"""Selective robot-only rewind support for synchronous LIBERO environments.

The wrapper deliberately avoids ``ControlEnv.set_state`` because that method
restores the complete flattened MuJoCo state, including objects and fixtures.
CycleVLA's backtrack contract is narrower: restore only the arm and gripper
configuration, keep the current scene state, clear the robot velocities, and
synchronize the OSC controller to the restored pose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jitpi05.jitrl.cycle import PhysicalEvidence


@dataclass(frozen=True)
class RobotConfigurationSnapshot:
    """A robot-only state captured from one synchronous LIBERO environment."""

    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    arm_qpos_indices: tuple[int, ...]
    arm_qpos: tuple[float, ...]
    gripper_qpos: tuple[float, ...]


@dataclass(frozen=True)
class RobotRestoreReport:
    """Numerical audit of one selective robot-state restore waypoint."""

    qpos_max_error: float
    robot_qvel_max_abs: float
    non_robot_qpos_max_change: float
    controller_resynchronized: bool


@dataclass(frozen=True)
class RobotRewindReport:
    """Audit of a smooth reverse replay through recorded robot configurations."""

    target_history_index: int
    final_history_index: int
    source_history_steps: int
    replayed_steps: int
    completed: bool
    waypoint_indices: tuple[int, ...]
    max_waypoint_qpos_delta: float
    qpos_max_error: float
    robot_qvel_max_abs: float
    non_robot_qpos_max_change: float
    controller_resynchronized: bool


def _flatten_indices(value: Any) -> tuple[int, ...]:
    """Normalize robosuite scalar/list/slice joint addresses to integer indices."""

    if isinstance(value, slice):
        if value.stop is None:
            raise ValueError("open-ended joint address slices are not supported")
        return tuple(range(value.start or 0, value.stop, value.step or 1))
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, np.ndarray):
        return tuple(int(item) for item in value.reshape(-1).tolist())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    raise TypeError(f"unsupported joint address type: {type(value).__name__}")


def _single_libero_env(vector_env: Any) -> Any:
    """Return the direct child of a batch-one synchronous vector environment."""

    children = getattr(vector_env, "envs", None)
    if children is None or len(children) != 1:
        raise TypeError(
            "Cycle recovery requires a batch-one synchronous vector environment "
            "with a public .envs[0] child"
        )
    return children[0]


def _control_env(libero_env: Any) -> Any:
    control = getattr(libero_env, "_env", None)
    if control is None or not hasattr(control, "sim") or not hasattr(control, "robots"):
        raise TypeError("the LIBERO child is not initialized with a robosuite ControlEnv")
    if len(control.robots) != 1:
        raise ValueError("Cycle recovery currently supports exactly one robot")
    return control


def _robot_indices(control: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    robot = control.robots[0]
    arm_qpos = _flatten_indices(robot._ref_joint_pos_indexes)
    gripper_qpos = _flatten_indices(robot._ref_gripper_joint_pos_indexes or ())
    arm_qvel = _flatten_indices(robot._ref_joint_vel_indexes)
    gripper_qvel = _flatten_indices(robot._ref_gripper_joint_vel_indexes or ())
    qpos_indices = tuple(dict.fromkeys((*arm_qpos, *gripper_qpos)))
    qvel_indices = tuple(dict.fromkeys((*arm_qvel, *gripper_qvel)))
    if not arm_qpos or not qpos_indices or not qvel_indices:
        raise ValueError("LIBERO robot references did not expose arm and gripper indices")
    return qpos_indices, qvel_indices, arm_qpos


def inspect_recovery_environment(vector_env: Any) -> dict[str, Any]:
    """Inspect the runtime unwrap and index chain without mutating simulator state."""

    libero_env = _single_libero_env(vector_env)
    control = _control_env(libero_env)
    robot = control.robots[0]
    qpos_indices, qvel_indices, arm_qpos_indices = _robot_indices(control)
    return {
        "vector_env_type": type(vector_env).__name__,
        "libero_env_type": type(libero_env).__name__,
        "control_env_type": type(control).__name__,
        "sim_type": type(control.sim).__name__,
        "robot_type": type(robot).__name__,
        "controller_type": type(robot.controller).__name__,
        "qpos_size": int(control.sim.data.qpos.shape[0]),
        "qvel_size": int(control.sim.data.qvel.shape[0]),
        "robot_qpos_indices": list(qpos_indices),
        "robot_qvel_indices": list(qvel_indices),
        "arm_qpos_indices": list(arm_qpos_indices),
        "controller_has_update_initial_joints": callable(
            getattr(robot.controller, "update_initial_joints", None)
        ),
        "controller_has_reset_goal": callable(getattr(robot.controller, "reset_goal", None)),
        "control_has_observation_refresh": callable(
            getattr(control, "_update_observables", None)
        )
        and callable(getattr(control, "_post_process", None)),
    }


def capture_robot_configuration(vector_env: Any) -> RobotConfigurationSnapshot:
    """Capture only arm and gripper qpos; no object state is copied."""

    control = _control_env(_single_libero_env(vector_env))
    qpos_indices, qvel_indices, arm_qpos_indices = _robot_indices(control)
    qpos = np.asarray(control.sim.data.qpos)
    gripper_indices = tuple(index for index in qpos_indices if index not in arm_qpos_indices)
    return RobotConfigurationSnapshot(
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        arm_qpos_indices=arm_qpos_indices,
        arm_qpos=tuple(float(qpos[index]) for index in arm_qpos_indices),
        gripper_qpos=tuple(float(qpos[index]) for index in gripper_indices),
    )


def _batch_single_observation(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _batch_single_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def refresh_libero_observation(vector_env: Any) -> Any:
    """Return the batch-one formatted child observation after a simulator write."""

    libero_env = _single_libero_env(vector_env)
    control = _control_env(libero_env)
    raw_observation = control.env._get_observations()
    formatter = getattr(libero_env, "_format_raw_obs", None)
    if not callable(formatter):
        raise TypeError("the LIBERO child does not expose _format_raw_obs")
    return _batch_single_observation(formatter(raw_observation))


def restore_robot_configuration(
    vector_env: Any,
    snapshot: RobotConfigurationSnapshot,
) -> RobotRestoreReport:
    """Restore arm/gripper qpos and resynchronize the low-level controller.

    The function never writes non-robot qpos or qvel.  The returned audit report
    therefore exposes the maximum scene-state change caused by this operation,
    which should be numerically zero in the MuJoCo probe.
    """

    control = _control_env(_single_libero_env(vector_env))
    robot = control.robots[0]
    qpos_indices, qvel_indices, arm_qpos_indices = _robot_indices(control)
    if qpos_indices != snapshot.qpos_indices or qvel_indices != snapshot.qvel_indices:
        raise ValueError("snapshot joint references do not match the current LIBERO environment")

    sim = control.sim
    all_qpos_indices = np.arange(sim.data.qpos.shape[0])
    robot_qpos_set = set(qpos_indices)
    non_robot_indices = [int(index) for index in all_qpos_indices if index not in robot_qpos_set]
    non_robot_before = np.asarray(sim.data.qpos[non_robot_indices]).copy()

    sim.data.qpos[list(snapshot.arm_qpos_indices)] = np.asarray(snapshot.arm_qpos)
    gripper_indices = tuple(index for index in qpos_indices if index not in arm_qpos_indices)
    if len(gripper_indices) != len(snapshot.gripper_qpos):
        raise ValueError("snapshot gripper dimension does not match the current environment")
    sim.data.qpos[list(gripper_indices)] = np.asarray(snapshot.gripper_qpos)
    sim.data.qvel[list(qvel_indices)] = 0.0
    sim.forward()

    controller = getattr(robot, "controller", None)
    resynchronized = False
    update_initial_joints = getattr(controller, "update_initial_joints", None)
    reset_goal = getattr(controller, "reset_goal", None)
    if callable(update_initial_joints):
        update_initial_joints(np.asarray(snapshot.arm_qpos))
        resynchronized = True
    elif callable(reset_goal):
        reset_goal()
        resynchronized = True
    else:
        raise RuntimeError("robot controller exposes neither update_initial_joints nor reset_goal")

    # Rebuild derived observations, but do not call set_init_state/set_state: both
    # operate on the full simulator state and would violate the object-state rule.
    post_process = getattr(control, "_post_process", None)
    update_observables = getattr(control, "_update_observables", None)
    if not callable(post_process) or not callable(update_observables):
        raise RuntimeError("LIBERO ControlEnv does not expose observation refresh hooks")
    post_process()
    update_observables(force=True)

    current_qpos = np.asarray(sim.data.qpos)
    current_qvel = np.asarray(sim.data.qvel)
    qpos_error = max(
        abs(float(current_qpos[index]) - expected)
        for index, expected in zip(
            snapshot.arm_qpos_indices, snapshot.arm_qpos, strict=True
        )
    )
    if snapshot.gripper_qpos:
        qpos_error = max(
            qpos_error,
            max(
                abs(float(current_qpos[index]) - expected)
                for index, expected in zip(gripper_indices, snapshot.gripper_qpos, strict=True)
            ),
        )
    non_robot_change = (
        float(np.max(np.abs(current_qpos[non_robot_indices] - non_robot_before)))
        if non_robot_indices
        else 0.0
    )
    robot_qvel_max_abs = float(np.max(np.abs(current_qvel[list(qvel_indices)])))
    return RobotRestoreReport(
        qpos_max_error=qpos_error,
        robot_qvel_max_abs=robot_qvel_max_abs,
        non_robot_qpos_max_change=non_robot_change,
        controller_resynchronized=resynchronized,
    )


def rewind_robot_configuration_history(
    vector_env: Any,
    history: Sequence[RobotConfigurationSnapshot],
    target_history_index: int,
    *,
    max_steps: int | None = None,
    on_waypoint: Callable[[int, RobotRestoreReport], None] | None = None,
) -> RobotRewindReport:
    """Reverse-replay recorded robot states without rewinding scene objects.

    Every replay step restores exactly the preceding recorded control-step state.
    When ``max_steps`` is smaller than the path length, replay stops at the last
    affordable intermediate state instead of skipping waypoints or teleporting
    to the target.  Callers can use ``on_waypoint`` to render each state and to
    charge the replay step against an episode budget.
    """

    if not history:
        raise ValueError("robot configuration history must not be empty")
    if (
        isinstance(target_history_index, bool)
        or not isinstance(target_history_index, (int, np.integer))
        or not 0 <= target_history_index < len(history)
    ):
        raise ValueError("target_history_index must reference the robot configuration history")
    if max_steps is not None and (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a non-negative integer or None")
    target_history_index = int(target_history_index)

    control = _control_env(_single_libero_env(vector_env))
    qpos_indices, qvel_indices, _ = _robot_indices(control)
    for snapshot in history:
        if snapshot.qpos_indices != qpos_indices or snapshot.qvel_indices != qvel_indices:
            raise ValueError("history joint references do not match the current LIBERO environment")

    current_history_index = len(history) - 1
    source_history_steps = current_history_index - target_history_index
    replayed_steps = (
        source_history_steps
        if max_steps is None
        else min(source_history_steps, int(max_steps))
    )
    waypoint_indices = tuple(
        range(current_history_index - 1, current_history_index - replayed_steps - 1, -1)
    )

    sim = control.sim
    robot_qpos_set = set(qpos_indices)
    non_robot_indices = [
        int(index)
        for index in range(int(sim.data.qpos.shape[0]))
        if index not in robot_qpos_set
    ]
    non_robot_before = np.asarray(sim.data.qpos[non_robot_indices]).copy()
    restore_reports: list[RobotRestoreReport] = []
    max_waypoint_qpos_delta = 0.0
    previous_snapshot = history[current_history_index]
    for history_index in waypoint_indices:
        snapshot = history[history_index]
        previous_qpos = np.asarray(
            (*previous_snapshot.arm_qpos, *previous_snapshot.gripper_qpos),
            dtype=np.float64,
        )
        waypoint_qpos = np.asarray(
            (*snapshot.arm_qpos, *snapshot.gripper_qpos),
            dtype=np.float64,
        )
        max_waypoint_qpos_delta = max(
            max_waypoint_qpos_delta,
            float(np.max(np.abs(waypoint_qpos - previous_qpos))),
        )
        report = restore_robot_configuration(vector_env, snapshot)
        restore_reports.append(report)
        if on_waypoint is not None:
            on_waypoint(history_index, report)
        previous_snapshot = snapshot

    final_history_index = current_history_index - replayed_steps
    non_robot_change = (
        float(np.max(np.abs(np.asarray(sim.data.qpos[non_robot_indices]) - non_robot_before)))
        if non_robot_indices
        else 0.0
    )
    return RobotRewindReport(
        target_history_index=target_history_index,
        final_history_index=final_history_index,
        source_history_steps=source_history_steps,
        replayed_steps=replayed_steps,
        completed=final_history_index == target_history_index,
        waypoint_indices=waypoint_indices,
        max_waypoint_qpos_delta=max_waypoint_qpos_delta,
        qpos_max_error=max((report.qpos_max_error for report in restore_reports), default=0.0),
        robot_qvel_max_abs=max(
            (report.robot_qvel_max_abs for report in restore_reports), default=0.0
        ),
        non_robot_qpos_max_change=max(
            non_robot_change,
            max(
                (report.non_robot_qpos_max_change for report in restore_reports),
                default=0.0,
            ),
        ),
        controller_resynchronized=all(
            report.controller_resynchronized for report in restore_reports
        ),
    )


def _robosuite_success(control: Any) -> bool:
    """Safely query the robosuite task success check; never raises."""

    check = getattr(control, "check_success", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception:
        return False


def _gripper_qpos_from_observation(observation: Any) -> tuple[float, ...] | None:
    """Read the batched gripper qpos from a formatted LeRobot observation."""

    robot_state = observation.get("robot_state") if isinstance(observation, Mapping) else None
    gripper = robot_state.get("gripper") if isinstance(robot_state, Mapping) else None
    qpos = gripper.get("qpos") if isinstance(gripper, Mapping) else None
    if qpos is None:
        return None
    try:
        return tuple(float(value) for value in np.asarray(qpos, dtype=np.float64).reshape(-1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scene object inspection: read-only MuJoCo body/site facts for object-level
# physical evidence. Everything here is a safe read and never mutates state.
# ---------------------------------------------------------------------------

_ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")
_IGNORED_OBJECT_BODIES = {"world", "table"}
_NON_SEMANTIC_TOKENS = {"main", "base", "default", "collision", "link"}


@dataclass(frozen=True)
class SceneObject:
    """One movable scene object with a canonical, task-usable name."""

    body_name: str
    canonical_name: str
    body_id: int
    position: tuple[float, float, float]


def _normalize_object_name(name: str) -> str:
    """Normalize any object label to a canonical lower-case space-joined name."""

    tokens: list[str] = []
    for part in str(name).lower().replace("-", " ").split("_"):
        for token in part.split():
            if not token or token.isdigit():
                continue
            if token in _NON_SEMANTIC_TOKENS:
                continue
            tokens.append(token)
    return " ".join(tokens)


def _canonical_object_name(body_name: str) -> str:
    """Normalize a MuJoCo body name to a task-usable canonical name.

    ``moka_pot_1_main`` -> ``moka pot``; ``chefmate_8_frypan_1_main`` ->
    ``chefmate frypan`` (numeric tokens are dropped, brand words are kept).
    """

    return _normalize_object_name(body_name)


def _is_scene_object_body(body_name: str) -> bool:
    if not body_name:
        return False
    if any(body_name.startswith(prefix) for prefix in _ROBOT_BODY_PREFIXES):
        return False
    return body_name not in _IGNORED_OBJECT_BODIES


def inspect_scene_objects(vector_env: Any) -> tuple[SceneObject, ...]:
    """Return deduplicated movable scene objects (prefer ``_main`` bodies).

    Robot, gripper, mount, world, and table bodies are excluded.  When any
    ``*_main`` bodies exist they are used as the authoritative object anchor;
    otherwise every remaining scene body is collected.
    """

    control = _control_env(_single_libero_env(vector_env))
    sim = control.sim
    model = getattr(sim, "model", None)
    if model is None:
        return ()
    main_bodies: list[SceneObject] = []
    fallback_bodies: list[SceneObject] = []
    nbody = int(getattr(model, "nbody", 0))
    for body_id in range(nbody):
        name = model.body_id2name(body_id) or ""
        if not _is_scene_object_body(name):
            continue
        canonical = _canonical_object_name(name)
        if not canonical:
            continue
        scene_object = SceneObject(
            body_name=name,
            canonical_name=canonical,
            body_id=body_id,
            position=tuple(float(v) for v in np.asarray(sim.data.body_xpos[body_id])),
        )
        if name.endswith("_main"):
            main_bodies.append(scene_object)
        else:
            fallback_bodies.append(scene_object)
    selected = main_bodies if main_bodies else fallback_bodies
    seen: dict[str, SceneObject] = {}
    for scene_object in selected:
        if scene_object.canonical_name not in seen:
            seen[scene_object.canonical_name] = scene_object
    return tuple(seen.values())


def gripper_pose(vector_env: Any) -> tuple[float, float, float] | None:
    """Return the end-effector grip site position, or None when unavailable."""

    control = _control_env(_single_libero_env(vector_env))
    sim = control.sim
    robot = control.robots[0]
    eef_site_id = getattr(robot, "eef_site_id", None)
    if eef_site_id is None:
        model = getattr(sim, "model", None)
        name2id = getattr(model, "site_name2id", None) if model is not None else None
        if callable(name2id):
            try:
                eef_site_id = name2id("gripper0_grip_site")
            except Exception:
                eef_site_id = None
    if eef_site_id is None:
        return None
    try:
        return tuple(float(v) for v in np.asarray(sim.data.site_xpos[eef_site_id]))
    except Exception:
        return None


def held_object_name(
    vector_env: Any,
    *,
    objects: Sequence[SceneObject] | None = None,
    gripper_position: Sequence[float] | None = None,
    xy_threshold: float = 0.06,
    z_threshold: float = 0.12,
) -> str | None:
    """Return the scene object closest to the gripper within the grasp cone.

    Returns ``None`` when no object is close enough, so the absence of a held
    object is never fabricated into a false identity claim.
    """

    if gripper_position is None:
        return None
    if objects is None:
        objects = inspect_scene_objects(vector_env)
    gx, gy, gz = (float(value) for value in gripper_position)
    best: str | None = None
    best_distance = float("inf")
    for scene_object in objects:
        ox, oy, oz = scene_object.position
        xy = math.hypot(ox - gx, oy - gy)
        dz = abs(oz - gz)
        if xy <= xy_threshold and dz <= z_threshold:
            distance = xy + dz
            if distance < best_distance:
                best_distance = distance
                best = scene_object.canonical_name
    return best


def match_target_object(
    action_text: str,
    objects: Sequence[SceneObject],
) -> str | None:
    """Return the scene object most referenced by a free-text action label."""

    text_tokens = set(re.findall(r"[a-z]+", str(action_text).lower()))
    if not text_tokens:
        return None
    best: str | None = None
    best_count = 0
    for scene_object in objects:
        overlap = len(set(scene_object.canonical_name.split()) & text_tokens)
        if overlap > best_count:
            best_count = overlap
            best = scene_object.canonical_name
    return best if best_count >= 1 else None


def _object_position(
    objects: Sequence[SceneObject],
    canonical_name: str,
) -> tuple[float, float, float] | None:
    for scene_object in objects:
        if scene_object.canonical_name == canonical_name:
            return scene_object.position
    return None


def build_physical_evidence(
    vector_env: Any,
    observation: Any,
    *,
    success: bool,
    gripper_closed_threshold: float | None = None,
    action_type: str = "",
    target_object: str | None = None,
    previous_held_relative: Sequence[float] | None = None,
    grasp_xy_threshold: float = 0.06,
    grasp_z_threshold: float = 0.12,
    follow_relative_threshold: float = 0.03,
) -> PhysicalEvidence:
    """Construct conservative object-level physical evidence from live state.

    Facts that can be read safely are filled in; everything else stays None
    (unknown) instead of guessing, matching the CycleVLA contract of never
    fabricating evidence:
    - ``postcondition_satisfied`` is True exactly when the robosuite success
      check passes;
    - ``gripper_closed`` is derived from the formatted observation when a
      threshold is configured;
    - ``target_identity_ok`` compares the object under the gripper with the
      requested target object (only for grasp-like actions with a target);
    - ``object_following_gripper`` compares the held object's offset to the
      gripper across two consecutive checks, so a carried object is detected
      without fabricating contact.
    """

    control = _control_env(_single_libero_env(vector_env))
    postcondition_satisfied = True if (success or _robosuite_success(control)) else None
    gripper_closed = None
    if gripper_closed_threshold is not None:
        gripper_qpos = _gripper_qpos_from_observation(observation)
        if gripper_qpos:
            gripper_closed = float(sum(gripper_qpos)) <= gripper_closed_threshold

    objects = inspect_scene_objects(vector_env)
    gripper_position = gripper_pose(vector_env)
    held = (
        held_object_name(
            vector_env,
            objects=objects,
            gripper_position=gripper_position,
            xy_threshold=grasp_xy_threshold,
            z_threshold=grasp_z_threshold,
        )
        if gripper_position is not None
        else None
    )

    target_identity_ok = None
    if target_object and held:
        target_identity_ok = (
            _normalize_object_name(held) == _normalize_object_name(target_object)
        )

    object_following_gripper = None
    if (
        held
        and gripper_position is not None
        and previous_held_relative is not None
    ):
        held_position = _object_position(objects, held)
        if held_position is not None:
            relative = (
                held_position[0] - gripper_position[0],
                held_position[1] - gripper_position[1],
                held_position[2] - gripper_position[2],
            )
            delta = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(relative, previous_held_relative))
            )
            object_following_gripper = delta <= follow_relative_threshold

    failure_reasons: list[str] = []
    if target_identity_ok is False:
        failure_reasons.append(
            f"gripper holds {held!r} but requested target is {target_object!r}"
        )
    if object_following_gripper is False:
        failure_reasons.append(f"held object {held!r} did not follow the gripper")
    return PhysicalEvidence(
        action_type=action_type,
        gripper_closed=gripper_closed,
        object_following_gripper=object_following_gripper,
        target_identity_ok=target_identity_ok,
        destination_reached=True if postcondition_satisfied else None,
        released=None,
        postcondition_satisfied=postcondition_satisfied,
        failure_detected=None,
        failure_reasons=tuple(failure_reasons),
    )
