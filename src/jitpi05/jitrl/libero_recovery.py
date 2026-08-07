"""Selective robot-only rewind support for synchronous LIBERO environments.

The wrapper deliberately avoids ``ControlEnv.set_state`` because that method
restores the complete flattened MuJoCo state, including objects and fixtures.
CycleVLA's backtrack contract is narrower: restore only the arm and gripper
configuration, keep the current scene state, clear the robot velocities, and
synchronize the OSC controller to the restored pose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    """Numerical audit of a selective rewind operation."""

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


def build_physical_evidence(
    vector_env: Any,
    observation: Any,
    *,
    success: bool,
    gripper_closed_threshold: float | None = None,
) -> PhysicalEvidence:
    """Construct conservative physical evidence from the live LIBERO state.

    Only facts that can be read safely are filled in:
    - ``postcondition_satisfied`` is True exactly when the robosuite success
      check passes (a mid-episode False is treated as unknown, never failure);
    - ``gripper_closed`` is derived from the formatted observation when a
      threshold is configured (calibrate per gripper type first).
    All other facts stay None (unknown) instead of guessing, matching the
    CycleVLA contract of never fabricating evidence.
    """

    control = _control_env(_single_libero_env(vector_env))
    postcondition_satisfied = True if (success or _robosuite_success(control)) else None
    gripper_closed = None
    if gripper_closed_threshold is not None:
        gripper_qpos = _gripper_qpos_from_observation(observation)
        if gripper_qpos:
            gripper_closed = float(sum(gripper_qpos)) <= gripper_closed_threshold
    return PhysicalEvidence(
        action_type="",
        gripper_closed=gripper_closed,
        postcondition_satisfied=postcondition_satisfied,
    )
