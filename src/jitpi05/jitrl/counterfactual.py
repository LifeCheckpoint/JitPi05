"""State-isolated primitives and summaries for Cycle recovery counterfactuals.

These utilities are intentionally separate from robot-only Cycle recovery.  A
counterfactual is a diagnostic experiment: it restores the entire MuJoCo state
between branches so each intervention begins from exactly the same failure
state.  Production Cycle rollout must continue using robot-only rewind.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from jitpi05.jitrl.libero_recovery import (
    _control_env,
    _robot_indices,
    _single_libero_env,
    refresh_libero_observation,
)

COUNTERFACTUAL_BRANCHES = (
    "no_op",
    "target_only",
    "rewind_only",
    "mbr_only",
    "full_cycle",
)


@dataclass(frozen=True)
class FullEnvironmentSnapshot:
    """Opaque complete simulator state for a diagnostic-only branch reset."""

    state: Any
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    control_bookkeeping: tuple[tuple[str, Any], ...]
    robosuite_bookkeeping: tuple[tuple[str, Any], ...]


def _capture_bookkeeping(instance: Any, names: tuple[str, ...]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, copy.deepcopy(getattr(instance, name)))
        for name in names
        if hasattr(instance, name)
    )


def _restore_bookkeeping(instance: Any, values: tuple[tuple[str, Any], ...]) -> None:
    for name, value in values:
        setattr(instance, name, copy.deepcopy(value))


def capture_full_environment_state(vector_env: Any) -> FullEnvironmentSnapshot:
    """Capture a detached full MuJoCo state without changing the environment."""

    control = _control_env(_single_libero_env(vector_env))
    sim = control.sim
    get_state = getattr(sim, "get_state", None)
    if not callable(get_state):
        raise TypeError("MuJoCo simulator does not expose get_state for counterfactuals")
    return FullEnvironmentSnapshot(
        state=copy.deepcopy(get_state()),
        qpos=tuple(float(value) for value in np.asarray(sim.data.qpos).reshape(-1)),
        qvel=tuple(float(value) for value in np.asarray(sim.data.qvel).reshape(-1)),
        control_bookkeeping=_capture_bookkeeping(control, ("timestep", "done")),
        robosuite_bookkeeping=_capture_bookkeeping(
            getattr(control, "env", None), ("timestep", "done")
        ),
    )


def restore_full_environment_state(
    vector_env: Any,
    snapshot: FullEnvironmentSnapshot,
) -> Any:
    """Restore a full diagnostic snapshot and resynchronize robot/controller state."""

    control = _control_env(_single_libero_env(vector_env))
    sim = control.sim
    set_state = getattr(sim, "set_state", None)
    if not callable(set_state):
        raise TypeError("MuJoCo simulator does not expose set_state for counterfactuals")
    set_state(copy.deepcopy(snapshot.state))
    _restore_bookkeeping(control, snapshot.control_bookkeeping)
    _restore_bookkeeping(getattr(control, "env", None), snapshot.robosuite_bookkeeping)
    sim.forward()

    qpos_indices, _qvel_indices, arm_qpos_indices = _robot_indices(control)
    qpos = np.asarray(sim.data.qpos)
    arm_qpos = qpos[list(arm_qpos_indices)].copy()
    robot = control.robots[0]
    controller = getattr(robot, "controller", None)
    update_initial_joints = getattr(controller, "update_initial_joints", None)
    reset_goal = getattr(controller, "reset_goal", None)
    if callable(update_initial_joints):
        update_initial_joints(arm_qpos)
    elif callable(reset_goal):
        reset_goal()
    else:
        raise RuntimeError("robot controller exposes neither update_initial_joints nor reset_goal")

    post_process = getattr(control, "_post_process", None)
    update_observables = getattr(control, "_update_observables", None)
    if not callable(post_process) or not callable(update_observables):
        raise RuntimeError("LIBERO ControlEnv does not expose observation refresh hooks")
    post_process()
    update_observables(force=True)

    restored_qpos = np.asarray(sim.data.qpos)
    restored_qvel = np.asarray(sim.data.qvel)
    expected_qpos = np.asarray(snapshot.qpos)
    expected_qvel = np.asarray(snapshot.qvel)
    if restored_qpos.shape != expected_qpos.shape or restored_qvel.shape != expected_qvel.shape:
        raise RuntimeError("counterfactual snapshot dimensions changed during restore")
    qpos_error = float(np.max(np.abs(restored_qpos - expected_qpos)))
    qvel_error = float(np.max(np.abs(restored_qvel - expected_qvel)))
    if qpos_error > 1e-8 or qvel_error > 1e-8:
        raise RuntimeError(
            "counterfactual full-state restore was inexact: "
            f"qpos_error={qpos_error:.3g} qvel_error={qvel_error:.3g}"
        )
    if len(qpos_indices) == 0:
        raise RuntimeError("counterfactual restore found no robot qpos indices")
    return refresh_libero_observation(vector_env)


def summarize_counterfactual_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce JSON-safe branch-level rescue/harm summaries from event records."""

    rows = list(events)
    branch_rows: dict[str, list[Mapping[str, Any]]] = {
        branch: [] for branch in COUNTERFACTUAL_BRANCHES
    }
    for event in rows:
        for branch in event.get("branches", []):
            name = str(branch.get("name", ""))
            if name in branch_rows:
                branch_rows[name].append(branch)

    branches: dict[str, dict[str, Any]] = {}
    for name, items in branch_rows.items():
        successes = sum(bool(item.get("success")) for item in items)
        mean_steps = (
            sum(int(item.get("executed_steps", 0)) for item in items) / len(items)
            if items
            else 0.0
        )
        branches[name] = {
            "events": len(items),
            "successes": successes,
            "success_rate": successes / len(items) if items else None,
            "mean_executed_steps": mean_steps,
            "termination_reasons": dict(
                sorted(
                    Counter(str(item.get("termination_reason", "unknown")) for item in items).items()
                )
            ),
        }

    no_op = branch_rows["no_op"]
    paired: dict[str, dict[str, int]] = {}
    for name, items in branch_rows.items():
        if name == "no_op":
            continue
        comparisons = zip(no_op, items, strict=False)
        rescue = harm = both_success = both_failure = 0
        for baseline, treatment in comparisons:
            baseline_success = bool(baseline.get("success"))
            treatment_success = bool(treatment.get("success"))
            rescue += int(not baseline_success and treatment_success)
            harm += int(baseline_success and not treatment_success)
            both_success += int(baseline_success and treatment_success)
            both_failure += int(not baseline_success and not treatment_success)
        paired[name] = {
            "paired_events": min(len(no_op), len(items)),
            "rescue": rescue,
            "harm": harm,
            "both_success": both_success,
            "both_failure": both_failure,
            "net_rescue": rescue - harm,
        }

    return {
        "event_count": len(rows),
        "branches": branches,
        "paired_vs_no_op": paired,
    }
