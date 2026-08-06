"""Probe selective robot-only rewind against a real synchronous LIBERO env.

Run from the repository root, for example:

    MUJOCO_GL=egl uv run python scripts/probe_libero_recovery.py \
        --suite libero_90 --task-id 79 --episode-index 0 --seed 17

This is a diagnostic script. It does not restore object state and does not
write any artifacts.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from jitpi05.jitrl.libero_recovery import (
    capture_robot_configuration,
    inspect_recovery_environment,
    restore_robot_configuration,
)
from jitpi05.simulation import close_envs, make_single_env


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=_positive_int, default=79)
    parser.add_argument("--episode-index", type=_positive_int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=_positive_int, default=400)
    return parser


def _max_abs(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def main() -> int:
    args = _parser().parse_args()
    task_spec = {
        "name": f"{args.suite}_task{args.task_id}",
        "suite": args.suite,
        "task_id": args.task_id,
        "max_steps": args.max_steps,
    }
    envs, vector_env, _, _ = make_single_env(task_spec, args.episode_index)
    try:
        print(f"reset: task={task_spec['name']} seed={args.seed}")
        vector_env.reset(seed=[args.seed])
        print("environment:", inspect_recovery_environment(vector_env))

        child = vector_env.envs[0]
        control = child._env
        snapshot = capture_robot_configuration(vector_env)
        robot_qpos_indices = tuple(snapshot.qpos_indices)
        all_indices = np.arange(control.sim.data.qpos.shape[0])
        object_indices = [
            int(index) for index in all_indices if int(index) not in robot_qpos_indices
        ]
        object_before = np.asarray(control.sim.data.qpos[object_indices]).copy()

        # Deliberately perturb only the robot configuration.  No object qpos or
        # qvel is changed by this diagnostic setup.
        control.sim.data.qpos[list(snapshot.arm_qpos_indices)] += 0.01
        control.sim.data.qvel[list(snapshot.qvel_indices)] = 0.25
        control.sim.forward()

        report = restore_robot_configuration(vector_env, snapshot)
        object_after = np.asarray(control.sim.data.qpos[object_indices]).copy()
        object_change = _max_abs(object_after - object_before)
        print("restore_report:", report)
        print("object_qpos_max_change:", object_change)
        print("sim_forward_calls: completed")

        if report.qpos_max_error >= 1e-6:
            raise AssertionError(f"robot qpos restore error too large: {report.qpos_max_error}")
        if report.robot_qvel_max_abs >= 1e-10:
            raise AssertionError(f"robot qvel was not cleared: {report.robot_qvel_max_abs}")
        if object_change >= 1e-10:
            raise AssertionError(f"object qpos changed during rewind: {object_change}")
        if not report.controller_resynchronized:
            raise AssertionError("controller was not resynchronized")
        print("probe_status: PASS")
        return 0
    finally:
        close_envs(envs)


if __name__ == "__main__":
    raise SystemExit(main())
