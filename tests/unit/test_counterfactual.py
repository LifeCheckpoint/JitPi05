from __future__ import annotations

import copy

import numpy as np

from jitpi05.jitrl.counterfactual import (
    capture_full_environment_state,
    restore_full_environment_state,
    summarize_counterfactual_events,
)


class _Data:
    def __init__(self) -> None:
        self.qpos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.qvel = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


class _Sim:
    def __init__(self) -> None:
        self.data = _Data()
        self.forward_calls = 0

    def get_state(self):
        return (self.data.qpos.copy(), self.data.qvel.copy())

    def set_state(self, state) -> None:
        qpos, qvel = copy.deepcopy(state)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    def forward(self) -> None:
        self.forward_calls += 1


class _Controller:
    def __init__(self) -> None:
        self.initial_joints = None

    def update_initial_joints(self, joints) -> None:
        self.initial_joints = np.asarray(joints).copy()


class _Robot:
    _ref_joint_pos_indexes = [0, 1, 2]
    _ref_joint_vel_indexes = [0, 1, 2]
    _ref_gripper_joint_pos_indexes = [3, 4]
    _ref_gripper_joint_vel_indexes = [3, 4]

    def __init__(self) -> None:
        self.controller = _Controller()


class _RobosuiteEnv:
    def __init__(self) -> None:
        self.timestep = 7
        self.done = False

    def _get_observations(self):
        return {"state": np.array([1.0])}


class _Control:
    def __init__(self) -> None:
        self.sim = _Sim()
        self.robots = [_Robot()]
        self.env = _RobosuiteEnv()
        self.timestep = 8
        self.done = False
        self.post_process_calls = 0
        self.update_observables_calls = 0

    def _post_process(self) -> None:
        self.post_process_calls += 1

    def _update_observables(self, *, force: bool) -> None:
        assert force is True
        self.update_observables_calls += 1


class _LiberoEnv:
    def __init__(self) -> None:
        self._env = _Control()

    def _format_raw_obs(self, observation):
        return observation


class _VectorEnv:
    def __init__(self) -> None:
        self.envs = [_LiberoEnv()]


def test_full_snapshot_restores_simulator_and_episode_bookkeeping() -> None:
    vector_env = _VectorEnv()
    control = vector_env.envs[0]._env
    snapshot = capture_full_environment_state(vector_env)

    control.sim.data.qpos[:] = 99.0
    control.sim.data.qvel[:] = -5.0
    control.timestep = 99
    control.done = True
    control.env.timestep = 98
    control.env.done = True

    observation = restore_full_environment_state(vector_env, snapshot)

    assert control.sim.data.qpos.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert control.sim.data.qvel.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert control.timestep == 8
    assert control.done is False
    assert control.env.timestep == 7
    assert control.env.done is False
    assert control.robots[0].controller.initial_joints.tolist() == [1.0, 2.0, 3.0]
    assert observation["state"].shape == (1, 1)


def test_counterfactual_summary_reports_branch_rates_and_paired_effects() -> None:
    events = [
        {
            "branches": [
                {"name": "no_op", "success": False, "executed_steps": 80, "termination_reason": "horizon"},
                {"name": "target_only", "success": True, "executed_steps": 40, "termination_reason": "environment_success"},
                {"name": "rewind_only", "success": False, "executed_steps": 70, "termination_reason": "horizon"},
                {"name": "mbr_only", "success": False, "executed_steps": 80, "termination_reason": "horizon"},
                {"name": "full_cycle", "success": True, "executed_steps": 55, "termination_reason": "environment_success"},
            ]
        },
        {
            "branches": [
                {"name": "no_op", "success": True, "executed_steps": 20, "termination_reason": "environment_success"},
                {"name": "target_only", "success": False, "executed_steps": 80, "termination_reason": "horizon"},
                {"name": "rewind_only", "success": True, "executed_steps": 30, "termination_reason": "environment_success"},
                {"name": "mbr_only", "success": True, "executed_steps": 20, "termination_reason": "environment_success"},
                {"name": "full_cycle", "success": False, "executed_steps": 80, "termination_reason": "horizon"},
            ]
        },
    ]

    summary = summarize_counterfactual_events(events)

    assert summary["event_count"] == 2
    assert summary["branches"]["no_op"]["success_rate"] == 0.5
    assert summary["branches"]["full_cycle"]["successes"] == 1
    assert summary["paired_vs_no_op"]["target_only"] == {
        "paired_events": 2,
        "rescue": 1,
        "harm": 1,
        "both_success": 0,
        "both_failure": 0,
        "net_rescue": 0,
    }
    assert summary["paired_vs_no_op"]["full_cycle"]["rescue"] == 1
    assert summary["paired_vs_no_op"]["full_cycle"]["harm"] == 1


def test_counterfactual_summary_preserves_empty_branch_contract() -> None:
    summary = summarize_counterfactual_events([])

    assert summary["event_count"] == 0
    assert summary["branches"]["no_op"]["success_rate"] is None
    assert summary["paired_vs_no_op"]["mbr_only"]["paired_events"] == 0
