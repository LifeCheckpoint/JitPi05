import numpy as np
import pytest

from jitpi05.jitrl.libero_recovery import (
    capture_robot_configuration,
    inspect_recovery_environment,
    restore_robot_configuration,
)


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        self.qvel = np.ones(9, dtype=np.float64)
        self.forward_calls = 0


class FakeSim:
    def __init__(self) -> None:
        self.data = FakeData()

    def forward(self) -> None:
        self.data.forward_calls += 1


class FakeController:
    def __init__(self, sim: FakeSim) -> None:
        self.sim = sim
        self.update_initial_joints_calls = 0
        self.reset_goal_calls = 0

    def update_initial_joints(self, joints) -> None:
        self.update_initial_joints_calls += 1
        assert np.asarray(joints).shape == (3,)

    def reset_goal(self) -> None:
        self.reset_goal_calls += 1


class FakeRobot:
    def __init__(self, sim: FakeSim) -> None:
        self._ref_joint_pos_indexes = [0, 1, 2]
        self._ref_joint_vel_indexes = [0, 1, 2]
        self._ref_gripper_joint_pos_indexes = [3, 4]
        self._ref_gripper_joint_vel_indexes = [3, 4]
        self.controller = FakeController(sim)


class FakeControlEnv:
    def __init__(self) -> None:
        self.sim = FakeSim()
        self.robots = [FakeRobot(self.sim)]
        self.post_process_calls = 0
        self.update_observables_calls = 0

    def _post_process(self) -> None:
        self.post_process_calls += 1

    def _update_observables(self, *, force: bool) -> None:
        assert force is True
        self.update_observables_calls += 1


class FakeLiberoEnv:
    def __init__(self) -> None:
        self._env = FakeControlEnv()


class FakeVectorEnv:
    def __init__(self) -> None:
        self.envs = [FakeLiberoEnv()]


def test_inspection_exposes_robot_only_recovery_chain() -> None:
    report = inspect_recovery_environment(FakeVectorEnv())

    assert report["vector_env_type"] == "FakeVectorEnv"
    assert report["robot_qpos_indices"] == [0, 1, 2, 3, 4]
    assert report["robot_qvel_indices"] == [0, 1, 2, 3, 4]
    assert report["controller_has_update_initial_joints"] is True
    assert report["controller_has_reset_goal"] is True


def test_restore_writes_only_robot_and_gripper_state() -> None:
    vector_env = FakeVectorEnv()
    control = vector_env.envs[0]._env
    snapshot = capture_robot_configuration(vector_env)
    control.sim.data.qpos[:] = np.arange(100.0, 109.0)
    control.sim.data.qvel[:] = 7.0
    non_robot_before = control.sim.data.qpos[5:].copy()

    report = restore_robot_configuration(vector_env, snapshot)

    assert report.qpos_max_error == pytest.approx(0.0)
    assert report.robot_qvel_max_abs == pytest.approx(0.0)
    assert report.non_robot_qpos_max_change == pytest.approx(0.0)
    assert report.controller_resynchronized is True
    assert control.sim.data.qpos[:5].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert control.sim.data.qpos[5:].tolist() == non_robot_before.tolist()
    assert control.sim.data.qvel[:5].tolist() == [0.0] * 5
    assert control.sim.data.qvel[5:].tolist() == [7.0] * 4
    assert control.sim.data.forward_calls == 1
    assert control.post_process_calls == 1
    assert control.update_observables_calls == 1
    assert control.robots[0].controller.update_initial_joints_calls == 1


def test_restore_rejects_snapshot_from_different_joint_layout() -> None:
    vector_env = FakeVectorEnv()
    snapshot = capture_robot_configuration(vector_env)
    invalid = snapshot.__class__(
        qpos_indices=(1, 2, 3, 4, 5),
        qvel_indices=snapshot.qvel_indices,
        arm_qpos_indices=snapshot.arm_qpos_indices,
        arm_qpos=snapshot.arm_qpos,
        gripper_qpos=snapshot.gripper_qpos,
    )

    with pytest.raises(ValueError, match="joint references"):
        restore_robot_configuration(vector_env, invalid)


def test_recovery_requires_batch_one_vector_env() -> None:
    class EmptyVectorEnv:
        envs = []

    with pytest.raises(TypeError, match="batch-one"):
        inspect_recovery_environment(EmptyVectorEnv())
