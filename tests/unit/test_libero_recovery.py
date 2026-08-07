import numpy as np
import pytest

from jitpi05.jitrl.libero_recovery import (
    build_physical_evidence,
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
    def __init__(self, check_success_result: bool = False) -> None:
        self.sim = FakeSim()
        self.robots = [FakeRobot(self.sim)]
        self.post_process_calls = 0
        self.update_observables_calls = 0
        self.check_success_result = check_success_result

    def _post_process(self) -> None:
        self.post_process_calls += 1

    def _update_observables(self, *, force: bool) -> None:
        assert force is True
        self.update_observables_calls += 1

    def check_success(self) -> bool:
        return self.check_success_result


class FakeLiberoEnv:
    def __init__(self, check_success: bool = False) -> None:
        self._env = FakeControlEnv(check_success_result=check_success)


class FakeVectorEnv:
    def __init__(self, check_success: bool = False) -> None:
        self.envs = [FakeLiberoEnv(check_success)]


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


def test_build_physical_evidence_reads_only_safe_facts() -> None:
    closed_obs = {"robot_state": {"gripper": {"qpos": np.array([0.01, 0.01])}}}
    open_obs = {"robot_state": {"gripper": {"qpos": np.array([0.04, 0.04])}}}

    # 无 gripper 阈值 + check_success 失败 → 所有事实 unknown（绝不臆测失败）
    env_unknown = FakeVectorEnv(check_success=False)
    evidence = build_physical_evidence(env_unknown, closed_obs, success=False)
    assert evidence.gripper_closed is None
    assert evidence.postcondition_satisfied is None
    assert evidence.explicit_failure is False

    # 传入 success=True → postcondition 明确成功
    assert build_physical_evidence(
        env_unknown, closed_obs, success=True
    ).postcondition_satisfied is True

    # robosuite check_success 通过 → 即便 success 标志为 False 也判定成功
    assert build_physical_evidence(
        FakeVectorEnv(check_success=True), closed_obs, success=False
    ).postcondition_satisfied is True

    # 配置 gripper 阈值后：闭合小 qpos → True，张开大 qpos → False
    assert build_physical_evidence(
        env_unknown, closed_obs, success=False, gripper_closed_threshold=0.05
    ).gripper_closed is True
    assert build_physical_evidence(
        env_unknown, open_obs, success=False, gripper_closed_threshold=0.05
    ).gripper_closed is False
