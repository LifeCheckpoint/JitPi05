import numpy as np
import pytest

from jitpi05.jitrl.libero_recovery import (
    build_physical_evidence,
    capture_robot_configuration,
    gripper_pose,
    held_object_name,
    inspect_recovery_environment,
    inspect_scene_objects,
    match_target_object,
    restore_robot_configuration,
    rewind_robot_configuration_history,
)


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        self.qvel = np.ones(9, dtype=np.float64)
        self.forward_calls = 0


class FakeSim:
    def __init__(self, model=None) -> None:
        self.data = FakeData()
        self.model = model

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
    def __init__(
        self,
        check_success_result: bool = False,
        sim: FakeSim | None = None,
    ) -> None:
        self.sim = sim if sim is not None else FakeSim()
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
    def __init__(self, check_success: bool = False, sim=None) -> None:
        self._env = FakeControlEnv(check_success_result=check_success, sim=sim)


class FakeVectorEnv:
    def __init__(self, check_success: bool = False, sim=None) -> None:
        self.envs = [FakeLiberoEnv(check_success, sim)]


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


def test_rewind_reverse_replays_every_recorded_robot_waypoint() -> None:
    vector_env = FakeVectorEnv()
    control = vector_env.envs[0]._env
    history = [capture_robot_configuration(vector_env)]
    control.sim.data.qpos[:5] = [20.0, 21.0, 22.0, 23.0, 24.0]
    history.append(capture_robot_configuration(vector_env))
    control.sim.data.qpos[:5] = [30.0, 31.0, 32.0, 33.0, 34.0]
    history.append(capture_robot_configuration(vector_env))
    control.sim.data.qpos[5:] = [105.0, 106.0, 107.0, 108.0]
    non_robot_before = control.sim.data.qpos[5:].copy()
    visited: list[tuple[int, list[float]]] = []

    report = rewind_robot_configuration_history(
        vector_env,
        history,
        0,
        on_waypoint=lambda index, _report: visited.append(
            (index, control.sim.data.qpos[:5].tolist())
        ),
    )

    assert report.completed is True
    assert report.source_history_steps == 2
    assert report.replayed_steps == 2
    assert report.waypoint_indices == (1, 0)
    assert report.final_history_index == 0
    assert report.qpos_max_error == pytest.approx(0.0)
    assert report.non_robot_qpos_max_change == pytest.approx(0.0)
    assert visited == [
        (1, [20.0, 21.0, 22.0, 23.0, 24.0]),
        (0, [10.0, 11.0, 12.0, 13.0, 14.0]),
    ]
    assert control.sim.data.qpos[5:].tolist() == non_robot_before.tolist()


def test_rewind_budget_stops_at_intermediate_waypoint_without_teleporting() -> None:
    vector_env = FakeVectorEnv()
    control = vector_env.envs[0]._env
    history = [capture_robot_configuration(vector_env)]
    control.sim.data.qpos[:5] = [20.0, 21.0, 22.0, 23.0, 24.0]
    history.append(capture_robot_configuration(vector_env))
    control.sim.data.qpos[:5] = [30.0, 31.0, 32.0, 33.0, 34.0]
    history.append(capture_robot_configuration(vector_env))

    report = rewind_robot_configuration_history(vector_env, history, 0, max_steps=1)

    assert report.completed is False
    assert report.replayed_steps == 1
    assert report.waypoint_indices == (1,)
    assert report.final_history_index == 1
    assert control.sim.data.qpos[:5].tolist() == [20.0, 21.0, 22.0, 23.0, 24.0]


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


class ObjectSimModel:
    def __init__(self) -> None:
        self.nbody = 6
        self.nsite = 1
        self._bodies = {
            0: "world",
            1: "table",
            2: "robot0_link0",
            3: "moka_pot_1_main",
            4: "chefmate_8_frypan_1_main",
            5: "flat_stove_1_main",
        }
        self._sites = {0: "gripper0_grip_site"}

    def body_id2name(self, body_id: int) -> str | None:
        return self._bodies.get(body_id)

    def site_name2id(self, name: str) -> int:
        for site_id, site_name in self._sites.items():
            if site_name == name:
                return site_id
        raise KeyError(name)


class ObjectSimData:
    def __init__(self, gripper_position=(0.05, 0.0, 0.97)) -> None:
        self.qpos = np.zeros(9, dtype=np.float64)
        self.qvel = np.zeros(9, dtype=np.float64)
        self.forward_calls = 0
        self.body_xpos = np.array(
            [
                [0.0, 0.0, 0.0],       # world
                [0.0, 0.0, 0.9],       # table
                [-0.66, 0.0, 1.0],     # robot link
                [0.05, 0.0, 0.96],     # moka pot
                [-0.05, -0.25, 0.95],  # frypan
                [-0.2, 0.2, 0.95],     # stove
            ],
            dtype=np.float64,
        )
        self.site_xpos = np.array([list(gripper_position)], dtype=np.float64)


def _object_vector_env(gripper_position=(0.05, 0.0, 0.97)) -> FakeVectorEnv:
    sim = FakeSim()
    sim.model = ObjectSimModel()
    sim.data = ObjectSimData(gripper_position)
    return FakeVectorEnv(sim=sim)


def test_inspect_scene_objects_returns_movable_bodies_only() -> None:
    objects = inspect_scene_objects(_object_vector_env())

    names = {obj.canonical_name for obj in objects}
    assert names == {"moka pot", "chefmate frypan", "flat stove"}
    assert all(obj.canonical_name for obj in objects)


def test_target_matching_and_held_object_identity() -> None:
    env = _object_vector_env(gripper_position=(0.05, 0.0, 0.97))
    objects = inspect_scene_objects(env)

    assert match_target_object("grasp the moka pot", objects) == "moka pot"
    assert match_target_object("place the moka pot on the stove", objects) == "moka pot"
    assert match_target_object("grasp the frypan", objects) == "chefmate frypan"

    gripper = gripper_pose(env)
    assert held_object_name(env, objects=objects, gripper_position=gripper) == "moka pot"
    evidence = build_physical_evidence(
        env,
        {},
        success=False,
        action_type="grasp",
        target_object="moka pot",
    )
    assert evidence.target_identity_ok is True


def test_wrong_object_identity_reports_failure_reason() -> None:
    # 夹爪移动到 frypan 上方，但目标是 moka pot → identity 不匹配
    env = _object_vector_env(gripper_position=(-0.05, -0.25, 0.97))
    evidence = build_physical_evidence(
        env,
        {},
        success=False,
        action_type="grasp",
        target_object="moka pot",
    )

    assert evidence.target_identity_ok is False
    assert evidence.failure_reasons


def test_object_following_gripper_across_two_checks() -> None:
    # 第一次检查：gripper 与 moka pot 相对偏移约 (0, 0, 0.01)
    env = _object_vector_env(gripper_position=(0.05, 0.0, 0.97))
    previous_relative = (0.0, 0.0, 0.01)
    evidence = build_physical_evidence(
        env,
        {},
        success=False,
        action_type="grasp",
        target_object="moka pot",
        previous_held_relative=previous_relative,
    )
    assert evidence.object_following_gripper is True
    assert evidence.target_identity_ok is True
