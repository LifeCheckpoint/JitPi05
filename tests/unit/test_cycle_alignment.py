"""官方 CycleVLA 对齐回归测试（A 类差异修复）。

覆盖：open-loop 步数、MBR 特征长度、failed repulsion 默认值、
反事实 horizon 语义、官方 1.5x 预算、真实 EEF 位姿特征。
"""

from __future__ import annotations

import pytest
import torch

from jitpi05.config import (
    CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS,
    CYCLE_MBR_ACTION_STEPS,
    CYCLE_MBR_USE_FAILED_REPULSION,
    CYCLE_NUM_STEPS_WAIT,
    CYCLE_OFFICIAL_BUDGET_MULTIPLIER,
    POLICY_ACTION_STEPS,
)
from jitpi05.jitrl.cycle import (
    cumulative_pose_trajectory_features,
    select_mbr_chunk,
)


def test_open_loop_steps_match_official_five() -> None:
    """官方 ``num_open_loop_steps=5``（模型 chunk_size 仍为 10）。"""

    assert POLICY_ACTION_STEPS == 5
    assert CYCLE_MBR_ACTION_STEPS == POLICY_ACTION_STEPS


def test_check_window_preserves_thirty_step_physical_window() -> None:
    """固定触发窗口应与官方 5 步 open-loop 下 30 步的检查时机一致。"""

    assert CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS * POLICY_ACTION_STEPS == 30


def test_failed_repulsion_is_off_by_default() -> None:
    """官方 ``mbr_use_failed_repulsion`` 默认 False。"""

    assert CYCLE_MBR_USE_FAILED_REPULSION is False


def test_official_budget_multiplier_matches_paper() -> None:
    """官方 episode 预算 = max_steps * 1.5 + num_steps_wait。"""

    assert CYCLE_OFFICIAL_BUDGET_MULTIPLIER == pytest.approx(1.5)
    assert CYCLE_NUM_STEPS_WAIT == 10


def test_cumulative_features_accept_real_initial_pose() -> None:
    """传入真实初始位姿后，特征应以该位姿为原点积分。"""

    chunk = torch.zeros(1, 5, 7)
    chunk[0, :, 0] = 1.0  # 每步 +1m 的 x 位移

    origin = cumulative_pose_trajectory_features(
        chunk, action_steps=5, initial_pose=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    )
    shifted = cumulative_pose_trajectory_features(
        chunk, action_steps=5, initial_pose=((10.0, 20.0, 30.0), (1.0, 0.0, 0.0, 0.0))
    )

    # 特征布局为 step-major：每步 6 维 = 位置(3) + 旋转(3)。
    shifted_steps = shifted[0].reshape(5, 6)
    origin_steps = origin[0].reshape(5, 6)
    assert shifted_steps[0, :3].tolist() == pytest.approx([11.0, 20.0, 30.0])
    for shifted_row, origin_row in zip(
        shifted_steps[:, 3:].tolist(), origin_steps[:, 3:].tolist(), strict=True
    ):
        assert shifted_row == pytest.approx(origin_row)
    # 位置平移差恒等于初始位置差。
    delta = (shifted_steps[:, :3] - origin_steps[:, :3]).tolist()
    for row in delta:
        assert row == pytest.approx([10.0, 20.0, 30.0])


def test_mbr_selection_is_translation_invariant() -> None:
    """候选轨迹整体平移不应改变 MBR 选择（距离矩阵不变）。"""

    torch.manual_seed(0)
    chunks = torch.randn(4, 5, 7)
    base = select_mbr_chunk(chunks, action_steps=5)[1]
    shifted = select_mbr_chunk(
        chunks,
        action_steps=5,
        initial_pose=((5.0, -3.0, 2.0), (1.0, 0.0, 0.0, 0.0)),
    )[1]

    assert base.selected_index == shifted.selected_index


def test_official_1p5x_budget_mode(monkeypatch) -> None:
    """``official_1p5x`` 返回 ``max_steps * 1.5 + num_steps_wait``。"""

    from jitpi05.jitrl import rollout

    monkeypatch.setattr(rollout, "CYCLE_BUDGET_MODE", "official_1p5x")
    assert rollout.evaluation_budget_max_steps({"max_steps": 400}) == 610
    assert rollout.evaluation_budget_max_steps({"max_steps": 520}) == 790
