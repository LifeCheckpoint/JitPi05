import pytest
import torch

from jitpi05.jitrl.cycle import (
    CycleSignalConfirmation,
    CycleSubtask,
    adapt_cycle_policy_output,
    build_cycle_subtask_program,
    cumulative_pose_trajectory_features,
    cumulative_trajectory_features,
    format_cycle_condition,
    format_cycle_subtask_program,
    match_cycle_subtask,
    pairwise_l2_distances,
    select_mbr_chunk,
    select_mbr_medoid,
)


def test_cumulative_features_use_executed_prefix_and_six_motion_dims() -> None:
    chunks = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
                [1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 200.0],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 300.0],
            ]
        ]
    )

    features = cumulative_trajectory_features(chunks, action_steps=2)

    assert features.shape == (1, 12)
    assert features.tolist() == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    ]


def test_pairwise_l2_distances_are_symmetric_with_zero_diagonal() -> None:
    features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]])

    distances = pairwise_l2_distances(features)

    assert distances.shape == (3, 3)
    assert torch.allclose(distances, distances.T)
    assert torch.allclose(torch.diag(distances), torch.zeros(3))
    assert distances[0, 1].item() == pytest.approx(5.0)
    assert distances[0, 2].item() == pytest.approx(4.0)


def test_pose_features_compose_rotation_deltas() -> None:
    chunks = torch.zeros(1, 2, 7)
    chunks[0, 0, 3] = torch.pi / 2
    chunks[0, 1, 3] = torch.pi / 2

    features = cumulative_pose_trajectory_features(chunks, action_steps=2)

    assert features.shape == (1, 12)
    assert features[0, 3:6].tolist() == pytest.approx([torch.pi / 2, 0.0, 0.0])
    assert features[0, 9:12].tolist() == pytest.approx([torch.pi, 0.0, 0.0])


def test_mbr_accepts_failed_trajectory_features_and_exposes_scores() -> None:
    chunks = torch.zeros(3, 2, 7)
    chunks[1, :, 0] = 1.0
    chunks[2, :, 0] = 2.0
    failed = cumulative_pose_trajectory_features(chunks[0:1], action_steps=2)[0]

    result = select_mbr_medoid(
        chunks,
        action_steps=2,
        failed_trajectories=[failed],
    )

    assert len(result.scores) == 3
    assert result.selection_mode == "rep"
    assert all(torch.isfinite(torch.tensor(result.scores)))


def test_mbr_selects_medoid_and_returns_original_candidate() -> None:
    chunks = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]],
            [[1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]],
        ]
    )

    selected, result = select_mbr_chunk(chunks, action_steps=1)

    assert result.selected_index == 1
    assert torch.equal(selected, chunks[1])
    assert result.risks[1] < result.risks[0]
    assert result.risks[1] < result.risks[2]
    assert result.pairwise_distances.shape == (3, 3)


def test_mbr_tie_breaks_to_first_candidate() -> None:
    chunks = torch.zeros(2, 2, 7)
    chunks[1, :, 0] = 2.0

    result = select_mbr_medoid(chunks)

    assert result.selected_index == 0


@pytest.mark.parametrize(
    "value, message",
    [
        (torch.empty(0, 2, 7), "at least one"),
        (torch.zeros(2, 0, 7), "non-empty"),
        (torch.zeros(2, 2, 5), "six"),
        (torch.full((2, 2, 7), float("nan")), "finite"),
    ],
)
def test_mbr_rejects_invalid_action_chunks(value: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_mbr_medoid(value)


def test_mbr_rejects_non_positive_prefix() -> None:
    with pytest.raises(ValueError, match="positive"):
        cumulative_trajectory_features(torch.zeros(2, 2, 7), action_steps=0)


def test_cycle_policy_adapter_preserves_7d_and_reads_9d_signals() -> None:
    seven = adapt_cycle_policy_output(torch.zeros(1, 4, 7))
    assert seven.signal_source == "missing_7d"
    assert seven.robot_action_chunk.shape == (4, 7)
    assert seven.progress_signal(0) is None

    nine_values = torch.zeros(2, 9)
    nine_values[:, 7] = torch.tensor([0.2, 0.8])
    nine_values[:, 8] = torch.tensor([0.91, 0.95])
    nine = adapt_cycle_policy_output(nine_values)
    assert nine.signal_source == "policy_9d"
    assert nine.robot_action_chunk.shape == (2, 7)
    assert nine.stop_signal(1) == pytest.approx(0.8)
    assert nine.progress_signal(0) == pytest.approx(0.91)


def test_cycle_policy_adapter_rejects_unknown_action_dimension() -> None:
    with pytest.raises(ValueError, match="exactly 7.*9"):
        adapt_cycle_policy_output(torch.zeros(2, 8))


def test_cycle_signal_confirmation_requires_consecutive_or_gap_confirmation() -> None:
    confirmation = CycleSignalConfirmation(threshold=0.9, required_consecutive=2, required_gap=2)
    assert confirmation.update(0.95) is False
    assert confirmation.update(0.91) is True
    confirmation.reset()
    assert confirmation.update(0.95) is False
    assert confirmation.update(0.1) is False
    assert confirmation.update(0.2) is False
    assert confirmation.update(0.95) is True


def test_cycle_program_has_stable_ids_and_matches_unused_nodes() -> None:
    program = build_cycle_subtask_program("pick up the mug and place it in the basket")
    formatted = format_cycle_subtask_program(program)
    assert formatted[0].startswith("subtask-00:")
    assert formatted[-1].startswith("subtask-03:")
    matched = match_cycle_subtask(
        {"type": "grasp", "text": "grasp the mug"},
        program,
    )
    repeated = match_cycle_subtask(
        {"type": "grasp", "text": "grasp the mug"},
        program,
        used_ids=(matched.subtask_id,),
    )
    assert matched.subtask_id == "subtask-01"
    assert repeated.subtask_id == matched.subtask_id
    assert all(isinstance(item, CycleSubtask) for item in program)


def test_cycle_program_uses_official_template_text() -> None:
    program = build_cycle_subtask_program("pick up the mug and place it in the basket")
    texts = [item.text for item in program]
    assert texts == [
        "Move the gripper above the mug.",
        "Close the gripper to grasp the mug.",
        "Move the gripper above the basket while holding the mug.",
        "Open the gripper to release the mug.",
    ]
    types = [item.action_type for item in program]
    assert types == ["approach", "grasp", "transport", "release"]


def test_cycle_program_handles_put_and_complex_tasks() -> None:
    program = build_cycle_subtask_program("put the red block on the wooden cabinet")
    assert program[0].text == "Move the gripper above the red block."
    assert program[-1].action_type == "release"

    complex_program = build_cycle_subtask_program("turn on the stove")
    assert [item.text for item in complex_program] == [
        "Move the gripper above the stove knob.",
        "Close the gripper to grasp the stove knob.",
        "Rotate the gripper to turn on the stove.",
    ]
    assert [item.action_type for item in complex_program] == [
        "approach",
        "grasp",
        "rotate",
    ]


def test_cycle_condition_uses_official_prompt_template() -> None:
    condition = format_cycle_condition(
        "pick up the mug and place it in the basket",
        "Move the gripper above the mug.",
    )
    assert condition == (
        "Task: pick up the mug and place it in the basket. "
        "The current subtask: Move the gripper above the mug."
    )
