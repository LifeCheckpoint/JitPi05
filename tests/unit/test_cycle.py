import pytest
import torch

from jitpi05.jitrl.cycle import (
    cumulative_trajectory_features,
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
