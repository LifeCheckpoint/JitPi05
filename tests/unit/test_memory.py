import pytest

from jitpi05.jitrl.memory import (
    JitRLMemory,
    action_similarity,
    discounted_returns,
    normalize_action,
)


def test_discounted_returns_preserve_order() -> None:
    assert discounted_returns([1.0, 0.0, 2.0], gamma=0.5) == [1.5, 1.0, 2.0]


def test_memory_never_adds_actions_to_the_query_set() -> None:
    memory = JitRLMemory(top_k=10)
    memory.append_episode(
        [
            {
                "state": "gripper holds mug near basket",
                "action_key": "transport|mug|basket",
                "action_text": "carry mug toward basket",
            },
            {
                "state": "gripper holds mug near basket",
                "action_key": "release|mug|basket",
                "action_text": "release mug at basket",
            },
        ],
        [1.0, 0.5],
        episode_index=0,
    )
    query_actions = ["transport|mug|basket"]
    estimate = memory.estimate_candidate_values(
        "gripper holds mug near basket",
        query_actions,
    )
    assert [candidate["action_key"] for candidate in estimate["candidates"]] == (
        query_actions
    )
    assert estimate["neighbor_action_coverage"] == {
        "transport|mug|basket": 1,
        "release|mug|basket": 1,
    }


def test_normalize_action_and_similarity() -> None:
    assert normalize_action("Pick up the Mug!") == "pick up the mug"
    assert normalize_action("STOP.") == "stop"
    assert action_similarity("pick up the mug", "pick up the mug") == 1.0
    assert action_similarity("pick up the mug", "grasp the mug") < 1.0
    assert action_similarity("stop", "grasp the mug") == 0.0


def test_free_memory_matches_candidates_by_text_similarity() -> None:
    memory = JitRLMemory(top_k=10)
    memory.append_episode(
        [
            {
                "state": "robot near mug stage grasp",
                "action_key": "grasp the mug",
                "action_text": "grasp the mug",
            },
            {
                "state": "robot near mug stage grasp",
                "action_key": "carry the mug to the basket",
                "action_text": "carry the mug to the basket",
            },
        ],
        [1.0, -0.5],
        episode_index=0,
    )
    estimate = memory.estimate_candidate_values_free(
        "robot near mug stage grasp",
        ["grasp the mug", "release the mug", "open the fridge"],
        exploration_uniforms=[0.9, 0.9, 0.01],
        exploration_rate=0.05,
        ucb_alpha=5.0,
        action_sim_threshold=0.6,
    )
    assert estimate["V"] == pytest.approx(0.25)
    assert [candidate["action_key"] for candidate in estimate["candidates"]] == [
        "grasp the mug",
        "release the mug",
        "open the fridge",
    ]
    by_key = {candidate["action_key"]: candidate for candidate in estimate["candidates"]}
    assert by_key["grasp the mug"]["seen"] is True
    assert by_key["grasp the mug"]["Q"] == pytest.approx(1.0)
    assert by_key["release the mug"]["seen"] is False
    assert by_key["release the mug"]["exploration_branch"] == "zero"
    assert by_key["release the mug"]["Q"] == pytest.approx(0.0)
    assert by_key["open the fridge"]["seen"] is False
    assert by_key["open the fridge"]["exploration_branch"] == "optimistic"
    assert by_key["open the fridge"]["ucb_bonus"] == pytest.approx(2.5)
    assert by_key["open the fridge"]["Q"] == pytest.approx(2.75)
