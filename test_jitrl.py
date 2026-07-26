from jitrl_eval import (
    _low_level_chunks_per_high_level_plan,
    build_augmented_candidates,
)
from jitrl_memory import JitRLMemory, discounted_returns
from jitrl_planner import parse_evaluator_json


def test_discounted_signed_returns() -> None:
    assert discounted_returns([0.5, -1.0, 1.0], gamma=0.95) == [
        0.45249999999999996,
        -0.050000000000000044,
        1.0,
    ]


def test_augmented_candidates_center_logits_and_deduplicate() -> None:
    result = build_augmented_candidates(
        [
            {"text": "move mug", "semantic_key": "move|mug|table"},
            {"text": "grasp mug", "semantic_key": "grasp|mug|handle"},
            {"text": "place mug", "semantic_key": "place|mug|plate"},
        ],
        [10.0, 8.0, 6.0],
        [
            {
                "id": 1,
                "action_key": "move|mug|table",
                "action_text": "approach mug",
                "similarity": 0.9,
            },
            {
                "id": 2,
                "action_key": "lift|mug|table",
                "action_text": "lift mug",
                "similarity": 0.8,
            },
            {
                "id": 3,
                "action_key": "lift|mug|table",
                "action_text": "raise mug",
                "similarity": 0.7,
            },
        ],
    )
    assert result["centered_generator_logits"] == [2.0, 0.0, -2.0]
    assert result["base_logits"] == [2.0, 0.0, -2.0, 0.0]
    assert result["candidates"][-1]["text"] == "lift mug"
    assert len(result["candidates"]) == 4


def test_ucb_known_optimistic_zero_and_empty_branches() -> None:
    memory = JitRLMemory(top_k=10)
    memory.append_episode(
        [
            {
                "state": "robot near mug stage grasp",
                "action_key": "grasp|mug|handle",
                "action_text": "grasp mug",
            },
            {
                "state": "robot near mug stage grasp",
                "action_key": "move|mug|table",
                "action_text": "move mug",
            },
        ],
        [1.0, -1.0],
        episode_index=0,
    )
    neighbors = memory.retrieve("robot near mug stage grasp")
    estimate = memory.estimate_candidate_values(
        "robot near mug stage grasp",
        ["grasp|mug|handle", "place|mug|plate", "release|mug|plate"],
        neighbors=neighbors,
        exploration_uniforms=[0.9, 0.01, 0.9],
        exploration_rate=0.05,
        ucb_alpha=5.0,
    )
    assert estimate["candidates"][0]["exploration_branch"] == "known"
    assert estimate["candidates"][1]["exploration_branch"] == "optimistic"
    assert estimate["candidates"][1]["Q"] == 2.5
    assert estimate["candidates"][2]["exploration_branch"] == "zero"

    empty = JitRLMemory().estimate_candidate_values(
        "new state",
        ["move|mug|table"],
        exploration_uniforms=[0.01],
        exploration_rate=0.05,
        ucb_alpha=5.0,
    )
    assert empty["candidates"][0]["exploration_branch"] == "empty_memory"
    assert empty["candidates"][0]["advantage"] == 0.0


def test_evaluator_schema() -> None:
    parsed = parse_evaluator_json(
        '{"result":"mug moved closer","usefulness":"useful",'
        '"certainty":"certain","score":2}'
    )
    assert parsed["score"] == 2


def test_high_to_low_planning_ratio() -> None:
    assert _low_level_chunks_per_high_level_plan() == 2
