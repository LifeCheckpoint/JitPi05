from jitpi05.jitrl.memory import JitRLMemory, discounted_returns


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
