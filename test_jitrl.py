from pathlib import Path

from eval_jitrl import build_summary, compute_run_metrics, resolve_tasks, run_dir_for
from jitrl_eval import (
    _low_level_chunks_per_high_level_plan,
    build_augmented_candidates,
)
from gemini_vlm import (
    ChunkEvaluation,
    VisualProposal,
    VisualProposalCandidate,
    evaluator_prompt,
    google_provider_base_url,
    proposal_prompt,
)
from jitrl_memory import JitRLMemory, discounted_returns
from jitrl_planner import candidate_token_ids, parse_evaluator_json, selection_prompt
from settings import (
    JITRL_BETA,
    JITRL_CANDIDATES,
    JITRL_EPISODES,
    JITRL_HIGH_LEVEL_STEPS,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_PLANNER_RETRIES,
    JITRL_QWEN_ID,
    JITRL_REWARD_VERSION,
    JITRL_TASKS,
)


def test_discounted_signed_returns() -> None:
    assert discounted_returns([0.5, -1.0, 1.0], gamma=0.95) == [
        0.45249999999999996,
        -0.050000000000000044,
        1.0,
    ]


def test_augmented_candidates_center_logits_and_deduplicate() -> None:
    result = build_augmented_candidates(
        [
            {"text": "move to center", "semantic_key": "move|gripper|center"},
            {"text": "move slightly left", "semantic_key": "move|gripper|left"},
            {"text": "lower gripper", "semantic_key": "lower|gripper|center"},
            {"text": "close gripper", "semantic_key": "grasp|central-object|gripper"},
            {"text": "move toward left plate", "semantic_key": "move|held-object|left-plate"},
        ],
        [10.0, 9.0, 8.0, 7.0, 6.0],
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
    assert result["centered_generator_logits"] == [2.0, 1.0, 0.0, -1.0, -2.0]
    assert result["base_logits"] == [2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 0.0]
    assert result["candidates"][-1]["text"] == "lift mug"
    assert len(result["candidates"]) == 7


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


def test_positive_only_evaluator_schema_and_parser() -> None:
    parsed = parse_evaluator_json(
        '{"result":"mug moved closer","usefulness":"useful",'
        '"certainty":"certain","score":2}'
    )
    assert parsed["score"] == 2
    structured = ChunkEvaluation.model_validate(parsed)
    assert structured.score == 2
    assert structured.normalized_usefulness() == "useful"

    neutral = ChunkEvaluation(
        result="no useful progress",
        usefulness="neutral",
        certainty="somewhat uncertain",
        score=0,
    )
    assert neutral.normalized_usefulness() == "neutral"

    for invalid in (
        '{"result":"regressed","usefulness":"harmful",'
        '"certainty":"certain","score":-1}',
        '{"result":"no effect","usefulness":"useful",'
        '"certainty":"certain","score":0}',
    ):
        try:
            parse_evaluator_json(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("strict positive-only parser must reject invalid reward")

    try:
        ChunkEvaluation(
            result="regressed",
            usefulness="neutral",
            certainty="certain",
            score=-1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("strict positive-only schema must reject negative scores")


def test_gemini_structured_proposal_and_physical_support_prompt() -> None:
    proposal = VisualProposal.model_validate(
        {
            "state_summary": "robot=near; gripper=open; object=cup; target=plate; relations=none; stage=approach",
            "candidates": [
                {"text": "move toward the object in the center", "semantic_key": "move|gripper|center"},
                {"text": "move slightly left", "semantic_key": "move|gripper|left"},
                {"text": "lower the gripper", "semantic_key": "lower|gripper|center"},
                {"text": "close around the nearby cup", "semantic_key": "grasp|nearby-cup|gripper"},
                {"text": "carry the held object toward the left plate", "semantic_key": "move|held-object|left-plate"},
            ],
        }
    )
    assert len(proposal.candidates) == JITRL_CANDIDATES == 5
    planning_prompt = proposal_prompt("move mug")
    assert "exactly 5" in planning_prompt
    assert "spatial or directional wording" in planning_prompt
    assert "no more than two" in planning_prompt
    prompt = evaluator_prompt(
        "put mug on plate",
        "before",
        "release mug",
        "after",
        False,
        0,
        1,
    )
    assert "supported by" in prompt
    assert "no longer held" in prompt
    assert "STRICT POSITIVE-ONLY" in prompt
    assert "must never be negative" in prompt
    assert "wrong receptacle/region/compartment" in prompt
    assert "assign score 0" in prompt


def test_missing_candidate_text_is_retried_by_outer_planner_loop() -> None:
    candidate = VisualProposalCandidate.model_validate(
        {"semantic_key": "move|gripper|center"}
    )
    try:
        candidate.normalized_text()
    except ValueError as error:
        assert "text, subtask_label, or label" in str(error)
    else:
        raise AssertionError("missing candidate text must fail normalization")
    assert JITRL_PLANNER_RETRIES == 7


def test_five_candidate_selection_prompt_and_token_ids() -> None:
    candidates = [
        {"text": f"action {index}", "semantic_key": f"move|gripper|region-{index}"}
        for index in range(1, 6)
    ]
    prompt = selection_prompt("task", "state", candidates)
    assert "1, 2, 3, 4, 5" in prompt

    class Tokenizer:
        @staticmethod
        def encode(label: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [int(label) + 100]

    assert candidate_token_ids(Tokenizer()) == [101, 102, 103, 104, 105]


def test_five_task_positive_only_experiment_configuration() -> None:
    assert JITRL_QWEN_ID == "Qwen/Qwen3.5-4B"
    assert JITRL_EPISODES == 15
    assert JITRL_HIGH_LEVEL_STEPS == 30
    assert JITRL_BETA == 0.40
    assert [task["task_id"] for task in JITRL_TASKS] == [18, 53, 59, 69, 79]
    assert all(task["suite"] == "libero_90" for task in JITRL_TASKS)
    assert [task["name"] for task in JITRL_TASKS] == [
        "libero_90_task18",
        "libero_90_task53",
        "libero_90_task59",
        "libero_90_task69",
        "libero_90_task79",
    ]
    assert [task["description"] for task in JITRL_TASKS] == [
        "put the frying pan on the stove",
        "pick up the orange juice and put it in the basket",
        "pick up the tomato sauce and put it in the tray",
        "put the chocolate pudding to the left of the plate",
        "pick up the book and place it in the left compartment of the caddy",
    ]
    assert JITRL_REWARD_VERSION == (
        "gemini36flash_positive_step_score_div3_terminal_plus1_v3"
    )
    assert str(JITRL_OUTPUT_DIR) == (
        "artifacts/jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3"
    )


def test_gemini_base_url_and_method_order() -> None:
    assert (
        google_provider_base_url("https://api.ikuncode.cc/v1beta/models")
        == "https://api.ikuncode.cc"
    )
    assert JITRL_METHODS == ("jitrl", "static")


def test_high_to_low_planning_ratio() -> None:
    assert _low_level_chunks_per_high_level_plan() == 3


def test_task_resolution_and_run_paths_are_isolated() -> None:
    selected = resolve_tasks(["libero_90_task79", "libero_90_task18"])
    assert [task["task_id"] for task in selected] == [79, 18]
    first = run_dir_for(Path("artifacts/test"), selected[0], "jitrl", 17)
    second = run_dir_for(Path("artifacts/test"), selected[1], "jitrl", 17)
    assert first == Path("artifacts/test/libero_90_task79/jitrl/seed_17")
    assert second == Path("artifacts/test/libero_90_task18/jitrl/seed_17")
    assert first != second


def test_multitask_summary_keeps_task_pairs_and_macro_average() -> None:
    tasks = [dict(task) for task in JITRL_TASKS[:2]]
    rows = []
    outcomes = {
        (tasks[0]["name"], "jitrl"): True,
        (tasks[0]["name"], "static"): False,
        (tasks[1]["name"], "jitrl"): False,
        (tasks[1]["name"], "static"): False,
    }
    for task in tasks:
        for method in JITRL_METHODS:
            rows.append(
                compute_run_metrics(
                    [
                        {
                            "success": outcomes[(task["name"], method)],
                            "steps": 400,
                            "chunks": [],
                        }
                    ],
                    [],
                    task_spec=task,
                    method=method,
                    seed=17,
                )
            )

    summary = build_summary(
        rows,
        requested_tasks=tasks,
        requested_methods=JITRL_METHODS,
        requested_seeds=(17,),
        requested_episodes=15,
        output_dir=Path("artifacts/test"),
    )
    assert summary["configuration"]["requested_runs"] == 4
    assert summary["configuration"]["requested_rollouts"] == 60
    assert summary["configuration"]["run_order"] == "task_then_method_then_seed"
    assert set(summary["tasks"]) == {task["name"] for task in tasks}
    assert (
        summary["tasks"][tasks[0]["name"]]["paired_differences"]
        ["success_rate"]["mean"]
        == 1.0
    )
    assert summary["task_macro"]["methods"]["jitrl"]["success_rate"]["mean"] == 0.5
    assert (
        summary["task_macro"]["paired_differences"]["success_rate"]["mean"]
        == 0.5
    )


def test_zero_evaluator_score_rate() -> None:
    metrics = compute_run_metrics(
        [
            {
                "success": False,
                "steps": 400,
                "chunks": [
                    {
                        "value_estimate": {"candidates": [{}]},
                        "logit_shifts": [0.0],
                        "choice_changed": False,
                        "augmentation_changed_choice": False,
                        "candidates": [{"source": "generator"}],
                        "selected_source": "generator",
                        "evaluator_score": 0,
                    },
                    {
                        "value_estimate": {"candidates": [{}]},
                        "logit_shifts": [0.0],
                        "choice_changed": False,
                        "augmentation_changed_choice": False,
                        "candidates": [{"source": "generator"}],
                        "selected_source": "generator",
                        "evaluator_score": 2,
                    },
                ],
            }
        ],
        [],
        task_spec=dict(JITRL_TASKS[0]),
        method="jitrl",
        seed=17,
    )
    assert metrics["negative_evaluator_score_rate"] == 0.0
    assert metrics["zero_evaluator_score_rate"] == 0.5
    assert metrics["positive_evaluator_score_rate"] == 0.5
