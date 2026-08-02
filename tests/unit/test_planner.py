import json
from pathlib import Path

import pytest
import torch

from jitpi05.cli.jitrl import resolve_tasks
from jitpi05.config import (
    JITRL_ACTION_WORKSPACE_VERSION,
    JITRL_BETA,
    JITRL_EPISODES,
    JITRL_HIGH_LEVEL_STEPS,
    JITRL_LOGIT_CALIBRATION,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_PLANNER_RETRIES,
    JITRL_QWEN_ID,
    JITRL_REWARD_VERSION,
    JITRL_TASKS,
    JITRL_TERMINATION_MODE,
)
from jitpi05.jitrl.memory import JitRLMemory, discounted_returns
from jitpi05.jitrl.metrics import build_summary, compute_run_metrics, run_dir_for
from jitpi05.jitrl.planner import (
    _planner_inputs,
    action_label_token_ids,
    apply_jitrl_update,
    deployment_candidates,
    generate_workspace,
    parse_evaluator_json,
    parse_workspace_json,
    selection_prompt,
    workspace_prompt,
)
from jitpi05.jitrl.rollout import _low_level_chunks_per_high_level_plan
from jitpi05.jitrl.semantic_actions import (
    ACTION_LABELS,
    ACTION_TYPES,
    action_schema_text,
    active_actions,
    instantiate_workspace,
    workspace_from_binding,
)
from jitpi05.jitrl.vlm import (
    ChunkEvaluation,
    evaluator_prompt,
    google_provider_base_url,
)


def workspace_bindings() -> list[dict]:
    return [
        {"type": "approach", "enabled": True, "target": "the mug"},
        {
            "type": "align",
            "enabled": True,
            "gripper": "the gripper",
            "target": "the mug",
        },
        {"type": "grasp", "enabled": True, "target": "the mug"},
        {"type": "lift", "enabled": False, "target": "the mug"},
        {
            "type": "transport",
            "enabled": False,
            "target": "the mug",
            "destination": "the basket",
        },
        {
            "type": "place",
            "enabled": False,
            "target": "the mug",
            "destination": "the basket",
        },
        {
            "type": "release",
            "enabled": False,
            "target": "the mug",
            "destination": "the basket",
        },
        {"type": "retract", "enabled": True, "direction": "upward"},
        {"type": "stop", "enabled": False},
    ]


def test_fixed_semantic_workspace_is_complete_and_rendered() -> None:
    workspace = instantiate_workspace(workspace_bindings())
    assert tuple(action["type"] for action in workspace) == ACTION_TYPES
    assert tuple(action["label"] for action in workspace) == tuple(
        ACTION_LABELS[action_type] for action_type in ACTION_TYPES
    )
    assert workspace[0]["text"] == "move toward the mug"
    assert workspace[4]["semantic_key"] == "transport|the-mug|the-basket"
    assert workspace[-1]["terminates_rollout"] is True
    assert all(action["terminates_rollout"] is False for action in workspace[:-1])
    assert [action["type"] for action in active_actions(workspace)] == [
        "approach",
        "align",
        "grasp",
        "retract",
    ]
    assert "9. stop()" in action_schema_text()


def test_workspace_rejects_dynamic_or_missing_action_types() -> None:
    invalid = workspace_bindings()[:-1]
    invalid.append({"type": "push", "enabled": True, "target": "the mug"})
    with pytest.raises(ValueError):
        instantiate_workspace(invalid)


def test_qwen_workspace_prompt_contains_schema_and_static_icl() -> None:
    prompt = workspace_prompt("put the mug in the basket", ["move toward the mug"])
    retry_prompt = workspace_prompt("put the mug in the basket", attempt=3)
    assert "fixed before evaluation" in prompt
    assert "must not invent" in prompt
    assert "Example 1" in prompt
    assert "approach(target)" in prompt
    assert '"enabled_actions"' in prompt
    assert '"actions"' not in prompt
    assert "action labels only, not verified facts" in prompt
    assert "current images as authoritative evidence" in prompt
    assert "never include stop" in prompt
    assert "environment-only termination diagnostic" in prompt
    assert "correction attempt 3" in retry_prompt


def test_workspace_inputs_continue_an_assistant_json_prefill() -> None:
    captured = {}

    class Inputs(dict):
        def to(self, device):
            captured["device"] = device
            return self

    class Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return Inputs(input_ids=torch.tensor([[1, 2]]))

    class Model:
        device = torch.device("cpu")

    _planner_inputs(Model(), Processor(), [], "prompt", assistant_prefill="{")
    assert captured["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "{"}],
    }
    assert captured["kwargs"]["add_generation_prompt"] is False
    assert captured["kwargs"]["continue_final_message"] is True
    assert captured["kwargs"]["enable_thinking"] is False
    assert captured["device"] == torch.device("cpu")


def compact_binding() -> dict:
    return {
        "state_summary": "robot=ready; gripper=open; object=mug; target=basket; relations=far; stage=approach",
        "object": "the mug",
        "destination": "the basket",
        "approach_target": "the mug",
        "align_target": "the mug",
        "retract_direction": "upward",
        "enabled_actions": ["approach", "align", "grasp", "retract"],
    }


def test_workspace_generation_prefills_json_and_reports_token_usage() -> None:
    continuation = json.dumps(compact_binding())[1:]
    captured = {}

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

    class Processor:
        tokenizer = Tokenizer()

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            captured["template_kwargs"] = kwargs
            return Inputs(
                input_ids=torch.tensor([[1, 2]]),
                attention_mask=torch.tensor([[1, 1]]),
            )

        @staticmethod
        def decode(token_ids, **kwargs):
            captured["decoded_token_count"] = int(token_ids.shape[0])
            return continuation

    class Model:
        device = torch.device("cpu")

        @staticmethod
        def generate(**kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 7, 8]])

    generated = generate_workspace(
        Model(),
        Processor(),
        [],
        "put the mug in the basket",
        max_new_tokens=8,
    )
    assert captured["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "{"}],
    }
    assert captured["template_kwargs"]["continue_final_message"] is True
    assert generated["binding_raw_output"] == json.dumps(compact_binding())
    assert generated["binding_assistant_prefill"] == "{"
    assert generated["binding_generated_tokens"] == 2
    assert generated["binding_generation_reached_limit"] is False


def test_workspace_json_constructs_all_fixed_actions_from_compact_binding() -> None:
    parsed = parse_workspace_json(json.dumps(compact_binding()))
    assert len(parsed["workspace"]) == 9
    assert [action["type"] for action in parsed["workspace"]] == list(ACTION_TYPES)
    assert parsed["workspace"][4]["semantic_key"] == "transport|the-mug|the-basket"
    assert parsed["termination_mode"] == "environment_only_no_stop_v1"
    assert [candidate["type"] for candidate in parsed["qwen_candidates"]] == [
        "approach",
        "align",
        "grasp",
        "retract",
    ]
    assert [candidate["type"] for candidate in parsed["candidates"]] == [
        "approach",
        "align",
        "grasp",
        "retract",
    ]
    assert all(candidate["label"] in {"1", "2", "3", "8"} for candidate in parsed["candidates"])


def test_no_stop_diagnostic_masks_stop_without_changing_fixed_workspace() -> None:
    binding = compact_binding()
    binding["enabled_actions"] = ["place", "retract", "stop"]
    parsed = parse_workspace_json(json.dumps(binding))
    assert len(parsed["workspace"]) == 9
    assert [candidate["type"] for candidate in parsed["qwen_candidates"]] == [
        "place",
        "retract",
        "stop",
    ]
    assert [candidate["type"] for candidate in parsed["candidates"]] == [
        "place",
        "retract",
    ]
    assert all(not candidate["terminates_rollout"] for candidate in parsed["candidates"])

    stop_only = compact_binding()
    stop_only["enabled_actions"] = ["stop"]
    with pytest.raises(ValueError, match="no executable actions"):
        parse_workspace_json(json.dumps(stop_only))

    workspace = workspace_from_binding(binding)
    assert [candidate["type"] for candidate in deployment_candidates(workspace)] == [
        "place",
        "retract",
    ]


def test_compact_binding_rejects_dynamic_or_duplicate_enabled_types() -> None:
    invalid = compact_binding()
    invalid["enabled_actions"] = ["approach", "push", "approach"]
    with pytest.raises(ValueError, match="unexpected=.*push.*duplicates=.*approach"):
        workspace_from_binding(invalid)


def test_selection_prompt_and_label_tokens_use_fixed_action_labels() -> None:
    candidates = active_actions(instantiate_workspace(workspace_bindings()))
    prompt = selection_prompt("task", "state", candidates)
    assert "1. approach" in prompt
    assert "8. retract" in prompt
    assert "1, 2, 3, 8" in prompt

    class Tokenizer:
        @staticmethod
        def encode(label: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [int(label) + 100]

    assert action_label_token_ids(Tokenizer(), candidates) == [101, 102, 103, 108]


def test_jitrl_update_uses_raw_qwen_logits_and_reports_policy_diagnostics() -> None:
    update = apply_jitrl_update(
        [4.0, 1.0, -2.0],
        [0.0, 1.0, -1.0],
        beta=0.4,
        temperature=0.8,
        uniform=0.7,
    )
    assert update["base_logits"] == [4.0, 1.0, -2.0]
    assert update["updated_logits"] == pytest.approx([4.0, 1.4, -2.4])
    assert update["update_kl"] >= 0.0
    assert update["base_entropy"] >= 0.0
    assert update["updated_entropy"] >= 0.0


def test_memory_estimates_only_the_supplied_qwen_action_set() -> None:
    memory = JitRLMemory(top_k=10)
    memory.append_episode(
        [
            {
                "state": "robot near mug stage grasp",
                "action_key": "grasp|the-mug|none",
                "action_text": "grasp the mug",
            },
            {
                "state": "robot near mug stage grasp",
                "action_key": "lift|the-mug|none",
                "action_text": "lift the mug",
            },
        ],
        [1.0, -1.0],
        episode_index=0,
    )
    requested = ["grasp|the-mug|none", "retract|gripper|upward"]
    estimate = memory.estimate_candidate_values(
        "robot near mug stage grasp",
        requested,
        exploration_uniforms=[0.9, 0.01],
        exploration_rate=0.05,
        ucb_alpha=5.0,
    )
    assert [candidate["action_key"] for candidate in estimate["candidates"]] == requested
    assert "lift|the-mug|none" not in [
        candidate["action_key"] for candidate in estimate["candidates"]
    ]


def test_positive_only_evaluator_schema_and_parser() -> None:
    parsed = parse_evaluator_json(
        '{"result":"mug moved closer","usefulness":"useful",'
        '"certainty":"certain","score":2}'
    )
    assert ChunkEvaluation.model_validate(parsed).score == 2
    with pytest.raises(ValueError):
        parse_evaluator_json(
            '{"result":"regressed","usefulness":"neutral",'
            '"certainty":"certain","score":-1}'
        )
    prompt = evaluator_prompt(
        "put mug on plate", "before", "release mug", "after", False, 0, 1
    )
    assert "STRICT POSITIVE-ONLY" in prompt
    assert "no longer held" in prompt


def test_experiment_configuration_is_qwen_workspace_only() -> None:
    assert JITRL_QWEN_ID == "Qwen/Qwen3.5-4B"
    assert (
        JITRL_ACTION_WORKSPACE_VERSION
        == "libero_semantic_actions_9_compact_binding_v2"
    )
    assert JITRL_TERMINATION_MODE == "environment_only_no_stop_v1"
    assert JITRL_LOGIT_CALIBRATION == "raw_qwen_fixed_workspace_v1"
    assert JITRL_EPISODES == 15
    assert JITRL_HIGH_LEVEL_STEPS == 30
    assert JITRL_BETA == 0.40
    assert JITRL_PLANNER_RETRIES == 7
    assert JITRL_OUTPUT_DIR.as_posix().endswith(
        "qwen4b_workspace_v2_no_stop_diagnostic"
    )
    assert JITRL_REWARD_VERSION == (
        "gemini36flash_positive_step_score_div3_terminal_plus1_v3"
    )


def test_gemini_is_evaluator_only_and_method_order_is_paired() -> None:
    assert google_provider_base_url("https://api.ikuncode.cc/v1beta/models") == (
        "https://api.ikuncode.cc"
    )
    assert JITRL_METHODS == ("jitrl", "static")
    assert _low_level_chunks_per_high_level_plan() == 3


def test_task_resolution_and_summary_paths() -> None:
    selected = resolve_tasks(["libero_90_task79", "libero_90_task18"])
    assert [task["task_id"] for task in selected] == [79, 18]
    first = run_dir_for(Path("artifacts/test"), selected[0], "jitrl", 17)
    assert first == Path("artifacts/test/libero_90_task79/jitrl/seed_17")

    tasks = [dict(task) for task in JITRL_TASKS[:2]]
    rows = []
    for task in tasks:
        for method in JITRL_METHODS:
            rows.append(
                compute_run_metrics(
                    [{"success": method == "jitrl", "steps": 400, "chunks": []}],
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
    assert summary["task_macro"]["paired_differences"]["success_rate"]["mean"] == 1.0


def test_discounted_signed_returns() -> None:
    assert discounted_returns([0.5, -1.0, 1.0], gamma=0.95) == [
        0.45249999999999996,
        -0.050000000000000044,
        1.0,
    ]
