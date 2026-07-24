import json

import torch

from data import load_episode
from pi05_pipeline import run_pi05_comparison
from planner import json_planning_result, run_high_level_planner
from settings import (
    BIAS_PHRASE,
    BIAS_VALUE,
    DATASET_ID,
    EPISODE_INDEX,
    ORACLE_SUBTASK,
    OUTPUT_DIR,
    PI05_ID,
    QWEN_ID,
)


def save_artifacts(
    plain_plan: dict,
    modulated_plan: dict,
    bias_target_ids: list[int],
    actions: dict[str, torch.Tensor],
    pi05_result: dict,
    demonstration: torch.Tensor,
    initial_noise: torch.Tensor,
    overall_task: str,
) -> None:
    planning_tensors = {
        "plain": {
            "generated_ids": plain_plan["generated_ids"],
            "raw_logits": plain_plan["raw_logits"],
            "modified_logits": plain_plan["modified_logits"],
        },
        "modulated": {
            "generated_ids": modulated_plan["generated_ids"],
            "raw_logits": modulated_plan["raw_logits"],
            "modified_logits": modulated_plan["modified_logits"],
        },
    }
    torch.save(planning_tensors, OUTPUT_DIR / "planning_logits.pt")
    torch.save(
        {
            "actions": actions,
            "demonstration": demonstration,
            "initial_noise": initial_noise,
            "traces": pi05_result["traces"],
        },
        OUTPUT_DIR / "pi05_actions.pt",
    )

    summary = {
        "models": {"planner": QWEN_ID, "policy": PI05_ID},
        "dataset": {"repo_id": DATASET_ID, "episode_index": EPISODE_INDEX},
        "overall_task": overall_task,
        "oracle_subtask": ORACLE_SUBTASK,
        "plain_planning": json_planning_result(plain_plan),
        "modulated_planning": json_planning_result(modulated_plan),
        "logits_bias": {
            "phrase": BIAS_PHRASE,
            "bias": BIAS_VALUE,
            "target_token_ids": bias_target_ids,
        },
        "pi05_conditions": pi05_result["conditions"],
        "metrics": pi05_result["metrics"],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved research artifacts to {OUTPUT_DIR.resolve()}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, frame, demonstration = load_episode()
    print(f"Task: {frame['task']}")
    print(f"Episode frames: {len(dataset)}")

    plain_plan, modulated_plan, bias_target_ids = run_high_level_planner(frame)
    actions, pi05_result, initial_noise = run_pi05_comparison(
        dataset,
        frame,
        demonstration,
        plain_plan["text"],
        modulated_plan["text"],
    )
    save_artifacts(
        plain_plan,
        modulated_plan,
        bias_target_ids,
        actions,
        pi05_result,
        demonstration,
        initial_noise,
        frame["task"],
    )


if __name__ == "__main__":
    main()
