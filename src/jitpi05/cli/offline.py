import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from jitpi05.config import (
    BIAS_PHRASE,
    BIAS_VALUE,
    DATASET_ID,
    EPISODE_INDEX,
    ORACLE_SUBTASK,
    OUTPUT_DIR,
    PI05_ID,
    QWEN_ID,
)
from jitpi05.datasets import load_episode
from jitpi05.planning import json_planning_result, run_high_level_planner
from jitpi05.policy import run_pi05_comparison


def save_artifacts(
    plain_plan: dict,
    modulated_plan: dict,
    bias_target_ids: list[int],
    actions: dict[str, torch.Tensor],
    pi05_result: dict,
    demonstration: torch.Tensor,
    initial_noise: torch.Tensor,
    overall_task: str,
    output_dir: Path,
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
    torch.save(planning_tensors, output_dir / "planning_logits.pt")
    torch.save(
        {
            "actions": actions,
            "demonstration": demonstration,
            "initial_noise": initial_noise,
            "traces": pi05_result["traces"],
        },
        output_dir / "pi05_actions.pt",
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved research artifacts to {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline JitPi05 comparison.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

    output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir,
    )


if __name__ == "__main__":
    main()
