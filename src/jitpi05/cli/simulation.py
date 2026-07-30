import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from jitpi05.config import SIM_OUTPUT_DIR
from jitpi05.simulation import (
    collect_plans,
    run_simulation_evaluation,
    save_simulation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired LIBERO simulations.")
    parser.add_argument("--output-dir", type=Path, default=SIM_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires an NVIDIA CUDA GPU.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plans = collect_plans()
    results = run_simulation_evaluation(plans, output_dir=output_dir)
    summary = save_simulation_artifacts(plans, results, output_dir=output_dir)

    rates = {
        task_name: {
            condition: metrics["success_rate"]
            for condition, metrics in task["conditions"].items()
        }
        for task_name, task in summary["tasks"].items()
    }
    print(json.dumps(rates, indent=2, ensure_ascii=False))
    print(f"Saved simulation artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
