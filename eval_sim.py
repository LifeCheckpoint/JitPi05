import json

import torch

from settings import SIM_OUTPUT_DIR
from sim_eval import collect_plans, run_simulation_evaluation, save_simulation_artifacts


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires an NVIDIA CUDA GPU.")

    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plans = collect_plans()
    results = run_simulation_evaluation(plans)
    summary = save_simulation_artifacts(plans, results)

    rates = {
        task_name: {
            condition: metrics["success_rate"]
            for condition, metrics in task["conditions"].items()
        }
        for task_name, task in summary["tasks"].items()
    }
    print(json.dumps(rates, indent=2, ensure_ascii=False))
    print(f"Saved simulation artifacts to {SIM_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
