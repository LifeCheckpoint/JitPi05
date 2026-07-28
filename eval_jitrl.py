"""CLI and seed-level statistics for JitRL evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from settings import (
    JITRL_BOOTSTRAP_CONFIDENCE,
    JITRL_BOOTSTRAP_SAMPLES,
    JITRL_EPISODES,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_SEEDS,
    JITRL_TASKS,
)

SCALAR_METRICS = (
    "success_rate",
    "final_10_success_rate",
    "mean_success_steps",
    "neighbor_action_coverage_rate",
    "nonzero_advantage_rate",
    "choice_change_rate",
    "augmentation_choice_change_rate",
    "ucb_application_rate",
    "memory_only_candidate_rate",
    "memory_only_selection_rate",
    "mean_evaluator_score",
    "negative_evaluator_score_rate",
    "zero_evaluator_score_rate",
    "positive_evaluator_score_rate",
    "mean_absolute_logit_shift",
    "mean_neighbor_count",
    "memory_size",
)
PAIRED_METRICS = (
    "success_rate",
    "final_10_success_rate",
    "mean_success_steps",
)
WARNINGS = (
    "Per task/method, the default design has only 1 seed-level run; sample "
    "standard deviation is 0 and the seed-level bootstrap interval collapses "
    "to the observed value, so neither supports across-seed inference.",
    "Episodes within one JitRL run are sequentially dependent through online "
    "memory adaptation and must not be treated as independent replicates.",
    "The five LIBERO-90 tasks were deliberately selected rather than randomly "
    "sampled; task-macro means and task-level bootstrap intervals are descriptive "
    "for this panel and do not establish suite-wide generalization.",
)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_run_metrics(
    episodes: Sequence[dict[str, Any]],
    memory: Sequence[Any],
    *,
    task_spec: dict[str, Any],
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Compute one task/method/seed record from complete episode chunk traces."""

    successes = [bool(episode["success"]) for episode in episodes]
    steps = [int(episode["steps"]) for episode in episodes]
    episode_count = len(successes)

    moving_success_rate_10 = [
        _rate(
            sum(successes[max(0, end - 10) : end]),
            end - max(0, end - 10),
        )
        for end in range(1, episode_count + 1)
    ]
    final_window = successes[-min(10, episode_count) :]
    successful_steps = [
        episode_steps
        for success, episode_steps in zip(successes, steps)
        if success
    ]

    chunks = [
        chunk
        for episode in episodes
        for chunk in episode.get("chunks", [])
    ]
    candidate_count = 0
    seen_count = 0
    nonzero_advantage_count = 0
    absolute_logit_shifts: list[float] = []
    neighbor_counts: list[float] = []
    evaluator_scores: list[int] = []
    changed_count = 0
    augmentation_changed_count = 0
    ucb_applied_count = 0
    memory_candidate_count = 0
    memory_selected_count = 0

    for chunk in chunks:
        value_estimate = chunk.get("value_estimate", {})
        candidates = value_estimate.get("candidates", [])
        candidate_count += len(candidates)
        seen_count += sum(bool(candidate.get("seen", False)) for candidate in candidates)
        nonzero_advantage_count += sum(
            abs(float(candidate.get("normalized_advantage", 0.0))) > 1e-12
            for candidate in candidates
        )
        logit_shifts = chunk.get("logit_shifts", [])
        if len(logit_shifts) != len(candidates):
            raise ValueError(
                "each chunk must have one logit shift per candidate slot"
            )
        absolute_logit_shifts.extend(abs(float(shift)) for shift in logit_shifts)
        neighbor_counts.append(float(value_estimate.get("neighbor_count", 0.0)))
        changed_count += bool(chunk.get("choice_changed", False))
        augmentation_changed_count += bool(
            chunk.get("augmentation_changed_choice", False)
        )
        ucb_applied_count += sum(
            bool(candidate.get("exploration_applied", False))
            for candidate in candidates
        )
        memory_candidate_count += sum(
            candidate.get("source") == "memory"
            for candidate in chunk.get("candidates", [])
        )
        memory_selected_count += chunk.get("selected_source") == "memory"
        if "evaluator_score" in chunk:
            evaluator_scores.append(int(chunk["evaluator_score"]))

    return {
        "task": dict(task_spec),
        "task_name": str(task_spec["name"]),
        "method": method,
        "seed": int(seed),
        "episodes": episode_count,
        "successes": successes,
        "steps": steps,
        "success_rate": _rate(sum(successes), episode_count),
        "final_10_success_rate": _rate(sum(final_window), len(final_window)),
        "moving_success_rate_10": moving_success_rate_10,
        "mean_success_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "memory_size": len(memory),
        "neighbor_action_coverage_rate": _rate(seen_count, candidate_count),
        "nonzero_advantage_rate": _rate(
            nonzero_advantage_count, candidate_count
        ),
        "choice_change_rate": _rate(changed_count, len(chunks)),
        "augmentation_choice_change_rate": _rate(
            augmentation_changed_count, len(chunks)
        ),
        "ucb_application_rate": _rate(ucb_applied_count, candidate_count),
        "memory_only_candidate_rate": _rate(memory_candidate_count, candidate_count),
        "memory_only_selection_rate": _rate(memory_selected_count, len(chunks)),
        "mean_evaluator_score": (
            float(np.mean(evaluator_scores)) if evaluator_scores else None
        ),
        # Retained for backward-compatible summaries of signed-reward artifacts.
        "negative_evaluator_score_rate": _rate(
            sum(score < 0 for score in evaluator_scores), len(evaluator_scores)
        ),
        "zero_evaluator_score_rate": _rate(
            sum(score == 0 for score in evaluator_scores), len(evaluator_scores)
        ),
        "positive_evaluator_score_rate": _rate(
            sum(score > 0 for score in evaluator_scores), len(evaluator_scores)
        ),
        "mean_absolute_logit_shift": (
            float(np.mean(absolute_logit_shifts))
            if absolute_logit_shifts
            else 0.0
        ),
        "mean_neighbor_count": (
            float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
        ),
    }


def run_dir_for(
    output_dir: Path,
    task_spec: dict[str, Any],
    method: str,
    seed: int,
) -> Path:
    """Return the collision-free artifact directory for one experimental run."""

    return output_dir / str(task_spec["name"]) / method / f"seed_{int(seed)}"


def load_and_write_run_metrics(
    output_dir: Path,
    task_spec: dict[str, Any],
    method: str,
    seed: int,
) -> dict[str, Any]:
    run_dir = run_dir_for(output_dir, task_spec, method, seed)
    episodes_path = run_dir / "episodes.json"
    memory_path = run_dir / "memory.json"
    episodes = _read_json(episodes_path)
    memory = _read_json(memory_path)
    if not isinstance(episodes, list):
        raise TypeError(f"{episodes_path} must contain a JSON list")
    if not isinstance(memory, list):
        raise TypeError(f"{memory_path} must contain a JSON list")

    metrics = compute_run_metrics(
        episodes,
        memory,
        task_spec=task_spec,
        method=method,
        seed=seed,
    )
    _atomic_write_json(run_dir / "metrics.json", metrics)
    return metrics


def _stable_seed(namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def aggregate_seed_values(
    seed_values: Sequence[tuple[int, float | int | None]],
    *,
    namespace: str,
) -> dict[str, Any]:
    """Aggregate non-null seed-level values with a reproducible bootstrap."""

    retained = sorted(
        (
            (int(seed), float(value))
            for seed, value in seed_values
            if value is not None
        ),
        key=lambda item: item[0],
    )
    seeds = [seed for seed, _ in retained]
    values = [value for _, value in retained]
    count = len(values)

    if count == 0:
        mean = None
        sample_std = 0.0
        interval: list[float | None] = [None, None]
    else:
        mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1)) if count >= 2 else 0.0
        if count == 1:
            interval = [values[0], values[0]]
        else:
            generator = np.random.default_rng(_stable_seed(namespace))
            samples = np.asarray(values, dtype=np.float64)[
                generator.integers(
                    0,
                    count,
                    size=(JITRL_BOOTSTRAP_SAMPLES, count),
                )
            ].mean(axis=1)
            alpha = 1.0 - JITRL_BOOTSTRAP_CONFIDENCE
            lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
            interval = [float(lower), float(upper)]

    return {
        "seeds": seeds,
        "values": values,
        "n": count,
        "mean": mean,
        "sample_std": sample_std,
        "seed_level_bootstrap_ci": interval,
    }


def aggregate_task_values(
    task_values: Sequence[tuple[str, float | int | None]],
    *,
    namespace: str,
) -> dict[str, Any]:
    """Aggregate one value per deliberately selected task for descriptive macros."""

    retained = [
        (str(task_name), float(value))
        for task_name, value in task_values
        if value is not None
    ]
    task_names = [task_name for task_name, _ in retained]
    values = [value for _, value in retained]
    count = len(values)
    if count == 0:
        mean = None
        sample_std = 0.0
        interval: list[float | None] = [None, None]
    else:
        mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1)) if count >= 2 else 0.0
        if count == 1:
            interval = [values[0], values[0]]
        else:
            generator = np.random.default_rng(_stable_seed(namespace))
            samples = np.asarray(values, dtype=np.float64)[
                generator.integers(
                    0,
                    count,
                    size=(JITRL_BOOTSTRAP_SAMPLES, count),
                )
            ].mean(axis=1)
            alpha = 1.0 - JITRL_BOOTSTRAP_CONFIDENCE
            lower, upper = np.quantile(
                samples,
                [alpha / 2.0, 1.0 - alpha / 2.0],
            )
            interval = [float(lower), float(upper)]
    return {
        "tasks": task_names,
        "values": values,
        "n": count,
        "mean": mean,
        "sample_std": sample_std,
        "descriptive_task_bootstrap_ci": interval,
    }


def _build_one_task_summary(
    run_metrics: Sequence[dict[str, Any]],
    *,
    task_spec: dict[str, Any],
    requested_methods: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    task_name = str(task_spec["name"])
    methods: dict[str, Any] = {}
    metrics_by_method: dict[str, dict[int, dict[str, Any]]] = {}

    for method in requested_methods:
        method_runs = sorted(
            (
                row
                for row in run_metrics
                if row["task_name"] == task_name and row["method"] == method
            ),
            key=lambda row: int(row["seed"]),
        )
        if not method_runs:
            continue
        by_seed = {int(row["seed"]): row for row in method_runs}
        metrics_by_method[method] = by_seed
        method_summary: dict[str, Any] = {
            "seeds": sorted(by_seed),
            "run_count": len(method_runs),
            "runs": [
                {
                    "seed": int(row["seed"]),
                    "episodes": int(row["episodes"]),
                    "metrics_path": str(
                        run_dir_for(
                            output_dir,
                            task_spec,
                            method,
                            int(row["seed"]),
                        )
                        / "metrics.json"
                    ),
                }
                for row in method_runs
            ],
        }
        for metric in SCALAR_METRICS:
            method_summary[metric] = aggregate_seed_values(
                [(int(row["seed"]), row[metric]) for row in method_runs],
                namespace=(
                    f"eval_jitrl|bootstrap|{task_name}|{method}|{metric}"
                ),
            )
        methods[method] = method_summary

    task_summary: dict[str, Any] = {
        "task": dict(task_spec),
        "run_count": sum(method["run_count"] for method in methods.values()),
        "methods": methods,
    }
    if "static" in metrics_by_method and "jitrl" in metrics_by_method:
        static_by_seed = metrics_by_method["static"]
        jitrl_by_seed = metrics_by_method["jitrl"]
        if static_by_seed and set(static_by_seed) == set(jitrl_by_seed):
            paired: dict[str, Any] = {
                "comparison": "jitrl-static",
                "seeds": sorted(static_by_seed),
            }
            for metric in PAIRED_METRICS:
                differences = []
                for seed in sorted(static_by_seed):
                    static_value = static_by_seed[seed][metric]
                    jitrl_value = jitrl_by_seed[seed][metric]
                    difference = (
                        None
                        if static_value is None or jitrl_value is None
                        else float(jitrl_value) - float(static_value)
                    )
                    differences.append((seed, difference))
                paired[metric] = aggregate_seed_values(
                    differences,
                    namespace=(
                        f"eval_jitrl|bootstrap|{task_name}|jitrl-static|{metric}"
                    ),
                )
            task_summary["paired_differences"] = paired
    return task_summary


def build_summary(
    run_metrics: Sequence[dict[str, Any]],
    *,
    requested_tasks: Sequence[dict[str, Any]],
    requested_methods: Sequence[str],
    requested_seeds: Sequence[int],
    requested_episodes: int,
    output_dir: Path,
) -> dict[str, Any]:
    task_summaries = {
        str(task_spec["name"]): _build_one_task_summary(
            run_metrics,
            task_spec=task_spec,
            requested_methods=requested_methods,
            output_dir=output_dir,
        )
        for task_spec in requested_tasks
    }
    macro_methods: dict[str, Any] = {}
    for method in requested_methods:
        available_tasks = [
            (task_name, task_summary["methods"][method])
            for task_name, task_summary in task_summaries.items()
            if method in task_summary["methods"]
        ]
        if not available_tasks:
            continue
        macro_methods[method] = {
            metric: aggregate_task_values(
                [
                    (task_name, method_summary[metric]["mean"])
                    for task_name, method_summary in available_tasks
                ],
                namespace=f"eval_jitrl|task-macro|{method}|{metric}",
            )
            for metric in SCALAR_METRICS
        }

    macro_paired: dict[str, Any] = {"comparison": "jitrl-static"}
    for metric in PAIRED_METRICS:
        macro_paired[metric] = aggregate_task_values(
            [
                (task_name, task_summary["paired_differences"][metric]["mean"])
                for task_name, task_summary in task_summaries.items()
                if "paired_differences" in task_summary
            ],
            namespace=f"eval_jitrl|task-macro|jitrl-static|{metric}",
        )

    return {
        "configuration": {
            "requested_tasks": [dict(task) for task in requested_tasks],
            "requested_methods": list(requested_methods),
            "requested_seeds": [int(seed) for seed in requested_seeds],
            "requested_episodes_per_run": int(requested_episodes),
            "requested_runs": (
                len(requested_tasks) * len(requested_methods) * len(requested_seeds)
            ),
            "requested_rollouts": (
                len(requested_tasks)
                * len(requested_methods)
                * len(requested_seeds)
                * requested_episodes
            ),
            "run_order": "task_then_method_then_seed",
            "bootstrap_samples": JITRL_BOOTSTRAP_SAMPLES,
            "bootstrap_confidence": JITRL_BOOTSTRAP_CONFIDENCE,
        },
        "run_count": len(run_metrics),
        "tasks": task_summaries,
        "task_macro": {
            "methods": macro_methods,
            "paired_differences": macro_paired,
        },
        "warnings": list(WARNINGS),
    }


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _deduplicate(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def resolve_tasks(task_names: Sequence[str] | None) -> list[dict[str, Any]]:
    """Resolve CLI task names while preserving the configured experiment order."""

    by_name = {str(task["name"]): dict(task) for task in JITRL_TASKS}
    requested = _deduplicate(task_names or tuple(by_name))
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown task names {unknown}; expected one of {tuple(by_name)}"
        )
    return [by_name[name] for name in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JitRL evaluations and aggregate seed-level statistics."
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(str(task["name"]) for task in JITRL_TASKS),
        help="Task to run or summarize; repeat for a subset (default: all).",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=JITRL_METHODS,
        help="Method to run or summarize; repeat for multiple methods (default: all).",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Seed to run or summarize; repeat for multiple seeds (default: settings).",
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=JITRL_EPISODES,
        help=f"Episodes per run (default: {JITRL_EPISODES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=JITRL_OUTPUT_DIR,
        help=f"Artifact root (default: {JITRL_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Read existing episodes.json and memory.json without running models.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = resolve_tasks(args.task)
    methods = _deduplicate(args.method or JITRL_METHODS)
    seeds = _deduplicate(args.seed or JITRL_SEEDS)
    output_dir = Path(args.output_dir)
    total_runs = len(tasks) * len(methods) * len(seeds)
    total_rollouts = total_runs * args.episodes

    run_experiment = None
    if not args.summarize_only:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --summarize-only is used")
        from jitrl_eval import run_jitrl_experiment

        run_experiment = run_jitrl_experiment

    mode = "summarize" if args.summarize_only else "evaluate"
    tqdm.write(
        f"[experiment] mode={mode} tasks={len(tasks)} methods={len(methods)} "
        f"seeds={len(seeds)} runs={total_runs} "
        f"episodes_per_run={args.episodes} total_rollouts={total_rollouts} "
        "order=task->method->seed"
    )
    run_progress = tqdm(
        total=total_runs,
        desc="experiment runs",
        unit="run",
        dynamic_ncols=True,
        position=0,
        leave=True,
    )
    collected: list[dict[str, Any]] = []
    completed_rollouts = 0
    try:
        for task_spec in tasks:
            task_name = str(task_spec["name"])
            for method in methods:
                for seed in seeds:
                    prefix = "summarize" if args.summarize_only else "run"
                    tqdm.write(
                        f"[start] {prefix} task={task_name} method={method} "
                        f"seed={seed} episodes={args.episodes} "
                        f"rollouts={completed_rollouts}/{total_rollouts}"
                    )
                    try:
                        if run_experiment is not None:
                            run_experiment(
                                task_spec,
                                method,
                                seed,
                                args.episodes,
                                output_dir,
                            )
                        metrics = load_and_write_run_metrics(
                            output_dir,
                            task_spec,
                            method,
                            seed,
                        )
                    except Exception as error:
                        if not args.summarize_only:
                            raise
                        tqdm.write(
                            f"[skip] task={task_name} method={method} "
                            f"seed={seed}: {error}",
                            file=sys.stderr,
                        )
                        run_progress.update(1)
                        continue

                    collected.append(metrics)
                    successes = sum(metrics["successes"])
                    completed_rollouts += int(metrics["episodes"])
                    run_progress.update(1)
                    run_progress.set_postfix(
                        task=task_name,
                        method=method,
                        seed=seed,
                        rollouts=f"{completed_rollouts}/{total_rollouts}",
                        refresh=True,
                    )
                    tqdm.write(
                        f"[done] task={task_name} method={method} seed={seed} "
                        f"success={successes}/{metrics['episodes']} "
                        f"rate={metrics['success_rate']:.1%} "
                        f"rollouts={completed_rollouts}/{total_rollouts}"
                    )
    finally:
        run_progress.close()

    summary = build_summary(
        collected,
        requested_tasks=tasks,
        requested_methods=methods,
        requested_seeds=seeds,
        requested_episodes=args.episodes,
        output_dir=output_dir,
    )
    summary_path = output_dir / "summary.json"
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
