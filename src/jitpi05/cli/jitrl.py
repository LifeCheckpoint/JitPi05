"""Command-line interface for JitRL evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from jitpi05.config import (
    JITRL_EPISODES,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_SEEDS,
    JITRL_TASKS,
)
from jitpi05.jitrl.metrics import (
    build_summary,
    load_and_write_run_metrics,
    write_json_atomic,
)


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
        from jitpi05.jitrl.rollout import run_jitrl_experiment

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
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
