"""Seed-level and task-level statistics for JitRL evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jitpi05.config import (
    JITRL_BOOTSTRAP_CONFIDENCE,
    JITRL_BOOTSTRAP_SAMPLES,
)

SCALAR_METRICS = (
    "success_rate",
    "final_10_success_rate",
    "mean_success_steps",
    "workspace_active_rate",
    "masked_stop_candidate_rate",
    "stop_selection_rate",
    "qwen_stop_termination_rate",
    "max_steps_termination_rate",
    "neighbor_action_coverage_rate",
    "free_action_match_rate",
    "free_unseen_rate",
    "nonzero_advantage_rate",
    "choice_change_rate",
    "ucb_application_rate",
    "mean_base_entropy",
    "mean_updated_entropy",
    "mean_update_kl",
    "mean_evaluator_score",
    "negative_evaluator_score_rate",
    "zero_evaluator_score_rate",
    "positive_evaluator_score_rate",
    "mean_absolute_logit_shift",
    "mean_neighbor_count",
    "memory_size",
    "cycle_enabled_rate",
    "cycle_check_count",
    "cycle_backtrack_count",
    "cycle_backtrack_rate",
    "cycle_veto_count",
    "cycle_veto_rate",
    "cycle_retry_count",
    "cycle_self_target_request_count",
    "cycle_self_target_request_rate",
    "cycle_replan_count",
    "cycle_replan_rate",
    "cycle_proxy_stop_finish_count",
    "cycle_success_after_backtrack_rate",
    "cycle_failure_after_backtrack_rate",
    "mbr_hypothesis_count",
    "mean_mbr_selected_risk",
    "mean_mbr_pairwise_distance",
    "mean_cycle_rewind_steps",
    "cycle_rewind_step_rate",
    "max_cycle_rewind_waypoint_qpos_delta",
    "mean_cycle_wall_time_seconds",
    "strict_budget_success_rate",
    "dynamic_budget_success_rate",
    "mean_cycle_extra_budget_granted_steps",
    "mean_cycle_extra_budget_used_steps",
)
# Success rate remains an auxiliary health metric. The primary JitRL effect is
# positive when successful episodes use fewer steps than the Static baseline.
PRIMARY_EFFECT_METRIC = "success_step_reduction"
PAIRED_METRICS = (
    PRIMARY_EFFECT_METRIC,
    "success_rate",
    # All methods have this field under the common 800-step protocol. It is
    # the fair success comparison once Cycle recovery may use extra steps.
    "strict_budget_success_rate",
    "final_10_success_rate",
)
# Free-candidate methods do not use the fixed nine-action workspace.
_FREE_METHODS = (
    "jitrl-free",
    "static-free",
    "jitrl-free-cycle",
    "static-free-cycle",
)
# Comparison names for the historical fixed-workspace/free-candidate pairs plus
# the Cycle learner effect and the two within-learner Cycle main effects.
PAIRED_COMPARISONS = (
    "jitrl-static",
    "jitrl-free-static-free",
    "jitrl-free-cycle-static-free-cycle",
    "cycle-static-free",
    "cycle-jitrl-free",
)
# (left_method, right_method, comparison_name) triples used for paired diffs.
_PAIRED_SPECS = (
    ("static", "jitrl", "jitrl-static"),
    ("static-free", "jitrl-free", "jitrl-free-static-free"),
    ("static-free-cycle", "jitrl-free-cycle", "jitrl-free-cycle-static-free-cycle"),
    ("static-free", "static-free-cycle", "cycle-static-free"),
    ("jitrl-free", "jitrl-free-cycle", "cycle-jitrl-free"),
)
WARNINGS = (
    "Success rate is retained as an auxiliary health metric; the primary "
    "JitRL effect metric is successful-episode step reduction versus Static.",
    "Cycle runs separately report strict_budget_success_rate under the common "
    "800-step cap and dynamic_budget_success_rate after recovery-only extra "
    "steps. Only the strict metric is used for cross-method paired success comparisons.",
    "Per task/method, the default design has only 1 seed-level run; sample "
    "standard deviation is 0 and the seed-level bootstrap interval collapses "
    "to the observed value, so neither supports across-seed inference.",
    "Episodes within one JitRL run are sequentially dependent through online "
    "memory adaptation and must not be treated as independent replicates.",
    "The default benchmark uses the complete 10-task LIBERO-10 long-horizon "
    "suite, but a CLI run may intentionally select a subset; results still use "
    "one seed and task-level intervals are descriptive.",
)


def write_json_atomic(path: Path, value: Any) -> None:
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
    termination_modes = {
        str(episode.get("termination_mode", "qwen_stop_v1")) for episode in episodes
    }
    if len(termination_modes) > 1:
        raise ValueError("all episodes in one run must share one termination mode")
    termination_mode = next(iter(termination_modes), "unknown")
    termination_reasons = [
        str(episode.get("termination_reason", "unknown")) for episode in episodes
    ]

    moving_success_rate_10 = [
        _rate(
            sum(successes[max(0, end - 10) : end]),
            end - max(0, end - 10),
        )
        for end in range(1, episode_count + 1)
    ]
    final_window = successes[-min(10, episode_count) :]
    successful_steps = [
        episode_steps for success, episode_steps in zip(successes, steps) if success
    ]

    chunks = [chunk for episode in episodes for chunk in episode.get("chunks", [])]
    cycle_episodes = [
        episode for episode in episodes if episode.get("cycle_enabled", False)
    ]
    cycle_checks = sum(int(episode.get("cycle_checks", 0)) for episode in episodes)
    cycle_backtracks = sum(
        int(episode.get("cycle_backtracks", 0)) for episode in episodes
    )
    cycle_vetoes = sum(int(episode.get("cycle_vetoes", 0)) for episode in episodes)
    cycle_retries = sum(int(episode.get("cycle_retries", 0)) for episode in episodes)
    cycle_self_target_requests = sum(
        int(episode.get("cycle_self_target_request_count", 0))
        for episode in cycle_episodes
    )
    cycle_replans = sum(
        int(episode.get("cycle_replan_count", 0)) for episode in cycle_episodes
    )
    cycle_proxy_stop_finishes = sum(
        int(episode.get("cycle_proxy_stop_finish_count", 0))
        for episode in cycle_episodes
    )
    mbr_hypotheses = sum(
        int(episode.get("mbr_hypothesis_count", 0)) for episode in episodes
    )
    mbr_risks = [
        float(event["selected_risk"])
        for episode in episodes
        for event in episode.get("mbr_events", [])
        if event.get("selected_risk") is not None
    ]
    mbr_pairwise_distances = [
        float(event["mean_pairwise_distance"])
        for episode in episodes
        for event in episode.get("mbr_events", [])
        if event.get("mean_pairwise_distance") is not None
    ]
    cycle_rewind_steps = sum(
        int(episode.get("cycle_rewind_steps", 0)) for episode in cycle_episodes
    )
    cycle_rewind_waypoint_deltas = [
        float(event["max_waypoint_qpos_delta"])
        for episode in cycle_episodes
        for event in episode.get("recovery_events", [])
        if event.get("event") == "backtrack"
        and event.get("max_waypoint_qpos_delta") is not None
    ]
    cycle_wall_times = [
        float(episode.get("cycle_wall_time_seconds", 0.0))
        for episode in cycle_episodes
    ]
    # Strict success is the common, fair comparison outcome. Non-Cycle
    # rollouts run exclusively under the unified 800-step cap, so their normal
    # success field is exactly their strict-budget result.
    strict_budget_successes = sum(
        bool(episode.get("strict_budget_success", episode.get("success", False)))
        for episode in episodes
    )
    dynamic_budget_successes = sum(
        bool(episode.get("dynamic_budget_success", episode.get("success", False)))
        for episode in cycle_episodes
    )
    cycle_extra_budget_granted = [
        int(episode.get("cycle_extra_budget_granted_steps", 0))
        for episode in cycle_episodes
    ]
    cycle_extra_budget_used = [
        int(episode.get("cycle_extra_budget_used_steps", 0))
        for episode in cycle_episodes
    ]
    backtracked_episodes = [
        episode for episode in episodes if int(episode.get("cycle_backtracks", 0)) > 0
    ]
    cycle_success_after_backtrack_count = sum(
        bool(episode.get("success", False)) for episode in backtracked_episodes
    )
    cycle_failure_after_backtrack_count = (
        len(backtracked_episodes) - cycle_success_after_backtrack_count
    )
    candidate_count = 0
    seen_count = 0
    nonzero_advantage_count = 0
    absolute_logit_shifts: list[float] = []
    neighbor_counts: list[float] = []
    base_entropies: list[float] = []
    updated_entropies: list[float] = []
    update_kls: list[float] = []
    evaluator_scores: list[int] = []
    changed_count = 0
    ucb_applied_count = 0
    masked_stop_candidate_count = 0
    stop_selection_count = 0

    for chunk in chunks:
        value_estimate = chunk.get("value_estimate", {})
        candidates = value_estimate.get("candidates", [])
        candidate_count += len(candidates)
        seen_count += sum(
            bool(candidate.get("seen", False)) for candidate in candidates
        )
        nonzero_advantage_count += sum(
            abs(float(candidate.get("normalized_advantage", 0.0))) > 1e-12
            for candidate in candidates
        )
        logit_shifts = chunk.get("logit_shifts", [])
        if len(logit_shifts) != len(candidates):
            raise ValueError("each chunk must have one logit shift per candidate slot")
        absolute_logit_shifts.extend(abs(float(shift)) for shift in logit_shifts)
        neighbor_counts.append(float(value_estimate.get("neighbor_count", 0.0)))
        base_entropies.append(float(chunk.get("base_entropy", 0.0)))
        updated_entropies.append(float(chunk.get("updated_entropy", 0.0)))
        update_kls.append(float(chunk.get("update_kl", 0.0)))
        changed_count += bool(chunk.get("choice_changed", False))
        ucb_applied_count += sum(
            bool(candidate.get("exploration_applied", False))
            for candidate in candidates
        )
        qwen_candidates = chunk.get("qwen_candidates", chunk.get("candidates", []))
        deployed_candidates = chunk.get("candidates", [])
        qwen_enabled_stop = any(
            candidate.get("type") == "stop" for candidate in qwen_candidates
        )
        deployed_stop = any(
            candidate.get("type") == "stop" for candidate in deployed_candidates
        )
        masked_stop_candidate_count += qwen_enabled_stop and not deployed_stop
        stop_selection_count += (
            chunk.get("selected_candidate", {}).get("type") == "stop"
        )
        if "evaluator_score" in chunk:
            evaluator_scores.append(int(chunk["evaluator_score"]))

    return {
        "task": dict(task_spec),
        "task_name": str(task_spec["name"]),
        "method": method,
        "seed": int(seed),
        "termination_mode": termination_mode,
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
        "workspace_active_rate": (
            None
            if method in _FREE_METHODS
            else _rate(candidate_count, 9 * len(chunks))
        ),
        "masked_stop_candidate_rate": _rate(masked_stop_candidate_count, len(chunks)),
        "stop_selection_rate": _rate(stop_selection_count, len(chunks)),
        "qwen_stop_termination_rate": _rate(
            sum(reason == "qwen_stop" for reason in termination_reasons), episode_count
        ),
        "max_steps_termination_rate": _rate(
            sum(reason == "max_steps" for reason in termination_reasons), episode_count
        ),
        "neighbor_action_coverage_rate": _rate(seen_count, candidate_count),
        "free_action_match_rate": (
            _rate(seen_count, candidate_count) if method in _FREE_METHODS else None
        ),
        "free_unseen_rate": (
            1.0 - _rate(seen_count, candidate_count)
            if method in _FREE_METHODS
            else None
        ),
        "nonzero_advantage_rate": _rate(nonzero_advantage_count, candidate_count),
        "choice_change_rate": _rate(changed_count, len(chunks)),
        "ucb_application_rate": _rate(ucb_applied_count, candidate_count),
        "mean_base_entropy": (
            float(np.mean(base_entropies)) if base_entropies else 0.0
        ),
        "mean_updated_entropy": (
            float(np.mean(updated_entropies)) if updated_entropies else 0.0
        ),
        "mean_update_kl": float(np.mean(update_kls)) if update_kls else 0.0,
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
            float(np.mean(absolute_logit_shifts)) if absolute_logit_shifts else 0.0
        ),
        "mean_neighbor_count": (
            float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
        ),
        "cycle_enabled_rate": _rate(len(cycle_episodes), episode_count),
        "cycle_check_count": float(cycle_checks),
        "cycle_backtrack_count": float(cycle_backtracks),
        "cycle_backtrack_rate": _rate(cycle_backtracks, cycle_checks),
        "cycle_veto_count": float(cycle_vetoes),
        "cycle_veto_rate": _rate(cycle_vetoes, cycle_checks),
        "cycle_retry_count": float(cycle_retries),
        "cycle_self_target_request_count": float(cycle_self_target_requests),
        "cycle_self_target_request_rate": _rate(
            cycle_self_target_requests, cycle_checks
        ),
        "cycle_replan_count": float(cycle_replans),
        "cycle_replan_rate": _rate(cycle_replans, cycle_backtracks),
        "cycle_proxy_stop_finish_count": float(cycle_proxy_stop_finishes),
        "mbr_hypothesis_count": float(mbr_hypotheses),
        "mean_mbr_selected_risk": (
            float(np.mean(mbr_risks)) if mbr_risks else None
        ),
        "mean_mbr_pairwise_distance": (
            float(np.mean(mbr_pairwise_distances))
            if mbr_pairwise_distances
            else None
        ),
        "mean_cycle_rewind_steps": (
            float(cycle_rewind_steps / len(cycle_episodes)) if cycle_episodes else None
        ),
        "cycle_rewind_step_rate": _rate(cycle_rewind_steps, sum(steps)),
        "max_cycle_rewind_waypoint_qpos_delta": (
            max(cycle_rewind_waypoint_deltas)
            if cycle_rewind_waypoint_deltas
            else None
        ),
        "mean_cycle_wall_time_seconds": (
            float(np.mean(cycle_wall_times)) if cycle_wall_times else None
        ),
        "strict_budget_success_rate": _rate(strict_budget_successes, episode_count),
        # This is intentionally undefined for non-Cycle methods rather than
        # reported as 0%; only Cycle can receive recovery-only extra steps.
        "dynamic_budget_success_rate": (
            _rate(dynamic_budget_successes, len(cycle_episodes))
            if cycle_episodes
            else None
        ),
        "mean_cycle_extra_budget_granted_steps": (
            float(np.mean(cycle_extra_budget_granted)) if cycle_episodes else None
        ),
        "mean_cycle_extra_budget_used_steps": (
            float(np.mean(cycle_extra_budget_used)) if cycle_episodes else None
        ),
        # These are within-run rescue/harm proxies, not counterfactual causal
        # claims: a true rescue/harm comparison requires the paired baseline run.
        "cycle_success_after_backtrack_rate": _rate(
            cycle_success_after_backtrack_count, len(backtracked_episodes)
        ),
        "cycle_failure_after_backtrack_rate": _rate(
            cycle_failure_after_backtrack_count, len(backtracked_episodes)
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
    write_json_atomic(run_dir / "metrics.json", metrics)
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
        ((int(seed), float(value)) for seed, value in seed_values if value is not None),
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
                namespace=(f"eval_jitrl|bootstrap|{task_name}|{method}|{metric}"),
            )
        methods[method] = method_summary

    task_summary: dict[str, Any] = {
        "task": dict(task_spec),
        "run_count": sum(method["run_count"] for method in methods.values()),
        "methods": methods,
    }
    paired_differences: dict[str, Any] = {}
    for static_method, jitrl_method, comparison in _PAIRED_SPECS:
        if (
            static_method not in metrics_by_method
            or jitrl_method not in metrics_by_method
        ):
            continue
        static_by_seed = metrics_by_method[static_method]
        jitrl_by_seed = metrics_by_method[jitrl_method]
        if not static_by_seed or set(static_by_seed) != set(jitrl_by_seed):
            continue
        paired: dict[str, Any] = {
            "comparison": comparison,
            "seeds": sorted(static_by_seed),
        }
        for metric in PAIRED_METRICS:
            differences = []
            for seed in sorted(static_by_seed):
                if metric == PRIMARY_EFFECT_METRIC:
                    static_value = static_by_seed[seed]["mean_success_steps"]
                    jitrl_value = jitrl_by_seed[seed]["mean_success_steps"]
                    # Positive means the learner completed successful episodes in
                    # fewer steps; None means one method had no successes.
                    difference = (
                        None
                        if static_value is None or jitrl_value is None
                        else float(static_value) - float(jitrl_value)
                    )
                else:
                    static_value = static_by_seed[seed][metric]
                    jitrl_value = jitrl_by_seed[seed][metric]
                    difference = float(jitrl_value) - float(static_value)
                differences.append((seed, difference))
            paired[metric] = aggregate_seed_values(
                differences,
                namespace=(
                    f"eval_jitrl|bootstrap|{task_name}|{comparison}|{metric}"
                ),
            )
        paired_differences[comparison] = paired
    if paired_differences:
        task_summary["paired_differences"] = paired_differences

    required_methods = (
        "static-free",
        "jitrl-free",
        "static-free-cycle",
        "jitrl-free-cycle",
    )
    if all(method in metrics_by_method for method in required_methods):
        seed_sets = [set(metrics_by_method[method]) for method in required_methods]
        if seed_sets and all(seed_set == seed_sets[0] for seed_set in seed_sets):
            interaction: dict[str, Any] = {
                "comparison": "cycle_x_jitrl_difference_in_differences",
                "seeds": sorted(seed_sets[0]),
            }
            for metric in PAIRED_METRICS:
                differences = []
                for seed in sorted(seed_sets[0]):
                    static_base = metrics_by_method["static-free"][seed]
                    jitrl_base = metrics_by_method["jitrl-free"][seed]
                    static_cycle = metrics_by_method["static-free-cycle"][seed]
                    jitrl_cycle = metrics_by_method["jitrl-free-cycle"][seed]
                    if metric == PRIMARY_EFFECT_METRIC:
                        static_delta = (
                            None
                            if static_base["mean_success_steps"] is None
                            or static_cycle["mean_success_steps"] is None
                            else float(static_base["mean_success_steps"])
                            - float(static_cycle["mean_success_steps"])
                        )
                        jitrl_delta = (
                            None
                            if jitrl_base["mean_success_steps"] is None
                            or jitrl_cycle["mean_success_steps"] is None
                            else float(jitrl_base["mean_success_steps"])
                            - float(jitrl_cycle["mean_success_steps"])
                        )
                    else:
                        static_delta = float(static_cycle[metric]) - float(static_base[metric])
                        jitrl_delta = float(jitrl_cycle[metric]) - float(jitrl_base[metric])
                    differences.append(
                        (
                            seed,
                            None
                            if static_delta is None or jitrl_delta is None
                            else jitrl_delta - static_delta,
                        )
                    )
                interaction[metric] = aggregate_seed_values(
                    differences,
                    namespace=(
                        f"eval_jitrl|bootstrap|{task_name}|"
                        f"cycle_x_jitrl_difference_in_differences|{metric}"
                    ),
                )
            task_summary["interaction_differences"] = interaction
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

    macro_paired: dict[str, Any] = {}
    for _, _, comparison in _PAIRED_SPECS:
        per_comparison: dict[str, Any] = {"comparison": comparison}
        for metric in PAIRED_METRICS:
            per_comparison[metric] = aggregate_task_values(
                [
                    (
                        task_name,
                        task_summary["paired_differences"][comparison][metric]["mean"],
                    )
                    for task_name, task_summary in task_summaries.items()
                    if "paired_differences" in task_summary
                    and comparison in task_summary["paired_differences"]
                ],
                namespace=f"eval_jitrl|task-macro|{comparison}|{metric}",
            )
        macro_paired[comparison] = per_comparison

    macro_interaction: dict[str, Any] = {
        "comparison": "cycle_x_jitrl_difference_in_differences"
    }
    for metric in PAIRED_METRICS:
        macro_interaction[metric] = aggregate_task_values(
            [
                (
                    task_name,
                    task_summary["interaction_differences"][metric]["mean"],
                )
                for task_name, task_summary in task_summaries.items()
                if "interaction_differences" in task_summary
            ],
            namespace=(
                f"eval_jitrl|task-macro|"
                f"cycle_x_jitrl_difference_in_differences|{metric}"
            ),
        )

    return {
        "primary_effect_metric": PRIMARY_EFFECT_METRIC,
        "primary_effect_definition": (
            "Static mean successful-episode steps minus JitRL mean "
            "successful-episode steps; positive values favor JitRL."
        ),
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
            "interaction_differences": macro_interaction,
        },
        "warnings": list(WARNINGS),
    }
