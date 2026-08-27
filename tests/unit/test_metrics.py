from pathlib import Path

from jitpi05.jitrl.metrics import compute_run_metrics, run_dir_for


def test_run_directory_isolated_by_task_method_and_seed() -> None:
    task = {"name": "libero_90_task79"}
    assert run_dir_for(Path("artifacts/test"), task, "jitrl", 17) == Path(
        "artifacts/test/libero_90_task79/jitrl/seed_17"
    )


def test_fixed_workspace_metrics_exclude_augmentation_fields() -> None:
    metrics = compute_run_metrics(
        [
            {
                "success": False,
                "termination_mode": "environment_only_no_stop_v1",
                "termination_reason": "max_steps",
                "steps": 100,
                "chunks": [
                    {
                        "workspace": [{"enabled": True}] * 9,
                        "qwen_candidates": [
                            {"type": "place"},
                            {"type": "retract"},
                            {"type": "stop"},
                        ],
                        "candidates": [
                            {"type": "place"},
                            {"type": "retract"},
                            {"type": "align"},
                        ],
                        "selected_candidate": {"type": "retract"},
                        "value_estimate": {
                            "neighbor_count": 2,
                            "candidates": [
                                {
                                    "seen": True,
                                    "normalized_advantage": 1.0,
                                    "exploration_applied": False,
                                },
                                {
                                    "seen": False,
                                    "normalized_advantage": -0.5,
                                    "exploration_applied": False,
                                },
                                {
                                    "seen": False,
                                    "normalized_advantage": 0.0,
                                    "exploration_applied": True,
                                },
                            ],
                        },
                        "logit_shifts": [0.4, -0.2, 0.0],
                        "choice_changed": True,
                        "base_entropy": 0.9,
                        "updated_entropy": 0.8,
                        "update_kl": 0.03,
                        "evaluator_score": 2,
                    }
                ],
            }
        ],
        [],
        task_spec={"name": "task"},
        method="jitrl",
        seed=17,
    )
    assert metrics["termination_mode"] == "environment_only_no_stop_v1"
    assert metrics["workspace_active_rate"] == 1 / 3
    assert metrics["masked_stop_candidate_rate"] == 1.0
    assert metrics["stop_selection_rate"] == 0.0
    assert metrics["qwen_stop_termination_rate"] == 0.0
    assert metrics["max_steps_termination_rate"] == 1.0
    assert metrics["neighbor_action_coverage_rate"] == 1 / 3
    assert metrics["choice_change_rate"] == 1.0
    assert metrics["ucb_application_rate"] == 1 / 3
    assert metrics["mean_base_entropy"] == 0.9
    assert metrics["mean_updated_entropy"] == 0.8
    assert metrics["mean_update_kl"] == 0.03
    assert "augmentation_choice_change_rate" not in metrics
    assert "memory_only_selection_rate" not in metrics


def test_cycle_metrics_report_backtrack_and_mbr_audits() -> None:
    metrics = compute_run_metrics(
        [
            {
                "success": True,
                "termination_mode": "environment_only_no_stop_v1",
                "termination_reason": "environment_success",
                "steps": 120,
                "cycle_enabled": True,
                "cycle_checks": 2,
                "cycle_backtracks": 1,
                "cycle_vetoes": 1,
                "cycle_retries": 1,
                "cycle_self_target_request_count": 1,
                "cycle_replan_count": 1,
                "cycle_proxy_stop_finish_count": 1,
                "cycle_wall_time_seconds": 0.5,
                "cycle_rewind_steps": 20,
                "strict_budget_success": True,
                "dynamic_budget_success": True,
                "cycle_extra_budget_granted_steps": 400,
                "cycle_extra_budget_used_steps": 40,
                "recovery_events": [
                    {
                        "event": "backtrack",
                        "max_waypoint_qpos_delta": 0.04,
                    }
                ],
                "mbr_hypothesis_count": 8,
                "mbr_events": [
                    {"selected_risk": 1.25, "mean_pairwise_distance": 2.5}
                ],
                "chunks": [],
            },
            {
                "success": False,
                "termination_mode": "environment_only_no_stop_v1",
                "termination_reason": "max_steps",
                "steps": 200,
                "cycle_enabled": True,
                "cycle_checks": 1,
                "cycle_backtracks": 0,
                "cycle_vetoes": 1,
                "cycle_retries": 0,
                "cycle_wall_time_seconds": 0.25,
                "strict_budget_success": False,
                "dynamic_budget_success": True,
                "cycle_extra_budget_granted_steps": 800,
                "cycle_extra_budget_used_steps": 600,
                "mbr_hypothesis_count": 0,
                "mbr_events": [],
                "chunks": [],
            },
        ],
        [],
        task_spec={"name": "cycle-task"},
        method="jitrl-free-cycle",
        seed=17,
    )

    assert metrics["cycle_enabled_rate"] == 1.0
    assert metrics["mean_cycle_rewind_steps"] == 10.0
    assert metrics["cycle_rewind_step_rate"] == 0.0625
    assert metrics["max_cycle_rewind_waypoint_qpos_delta"] == 0.04
    assert metrics["cycle_check_count"] == 3.0
    assert metrics["cycle_backtrack_count"] == 1.0
    assert metrics["cycle_backtrack_rate"] == 1 / 3
    assert metrics["cycle_veto_count"] == 2.0
    assert metrics["cycle_veto_rate"] == 2 / 3
    assert metrics["cycle_self_target_request_count"] == 1.0
    assert metrics["cycle_self_target_request_rate"] == 1 / 3
    assert metrics["cycle_replan_count"] == 1.0
    assert metrics["cycle_replan_rate"] == 1.0
    assert metrics["cycle_proxy_stop_finish_count"] == 1.0
    assert metrics["cycle_success_after_backtrack_rate"] == 1.0
    assert metrics["cycle_failure_after_backtrack_rate"] == 0.0
    assert metrics["mbr_hypothesis_count"] == 8.0
    assert metrics["mean_mbr_selected_risk"] == 1.25
    assert metrics["mean_mbr_pairwise_distance"] == 2.5
    assert metrics["mean_cycle_wall_time_seconds"] == 0.375
    # `success_rate` is the final Cycle outcome. The two budget-scoped metrics
    # make the fair 800-step result and recovery-only result explicit.
    assert metrics["strict_budget_success_rate"] == 0.5
    assert metrics["dynamic_budget_success_rate"] == 1.0
    assert metrics["mean_cycle_extra_budget_granted_steps"] == 600.0
    assert metrics["mean_cycle_extra_budget_used_steps"] == 320.0


def test_non_cycle_metrics_keep_strict_budget_comparable_and_dynamic_undefined() -> None:
    metrics = compute_run_metrics(
        [
            {
                "success": True,
                "termination_mode": "environment_only_no_stop_v1",
                "termination_reason": "environment_success",
                "steps": 120,
                "chunks": [],
            },
            {
                "success": False,
                "termination_mode": "environment_only_no_stop_v1",
                "termination_reason": "max_steps",
                "steps": 800,
                "chunks": [],
            },
        ],
        [],
        task_spec={"name": "non-cycle-task"},
        method="static-free",
        seed=17,
    )

    assert metrics["strict_budget_success_rate"] == 0.5
    assert metrics["dynamic_budget_success_rate"] is None
