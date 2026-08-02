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
