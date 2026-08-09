import pytest

from jitpi05.config import (
    CYCLE_BUDGET_MODE,
    CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS,
    CYCLE_EVALUATION,
    CYCLE_JITRL_METHODS,
    CYCLE_MAX_RETRIES,
    CYCLE_MBR_HYPOTHESES,
    CYCLE_PROGRESS_THRESHOLD,
    CYCLE_RETRY_BUDGET_PER_BACKTRACK,
    CYCLE_RETRY_TOTAL_BUDGET,
)
from jitpi05.jitrl.cycle import (
    CycleGate,
    PhysicalEvidence,
    SemanticAnchor,
    proxy_subtask_stop,
    resolve_cycle_decision,
    semantic_action_type,
)


def anchors() -> list[SemanticAnchor]:
    return [
        SemanticAnchor("a0", "approach the mug", "condition-a0", 0, "approach"),
        SemanticAnchor("a1", "grasp the mug", "condition-a1", 1, "grasp"),
        SemanticAnchor("a2", "place the mug at the basket", "condition-a2", 2, "place"),
    ]


def test_cycle_configuration_keeps_four_way_comparison_opt_in() -> None:
    assert CYCLE_BUDGET_MODE == "unified_800"
    assert CYCLE_JITRL_METHODS == (
        "static-free",
        "jitrl-free",
        "static-free-cycle",
        "jitrl-free-cycle",
    )
    assert CYCLE_PROGRESS_THRESHOLD == pytest.approx(0.90)
    assert CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS == 3
    assert CYCLE_MBR_HYPOTHESES == 8
    assert 59 in CYCLE_EVALUATION.libero90_candidates
    assert (
        CYCLE_RETRY_TOTAL_BUDGET
        == CYCLE_MAX_RETRIES * CYCLE_RETRY_BUDGET_PER_BACKTRACK
    )
    assert CYCLE_RETRY_TOTAL_BUDGET > 0


def test_official_budget_helper_keeps_both_rollout_paths_equal() -> None:
    from jitpi05.jitrl.rollout import evaluation_budget_max_steps

    task = {"max_steps": 400}
    assert evaluation_budget_max_steps(task) == 800


def test_proxy_gate_triggers_at_three_of_four_chunks() -> None:
    gate = CycleGate()

    assert gate.proxy_progress(2, 4) == pytest.approx(0.5)
    assert gate.triggered(2, 4) is False
    assert gate.triggered(3, 4) is True


def test_free_text_semantic_action_is_classified_from_text() -> None:
    assert semantic_action_type({"type": "free", "text": "grasp the book"}) == "grasp"
    assert semantic_action_type("place the book at the left compartment") == "place"
    assert semantic_action_type("release the book at the caddy") == "release"
    assert semantic_action_type("move toward the book") == "approach"


def test_explicit_physical_failure_backtracks_even_without_vlm() -> None:
    decision = resolve_cycle_decision(
        vlm=None,
        physical=PhysicalEvidence(
            action_type="grasp",
            gripper_closed=True,
            object_following_gripper=False,
        ),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_count=0,
    )

    assert decision.decision == "backtrack"
    assert decision.next_anchor_id == "a1"
    assert decision.physical_failure is True
    assert decision.accepted is True


def test_physical_success_vetoes_vlm_backtrack() -> None:
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "a1",
            "assessment": {"success_likelihood": "low", "view_agreement": "agree"},
            "front_view_evidence": ["gripper is misaligned"],
            "wrist_view_evidence": ["contact appears unstable"],
        },
        physical=PhysicalEvidence(action_type="grasp", postcondition_satisfied=True),
        anchors=anchors(),
        current_anchor=anchors()[1],
        retry_count=0,
    )

    assert decision.decision == "transit"
    assert decision.vetoed is True
    assert "contradicts" in decision.reason


def test_vlm_needs_strong_likelihood_and_exact_anchor() -> None:
    weak = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "grasp the mug",
            "success_likelihood": "medium",
        },
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[1],
        retry_count=0,
    )
    strong = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "grasp the mug",
            "success_likelihood": "low",
        },
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_count=0,
    )

    assert weak.decision == "transit"
    assert weak.vetoed is True
    assert strong.decision == "backtrack"
    assert strong.next_anchor_id == "a1"
    assert strong.retry_count == 1


def test_vlm_medium_backtracks_when_likelihood_set_is_widened() -> None:
    medium_vlm = {
        "type": "backtrack",
        "next_subtask": "grasp the mug",
        "success_likelihood": "medium",
    }
    widened = resolve_cycle_decision(
        vlm=medium_vlm,
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_count=0,
        backtrack_likelihoods=("low", "medium"),
    )
    strict = resolve_cycle_decision(
        vlm=medium_vlm,
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_count=0,
    )

    assert widened.decision == "backtrack"
    assert widened.next_anchor_id == "a1"
    assert widened.retry_count == 1
    assert strict.decision == "transit"


def test_retry_limit_forces_transit() -> None:
    decision = resolve_cycle_decision(
        vlm={"type": "backtrack", "next_subtask": "a1"},
        physical=PhysicalEvidence(action_type="grasp", failure_detected=True),
        anchors=anchors(),
        current_anchor=anchors()[1],
        retry_count=3,
        max_retries=3,
    )

    assert decision.decision == "transit"
    assert decision.vetoed is True
    assert "retry limit" in decision.reason
    assert decision.retry_count == 3


def test_retry_limit_is_scoped_to_selected_target() -> None:
    retry_counts = {"a1": 3}
    decision = resolve_cycle_decision(
        vlm={"type": "backtrack", "next_subtask": "a0", "success_likelihood": "low"},
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_counts=retry_counts,
        max_retries=3,
    )

    assert decision.decision == "backtrack"
    assert decision.next_anchor_id == "a0"
    assert decision.retry_count == 1


def test_recovery_target_binds_to_current_object_identity() -> None:
    folded_anchors = [
        SemanticAnchor(
            "a0", "approach the mug", "condition-a0", 0, "approach",
            program_id="subtask-00", target_object="mug",
        ),
        SemanticAnchor(
            "a1", "grasp the bowl", "condition-a1", 1, "grasp",
            program_id="subtask-00", target_object="bowl",
        ),
        SemanticAnchor(
            "a2", "place the mug at the basket", "condition-a2", 2, "place",
            program_id="subtask-00", target_object="mug",
        ),
    ]
    # 粗粒度 program 把三个动作都折叠到 subtask-00。当前失败的是 place mug，
    # 因此回溯必须落在同样针对 mug 的 a0，而不是针对 bowl 的 a1。
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "subtask-00",
            "success_likelihood": "low",
        },
        physical=PhysicalEvidence(action_type="place"),
        anchors=folded_anchors,
        current_anchor=folded_anchors[2],
        retry_count=0,
    )

    assert decision.decision == "backtrack"
    assert decision.next_anchor_id == "a0"


def test_self_target_backtrack_is_rejected() -> None:
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "grasp the mug",
            "success_likelihood": "low",
        },
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[1],
        retry_count=0,
    )

    # current_anchor (a1, "grasp the mug") requested itself as the rewind target;
    # rewinding onto the current anchor is not a recovery and must be rejected,
    # otherwise a broken subtask loops to its own state forever.
    assert decision.decision == "transit"
    assert decision.next_anchor_id is None
    assert decision.accepted is False
    assert decision.vetoed is True


def test_backtrack_to_strictly_earlier_anchor_is_allowed() -> None:
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "grasp the mug",
            "success_likelihood": "low",
        },
        physical=PhysicalEvidence(action_type="place"),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_count=0,
    )

    # a1 is strictly earlier than the current a2, so it remains a valid target.
    assert decision.decision == "backtrack"
    assert decision.next_anchor_id == "a1"
    assert decision.retry_count == 1


def test_anchor_requires_exact_recorded_target() -> None:
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "invented action",
            "assessment": {"success_likelihood": "low", "view_agreement": "agree"},
            "front_view_evidence": ["wrong target"],
            "wrist_view_evidence": ["wrong contact"],
        },
        physical=PhysicalEvidence(action_type="grasp"),
        anchors=anchors(),
        current_anchor=anchors()[1],
        retry_count=0,
    )

    assert decision.decision == "transit"
    assert decision.next_anchor_id is None


def test_invalid_gate_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        CycleGate(threshold=0.0)


def test_rotate_action_type_is_classified_from_text() -> None:
    assert semantic_action_type("Rotate the gripper to turn on the stove.") == "rotate"
    assert semantic_action_type("turn the knob clockwise") == "rotate"


def test_retry_limit_forces_transit_without_vetoed_backtrack() -> None:
    decision = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "a0",
            "success_likelihood": "low",
        },
        physical=PhysicalEvidence(action_type="grasp", gripper_closed=True),
        anchors=anchors(),
        current_anchor=anchors()[2],
        retry_counts={"a0": 3},
        max_retries=3,
    )

    assert decision.decision == "transit"
    assert decision.next_anchor_id is None
    assert decision.vetoed is True
    assert "retry limit" in decision.reason


def test_proxy_subtask_stop_is_scoped_to_current_subtask_physics() -> None:
    # 全局 success 未知（中途）时，grasp 抓稳仍应触发子任务级完成。
    assert proxy_subtask_stop(
        PhysicalEvidence(
            action_type="grasp",
            object_following_gripper=True,
            postcondition_satisfied=None,
        )
    ) is True
    # grasp 但物体没跟随（抓空）不应判定完成。
    assert proxy_subtask_stop(
        PhysicalEvidence(
            action_type="grasp",
            object_following_gripper=False,
            gripper_closed=False,
            postcondition_satisfied=None,
        )
    ) is False
    # transport 到位即完成。
    assert proxy_subtask_stop(
        PhysicalEvidence(
            action_type="transport",
            destination_reached=True,
            postcondition_satisfied=None,
        )
    ) is True
    # release 释放即完成。
    assert proxy_subtask_stop(
        PhysicalEvidence(
            action_type="release",
            released=True,
            destination_reached=True,
            postcondition_satisfied=None,
        )
    ) is True
    # 事实未知时保持 None（不臆断）。
    assert (
        proxy_subtask_stop(
            PhysicalEvidence(action_type="approach", postcondition_satisfied=None)
        )
        is None
    )


def test_proxy_subtask_stop_uses_template_types_not_free_text_place() -> None:
    # v7 根因回归测试：transport 阶段物体仍在跟随夹爪是正常搬运状态，
    # 未到位时既不能确认完成也不能确认失败 → 保持 None（不臆断）；若
    # action_type 被误标成 place，proxy stop 会错误地走「未释放」逻辑。
    assert (
        proxy_subtask_stop(
            PhysicalEvidence(
                action_type="transport",
                object_following_gripper=True,
                destination_reached=False,
                postcondition_satisfied=None,
            )
        )
        is None
    )
    # release 且物体仍跟随（未真正释放）→ 未完成。
    assert (
        proxy_subtask_stop(
            PhysicalEvidence(
                action_type="release",
                object_following_gripper=True,
                released=False,
                destination_reached=False,
                postcondition_satisfied=None,
            )
        )
        is False
    )
    # grasp 抓稳（目标在夹爪下）→ 完成。
    assert proxy_subtask_stop(
        PhysicalEvidence(
            action_type="grasp",
            target_identity_ok=True,
            postcondition_satisfied=None,
        )
    ) is True
