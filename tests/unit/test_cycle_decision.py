import pytest

from jitpi05.config import (
    CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS,
    CYCLE_JITRL_METHODS,
    CYCLE_LIBERO90_CANDIDATES,
    CYCLE_MBR_HYPOTHESES,
    CYCLE_PROGRESS_THRESHOLD,
)
from jitpi05.jitrl.cycle import (
    CycleGate,
    PhysicalEvidence,
    SemanticAnchor,
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
    assert CYCLE_JITRL_METHODS == (
        "static-free",
        "jitrl-free",
        "static-free-cycle",
        "jitrl-free-cycle",
    )
    assert CYCLE_PROGRESS_THRESHOLD == pytest.approx(0.75)
    assert CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS == 3
    assert CYCLE_MBR_HYPOTHESES == 8
    assert 59 in CYCLE_LIBERO90_CANDIDATES


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


def test_vlm_needs_strong_two_view_evidence_and_exact_anchor() -> None:
    weak = resolve_cycle_decision(
        vlm={
            "type": "backtrack",
            "next_subtask": "grasp the mug",
            "assessment": {"success_likelihood": "medium", "view_agreement": "agree"},
            "front_view_evidence": ["unclear pose"],
            "wrist_view_evidence": [],
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
            "assessment": {"success_likelihood": "low", "view_agreement": "agree"},
            "front_view_evidence": ["gripper is beside mug"],
            "wrist_view_evidence": ["no stable contact"],
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
