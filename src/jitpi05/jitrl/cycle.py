"""Pure inference-time utilities for the CycleVLA-lite wrapper.

This module intentionally has no MuJoCo, model, or network dependencies.  The
MBR implementation follows CycleVLA Appendix C: candidate action chunks are
converted to cumulative six-dimensional end-effector trajectories, the full
pairwise L2 risk matrix is computed, and the medoid candidate is selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class MBRSelection:
    """Auditable result of selecting one candidate by sampling-based MBR."""

    selected_index: int
    risks: tuple[float, ...]
    features: torch.Tensor
    pairwise_distances: torch.Tensor

    @property
    def selected_risk(self) -> float:
        return self.risks[self.selected_index]

    @property
    def mean_pairwise_distance(self) -> float:
        return float(self.pairwise_distances.mean().item())


def _as_action_tensor(action_chunks: torch.Tensor | Sequence[Any]) -> torch.Tensor:
    """Normalize candidate chunks to a finite ``[N, H, D]`` float tensor."""

    chunks = torch.as_tensor(action_chunks, dtype=torch.float32)
    if chunks.ndim != 3:
        raise ValueError(
            "action_chunks must have shape [num_candidates, horizon, action_dim]; "
            f"got {tuple(chunks.shape)}"
        )
    if chunks.shape[0] == 0 or chunks.shape[1] == 0:
        raise ValueError("action_chunks must contain at least one non-empty chunk")
    if chunks.shape[2] < 6:
        raise ValueError(
            "action_chunks must expose at least six translational/rotational deltas"
        )
    if not bool(torch.isfinite(chunks).all()):
        raise ValueError("action_chunks must contain only finite values")
    return chunks


def cumulative_trajectory_features(
    action_chunks: torch.Tensor | Sequence[Any],
    *,
    action_steps: int | None = None,
    delta_dims: int = 6,
) -> torch.Tensor:
    """Build CycleVLA's cumulative 6D trajectory feature for each hypothesis.

    The first ``action_steps`` actions are used because JitPi05 executes only a
    prefix of each predicted chunk.  Each translational/rotational delta is
    cumulatively summed before flattening, so early deviations compound over
    the horizon as in CycleVLA Appendix C.
    """

    chunks = _as_action_tensor(action_chunks)
    if isinstance(action_steps, bool) or (action_steps is not None and action_steps <= 0):
        raise ValueError("action_steps must be a positive integer or None")
    if isinstance(delta_dims, bool) or delta_dims <= 0 or delta_dims > chunks.shape[2]:
        raise ValueError("delta_dims must be in [1, action_dim]")
    horizon = chunks.shape[1] if action_steps is None else min(action_steps, chunks.shape[1])
    deltas = chunks[:, :horizon, :delta_dims]
    return deltas.cumsum(dim=1).reshape(chunks.shape[0], horizon * delta_dims)


def pairwise_l2_distances(features: torch.Tensor) -> torch.Tensor:
    """Return the symmetric candidate-by-candidate Euclidean distance matrix."""

    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must have shape [num_candidates, feature_dim]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must contain only finite values")
    return torch.cdist(values, values, p=2)


def select_mbr_medoid(
    action_chunks: torch.Tensor | Sequence[Any],
    *,
    action_steps: int | None = None,
    delta_dims: int = 6,
) -> MBRSelection:
    """Select the minimum-risk action hypothesis and expose its audit matrix.

    Risk is the mean distance to all sampled hypotheses, including the
    candidate's zero self-distance.  ``torch.argmin`` makes ties deterministic
    by selecting the first candidate, which is important for paired runs.
    """

    features = cumulative_trajectory_features(
        action_chunks,
        action_steps=action_steps,
        delta_dims=delta_dims,
    )
    distances = pairwise_l2_distances(features)
    risks_tensor = distances.mean(dim=1)
    selected_index = int(torch.argmin(risks_tensor).item())
    return MBRSelection(
        selected_index=selected_index,
        risks=tuple(float(value) for value in risks_tensor.tolist()),
        features=features,
        pairwise_distances=distances,
    )


def select_mbr_chunk(
    action_chunks: torch.Tensor | Sequence[Any],
    *,
    action_steps: int | None = None,
    delta_dims: int = 6,
) -> tuple[torch.Tensor, MBRSelection]:
    """Return the selected original chunk without averaging or modifying it."""

    chunks = _as_action_tensor(action_chunks)
    selection = select_mbr_medoid(
        chunks,
        action_steps=action_steps,
        delta_dims=delta_dims,
    )
    return chunks[selection.selected_index], selection


class CyclePhase(StrEnum):
    """Inference phases used by the CycleVLA-lite rollout state machine."""

    IN_PROGRESS = "in_progress"
    CHECK = "check"
    BACKTRACK = "backtrack"
    MBR_RETRY = "mbr_retry"


@dataclass(frozen=True)
class CycleGate:
    """A zero-training proxy for CycleVLA's learned progress trigger."""

    threshold: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("CycleGate.threshold must be in (0, 1]")

    def proxy_progress(self, completed_chunks: int, total_chunks: int) -> float:
        if completed_chunks < 0 or total_chunks <= 0:
            raise ValueError("completed_chunks must be non-negative and total_chunks positive")
        return min(1.0, float(completed_chunks) / float(total_chunks))

    def triggered(self, completed_chunks: int, total_chunks: int) -> bool:
        return self.proxy_progress(completed_chunks, total_chunks) >= self.threshold


@dataclass(frozen=True)
class SemanticAnchor:
    """The language and history identity of one high-level Qwen decision."""

    anchor_id: str
    action_text: str
    condition: str
    high_level_step_index: int
    action_type: str

    def __post_init__(self) -> None:
        if not self.anchor_id.strip() or not self.action_text.strip():
            raise ValueError("semantic anchors require non-empty id and action_text")
        if self.high_level_step_index < 0:
            raise ValueError("high_level_step_index must be non-negative")


@dataclass(frozen=True)
class PhysicalEvidence:
    """Explicit simulator/proprioceptive facts used to gate VLM decisions."""

    action_type: str
    gripper_closed: bool | None = None
    object_following_gripper: bool | None = None
    target_identity_ok: bool | None = None
    destination_reached: bool | None = None
    released: bool | None = None
    postcondition_satisfied: bool | None = None
    failure_detected: bool | None = None
    failure_reasons: tuple[str, ...] = ()

    @property
    def explicit_failure(self) -> bool:
        return self.failure_detected is True or self.postcondition_satisfied is False

    @property
    def explicit_success(self) -> bool:
        return self.postcondition_satisfied is True


@dataclass(frozen=True)
class CycleDecision:
    """Conservative, auditable result of fusing physical and VLM evidence."""

    decision: Literal["transit", "backtrack"]
    next_anchor_id: str | None
    reason: str
    physical_failure: bool
    vlm_requested_backtrack: bool
    accepted: bool
    vetoed: bool
    retry_count: int
    max_retries: int


def semantic_action_type(action: str | Mapping[str, Any]) -> str:
    """Map a Qwen action instance to a small recovery-relevant stage vocabulary."""

    if isinstance(action, Mapping):
        raw_type = str(action.get("type", "")).strip().lower()
        if raw_type and raw_type not in {"free", "unknown"}:
            return raw_type
        text = str(action.get("text", ""))
    else:
        text = action
    normalized = " ".join(str(text).lower().split())
    if normalized == "stop" or normalized.startswith("stop "):
        return "stop"
    if any(token in normalized for token in ("release", "open the gripper", "let go")):
        return "release"
    if any(token in normalized for token in ("place ", "put ", "set ")):
        return "place"
    if any(token in normalized for token in ("carry ", "transport", "holding ", "with the ")):
        return "transport"
    if any(token in normalized for token in ("grasp", "pick up", "close the gripper")):
        return "grasp"
    if any(token in normalized for token in ("lift", "raise")):
        return "lift"
    if any(token in normalized for token in ("align", "center", "position over")):
        return "align"
    if any(token in normalized for token in ("retract", "move away", "back away")):
        return "retract"
    if any(token in normalized for token in ("move", "approach", "reach")):
        return "approach"
    return "unknown"


def physical_failure_signal(evidence: PhysicalEvidence) -> bool | None:
    """Return a three-valued postcondition result without guessing unknown facts."""

    if evidence.explicit_failure:
        return True
    if evidence.explicit_success:
        return False
    if evidence.target_identity_ok is False:
        return True
    action_type = semantic_action_type(evidence.action_type)
    if action_type in {"grasp", "lift", "transport"}:
        if evidence.object_following_gripper is False:
            return True
        if action_type == "grasp" and evidence.gripper_closed is False:
            return True
    if action_type in {"place", "release"}:
        if evidence.destination_reached is False or evidence.released is False:
            return True
    return None


def select_recovery_anchor(
    target: str | None,
    anchors: Sequence[SemanticAnchor],
) -> SemanticAnchor | None:
    """Resolve an exact VLM target to the earliest matching recorded anchor."""

    requested = "" if target is None else str(target).strip()
    if not requested:
        return None
    matches = [
        anchor
        for anchor in anchors
        if requested in {anchor.anchor_id, anchor.action_text}
    ]
    return min(matches, key=lambda anchor: anchor.high_level_step_index) if matches else None


def _vlm_evidence_is_strong(vlm: Mapping[str, Any]) -> bool:
    assessment = vlm.get("assessment")
    assessment_map = assessment if isinstance(assessment, Mapping) else {}
    likelihood = str(
        vlm.get("success_likelihood", assessment_map.get("success_likelihood", ""))
    ).lower()
    agreement = str(vlm.get("view_agreement", assessment_map.get("view_agreement", ""))).lower()
    front = vlm.get("front_view_evidence", ())
    wrist = vlm.get("wrist_view_evidence", ())
    front_count = len(front) if isinstance(front, Sequence) and not isinstance(front, str) else 0
    wrist_count = len(wrist) if isinstance(wrist, Sequence) and not isinstance(wrist, str) else 0
    return likelihood == "low" and agreement != "disagree" and front_count >= 1 and wrist_count >= 1


def resolve_cycle_decision(
    *,
    vlm: Mapping[str, Any] | None,
    physical: PhysicalEvidence,
    anchors: Sequence[SemanticAnchor],
    current_anchor: SemanticAnchor,
    retry_count: int,
    max_retries: int = 3,
) -> CycleDecision:
    """Fuse evidence with transit-default and bounded, reversible recovery.

    A VLM cannot terminate an episode.  Its backtrack request is accepted only
    with strong two-view evidence and an exact recorded target, unless an
    explicit physical postcondition failure already proves the need to retry.
    """

    if retry_count < 0 or max_retries <= 0:
        raise ValueError("retry_count must be non-negative and max_retries positive")
    if retry_count >= max_retries:
        return CycleDecision(
            decision="transit",
            next_anchor_id=None,
            reason="retry limit reached; force transit without another rewind",
            physical_failure=physical_failure_signal(physical) is True,
            vlm_requested_backtrack=False,
            accepted=False,
            vetoed=True,
            retry_count=retry_count,
            max_retries=max_retries,
        )

    physical_signal = physical_failure_signal(physical)
    raw_type = str((vlm or {}).get("type", (vlm or {}).get("decision", "transit"))).lower()
    vlm_requested = raw_type == "backtrack"
    target = (vlm or {}).get("next_subtask", (vlm or {}).get("next_anchor_id"))
    target_anchor = select_recovery_anchor(target, anchors)
    physical_failure = physical_signal is True
    physical_contradiction = physical_signal is False

    # A deterministic physical failure must still be recoverable when the VLM
    # is unavailable. Prefer the earliest anchor of the failed semantic stage;
    # the current anchor is the last safe fallback.
    if physical_failure and target_anchor is None:
        failed_type = semantic_action_type(physical.action_type)
        same_stage = [
            anchor for anchor in anchors if semantic_action_type(anchor.action_type) == failed_type
        ]
        target_anchor = (
            min(same_stage, key=lambda anchor: anchor.high_level_step_index)
            if same_stage
            else current_anchor
        )

    if physical_failure and target_anchor is not None:
        return CycleDecision(
            decision="backtrack",
            next_anchor_id=target_anchor.anchor_id,
            reason="explicit physical postcondition failure; rewind to earliest valid anchor",
            physical_failure=True,
            vlm_requested_backtrack=vlm_requested,
            accepted=True,
            vetoed=False,
            retry_count=retry_count + 1,
            max_retries=max_retries,
        )
    if physical_contradiction and vlm_requested:
        return CycleDecision(
            decision="transit",
            next_anchor_id=None,
            reason="physical postcondition contradicts VLM backtrack request",
            physical_failure=False,
            vlm_requested_backtrack=True,
            accepted=False,
            vetoed=True,
            retry_count=retry_count,
            max_retries=max_retries,
        )
    if vlm_requested and target_anchor is not None and _vlm_evidence_is_strong(vlm or {}):
        return CycleDecision(
            decision="backtrack",
            next_anchor_id=target_anchor.anchor_id,
            reason="strong two-view low-success evidence with exact recovery anchor",
            physical_failure=False,
            vlm_requested_backtrack=True,
            accepted=True,
            vetoed=False,
            retry_count=retry_count + 1,
            max_retries=max_retries,
        )
    return CycleDecision(
        decision="transit",
        next_anchor_id=None,
        reason="transit default: insufficient unambiguous failure evidence",
        physical_failure=physical_failure,
        vlm_requested_backtrack=vlm_requested,
        accepted=False,
        vetoed=vlm_requested,
        retry_count=retry_count,
        max_retries=max_retries,
    )
