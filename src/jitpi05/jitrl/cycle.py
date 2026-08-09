"""Pure inference-time utilities for the CycleVLA-lite wrapper.

This module intentionally has no MuJoCo, model, or network dependencies. The
MBR implementation follows CycleVLA Appendix C: candidate action chunks are
converted to cumulative six-dimensional end-effector trajectories, the full
pairwise L2 risk matrix is computed, and the medoid candidate is selected.
"""

from __future__ import annotations

import re
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
    """Build CycleVLA's cumulative 6D trajectory feature for each hypothesis."""

    chunks = _as_action_tensor(action_chunks)
    if isinstance(action_steps, bool) or (
        action_steps is not None and action_steps <= 0
    ):
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
    """Select the minimum-risk action hypothesis and expose its audit matrix."""

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
    """Inference phases used by the CycleVLA state machine."""

    IN_PROGRESS = "in_progress"
    CHECK = "check"
    COMPLETE = "complete"
    BACKTRACK = "backtrack"
    MBR_RETRY = "mbr_retry"


@dataclass(frozen=True)
class CyclePolicyOutput:
    """One policy chunk plus optional learned CycleVLA signals.

    The stock LeRobot policy emits seven robot-action dimensions, so its stop
    and progress signals are represented as ``None`` rather than fabricated
    values. A CycleVLA/OpenPI policy emits nine dimensions and is adapted here.
    """

    robot_action_chunk: torch.Tensor
    stop_signals: tuple[float | None, ...]
    progress_signals: tuple[float | None, ...]
    signal_source: Literal["policy_9d", "missing_7d"]

    @property
    def has_policy_signals(self) -> bool:
        return self.signal_source == "policy_9d"

    @property
    def horizon(self) -> int:
        return int(self.robot_action_chunk.shape[0])

    def stop_signal(self, index: int) -> float | None:
        return self.stop_signals[index] if 0 <= index < len(self.stop_signals) else None

    def progress_signal(self, index: int) -> float | None:
        return (
            self.progress_signals[index]
            if 0 <= index < len(self.progress_signals)
            else None
        )


def adapt_cycle_policy_output(action_chunk: torch.Tensor) -> CyclePolicyOutput:
    """Adapt a 7-D LeRobot or 9-D CycleVLA action chunk.

    Any other dimensionality is rejected instead of silently truncating model
    output, because truncation can turn a future policy mismatch into a false
    Cycle transition.
    """

    values = torch.as_tensor(action_chunk, dtype=torch.float32).detach().cpu()
    if values.ndim == 3 and values.shape[0] == 1:
        values = values.squeeze(0)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "policy action chunk must have shape [horizon, action_dim] or "
            "[1, horizon, action_dim]"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("policy action chunk must contain only finite values")
    action_dim = int(values.shape[1])
    horizon = int(values.shape[0])
    if action_dim == 7:
        return CyclePolicyOutput(
            robot_action_chunk=values,
            stop_signals=(None,) * horizon,
            progress_signals=(None,) * horizon,
            signal_source="missing_7d",
        )
    if action_dim == 9:
        return CyclePolicyOutput(
            robot_action_chunk=values[:, :7].contiguous(),
            stop_signals=tuple(float(value) for value in values[:, 7].tolist()),
            progress_signals=tuple(float(value) for value in values[:, 8].tolist()),
            signal_source="policy_9d",
        )
    raise ValueError(
        "Cycle policy output must expose exactly 7 robot dimensions or 9 "
        f"CycleVLA dimensions; got action_dim={action_dim}"
    )


@dataclass
class CycleSignalConfirmation:
    """Appendix-D confirmation rule for noisy stop/progress signals."""

    threshold: float = 0.9
    required_consecutive: int = 2
    required_gap: int = 2
    first_seen: bool = False
    consecutive_high: int = 0
    low_gap: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("CycleSignalConfirmation.threshold must be in (0, 1]")
        if self.required_consecutive <= 0 or self.required_gap <= 0:
            raise ValueError("confirmation counts must be positive")

    def reset(self) -> None:
        self.first_seen = False
        self.consecutive_high = 0
        self.low_gap = 0

    def update(self, signal: float | bool | None) -> bool:
        """Consume one signal and return whether the condition is confirmed."""

        if signal is None:
            self.consecutive_high = 0
            return False
        high = bool(signal) if isinstance(signal, bool) else float(signal) >= self.threshold
        if high:
            confirmed = (
                self.consecutive_high + 1 >= self.required_consecutive
                or (self.first_seen and self.low_gap >= self.required_gap)
            )
            self.first_seen = True
            self.consecutive_high += 1
            self.low_gap = 0
            return confirmed
        if self.first_seen:
            self.low_gap += 1
        self.consecutive_high = 0
        return False


@dataclass(frozen=True)
class CycleSubtask:
    """Stable, immutable subtask node used by the VLM recovery planner."""

    subtask_id: str
    text: str
    action_type: str
    position: int

    def __post_init__(self) -> None:
        if not self.subtask_id.strip() or not self.text.strip():
            raise ValueError("Cycle subtasks require non-empty id and text")
        if self.position < 0:
            raise ValueError("Cycle subtask position must be non-negative")


def format_cycle_subtask_program(program: Sequence[CycleSubtask]) -> tuple[str, ...]:
    """Serialize a program as exact ``id: text`` targets for the VLM."""

    return tuple(f"{item.subtask_id}: {item.text}" for item in program)


def format_cycle_condition(task: str, subtask: str) -> str:
    """Build the official paper prompt for the low-level policy.

    The released ``pi05_libero_cyclevla`` model and the OpenVLA-OFT
    ``libero_sub_decomposed_progress`` checkpoint were both trained with
    ``prompt_from_task=True`` where the prompt is exactly
    ``Task: <task>. The current subtask: <subtask>``.  Reusing that template
    keeps the 7-D condition distribution aligned with the training distribution
    instead of injecting free-text Qwen output.
    """

    return f"Task: {str(task).strip()}. The current subtask: {str(subtask).strip()}"


def _extract_pick_place_objects(task: str) -> tuple[str, str] | None:
    """Extract ``(pick_object, place_object)`` following the paper's FSM utils.

    Mirrors ``fsm_utils/utils.py``: for ``pick up the X and place it in/on the
    Y`` and ``put the X in/on the Y`` the two objects are read back verbatim so
    the resulting program texts match the subtask prompt distribution the
    released ``libero_decomposed_progress`` checkpoints were trained on.
    """

    normalized = " ".join(str(task).strip().split())
    lower = normalized.lower()
    match = re.search(
        r"^(?:pick up|grasp|take|get)\s+the\s+(?P<pick>.+?)\s+and\s+"
        r"(?:place|put|set)\s+it\s+(?:in|on|at|into)\s+the\s+(?P<place>.+)$",
        lower,
    )
    if match:
        return match.group("pick").strip(), match.group("place").strip()
    match = re.search(
        r"^put\s+the\s+(?P<pick>.+?)\s+(?:in|on|at|into)\s+the\s+(?P<place>.+)$",
        lower,
    )
    if match:
        return match.group("pick").strip(), match.group("place").strip()
    return None


_COMPLEX_TASK_STATES: dict[str, tuple[str, ...]] = {
    "open the middle drawer of the cabinet": (
        "Move the gripper toward the handle of the middle drawer of the cabinet.",
        "Close the gripper to grasp the drawer handle.",
        "Move the gripper to pull the drawer outward.",
    ),
    "open the top drawer and put the bowl inside": (
        "Move the gripper above the handle of the top drawer and insert it into the gap.",
        "Move the gripper to pull the drawer outward.",
        "Move the gripper above the bowl.",
        "Close the gripper to grasp the bowl.",
        "Move the gripper above the open top drawer while holding the bowl.",
        "Open the gripper to release the bowl.",
    ),
    "push the plate to the front of the stove": (
        "Move the gripper toward the plate and make contact.",
        "Close the gripper to secure attachment to the plate.",
        "Move the gripper forward to push the plate toward the front of the stove while maintaining contact.",
    ),
    "turn on the stove": (
        "Move the gripper above the stove knob.",
        "Close the gripper to grasp the stove knob.",
        "Rotate the gripper to turn on the stove.",
    ),
    "turn on the stove and put the moka pot on it": (
        "Move the gripper above the stove knob.",
        "Close the gripper to grasp the stove knob.",
        "Rotate the gripper to turn on the stove.",
        "Open the gripper to release the stove knob.",
        "Move the gripper above the moka pot.",
        "Close the gripper to grasp the moka pot.",
        "Move the gripper above the stove while holding the moka pot.",
        "Open the gripper to release the moka pot.",
    ),
    "put the black bowl in the bottom drawer of the cabinet and close it": (
        "Move the gripper above the black bowl.",
        "Close the gripper to grasp the black bowl.",
        "Move the gripper above the bottom drawer of the cabinet while holding the black bowl.",
        "Open the gripper to release the black bowl.",
        "Move the gripper to close the bottom drawer of the cabinet.",
    ),
    "put the yellow and white mug in the microwave and close it": (
        "Move the gripper above the yellow and white mug.",
        "Close the gripper to grasp the yellow and white mug.",
        "Move the gripper inside the microwave while holding the yellow and white mug.",
        "Open the gripper to release the yellow and white mug.",
        "Move the gripper to close the microwave.",
    ),
}


def build_cycle_subtask_program(task: str) -> tuple[CycleSubtask, ...]:
    """Build an official-style deterministic subtask program.

    Precedence mirrors the paper's eval entry point
    (``pick_place_states`` -> ``complex_states``):
      1. a pick-place task expands to the official four-state template
         (Move above -> Close to grasp -> Move above the place while holding ->
         Open to release);
      2. otherwise a known complex task uses its hard-coded state sequence;
      3. as a last resort a single semantic node keeps the program non-empty.
    The texts are verbatim from ``fsm_utils/build.py`` so the condition
    distribution stays aligned with the released decomposed checkpoints.
    """

    normalized = " ".join(str(task).strip().split())
    lower = normalized.lower()
    pick_place = _extract_pick_place_objects(normalized)
    if pick_place is not None:
        pick_object, place_object = pick_place
        texts = (
            ("approach", f"Move the gripper above the {pick_object}."),
            ("grasp", f"Close the gripper to grasp the {pick_object}."),
            ("transport", f"Move the gripper above the {place_object} while holding the {pick_object}."),
            ("release", f"Open the gripper to release the {pick_object}."),
        )
    elif lower in _COMPLEX_TASK_STATES:
        texts = tuple(
            (semantic_action_type(state), state) for state in _COMPLEX_TASK_STATES[lower]
        )
    else:
        texts = ((semantic_action_type(normalized), normalized or "complete the task"),)
    return tuple(
        CycleSubtask(
            subtask_id=f"subtask-{index:02d}",
            text=text,
            action_type=action_type,
            position=index,
        )
        for index, (action_type, text) in enumerate(texts)
    )


def match_cycle_subtask(
    action: str | Mapping[str, Any],
    program: Sequence[CycleSubtask],
    *,
    preferred_position: int = 0,
    used_ids: Sequence[str] = (),
) -> CycleSubtask:
    """Map a live Qwen action to the closest immutable program node."""

    if not program:
        raise ValueError("Cycle subtask program cannot be empty")
    action_text = str(action.get("text", "") if isinstance(action, Mapping) else action)
    action_type = semantic_action_type(action)
    tokens = set(re.findall(r"[a-z0-9]+", action_text.lower()))
    used = set(used_ids)

    def score(item: CycleSubtask) -> tuple[int, int, int, int]:
        overlap = len(tokens & set(re.findall(r"[a-z0-9]+", item.text.lower())))
        unused = int(item.subtask_id not in used)
        type_match = int(action_type == item.action_type)
        distance = -abs(item.position - preferred_position)
        return type_match, unused, overlap, distance

    return max(program, key=score)


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
    """The language and stable program identity of one high-level decision."""

    anchor_id: str
    action_text: str
    condition: str
    high_level_step_index: int
    action_type: str
    program_id: str = ""
    program_position: int = -1
    target_object: str = ""

    def __post_init__(self) -> None:
        if not self.anchor_id.strip() or not self.action_text.strip():
            raise ValueError("semantic anchors require non-empty id and action_text")
        if self.high_level_step_index < 0:
            raise ValueError("high_level_step_index must be non-negative")
        if self.program_position < -1:
            raise ValueError("program_position must be -1 or non-negative")


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


def proxy_subtask_stop(evidence: PhysicalEvidence) -> bool | None:
    """Return whether the *current subtask* is physically complete.

    The 7-D stock policy has no learned stop signal, so the official stop
    semantics (subtask done) must be proxied from simulator object-level
    evidence.  Unlike ``explicit_success`` (which is only True when the whole
    task succeeded and therefore never fires mid-episode), this judgment is
    scoped to the current action type:

    - grasp / lift: complete when the object follows the gripper (or the
      gripper holds the target identity);
    - transport: complete when the destination is reached;
    - place / release: complete when the object is released at the
      destination;
    - approach / align: complete when the target is under the gripper.

    Returns True/False when decidable and None when the facts are unknown.
    """

    action_type = semantic_action_type(evidence.action_type)
    if action_type in {"grasp", "lift"}:
        if evidence.object_following_gripper is True or evidence.target_identity_ok is True:
            return True
        if evidence.gripper_closed is False:
            return False
        return None
    if action_type in {"transport"}:
        if evidence.destination_reached is True:
            return True
        if evidence.object_following_gripper is False:
            return False
        return None
    if action_type in {"place", "release"}:
        if evidence.released is True or evidence.destination_reached is True:
            return True
        if evidence.object_following_gripper is True:
            return False
        return None
    if action_type in {"approach", "align"}:
        if evidence.target_identity_ok is True:
            return True
        return None
    return evidence.explicit_success


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
    if any(token in normalized for token in ("rotate", "turn the knob", "turn the handle")):
        return "rotate"
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
    *,
    current_anchor: SemanticAnchor | None = None,
) -> SemanticAnchor | None:
    """Resolve an exact stable program target to a previously reached anchor.

    Recovery must always move *backwards*: the target is required to be strictly
    earlier than the current anchor.  Rewinding onto the current anchor itself
    (self-target) is never a recovery and is rejected here, so a broken subtask
    cannot repeatedly rewind to its own state and loop forever.  Only explicit
    physical-failure retries may target the current anchor, and those are still
    bounded by the per-target retry limit.

    When several anchors share the same coarse program node (coarse fallback
    programs fold many actions onto one ``subtask-00``), the resolved target is
    additionally bound to the current anchor's ``target_object``: the recovery
    point must concern the same object as the failing step, so a ``grasp the mug``
    failure cannot rewind into an earlier ``approach the bowl`` step.
    """

    requested = "" if target is None else str(target).strip()
    if not requested:
        return None
    matches = [
        anchor
        for anchor in anchors
        if requested in {anchor.anchor_id, anchor.action_text, anchor.program_id}
    ]
    if current_anchor is not None:
        matches = [
            anchor
            for anchor in matches
            if anchor.high_level_step_index < current_anchor.high_level_step_index
        ]
    if current_anchor is not None and current_anchor.target_object and matches:
        same_object = [
            anchor
            for anchor in matches
            if anchor.target_object
            and anchor.target_object == current_anchor.target_object
        ]
        if same_object:
            return min(same_object, key=lambda anchor: anchor.high_level_step_index)
    return min(matches, key=lambda anchor: anchor.high_level_step_index) if matches else None


def _vlm_evidence_is_strong(
    vlm: Mapping[str, Any],
    *,
    backtrack_likelihoods: Sequence[str] = ("low",),
) -> bool:
    """Return whether the VLM likelihood passes the configured strict gate."""

    likelihood = str(vlm.get("success_likelihood", "")).lower()
    return likelihood in {str(value).lower() for value in backtrack_likelihoods}


def resolve_cycle_decision(
    *,
    vlm: Mapping[str, Any] | None,
    physical: PhysicalEvidence,
    anchors: Sequence[SemanticAnchor],
    current_anchor: SemanticAnchor,
    retry_count: int = 0,
    retry_counts: Mapping[str, int] | None = None,
    max_retries: int = 3,
    backtrack_likelihoods: Sequence[str] = ("low",),
) -> CycleDecision:
    """Fuse evidence with transit-default and bounded per-target recovery."""

    if retry_count < 0 or max_retries <= 0:
        raise ValueError("retry_count must be non-negative and max_retries positive")
    if retry_counts is not None and any(value < 0 for value in retry_counts.values()):
        raise ValueError("retry_counts must contain non-negative values")

    physical_signal = physical_failure_signal(physical)
    raw_type = str((vlm or {}).get("type", (vlm or {}).get("decision", "transit"))).lower()
    vlm_requested = raw_type == "backtrack"
    target = (vlm or {}).get("next_subtask", (vlm or {}).get("next_anchor_id"))
    target_anchor = select_recovery_anchor(target, anchors, current_anchor=current_anchor)
    physical_failure = physical_signal is True
    physical_contradiction = physical_signal is False

    if physical_failure and target_anchor is None:
        failed_type = semantic_action_type(physical.action_type)
        same_stage = [
            anchor
            for anchor in anchors
            if semantic_action_type(anchor.action_type) == failed_type
            and anchor.high_level_step_index <= current_anchor.high_level_step_index
        ]
        target_anchor = (
            min(same_stage, key=lambda anchor: anchor.high_level_step_index)
            if same_stage
            else current_anchor
        )

    target_retry_count = retry_count
    if target_anchor is not None and retry_counts is not None:
        target_retry_count = retry_counts.get(target_anchor.anchor_id, 0)

    if target_retry_count >= max_retries:
        return CycleDecision(
            decision="transit",
            next_anchor_id=None,
            reason="target subtask retry limit reached; force transit without another rewind",
            physical_failure=physical_failure,
            vlm_requested_backtrack=False,
            accepted=False,
            vetoed=True,
            retry_count=target_retry_count,
            max_retries=max_retries,
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
            retry_count=target_retry_count + 1,
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
            retry_count=target_retry_count,
            max_retries=max_retries,
        )

    if (
        vlm_requested
        and target_anchor is not None
        and _vlm_evidence_is_strong(vlm or {}, backtrack_likelihoods=backtrack_likelihoods)
    ):
        return CycleDecision(
            decision="backtrack",
            next_anchor_id=target_anchor.anchor_id,
            reason="strong low-success evidence with exact prior subtask target",
            physical_failure=False,
            vlm_requested_backtrack=True,
            accepted=True,
            vetoed=False,
            retry_count=target_retry_count + 1,
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
        retry_count=target_retry_count,
        max_retries=max_retries,
    )
