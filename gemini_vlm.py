"""PydanticAI Gemini multimodal services that do not require local logits."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from settings import (
    JITRL_CANDIDATES,
    JITRL_GEMINI_BASE_URL,
    JITRL_GEMINI_CREDENTIALS_PATH,
    JITRL_GEMINI_TIMEOUT_SECONDS,
    JITRL_MAX_EVALUATOR_TOKENS,
    JITRL_EVALUATOR_MODEL,
)


def google_provider_base_url(collection_url: str) -> str:
    """Convert a supplied ``.../v1beta/models`` collection URL into its API root."""

    normalized = collection_url.rstrip("/")
    suffix = "/v1beta/models"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized


class VisualProposalCandidate(BaseModel):
    """One executable high-level action proposed by Gemini."""

    # Gemini-compatible endpoints may use label/subtask_label instead of text.
    # Accept harmless extra explanation fields and normalize the label.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("text", "subtask_label", "label"),
    )
    semantic_key: str = Field(min_length=1)

    def normalized_text(self) -> str:
        value = self.text
        if value is None or not value.strip():
            raise ValueError("Gemini candidate must contain text, subtask_label, or label")
        return value.strip()


class VisualProposal(BaseModel):
    """Gemini visual state summary and candidate proposal."""

    model_config = ConfigDict(extra="forbid")

    state_summary: str = Field(min_length=1)
    candidates: list[VisualProposalCandidate] = Field(
        min_length=JITRL_CANDIDATES,
        max_length=JITRL_CANDIDATES,
    )


class ChunkEvaluation(BaseModel):
    """Structured, schema-validated positive-only reward from Gemini."""

    model_config = ConfigDict(extra="ignore")

    result: str = Field(min_length=1)
    usefulness: Literal["useful", "neutral"]
    certainty: Literal["certain", "somewhat uncertain", "very uncertain"]
    score: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def validate_positive_only_consistency(self) -> "ChunkEvaluation":
        expected = "useful" if self.score > 0 else "neutral"
        if self.usefulness != expected:
            raise ValueError(
                f"usefulness must be {expected!r} when score is {self.score}"
            )
        return self

    def normalized_usefulness(self) -> Literal["useful", "neutral"]:
        return "useful" if self.score > 0 else "neutral"


def load_gemini_api_key(path: Path = JITRL_GEMINI_CREDENTIALS_PATH) -> str:
    """Load the API key from a local ignored JSON credential file."""

    credentials_path = Path(path)
    if not credentials_path.is_file():
        raise RuntimeError(
            f"Gemini credentials file not found: {credentials_path}. "
            'Create JSON with one field: {"api_key":"..."}.'
        )
    try:
        value = json.loads(credentials_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Gemini credentials file is invalid JSON: {credentials_path}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("Gemini credentials root must be a JSON object")
    api_key = value.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError("Gemini credentials must contain a non-empty api_key")
    return api_key.strip()


def _load_gemini_agent(output_type, max_tokens: int) -> Agent:
    provider = GoogleProvider(
        api_key=load_gemini_api_key(),
        base_url=google_provider_base_url(JITRL_GEMINI_BASE_URL),
    )
    model = GoogleModel(JITRL_EVALUATOR_MODEL, provider=provider)
    return Agent(
        model,
        output_type=output_type,
        retries=0,
        model_settings=GoogleModelSettings(
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=JITRL_GEMINI_TIMEOUT_SECONDS,
        ),
    )


def load_gemini_planner() -> Agent[None, VisualProposal]:
    """Create a Gemini visual-proposal agent for output that needs no logits."""

    return _load_gemini_agent(VisualProposal, JITRL_MAX_EVALUATOR_TOKENS)


def load_gemini_evaluator() -> Agent[None, ChunkEvaluation]:
    """Create one reusable Gemini step-reward evaluator."""

    return _load_gemini_agent(ChunkEvaluation, JITRL_MAX_EVALUATOR_TOKENS)


def _image_content(image: Image.Image) -> BinaryContent:
    if not isinstance(image, Image.Image):
        raise TypeError("images must contain PIL.Image.Image instances")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return BinaryContent(data=buffer.getvalue(), media_type="image/jpeg")


def proposal_prompt(
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    attempt: int = 1,
) -> str:
    """Build a Gemini prompt for five broad, VLA-groundable action candidates."""

    history = [item.strip() for item in (recent_subtasks or [])[-2:] if item.strip()]
    history_text = "\n".join(f"- {item}" for item in history) or "None"
    retry = (
        ""
        if attempt == 1
        else (
            f"A previous call failed. On correction attempt {attempt}, verify that "
            f"every candidate has non-empty text and semantic_key fields and return "
            f"exactly {JITRL_CANDIDATES} candidates.\n"
        )
    )
    return (
        "You are the visual high-level planner for a Franka robot controlled by a "
        "low-level VLA. The two attached images are ordered as external camera then "
        "wrist camera.\n"
        f"Overall task: {task}\n"
        f"Recently selected subtasks, oldest to newest:\n{history_text}\n"
        f"{retry}"
        f"Infer the current manipulation state and propose exactly {JITRL_CANDIDATES} "
        "distinct, immediately executable next subtasks. state_summary must be concise, "
        "stable English with structure 'robot=...; gripper=...; object=...; target=...; "
        "relations=...; stage=...'.\n"
        "IMPORTANT LANGUAGE-GROUNDING RULE: the low-level VLA may not reliably ground "
        "fine visual attributes such as color, texture, pattern, brand, or material. "
        "Do not make every candidate depend on a phrase such as 'red mug'. Prefer broad, "
        "embodied and spatially grounded commands using image-relative position, table "
        "region, gripper-relative geometry, direction, stage, and generic object class. "
        "Useful examples include 'move toward the object in the center', 'move slightly "
        "left', 'lower the gripper', 'close the gripper around the central cup', and "
        "'carry the held object toward the plate on the left'. Use fine attributes only "
        "when needed to disambiguate the state summary, not as the sole grounding cue in "
        "all action texts.\n"
        "Diversify the five candidates across plausible motion/interaction alternatives: "
        "at least two should use spatial or directional wording; at least one should use "
        "a generic object phrase such as 'central object', 'nearby cup', or 'held object'; "
        "and no more than two may repeat the task's exact color-qualified object phrase. "
        "Do not output tiny paraphrases of one command. Each candidate text must be a "
        "short imperative English subtask. Each semantic_key must have exactly three "
        "descriptive lowercase fields separated by two | characters: skill|object|target."
    )


def generate_proposal(
    planner: Agent[None, VisualProposal],
    images: Sequence[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    attempt: int = 1,
) -> dict:
    """Use Gemini for visual understanding and structured candidate generation."""

    if len(images) != 2:
        raise ValueError("visual proposal requires external and wrist images")
    prompt = proposal_prompt(task, recent_subtasks, attempt=attempt)
    run_result = planner.run_sync([prompt, *(_image_content(image) for image in images)])
    output = run_result.output
    proposal = output.model_dump(mode="json")
    proposal["candidates"] = [
        {
            "text": candidate.normalized_text(),
            "semantic_key": candidate.semantic_key.strip(),
        }
        for candidate in output.candidates
    ]
    if len(proposal["candidates"]) != JITRL_CANDIDATES:
        raise ValueError(
            f"Gemini proposal must contain exactly {JITRL_CANDIDATES} candidates"
        )
    return {
        "backend": "pydantic_ai_google",
        "model": JITRL_EVALUATOR_MODEL,
        "prompt": prompt,
        "raw_output": output.model_dump_json(),
        **proposal,
    }


def evaluator_prompt(
    task: str,
    state_summary: str,
    action_text: str,
    next_state_summary: str,
    success: bool,
    chunk_index: int,
    chunk_count: int,
    attempt: int = 1,
) -> str:
    """Build the strict positive-only evaluator prompt with physical safeguards."""

    retry = (
        ""
        if attempt == 1
        else (
            "A previous call failed. Re-evaluate conservatively and ensure every "
            "structured field is valid.\n"
        )
    )
    return (
        "You evaluate one completed high-level Franka robot action for an online "
        "experience memory. The four attached images are ordered as: before external "
        "camera, before wrist camera, after external camera, after wrist camera.\n"
        f"Overall task: {task}\n"
        f"Chunk: {chunk_index + 1}/{chunk_count}\n"
        f"State before: {state_summary}\n"
        f"Selected subtask: {action_text}\n"
        f"State after: {next_state_summary}\n"
        f"Environment task success at episode end: {str(bool(success)).lower()}\n"
        f"{retry}"
        "Judge only the action's actual visual effect in the full task context. This is "
        "a STRICT POSITIVE-ONLY evaluator: score must be an integer from 0 to 3 and must "
        "never be negative. Assign 1, 2, or 3 only for visually clear, task-relevant "
        "progress of increasing significance. Assign 0 to every harmful, regressive, "
        "wrong-object, wrong-target, repeated, ineffective, no-effect, ambiguous, or "
        "uncertain action. usefulness must be 'useful' exactly when score is positive "
        "and 'neutral' exactly when score is zero. Do not assign credit merely because "
        "the complete episode eventually succeeded.\n"
        "For placement/release/drop actions, distinguish 2D overlap or proximity from "
        "physical completion. A positive placement score requires clear visual evidence "
        "that the correct target object is no longer held by the gripper, is supported "
        "by the specifically requested receptacle, surface, side, region, or compartment "
        "rather than hovering above it, and is stable after release. If the gripper is "
        "still holding it, it moves with the gripper, support/contact is ambiguous, the "
        "object is in the wrong receptacle/region/compartment, or the wrong object is near "
        "the target, assign score 0. The final environment success flag is context, not "
        "proof that this individual chunk deserves positive credit."
    )


def evaluate_chunk(
    evaluator: Agent[None, ChunkEvaluation],
    images: Sequence[Image.Image],
    task: str,
    state_summary: str,
    action_text: str,
    next_state_summary: str,
    success: bool,
    chunk_index: int,
    chunk_count: int,
    attempt: int = 1,
) -> dict:
    """Send four before/after views to Gemini and return an auditable reward record."""

    if len(images) != 4:
        raise ValueError("chunk evaluator requires four before/after images")
    prompt = evaluator_prompt(
        task,
        state_summary,
        action_text,
        next_state_summary,
        success,
        chunk_index,
        chunk_count,
        attempt=attempt,
    )
    user_content = [prompt, *(_image_content(image) for image in images)]
    run_result = evaluator.run_sync(user_content)
    output = run_result.output
    record = output.model_dump(mode="json")
    record["result"] = record["result"].strip()
    record["usefulness"] = output.normalized_usefulness()
    return {
        "backend": "pydantic_ai_google",
        "model": JITRL_EVALUATOR_MODEL,
        "prompt": prompt,
        "raw_output": output.model_dump_json(),
        **record,
    }
