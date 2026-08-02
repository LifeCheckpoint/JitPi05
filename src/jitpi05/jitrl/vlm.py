"""Gemini visual step evaluator used after each completed JitRL episode."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from jitpi05.config import (
    JITRL_EVALUATOR_MODEL,
    JITRL_GEMINI_BASE_URL,
    JITRL_GEMINI_CREDENTIALS_PATH,
    JITRL_GEMINI_TIMEOUT_SECONDS,
    JITRL_MAX_EVALUATOR_TOKENS,
)


def google_provider_base_url(collection_url: str) -> str:
    """Convert ``.../v1beta/models`` into the Google-compatible API root."""

    normalized = collection_url.rstrip("/")
    suffix = "/v1beta/models"
    return normalized[: -len(suffix)] if normalized.endswith(suffix) else normalized


class ChunkEvaluation(BaseModel):
    """Schema-validated positive-only visual step reward."""

    model_config = ConfigDict(extra="ignore")

    result: str = Field(min_length=1)
    usefulness: Literal["useful", "neutral"]
    certainty: Literal["certain", "somewhat uncertain", "very uncertain"]
    score: int = Field(ge=0, le=3)

    @model_validator(mode="after")
    def validate_positive_only_consistency(self) -> ChunkEvaluation:
        expected = "useful" if self.score > 0 else "neutral"
        if self.usefulness != expected:
            raise ValueError(
                f"usefulness must be {expected!r} when score is {self.score}"
            )
        return self

    def normalized_usefulness(self) -> Literal["useful", "neutral"]:
        return "useful" if self.score > 0 else "neutral"


def load_gemini_api_key(path: Path = JITRL_GEMINI_CREDENTIALS_PATH) -> str:
    """Load the evaluator API key from the ignored credential file."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(value["api_key"]).strip()


def load_gemini_evaluator() -> Agent[None, ChunkEvaluation]:
    """Create the reusable Gemini episode-end step evaluator."""

    provider = GoogleProvider(
        api_key=load_gemini_api_key(),
        base_url=google_provider_base_url(JITRL_GEMINI_BASE_URL),
    )
    model = GoogleModel(JITRL_EVALUATOR_MODEL, provider=provider)
    return Agent(
        model,
        output_type=ChunkEvaluation,
        retries=0,
        model_settings=GoogleModelSettings(
            temperature=0.0,
            max_tokens=JITRL_MAX_EVALUATOR_TOKENS,
            timeout=JITRL_GEMINI_TIMEOUT_SECONDS,
        ),
    )


def _image_content(image: Image.Image) -> BinaryContent:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return BinaryContent(data=buffer.getvalue(), media_type="image/jpeg")


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
    """Build the conservative positive-only visual reward prompt."""

    retry = (
        ""
        if attempt == 1
        else "The previous output was invalid. Re-evaluate conservatively.\n"
    )
    return (
        "You evaluate one completed high-level Franka robot action for an online "
        "experience memory. The four images are before external, before wrist, after "
        "external, and after wrist.\n"
        f"Overall task: {task}\n"
        f"Chunk: {chunk_index + 1}/{chunk_count}\n"
        f"State before: {state_summary}\n"
        f"Selected semantic action: {action_text}\n"
        f"State after: {next_state_summary}\n"
        f"Environment task success at episode end: {str(bool(success)).lower()}\n"
        f"{retry}"
        "This is a STRICT POSITIVE-ONLY evaluator. score is an integer from 0 to 3. "
        "Assign 1, 2, or 3 only for visually clear task-relevant progress. Assign 0 "
        "to harmful, regressive, wrong-object, wrong-target, repeated, ineffective, "
        "ambiguous, or uncertain actions. usefulness is useful exactly when score is "
        "positive and neutral exactly when score is zero. Do not give credit merely "
        "because the episode eventually succeeded.\n"
        "For place or release, positive credit requires the correct object to be no "
        "longer held, supported by the requested receptacle, surface, side, region, or "
        "compartment, and stable after release. Two-dimensional overlap, hovering, an "
        "incorrect region, or an object still moving with the gripper receives score 0."
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
    """Evaluate one semantic-action transition after the episode finishes."""

    prompt = evaluator_prompt(
        task,
        state_summary,
        action_text,
        next_state_summary,
        success,
        chunk_index,
        chunk_count,
        attempt,
    )
    result = evaluator.run_sync(
        [prompt, *(_image_content(image) for image in images)]
    ).output
    record = result.model_dump(mode="json")
    record["result"] = record["result"].strip()
    record["usefulness"] = result.normalized_usefulness()
    return {
        "backend": "pydantic_ai_google",
        "model": JITRL_EVALUATOR_MODEL,
        "prompt": prompt,
        "raw_output": result.model_dump_json(),
        **record,
    }
