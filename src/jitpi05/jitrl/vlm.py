"""Gemini visual step evaluator used after each completed JitRL episode."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from jitpi05.config import (
    CYCLE_PREDICTOR_RETRIES,
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


class CycleFailurePrediction(BaseModel):
    """Flat, transit-default failure forecast with auditable view evidence."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["transit", "backtrack"] = "transit"
    success_likelihood: Literal["high", "medium", "low"] = "medium"
    next_subtask: str = Field(
        default="", description="Exact stable subtask ID when backtracking, else empty."
    )
    reason: str = Field(default="", description="One short sentence of observable evidence.")
    front_evidence: str = Field(default="", description="Observable FRONT-view evidence.")
    wrist_evidence: str = Field(default="", description="Observable WRIST-view evidence.")


def validate_cycle_prediction_target(
    prediction: CycleFailurePrediction | dict,
    valid_subtasks: Sequence[str],
) -> CycleFailurePrediction:
    """Reject hallucinated rewind targets before they reach the simulator."""

    validated = (
        prediction
        if isinstance(prediction, CycleFailurePrediction)
        else CycleFailurePrediction.model_validate(prediction)
    )
    allowed: set[str] = set()
    for subtask in valid_subtasks:
        value = str(subtask).strip()
        if not value:
            continue
        allowed.add(value)
        if ":" in value:
            allowed.add(value.split(":", 1)[0].strip())
    if validated.type == "backtrack" and validated.next_subtask not in allowed:
        raise ValueError(
            "Cycle VLM returned a next_subtask that is not an exact stable "
            f"program target: {validated.next_subtask!r}"
        )
    return validated


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


def load_cycle_failure_predictor() -> Agent[None, CycleFailurePrediction]:
    """Create the independent in-loop Cycle failure predictor."""

    provider = GoogleProvider(
        api_key=load_gemini_api_key(),
        base_url=google_provider_base_url(JITRL_GEMINI_BASE_URL),
    )
    model = GoogleModel(JITRL_EVALUATOR_MODEL, provider=provider)
    return Agent(
        model,
        output_type=CycleFailurePrediction,
        retries=CYCLE_PREDICTOR_RETRIES,
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


def cycle_failure_predictor_prompt(
    task: str,
    subtasks: Sequence[str],
    current_subtask: str,
    *,
    attempt: int = 1,
) -> str:
    """Build the paper-style FRONT/WRIST, transit-default failure prompt."""

    retry = (
        ""
        if attempt == 1
        else "The previous output was invalid; return exactly one flat JSON object.\n"
    )
    subtask_lines = "\n".join(
        f"{index + 1}. {subtask}" for index, subtask in enumerate(subtasks)
    )
    return (
        "You are an expert robot behavior annotator doing an in-loop failure "
        "forecast at approximately 90% of the current subtask, not a post-hoc "
        "success report. Image 1 is FRONT; image 2 is WRIST.\n"
        f"Overall task: {task}\n"
        f"Current subtask: {current_subtask}\n"
        f"Complete immutable subtask program (targets are exact IDs):\n{subtask_lines}\n"
        f"{retry}"
        "FRONT provides global object identity, spatial alignment, reachability, "
        "and path context. WRIST provides local gripper alignment, contact, slip, "
        "and affordance evidence. Fuse both views. Default to type=transit unless "
        "strong, unambiguous observable evidence says continuing will fail without "
        "repositioning. Never terminate the episode. If backtracking, choose the "
        "earliest already-reached program ID that restores the missing precondition.\n"
        "Use this complete valid output example only as a formatting guide; do not "
        "copy its evidence, and do not emit Markdown fences or explanatory text:\n"
        '{"type": "transit", "success_likelihood": "medium", '
        '"next_subtask": "", '
        '"reason": "Both views show a reachable and stable continuation.", '
        '"front_evidence": "FRONT shows the intended object and path clearly.", '
        '"wrist_evidence": "WRIST shows stable gripper alignment and contact."}\n'
        "For type=transit, next_subtask MUST be the empty string. For type=backtrack, "
        "next_subtask MUST be copied exactly from one stable program ID above; never "
        "invent, paraphrase, or include the subtask text. Return exactly ONE flat JSON "
        "object with ONLY these six fields and no text outside it:\n"
        '{"type": "transit"|"backtrack", '
        '"success_likelihood": "high"|"medium"|"low", '
        '"next_subtask": "<exact program ID, or empty string>", '
        '"reason": "<one short sentence>", '
        '"front_evidence": "<observable FRONT cue>", '
        '"wrist_evidence": "<observable WRIST cue>"}'
    )


def predict_cycle_failure(
    predictor: Agent[None, CycleFailurePrediction],
    images: Sequence[Image.Image],
    task: str,
    subtasks: Sequence[str],
    current_subtask: str,
    *,
    attempt: int = 1,
) -> dict:
    """Run the in-loop predictor on synchronized front and wrist views."""

    if len(images) != 2:
        raise ValueError("Cycle failure prediction requires exactly front and wrist images")
    prompt = cycle_failure_predictor_prompt(
        task,
        subtasks,
        current_subtask,
        attempt=attempt,
    )
    result = predictor.run_sync([prompt, *(_image_content(image) for image in images)]).output
    if result.type == "backtrack":
        result = validate_cycle_prediction_target(result, subtasks)
    return {
        "backend": "pydantic_ai_google",
        "model": JITRL_EVALUATOR_MODEL,
        "prompt": prompt,
        "raw_output": result.model_dump_json(),
        "schema_valid": True,
        **result.model_dump(mode="json"),
    }


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


def _extract_json_object(text: str) -> dict:
    """从 Qwen 输出中稳健提取 JSON 对象（容忍 markdown fence 与前后缀）。"""

    stripped = text.strip()
    if stripped.startswith("```"):
        blocks = stripped.split("```")
        stripped = blocks[1] if len(blocks) > 1 else blocks[0]
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in Qwen output: {text!r}")
    return json.loads(stripped[start : end + 1])


def qwen_evaluator_prompt(
    task: str,
    state_summary: str,
    action_text: str,
    next_state_summary: str,
    success: bool,
    chunk_index: int,
    chunk_count: int,
    attempt: int = 1,
) -> str:
    """在 Gemini evaluator prompt 基础上追加 Qwen 需要的 JSON 输出格式说明。"""

    base = evaluator_prompt(
        task,
        state_summary,
        action_text,
        next_state_summary,
        success,
        chunk_index,
        chunk_count,
        attempt,
    )
    return (
        base
        + "\nReturn exactly one JSON object and no other text, with these fields: "
        '{"result": "<one short sentence>", "usefulness": "useful" | "neutral", '
        '"certainty": "certain" | "somewhat uncertain" | "very uncertain", '
        '"score": <integer 0-3>}. usefulness must be useful exactly when score is '
        "positive and neutral exactly when score is zero."
    )


class QwenChunkEvaluator:
    """本地 Qwen3.5-2B 视觉 chunk 评估器（复用高层规划器的同一模型实例）。

    与 Gemini 评估器等价地返回 ``result/usefulness/certainty/score``，仅 backend
    与 model 字段不同。注意：评估与规划共享同一模型，存在自评偏差（见 README
    credit assignment 说明）。
    """

    def __init__(self, model, processor, *, model_id: str | None = None) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id or getattr(model, "name_or_path", "Qwen3.5-2B")

    @torch.inference_mode()
    def evaluate(
        self,
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
        """评估一个语义动作转移，返回与 Gemini 版一致的 dict。"""

        prompt = qwen_evaluator_prompt(
            task,
            state_summary,
            action_text,
            next_state_summary,
            success,
            chunk_index,
            chunk_count,
            attempt,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=JITRL_MAX_EVALUATOR_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids = generated[0, inputs["input_ids"].shape[1] :]
        raw_output = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        parsed = _extract_json_object(raw_output)
        try:
            score = int(parsed["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Qwen evaluator missing/invalid score: {raw_output!r}") from error
        if not 0 <= score <= 3:
            raise ValueError(f"Qwen evaluator score out of range: {score}")
        usefulness = "useful" if score > 0 else "neutral"
        certainty = str(parsed.get("certainty", "certain"))
        result = str(parsed.get("result", "")).strip()
        if not result:
            result = usefulness

        return {
            "backend": "qwen_local",
            "model": self.model_id,
            "prompt": prompt,
            "raw_output": raw_output,
            "result": result,
            "usefulness": usefulness,
            "certainty": certainty,
            "score": score,
        }
