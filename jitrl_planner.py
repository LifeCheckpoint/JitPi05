"""First-stage visual proposals and paired JitRL candidate selection."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

from settings import JITRL_QWEN_ID


_SEMANTIC_PART_WHITESPACE = re.compile(r"\s+")


def load_jitrl_planner():
    """Load the resident Qwen visual planner in NF4 on the first CUDA device."""

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(JITRL_QWEN_ID)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        JITRL_QWEN_ID,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()
    return model, processor


def _recent_subtasks(history: Sequence[str] | None) -> list[str]:
    if history is None:
        return []
    items = list(history)[-2:]
    if any(not isinstance(item, str) for item in items):
        raise TypeError("recent_subtasks must contain only strings")
    return [item.strip() for item in items if item.strip()]


def _history_text(history: Sequence[str]) -> str:
    if not history:
        return "None"
    return "\n".join(f"- {item}" for item in history)


def proposal_prompt(task: str, recent_subtasks: Sequence[str] | None = None) -> str:
    """Build the strict three-candidate visual planning prompt."""

    history = _recent_subtasks(recent_subtasks)
    return (
        "You are the high-level planner for a Franka robot. Image 1 is the external "
        "camera and image 2 is the wrist camera. Thinking is disabled: do not emit "
        "reasoning or <think> tags.\n"
        f"Overall task: {task}\n"
        "Recently selected subtasks, oldest to newest:\n"
        f"{_history_text(history)}\n"
        "Infer the current manipulation state and propose exactly three distinct, "
        "immediately executable next subtasks. Return exactly ONE valid JSON object "
        "on one line, with no Markdown or surrounding text. The outer object MUST "
        "contain both state_summary and candidates; do not close it after "
        "state_summary. candidates MUST be a JSON array of exactly three objects. "
        "Copy this skeleton exactly and replace only string contents:\n"
        '{"state_summary":"STATE","candidates":['
        '{"text":"ACTION1","semantic_key":"skill|object|target"},'
        '{"text":"ACTION2","semantic_key":"skill|object|target"},'
        '{"text":"ACTION3","semantic_key":"skill|object|target"}]}\n'
        "state_summary must be concise, stable English in the structure "
        "'robot=...; gripper=...; object=...; target=...; relations=...; stage=...'. "
        "Each candidate text must be a short imperative English subtask. Each "
        "semantic_key must have exactly three lowercase fields separated by two | "
        "characters; choose descriptive skill, object, and target fields without "
        "using a fixed vocabulary. Output JSON only."
    )


def selection_prompt(
    task: str,
    state_summary: str,
    candidates: Sequence[dict],
    recent_subtasks: Sequence[str] | None = None,
) -> str:
    """Build the prompt whose first token selects one numbered candidate."""

    history = _recent_subtasks(recent_subtasks)
    if len(candidates) != 3:
        raise ValueError("selection prompt requires exactly 3 candidates")
    candidate_lines = "\n".join(
        f"{index}. {candidate['text']}" for index, candidate in enumerate(candidates, 1)
    )
    return (
        "You are selecting the immediate high-level subtask for a Franka robot. "
        "Image 1 is the external camera and image 2 is the wrist camera. Thinking "
        "is disabled.\n"
        f"Overall task: {task}\n"
        f"Current state: {state_summary}\n"
        "Recently selected subtasks, oldest to newest:\n"
        f"{_history_text(history)}\n"
        "Candidates:\n"
        f"{candidate_lines}\n"
        "Select the best candidate now. Output exactly one digit: 1, 2, or 3. "
        "Output no whitespace, explanation, or punctuation."
    )


def normalize_semantic_key(value: str) -> str:
    """Normalize a structurally valid ``skill|object|target`` key."""

    if not isinstance(value, str):
        raise ValueError("candidate semantic_key must be a string")
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError(
            "candidate semantic_key must contain exactly 3 fields separated by 2 '|'"
        )
    normalized = [
        _SEMANTIC_PART_WHITESPACE.sub("-", part.strip().lower()) or "none"
        for part in parts
    ]
    return "|".join(normalized)


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def parse_proposal_json(text: str) -> dict:
    """Extract and validate the first proposal JSON object."""

    if not isinstance(text, str):
        raise ValueError("planner output must be a string")
    cleaned = _strip_markdown_fence(text)
    object_start = cleaned.find("{")
    if object_start < 0:
        raise ValueError("planner output does not contain a JSON object")
    try:
        proposal, _ = json.JSONDecoder().raw_decode(cleaned[object_start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid planner JSON: {error.msg}") from error

    if not isinstance(proposal, dict):
        raise ValueError("planner JSON root must be an object")
    state_summary = proposal.get("state_summary")
    if not isinstance(state_summary, str) or not state_summary.strip():
        raise ValueError("planner JSON state_summary must be a non-empty string")
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("planner JSON must contain exactly 3 candidates")

    validated_candidates = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index + 1} must be an object")
        candidate_text = candidate.get("text")
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            raise ValueError(f"candidate {index + 1} text must be a non-empty string")
        try:
            semantic_key = normalize_semantic_key(candidate.get("semantic_key"))
        except ValueError as error:
            raise ValueError(f"candidate {index + 1}: {error}") from error
        validated_candidates.append(
            {"text": candidate_text.strip(), "semantic_key": semantic_key}
        )

    return {
        "state_summary": state_summary.strip(),
        "candidates": validated_candidates,
    }


def _validate_images(images: list[Image.Image]) -> None:
    if len(images) != 2:
        raise ValueError("exactly 2 camera images are required")
    if any(not isinstance(image, Image.Image) for image in images):
        raise TypeError("images must contain PIL.Image.Image instances")


def _planner_inputs(model, processor, images: list[Image.Image], prompt: str):
    _validate_images(images)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


@torch.inference_mode()
def generate_proposal(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: list[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = 256,
) -> dict:
    """Generate and parse one state summary with exactly three candidates."""

    prompt = proposal_prompt(task, recent_subtasks)
    inputs = _planner_inputs(model, processor, images, prompt)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids = generated[0, inputs["input_ids"].shape[1] :]
    output_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    proposal = parse_proposal_json(output_text)
    return {"prompt": prompt, **proposal}


def candidate_token_ids(tokenizer) -> list[int]:
    """Return single-token ids for the literal candidate labels 1, 2, and 3."""

    result = []
    for label in ("1", "2", "3"):
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"candidate label {label!r} must encode to exactly one token; got {token_ids}"
            )
        result.append(int(token_ids[0]))
    if len(set(result)) != 3:
        raise ValueError("candidate labels '1', '2', and '3' must have distinct token ids")
    return result


@torch.inference_mode()
def candidate_base_logits(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: list[Image.Image],
    prompt: str,
) -> tuple[list[int], torch.Tensor]:
    """Read only the three next-token logits at the final valid prompt token."""

    inputs = _planner_inputs(model, processor, images, prompt)
    outputs = model(**inputs)
    logits = outputs.logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("planner forward pass must return [1, sequence, vocabulary] logits")

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        last_token_index = inputs["input_ids"].shape[1] - 1
    else:
        valid_positions = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("planner prompt has no valid input tokens")
        last_token_index = int(valid_positions[-1])

    token_ids = candidate_token_ids(processor.tokenizer)
    selected = logits[0, last_token_index, token_ids]
    return token_ids, selected.detach().to(device="cpu", dtype=torch.float32)


@torch.inference_mode()
def propose_and_score(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: list[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = 256,
) -> dict:
    """Generate candidates and return their JSON-safe base-logit planning record."""

    history = _recent_subtasks(recent_subtasks)
    proposal = generate_proposal(
        model,
        processor,
        images,
        task,
        recent_subtasks=history,
        max_new_tokens=max_new_tokens,
    )
    score_prompt = selection_prompt(
        task,
        proposal["state_summary"],
        proposal["candidates"],
        recent_subtasks=history,
    )
    token_ids, base_logits = candidate_base_logits(
        model, processor, images, score_prompt
    )
    return {
        "prompt": proposal["prompt"],
        "selection_prompt": score_prompt,
        "state_summary": proposal["state_summary"],
        "candidates": proposal["candidates"],
        "candidate_token_ids": token_ids,
        "base_logits": base_logits.tolist(),
    }


def _float_vector(values: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        vector = values.detach().to(device="cpu", dtype=torch.float32)
    else:
        vector = torch.tensor(list(values), dtype=torch.float32)
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _cdf_choice(probabilities: torch.Tensor, uniform: float) -> int:
    cdf = probabilities.cumsum(dim=0)
    cdf[-1] = 1.0
    return int(torch.searchsorted(cdf, torch.tensor(uniform), right=True))


def apply_jitrl_update(
    base_logits: Sequence[float] | torch.Tensor,
    normalized_advantages: Sequence[float] | torch.Tensor,
    beta: float,
    temperature: float,
    uniform: float,
) -> dict:
    """Shift logits and make paired CDF draws; returned choice indices are 0-based."""

    base = _float_vector(base_logits, "base_logits")
    advantages = _float_vector(normalized_advantages, "normalized_advantages")
    if base.shape != advantages.shape:
        raise ValueError("base_logits and normalized_advantages must have the same length")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("uniform must be finite and in [0, 1)")

    updated = base + float(beta) * advantages
    base_probs = torch.softmax(base / float(temperature), dim=0)
    updated_probs = torch.softmax(updated / float(temperature), dim=0)
    base_choice_index = _cdf_choice(base_probs, float(uniform))
    updated_choice_index = _cdf_choice(updated_probs, float(uniform))

    return {
        "base_logits": base.tolist(),
        "updated_logits": updated.tolist(),
        "logit_shifts": (updated - base).tolist(),
        "base_probs": base_probs.tolist(),
        "updated_probs": updated_probs.tolist(),
        "uniform": float(uniform),
        "base_choice_index": base_choice_index,
        "updated_choice_index": updated_choice_index,
        "choice_changed": base_choice_index != updated_choice_index,
    }
