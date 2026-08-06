"""Qwen-only semantic-action planning and JitRL logit updates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

from jitpi05.config import (
    JITRL_FREE_CANDIDATES,
    JITRL_FREE_PLAN_TOKENS,
    JITRL_QWEN_ID,
    JITRL_TERMINATION_MODE,
)
from jitpi05.jitrl.memory import normalize_action
from jitpi05.jitrl.semantic_actions import (
    ACTION_LABELS,
    ACTION_TYPES,
    action_schema_text,
    active_actions,
    workspace_from_binding,
)

_ICL = """Example 1
Task: pick up the mug and put it in the basket
State: gripper open above the mug; mug not grasped
Binding: approach/align/grasp enabled for the mug; later-stage actions disabled; retract enabled; stop disabled.

Example 2
Task: put the mug in the basket
State: gripper holds the mug above the table; basket visible
Binding: lift/transport/place enabled for the mug and basket; approach/align/grasp disabled; release disabled until placement contact; retract enabled; stop disabled.

Example 3
Task: put the mug in the basket
State: mug appears stable in the basket and gripper is open
Binding: retract enabled to clear the object and expose the final relation; stop disabled by the diagnostic protocol."""

_SELECTION_ICL = """Selection examples:
- Open gripper far from the target: choose approach (label 1).
- Gripper holds the target away from its destination: choose transport (label 5).
- Object appears placed and the gripper is open: choose retract (label 8)."""


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
    return [item.strip() for item in (history or [])[-2:] if item.strip()]


def _history_text(history: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in history) or "None"


def workspace_prompt(
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    attempt: int = 1,
) -> str:
    """Prompt Qwen to bind the fixed semantic-action workspace to the current state."""

    retry = (
        ""
        if attempt == 1
        else (
            f"Schema correction attempt {attempt}: the previous JSON was invalid. "
            "Re-evaluate the images and return the exact compact binding schema.\n"
        )
    )
    return (
        "You are the frozen high-level policy for a Franka robot. Image 1 is the "
        "external camera and image 2 is the wrist camera. Thinking is disabled.\n"
        f"Overall task: {task}\n"
        "Recently executed semantic actions (action labels only, not verified facts):\n"
        f"{_history_text(_recent_subtasks(recent_subtasks))}\n"
        "Use the current images as authoritative evidence. Do not infer that an object "
        "is held, placed, or that the task is complete only because an earlier action "
        "label says grasp, place, release, or retract. The history only provides context "
        "for choosing the next action.\n"
        f"{retry}"
        "The action vocabulary below is fixed before evaluation. You must not invent "
        "or rename action types. Code constructs all nine action objects; you only bind "
        "the shared scene parameters and list the currently executable types.\n"
        f"{action_schema_text()}\n\n"
        f"{_ICL}\n\n"
        "Return one JSON object and no other text. Use this exact compact structure:\n"
        '{"state_summary":"robot=...; gripper=...; object=...; target=...; '
        'relations=...; stage=...","object":"...","destination":"...",'
        '"approach_target":"...","align_target":"...",'
        '"retract_direction":"upward","enabled_actions":'
        '["approach","align","grasp","retract"]}\n'
        "enabled_actions may contain only names from the fixed vocabulary and must not "
        "contain duplicates. Use concise English bindings grounded in visible objects, "
        "regions, directions, gripper state, and task progress. The object is the item "
        "to manipulate; destination is its task destination. approach_target and "
        "align_target may be the object or destination according to the current stage. "
        "This is an environment-only termination diagnostic: never include stop in "
        "enabled_actions, even if the task appears complete. When placement appears "
        "complete, enable retract to clear the object and permit another observation."
    )


def selection_prompt(
    task: str,
    state_summary: str,
    candidates: Sequence[dict],
    recent_subtasks: Sequence[str] | None = None,
) -> str:
    """Build the Qwen policy prompt over its own active workspace instances."""

    candidate_lines = "\n".join(
        f"{action['label']}. {action['type']}: {action['text']}" for action in candidates
    )
    labels = ", ".join(action["label"] for action in candidates)
    return (
        "You are the frozen high-level policy for a Franka robot. Image 1 is the "
        "external camera and image 2 is the wrist camera. Thinking is disabled.\n"
        f"Overall task: {task}\n"
        f"Current state: {state_summary}\n"
        "Recently executed semantic actions:\n"
        f"{_history_text(_recent_subtasks(recent_subtasks))}\n"
        "The following actions were bound by this same Qwen policy from the fixed "
        "pre-deployment workspace. Only these actions are currently valid:\n"
        f"{candidate_lines}\n"
        f"{_SELECTION_ICL}\n"
        f"Select the best immediate action. Output exactly one label from: {labels}. "
        "Output no whitespace, explanation, or punctuation."
    )


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _workspace_binding_diagnostics(binding: dict) -> str:
    required = (
        "state_summary",
        "object",
        "destination",
        "approach_target",
        "align_target",
        "retract_direction",
        "enabled_actions",
    )
    missing_fields = [field for field in required if field not in binding]
    enabled = list(binding.get("enabled_actions", []))
    unexpected = [action_type for action_type in enabled if action_type not in ACTION_TYPES]
    duplicates = sorted(
        {action_type for action_type in enabled if enabled.count(action_type) > 1}
    )
    return (
        f"missing_fields={missing_fields}, unexpected_actions={unexpected}, "
        f"duplicate_actions={duplicates}, enabled_actions={enabled}"
    )


def parse_evaluator_json(text: str) -> dict:
    """Extract one strict positive-only step-reward record."""

    cleaned = _strip_markdown_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("evaluator response must contain one JSON object")
    value = json.loads(cleaned[start : end + 1])
    score = value["score"]
    usefulness = value["usefulness"]
    expected = "useful" if score > 0 else "neutral"
    if not isinstance(score, int) or not 0 <= score <= 3 or usefulness != expected:
        raise ValueError("invalid positive-only evaluator record")
    return {
        "score": score,
        "usefulness": usefulness,
        "certainty": value["certainty"],
        "result": value["result"].strip(),
    }


def deployment_candidates(
    workspace: Sequence[dict],
    termination_mode: str = JITRL_TERMINATION_MODE,
) -> list[dict]:
    """Apply the method-independent deployment mask to Qwen's active workspace."""

    candidates = active_actions(list(workspace))
    if termination_mode == "environment_only_no_stop_v1":
        candidates = [candidate for candidate in candidates if candidate["type"] != "stop"]
    elif termination_mode != "qwen_stop_v1":
        raise ValueError(f"unknown JITRL termination mode: {termination_mode!r}")
    if not candidates:
        raise ValueError(
            "deployment mask left no executable actions; Qwen must enable at least "
            "one non-stop fixed action"
        )
    return candidates


def parse_workspace_json(text: str) -> dict:
    """Parse Qwen's compact binding and construct the complete fixed workspace."""

    cleaned = _strip_markdown_fence(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Qwen workspace response must contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    try:
        workspace = workspace_from_binding(value)
        qwen_candidates = active_actions(workspace)
        candidates = deployment_candidates(workspace)
    except Exception as error:
        diagnostics = _workspace_binding_diagnostics(value)
        raise ValueError(
            f"compact workspace binding invalid: {diagnostics}; reason={error}"
        ) from error
    return {
        "state_summary": value["state_summary"].strip(),
        "workspace": workspace,
        "qwen_candidates": qwen_candidates,
        "candidates": candidates,
        "termination_mode": JITRL_TERMINATION_MODE,
        "binding": value,
    }


def _planner_inputs(
    model,
    processor,
    images: Sequence[Image.Image],
    prompt: str,
    assistant_prefill: str | None = None,
):
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if assistant_prefill is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_prefill}],
            }
        )
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=assistant_prefill is None,
        continue_final_message=assistant_prefill is not None,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


@torch.inference_mode()
def generate_workspace(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = 768,
    attempt: int = 1,
) -> dict:
    """Use Qwen to abstract state and instantiate all fixed semantic templates."""

    prompt = workspace_prompt(task, recent_subtasks, attempt)
    assistant_prefill = "{"
    inputs = _planner_inputs(
        model,
        processor,
        images,
        prompt,
        assistant_prefill=assistant_prefill,
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids = generated[0, inputs["input_ids"].shape[1] :]
    generated_tokens = int(generated_ids.shape[0])
    generation_reached_limit = generated_tokens >= max_new_tokens
    raw_output = assistant_prefill + processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    try:
        parsed = parse_workspace_json(raw_output)
    except Exception as error:
        raise ValueError(
            f"{error}; generated_tokens={generated_tokens}; "
            f"max_new_tokens={max_new_tokens}; "
            f"generation_reached_limit={generation_reached_limit}; "
            f"Qwen output={raw_output!r}"
        ) from error
    return {
        "binding_prompt": prompt,
        "binding_raw_output": raw_output,
        "binding_assistant_prefill": assistant_prefill,
        "binding_generated_tokens": generated_tokens,
        "binding_generation_reached_limit": generation_reached_limit,
        **parsed,
    }


def action_label_token_ids(tokenizer, candidates: Sequence[dict]) -> list[int]:
    """Map active fixed-action labels to their single-token Qwen ids."""

    token_ids = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate["label"], add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action label {candidate['label']} is not one token")
        token_ids.append(int(encoded[0]))
    return token_ids


@torch.inference_mode()
def score_workspace(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    binding: dict,
    recent_subtasks: Sequence[str] | None = None,
) -> dict:
    """Read Qwen's raw logits for its current active semantic actions."""

    candidates = binding["candidates"]
    prompt = selection_prompt(
        task,
        binding["state_summary"],
        candidates,
        recent_subtasks,
    )
    inputs = _planner_inputs(model, processor, images, prompt)
    outputs = model(**inputs)
    last_token_index = int(torch.nonzero(inputs["attention_mask"][0])[-1])
    token_ids = action_label_token_ids(processor.tokenizer, candidates)
    logits = outputs.logits[0, last_token_index, token_ids]
    return {
        **binding,
        "selection_prompt": prompt,
        "candidate_token_ids": token_ids,
        "base_logits": logits.detach().to(device="cpu", dtype=torch.float32).tolist(),
    }


@torch.inference_mode()
def plan_and_score(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = 768,
    attempt: int = 1,
) -> dict:
    """Bind and score the fixed workspace with one frozen Qwen model."""

    binding = generate_workspace(
        model,
        processor,
        images,
        task,
        recent_subtasks,
        max_new_tokens,
        attempt,
    )
    return score_workspace(
        model,
        processor,
        images,
        task,
        binding,
        recent_subtasks,
    )


def _float_vector(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu", dtype=torch.float32)
    return torch.tensor(list(values), dtype=torch.float32)


def _cdf_choice(probabilities: torch.Tensor, uniform: float) -> int:
    cdf = probabilities.cumsum(dim=0)
    cdf[-1] = 1.0
    return int(torch.searchsorted(cdf, torch.tensor(uniform), right=True))


def _entropy(probabilities: torch.Tensor) -> float:
    return float(-(probabilities * probabilities.log()).sum())


def apply_jitrl_update(
    base_logits: Sequence[float] | torch.Tensor,
    normalized_advantages: Sequence[float] | torch.Tensor,
    beta: float,
    temperature: float,
    uniform: float,
) -> dict:
    """Apply the closed-form JitRL update to Qwen's fixed-workspace logits."""

    base = _float_vector(base_logits)
    advantages = _float_vector(normalized_advantages)
    updated = base + float(beta) * advantages
    base_probs = torch.softmax(base / float(temperature), dim=0)
    updated_probs = torch.softmax(updated / float(temperature), dim=0)
    base_choice_index = _cdf_choice(base_probs, float(uniform))
    updated_choice_index = _cdf_choice(updated_probs, float(uniform))
    update_kl = float((updated_probs * (updated_probs.log() - base_probs.log())).sum())
    return {
        "base_logits": base.tolist(),
        "updated_logits": updated.tolist(),
        "logit_shifts": (updated - base).tolist(),
        "base_probs": base_probs.tolist(),
        "updated_probs": updated_probs.tolist(),
        "base_entropy": _entropy(base_probs),
        "updated_entropy": _entropy(updated_probs),
        "update_kl": update_kl,
        "uniform": float(uniform),
        "base_choice_index": base_choice_index,
        "updated_choice_index": updated_choice_index,
        "choice_changed": base_choice_index != updated_choice_index,
    }


_FREE_PROPOSAL_ICL = """Example 1
Task: put both the alphabet soup and the tomato sauce in the basket
State: alphabet soup in hand above the basket; tomato sauce still on the table
Candidates: ["place the alphabet soup in the basket", "approach the tomato sauce", "grasp the tomato sauce", "carry the alphabet soup to the basket", "stop"]

Example 2
Task: put the white mug on the plate and put the chocolate pudding to the right of the plate
State: white mug held above the plate; chocolate pudding on the table
Candidates: ["place the white mug on the plate", "approach the chocolate pudding", "grasp the chocolate pudding", "carry the chocolate pudding toward the plate", "stop"]"""

# ICL variant used when Qwen is forbidden from proposing "stop": every example
# candidate is a concrete manipulation action and termination is never suggested.
_FREE_PROPOSAL_ICL_NO_STOP = """Example 1
Task: put both the alphabet soup and the tomato sauce in the basket
State: alphabet soup in hand above the basket; tomato sauce still on the table
Candidates: ["place the alphabet soup in the basket", "approach the tomato sauce", "grasp the tomato sauce", "carry the tomato sauce to the basket", "open the basket"]

Example 2
Task: put the white mug on the plate and put the chocolate pudding to the right of the plate
State: white mug held above the plate; chocolate pudding on the table
Candidates: ["place the white mug on the plate", "approach the chocolate pudding", "grasp the chocolate pudding", "carry the chocolate pudding toward the plate", "move the plate aside"]"""

_FREE_SELECTION_ICL = """Selection examples:
- Object not yet grasped: choose an approach or grasp action.
- Object held away from its destination: choose a carry or place action.
- Task appears complete and the gripper is open: choose stop."""


def free_proposal_prompt(
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    attempt: int = 1,
    candidate_count: int = JITRL_FREE_CANDIDATES,
    termination_mode: str = JITRL_TERMINATION_MODE,
) -> str:
    """Prompt Qwen to freely propose candidate actions plus a compact state summary.

    Under ``environment_only_no_stop_v1`` the prompt forbids a ``stop`` candidate
    outright, so the model never emits a termination suggestion; termination is
    decided solely by the environment evaluator.
    """

    retry = (
        ""
        if attempt == 1
        else (
            f"Proposal correction attempt {attempt}: the previous JSON was invalid. "
            "Re-evaluate the images and return the exact compact proposal schema.\n"
        )
    )
    if termination_mode == "environment_only_no_stop_v1":
        stop_instruction = (
            'Do NOT include the candidate "stop" in the list. The robot never '
            "decides to stop; task completion and episode termination are decided "
            "by the environment evaluator. Every candidate must be a concrete "
            "manipulation action the robot should perform next."
        )
        icl = _FREE_PROPOSAL_ICL_NO_STOP
    elif termination_mode == "qwen_stop_v1":
        stop_instruction = (
            'Include the literal candidate "stop" only when the task appears '
            "complete or cannot continue."
        )
        icl = _FREE_PROPOSAL_ICL
    else:
        raise ValueError(f"unknown JITRL termination mode: {termination_mode!r}")
    return (
        "You are the frozen high-level policy for a Franka robot. Image 1 is the "
        "external camera and image 2 is the wrist camera. Thinking is disabled.\n"
        f"Overall task: {task}\n"
        "Recently executed actions (action text only, not verified facts):\n"
        f"{_history_text(_recent_subtasks(recent_subtasks))}\n"
        "Use the current images as authoritative evidence. Do not infer that an "
        "object is held, placed, or that the task is complete only because an earlier "
        "action says so.\n"
        f"{retry}"
        f"Propose exactly {candidate_count} distinct immediate manipulation actions "
        "the robot could take next. Each candidate must be a single short imperative "
        'verb phrase (for example "grasp the alphabet soup"). '
        f"{stop_instruction} "
        "Do not explain or number the candidates.\n"
        f"{icl}\n\n"
        "Return one JSON object and no other text. Use this exact compact structure:\n"
        '{"state_summary":"robot=...; gripper=...; object=...; target=...; '
        'relations=...; stage=...","candidates":["...", "...", "..."]}\n'
        "candidates must be a JSON array of distinct non-empty strings."
    )


def free_selection_prompt(
    task: str,
    state_summary: str,
    candidates: Sequence[dict],
    recent_subtasks: Sequence[str] | None = None,
) -> str:
    """Build the Qwen selection prompt over free-text candidate actions."""

    candidate_lines = "\n".join(
        f"{action['label']}. {action['text']}" for action in candidates
    )
    labels = ", ".join(action["label"] for action in candidates)
    return (
        "You are the frozen high-level policy for a Franka robot. Image 1 is the "
        "external camera and image 2 is the wrist camera. Thinking is disabled.\n"
        f"Overall task: {task}\n"
        f"Current state: {state_summary}\n"
        "Recently executed actions:\n"
        f"{_history_text(_recent_subtasks(recent_subtasks))}\n"
        "The following candidate actions were proposed by this same Qwen policy:\n"
        f"{candidate_lines}\n"
        f"{_FREE_SELECTION_ICL}\n"
        f"Select the best immediate action. Output exactly one label from: {labels}. "
        "Output no whitespace, explanation, or punctuation."
    )


def parse_free_planning_json(
    text: str, candidate_count: int = JITRL_FREE_CANDIDATES
) -> dict:
    """Parse Qwen's free proposal into a deployment-compatible candidate list."""

    cleaned = _strip_markdown_fence(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("free proposal must contain a JSON object")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    state_summary = str(value.get("state_summary", "")).strip()
    if not state_summary:
        raise ValueError("free proposal state_summary must be non-empty")
    raw_candidates = value.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("free proposal candidates must be a JSON array")

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        candidate_text = str(raw).strip()
        if not candidate_text:
            continue
        normalized = normalize_action(candidate_text)
        if normalized in seen:
            continue
        seen.add(normalized)
        is_stop = normalized == "stop"
        candidates.append(
            {
                "label": str(len(candidates) + 1),
                "type": "stop" if is_stop else "free",
                "text": candidate_text,
                "semantic_key": normalized,
                "enabled": True,
                "terminates_rollout": is_stop,
            }
        )
    if len(candidates) == 0:
        raise ValueError("free proposal must contain at least 1 candidate (got 0)")
    if len(candidates) > candidate_count:
        candidates = candidates[:candidate_count]
        for index, candidate in enumerate(candidates):
            candidate["label"] = str(index + 1)
    # A single candidate is accepted: if it is "stop" it is a termination signal,
    # otherwise it is the only actionable choice (JitRL cannot re-rank one option).
    stop_only = len(candidates) == 1 and candidates[0]["type"] == "stop"
    return {
        "state_summary": state_summary,
        "workspace": [],
        "qwen_candidates": [candidate.copy() for candidate in candidates],
        "candidates": candidates,
        "termination_mode": JITRL_TERMINATION_MODE,
        "binding": value,
        "stop_only": stop_only,
    }


def deployment_candidates_free(
    candidates: Sequence[dict],
    termination_mode: str = JITRL_TERMINATION_MODE,
) -> list[dict]:
    """Apply the method-independent deployment mask to free-text candidates.

    In the no-stop diagnostic the literal ``"stop"`` candidate is removed so the
    episode can only terminate through the environment predicate, mirroring the
    fixed-workspace deployment mask. Labels are renumbered so the surviving set is
    contiguous for selection scoring. An empty result means Qwen proposed only
    ``"stop"`` (a disabled termination) and the caller decides how to continue.
    """

    if termination_mode == "environment_only_no_stop_v1":
        deployed = [
            candidate.copy()
            for candidate in candidates
            if candidate["type"] != "stop"
        ]
    elif termination_mode == "qwen_stop_v1":
        deployed = [candidate.copy() for candidate in candidates]
    else:
        raise ValueError(f"unknown JITRL termination mode: {termination_mode!r}")
    for index, candidate in enumerate(deployed):
        candidate["label"] = str(index + 1)
    return deployed


@torch.inference_mode()
def generate_free_candidates(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = JITRL_FREE_PLAN_TOKENS,
    candidate_count: int = JITRL_FREE_CANDIDATES,
    attempt: int = 1,
) -> dict:
    """Use Qwen to propose free-text candidate actions and a state summary."""

    prompt = free_proposal_prompt(task, recent_subtasks, attempt, candidate_count)
    assistant_prefill = "{"
    inputs = _planner_inputs(
        model,
        processor,
        images,
        prompt,
        assistant_prefill=assistant_prefill,
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids = generated[0, inputs["input_ids"].shape[1] :]
    generated_tokens = int(generated_ids.shape[0])
    generation_reached_limit = generated_tokens >= max_new_tokens
    raw_output = assistant_prefill + processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    try:
        parsed = parse_free_planning_json(raw_output, candidate_count)
    except Exception as error:
        raise ValueError(
            f"{error}; generated_tokens={generated_tokens}; "
            f"max_new_tokens={max_new_tokens}; "
            f"generation_reached_limit={generation_reached_limit}; "
            f"Qwen output={raw_output!r}"
        ) from error
    return {
        "binding_prompt": prompt,
        "binding_raw_output": raw_output,
        "binding_assistant_prefill": assistant_prefill,
        "binding_generated_tokens": generated_tokens,
        "binding_generation_reached_limit": generation_reached_limit,
        **parsed,
    }


@torch.inference_mode()
def score_free_candidates(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    proposal: dict,
    recent_subtasks: Sequence[str] | None = None,
) -> dict:
    """Read Qwen's raw logits for its own free-text candidate actions."""

    candidates = proposal["candidates"]
    prompt = free_selection_prompt(
        task,
        proposal["state_summary"],
        candidates,
        recent_subtasks,
    )
    inputs = _planner_inputs(model, processor, images, prompt)
    outputs = model(**inputs)
    last_token_index = int(torch.nonzero(inputs["attention_mask"][0])[-1])
    token_ids = action_label_token_ids(processor.tokenizer, candidates)
    logits = outputs.logits[0, last_token_index, token_ids]
    return {
        **proposal,
        "selection_prompt": prompt,
        "candidate_token_ids": token_ids,
        "base_logits": logits.detach().to(device="cpu", dtype=torch.float32).tolist(),
    }


@torch.inference_mode()
def plan_and_score_free(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: Sequence[Image.Image],
    task: str,
    recent_subtasks: Sequence[str] | None = None,
    max_new_tokens: int = JITRL_FREE_PLAN_TOKENS,
    attempt: int = 1,
) -> dict:
    """Propose free-text candidates, apply the deployment mask, then score them.

    Under ``environment_only_no_stop_v1`` the ``stop`` candidate is masked before
    scoring, so episodes can only terminate through the environment predicate.
    ``qwen_candidates`` retains Qwen's raw (possibly stop-containing) proposal for
    the audit trace while ``candidates`` holds the deployed scoring set.
    """

    proposal = generate_free_candidates(
        model,
        processor,
        images,
        task,
        recent_subtasks,
        max_new_tokens,
        JITRL_FREE_CANDIDATES,
        attempt,
    )
    qwen_candidates = proposal["candidates"]
    deployed = deployment_candidates_free(qwen_candidates, JITRL_TERMINATION_MODE)
    proposal = {
        **proposal,
        "qwen_candidates": [candidate.copy() for candidate in qwen_candidates],
        "candidates": deployed,
    }
    if not deployed:
        # Qwen proposed only "stop" and stop is disabled under environment-only
        # termination. Return a signal so the rollout can continue with the
        # previous action instead of terminating or crashing.
        return {
            **proposal,
            "stop_only_masked": True,
        }
    return score_free_candidates(
        model,
        processor,
        images,
        task,
        proposal,
        recent_subtasks,
    )
