import gc
from typing import Callable

import torch
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, Qwen3_5ForConditionalGeneration

from data import tensor_to_pil
from settings import BIAS_PHRASE, BIAS_VALUE, MAX_PLAN_TOKENS, QWEN_ID, TOP_K

LogitsModifier = Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]


def planning_prompt(task: str) -> str:
    return (
        "You are the high-level planner for a Franka robot. "
        "Image 1 is the external camera and image 2 is the wrist camera. "
        f"Overall task: {task}. "
        "Infer the single immediate manipulation subtask that should be executed now. "
        "Return only one short imperative English verb phrase, without explanation, labels, or punctuation."
    )


def topk_summary(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    probabilities = logits.float().softmax(dim=-1)
    values, indices = probabilities.topk(k, dim=-1)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
            "probability": float(probability),
        }
        for token_id, probability in zip(indices[0], values[0], strict=True)
    ]


class InspectableLogitsProcessor(LogitsProcessor):
    """记录逐 token logits，并在确定性 argmax 前暴露一个可编辑回调。"""

    def __init__(
        self,
        prompt_length: int,
        tokenizer,
        modifier: LogitsModifier | None = None,
    ):
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.modifier = modifier
        self.raw_logits: list[torch.Tensor] = []
        self.modified_logits: list[torch.Tensor] = []
        self.steps: list[dict] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = input_ids.shape[1] - self.prompt_length
        raw = scores.detach().clone()
        modified = raw.clone()
        if self.modifier is not None:
            modified = self.modifier(step, input_ids, modified)

        selected_id = int(modified.argmax(dim=-1)[0])
        self.raw_logits.append(raw.to(device="cpu", dtype=torch.bfloat16))
        self.modified_logits.append(modified.detach().to(device="cpu", dtype=torch.bfloat16))
        self.steps.append(
            {
                "step": step,
                "raw_top_k": topk_summary(raw, self.tokenizer, TOP_K),
                "modified_top_k": topk_summary(modified, self.tokenizer, TOP_K),
                "selected_token_id": selected_id,
                "selected_token": self.tokenizer.decode(
                    [selected_id], clean_up_tokenization_spaces=False
                ),
            }
        )
        return modified


def make_phrase_bias(tokenizer, phrase: str, bias: float) -> tuple[LogitsModifier, list[int]]:
    """依次提高目标短语 token 的 logit；这是待替换的最小调制实验入口。"""
    # assistant 回复通常以空格开头，因此显式加入前导空格以匹配首 token。
    target_ids = tokenizer.encode(f" {phrase}", add_special_tokens=False)

    def modifier(step: int, _input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if step < len(target_ids):
            logits[:, target_ids[step]] += bias
        return logits

    return modifier, target_ids


def clean_subtask(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text.strip()
    for prefix in ("Subtask:", "Current subtask:", "Action:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text.strip(" \t\n\r.\"'")


@torch.inference_mode()
def generate_subtask(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: list[Image.Image],
    task: str,
    modifier: LogitsModifier | None = None,
) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": planning_prompt(task)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    logits_processor = InspectableLogitsProcessor(
        prompt_length=inputs["input_ids"].shape[1],
        tokenizer=processor.tokenizer,
        modifier=modifier,
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_PLAN_TOKENS,
        do_sample=False,
        use_cache=True,
        logits_processor=[logits_processor],
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    generated_ids = generated[0, inputs["input_ids"].shape[1] :].detach().cpu()
    full_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "prompt": planning_prompt(task),
        "text": clean_subtask(full_text),
        "full_text": full_text,
        "generated_ids": generated_ids,
        "raw_logits": torch.cat(logits_processor.raw_logits, dim=0),
        "modified_logits": torch.cat(logits_processor.modified_logits, dim=0),
        "steps": logits_processor.steps,
    }


def load_high_level_planner():
    print(f"Loading high-level planner: {QWEN_ID}")
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        QWEN_ID,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    ).eval()
    return model, processor


def release_high_level_planner(model, processor) -> None:
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def generate_plan_pair(
    model: Qwen3_5ForConditionalGeneration,
    processor,
    images: list[Image.Image],
    task: str,
    bias_phrase: str,
    bias_value: float = BIAS_VALUE,
) -> tuple[dict, dict, list[int]]:
    plain = generate_subtask(model, processor, images, task)
    modifier, target_ids = make_phrase_bias(processor.tokenizer, bias_phrase, bias_value)
    modulated = generate_subtask(model, processor, images, task, modifier)
    return plain, modulated, target_ids


def run_high_level_planner(frame: dict) -> tuple[dict, dict, list[int]]:
    model, processor = load_high_level_planner()

    images = [
        tensor_to_pil(frame["observation.images.image"]),
        tensor_to_pil(frame["observation.images.image2"]),
    ]
    plain, modulated, target_ids = generate_plan_pair(
        model, processor, images, frame["task"], BIAS_PHRASE
    )

    print(f"Qwen subtask: {plain['text']}")
    print(f"Modulated Qwen subtask: {modulated['text']}")

    release_high_level_planner(model, processor)
    return plain, modulated, target_ids


def json_planning_result(result: dict) -> dict:
    return {
        "prompt": result["prompt"],
        "text": result["text"],
        "full_text": result["full_text"],
        "generated_ids": result["generated_ids"].tolist(),
        "steps": result["steps"],
    }
