import torch
from transformers import AutoTokenizer

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from settings import ORACLE_SUBTASK, PI05_ID, PI05_TOKENIZER_ID, SEED


def make_pi05_processors(config, stats, device: str):
    tokenizer = AutoTokenizer.from_pretrained(PI05_TOKENIZER_ID)
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=norm_map,
                stats=stats,
            ),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
            TokenizerProcessorStep(
                tokenizer=tokenizer,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=norm_map,
                stats=stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


@torch.inference_mode()
def explicit_pi05_flow(
    policy: PI05Policy,
    batch: dict,
    initial_noise: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """显式展开 π₀.₅ prefix 编码、KV cache 和 flow-matching 去噪循环。"""
    core = policy.model
    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embeddings, prefix_pad_masks, prefix_attention_masks = core.embed_prefix(
        images, image_masks, tokens, token_masks
    )
    prefix_attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_attention_masks)
    prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_attention_4d = core._prepare_attention_masks_4d(prefix_attention_2d)
    core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )
    _, past_key_values = core.paligemma_with_expert.forward(
        attention_mask=prefix_attention_4d,
        position_ids=prefix_positions,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=True,
    )

    actions = initial_noise.clone()
    trajectory = [actions.detach().cpu()]
    vector_fields = []
    dt = -1.0 / policy.config.num_inference_steps
    batch_size = tokens.shape[0]
    for step in range(policy.config.num_inference_steps):
        time = torch.full(
            (batch_size,),
            1.0 + step * dt,
            dtype=torch.float32,
            device=tokens.device,
        )
        vector_field = core.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=actions,
            timestep=time,
        )
        actions = actions + dt * vector_field
        vector_fields.append(vector_field.detach().cpu())
        trajectory.append(actions.detach().cpu())

    action_dim = policy.config.output_features["action"].shape[0]
    return actions[:, :, :action_dim], {
        "trajectory": torch.cat(trajectory, dim=0),
        "vector_fields": torch.cat(vector_fields, dim=0),
        "prefix_tokens": tokens.detach().cpu(),
        "prefix_token_mask": token_masks.detach().cpu(),
    }


def conditioned_task(overall_task: str, subtask: str) -> str:
    return f"Overall task: {overall_task}. Current subtask: {subtask}."


def action_metrics(
    action: torch.Tensor,
    demonstration: torch.Tensor,
    baseline: torch.Tensor,
) -> dict:
    horizon = min(action.shape[0], demonstration.shape[0])
    error = action[:horizon] - demonstration[:horizon]
    baseline_delta = action - baseline
    return {
        "shape": list(action.shape),
        "all_finite": bool(torch.isfinite(action).all()),
        "min": float(action.min()),
        "max": float(action.max()),
        "mean": float(action.mean()),
        "mean_step_l2_norm": float(torch.linalg.vector_norm(action, dim=-1).mean()),
        "demo_mae": float(error.abs().mean()),
        "demo_rmse": float(error.square().mean().sqrt()),
        "delta_from_original_rmse": float(baseline_delta.square().mean().sqrt()),
    }


def run_pi05_comparison(
    dataset: LeRobotDataset,
    frame: dict,
    demonstration: torch.Tensor,
    plain_subtask: str,
    modulated_subtask: str,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], torch.Tensor]:
    print(f"Loading low-level policy: {PI05_ID}")
    config = PreTrainedConfig.from_pretrained(PI05_ID)
    config.device = "cuda"
    config.dtype = "bfloat16"
    policy = PI05Policy.from_pretrained(PI05_ID, config=config).eval()
    preprocessor, postprocessor = make_pi05_processors(config, dataset.meta.stats, "cuda")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    initial_noise = torch.randn(
        1,
        config.chunk_size,
        config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    ).to("cuda")

    task = frame["task"]
    conditions = {
        "original_task": task,
        "oracle_subtask": conditioned_task(task, ORACLE_SUBTASK),
        "qwen_subtask": conditioned_task(task, plain_subtask),
        "modulated_qwen_subtask": conditioned_task(task, modulated_subtask),
    }
    actions: dict[str, torch.Tensor] = {}
    traces: dict[str, dict] = {}
    for name, condition in conditions.items():
        conditioned_frame = dict(frame)
        conditioned_frame["task"] = condition
        batch = preprocessor(conditioned_frame)
        normalized_action, trace = explicit_pi05_flow(policy, batch, initial_noise)
        actions[name] = postprocessor(normalized_action).squeeze(0)
        traces[name] = trace
        print(f"Finished π₀.₅ condition: {name}")

    baseline = actions["original_task"]
    metrics = {
        name: action_metrics(action, demonstration, baseline)
        for name, action in actions.items()
    }
    return actions, {"conditions": conditions, "metrics": metrics, "traces": traces}, initial_noise.cpu()
