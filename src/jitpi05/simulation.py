import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.io_utils import write_video
from PIL import Image
from transformers import AutoTokenizer

from jitpi05.config import (
    BIAS_VALUE,
    JITRL_BENCHMARK_PANEL,
    JITRL_LOW_LEVEL_BACKEND,
    PI05_TOKENIZER_ID,
    QWEN_ID,
    SEED,
    SIM_ACTION_STEPS,
    SIM_EPISODES,
    SIM_OUTPUT_DIR,
    SIM_PI05_ID,
    SIM_TASKS,
)
from jitpi05.planning import (
    generate_plan_pair,
    json_planning_result,
    load_high_level_planner,
    release_high_level_planner,
)
from jitpi05.policy import conditioned_task

# LIBERO/MuJoCo 在无显示器 Linux 机器上通常需要 EGL；保留用户显式设置。
os.environ.setdefault("MUJOCO_GL", "egl")


def make_single_env(
    task_spec: dict,
    episode_index: int,
    budget_max_steps: int | None = None,
):
    """创建 batch=1 环境，并把唯一子环境固定到指定 LIBERO init state。

    ``budget_max_steps`` 覆盖任务的官方 ``max_steps`` 作为环境 episode_length
    （JitRL 评测用它为 Cycle 回溯叠加统一预算；SIM/离线对比不传则保持原值）。
    """
    config = LiberoEnvConfig(
        task=task_spec["suite"],
        task_ids=[task_spec["task_id"]],
        episode_length=(
            budget_max_steps
            if budget_max_steps is not None
            else task_spec["max_steps"]
        ),
        observation_height=256,
        observation_width=256,
    )
    envs = make_env(config, n_envs=1)
    env = envs[task_spec["suite"]][task_spec["task_id"]]
    env.set_attr("init_state_id", episode_index)
    env_preprocessor, env_postprocessor = config.get_env_processors()
    return envs, env, env_preprocessor, env_postprocessor


def disable_robosuite_horizon_done(vector_env: Any) -> None:
    """让底层 robosuite 环境永不自行按固定 ``horizon`` 终止。

    LIBERO 包装层（LiberoEnv）在下层 robosuite ``bddl_base_domain.step`` 里会用
    ``_check_success()`` 覆盖 robosuite 返回的 ``done``，因此上层永远看不到
    robosuite 基于 ``horizon`` 的终止信号，也不会触发 ``reset()``。此时 robosuite
    内部的 ``self.done`` 残留在 ``True``，下一次 ``env.step`` 就会抛
    ``ValueError("executing action in terminated episode")`` —— 只要外层
    ``episode_length``（动态预算可能到 1600）超过 robosuite 硬编码的
    ``horizon``（1000）就会复现。

    解法：把 ``ignore_done`` 置为 ``True``，让 robosuite 不再自行按 horizon
    终止；终止完全交给上层 ``max_steps`` 步数预算，任务成功仍由
    ``_check_success()`` 独立提供（LiberoEnv.step 中 ``terminated = done or
    is_success``，其中 ``done`` 已被覆盖为 ``_check_success()``）。

    该属性在 robosuite 环境对象生命周期内保持不变（``reset()`` 不会重设
    ``ignore_done``），因此对同一 episode 只需在首次 ``reset`` 后设置一次。
    """

    try:
        children = getattr(vector_env, "envs", None)
        if not children:
            return
        libero_env = children[0]
        control = getattr(libero_env, "_env", None)
        if control is None or not hasattr(control, "env"):
            return
        robosuite_env = getattr(control, "env", None)
        if robosuite_env is None or not hasattr(robosuite_env, "ignore_done"):
            return
        robosuite_env.ignore_done = True
    except Exception:
        # 环境内省失败不应阻断 rollout；保持 best-effort。
        return


def policy_frame(observation: dict, task: str, env_preprocessor) -> dict:
    """沿用官方 eval 顺序：numpy observation -> LeRobot 格式 -> LIBERO processor。"""
    frame = preprocess_observation(observation)
    frame["task"] = [task]
    return env_preprocessor(frame)


def planning_images(frame: dict) -> list[Image.Image]:
    images = []
    for key in ("observation.images.image", "observation.images.image2"):
        array = (
            frame[key][0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .numpy()
        )
        images.append(Image.fromarray(array))
    return images


def collect_plans() -> dict:
    """一次加载 Qwen，依次读取所有固定首帧并保存 plain/modulated 规划。"""
    model, processor = load_high_level_planner()
    plans: dict[str, list[dict]] = {}
    try:
        for task_spec in SIM_TASKS:
            task_name = task_spec["name"]
            plans[task_name] = []
            for episode_index in range(SIM_EPISODES):
                envs, env, env_preprocessor, _ = make_single_env(
                    task_spec, episode_index
                )
                try:
                    observation, _ = env.reset(seed=[SEED + episode_index])
                    task = list(env.call("task_description"))[0]
                    frame = policy_frame(observation, task, env_preprocessor)
                    plain, modulated, target_ids = generate_plan_pair(
                        model,
                        processor,
                        planning_images(frame),
                        task,
                        task_spec["bias_phrase"],
                    )
                    plans[task_name].append(
                        {
                            "task": task,
                            "plain": plain,
                            "modulated": modulated,
                            "target_ids": target_ids,
                        }
                    )
                    print(
                        f"Planned {task_name} episode {episode_index}: "
                        f"{plain['text']} -> {modulated['text']}"
                    )
                finally:
                    close_envs(envs)
    finally:
        release_high_level_planner(model, processor)
    return plans


def load_policy():
    if JITRL_LOW_LEVEL_BACKEND == "rlinf":
        return _load_rlinf_policy()
    print(f"Loading low-level policy: {SIM_PI05_ID}")
    config = PreTrainedConfig.from_pretrained(SIM_PI05_ID)
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.compile_model = False
    config.gradient_checkpointing = False
    config.n_action_steps = SIM_ACTION_STEPS
    policy = PI05Policy.from_pretrained(SIM_PI05_ID, config=config).eval()
    tokenizer = AutoTokenizer.from_pretrained(PI05_TOKENIZER_ID)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        SIM_PI05_ID,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer": tokenizer},
            "device_processor": {"device": "cuda"},
        },
    )
    return policy, preprocessor, postprocessor


def _load_rlinf_policy():
    """加载 RLinf π0.5 远程低层策略（独立 openpi 子进程）。"""
    from jitpi05.rlinf_client import (
        IdentityProcessor,
        RlinfPi05Client,
        RlinfPi05PolicyAdapter,
    )

    print("Loading low-level policy: RLinf-Pi05-LIBERO-130-fullshot-SFT (remote)")
    client = RlinfPi05Client()
    policy = RlinfPi05PolicyAdapter(client)
    identity = IdentityProcessor()
    return policy, identity, identity


def reset_rollout_state(policy, preprocessor, postprocessor, seed: int) -> None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_shared_noise(policy, max_steps: int, seed: int) -> torch.Tensor:
    """同一 episode 的四个条件复用一组按 chunk 排列的 flow-matching 初始噪声。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_chunks = (max_steps + SIM_ACTION_STEPS - 1) // SIM_ACTION_STEPS
    return torch.randn(
        max_chunks,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    ).to("cuda")


def success_from_info(info: dict) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            success = final_info.get("is_success", [False])
            return bool(success[0] if hasattr(success, "__getitem__") else success)
        item = final_info[0]
        return bool(item.get("is_success", False)) if isinstance(item, dict) else False
    success = info.get("is_success", [False])
    return bool(success[0] if hasattr(success, "__getitem__") else success)


def rollout_condition(
    task_spec: dict,
    episode_index: int,
    condition_name: str,
    condition: str,
    shared_noise: torch.Tensor,
    policy,
    preprocessor,
    postprocessor,
    output_dir: Path = SIM_OUTPUT_DIR,
) -> dict:
    envs, env, env_preprocessor, env_postprocessor = make_single_env(
        task_spec, episode_index
    )
    episode_seed = SEED + episode_index
    reset_rollout_state(policy, preprocessor, postprocessor, episode_seed)

    task = condition
    video_path = (
        output_dir
        / "videos"
        / task_spec["name"]
        / condition_name
        / f"episode_{episode_index}.mp4"
    )
    actions: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    rewards: list[float] = []
    frames: list[np.ndarray] = []
    success = False
    try:
        observation, _ = env.reset(seed=[episode_seed])
        task = list(env.call("task_description"))[0]
        max_steps = int(env.call("_max_episode_steps")[0])
        frames.append(env.render()[0])

        while len(actions) < max_steps and not success:
            chunk_index = len(action_chunks)
            frame = policy_frame(observation, condition, env_preprocessor)
            batch = preprocessor(frame)
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(
                    batch,
                    noise=shared_noise[chunk_index].unsqueeze(0),
                )
            action_chunk = postprocessor(normalized_chunk).squeeze(0)
            action_chunks.append(action_chunk.detach().cpu())

            for action in action_chunk[:SIM_ACTION_STEPS]:
                action_transition = env_postprocessor({"action": action.unsqueeze(0)})
                env_action = action_transition["action"].detach().cpu()
                observation, reward, terminated, truncated, info = env.step(
                    env_action.numpy()
                )
                actions.append(env_action.squeeze(0))
                rewards.append(float(reward[0]))
                frames.append(env.render()[0])

                done = bool(terminated[0] or truncated[0])
                success = success or success_from_info(info)
                if done or success or len(actions) >= max_steps:
                    break
    finally:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            write_video(video_path, frames, env.metadata.get("render_fps", 80))
        close_envs(envs)

    return {
        "task": task,
        "condition": condition,
        "success": success,
        "steps": len(actions),
        "sum_reward": sum(rewards),
        "max_reward": max(rewards, default=0.0),
        "actions": torch.stack(actions) if actions else torch.empty(0, 7),
        "action_chunks": torch.stack(action_chunks)
        if action_chunks
        else torch.empty(0, 50, 7),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "video_path": str(video_path),
    }


def run_simulation_evaluation(plans: dict, output_dir: Path = SIM_OUTPUT_DIR) -> dict:
    policy, preprocessor, postprocessor = load_policy()
    results: dict[str, list[dict]] = {}
    try:
        for task_spec in SIM_TASKS:
            task_name = task_spec["name"]
            results[task_name] = []
            for episode_index, plan in enumerate(plans[task_name]):
                task = plan["task"]
                shared_noise = make_shared_noise(
                    policy,
                    task_spec["max_steps"],
                    SEED + episode_index,
                )
                conditions = {
                    "original_task": task,
                    "oracle_subtask": conditioned_task(
                        task, task_spec["oracle_subtask"]
                    ),
                    "qwen_subtask": conditioned_task(task, plan["plain"]["text"]),
                    "modulated_qwen_subtask": conditioned_task(
                        task, plan["modulated"]["text"]
                    ),
                }
                episode_result = {"episode_index": episode_index, "conditions": {}}
                for name, condition in conditions.items():
                    print(f"Running {task_name} episode {episode_index}: {name}")
                    episode_result["conditions"][name] = rollout_condition(
                        task_spec,
                        episode_index,
                        name,
                        condition,
                        shared_noise,
                        policy,
                        preprocessor,
                        postprocessor,
                        output_dir,
                    )
                results[task_name].append(episode_result)
                del shared_noise
    finally:
        del policy, preprocessor, postprocessor
        gc.collect()
        torch.cuda.empty_cache()
    return results


def summarize(plans: dict, results: dict) -> dict:
    summary: dict[str, Any] = {
        "models": {"planner": QWEN_ID, "policy": SIM_PI05_ID},
        "episodes_per_task": SIM_EPISODES,
        "action_steps_before_replan": SIM_ACTION_STEPS,
        "seed": SEED,
        "tasks": {},
    }
    for task_spec in SIM_TASKS:
        task_name = task_spec["name"]
        task_results = results[task_name]
        condition_names = task_results[0]["conditions"].keys()
        summary["tasks"][task_name] = {
            "suite": task_spec["suite"],
            "task_id": task_spec["task_id"],
            "zero_shot": task_spec["zero_shot"],
            "task": plans[task_name][0]["task"],
            "oracle_subtask": task_spec["oracle_subtask"],
            "bias_phrase": task_spec["bias_phrase"],
            "planning": [
                {
                    "episode_index": index,
                    "plain": json_planning_result(plan["plain"]),
                    "modulated": json_planning_result(plan["modulated"]),
                    "target_token_ids": plan["target_ids"],
                    "bias_value": BIAS_VALUE,
                }
                for index, plan in enumerate(plans[task_name])
            ],
            "conditions": {
                name: {
                    "successes": [
                        episode["conditions"][name]["success"]
                        for episode in task_results
                    ],
                    "success_rate": sum(
                        episode["conditions"][name]["success"]
                        for episode in task_results
                    )
                    / len(task_results),
                    "mean_steps": sum(
                        episode["conditions"][name]["steps"] for episode in task_results
                    )
                    / len(task_results),
                    "mean_sum_reward": sum(
                        episode["conditions"][name]["sum_reward"]
                        for episode in task_results
                    )
                    / len(task_results),
                    "videos": [
                        episode["conditions"][name]["video_path"]
                        for episode in task_results
                    ],
                }
                for name in condition_names
            },
        }
    return summary


def save_simulation_artifacts(
    plans: dict, results: dict, output_dir: Path = SIM_OUTPUT_DIR
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    planning_tensors = {
        task_name: [
            {
                "plain": {
                    "generated_ids": plan["plain"]["generated_ids"],
                    "raw_logits": plan["plain"]["raw_logits"],
                    "modified_logits": plan["plain"]["modified_logits"],
                },
                "modulated": {
                    "generated_ids": plan["modulated"]["generated_ids"],
                    "raw_logits": plan["modulated"]["raw_logits"],
                    "modified_logits": plan["modulated"]["modified_logits"],
                },
                "target_token_ids": plan["target_ids"],
            }
            for plan in task_plans
        ]
        for task_name, task_plans in plans.items()
    }
    rollout_tensors = {
        task_name: [
            {
                "episode_index": episode["episode_index"],
                "conditions": {
                    name: {
                        "actions": result["actions"],
                        "action_chunks": result["action_chunks"],
                        "rewards": result["rewards"],
                    }
                    for name, result in episode["conditions"].items()
                },
            }
            for episode in task_results
        ]
        for task_name, task_results in results.items()
    }
    torch.save(planning_tensors, output_dir / "planning_logits.pt")
    torch.save(rollout_tensors, output_dir / "rollouts.pt")

    summary = summarize(plans, results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
