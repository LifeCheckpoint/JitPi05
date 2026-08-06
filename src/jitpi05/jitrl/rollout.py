"""Single-method, single-seed first-stage JitRL simulation runs."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.utils import close_envs
from lerobot.utils.io_utils import write_video
from tqdm.auto import tqdm

from jitpi05.config import (
    JITRL_ACTION_WORKSPACE_VERSION,
    JITRL_BETA,
    JITRL_EPISODES,
    JITRL_EVALUATOR_BACKEND,
    JITRL_EVALUATOR_MODEL,
    JITRL_EVALUATOR_RETRIES,
    JITRL_EVALUATOR_SCORE_SCALE,
    JITRL_EXPLORATION_RATE,
    JITRL_FREE_ACTION_SIM_THRESHOLD,
    JITRL_FREE_MAX_PLAN_RETRIES,
    JITRL_FREE_PLAN_TOKENS,
    JITRL_GAMMA,
    JITRL_HIGH_LEVEL_STEPS,
    JITRL_HISTORY_SIZE,
    JITRL_LOGIT_CALIBRATION,
    JITRL_MAX_PLAN_TOKENS,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_PLANNER_RETRIES,
    JITRL_QWEN_ID,
    JITRL_REWARD_VERSION,
    JITRL_TEMPERATURE,
    JITRL_TERMINAL_SUCCESS_BONUS,
    JITRL_TERMINATION_MODE,
    JITRL_TOP_K,
    JITRL_UCB_ALPHA,
    SIM_ACTION_STEPS,
    SIM_PI05_ID,
)
from jitpi05.jitrl.memory import JitRLMemory, discounted_returns
from jitpi05.jitrl.planner import (
    apply_jitrl_update,
    load_jitrl_planner,
    plan_and_score,
    plan_and_score_free,
    score_free_candidates,
)

# Free-candidate methods propose free-text actions from Qwen (no fixed workspace).
FREE_METHODS = ("jitrl-free", "static-free")
# Methods that populate/read the online experience memory.
MEMORY_METHODS = ("jitrl", "jitrl-free")
# Methods that apply the memory advantage logit update (beta > 0).
UPDATED_METHODS = ("jitrl", "jitrl-free")
from jitpi05.jitrl.vlm import (
    evaluate_chunk as evaluate_chunk_with_gemini,
)
from jitpi05.jitrl.vlm import load_gemini_evaluator
from jitpi05.policy import conditioned_task
from jitpi05.simulation import (
    load_policy,
    make_single_env,
    planning_images,
    policy_frame,
    reset_rollout_state,
    success_from_info,
)

# Fixed-workspace JitRL: Qwen policy logits, memory advantages, visual step rewards.


def coordinate_seed(kind: str, seed: int, episode: int, chunk: int) -> int:
    """Map an experiment coordinate to a stable torch-compatible integer seed."""

    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    coordinate = f"{kind}|{int(seed)}|{int(episode)}|{int(chunk)}".encode()
    digest = hashlib.sha256(coordinate).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def coordinate_uniform(
    seed: int, episode: int, chunk: int, kind: str = "candidate_uniform"
) -> float:
    """Draw one deterministic coordinate-local uniform from a private generator."""

    generator = torch.Generator(device="cpu").manual_seed(
        coordinate_seed(kind, seed, episode, chunk)
    )
    return float(torch.rand((), generator=generator, dtype=torch.float32).item())


def coordinate_exploration_uniforms(
    seed: int, episode: int, chunk: int, count: int
) -> list[float]:
    return [
        coordinate_uniform(seed, episode, chunk, kind=f"ucb_uniform_{index}")
        for index in range(count)
    ]


def coordinate_flow_noise(
    policy,
    seed: int,
    episode: int,
    chunk: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Rebuild one paired flow-noise tensor without materializing a whole rollout."""

    generator = torch.Generator(device="cpu").manual_seed(
        coordinate_seed("flow_noise", seed, episode, chunk)
    )
    noise = torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return noise.to(device)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def _static_value_estimate(action_keys: Sequence[str]) -> dict:
    return {
        "V": 0.0,
        "Q": [0.0 for _ in action_keys],
        "A": [0.0 for _ in action_keys],
        "normalized_A": [0.0 for _ in action_keys],
        "coverage": {},
        "neighbor_count": 0,
        "neighbor_action_coverage": {},
        "candidates": [
            {
                "action_key": action_key,
                "Q": 0.0,
                "advantage": 0.0,
                "normalized_advantage": 0.0,
                "neighbor_count": 0,
                "seen": False,
                "exploration_uniform": None,
                "exploration_applied": False,
                "exploration_branch": "static",
                "ucb_bonus": 0.0,
            }
            for action_key in action_keys
        ],
    }


def _trace_value_estimate(estimate: dict) -> dict:
    """Add compact vectors while retaining the complete memory API estimate."""

    result = {
        **estimate,
        "Q": [float(candidate["Q"]) for candidate in estimate["candidates"]],
        "A": [float(candidate["advantage"]) for candidate in estimate["candidates"]],
        "normalized_A": [
            float(candidate["normalized_advantage"])
            for candidate in estimate["candidates"]
        ],
        "coverage": dict(estimate["neighbor_action_coverage"]),
    }
    return result


def _episode_tensors(
    action_chunks: Sequence[torch.Tensor],
    actions: Sequence[torch.Tensor],
    rewards: Sequence[float],
    policy,
) -> dict[str, torch.Tensor]:
    action_dim = int(policy.config.output_features["action"].shape[0])
    stacked_chunks = (
        torch.stack(list(action_chunks))
        if action_chunks
        else torch.empty((0, policy.config.chunk_size, action_dim), dtype=torch.float32)
    )
    stacked_actions = (
        torch.stack(list(actions))
        if actions
        else torch.empty((0, action_dim), dtype=torch.float32)
    )
    return {
        "action_chunks": stacked_chunks,
        "actions": stacked_actions,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
    }


def _save_episode_tensors(
    tensor_dir: Path, episode_index: int, tensors: dict[str, torch.Tensor]
) -> dict[str, str]:
    tensor_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: tensor_dir / f"episode_{episode_index}_{name}.pt" for name in tensors
    }
    for name, path in paths.items():
        torch.save(tensors[name], path)
    return {f"{name}_path": str(path) for name, path in paths.items()}


def _memory_snapshot(memory: JitRLMemory | None) -> list[dict]:
    return memory.snapshot() if memory is not None else []


def _trace_neighbors(neighbors: Sequence[dict]) -> list[dict]:
    return [
        {
            "id": int(neighbor["id"]),
            "action_key": neighbor["action_key"],
            "action_text": neighbor["action_text"],
            "return": float(neighbor["return"]),
            "episode": int(neighbor["episode_index"]),
            "chunk": int(neighbor["chunk_index"]),
            "similarity": float(neighbor["similarity"]),
        }
        for neighbor in neighbors
    ]


def _low_level_chunks_per_high_level_plan() -> int:
    if JITRL_HIGH_LEVEL_STEPS <= 0:
        raise ValueError("JITRL_HIGH_LEVEL_STEPS must be positive")
    if JITRL_HIGH_LEVEL_STEPS % SIM_ACTION_STEPS != 0:
        raise ValueError("JITRL_HIGH_LEVEL_STEPS must be divisible by SIM_ACTION_STEPS")
    return JITRL_HIGH_LEVEL_STEPS // SIM_ACTION_STEPS


def _plan_and_score_with_retry(
    planner_model,
    planner_processor,
    images,
    overall_task: str,
    recent_subtasks: Sequence[str],
    *,
    method: str,
    seed: int,
    episode_index: int,
    high_level_step_index: int,
) -> tuple[dict, int]:
    """Retry malformed Qwen workspace bindings without restarting the episode."""

    max_retries = (
        JITRL_FREE_MAX_PLAN_RETRIES
        if method in FREE_METHODS
        else JITRL_PLANNER_RETRIES
    )
    failures = []
    for attempt in range(1, max_retries + 1):
        try:
            if method in FREE_METHODS:
                planning = plan_and_score_free(
                    planner_model,
                    planner_processor,
                    images,
                    overall_task,
                    recent_subtasks=recent_subtasks,
                    max_new_tokens=JITRL_FREE_PLAN_TOKENS,
                    attempt=attempt,
                )
            else:
                planning = plan_and_score(
                    planner_model,
                    planner_processor,
                    images,
                    overall_task,
                    recent_subtasks=recent_subtasks,
                    max_new_tokens=JITRL_MAX_PLAN_TOKENS,
                    attempt=attempt,
                )
            return planning, attempt
        except Exception as error:
            failures.append(f"attempt {attempt}: {error}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"planner failed after {max_retries} attempts:\n"
                    + "\n".join(failures)
                ) from error
            tqdm.write(
                f"[planner-retry] method={method} seed={seed} "
                f"episode={episode_index + 1} high={high_level_step_index} "
                f"attempt={attempt}/{max_retries} error={error}"
            )

    raise RuntimeError("planner retry loop terminated unexpectedly")


def _evaluate_episode_chunks(
    evaluator,
    episode_result: dict,
    evaluator_images: Sequence[tuple[list, list]],
    *,
    method: str,
    seed: int,
    episode_index: int,
) -> tuple[list[float], list[float]]:
    """Assign positive-only visual step rewards and discounted returns."""

    chunks = episode_result["chunks"]
    if len(chunks) != len(evaluator_images):
        raise ValueError("chunk traces and evaluator image pairs must align")
    scores: list[int] = []
    for chunk_index, (chunk, (before_images, after_images)) in enumerate(
        zip(chunks, evaluator_images)
    ):
        next_state_summary = (
            chunks[chunk_index + 1]["state_summary"]
            if chunk_index + 1 < len(chunks)
            else (
                "terminal task success"
                if episode_result["success"]
                else "terminal or time-limit state; task not completed"
            )
        )
        evaluation = None
        for attempt in range(1, JITRL_EVALUATOR_RETRIES + 1):
            try:
                evaluation = evaluate_chunk_with_gemini(
                    evaluator,
                    [*before_images, *after_images],
                    episode_result["task"],
                    chunk["state_summary"],
                    chunk["selected_candidate"]["text"],
                    next_state_summary,
                    episode_result["success"],
                    chunk_index,
                    len(chunks),
                    attempt=attempt,
                )
                break
            except Exception as error:
                if attempt == JITRL_EVALUATOR_RETRIES:
                    raise RuntimeError(
                        "chunk evaluator failed after "
                        f"{JITRL_EVALUATOR_RETRIES} attempts: {error}"
                    ) from error
                tqdm.write(
                    f"[evaluator-retry] method={method} seed={seed} "
                    f"episode={episode_index + 1} chunk={chunk_index} "
                    f"attempt={attempt}/{JITRL_EVALUATOR_RETRIES} error={error}"
                )
        if evaluation is None:
            raise RuntimeError("chunk evaluator returned no result")
        evaluation["attempts"] = attempt
        evaluation["next_state_summary"] = next_state_summary
        chunk["evaluator"] = evaluation
        score = int(evaluation["score"])
        if not 0 <= score <= 3:
            raise ValueError("positive-only evaluator score must be in [0, 3]")
        scores.append(score)

    step_rewards = [score * JITRL_EVALUATOR_SCORE_SCALE for score in scores]
    if episode_result["success"] and step_rewards:
        step_rewards[-1] += JITRL_TERMINAL_SUCCESS_BONUS
    returns = discounted_returns(step_rewards, gamma=JITRL_GAMMA)
    for chunk, score, reward, return_value in zip(
        chunks, scores, step_rewards, returns
    ):
        chunk["evaluator_score"] = score
        chunk["step_reward"] = float(reward)
        chunk["return"] = float(return_value)
    episode_result["evaluator_scores"] = scores
    episode_result["step_rewards"] = [float(value) for value in step_rewards]
    episode_result["discounted_returns"] = [float(value) for value in returns]
    return step_rewards, returns


def _run_episode(
    *,
    task_spec: dict[str, Any],
    method: str,
    seed: int,
    episode_index: int,
    run_dir: Path,
    memory: JitRLMemory | None,
    planner_model,
    planner_processor,
    policy,
    preprocessor,
    postprocessor,
    step_progress=None,
) -> tuple[dict, list[dict], dict[str, torch.Tensor], list[tuple[list, list]]]:
    envs, env, env_preprocessor, env_postprocessor = make_single_env(
        task_spec, episode_index
    )
    episode_seed = int(seed) + episode_index

    video_path = run_dir / "videos" / f"episode_{episode_index}.mp4"
    frames: list[np.ndarray] = []
    action_chunks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[float] = []
    chunks: list[dict] = []
    evaluator_images: list[tuple[list, list]] = []
    episode_records: list[dict] = []
    selected_subtasks: list[str] = []
    active_condition = ""
    active_chunk_trace: dict | None = None
    active_before_images: list | None = None
    overall_task = ""
    success = False
    episode_done = False
    termination_reason: str | None = None
    low_level_chunks_per_plan = _low_level_chunks_per_high_level_plan()

    try:
        reset_rollout_state(policy, preprocessor, postprocessor, episode_seed)
        observation, _ = env.reset(seed=[episode_seed])
        overall_task = list(env.call("task_description"))[0]
        max_steps = int(env.call("_max_episode_steps")[0])
        frames.append(env.render()[0])

        while len(actions) < max_steps and not success and not episode_done:
            low_level_chunk_index = len(action_chunks)
            high_level_step_index = low_level_chunk_index // low_level_chunks_per_plan
            high_level_replan = low_level_chunk_index % low_level_chunks_per_plan == 0
            progress = {
                "task": task_spec["name"],
                "method": method,
                "seed": int(seed),
                "episode_index": episode_index,
                "chunk_index": low_level_chunk_index,
                "low_level_chunk_index": low_level_chunk_index,
                "high_level_step_index": high_level_step_index,
                "high_level_replan": high_level_replan,
                "steps": len(actions),
                "max_steps": max_steps,
            }
            _atomic_write_json(run_dir / "progress.json", progress)
            if step_progress is not None:
                step_progress.set_postfix_str(
                    f"low={low_level_chunk_index} high={high_level_step_index} "
                    f"planner={'qwen-workspace' if high_level_replan else 'cached'}",
                    refresh=True,
                )
            frame = policy_frame(observation, overall_task, env_preprocessor)

            if high_level_replan:
                images = planning_images(frame)
                active_before_images = [image.copy() for image in images]
                planning, planner_attempts = _plan_and_score_with_retry(
                    planner_model,
                    planner_processor,
                    images,
                    overall_task,
                    selected_subtasks[-JITRL_HISTORY_SIZE:],
                    method=method,
                    seed=seed,
                    episode_index=episode_index,
                    high_level_step_index=high_level_step_index,
                )
                if method in FREE_METHODS and planning.get("stop_only_masked"):
                    # Qwen proposed only "stop" (a visual completion judgement)
                    # which is disabled under environment-only termination. On the
                    # first decision there is no previous action to reuse, so the
                    # episode fails as a stop termination; otherwise repeat the
                    # previous high-level action so the episode keeps running.
                    if not selected_subtasks:
                        termination_reason = "qwen_stop"
                        evaluator_images.append(
                            (
                                active_before_images,
                                [image.copy() for image in images],
                            )
                        )
                        active_before_images = None
                        break
                    repeated = dict(chunks[-1]["selected_candidate"])
                    repeated["label"] = "1"
                    planning = score_free_candidates(
                        planner_model,
                        planner_processor,
                        images,
                        overall_task,
                        {
                            **planning,
                            "candidates": [repeated],
                            "qwen_candidates": [dict(repeated)],
                        },
                        selected_subtasks[-JITRL_HISTORY_SIZE:],
                    )
                    planning["stop_only_masked"] = True
                    planning["repeated_previous_action"] = True
                candidates = planning["candidates"]
                raw_neighbors = (
                    memory.retrieve(planning["state_summary"])
                    if method in MEMORY_METHODS and memory is not None
                    else []
                )
                action_keys = [candidate["semantic_key"] for candidate in candidates]
                retrieved_neighbors = _trace_neighbors(raw_neighbors)

                if method in MEMORY_METHODS:
                    if memory is None:
                        raise RuntimeError(f"{method} method requires a task-local memory")
                    exploration_uniforms = coordinate_exploration_uniforms(
                        seed,
                        episode_index,
                        high_level_step_index,
                        len(candidates),
                    )
                    if method == "jitrl":
                        value_estimate = memory.estimate_candidate_values(
                            planning["state_summary"],
                            action_keys,
                            neighbors=raw_neighbors,
                            exploration_uniforms=exploration_uniforms,
                            exploration_rate=JITRL_EXPLORATION_RATE,
                            ucb_alpha=JITRL_UCB_ALPHA,
                        )
                    else:
                        value_estimate = memory.estimate_candidate_values_free(
                            planning["state_summary"],
                            [candidate["text"] for candidate in candidates],
                            neighbors=raw_neighbors,
                            exploration_uniforms=exploration_uniforms,
                            exploration_rate=JITRL_EXPLORATION_RATE,
                            ucb_alpha=JITRL_UCB_ALPHA,
                            action_sim_threshold=JITRL_FREE_ACTION_SIM_THRESHOLD,
                        )
                else:
                    value_estimate = _static_value_estimate(action_keys)
                value_estimate = _trace_value_estimate(value_estimate)

                normalized_advantages = [
                    candidate_value["normalized_advantage"]
                    for candidate_value in value_estimate["candidates"]
                ]
                uniform = coordinate_uniform(seed, episode_index, high_level_step_index)
                update = apply_jitrl_update(
                    planning["base_logits"],
                    normalized_advantages,
                    beta=JITRL_BETA if method in UPDATED_METHODS else 0.0,
                    temperature=JITRL_TEMPERATURE,
                    uniform=uniform,
                )
                selected_index = (
                    update["updated_choice_index"]
                    if method in UPDATED_METHODS
                    else update["base_choice_index"]
                )
                selected_candidate = candidates[selected_index]
                active_condition = conditioned_task(
                    overall_task, selected_candidate["text"]
                )
                action_start_step = len(actions)

                episode_records.append(
                    {
                        "state": planning["state_summary"],
                        "action_key": selected_candidate["semantic_key"],
                        "action_text": selected_candidate["text"],
                        "chunk_index": high_level_step_index,
                    }
                )
                selected_subtasks.append(selected_candidate["text"])
                active_chunk_trace = {
                    "chunk_index": high_level_step_index,
                    "planner_attempts": planner_attempts,
                    "binding_prompt": planning["binding_prompt"],
                    "binding_raw_output": planning["binding_raw_output"],
                    "binding_assistant_prefill": planning[
                        "binding_assistant_prefill"
                    ],
                    "binding_generated_tokens": planning["binding_generated_tokens"],
                    "binding_generation_reached_limit": planning[
                        "binding_generation_reached_limit"
                    ],
                    "binding": planning["binding"],
                    "selection_prompt": planning["selection_prompt"],
                    "state_summary": planning["state_summary"],
                    "workspace": planning["workspace"],
                    "qwen_candidates": planning["qwen_candidates"],
                    "candidates": candidates,
                    "termination_mode": planning["termination_mode"],
                    "candidate_token_ids": planning["candidate_token_ids"],
                    "value_estimate": value_estimate,
                    "retrieved_neighbors": retrieved_neighbors,
                    **update,
                    "selected_candidate_index": selected_index,
                    "selected_candidate": selected_candidate,
                    "condition": active_condition,
                    "return": None,
                    "action_start_step": action_start_step,
                    "action_end_step": action_start_step,
                    "low_level_chunks": [],
                }
                chunks.append(active_chunk_trace)
                if selected_candidate["terminates_rollout"]:
                    if (
                        JITRL_TERMINATION_MODE == "environment_only_no_stop_v1"
                        and method not in FREE_METHODS
                    ):
                        raise RuntimeError(
                            "stop reached rollout selection under the no-stop diagnostic"
                        )
                    termination_reason = "qwen_stop"
                    evaluator_images.append(
                        (
                            active_before_images,
                            [image.copy() for image in images],
                        )
                    )
                    active_before_images = None
                    break
            elif active_chunk_trace is None:
                raise RuntimeError("low-level replan has no cached high-level subtask")

            low_level_action_start_step = len(actions)
            conditioned_frame = dict(frame)
            conditioned_frame["task"] = [active_condition]
            batch = preprocessor(conditioned_frame)
            noise = coordinate_flow_noise(
                policy, seed, episode_index, low_level_chunk_index, device="cuda"
            )
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(batch, noise=noise)
            del noise
            action_chunk = postprocessor(normalized_chunk).squeeze(0)
            if action_chunk.shape[0] != policy.config.chunk_size:
                raise ValueError(
                    "predict_action_chunk must return the complete configured action chunk"
                )
            action_chunks.append(action_chunk.detach().cpu())
            del batch, normalized_chunk

            for action in action_chunk[:SIM_ACTION_STEPS]:
                action_transition = env_postprocessor({"action": action.unsqueeze(0)})
                env_action = action_transition["action"].detach().cpu()
                observation, reward, terminated, truncated, info = env.step(
                    env_action.numpy()
                )
                actions.append(env_action.squeeze(0).clone())
                rewards.append(float(reward[0]))
                frames.append(env.render()[0])
                if step_progress is not None:
                    step_progress.update(1)

                episode_done = bool(terminated[0] or truncated[0])
                success = success or success_from_info(info)
                if success:
                    termination_reason = "environment_success"
                elif episode_done:
                    termination_reason = "environment_done"
                elif len(actions) >= max_steps:
                    termination_reason = "max_steps"
                if episode_done or success or len(actions) >= max_steps:
                    break

            active_chunk_trace["action_end_step"] = len(actions)
            active_chunk_trace["low_level_chunks"].append(
                {
                    "chunk_index": low_level_chunk_index,
                    "action_start_step": low_level_action_start_step,
                    "action_end_step": len(actions),
                }
            )
            if (
                len(active_chunk_trace["low_level_chunks"]) == low_level_chunks_per_plan
                or episode_done
                or success
                or len(actions) >= max_steps
            ):
                if active_before_images is None:
                    raise RuntimeError(
                        "completed high-level chunk has no before images"
                    )
                after_frame = policy_frame(observation, overall_task, env_preprocessor)
                evaluator_images.append(
                    (
                        active_before_images,
                        [image.copy() for image in planning_images(after_frame)],
                    )
                )
                active_before_images = None
    finally:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if frames:
                write_video(video_path, frames, env.metadata.get("render_fps", 80))
        finally:
            close_envs(envs)

    _atomic_write_json(
        run_dir / "progress.json",
        {
            "task": task_spec["name"],
            "method": method,
            "seed": int(seed),
            "episode_index": episode_index,
            "chunk_index": len(action_chunks),
            "low_level_chunk_index": len(action_chunks),
            "high_level_step_index": len(chunks),
            "steps": len(actions),
            "max_steps": max_steps,
            "episode_complete": True,
        },
    )
    if termination_reason is None:
        termination_reason = "max_steps"
    tensors = _episode_tensors(action_chunks, actions, rewards, policy)
    episode_result = {
        "episode_index": episode_index,
        "init_state_id": episode_index,
        "task": overall_task,
        "success": bool(success),
        "termination_mode": JITRL_TERMINATION_MODE,
        "termination_reason": termination_reason,
        "steps": len(actions),
        "sum_reward": float(sum(rewards)),
        "max_reward": float(max(rewards, default=0.0)),
        "chunk_count": len(chunks),
        "high_level_step_count": len(chunks),
        "low_level_chunk_count": len(action_chunks),
        "video_path": str(video_path),
        "chunks": chunks,
    }
    return episode_result, episode_records, tensors, evaluator_images


def summarize_run(
    episodes: Sequence[dict],
    memory_size: int,
    *,
    task_spec: dict[str, Any],
    method: str,
    seed: int,
    run_dir: Path,
) -> dict:
    """Build the JSON-safe run-level record consumed by the later CLI/statistics layer."""

    episode_rows = [
        {
            "episode_index": episode["episode_index"],
            "init_state_id": episode["init_state_id"],
            "success": episode["success"],
            "steps": episode["steps"],
        }
        for episode in episodes
    ]
    task_descriptions = list(dict.fromkeys(episode["task"] for episode in episodes))
    models = {
        "high_level_policy": JITRL_QWEN_ID,
        "policy": SIM_PI05_ID,
        "evaluator": JITRL_EVALUATOR_MODEL,
        "planning_mode": (
            "free-candidates" if method in FREE_METHODS else "fixed-workspace"
        ),
    }
    return {
        "model": models,
        "models": models,
        "task": {**task_spec, "descriptions": task_descriptions},
        "method": method,
        "seed": int(seed),
        "episodes": len(episodes),
        "high_level_steps_before_replan": JITRL_HIGH_LEVEL_STEPS,
        "action_steps_before_replan": SIM_ACTION_STEPS,
        "planner_max_attempts": JITRL_PLANNER_RETRIES,
        "evaluator_backend": JITRL_EVALUATOR_BACKEND,
        "evaluator_max_attempts": JITRL_EVALUATOR_RETRIES,
        "gamma": JITRL_GAMMA,
        "beta": JITRL_BETA,
        "temperature": JITRL_TEMPERATURE,
        "exploration_rate": JITRL_EXPLORATION_RATE,
        "ucb_alpha": JITRL_UCB_ALPHA,
        "action_workspace_version": JITRL_ACTION_WORKSPACE_VERSION,
        "termination_mode": JITRL_TERMINATION_MODE,
        "logit_calibration": JITRL_LOGIT_CALIBRATION,
        "reward_version": JITRL_REWARD_VERSION,
        "episode_results": episode_rows,
        "successes": [row["success"] for row in episode_rows],
        "steps": [row["steps"] for row in episode_rows],
        "success_rate": (
            sum(row["success"] for row in episode_rows) / len(episode_rows)
            if episode_rows
            else 0.0
        ),
        "memory_size": int(memory_size),
        "run_dir": str(run_dir),
        "episodes_path": str(run_dir / "episodes.json"),
        "memory_path": str(run_dir / "memory.json"),
    }


def load_experiment_models(*, need_evaluator: bool) -> dict:
    """Load planner, policy and (optionally) evaluator once for reuse across runs.

    Models stay resident on the GPU across task/method/seed runs so the CLI does not
    repeatedly unload and reload them.
    """

    planner_model, planner_processor = load_jitrl_planner()
    policy, preprocessor, postprocessor = load_policy()
    evaluator = load_gemini_evaluator() if need_evaluator else None
    return {
        "planner_model": planner_model,
        "planner_processor": planner_processor,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "evaluator": evaluator,
    }


def release_experiment_models(models: dict) -> None:
    """Release the shared experiment models and free CUDA memory."""

    for key in (
        "planner_model",
        "planner_processor",
        "policy",
        "preprocessor",
        "postprocessor",
        "evaluator",
    ):
        models[key] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_jitrl_experiment(
    task_spec: dict[str, Any],
    method: str,
    seed: int,
    episodes: int = JITRL_EPISODES,
    output_dir: Path = JITRL_OUTPUT_DIR,
    models: dict | None = None,
) -> dict:
    """Run one task/method/seed pairing from an independent empty memory.

    When ``models`` (from :func:`load_experiment_models`) is provided, the resident
    model objects are reused and not unloaded when this run finishes; otherwise the
    models are loaded here and released in the ``finally`` block.
    """

    if method not in JITRL_METHODS:
        raise ValueError(f"method must be one of {JITRL_METHODS}; got {method!r}")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    task_name = str(task_spec.get("name", "")).strip()
    if not task_name:
        raise ValueError("task_spec must contain a non-empty name")
    for field in ("suite", "task_id", "max_steps"):
        if field not in task_spec:
            raise ValueError(f"task_spec must contain {field!r}")

    run_dir = Path(output_dir) / task_name / method / f"seed_{int(seed)}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    video_dir = run_dir / "videos"
    tensor_dir = run_dir / "tensors"
    snapshot_dir = run_dir / "memory_snapshots"
    for directory in (run_dir, video_dir, tensor_dir, snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict] = []
    memory = (
        JitRLMemory(top_k=JITRL_TOP_K) if method in MEMORY_METHODS else None
    )
    _atomic_write_json(run_dir / "episodes.json", episode_results)
    _atomic_write_json(run_dir / "memory.json", _memory_snapshot(memory))

    planner_model = None
    planner_processor = None
    evaluator = None
    policy = None
    preprocessor = None
    postprocessor = None
    current_episode = None
    owns_models = models is None
    episode_progress = tqdm(
        total=episodes,
        desc=f"{task_name} {method} seed={seed} episodes",
        unit="ep",
        dynamic_ncols=True,
        position=1,
        leave=True,
    )
    try:
        if models is not None:
            planner_model = models["planner_model"]
            planner_processor = models["planner_processor"]
            policy = models["policy"]
            preprocessor = models["preprocessor"]
            postprocessor = models["postprocessor"]
            evaluator = models.get("evaluator")
            if method in MEMORY_METHODS and evaluator is None:
                raise RuntimeError(f"{method} method requires a Gemini evaluator")
        else:
            tqdm.write(
                f"[load] task={task_name} method={method} seed={seed}: "
                "loading Qwen planner"
            )
            planner_model, planner_processor = load_jitrl_planner()
            if method in MEMORY_METHODS:
                tqdm.write(
                    f"[load] task={task_name} method={method} seed={seed}: "
                    f"configuring {JITRL_EVALUATOR_MODEL} evaluator"
                )
                evaluator = load_gemini_evaluator()
            tqdm.write(
                f"[load] task={task_name} method={method} seed={seed}: "
                "loading π₀.₅ policy"
            )
            policy, preprocessor, postprocessor = load_policy()
        tqdm.write(
            f"[ready] task={task_name} method={method} seed={seed}: models loaded"
        )

        for episode_index in range(episodes):
            current_episode = episode_index
            max_steps = int(task_spec["max_steps"])
            step_progress = tqdm(
                total=max_steps,
                desc=(
                    f"{task_name} {method} seed={seed} episode "
                    f"{episode_index + 1}/{episodes} steps"
                ),
                unit="step",
                dynamic_ncols=True,
                position=2,
                leave=False,
            )
            try:
                (
                    episode_result,
                    episode_records,
                    tensors,
                    evaluator_images,
                ) = _run_episode(
                    task_spec=task_spec,
                    method=method,
                    seed=int(seed),
                    episode_index=episode_index,
                    run_dir=run_dir,
                    memory=memory,
                    planner_model=planner_model,
                    planner_processor=planner_processor,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    step_progress=step_progress,
                )
            finally:
                step_progress.close()
            if method in MEMORY_METHODS:
                if evaluator is None:
                    raise RuntimeError(f"{method} method lost its Gemini evaluator")
                _step_rewards, returns = _evaluate_episode_chunks(
                    evaluator,
                    episode_result,
                    evaluator_images,
                    method=method,
                    seed=int(seed),
                    episode_index=episode_index,
                )
            else:
                returns = discounted_returns(
                    [0.0] * len(episode_records), gamma=JITRL_GAMMA
                )
                for chunk, return_value in zip(episode_result["chunks"], returns):
                    chunk["return"] = float(return_value)

            if method in MEMORY_METHODS:
                if memory is None:
                    raise RuntimeError(f"{method} method lost its task-local memory")
                memory.append_episode(episode_records, returns, episode_index)

            tensor_paths = _save_episode_tensors(tensor_dir, episode_index, tensors)
            episode_result.update(tensor_paths)
            episode_results.append(episode_result)

            snapshot = _memory_snapshot(memory)
            _atomic_write_json(run_dir / "episodes.json", episode_results)
            _atomic_write_json(run_dir / "memory.json", snapshot)
            _atomic_write_json(snapshot_dir / f"episode_{episode_index}.json", snapshot)

            completed = len(episode_results)
            successes = sum(bool(row["success"]) for row in episode_results)
            recent_rows = episode_results[-10:]
            recent_successes = sum(bool(row["success"]) for row in recent_rows)
            success_rate = successes / completed
            recent_rate = recent_successes / len(recent_rows)
            memory_size = len(snapshot)
            episode_progress.update(1)
            episode_progress.set_postfix(
                success=f"{successes}/{completed}",
                rate=f"{success_rate:.1%}",
                last10=f"{recent_rate:.1%}",
                memory=memory_size,
                refresh=True,
            )
            tqdm.write(
                f"[episode] task={task_name} method={method} seed={seed} "
                f"episode={episode_index + 1}/{episodes} "
                f"result={'SUCCESS' if episode_result['success'] else 'FAIL'} "
                f"steps={episode_result['steps']} "
                f"high={episode_result['high_level_step_count']} "
                f"low={episode_result['low_level_chunk_count']} "
                f"cumulative={successes}/{completed} ({success_rate:.1%}) "
                f"last10={recent_successes}/{len(recent_rows)} ({recent_rate:.1%}) "
                f"memory={memory_size}"
            )
            current_episode = None
    except Exception as error:
        _atomic_write_json(
            run_dir / "failure.json",
            {
                "task": task_name,
                "method": method,
                "seed": int(seed),
                "episode_index": current_episode,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        episode_progress.close()
        if owns_models:
            planner_model = None
            planner_processor = None
            evaluator = None
            policy = None
            preprocessor = None
            postprocessor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    memory_size = len(memory) if memory is not None else 0
    summary = summarize_run(
        episode_results,
        memory_size,
        task_spec=task_spec,
        method=method,
        seed=int(seed),
        run_dir=run_dir,
    )
    _atomic_write_json(run_dir / "run_summary.json", summary)

    # Successful runs persist everything before the potentially expensive cleanup.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
