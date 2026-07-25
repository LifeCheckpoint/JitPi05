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

from jitrl_memory import JitRLMemory, discounted_chunk_returns
from jitrl_planner import apply_jitrl_update, load_jitrl_planner, propose_and_score
from pi05_pipeline import conditioned_task
from settings import (
    JITRL_BETA,
    JITRL_CANDIDATES,
    JITRL_EPISODES,
    JITRL_GAMMA,
    JITRL_HIGH_LEVEL_STEPS,
    JITRL_HISTORY_SIZE,
    JITRL_MAX_PLAN_TOKENS,
    JITRL_METHODS,
    JITRL_OUTPUT_DIR,
    JITRL_QWEN_ID,
    JITRL_TASK,
    JITRL_TEMPERATURE,
    JITRL_TOP_K,
    SIM_ACTION_STEPS,
    SIM_PI05_ID,
)
from sim_eval import (
    load_policy,
    make_single_env,
    planning_images,
    policy_frame,
    reset_rollout_state,
    success_from_info,
)

# Stage 1 only: visual-planner candidates; no memory-only actions/UCB/evaluator.


def coordinate_seed(kind: str, seed: int, episode: int, chunk: int) -> int:
    """Map an experiment coordinate to a stable torch-compatible integer seed."""

    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    coordinate = f"{kind}|{int(seed)}|{int(episode)}|{int(chunk)}".encode("utf-8")
    digest = hashlib.sha256(coordinate).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def coordinate_uniform(seed: int, episode: int, chunk: int) -> float:
    """Draw the paired candidate-selection uniform from a private CPU generator."""

    generator = torch.Generator(device="cpu").manual_seed(
        coordinate_seed("candidate_uniform", seed, episode, chunk)
    )
    return float(torch.rand((), generator=generator, dtype=torch.float32).item())


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
        raise ValueError(
            "JITRL_HIGH_LEVEL_STEPS must be divisible by SIM_ACTION_STEPS"
        )
    return JITRL_HIGH_LEVEL_STEPS // SIM_ACTION_STEPS


def _run_episode(
    *,
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
) -> tuple[dict, list[dict], dict[str, torch.Tensor]]:
    envs, env, env_preprocessor, env_postprocessor = make_single_env(
        JITRL_TASK, episode_index
    )
    episode_seed = int(seed) + episode_index

    video_path = run_dir / "videos" / f"episode_{episode_index}.mp4"
    frames: list[np.ndarray] = []
    action_chunks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[float] = []
    chunks: list[dict] = []
    episode_records: list[dict] = []
    selected_subtasks: list[str] = []
    active_condition = ""
    active_chunk_trace: dict | None = None
    overall_task = ""
    success = False
    episode_done = False
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
                    f"qwen={'yes' if high_level_replan else 'cached'}",
                    refresh=True,
                )
            frame = policy_frame(observation, overall_task, env_preprocessor)

            if high_level_replan:
                images = planning_images(frame)
                planning = propose_and_score(
                    planner_model,
                    planner_processor,
                    images,
                    overall_task,
                    recent_subtasks=selected_subtasks[-JITRL_HISTORY_SIZE:],
                    max_new_tokens=JITRL_MAX_PLAN_TOKENS,
                )
                candidates = planning["candidates"]
                if len(candidates) != JITRL_CANDIDATES:
                    raise ValueError(
                        f"planner returned {len(candidates)} candidates; expected {JITRL_CANDIDATES}"
                    )
                action_keys = [
                    candidate["semantic_key"] for candidate in candidates
                ]

                if method == "jitrl":
                    if memory is None:
                        raise RuntimeError("jitrl method requires a task-local memory")
                    retrieved_neighbors = _trace_neighbors(
                        memory.retrieve(planning["state_summary"])
                    )
                    value_estimate = memory.estimate_candidate_values(
                        planning["state_summary"], action_keys
                    )
                else:
                    retrieved_neighbors = []
                    value_estimate = _static_value_estimate(action_keys)
                value_estimate = _trace_value_estimate(value_estimate)

                normalized_advantages = [
                    candidate_value["normalized_advantage"]
                    for candidate_value in value_estimate["candidates"]
                ]
                uniform = coordinate_uniform(
                    seed, episode_index, high_level_step_index
                )
                update = apply_jitrl_update(
                    planning["base_logits"],
                    normalized_advantages,
                    beta=JITRL_BETA,
                    temperature=JITRL_TEMPERATURE,
                    uniform=uniform,
                )
                selected_index = (
                    update["updated_choice_index"]
                    if method == "jitrl"
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
                    "proposal_prompt": planning["prompt"],
                    "selection_prompt": planning["selection_prompt"],
                    "state_summary": planning["state_summary"],
                    "candidates": candidates,
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
    tensors = _episode_tensors(action_chunks, actions, rewards, policy)
    episode_result = {
        "episode_index": episode_index,
        "init_state_id": episode_index,
        "task": overall_task,
        "success": bool(success),
        "steps": len(actions),
        "sum_reward": float(sum(rewards)),
        "max_reward": float(max(rewards, default=0.0)),
        "chunk_count": len(chunks),
        "high_level_step_count": len(chunks),
        "low_level_chunk_count": len(action_chunks),
        "video_path": str(video_path),
        "chunks": chunks,
    }
    return episode_result, episode_records, tensors


def summarize_run(
    episodes: Sequence[dict],
    memory_size: int,
    *,
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
    models = {"planner": JITRL_QWEN_ID, "policy": SIM_PI05_ID}
    return {
        "model": models,
        "models": models,
        "task": {**JITRL_TASK, "descriptions": task_descriptions},
        "method": method,
        "seed": int(seed),
        "episodes": len(episodes),
        "high_level_steps_before_replan": JITRL_HIGH_LEVEL_STEPS,
        "action_steps_before_replan": SIM_ACTION_STEPS,
        "gamma": JITRL_GAMMA,
        "beta": JITRL_BETA,
        "temperature": JITRL_TEMPERATURE,
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


def run_jitrl_experiment(
    method: str,
    seed: int,
    episodes: int = JITRL_EPISODES,
    output_dir: Path = JITRL_OUTPUT_DIR,
) -> dict:
    """Run one method/seed pairing from empty task-local memory and persist artifacts."""

    if method not in JITRL_METHODS:
        raise ValueError(f"method must be one of {JITRL_METHODS}; got {method!r}")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")

    run_dir = Path(output_dir) / method / f"seed_{int(seed)}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    video_dir = run_dir / "videos"
    tensor_dir = run_dir / "tensors"
    snapshot_dir = run_dir / "memory_snapshots"
    for directory in (run_dir, video_dir, tensor_dir, snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict] = []
    memory = JitRLMemory(top_k=JITRL_TOP_K) if method == "jitrl" else None
    _atomic_write_json(run_dir / "episodes.json", episode_results)
    _atomic_write_json(run_dir / "memory.json", _memory_snapshot(memory))

    planner_model = None
    planner_processor = None
    policy = None
    preprocessor = None
    postprocessor = None
    current_episode = None
    episode_progress = tqdm(
        total=episodes,
        desc=f"{method} seed={seed} episodes",
        unit="ep",
        dynamic_ncols=True,
        position=1,
        leave=True,
    )
    try:
        tqdm.write(f"[load] method={method} seed={seed}: loading Qwen planner")
        planner_model, planner_processor = load_jitrl_planner()
        tqdm.write(f"[load] method={method} seed={seed}: loading π₀.₅ policy")
        policy, preprocessor, postprocessor = load_policy()
        tqdm.write(f"[ready] method={method} seed={seed}: models loaded")

        for episode_index in range(episodes):
            current_episode = episode_index
            max_steps = int(JITRL_TASK["max_steps"])
            step_progress = tqdm(
                total=max_steps,
                desc=(
                    f"{method} seed={seed} episode "
                    f"{episode_index + 1}/{episodes} steps"
                ),
                unit="step",
                dynamic_ncols=True,
                position=2,
                leave=False,
            )
            try:
                episode_result, episode_records, tensors = _run_episode(
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
            returns = discounted_chunk_returns(
                len(episode_records), episode_result["success"], gamma=JITRL_GAMMA
            )
            for chunk, return_value in zip(episode_result["chunks"], returns):
                chunk["return"] = float(return_value)

            if method == "jitrl":
                if memory is None:
                    raise RuntimeError("jitrl method lost its task-local memory")
                memory.append_episode(episode_records, returns, episode_index)

            tensor_paths = _save_episode_tensors(tensor_dir, episode_index, tensors)
            episode_result.update(tensor_paths)
            episode_results.append(episode_result)

            snapshot = _memory_snapshot(memory)
            _atomic_write_json(run_dir / "episodes.json", episode_results)
            _atomic_write_json(run_dir / "memory.json", snapshot)
            _atomic_write_json(
                snapshot_dir / f"episode_{episode_index}.json", snapshot
            )

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
                f"[episode] method={method} seed={seed} "
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
        planner_model = None
        planner_processor = None
        policy = None
        preprocessor = None
        postprocessor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    memory_size = len(memory) if memory is not None else 0
    summary = summarize_run(
        episode_results,
        memory_size,
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
