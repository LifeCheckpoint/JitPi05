"""Single-method, single-seed first-stage JitRL simulation runs."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.utils import close_envs
from lerobot.utils.io_utils import write_video
from tqdm.auto import tqdm

from jitpi05.config import (
    ALL_METHODS,
    CYCLE_BUDGET_MODE,
    CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS,
    CYCLE_CONFIG_VERSION,
    CYCLE_GRIPPER_CLOSED_QPOS_THRESHOLD,
    CYCLE_MAX_RETRIES,
    CYCLE_MBR_ACTION_STEPS,
    CYCLE_MBR_DELTA_DIMS,
    CYCLE_MBR_HYPOTHESES,
    CYCLE_METHODS,
    CYCLE_PREDICTOR_RETRIES,
    CYCLE_PROGRESS_THRESHOLD,
    CYCLE_PROXY_PROGRESS_THRESHOLD,
    CYCLE_SIGNAL_CONFIRM_CONSECUTIVE,
    CYCLE_SIGNAL_CONFIRM_GAP,
    CYCLE_STOP_SIGNAL_THRESHOLD,
    CYCLE_UNIFIED_MAX_STEPS,
    CYCLE_VLM_BACKTRACK_LIKELIHOODS,
    JITRL_ACTION_WORKSPACE_VERSION,
    JITRL_BETA,
    JITRL_EPISODES,
    JITRL_EVALUATOR_BACKEND,
    JITRL_EVALUATOR_MODEL,
    JITRL_EVALUATOR_RETRIES,
    JITRL_EVALUATOR_SCORE_SCALE,
    JITRL_EXPLORATION_RATE,
    JITRL_FREE_ACTION_SIM_THRESHOLD,
    JITRL_FREE_PLAN_TOKENS,
    JITRL_GAMMA,
    JITRL_HIGH_LEVEL_STEPS,
    JITRL_HISTORY_SIZE,
    JITRL_LOGIT_CALIBRATION,
    JITRL_LOW_LEVEL_BACKEND,
    JITRL_MAX_PLAN_TOKENS,
    JITRL_OUTPUT_DIR,
    JITRL_PLANNER_RETRIES,
    JITRL_QWEN_ID,
    JITRL_REWARD_VERSION,
    JITRL_TEMPERATURE,
    JITRL_TERMINAL_SUCCESS_BONUS,
    JITRL_TERMINATION_MODE,
    JITRL_TOP_K,
    JITRL_UCB_ALPHA,
    POLICY_ACTION_STEPS,
    RLINF_USE_RAW_TASK_PROMPT,
    SIM_ACTION_STEPS,
    SIM_PI05_ID,
)
from jitpi05.jitrl.cycle import (
    CycleGate,
    CyclePhase,
    CycleSignalConfirmation,
    PhysicalEvidence,
    SemanticAnchor,
    adapt_cycle_policy_output,
    build_cycle_subtask_program,
    format_cycle_condition,
    format_cycle_subtask_program,
    match_cycle_subtask,
    proxy_subtask_stop,
    resolve_cycle_decision,
    select_mbr_chunk,
)
from jitpi05.jitrl.libero_recovery import (
    build_physical_evidence,
    capture_robot_configuration,
    gripper_pose,
    held_object_name,
    inspect_scene_objects,
    match_target_object,
    refresh_libero_observation,
    rewind_robot_configuration_history,
)
from jitpi05.jitrl.memory import JitRLMemory, discounted_returns
from jitpi05.jitrl.planner import (
    apply_jitrl_update,
    load_jitrl_planner,
    plan_and_score,
    plan_and_score_free,
    score_free_candidates,
)
from jitpi05.jitrl.vlm import (
    QwenChunkEvaluator,
    load_cycle_failure_predictor,
    predict_cycle_failure,
)
from jitpi05.policy import conditioned_task
from jitpi05.simulation import (
    load_policy,
    make_single_env,
    planning_images,
    policy_frame,
    reset_rollout_state,
    success_from_info,
)

# Free-candidate methods propose free-text actions from Qwen (no fixed workspace).
FREE_METHODS = (
    "jitrl-free",
    "static-free",
    "jitrl-free-cycle",
    "static-free-cycle",
)


def _low_level_condition(overall_task: str, subtask: str) -> str:
    """选择低层 VLA prompt；RLinf 诊断必须遵循官方 raw-task 协议。"""
    if RLINF_USE_RAW_TASK_PROMPT and JITRL_LOW_LEVEL_BACKEND == "rlinf":
        return overall_task
    return conditioned_task(overall_task, subtask)
# Methods that populate/read the online experience memory. Recovery actions are
# deliberately excluded from episode_records, so Cycle does not change the JitRL
# learning boundary.
MEMORY_METHODS = ("jitrl", "jitrl-free", "jitrl-free-cycle")
# Methods that apply the memory advantage logit update (beta > 0).
UPDATED_METHODS = ("jitrl", "jitrl-free", "jitrl-free-cycle")
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
    kind: str = "flow_noise",
) -> torch.Tensor:
    """Rebuild one coordinate-local flow-noise tensor for paired rollout or MBR."""

    generator = torch.Generator(device="cpu").manual_seed(
        coordinate_seed(kind, seed, episode, chunk)
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


def evaluation_budget_max_steps(task_spec: dict[str, Any]) -> int:
    """Return the single episode budget shared by baseline and Cycle rollouts."""

    try:
        official_max_steps = int(task_spec["max_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("task_spec must contain an integer max_steps") from error
    if official_max_steps <= 0:
        raise ValueError("task_spec max_steps must be positive")
    if CYCLE_BUDGET_MODE == "official":
        return official_max_steps
    if CYCLE_BUDGET_MODE == "unified_800":
        if CYCLE_UNIFIED_MAX_STEPS < official_max_steps:
            raise ValueError(
                "CYCLE_UNIFIED_MAX_STEPS must be at least the task's official max_steps"
            )
        return CYCLE_UNIFIED_MAX_STEPS
    raise ValueError(
        "CYCLE_BUDGET_MODE must be 'official' or 'unified_800'; "
        f"got {CYCLE_BUDGET_MODE!r}"
    )


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

    max_retries = JITRL_PLANNER_RETRIES
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
                evaluation = evaluator.evaluate(
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
                # Exponential backoff for every retry so the attempts spread
                # across a longer outage window instead of all landing in one.
                time.sleep(min(2.0**attempt, 60.0))
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


def _run_episode_cycle(
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
    cycle_predictor=None,
    step_progress=None,
) -> tuple[dict, list[dict], dict[str, torch.Tensor], list[tuple[list, list]]]:
    """Run the zero-shot CycleVLA-lite state machine for one episode.

    The ordinary Qwen/JitRL path remains separate below.  Cycle recovery is a
    physical intervention on the existing environment state: it never calls the
    planner, never creates an experience record, and never appends a synthetic
    evaluator frame.  MBR candidates are generated only after an accepted
    backtrack and are selected as an original pi0.5 hypothesis.
    """

    cycle_wall_start = time.perf_counter()
    budget_max_steps = evaluation_budget_max_steps(task_spec)
    envs, env, env_preprocessor, env_postprocessor = make_single_env(
        task_spec,
        episode_index,
        budget_max_steps=budget_max_steps,
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
    anchors: list[SemanticAnchor] = []
    robot_history: list[Any] = []
    anchor_history_indices: dict[str, int] = {}
    anchor_traces: dict[str, dict] = {}
    evaluated_anchor_ids: set[str] = set()
    recovery_events: list[dict] = []
    cycle_trace: list[dict] = []
    active_anchor: SemanticAnchor | None = None
    active_trace: dict | None = None
    active_condition = ""
    active_before_images: list | None = None
    active_attempt_chunks = 0
    active_attempt_is_recovery = False
    active_attempt_checked = False
    # 回溯后 MBR 选定的 rollout 坐标基址：恢复期间后续 chunk 沿用同一 MBR seed
    # 系列滚动生成（重 rollout 整个子任务），而不是每个 chunk 用全局递增坐标。
    mbr_rollout_coordinate: int | None = None
    phase = CyclePhase.IN_PROGRESS
    cycle_program = ()
    cycle_program_text: tuple[str, ...] = ()
    cycle_signal_mode = "unknown"
    cycle_trigger_mode = "not_triggered"
    progress_confirmation = CycleSignalConfirmation(
        threshold=CYCLE_PROGRESS_THRESHOLD,
        required_consecutive=CYCLE_SIGNAL_CONFIRM_CONSECUTIVE,
        required_gap=CYCLE_SIGNAL_CONFIRM_GAP,
    )
    stop_confirmation = CycleSignalConfirmation(
        threshold=CYCLE_STOP_SIGNAL_THRESHOLD,
        required_consecutive=CYCLE_SIGNAL_CONFIRM_CONSECUTIVE,
        required_gap=CYCLE_SIGNAL_CONFIRM_GAP,
    )
    retry_counts: dict[str, int] = {}
    last_info: dict = {}
    overall_task = ""
    success = False
    episode_done = False
    termination_reason: str | None = None
    cycle_retry_count = 0
    cycle_check_count = 0
    cycle_backtrack_count = 0
    cycle_veto_count = 0
    cycle_proxy_stop_finish_count = 0
    mbr_events: list[dict] = []
    cycle_rewind_steps = 0
    low_level_chunks_per_plan = _low_level_chunks_per_high_level_plan()
    gate = CycleGate(threshold=CYCLE_PROXY_PROGRESS_THRESHOLD)
    last_held_relative: tuple[float, float, float] | None = None

    def _current_images() -> list:
        current_frame = policy_frame(observation, overall_task, env_preprocessor)
        return [image.copy() for image in planning_images(current_frame)]

    def _finish_active_subtask() -> None:
        """Close one anchor only after completion is confirmed or fallback ends."""

        nonlocal active_anchor, active_trace, active_condition
        nonlocal active_attempt_chunks, active_attempt_is_recovery
        nonlocal active_attempt_checked, phase, active_before_images
        _finalize_evaluator_images()
        active_anchor = None
        active_trace = None
        active_condition = ""
        active_attempt_chunks = 0
        active_attempt_is_recovery = False
        active_attempt_checked = False
        active_before_images = None
        progress_confirmation.reset()
        stop_confirmation.reset()
        phase = CyclePhase.IN_PROGRESS

    def _finalize_evaluator_images() -> None:
        nonlocal active_before_images
        if (
            active_anchor is None
            or active_before_images is None
            or active_anchor.anchor_id in evaluated_anchor_ids
        ):
            active_before_images = None
            return
        evaluator_images.append((active_before_images, _current_images()))
        evaluated_anchor_ids.add(active_anchor.anchor_id)
        active_before_images = None

    def _budget_steps() -> int:
        return len(actions) + cycle_rewind_steps

    def _execute_action_chunk(
        policy_output,
        *,
        source: str,
        anchor_id: str,
    ) -> dict:
        nonlocal observation, episode_done, success, termination_reason, last_info, phase
        nonlocal cycle_signal_mode
        chunk_index = len(action_chunks)
        action_start_step = len(actions)
        adapted = (
            policy_output
            if hasattr(policy_output, "robot_action_chunk")
            else adapt_cycle_policy_output(policy_output)
        )
        if cycle_signal_mode == "unknown":
            cycle_signal_mode = adapted.signal_source
        elif cycle_signal_mode != adapted.signal_source:
            cycle_signal_mode = "mixed"
        cpu_chunk = adapted.robot_action_chunk.detach().cpu().clone()
        action_chunks.append(cpu_chunk)
        progress_confirmed = False
        stop_confirmed = False
        for local_index, action in enumerate(cpu_chunk[:POLICY_ACTION_STEPS]):
            if _budget_steps() >= max_steps:
                break
            action_transition = env_postprocessor({"action": action.unsqueeze(0)})
            env_action = action_transition["action"].detach().cpu()
            observation, reward, terminated, truncated, info = env.step(
                env_action.numpy()
            )
            actions.append(env_action.squeeze(0).clone())
            rewards.append(float(reward[0]))
            robot_history.append(capture_robot_configuration(env))
            frames.append(env.render()[0])
            last_info = info if isinstance(info, dict) else {}
            if adapted.has_policy_signals:
                progress_confirmed = (
                    progress_confirmation.update(adapted.progress_signal(local_index))
                    or progress_confirmed
                )
                stop_confirmed = (
                    stop_confirmation.update(adapted.stop_signal(local_index))
                    or stop_confirmed
                )
            if step_progress is not None:
                step_progress.update(1)
            episode_done = bool(terminated[0] or truncated[0])
            success = success or success_from_info(last_info)
            if success:
                termination_reason = "environment_success"
            elif episode_done:
                termination_reason = "environment_done"
            elif _budget_steps() >= max_steps:
                termination_reason = "max_steps"
            if episode_done or success or _budget_steps() >= max_steps:
                break
        return {
            "chunk_index": chunk_index,
            "action_start_step": action_start_step,
            "action_end_step": len(actions),
            "source": source,
            "anchor_id": anchor_id,
            "signal_source": adapted.signal_source,
            "progress_confirmed": progress_confirmed,
            "stop_confirmed": stop_confirmed,
            "executed_steps": len(actions) - action_start_step,
        }

    def _predict_action_chunk(frame: dict, condition: str, noise: torch.Tensor):
        conditioned_frame = dict(frame)
        conditioned_frame["task"] = [condition]
        batch = preprocessor(conditioned_frame)
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(batch, noise=noise)
        action_chunk = postprocessor(normalized_chunk).squeeze(0)
        del batch, normalized_chunk, noise
        if action_chunk.shape[0] != policy.config.chunk_size:
            raise ValueError(
                "predict_action_chunk must return the complete configured action chunk"
            )
        return adapt_cycle_policy_output(action_chunk)

    def _run_mbr_retry(
        anchor: SemanticAnchor,
        restored_frame: dict,
        condition: str | None = None,
    ) -> dict:
        nonlocal mbr_rollout_coordinate
        exec_condition = anchor.condition if condition is None else condition
        candidates = []
        candidate_seeds: list[int] = []
        mbr_retry_index = max(0, cycle_retry_count - 1)
        mbr_coordinate = (
            anchor.high_level_step_index * CYCLE_MAX_RETRIES * CYCLE_MBR_HYPOTHESES
            + mbr_retry_index * CYCLE_MBR_HYPOTHESES
        )
        for candidate_index in range(CYCLE_MBR_HYPOTHESES):
            coordinate = mbr_coordinate + candidate_index
            candidate_seeds.append(
                coordinate_seed("mbr_flow_noise", seed, episode_index, coordinate)
            )
            noise = coordinate_flow_noise(
                policy,
                seed,
                episode_index,
                coordinate,
                device="cuda",
                kind="mbr_flow_noise",
            )
            candidates.append(_predict_action_chunk(restored_frame, exec_condition, noise))
        robot_chunks = torch.stack([candidate.robot_action_chunk for candidate in candidates])
        _, selection = select_mbr_chunk(
            robot_chunks,
            action_steps=CYCLE_MBR_ACTION_STEPS,
            delta_dims=CYCLE_MBR_DELTA_DIMS,
        )
        selected_output = candidates[selection.selected_index]
        action_record = _execute_action_chunk(
            selected_output,
            source="mbr_retry",
            anchor_id=anchor.anchor_id,
        )
        # 官方 MBR 是在回溯后选定代表 seed 然后重新 rollout 整个子任务（多个
        # chunk 连续执行），而不是只修正一个动作块。记录选中 seed 的坐标基址，
        # 主循环恢复期间沿用该基址滚动生成后续 chunk，实现「重 rollout 子任务」。
        mbr_rollout_coordinate = mbr_coordinate + selection.selected_index
        event = {
            "event": "mbr_retry",
            "anchor_id": anchor.anchor_id,
            "high_level_step_index": anchor.high_level_step_index,
            "condition": exec_condition,
            "hypothesis_count": CYCLE_MBR_HYPOTHESES,
            "action_steps": CYCLE_MBR_ACTION_STEPS,
            "delta_dims": CYCLE_MBR_DELTA_DIMS,
            "candidate_flow_noise_seeds": candidate_seeds,
            "rollout_coordinate": mbr_rollout_coordinate,
            "selected_index": selection.selected_index,
            "selected_risk": selection.selected_risk,
            "mean_pairwise_distance": selection.mean_pairwise_distance,
            "risks": list(selection.risks),
            "features": selection.features.tolist(),
            "pairwise_distances": selection.pairwise_distances.tolist(),
            "executed_chunk": action_record,
        }
        if active_trace is not None:
            active_trace["retry_low_level_chunks"].append(action_record)
        mbr_events.append(event)
        recovery_events.append(event)
        return event

    def _replan_after_rewind(anchor: SemanticAnchor) -> dict:
        """Re-confirm the high-level target from the post-rewind observation.

        The arm and gripper have been rewound to an earlier state but the object
        scene is deliberately *not* rolled back.  Reusing the stale
        ``anchor.condition`` against the changed scene is exactly what lets the
        7-D policy grab the wrong object after a backtrack.  We therefore plan
        once more from the current observation and return a fresh execution
        condition.  Recovery never queries JitRL memory and never appends an
        experience record, keeping this an inference-only intervention.
        """

        replan_images = planning_images(
            policy_frame(observation, overall_task, env_preprocessor)
        )
        planning, planner_attempts = _plan_and_score_with_retry(
            planner_model,
            planner_processor,
            replan_images,
            overall_task,
            selected_subtasks[-JITRL_HISTORY_SIZE:],
            method=method,
            seed=seed,
            episode_index=episode_index,
            high_level_step_index=anchor.high_level_step_index,
        )
        replan_candidates = planning["candidates"]
        value_estimate = _trace_value_estimate(
            _static_value_estimate(
                [candidate["semantic_key"] for candidate in replan_candidates]
            )
        )
        update = apply_jitrl_update(
            planning["base_logits"],
            [item["normalized_advantage"] for item in value_estimate["candidates"]],
            beta=0.0,
            temperature=JITRL_TEMPERATURE,
            uniform=coordinate_uniform(seed, episode_index, anchor.high_level_step_index),
        )
        selected_index = update["base_choice_index"]
        selected_candidate = replan_candidates[selected_index]
        program_node = match_cycle_subtask(
            selected_candidate,
            cycle_program,
            preferred_position=anchor.program_position,
            used_ids=[existing.program_id for existing in anchors if existing.program_id],
        )
        # 与 anchor 创建一致：condition 用官方模板（程序节点文本），避免自由文本
        # 导致的 prompt 分布漂移。
        condition = format_cycle_condition(overall_task, program_node.text)
        return {
            "planning": planning,
            "planner_attempts": planner_attempts,
            "value_estimate": value_estimate,
            "update": update,
            "selected_candidate": selected_candidate,
            "selected_index": selected_index,
            "condition": condition,
            "program_node": program_node,
        }

    def _physical_evidence(anchor: SemanticAnchor) -> PhysicalEvidence:
        nonlocal last_held_relative
        objects = inspect_scene_objects(env)
        target_object = match_target_object(anchor.action_text, objects)
        base = build_physical_evidence(
            env,
            observation,
            success=success,
            gripper_closed_threshold=CYCLE_GRIPPER_CLOSED_QPOS_THRESHOLD,
            action_type=anchor.action_type,
            target_object=target_object,
            previous_held_relative=last_held_relative,
        )
        gripper_position = gripper_pose(env)
        held = (
            held_object_name(env, objects=objects, gripper_position=gripper_position)
            if gripper_position is not None
            else None
        )
        if held and gripper_position is not None:
            held_position = next(
                (
                    scene_object.position
                    for scene_object in objects
                    if scene_object.canonical_name == held
                ),
                None,
            )
            last_held_relative = (
                (
                    held_position[0] - gripper_position[0],
                    held_position[1] - gripper_position[1],
                    held_position[2] - gripper_position[2],
                )
                if held_position is not None
                else None
            )
        else:
            last_held_relative = None
        raw = last_info.get("cycle_physical_evidence") if isinstance(last_info, dict) else None
        values = raw if isinstance(raw, dict) else {}
        success_fact = success or values.get("postcondition_satisfied") is True
        failure_fact = values.get("failure_detected")
        reasons = values.get("failure_reasons", ())
        if isinstance(reasons, str):
            reasons = (reasons,)
        if failure_fact is None:
            reasons = (*reasons, *base.failure_reasons)
        return PhysicalEvidence(
            action_type=anchor.action_type,
            gripper_closed=(
                base.gripper_closed
                if values.get("gripper_closed") is None
                else values.get("gripper_closed")
            ),
            object_following_gripper=(
                values.get("object_following_gripper")
                if values.get("object_following_gripper") is not None
                else base.object_following_gripper
            ),
            target_identity_ok=(
                values.get("target_identity_ok")
                if values.get("target_identity_ok") is not None
                else base.target_identity_ok
            ),
            destination_reached=(
                values.get("destination_reached")
                if values.get("destination_reached") is not None
                else base.destination_reached
            ),
            released=values.get("released"),
            postcondition_satisfied=(
                True
                if success_fact
                else (
                    base.postcondition_satisfied
                    if values.get("postcondition_satisfied") is None
                    else values.get("postcondition_satisfied")
                )
            ),
            failure_detected=failure_fact,
            failure_reasons=tuple(str(reason) for reason in reasons),
        )

    try:
        reset_rollout_state(policy, preprocessor, postprocessor, episode_seed)
        observation, _ = env.reset(seed=[episode_seed])
        robot_history.append(capture_robot_configuration(env))
        overall_task = list(env.call("task_description"))[0]
        cycle_program = build_cycle_subtask_program(overall_task)
        cycle_program_text = format_cycle_subtask_program(cycle_program)
        max_steps = int(env.call("_max_episode_steps")[0])
        frames.append(env.render()[0])

        while _budget_steps() < max_steps and not success and not episode_done:
            if active_anchor is None:
                low_level_chunk_index = len(action_chunks)
                high_level_step_index = len(chunks)
                frame = policy_frame(observation, overall_task, env_preprocessor)
                images = planning_images(frame)
                active_before_images = [image.copy() for image in images]
                if step_progress is not None:
                    step_progress.set_postfix_str(
                        f"low={low_level_chunk_index} high={high_level_step_index} planner=qwen-cycle",
                        refresh=True,
                    )
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
                if planning.get("stop_only_masked"):
                    if not selected_subtasks:
                        termination_reason = "qwen_stop"
                        evaluator_images.append((active_before_images, [image.copy() for image in images]))
                        active_before_images = None
                        break
                    repeated = dict(chunks[-1]["selected_candidate"])
                    repeated["label"] = "1"
                    planning = score_free_candidates(
                        planner_model,
                        planner_processor,
                        images,
                        overall_task,
                        {**planning, "candidates": [repeated], "qwen_candidates": [dict(repeated)]},
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
                        seed, episode_index, high_level_step_index, len(candidates)
                    )
                    if method == "jitrl":
                        value_estimate = memory.estimate_candidate_values(
                            planning["state_summary"], action_keys,
                            neighbors=raw_neighbors,
                            exploration_uniforms=exploration_uniforms,
                            exploration_rate=JITRL_EXPLORATION_RATE,
                            ucb_alpha=JITRL_UCB_ALPHA,
                        )
                    else:
                        value_estimate = memory.estimate_candidate_values_free(
                            planning["state_summary"], [candidate["text"] for candidate in candidates],
                            neighbors=raw_neighbors,
                            exploration_uniforms=exploration_uniforms,
                            exploration_rate=JITRL_EXPLORATION_RATE,
                            ucb_alpha=JITRL_UCB_ALPHA,
                            action_sim_threshold=JITRL_FREE_ACTION_SIM_THRESHOLD,
                        )
                else:
                    value_estimate = _static_value_estimate(action_keys)
                value_estimate = _trace_value_estimate(value_estimate)
                update = apply_jitrl_update(
                    planning["base_logits"],
                    [item["normalized_advantage"] for item in value_estimate["candidates"]],
                    beta=JITRL_BETA if method in UPDATED_METHODS else 0.0,
                    temperature=JITRL_TEMPERATURE,
                    uniform=coordinate_uniform(seed, episode_index, high_level_step_index),
                )
                selected_index = (
                    update["updated_choice_index"] if method in UPDATED_METHODS
                    else update["base_choice_index"]
                )
                selected_candidate = candidates[selected_index]
                anchor_id = f"anchor-{high_level_step_index}"
                program_node = match_cycle_subtask(
                    selected_candidate,
                    cycle_program,
                    preferred_position=high_level_step_index,
                    used_ids=[anchor.program_id for anchor in anchors if anchor.program_id],
                )
                # 官方模板 condition：低层 7-D policy 的训练 prompt 分布就是
                # ``Task: ... The current subtask: ...``，且 subtask 文本取自确定性
                # 程序节点（与 released decomposed checkpoint 训练分布一致），
                # 而不是注入 Qwen 自由文本（避免分布漂移导致抓错物体）。
                active_condition = format_cycle_condition(overall_task, program_node.text)
                anchor_target_object = (
                    match_target_object(
                        selected_candidate["text"], inspect_scene_objects(env)
                    )
                    or ""
                )
                active_anchor = SemanticAnchor(
                    anchor_id=anchor_id,
                    action_text=selected_candidate["text"],
                    condition=active_condition,
                    high_level_step_index=high_level_step_index,
                    # action_type 必须用程序节点的模板类型（approach/grasp/
                    # transport/release），而不是 Qwen 自由文本分类：官方模板
                    # 文本语义（如 "Move ... while holding the book" = transport）
                    # 与 Qwen 自由文本（如 "place the book at the caddy" = place）
                    # 不一致，用 Qwen 分类会让 proxy_subtask_stop / 物理失败信号
                    # / gripper 特判全部用错动作语义（v7 中 transport/release 被
                    # 误标成 place 数十次，是 0/10 的直接原因）。
                    action_type=program_node.action_type,
                    program_id=program_node.subtask_id,
                    program_position=program_node.position,
                    target_object=anchor_target_object,
                )
                anchors.append(active_anchor)
                anchor_history_indices[anchor_id] = len(robot_history) - 1
                episode_records.append(
                    {
                        "state": planning["state_summary"],
                        "action_key": selected_candidate["semantic_key"],
                        "action_text": selected_candidate["text"],
                        "chunk_index": high_level_step_index,
                    }
                )
                selected_subtasks.append(selected_candidate["text"])
                progress_confirmation.reset()
                stop_confirmation.reset()
                phase = CyclePhase.IN_PROGRESS
                active_trace = {
                    "chunk_index": high_level_step_index,
                    "anchor_id": anchor_id,
                    "program_id": active_anchor.program_id,
                    "program_position": active_anchor.program_position,
                    "planner_attempts": planner_attempts,
                    "binding_prompt": planning["binding_prompt"],
                    "binding_raw_output": planning["binding_raw_output"],
                    "binding_assistant_prefill": planning["binding_assistant_prefill"],
                    "binding_generated_tokens": planning["binding_generated_tokens"],
                    "binding_generation_reached_limit": planning["binding_generation_reached_limit"],
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
                    "action_start_step": len(actions),
                    "action_end_step": len(actions),
                    "low_level_chunks": [],
                    "retry_low_level_chunks": [],
                }
                chunks.append(active_trace)
                anchor_traces[anchor_id] = active_trace
                active_attempt_chunks = 0
                active_attempt_is_recovery = False
                active_attempt_checked = False
                if selected_candidate["terminates_rollout"]:
                    termination_reason = "qwen_stop"
                    evaluator_images.append((active_before_images, [image.copy() for image in images]))
                    active_before_images = None
                    break

            frame = policy_frame(observation, overall_task, env_preprocessor)
            low_level_chunk_index = len(action_chunks)
            # 回溯恢复期间沿用 MBR 选定的 seed 基址滚动生成后续 chunk，重新
            # rollout 整个子任务；非恢复路径保持全局递增坐标（确定性配对）。
            if active_attempt_is_recovery and mbr_rollout_coordinate is not None:
                noise_coordinate = mbr_rollout_coordinate + active_attempt_chunks
                noise_kind = "mbr_rollout_noise"
            else:
                noise_coordinate = low_level_chunk_index
                noise_kind = "flow_noise"
            noise = coordinate_flow_noise(
                policy, seed, episode_index, noise_coordinate, device="cuda", kind=noise_kind
            )
            action_chunk = _predict_action_chunk(frame, active_condition, noise)
            action_record = _execute_action_chunk(
                action_chunk, source="policy", anchor_id=active_anchor.anchor_id
            )
            if active_trace is None or active_anchor is None:
                raise RuntimeError("Cycle active action trace is missing")
            active_trace["action_end_step"] = len(actions)
            low_level_key = (
                "retry_low_level_chunks" if active_attempt_is_recovery else "low_level_chunks"
            )
            active_trace[low_level_key].append(action_record)
            active_attempt_chunks += 1

            if episode_done or success or _budget_steps() >= max_steps:
                _finalize_evaluator_images()
                active_anchor = None
                break

            signal_source = action_record["signal_source"]
            progress_triggered = False
            trigger_mode = None
            if not active_attempt_checked and phase == CyclePhase.IN_PROGRESS:
                if signal_source == "policy_9d":
                    progress_triggered = bool(action_record["progress_confirmed"])
                    trigger_mode = "learned_progress_confirmed"
                else:
                    progress_triggered = (
                        active_attempt_chunks >= CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS
                        and gate.triggered(active_attempt_chunks, low_level_chunks_per_plan)
                    )
                    trigger_mode = "fixed_horizon_degraded"
            if progress_triggered:
                active_attempt_checked = True
                phase = CyclePhase.CHECK
                cycle_trigger_mode = trigger_mode or cycle_trigger_mode
                cycle_check_count += 1
                current_images = _current_images()
                physical = _physical_evidence(active_anchor)
                # 官方对 gripper 子任务（close/open the gripper）跳过 VLM 检查，
                # 直接进入完成阶段。这里沿用该特判：不再把预测成本花在纯夹爪
                # 开合上，但物理证据检查仍然保留（抓空/夹爪未闭合仍可触发回溯）。
                is_gripper_subtask = (
                    active_anchor.action_type in {"grasp", "release"}
                    or "close the gripper to grasp" in active_anchor.action_text.lower()
                    or "open the gripper to release" in active_anchor.action_text.lower()
                )
                # 官方 per-subtask retry 达上限后强制 to_complete（跳过 VLM 检查
                # 直接推进），避免对同一子任务无限回溯。resolve_cycle_decision 的
                # retry-limit 分支会返回 transit，此处提前跳过 VLM 预测成本。
                retry_limit_reached = (
                    retry_counts.get(active_anchor.anchor_id, 0) >= CYCLE_MAX_RETRIES
                )
                prediction = None
                prediction_error = None
                if cycle_predictor is not None and not is_gripper_subtask and not retry_limit_reached:
                    for predictor_attempt in range(1, CYCLE_PREDICTOR_RETRIES + 1):
                        try:
                            prediction = predict_cycle_failure(
                                cycle_predictor,
                                current_images,
                                overall_task,
                                cycle_program_text,
                                f"{active_anchor.program_id}: {active_anchor.action_text}",
                                attempt=cycle_check_count,
                            )
                            break
                        except Exception as error:
                            if predictor_attempt == CYCLE_PREDICTOR_RETRIES:
                                prediction_error = f"{type(error).__name__}: {error}"
                            else:
                                time.sleep(min(2.0**predictor_attempt, 10.0))
                decision = resolve_cycle_decision(
                    vlm=prediction,
                    physical=physical,
                    anchors=anchors,
                    current_anchor=active_anchor,
                    retry_count=0,
                    retry_counts=retry_counts,
                    max_retries=CYCLE_MAX_RETRIES,
                    backtrack_likelihoods=CYCLE_VLM_BACKTRACK_LIKELIHOODS,
                )
                raw_vlm_target = ""
                if isinstance(prediction, dict):
                    raw_vlm_target = prediction.get(
                        "next_subtask", prediction.get("next_anchor_id", "")
                    )
                self_target_requested = bool(
                    raw_vlm_target
                    and (
                        raw_vlm_target == active_anchor.anchor_id
                        or raw_vlm_target == active_anchor.action_text
                        or raw_vlm_target == active_anchor.program_id
                    )
                )
                trace_event = {
                    "event": "cycle_check",
                    "phase": phase.value,
                    "trigger_mode": trigger_mode,
                    "signal_source": signal_source,
                    "anchor_id": active_anchor.anchor_id,
                    "program_id": active_anchor.program_id,
                    "completed_chunks": active_attempt_chunks,
                    "total_chunks": low_level_chunks_per_plan,
                    "proxy_progress": gate.proxy_progress(active_attempt_chunks, low_level_chunks_per_plan),
                    "progress_confirmed": action_record["progress_confirmed"],
                    "stop_confirmed": action_record["stop_confirmed"],
                    "physical_evidence": {
                        "action_type": physical.action_type,
                        "gripper_closed": physical.gripper_closed,
                        "target_identity_ok": physical.target_identity_ok,
                        "object_following_gripper": physical.object_following_gripper,
                        "destination_reached": physical.destination_reached,
                        "postcondition_satisfied": physical.postcondition_satisfied,
                        "failure_detected": physical.failure_detected,
                        "failure_reasons": list(physical.failure_reasons),
                    },
                    "prediction": prediction,
                    "prediction_error": prediction_error,
                    "gripper_subtask": is_gripper_subtask,
                    "retry_limit_reached": retry_limit_reached,
                    "self_target_requested": self_target_requested,
                    "decision": decision.decision,
                    "decision_reason": decision.reason,
                    "next_anchor_id": decision.next_anchor_id,
                    "retry_count": decision.retry_count,
                    "accepted": decision.accepted,
                    "physical_failure": decision.physical_failure,
                    "vlm_requested_backtrack": decision.vlm_requested_backtrack,
                    "vetoed": decision.vetoed,
                }
                cycle_trace.append(trace_event)
                if decision.vetoed:
                    cycle_veto_count += 1
                if decision.decision == "backtrack" and decision.next_anchor_id is not None:
                    phase = CyclePhase.BACKTRACK
                    target_anchor = next(
                        anchor for anchor in anchors if anchor.anchor_id == decision.next_anchor_id
                    )
                    _finalize_evaluator_images()
                    target_history_index = anchor_history_indices[target_anchor.anchor_id]
                    rewind_budget = max(0, max_steps - _budget_steps())

                    def _record_rewind_waypoint(
                        _history_index: int,
                        _restore_report,
                    ) -> None:
                        nonlocal cycle_rewind_steps
                        cycle_rewind_steps += 1
                        frames.append(env.render()[0])
                        if step_progress is not None:
                            step_progress.update(1)

                    rewind_report = rewind_robot_configuration_history(
                        env,
                        robot_history,
                        target_history_index,
                        max_steps=rewind_budget,
                        on_waypoint=_record_rewind_waypoint,
                    )
                    observation = refresh_libero_observation(env)
                    recovery_event = {
                        "event": "backtrack",
                        "phase": "backtrack",
                        "mode": "reverse_robot_state_replay",
                        "from_anchor_id": active_anchor.anchor_id,
                        "from_target_object": active_anchor.target_object,
                        "to_anchor_id": target_anchor.anchor_id,
                        "to_target_object": target_anchor.target_object,
                        "decision_reason": decision.reason,
                        "target_history_index": target_history_index,
                        "final_history_index": rewind_report.final_history_index,
                        "source_history_steps": rewind_report.source_history_steps,
                        "replayed_steps": rewind_report.replayed_steps,
                        "completed": rewind_report.completed,
                        "waypoint_indices": list(rewind_report.waypoint_indices),
                        "max_waypoint_qpos_delta": rewind_report.max_waypoint_qpos_delta,
                        "restore_report": {
                            "qpos_max_error": rewind_report.qpos_max_error,
                            "robot_qvel_max_abs": rewind_report.robot_qvel_max_abs,
                            "non_robot_qpos_max_change": rewind_report.non_robot_qpos_max_change,
                            "controller_resynchronized": rewind_report.controller_resynchronized,
                        },
                    }
                    recovery_events.append(recovery_event)
                    if not rewind_report.completed:
                        termination_reason = "max_steps"
                        break
                    del robot_history[target_history_index + 1 :]
                    valid_anchor_ids = {
                        anchor.anchor_id
                        for anchor in anchors
                        if anchor_history_indices[anchor.anchor_id] <= target_history_index
                    }
                    anchors[:] = [
                        anchor for anchor in anchors if anchor.anchor_id in valid_anchor_ids
                    ]
                    anchor_history_indices = {
                        anchor_id: history_index
                        for anchor_id, history_index in anchor_history_indices.items()
                        if anchor_id in valid_anchor_ids
                    }
                    recovery_event["invalidated_anchor_ids"] = sorted(
                        set(anchor_traces) - valid_anchor_ids
                    )
                    cycle_backtrack_count += 1
                    retry_counts[target_anchor.anchor_id] = decision.retry_count
                    cycle_retry_count = max(retry_counts.values(), default=0)
                    active_anchor = target_anchor
                    active_trace = anchor_traces[target_anchor.anchor_id]
                    active_attempt_chunks = 0
                    active_attempt_is_recovery = True
                    active_attempt_checked = False
                    progress_confirmation.reset()
                    stop_confirmation.reset()
                    if _budget_steps() >= max_steps:
                        termination_reason = "max_steps"
                        break
                    # 回溯后重新进行高层目标确认/规划：机器人已回到旧姿态，但
                    # 物体场景未回滚，旧 anchor.condition 与新场景错配正是回溯后
                    # 抓错物体的直接原因。必须基于当前观测重新确认目标，再把新的
                    # 执行条件交给 MBR 重试，而不是直接沿用旧条件。
                    target_anchor_list_index = next(
                        index
                        for index, existing_anchor in enumerate(anchors)
                        if existing_anchor.anchor_id == target_anchor.anchor_id
                    )
                    selected_subtasks[:] = selected_subtasks[
                        : target_anchor_list_index + 1
                    ]
                    replan = None
                    replan_error = None
                    try:
                        replan = _replan_after_rewind(target_anchor)
                    except Exception as error:
                        replan_error = f"{type(error).__name__}: {error}"
                    if replan is not None:
                        replan_program_node = replan["program_node"]
                        replan_attempts = replan["planner_attempts"]
                        replan_candidate = replan["selected_candidate"]["text"]
                        replan_selected_index = replan["selected_index"]
                        replan_program_id = replan_program_node.subtask_id
                        replan_program_position = replan_program_node.position
                        if replan_program_id == target_anchor.program_id:
                            # replan 与回溯目标一致：采用 replan 的 condition。
                            replanned_condition = replan["condition"]
                            replan_forced_alignment = False
                        else:
                            # 官方语义：回溯后就是重做回溯目标子任务本身，不自由
                            # 切换到其它节点。当 replan 与目标不一致时强制对齐到
                            # 回溯目标节点的官方模板 condition，避免 condition 与
                            # 场景错配导致抓错物体。
                            if (
                                target_anchor.program_position >= 0
                                and target_anchor.program_position < len(cycle_program)
                            ):
                                target_template = cycle_program[
                                    target_anchor.program_position
                                ].text
                                replanned_condition = format_cycle_condition(
                                    overall_task, target_template
                                )
                            else:
                                replanned_condition = target_anchor.condition
                            replan_forced_alignment = True
                    else:
                        # Planner outage must not crash the episode: fall back to
                        # the recovered anchor's original condition and audit the
                        # degraded re-confirmation in the recovery event.
                        replanned_condition = target_anchor.condition
                        replan_attempts = 0
                        replan_candidate = target_anchor.action_text
                        replan_selected_index = None
                        replan_program_id = target_anchor.program_id
                        replan_program_position = target_anchor.program_position
                        replan_forced_alignment = True
                    active_condition = replanned_condition
                    recovery_event["replanned"] = {
                        "triggered": replan is not None,
                        "error": replan_error,
                        "planner_attempts": replan_attempts,
                        "selected_candidate": replan_candidate,
                        "selected_index": replan_selected_index,
                        "program_id": replan_program_id,
                        "program_position": replan_program_position,
                        "target_program_id": target_anchor.program_id,
                        "consistent_with_target": (
                            replan_program_id == target_anchor.program_id
                        ),
                        "forced_alignment": replan_forced_alignment,
                        "condition": replanned_condition,
                        "previous_condition": target_anchor.condition,
                    }
                    restored_frame = policy_frame(observation, overall_task, env_preprocessor)
                    phase = CyclePhase.MBR_RETRY
                    _run_mbr_retry(
                        target_anchor,
                        restored_frame,
                        condition=replanned_condition,
                    )
                    phase = CyclePhase.IN_PROGRESS
                    active_attempt_chunks = 1
                    if success or episode_done or _budget_steps() >= max_steps:
                        _finalize_evaluator_images()
                        break
                else:
                    # Transit is not completion. In 9-D mode the current subtask
                    # remains active until the confirmed stop signal arrives.
                    # The 7-D path reaches this phase too, then uses its explicit
                    # fixed-horizon fallback below.
                    phase = CyclePhase.COMPLETE

            if phase == CyclePhase.COMPLETE:
                # 7-D 代理没有 learned stop 信号，官方用 stop 信号推进子任务。
                # 这里用子任务级物理完成判定作为代理 stop：按当前动作类型检查
                # 场景物理状态（抓稳/到位/释放），而不是绑定全局任务 success
                # （全局 success 中途恒为 False，proxy stop 永远不会触发）。
                proxy_stop_finish = False
                if active_anchor is not None:
                    try:
                        complete_evidence = _physical_evidence(active_anchor)
                        proxy_stop_finish = bool(
                            proxy_subtask_stop(complete_evidence)
                        )
                    except Exception:
                        proxy_stop_finish = False
                if proxy_stop_finish:
                    cycle_proxy_stop_finish_count += 1
                if (
                    action_record["stop_confirmed"]
                    or proxy_stop_finish
                    or (
                        signal_source == "missing_7d"
                        and active_attempt_chunks >= low_level_chunks_per_plan
                    )
                ):
                    _finish_active_subtask()
            elif (
                signal_source == "missing_7d"
                and active_attempt_chunks >= low_level_chunks_per_plan
            ):
                _finish_active_subtask()

    finally:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if frames:
                write_video(video_path, frames, env.metadata.get("render_fps", 80))
        finally:
            close_envs(envs)

    if len(chunks) != len(evaluator_images):
        raise RuntimeError(
            "Cycle episode produced misaligned semantic traces and evaluator images: "
            f"{len(chunks)} != {len(evaluator_images)}"
        )
    if termination_reason is None:
        termination_reason = "max_steps"
    tensors = _episode_tensors(action_chunks, actions, rewards, policy)
    cycle_self_target_request_count = sum(
        1 for event in cycle_trace if event.get("self_target_requested")
    )
    cycle_replan_count = sum(
        1 for event in recovery_events if event.get("replanned", {}).get("triggered")
    )
    episode_result = {
        "episode_index": episode_index,
        "init_state_id": episode_index,
        "task": overall_task,
        "success": bool(success),
        "termination_mode": JITRL_TERMINATION_MODE,
        "termination_reason": termination_reason,
        "steps": _budget_steps(),
        "policy_action_steps": len(actions),
        "cycle_rewind_steps": cycle_rewind_steps,
        "sum_reward": float(sum(rewards)),
        "max_reward": float(max(rewards, default=0.0)),
        "chunk_count": len(chunks),
        "high_level_step_count": len(chunks),
        "low_level_chunk_count": len(action_chunks),
        "video_path": str(video_path),
        "chunks": chunks,
        "cycle_enabled": True,
        "cycle_signal_mode": cycle_signal_mode,
        "cycle_trigger_mode": cycle_trigger_mode,
        "cycle_program": list(cycle_program_text),
        "cycle_budget_mode": CYCLE_BUDGET_MODE,
        "cycle_budget_max_steps": int(budget_max_steps),
        "cycle_retry_counts": dict(retry_counts),
        "cycle_rewind_mode": "reverse_robot_state_replay",
        "cycle_config_version": CYCLE_CONFIG_VERSION,
        "cycle_progress_threshold": CYCLE_PROGRESS_THRESHOLD,
        "cycle_proxy_progress_threshold": CYCLE_PROXY_PROGRESS_THRESHOLD,
        "cycle_check_after_low_level_chunks": CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS,
        "cycle_checks": cycle_check_count,
        "cycle_backtracks": cycle_backtrack_count,
        "cycle_vetoes": cycle_veto_count,
        "cycle_retries": cycle_retry_count,
        "cycle_max_retries": CYCLE_MAX_RETRIES,
        "cycle_self_target_request_count": cycle_self_target_request_count,
        "cycle_replan_count": cycle_replan_count,
        "cycle_proxy_stop_finish_count": cycle_proxy_stop_finish_count,
        "cycle_trace": cycle_trace,
        "recovery_events": recovery_events,
        "mbr_events": mbr_events,
        "mbr_hypothesis_count": sum(
            int(event["hypothesis_count"]) for event in mbr_events
        ),
        "cycle_wall_time_seconds": float(time.perf_counter() - cycle_wall_start),
    }
    return episode_result, episode_records, tensors, evaluator_images


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
    cycle_predictor=None,
    step_progress=None,
) -> tuple[dict, list[dict], dict[str, torch.Tensor], list[tuple[list, list]]]:
    if method in CYCLE_METHODS:
        return _run_episode_cycle(
            task_spec=task_spec,
            method=method,
            seed=seed,
            episode_index=episode_index,
            run_dir=run_dir,
            memory=memory,
            planner_model=planner_model,
            planner_processor=planner_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cycle_predictor=cycle_predictor,
            step_progress=step_progress,
        )
    budget_max_steps = evaluation_budget_max_steps(task_spec)
    envs, env, env_preprocessor, env_postprocessor = make_single_env(
        task_spec,
        episode_index,
        budget_max_steps=budget_max_steps,
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
                active_condition = _low_level_condition(
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

            for action in action_chunk[:POLICY_ACTION_STEPS]:
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
        "cycle_enabled": False,
        "cycle_budget_mode": CYCLE_BUDGET_MODE,
        "cycle_budget_max_steps": int(budget_max_steps),
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
        "budget_mode": CYCLE_BUDGET_MODE,
        "budget_max_steps": evaluation_budget_max_steps(task_spec),
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


def load_experiment_models(
    *, need_evaluator: bool, need_cycle_predictor: bool = False
) -> dict:
    """Load planner, policy and (optionally) evaluator once for reuse across runs.

    Models stay resident on the GPU across task/method/seed runs so the CLI does not
    repeatedly unload and reload them.
    """

    planner_model, planner_processor = load_jitrl_planner()
    policy, preprocessor, postprocessor = load_policy()
    evaluator = (
        QwenChunkEvaluator(planner_model, planner_processor) if need_evaluator else None
    )
    cycle_predictor = load_cycle_failure_predictor() if need_cycle_predictor else None
    return {
        "planner_model": planner_model,
        "planner_processor": planner_processor,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "evaluator": evaluator,
        "cycle_predictor": cycle_predictor,
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
        "cycle_predictor",
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
    init_state_start: int = 0,
) -> dict:
    """Run one task/method/seed pairing from an independent empty memory.

    When ``models`` (from :func:`load_experiment_models`) is provided, the resident
    model objects are reused and not unloaded when this run finishes; otherwise the
    models are loaded here and released in the ``finally`` block.
    """

    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}; got {method!r}")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if (
        isinstance(init_state_start, bool)
        or not isinstance(init_state_start, int)
        or init_state_start < 0
    ):
        raise ValueError("init_state_start must be a non-negative integer")
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
    cycle_predictor = None
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
            cycle_predictor = models.get("cycle_predictor")
            if method in MEMORY_METHODS and evaluator is None:
                raise RuntimeError(f"{method} method requires a Qwen evaluator")
            if method in CYCLE_METHODS and cycle_predictor is None:
                raise RuntimeError(f"{method} method requires a Cycle predictor")
        else:
            tqdm.write(
                f"[load] task={task_name} method={method} seed={seed}: "
                "loading Qwen planner"
            )
            planner_model, planner_processor = load_jitrl_planner()
            if method in MEMORY_METHODS:
                tqdm.write(
                    f"[load] task={task_name} method={method} seed={seed}: "
                    "configuring local Qwen evaluator"
                )
                evaluator = QwenChunkEvaluator(planner_model, planner_processor)
            if method in CYCLE_METHODS:
                tqdm.write(
                    f"[load] task={task_name} method={method} seed={seed}: "
                    f"configuring {JITRL_EVALUATOR_MODEL} Cycle predictor"
                )
                cycle_predictor = load_cycle_failure_predictor()
            tqdm.write(
                f"[load] task={task_name} method={method} seed={seed}: "
                "loading π₀.₅ policy"
            )
            policy, preprocessor, postprocessor = load_policy()
        tqdm.write(
            f"[ready] task={task_name} method={method} seed={seed}: models loaded"
        )

        for local_index, episode_index in enumerate(
            range(init_state_start, init_state_start + episodes)
        ):
            current_episode = episode_index
            max_steps = evaluation_budget_max_steps(task_spec)
            step_progress = tqdm(
                total=max_steps,
                desc=(
                    f"{task_name} {method} seed={seed} episode "
                    f"{local_index + 1}/{episodes} steps"
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
                    cycle_predictor=cycle_predictor,
                    step_progress=step_progress,
                )
            finally:
                step_progress.close()
            if method in MEMORY_METHODS:
                if evaluator is None:
                    raise RuntimeError(f"{method} method lost its Qwen evaluator")
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
            cycle_predictor = None
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
