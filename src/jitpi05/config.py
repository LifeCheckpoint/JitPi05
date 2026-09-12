"""集中配置：模型、路径、模拟、JitRL 学习器、CycleVLA-lite 包装与评测任务。

所有常量按功能域分组。方法集合由 ``JITRL_METHODS`` / ``CYCLE_JITRL_METHODS``
派生为 ``CYCLE_METHODS`` / ``ALL_METHODS``；LIBERO-90 的难度筛选与正式评测
设计集中在一个 ``frozen`` 数据类 ``CYCLE_EVALUATION`` 中，避免散落的扁平常量。
"""

import os
from dataclasses import dataclass

from jitpi05.paths import artifact_root, gemini_credentials_path

# =========================================================================
# 模型与数据集
# =========================================================================
DATASET_ID = "HuggingFaceVLA/libero"  # 离线对比使用的 LIBERO 数据集
EPISODE_INDEX = 0  # 离线对比读取的固定 episode

# 高层语言规划器
QWEN_ID = "Qwen/Qwen3.5-4B"  # 离线/模拟规划器
JITRL_QWEN_ID = "Qwen/Qwen3.5-2B"  # JitRL 在线规划器
MAX_PLAN_TOKENS = 20  # 离线规划的生成 token 上限
TOP_K = 5  # 离线规划 logits top-k 分析

# 底层动作策略
PI05_ID = "lerobot/pi05_libero_base"  # 离线仿真策略
SIM_PI05_ID = "lerobot/pi05-libero"  # JitRL 与模拟共享的策略
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

# =========================================================================
# 输出目录
# =========================================================================
OUTPUT_DIR = artifact_root()
SIM_OUTPUT_DIR = OUTPUT_DIR / "sim_eval"
JITRL_OUTPUT_DIR = OUTPUT_DIR / "jitrl_cycle"

# =========================================================================
# 模拟与 rollout 基础
# =========================================================================
SEED = 7
SIM_EPISODES = 5
SIM_ACTION_STEPS = 10  # 每个底层 chunk 执行的动作步数

# =========================================================================
# 评测方法与派生能力
# =========================================================================
# 历史默认两方法；四方法二因素比较（Cycle × JitRL）通过 CLI --method 显式开启。
JITRL_METHODS = ("jitrl-free", "static-free")
# 无高层规划器的 raw-task 对照：冻结低层 VLA 在整个 episode 中始终接收
# LIBERO 原始任务描述。direct-cycle 只把确定性程序用于 Cycle 内部恢复状态，
# 不把程序节点文本作为低层语言条件。
DIRECT_METHODS = ("direct", "direct-cycle")
# 固定工作集（HarnessVLA 固定原语）方法：jitrl（在线记忆调制）/ static（无调制）。
JITRL_FIXED_METHODS = ("jitrl", "static")
CYCLE_JITRL_METHODS = (
    "static-free",
    "jitrl-free",
    "static-free-cycle",
    "jitrl-free-cycle",
)
# Cycle 包装方法 = 后缀为 "-cycle" 的方法。
CYCLE_METHODS = tuple(
    method
    for method in (*CYCLE_JITRL_METHODS, *DIRECT_METHODS)
    if method.endswith("-cycle")
)
# 全部方法 = 自由候选 + 固定工作集 + Cycle 包装方法。
# rollout.run_jitrl_experiment 使用本常量校验 method，因此必须包含固定工作集方法。
ALL_METHODS = tuple(
    dict.fromkeys(
        (*JITRL_METHODS, *JITRL_FIXED_METHODS, *DIRECT_METHODS, *CYCLE_METHODS)
    )
)
# CLI --method 可选全集与 ALL_METHODS 保持一致（含自由候选、固定工作集与 Cycle 方法）。
JITRL_CLI_METHODS = ALL_METHODS

# =========================================================================
# JitRL 学习器
# =========================================================================
JITRL_SEEDS = (17,)
JITRL_EPISODES = 10
JITRL_HIGH_LEVEL_STEPS = 40  # 高层决策总步数（须能被 SIM_ACTION_STEPS 整除）
JITRL_PLANNER_RETRIES = 7  # 高层规划重试（free 与固定工作区统一）
JITRL_EVALUATOR_RETRIES = 15  # 每个 chunk 的 Gemini 评估重试（指数退避）
JITRL_HISTORY_SIZE = 2
JITRL_TOP_K = 10
JITRL_GAMMA = 0.95
JITRL_BETA = 0.40
JITRL_TEMPERATURE = 0.8
JITRL_EXPLORATION_RATE = 0.05
JITRL_UCB_ALPHA = 5.0
JITRL_EVALUATOR_SCORE_SCALE = 1.0 / 3.0
JITRL_TERMINAL_SUCCESS_BONUS = 1.0
JITRL_BOOTSTRAP_SAMPLES = 10_000
JITRL_BOOTSTRAP_CONFIDENCE = 0.95

# Free-candidate 规划：Qwen 提出自由文本高层动作，memory 按 Jaccard 相似度匹配。
JITRL_FREE_CANDIDATES = 5
JITRL_FREE_ACTION_SIM_THRESHOLD = 0.6
JITRL_FREE_PLAN_TOKENS = 512
JITRL_MAX_PLAN_TOKENS = 768  # 固定工作区规划

# 版本标识（写入 artifact 用于可复现审计）
JITRL_ACTION_WORKSPACE_VERSION = "libero_semantic_actions_9_compact_binding_v2"
JITRL_TERMINATION_MODE = "environment_only_no_stop_v1"
JITRL_LOGIT_CALIBRATION = "raw_qwen_fixed_workspace_v1"
JITRL_REWARD_VERSION = "gemini36flash_positive_step_score_div3_terminal_plus1_v3"

# =========================================================================
# Gemini 评估器 / Cycle 失败预测器
# =========================================================================
JITRL_EVALUATOR_BACKEND = "gemini_pydantic_ai"
JITRL_EVALUATOR_MODEL = "gemini-3.6-flash"
JITRL_MAX_EVALUATOR_TOKENS = 192
JITRL_GEMINI_BASE_URL = "https://api.ikuncode.cc/v1beta/models"
JITRL_GEMINI_CREDENTIALS_PATH = gemini_credentials_path()
JITRL_GEMINI_TIMEOUT_SECONDS = 120.0

# =========================================================================
# CycleVLA-lite 包装（零训练推理包装，不改变 JitRL 学习边界）
# =========================================================================
CYCLE_CONFIG_VERSION = "cyclevla_lite_inference_p1_v4_smooth_rewind_unified_800"
# 论文 progress/stop 阈值；当前 stock 7-D 策略没有 learned signals，不能伪造。
CYCLE_PROGRESS_THRESHOLD = 0.90
CYCLE_PROXY_PROGRESS_THRESHOLD = 0.75
CYCLE_STOP_SIGNAL_THRESHOLD = 0.50
# 官方用 learned progress 信号触发 VLM 检查（子任务 ~90% 进度）；7D stock
# 策略没有该信号，这里退化为固定 chunk 数代理。open-loop 已对齐官方 5 步，
# 因此 6 个 chunk = 30 步，与旧 3 chunks × 10 步保持相同的物理检查窗口。
CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS = 6
CYCLE_SIGNAL_CONFIRM_CONSECUTIVE = 2
CYCLE_SIGNAL_CONFIRM_GAP = 2
CYCLE_MAX_RETRIES = 3
# Keep baseline and Cycle comparable while allowing enough time for bounded
# recovery attempts. The official LIBERO horizon remains 400; this experiment
# explicitly selects the same unified 800-step budget for both methods.
CYCLE_BUDGET_MODE = "unified_800"
CYCLE_UNIFIED_MAX_STEPS = 800
# 官方 evaluator 的 episode 预算 = TASK_MAX_STEPS * 1.5 + num_steps_wait。
# 选择 ``official_1p5x`` 时按该公式计算，与 CycleVLA 论文口径一致。
CYCLE_OFFICIAL_BUDGET_MULTIPLIER = 1.5
CYCLE_RETRY_BUDGET_PER_BACKTRACK = 120
CYCLE_RETRY_TOTAL_BUDGET = CYCLE_MAX_RETRIES * CYCLE_RETRY_BUDGET_PER_BACKTRACK
# Cycle recovery 诊断预算：接受一次有效回溯时立即增加预算，保证 rewind 本身
# 不会在原始 800 步上限处被截断。严格公平指标仍固定使用 CYCLE_UNIFIED_MAX_STEPS。
CYCLE_DYNAMIC_BUDGET_ENABLED = True
CYCLE_BACKTRACK_EXTRA_BUDGET = 400
CYCLE_MAX_EXTRA_BUDGET = 800
CYCLE_MAX_DYNAMIC_STEPS = CYCLE_UNIFIED_MAX_STEPS + CYCLE_MAX_EXTRA_BUDGET
CYCLE_MBR_HYPOTHESES = 8
# 官方 MBR 特征长度 = num_open_loop_steps * 6 = 5 * 6 = 30。
CYCLE_MBR_ACTION_STEPS = 5
CYCLE_MBR_DELTA_DIMS = 6
# 官方 ``mbr_use_failed_repulsion`` 默认 False；失败轨迹排斥仅作显式消融。
CYCLE_MBR_USE_FAILED_REPULSION = False
CYCLE_PREDICTOR_RETRIES = 3
# 同状态反事实恢复诊断：仅在已接受的回溯事件上执行；分支动作不写回正式 rollout。
# 步数上限 0 表示使用「剩余 episode 预算」，对齐官方「回溯后继续到 episode
# 结束」的语义（官方不存在独立短窗口）。>0 时截断，仅用于快速调试。
CYCLE_COUNTERFACTUAL_HORIZON_STEPS = 0
# 每个 episode 最多诊断多少个回溯事件。分支会各自跑完剩余预算，成本近似
# 「事件数 × 分支数 × 剩余步数」，因此默认 1 已足以获得事件级配对样本。
CYCLE_COUNTERFACTUAL_MAX_EVENTS_PER_EPISODE = 1
# 只对每第 N 个 episode 做反事实诊断（1 = 全部）。诊断成本与 episode 数线性
# 相关，抽样可在保持配对统计的同时按比例节省时间。
CYCLE_COUNTERFACTUAL_EPISODE_STRIDE = 1
# 可诊断分支子集。默认全开；速度受限时可裁剪为
# ("no_op", "full_cycle") 或 ("no_op", "target_only", "full_cycle")。
# 环境变量 ``JITPI05_COUNTERFACTUAL_BRANCHES`` 以逗号分隔覆盖。
CYCLE_COUNTERFACTUAL_BRANCHES: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get(
        "JITPI05_COUNTERFACTUAL_BRANCHES",
        "no_op,target_only,rewind_only,mbr_only,full_cycle",
    ).split(",")
    if name.strip()
)
# 关闭视频渲染可省去每控制步一次 ``env.render()``（CPU 侧开销，推理侧无关）。
# 环境变量 ``JITPI05_RECORD_VIDEO=false`` 可关闭。
CYCLE_RECORD_VIDEO = os.environ.get("JITPI05_RECORD_VIDEO", "true").strip().lower() not in (
    "0",
    "false",
    "no",
)
# 官方 ``num_steps_wait=10``：episode 起始用 dummy action 让物体稳定。
CYCLE_NUM_STEPS_WAIT = 10
# Paper-faithful anti-sycophancy default: only low likelihood can backtrack.
CYCLE_VLM_BACKTRACK_LIKELIHOODS = ("low",)
# None keeps simulator-side gripper inference disabled until calibrated.
CYCLE_GRIPPER_CLOSED_QPOS_THRESHOLD = None


@dataclass(frozen=True)
class CycleEvaluationDesign:
    """LIBERO-90 难度筛选与正式评测的预注册设计。

    字段被测试锁定；CLI 的 ``--screen`` 与 ``--init-state-start`` 直接消费本对象。
    """

    difficulty_screen_episodes: int = 10
    difficulty_screen_state_range: tuple[int, int] = (0, 9)
    formal_state_range: tuple[int, int] = (10, 29)
    # 天然包含可恢复 pick/place 或对齐错误的 LIBERO-90 候选任务。
    libero90_candidates: tuple[int, ...] = (19, 27, 53, 59, 60, 62, 69, 79)


CYCLE_EVALUATION = CycleEvaluationDesign()

# =========================================================================
# 评测任务定义
# =========================================================================
# LIBERO-90 官方 max_steps（TASK_SUITE_MAX_STEPS）。
LIBERO90_MAX_STEPS = 400
# LIBERO-10 官方 max_steps（与 published pi0.5-LIBERO 一致）。
LIBERO10_MAX_STEPS = 520

# 评测面板可选：环境变量 JITPI05_JITRL_PANEL=libero10 | libero90 | libero_pro
# （默认 libero90）。
# - libero10：完整 LIBERO-10 长任务 suite 全 10 任务，对应 ablation 报告的
#   HarnessVLA × JitRL 四格配置。
# - libero90：LIBERO-90 预注册候选池（难度筛选 + CycleVLA 正式比较）。
# - libero_pro：LIBERO-Pro 扰动面板（object/position/semantic/task 四维扰动，
#   基于 libero_10 的 10 个长任务），用于泛化 robustness 消融。
JITRL_BENCHMARK_PANEL = os.environ.get("JITPI05_JITRL_PANEL", "libero90").strip().lower()
if JITRL_BENCHMARK_PANEL not in ("libero10", "libero90", "libero_pro"):
    raise ValueError(
        "JITPI05_JITRL_PANEL must be 'libero10', 'libero90' or 'libero_pro'; "
        f"got {JITRL_BENCHMARK_PANEL!r}"
    )

# 低层策略后端：lerobot（默认，上一代 `lerobot/pi05-libero`）| rlinf
# （RLinf-Pi05-LIBERO-130-fullshot-SFT 远程 OpenPI 子进程）。用于在 libero_pro
# 面板下对照观察低层模型行为，排查异常时快速回退。
JITRL_LOW_LEVEL_BACKEND = os.environ.get("JITPI05_LOW_LEVEL_BACKEND", "lerobot").strip().lower()
if JITRL_LOW_LEVEL_BACKEND not in ("lerobot", "rlinf"):
    raise ValueError(
        "JITPI05_LOW_LEVEL_BACKEND must be 'lerobot' or 'rlinf'; "
        f"got {JITRL_LOW_LEVEL_BACKEND!r}"
    )

# OpenPI 官方 LIBERO evaluator 每次只执行预测 chunk 的前 5 步，然后重新
# 观测并推理（``num_open_loop_steps=5``，模型 chunk_size 仍为 10）。为与
# 官方协议对齐，LeRobot 与 RLinf 两个后端统一使用 5 步 open-loop。
RLINF_ACTION_STEPS = 5
POLICY_ACTION_STEPS = RLINF_ACTION_STEPS
# RLinf 的正式 JitRL 消融必须让高层选出的 semantic subtask 进入低层语言条件，
# 否则 memory advantage 的策略变化无法传导到环境动作。默认使用与 LeRobot 相同的
# ``Overall task + Current subtask`` 条件化格式。若只想复现 OpenPI 官方的 raw-task
# evaluator 协议，可显式设 ``JITPI05_RLINF_USE_RAW_TASK_PROMPT=true``；该兼容诊断
# 模式不应用于 JitRL 与 Static 的高低层消融比较。
RLINF_USE_RAW_TASK_PROMPT = os.environ.get(
    "JITPI05_RLINF_USE_RAW_TASK_PROMPT", "false"
).strip().lower() in ("1", "true", "yes", "on")

# 完整 LIBERO-10 长任务 suite 的任务描述（published pi0.5-LIBERO benchmark）。
_LIBERO10_TASK_DESCRIPTIONS = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)

if JITRL_BENCHMARK_PANEL == "libero10":
    JITRL_TASKS = tuple(
        {
            "name": f"libero_10_task{task_id}",
            "suite": "libero_10",
            "task_id": task_id,
            "description": _LIBERO10_TASK_DESCRIPTIONS[task_id],
            "max_steps": LIBERO10_MAX_STEPS,
            "zero_shot": False,
        }
        for task_id in range(10)
    )
elif JITRL_BENCHMARK_PANEL == "libero_pro":
    # LIBERO-Pro 扰动面板。注册扰动 suite 并生成任务描述；任务描述由环境在
    # 运行时提供（env.call("task_description")），此处留空占位。每个任务
    # 携带 "perturbation" 维度标记，便于报告按扰动维度聚合。
    from jitpi05.libero_pro import libero_pro_task_specs, register_libero_pro_suites

    register_libero_pro_suites()
    JITRL_TASKS = libero_pro_task_specs(
        base_suite="libero_10", max_steps=LIBERO10_MAX_STEPS
    )
else:
    # LIBERO-90 预注册候选池。任务描述由环境在运行时提供
    # （env.call("task_description")），此处留空占位。
    JITRL_TASKS = tuple(
        {
            "name": f"libero_90_task{task_id}",
            "suite": "libero_90",
            "task_id": task_id,
            "description": "",
            "max_steps": LIBERO90_MAX_STEPS,
            "zero_shot": True,
        }
        for task_id in CYCLE_EVALUATION.libero90_candidates
    )

SIM_TASKS = (
    {
        "name": "libero_object_task0",
        "suite": "libero_object",
        "task_id": 0,
        "oracle_subtask": "pick up the alphabet soup",
        "bias_phrase": "move toward the alphabet soup",
        "max_steps": 280,
        "zero_shot": False,
    },
    {
        "name": "libero_90_task79",
        "suite": "libero_90",
        "task_id": 79,
        "oracle_subtask": "pick up the book",
        "bias_phrase": "move toward the book",
        "max_steps": 400,
        "zero_shot": True,
    },
)

# =========================================================================
# 离线对比的调制示例
# =========================================================================
# episode 0 的第一个人工高层步骤；切换 episode 时应同步修改。
ORACLE_SUBTASK = "pick up the white mug"
# 最小 logits 调制示例。研究时直接修改短语、bias，或替换 make_phrase_bias()。
BIAS_PHRASE = "move toward the white mug"
BIAS_VALUE = 8.0
