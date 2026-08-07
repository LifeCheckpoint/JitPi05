"""集中配置：模型、路径、模拟、JitRL 学习器、CycleVLA-lite 包装与评测任务。

所有常量按功能域分组。方法集合由 ``JITRL_METHODS`` / ``CYCLE_JITRL_METHODS``
派生为 ``CYCLE_METHODS`` / ``ALL_METHODS``；LIBERO-90 的难度筛选与正式评测
设计集中在一个 ``frozen`` 数据类 ``CYCLE_EVALUATION`` 中，避免散落的扁平常量。
"""

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
CYCLE_JITRL_METHODS = (
    "static-free",
    "jitrl-free",
    "static-free-cycle",
    "jitrl-free-cycle",
)
# Cycle 包装方法 = 后缀为 "-cycle" 的方法；全部方法 = 默认方法 + Cycle 方法。
CYCLE_METHODS = tuple(method for method in CYCLE_JITRL_METHODS if method.endswith("-cycle"))
ALL_METHODS = tuple(dict.fromkeys((*JITRL_METHODS, *CYCLE_METHODS)))

# =========================================================================
# JitRL 学习器
# =========================================================================
JITRL_SEEDS = (17,)
JITRL_EPISODES = 10
JITRL_HIGH_LEVEL_STEPS = 40  # 高层决策总步数（须能被 SIM_ACTION_STEPS 整除）
JITRL_PLANNER_RETRIES = 7  # 高层规划重试（free 与固定工作区统一）
JITRL_EVALUATOR_RETRIES = 10  # 每个 chunk 的 Gemini 评估重试（指数退避）
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
CYCLE_CONFIG_VERSION = "cyclevla_lite_zero_shot_v1"
CYCLE_PROGRESS_THRESHOLD = 0.75  # 代理进度门控（3/4 chunk）
CYCLE_CHECK_AFTER_LOW_LEVEL_CHUNKS = 3  # 每次尝试的第 N 个 chunk 后触发检查
CYCLE_MAX_RETRIES = 3  # 每个锚点的回溯/重试上限
# 为 Cycle 回溯预留的统一步数预算：所有方法共用同一 max_steps =
# 任务官方上限 + CYCLE_RETRY_TOTAL_BUDGET，保证回退链有足够步数执行完，
# 且不因预算不同而破坏配对公平。单次回溯实际成本约 40 步，120 留三倍裕量。
CYCLE_RETRY_BUDGET_PER_BACKTRACK = 120
CYCLE_RETRY_TOTAL_BUDGET = CYCLE_MAX_RETRIES * CYCLE_RETRY_BUDGET_PER_BACKTRACK
CYCLE_MBR_HYPOTHESES = 8  # MBR 采样假设数
CYCLE_MBR_ACTION_STEPS = 10  # MBR 轨迹特征使用的前缀动作步数
CYCLE_MBR_DELTA_DIMS = 6  # MBR 累积轨迹的平移/旋转维度
# 可接受的回溯低置信集合（transit-default 的召回调节）。默认含 medium 以激活
# 回溯；若误回溯（harm）过高可收紧为 ("low",)。
CYCLE_VLM_BACKTRACK_LIKELIHOODS = ("low", "medium")
# 夹爪闭合的 qpos 阈值（两指 qpos 之和）。None 表示不启用 gripper 物理失败判断
# （该阈值依赖夹爪型号，需在真实环境校准后再启用）。
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

# JitRL 默认评测任务池 = LIBERO-90 预注册候选池。正式评测前先用 static-free
# 在筛选状态范围做难度筛选，再用筛选出的难任务面板做四方法正式比较。
# 任务描述由环境在运行时提供（env.call("task_description")），此处留空占位。
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
