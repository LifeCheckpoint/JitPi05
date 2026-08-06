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
CYCLE_MBR_HYPOTHESES = 8  # MBR 采样假设数
CYCLE_MBR_ACTION_STEPS = 10  # MBR 轨迹特征使用的前缀动作步数
CYCLE_MBR_DELTA_DIMS = 6  # MBR 累积轨迹的平移/旋转维度


@dataclass(frozen=True)
class CycleEvaluationDesign:
    """LIBERO-90 难度筛选与正式评测的预注册设计。

    字段当前被测试锁定但尚未接入 CLI/脚本；接入筛选与正式比较时应直接使用
    本对象，避免再次散落为扁平常量。
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
# 官方 LIBERO 任务描述（suite, max_steps, 描述列表）。导入时不初始化 LIBERO 包。
_STANDARD_LIBERO_TASKS = (
    (
        "libero_spatial",
        280,
        (
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "pick up the black bowl next to the ramekin and place it on the plate",
            "pick up the black bowl from table center and place it on the plate",
            "pick up the black bowl on the cookie box and place it on the plate",
            "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
            "pick up the black bowl on the ramekin and place it on the plate",
            "pick up the black bowl next to the cookie box and place it on the plate",
            "pick up the black bowl on the stove and place it on the plate",
            "pick up the black bowl next to the plate and place it on the plate",
            "pick up the black bowl on the wooden cabinet and place it on the plate",
        ),
    ),
    (
        "libero_object",
        280,
        (
            "pick up the alphabet soup and place it in the basket",
            "pick up the cream cheese and place it in the basket",
            "pick up the salad dressing and place it in the basket",
            "pick up the bbq sauce and place it in the basket",
            "pick up the ketchup and place it in the basket",
            "pick up the tomato sauce and place it in the basket",
            "pick up the butter and place it in the basket",
            "pick up the milk and place it in the basket",
            "pick up the chocolate pudding and place it in the basket",
            "pick up the orange juice and place it in the basket",
        ),
    ),
    (
        "libero_goal",
        300,
        (
            "open the middle drawer of the cabinet",
            "put the bowl on the stove",
            "put the wine bottle on top of the cabinet",
            "open the top drawer and put the bowl inside",
            "put the bowl on top of the cabinet",
            "push the plate to the front of the stove",
            "put the cream cheese in the bowl",
            "turn on the stove",
            "put the bowl on the plate",
            "put the wine bottle on the rack",
        ),
    ),
    (
        "libero_10",
        520,
        (
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
        ),
    ),
)

_STANDARD_LIBERO_TASK_LOOKUP = {
    suite: {
        task_id: description
        for task_id, description in enumerate(descriptions)
    }
    for suite, _, descriptions in _STANDARD_LIBERO_TASKS
}

# 使用完整的 LIBERO-10 长程套件（位于已发表微调分布内），避免手挑跨套件面板。
_JITRL_BENCHMARK_TASK_IDS = (("libero_10", tuple(range(10))),)

JITRL_TASKS = tuple(
    {
        "name": f"{suite}_task{task_id}",
        "suite": suite,
        "task_id": task_id,
        "description": _STANDARD_LIBERO_TASK_LOOKUP[suite][task_id],
        "max_steps": next(
            max_steps
            for configured_suite, max_steps, _ in _STANDARD_LIBERO_TASKS
            if configured_suite == suite
        ),
        "zero_shot": False,
    }
    for suite, task_ids in _JITRL_BENCHMARK_TASK_IDS
    for task_id in task_ids
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
