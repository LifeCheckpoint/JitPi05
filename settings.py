from pathlib import Path

DATASET_ID = "HuggingFaceVLA/libero"
EPISODE_INDEX = 0
QWEN_ID = "Qwen/Qwen3.5-4B"
JITRL_QWEN_ID = "Qwen/Qwen3.5-9B"
PI05_ID = "lerobot/pi05_libero_base"
SIM_PI05_ID = "lerobot/pi05-libero"
# 与 google/paligemma-3b-pt-224 使用相同 tokenizer，避免仅为 tokenizer 要求 gated repo 登录。
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

OUTPUT_DIR = Path("artifacts")
SIM_OUTPUT_DIR = OUTPUT_DIR / "sim_eval"
JITRL_OUTPUT_DIR = OUTPUT_DIR / "jitrl_eval_task79_seed17_gemini5"
MAX_PLAN_TOKENS = 20
TOP_K = 5
SEED = 7
SIM_EPISODES = 5
SIM_ACTION_STEPS = 10

# 默认按用户要求先运行带记忆方法，再运行无记忆基线。
JITRL_METHODS = ("jitrl", "static")
JITRL_SEEDS = (17,)
JITRL_EPISODES = 25
JITRL_HIGH_LEVEL_STEPS = 20
JITRL_PLANNER_RETRIES = 7
JITRL_EVALUATOR_RETRIES = 3
JITRL_CANDIDATES = 5
JITRL_HISTORY_SIZE = 2
JITRL_TOP_K = 10
JITRL_GAMMA = 0.95
JITRL_BETA = 0.25
JITRL_TEMPERATURE = 0.8
JITRL_EXPLORATION_RATE = 0.05
JITRL_UCB_ALPHA = 5.0
JITRL_MEMORY_BASE_LOGIT = 0.0
JITRL_EVALUATOR_SCORE_SCALE = 1.0 / 3.0
JITRL_TERMINAL_SUCCESS_BONUS = 1.0
JITRL_MAX_PLAN_TOKENS = 256
JITRL_MAX_EVALUATOR_TOKENS = 192
JITRL_EVALUATOR_BACKEND = "gemini_pydantic_ai"
JITRL_EVALUATOR_MODEL = "gemini-3.6-flash"
# 服务商给出的 Gemini generate-content collection URL。适配器会为
# PydanticAI/GoogleProvider 推导 API root，最终仍请求此 collection 下的模型。
JITRL_GEMINI_BASE_URL = "https://api.ikuncode.cc/v1beta/models"
JITRL_GEMINI_CREDENTIALS_PATH = Path(".secrets/gemini.json")
JITRL_GEMINI_TIMEOUT_SECONDS = 120.0
JITRL_LOGIT_CALIBRATION = "mean_center_v1"
JITRL_REWARD_VERSION = "gemini36flash_step_score_div3_terminal_plus1_v2"
JITRL_BOOTSTRAP_SAMPLES = 10_000
JITRL_BOOTSTRAP_CONFIDENCE = 0.95
JITRL_TASK = {
    "name": "libero_90_task79",
    "suite": "libero_90",
    "task_id": 79,
    "max_steps": 400,
    "zero_shot": True,
}

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

# episode 0 的第一个人工高层步骤；切换 episode 时应同步修改。
ORACLE_SUBTASK = "pick up the white mug"
# 最小 logits 调制示例。研究时直接修改短语、bias，或替换 make_phrase_bias()。
BIAS_PHRASE = "move toward the white mug"
BIAS_VALUE = 8.0
