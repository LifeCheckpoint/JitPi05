from jitpi05.paths import artifact_root, gemini_credentials_path

DATASET_ID = "HuggingFaceVLA/libero"
EPISODE_INDEX = 0
QWEN_ID = "Qwen/Qwen3.5-4B"
JITRL_QWEN_ID = "Qwen/Qwen3.5-4B"
PI05_ID = "lerobot/pi05_libero_base"
SIM_PI05_ID = "lerobot/pi05-libero"
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

OUTPUT_DIR = artifact_root()
SIM_OUTPUT_DIR = OUTPUT_DIR / "sim_eval"
JITRL_OUTPUT_DIR = OUTPUT_DIR / "jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3"
MAX_PLAN_TOKENS = 20
TOP_K = 5
SEED = 7
SIM_EPISODES = 5
SIM_ACTION_STEPS = 10

JITRL_METHODS = ("jitrl", "static")
JITRL_SEEDS = (17,)
JITRL_EPISODES = 15
JITRL_HIGH_LEVEL_STEPS = 30
JITRL_PLANNER_RETRIES = 7
JITRL_EVALUATOR_RETRIES = 3
JITRL_CANDIDATES = 5
JITRL_HISTORY_SIZE = 2
JITRL_TOP_K = 10
JITRL_GAMMA = 0.95
JITRL_BETA = 0.40
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
JITRL_GEMINI_BASE_URL = "https://api.ikuncode.cc/v1beta/models"
JITRL_GEMINI_CREDENTIALS_PATH = gemini_credentials_path()
JITRL_GEMINI_TIMEOUT_SECONDS = 120.0
JITRL_LOGIT_CALIBRATION = "mean_center_v1"
JITRL_REWARD_VERSION = "gemini36flash_positive_step_score_div3_terminal_plus1_v3"
JITRL_BOOTSTRAP_SAMPLES = 10_000
JITRL_BOOTSTRAP_CONFIDENCE = 0.95
JITRL_TASKS = (
    {
        "name": "libero_90_task18",
        "suite": "libero_90",
        "task_id": 18,
        "description": "put the frying pan on the stove",
        "max_steps": 400,
        "zero_shot": True,
    },
    {
        "name": "libero_90_task53",
        "suite": "libero_90",
        "task_id": 53,
        "description": "pick up the orange juice and put it in the basket",
        "max_steps": 400,
        "zero_shot": True,
    },
    {
        "name": "libero_90_task59",
        "suite": "libero_90",
        "task_id": 59,
        "description": "pick up the tomato sauce and put it in the tray",
        "max_steps": 400,
        "zero_shot": True,
    },
    {
        "name": "libero_90_task69",
        "suite": "libero_90",
        "task_id": 69,
        "description": "put the chocolate pudding to the left of the plate",
        "max_steps": 400,
        "zero_shot": True,
    },
    {
        "name": "libero_90_task79",
        "suite": "libero_90",
        "task_id": 79,
        "description": (
            "pick up the book and place it in the left compartment of the caddy"
        ),
        "max_steps": 400,
        "zero_shot": True,
    },
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

# episode 0 的第一个人工高层步骤；切换 episode 时应同步修改。
ORACLE_SUBTASK = "pick up the white mug"
# 最小 logits 调制示例。研究时直接修改短语、bias，或替换 make_phrase_bias()。
BIAS_PHRASE = "move toward the white mug"
BIAS_VALUE = 8.0
