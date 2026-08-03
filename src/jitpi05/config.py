from jitpi05.paths import artifact_root, gemini_credentials_path

DATASET_ID = "HuggingFaceVLA/libero"
EPISODE_INDEX = 0
QWEN_ID = "Qwen/Qwen3.5-4B"
JITRL_QWEN_ID = "Qwen/Qwen3.5-2B"
PI05_ID = "lerobot/pi05_libero_base"
SIM_PI05_ID = "lerobot/pi05-libero"
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

OUTPUT_DIR = artifact_root()
SIM_OUTPUT_DIR = OUTPUT_DIR / "sim_eval"
JITRL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "jitrl_eval_libero10_long_seed17_qwen2b_workspace_v2_no_stop_diagnostic"
)
MAX_PLAN_TOKENS = 20
TOP_K = 5
SEED = 7
SIM_EPISODES = 5
SIM_ACTION_STEPS = 10

JITRL_METHODS = ("jitrl", "static")
JITRL_SEEDS = (17,)
JITRL_EPISODES = 10
JITRL_HIGH_LEVEL_STEPS = 30
JITRL_PLANNER_RETRIES = 7
JITRL_EVALUATOR_RETRIES = 5
JITRL_ACTION_WORKSPACE_VERSION = "libero_semantic_actions_9_compact_binding_v2"
JITRL_TERMINATION_MODE = "environment_only_no_stop_v1"
JITRL_HISTORY_SIZE = 2
JITRL_TOP_K = 10
JITRL_GAMMA = 0.95
JITRL_BETA = 0.40
JITRL_TEMPERATURE = 0.8
JITRL_EXPLORATION_RATE = 0.05
JITRL_UCB_ALPHA = 5.0
JITRL_EVALUATOR_SCORE_SCALE = 1.0 / 3.0
JITRL_TERMINAL_SUCCESS_BONUS = 1.0
JITRL_MAX_PLAN_TOKENS = 768
JITRL_MAX_EVALUATOR_TOKENS = 192
JITRL_EVALUATOR_BACKEND = "gemini_pydantic_ai"
JITRL_EVALUATOR_MODEL = "gemini-3.6-flash"
JITRL_GEMINI_BASE_URL = "https://api.ikuncode.cc/v1beta/models"
JITRL_GEMINI_CREDENTIALS_PATH = gemini_credentials_path()
JITRL_GEMINI_TIMEOUT_SECONDS = 120.0
JITRL_LOGIT_CALIBRATION = "raw_qwen_fixed_workspace_v1"
JITRL_REWARD_VERSION = "gemini36flash_positive_step_score_div3_terminal_plus1_v3"
JITRL_BOOTSTRAP_SAMPLES = 10_000
JITRL_BOOTSTRAP_CONFIDENCE = 0.95

# The published LeRobot pi0.5-LIBERO result includes the standard LIBERO-10
# long-horizon suite. Keep descriptions explicit so config import does not
# initialize the LIBERO benchmark package.
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

# Use the complete LIBERO-10 long-horizon suite rather than mixing short and
# long tasks. This keeps the benchmark inside the published fine-tuning
# distribution while avoiding a hand-picked cross-suite task panel.
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

# episode 0 的第一个人工高层步骤；切换 episode 时应同步修改。
ORACLE_SUBTASK = "pick up the white mug"
# 最小 logits 调制示例。研究时直接修改短语、bias，或替换 make_phrase_bias()。
BIAS_PHRASE = "move toward the white mug"
BIAS_VALUE = 8.0
