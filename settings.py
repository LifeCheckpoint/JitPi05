from pathlib import Path

DATASET_ID = "HuggingFaceVLA/libero"
EPISODE_INDEX = 0
QWEN_ID = "Qwen/Qwen3.5-4B"
PI05_ID = "lerobot/pi05_libero_base"
SIM_PI05_ID = "lerobot/pi05-libero"
# 与 google/paligemma-3b-pt-224 使用相同 tokenizer，避免仅为 tokenizer 要求 gated repo 登录。
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

OUTPUT_DIR = Path("artifacts")
SIM_OUTPUT_DIR = OUTPUT_DIR / "sim_eval"
MAX_PLAN_TOKENS = 20
TOP_K = 5
SEED = 7
SIM_EPISODES = 5
SIM_ACTION_STEPS = 10

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
