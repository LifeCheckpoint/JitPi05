from pathlib import Path

DATASET_ID = "HuggingFaceVLA/libero"
EPISODE_INDEX = 0
QWEN_ID = "Qwen/Qwen3.5-4B"
PI05_ID = "lerobot/pi05_libero_base"
# 与 google/paligemma-3b-pt-224 使用相同 tokenizer，避免仅为 tokenizer 要求 gated repo 登录。
PI05_TOKENIZER_ID = "nnh-pbbb/paligemma-3b-pt-224"

OUTPUT_DIR = Path("artifacts")
MAX_PLAN_TOKENS = 20
TOP_K = 5
SEED = 7

# episode 0 的第一个人工高层步骤；切换 episode 时应同步修改。
ORACLE_SUBTASK = "pick up the white mug"
# 最小 logits 调制示例。研究时直接修改短语、bias，或替换 make_phrase_bias()。
BIAS_PHRASE = "move toward the white mug"
BIAS_VALUE = 8.0
