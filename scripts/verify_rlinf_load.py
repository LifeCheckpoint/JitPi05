"""验证 RLinf-Pi05-LIBERO-130-fullshot-SFT 可用 openpi pi05_libero 加载并单步推理。

在 openpi 独立环境（third_party/openpi/.venv）中运行：

    cd third_party/openpi
    uv run python ../../scripts/verify_rlinf_load.py \
        ../../artifacts/models/rlinf-pi05-libero130-fullshot-sft

用法：python verify_rlinf_load.py [checkpoint_dir]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


def main() -> int:
    from openpi.policies import policy_config
    from openpi.training import config as _config

    checkpoint_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "artifacts/models/rlinf-pi05-libero130-fullshot-sft"
    )
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        raise FileNotFoundError(
            f"model.safetensors not found under {checkpoint_dir}"
        )

    config = _config.get_config("pi05_libero")
    print(f"[verify] loading policy from {checkpoint_dir}")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("[verify] policy loaded OK")

    # 与 RLinf libero_eval.py 一致的输入：双相机 + 7 维状态 + prompt。
    # 图像 180° 旋转在真实 rollout 中处理；此处仅验证加载与单步推理形状。
    example = {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(
            256, size=(256, 256, 3), dtype=np.uint8
        ),
        "prompt": "put both the alphabet soup and the tomato sauce in the basket",
    }
    result = policy.infer(example)
    actions = np.asarray(result["actions"])
    print(f"[verify] action chunk shape: {actions.shape}")
    print(f"[verify] action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print("[verify] LOAD + INFER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
