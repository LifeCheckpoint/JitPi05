"""RLinf π0.5 低层策略推理服务（在 openpi 独立 venv 中运行）。

通过 stdin/stdout 的 JSON 行协议提供推理。客户端逐行发送请求，服务端逐行
返回响应，便于当前项目（Python 3.12 + LeRobot）通过子进程调用。

请求（JSON 对象，单行）::

    {
        "image": "<jpeg base64>",        # 外部相机 RGB，256x256
        "wrist_image": "<jpeg base64>",  # 腕部相机 RGB，256x256
        "state": [8 floats],             # eef_pos(3) + axisangle(3) + gripper_qpos(2)
        "prompt": "task language",
        "noise": [[10 x 7 floats]]       # 可选，显式 flow-matching 初始噪声
    }

响应（JSON 对象，单行）::

    {"actions": [[10 x 7 floats]]}       # 反归一化后的动作 chunk

用法（在 third_party/openpi 目录内）::

    .venv/bin/python ../../scripts/rlinf_pi05_server.py \\
        ../../artifacts/models/rlinf-pi05-libero130-fullshot-sft
"""

from __future__ import annotations

import base64
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def load_policy(checkpoint_dir: str, num_steps: int = 10):
    from openpi.policies import policy_config
    from openpi.training import config as _config

    config = _config.get_config("pi05_libero")
    return policy_config.create_trained_policy(
        config, checkpoint_dir, sample_kwargs={"num_steps": num_steps}
    )


def _decode_jpeg(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.encode("ascii"))
    arr = np.frombuffer(data, dtype=np.uint8)
    # cv2 返回 BGR，转为 RGB。
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image")
    return image[..., ::-1].copy()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: rlinf_pi05_server.py <checkpoint_dir>", file=sys.stderr)
        return 2
    checkpoint_dir = os.path.abspath(sys.argv[1])

    print(f"[rlinf-server] loading policy from {checkpoint_dir}", file=sys.stderr, flush=True)
    policy = load_policy(checkpoint_dir)
    print("[rlinf-server] READY", file=sys.stderr, flush=True)
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as error:
            print(json.dumps({"error": f"bad request json: {error}"}), flush=True)
            continue

        try:
            image = _decode_jpeg(req["image"])
            wrist = _decode_jpeg(req["wrist_image"])
            # The client sends images after LeRobot's LiberoProcessorStep has
            # already applied the official LIBERO 180-degree flip. Do not flip
            # here again: doing so would cancel the training-time orientation
            # transform and make the policy act on an OOD camera view.
            image = np.ascontiguousarray(image)
            wrist = np.ascontiguousarray(wrist)

            example = {
                "observation/state": np.asarray(req["state"], dtype=np.float32),
                "observation/image": image,
                "observation/wrist_image": wrist,
                "prompt": req["prompt"],
            }

            noise = None
            if req.get("noise") is not None:
                noise = np.asarray(req["noise"], dtype=np.float32)
                # openpi π0.5 模型内部 action_dim=32（LIBERO 环境维度为 7，输出
                # 时截取前 7 维）。客户端传入的噪声已是 32 维（与 LeRobot pi05
                # 的 max_action_dim 一致，全随机）；此处保留 7 维兜底 pad，以兼容
                # 旧调用方。
                model_action_dim = 32
                if noise.shape[-1] < model_action_dim:
                    # 支持 (H, D) 与 (1, H, D) 等任意前置 batch 维度，尾部 pad。
                    padded_shape = noise.shape[:-1] + (model_action_dim,)
                    padded = np.zeros(padded_shape, dtype=np.float32)
                    padded[..., : noise.shape[-1]] = noise
                    noise = padded

            result = policy.infer(example, noise=noise)
            actions = np.asarray(result["actions"], dtype=np.float32)
            print(json.dumps({"actions": actions.tolist()}), flush=True)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"error": str(error)}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
