"""RLinf π0.5 低层策略客户端 adapter。

在独立 openpi 环境（Python 3.11）中运行服务端推理进程，通过 stdin/stdout 的
JSON 行协议通信。本模块运行在当前项目（Python 3.12）中，负责子进程生命周期、
图像编码与请求/响应序列化，并向 rollout 暴露与本地 LeRobot 策略等价的接口。

协议与服务端见 [`scripts/rlinf_pi05_server.py`](../../scripts/rlinf_pi05_server.py)。
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SERVER_PYTHON = _PROJECT_ROOT / "third_party" / "openpi" / ".venv" / "bin" / "python"
_DEFAULT_SERVER_SCRIPT = _PROJECT_ROOT / "scripts" / "rlinf_pi05_server.py"
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "artifacts" / "models" / "rlinf-pi05-libero130-fullshot-sft"

# openpi pi05_libero 的维度：
# - action_horizon（低层预测步数）= 10
# - action_dim（环境动作维度）= 7
# - 模型内部 action_dim = 32（与 LeRobot pi05 的 max_action_dim 一致）。
CHUNK_SIZE = 10
ACTION_DIM = 7
MODEL_ACTION_DIM = 32


def _encode_jpeg(array_rgb: np.ndarray, quality: int = 95) -> str:
    image = Image.fromarray(np.ascontiguousarray(array_rgb))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RlinfPi05Client:
    """与 RLinf π0.5 服务端通信的客户端。

    输入图像为 RGB HxWx3 uint8；噪声为 (action_horizon, action_dim) 或 None。
    """

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike | None = None,
        server_python: str | os.PathLike | None = None,
        server_script: str | os.PathLike | None = None,
    ) -> None:
        self.checkpoint_dir = str(checkpoint_dir or _DEFAULT_CHECKPOINT)
        self.server_python = str(server_python or _DEFAULT_SERVER_PYTHON)
        self.server_script = str(server_script or _DEFAULT_SERVER_SCRIPT)

        self.chunk_size = CHUNK_SIZE
        self.max_action_dim = ACTION_DIM

        command = [
            self.server_python,
            self.server_script,
            self.checkpoint_dir,
        ]
        env = dict(os.environ)
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        self._proc = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if line == "READY":
                return
            if line:
                raise RuntimeError(f"unexpected server startup output: {line}")

    def reset(self) -> None:
        """no-op：无状态客户端。"""

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except OSError:
                pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def _request(self, payload: dict) -> dict:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"rlinf server error: {response['error']}")
            return response
        raise RuntimeError("rlinf server closed unexpectedly")

    def predict(
        self,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
        noise: np.ndarray | None = None,
    ) -> torch.Tensor:
        """执行一次推理，返回反归一化 action chunk，shape ``(chunk_size, action_dim)``。

        参数:
            image: 外部相机 RGB HxWx3 uint8。
            wrist_image: 腕部相机 RGB HxWx3 uint8。
            state: 7 维 proprioceptive state（eef_pos + axisangle + gripper）。
            prompt: 低层语言条件。
            noise: 可选 flow-matching 初始噪声，shape ``(action_horizon, action_dim)``。
        """
        payload: dict = {
            "image": _encode_jpeg(image),
            "wrist_image": _encode_jpeg(wrist_image),
            "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
            "prompt": prompt,
        }
        if noise is not None:
            payload["noise"] = np.asarray(noise, dtype=np.float32).tolist()
        response = self._request(payload)
        actions = np.asarray(response["actions"], dtype=np.float32)
        return torch.from_numpy(actions)  # (action_horizon, action_dim)

    def __enter__(self) -> RlinfPi05Client:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class IdentityProcessor:
    """替代 LeRobot preprocessor/postprocessor 的 identity 处理。

    RLinf 后端直接返回反归一化 action chunk，无需 LeRobot 的归一化/反归一化
    链路；rollout 仍会调用 ``reset()``，因此需要提供该方法。
    """

    def __call__(self, value):  # noqa: ANN001
        return value

    def reset(self) -> None:
        pass


def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    """robosuite 的四元数转轴角（与 RLinf libero_eval 一致）。"""
    quat = torch.as_tensor(quat, dtype=torch.float32)
    if float(quat[3]) > 1.0:
        quat = quat / torch.norm(quat)
    den = torch.sqrt(1.0 - quat[3] ** 2)
    if float(den) < 1e-6:
        return torch.zeros(3, dtype=torch.float32)
    return quat[:3] * 2.0 * torch.acos(quat[3].clamp(-1.0, 1.0)) / den


class RlinfPi05PolicyAdapter:
    """把 :class:`RlinfPi05Client` 包装成 rollout 期望的 LeRobot policy 接口。

    与本地 ``PI05Policy`` 的区别：
    - ``predict_action_chunk`` 直接返回反归一化 action chunk（服务端已完成
      unnormalize），配套的 postprocessor 应为 identity。
    - 输入直接从 LeRobot ``frame`` 提取图像/state/prompt，绕过 LeRobot
      preprocessor（归一化、tokenize）与 OpenPI 的输入命名差异。
    """

    def __init__(self, client: RlinfPi05Client) -> None:
        self.client = client
        self._state_fallback_count = 0
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            # max_action_dim 与 LeRobot pi05 一致为 32，保证 coordinate_flow_noise
            # 生成的 flow-matching 噪声同为 32 维全随机（前 7 维对应环境动作，
            # 后 25 维为模型内部 padding 维），与之前 libero10 实验的噪声协议一致。
            max_action_dim=MODEL_ACTION_DIM,
            output_features={"action": SimpleNamespace(shape=(ACTION_DIM,))},
        )

    def reset(self) -> None:
        """no-op：服务端无跨 episode 状态。"""

    @staticmethod
    def _image_from_frame(frame: dict, key: str) -> np.ndarray:
        """从 LeRobot frame 提取 RGB HxWx3 uint8 图像。"""
        tensor = frame[key][0]  # [3, H, W] float32 [0,1]
        array = (
            tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0)
        ).to(torch.uint8).numpy()
        return np.ascontiguousarray(array)

    @staticmethod
    def _state_from_frame(frame: dict) -> np.ndarray:
        """从 rollout frame 构造 RLinf/OpenPI 的 8 维 LIBERO state。

        ``LiberoEnv`` 的 ``env_preprocessor`` 已经把原始 nested
        ``robot_state`` 整理为 flat ``observation.state``，其顺序与官方
        OpenPI LIBERO evaluator 一致：eef_pos(3) + axisangle(3) +
        gripper_qpos(2)。必须优先读取这个 flat key；旧版仅查 nested key
        会异常后静默返回全零 state，导致策略看起来像随机权重。
        """
        flat_state = frame.get("observation.state")
        if flat_state is not None:
            state = torch.as_tensor(flat_state, dtype=torch.float32).reshape(-1)
            if state.numel() >= 8 and torch.isfinite(state[:8]).all():
                return state[:8].detach().cpu().numpy().astype(np.float32)

        try:
            robot_state = frame["observation.state.robot_state"]
            eef_pos = robot_state["eef"]["pos"][0]  # [3]
            quat = robot_state["eef"]["quat"][0]  # [4]
            axisangle = _quat2axisangle(quat)  # [3]
            gripper = robot_state["gripper"]["qpos"][0]  # [2]
            state = torch.cat([eef_pos, axisangle, gripper], dim=-1)
            return state.detach().cpu().numpy().astype(np.float32)
        except (KeyError, IndexError, TypeError, RuntimeError, ValueError):
            raise ValueError(
                "RLinf adapter could not extract the required 8-D LIBERO state; "
                f"available frame keys={sorted(frame.keys())}"
            ) from None

    def predict_action_chunk(
        self,
        frame: dict,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回反归一化 action chunk，shape ``[1, chunk_size, action_dim]``。"""
        image = self._image_from_frame(frame, "observation.images.image")
        wrist = self._image_from_frame(frame, "observation.images.image2")
        state = self._state_from_frame(frame)
        prompt = frame["task"]
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0]

        noise_np = None
        if noise is not None:
            noise_np = noise.detach().cpu().numpy().astype(np.float32)

        actions = self.client.predict(image, wrist, state, prompt, noise_np)
        return actions.unsqueeze(0)  # [1, chunk_size, action_dim]

    def close(self) -> None:
        self.client.close()
