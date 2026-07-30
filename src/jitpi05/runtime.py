from __future__ import annotations

import importlib
import platform
import sys
from typing import Any


def inspect_gpu_runtime() -> dict[str, Any]:
    """Validate the Linux CUDA stack without downloading models or datasets."""
    if sys.platform != "linux":
        raise RuntimeError(
            "JitPi05 GPU execution requires Linux (Ubuntu 24.04 under WSL2 is supported)"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    if torch.version.cuda is None:
        raise RuntimeError("the installed PyTorch build has no CUDA runtime")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the active GPU does not support bfloat16")

    device = torch.device("cuda:0")
    left = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    right = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    result = left @ right
    if float(result[0, 0]) != 16.0:
        raise RuntimeError(
            "CUDA bfloat16 matrix multiplication returned an invalid result"
        )

    modules: dict[str, str] = {}
    for import_name, display_name in (
        ("bitsandbytes", "bitsandbytes"),
        ("fla", "flash-linear-attention"),
        ("triton", "triton"),
        ("lerobot", "lerobot"),
    ):
        module = importlib.import_module(import_name)
        modules[display_name] = str(getattr(module, "__version__", "installed"))

    from fla.utils import _device as fla_device

    if fla_device.device_platform != "cuda" or not fla_device.IS_NVIDIA:
        raise RuntimeError(
            "flash-linear-attention/Triton did not initialize its NVIDIA CUDA backend"
        )

    properties = torch.cuda.get_device_properties(device)
    return {
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bfloat16": True,
        "fla_backend": fla_device.device_platform,
        "modules": modules,
    }
