import sys

import pytest


@pytest.mark.gpu
@pytest.mark.integration
def test_supported_gpu_runtime() -> None:
    if sys.platform != "linux":
        pytest.skip("the supported CUDA runtime is Linux-only")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not exposed to this Linux environment")

    from jitpi05.runtime import inspect_gpu_runtime

    report = inspect_gpu_runtime()
    assert report["cuda_runtime"]
    assert report["bfloat16"] is True
    assert report["fla_backend"] == "cuda"
    assert "RTX 4090" in report["device"]
