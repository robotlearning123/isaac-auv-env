"""Environment validation tests — verify the stack itself, before any sim code."""

from __future__ import annotations

import sys

import pytest


def test_python_version() -> None:
    assert sys.version_info >= (3, 12), f"need Python 3.12+, got {sys.version_info}"
    assert sys.version_info < (3, 14), f"max Python 3.13.x, got {sys.version_info}"


def test_oceanscale_importable() -> None:
    import oceanscale

    assert oceanscale.__version__ == "0.0.1"


def test_numpy() -> None:
    import numpy as np

    a = np.arange(12).reshape(3, 4)
    assert a.sum() == 66


def test_scipy() -> None:
    import scipy

    assert scipy.__version__ >= "1.13"


def test_torch_import() -> None:
    import torch

    assert torch.__version__ >= "2.7"


@pytest.mark.gpu
def test_torch_cuda_available() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA not available — check driver / wheel"
    assert torch.cuda.device_count() >= 1
    name = torch.cuda.get_device_name(0)
    assert "RTX 5090" in name or "Blackwell" in name or "RTX" in name, f"got {name}"


def test_warp_import() -> None:
    import warp as wp

    assert wp.__version__ >= "1.13"


@pytest.mark.gpu
def test_warp_gpu_devices() -> None:
    import warp as wp

    wp.init()
    devices = wp.get_devices()
    cuda_devices = [d for d in devices if d.is_cuda]
    assert len(cuda_devices) >= 1, f"no CUDA devices found, only: {devices}"


def test_newton_import() -> None:
    import newton

    assert newton.__version__ >= "1.2"


def test_usd_import() -> None:
    from pxr import Usd

    assert Usd.GetVersion() >= (0, 25, 11)
