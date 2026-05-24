"""Version compatibility matrix — detect installed versions, check known issues,
report compatibility tier, and test version-specific features."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Version detection helpers
# ---------------------------------------------------------------------------

class StackVersions(NamedTuple):
    python: str
    cuda_runtime: str
    cuda_driver: str
    gpu_name: str
    torch: str
    torch_cuda: str
    warp: str
    newton: str
    cupy: str
    jax: str
    numpy: str
    scipy: str
    usd: str


def _detect_versions() -> StackVersions:
    """Detect all installed stack versions."""
    import numpy as np
    import scipy

    python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # CUDA runtime
    try:
        import torch
        torch_ver = torch.__version__
        torch_cuda = torch.version.cuda or "none"
        cuda_runtime = torch_cuda
    except ImportError:
        torch_ver = "not installed"
        torch_cuda = "none"
        cuda_runtime = "unknown"

    # CUDA driver via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
        parts = out.split(", ")
        cuda_driver = parts[0] if parts else "unknown"
        gpu_name = parts[1] if len(parts) > 1 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        cuda_driver = "unknown"
        gpu_name = "unknown"

    # Warp
    try:
        import warp as wp
        warp_ver = wp.__version__
    except ImportError:
        warp_ver = "not installed"

    # Newton
    try:
        import newton
        newton_ver = newton.__version__
    except ImportError:
        newton_ver = "not installed"

    # CuPy
    try:
        import cupy
        cupy_ver = cupy.__version__
    except ImportError:
        cupy_ver = "not installed"

    # JAX
    try:
        import jax
        jax_ver = jax.__version__
    except ImportError:
        jax_ver = "not installed"

    # USD
    try:
        from pxr import Usd
        usd_ver = ".".join(str(x) for x in Usd.GetVersion())
    except ImportError:
        usd_ver = "not installed"

    return StackVersions(
        python=python,
        cuda_runtime=cuda_runtime,
        cuda_driver=cuda_driver,
        gpu_name=gpu_name,
        torch=torch_ver,
        torch_cuda=torch_cuda,
        warp=warp_ver,
        newton=newton_ver,
        cupy=cupy_ver,
        jax=jax_ver,
        numpy=np.__version__,
        scipy=scipy.__version__,
        usd=usd_ver,
    )


def _ver_tuple(v: str) -> tuple[int, ...]:
    """Parse '1.13.0' → (1, 13, 0). Strips non-numeric suffixes."""
    return tuple(int(x) for x in re.match(r"[\d.]+", v).group().split("."))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Compatibility matrix definition
# ---------------------------------------------------------------------------

# Known incompatible combos: (condition_description, check_function)
KNOWN_ISSUES: list[tuple[str, callable]] = []


def _check_torch_cuda13_cu128(v: StackVersions) -> str | None:
    """PyTorch built for cu128 but CUDA runtime is 13.x — ABI mismatch risk."""
    if v.torch == "not installed":
        return None
    if "cu128" in v.torch_cuda and v.cuda_runtime.startswith("13"):
        return (
            f"PyTorch built for CUDA 12.8 (cu128) but runtime is CUDA {v.cuda_runtime}. "
            "May work via forward compat but kernel launches could fail. "
            "Rebuild with cu130 wheels for CUDA 13."
        )
    return None


def _check_warp_cuda13_old(v: StackVersions) -> str | None:
    """Warp < 1.9 does not support CUDA 13."""
    if v.warp == "not installed":
        return None
    if _ver_tuple(v.warp) < (1, 9) and v.cuda_runtime.startswith("13"):
        return f"Warp {v.warp} does not support CUDA 13. Upgrade to >= 1.9."
    return None


def _check_numpy2_torch_old(v: StackVersions) -> str | None:
    """NumPy 2.x with PyTorch < 2.3 causes ABI issues."""
    if v.torch == "not installed":
        return None
    if _ver_tuple(v.numpy) >= (2,) and _ver_tuple(v.torch) < (2, 3):
        return f"NumPy {v.numpy} + PyTorch {v.torch}: ABI incompatibility. Upgrade PyTorch >= 2.3."
    return None


def _check_python313_newton(v: StackVersions) -> str | None:
    """Newton 1.2.0 may have issues on Python 3.13 (not yet validated)."""
    if v.newton == "not installed":
        return None
    if _ver_tuple(v.python) >= (3, 13) and _ver_tuple(v.newton) <= (1, 2, 0):
        return (
            f"Newton {v.newton} on Python {v.python}: not validated. "
            "Check newton-physics/newton for 3.13 support status."
        )
    return None


KNOWN_ISSUES = [
    ("PyTorch cu128 + CUDA 13 mismatch", _check_torch_cuda13_cu128),
    ("Warp < 1.9 + CUDA 13", _check_warp_cuda13_old),
    ("NumPy 2 + old PyTorch ABI", _check_numpy2_torch_old),
    ("Python 3.13 + Newton 1.2.0", _check_python313_newton),
]

# Compatibility tiers
COMPAT_TIERS = {
    "T1_REFERENCE": "Python 3.12 + CUDA 12.8 + PyTorch 2.7 + Warp 1.13 + Newton 1.2 (reference platform)",
    "T2_FORWARD": "CUDA 13.x + PyTorch 2.12 + Warp 1.13 + Newton 1.2 (forward compat)",
    "T2_PY313": "Python 3.13 + CUDA 12.8 + same deps (Python forward compat)",
    "T3_COLAB": "Colab T4/L4 + whatever CUDA/PyTorch versions available",
    "T3_BLEEDING": "CUDA 13.2 + latest everything (bleeding edge)",
}


def _classify_tier(v: StackVersions) -> str:
    """Classify detected stack into a compatibility tier."""
    py = _ver_tuple(v.python)
    cuda = v.cuda_runtime

    if py[:2] == (3, 12) and cuda.startswith("12.8"):
        return "T1_REFERENCE"
    if cuda.startswith("13"):
        if _ver_tuple(v.cuda_runtime) >= (13, 2):
            return "T3_BLEEDING"
        return "T2_FORWARD"
    if py[:2] == (3, 13):
        return "T2_PY313"
    if "T4" in v.gpu_name or "L4" in v.gpu_name:
        return "T3_COLAB"
    return "T1_REFERENCE"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack_versions() -> StackVersions:
    return _detect_versions()


class TestVersionDetection:
    """Detect and report all installed versions."""

    def test_report_versions(self, stack_versions: StackVersions) -> None:
        v = stack_versions
        report = [
            "",
            "═══════════════════════════════════════════════",
            "  OceanScale Version Compatibility Report",
            "═══════════════════════════════════════════════",
            f"  Python:       {v.python}",
            f"  CUDA runtime: {v.cuda_runtime}",
            f"  CUDA driver:  {v.cuda_driver}",
            f"  GPU:          {v.gpu_name}",
            f"  PyTorch:      {v.torch} (CUDA {v.torch_cuda})",
            f"  Warp:         {v.warp}",
            f"  Newton:       {v.newton}",
            f"  CuPy:         {v.cupy}",
            f"  JAX:          {v.jax}",
            f"  NumPy:        {v.numpy}",
            f"  SciPy:        {v.scipy}",
            f"  USD:          {v.usd}",
            f"  Tier:         {_classify_tier(v)}",
            "═══════════════════════════════════════════════",
        ]
        print("\n".join(report))

    def test_all_core_deps_installed(self, stack_versions: StackVersions) -> None:
        v = stack_versions
        missing = []
        for name, ver in [
            ("torch", v.torch), ("warp", v.warp), ("newton", v.newton),
        ]:
            if ver == "not installed":
                missing.append(name)
        assert not missing, f"Missing core dependencies: {missing}"


class TestKnownIncompatibilities:
    """Check for known version incompatibilities."""

    @pytest.mark.parametrize("name,checker", KNOWN_ISSUES, ids=[n for n, _ in KNOWN_ISSUES])
    def test_no_known_issues(self, stack_versions: StackVersions, name: str, checker) -> None:
        issue = checker(stack_versions)
        if issue:
            pytest.xfail(f"[{name}] {issue}")


class TestCUDACompat:
    """CUDA-specific compatibility checks."""

    @pytest.mark.gpu
    def test_cuda_runtime_matches_torch(self, stack_versions: StackVersions) -> None:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        runtime_major = int(stack_versions.cuda_runtime.split(".")[0])
        torch_cuda_major = int(stack_versions.torch_cuda.split(".")[0])
        assert abs(runtime_major - torch_cuda_major) <= 1, (
            f"CUDA runtime {stack_versions.cuda_runtime} vs PyTorch CUDA {stack_versions.torch_cuda}: "
            "major version gap > 1, likely ABI issue"
        )

    @pytest.mark.gpu
    def test_warp_cuda_device_matches(self) -> None:
        import torch
        import warp as wp
        wp.init()
        warp_devices = [d for d in wp.get_devices() if d.is_cuda]
        torch_count = torch.cuda.device_count()
        assert len(warp_devices) == torch_count, (
            f"Warp sees {len(warp_devices)} CUDA devices, PyTorch sees {torch_count}"
        )

    @pytest.mark.gpu
    def test_cupy_cuda_matches(self) -> None:
        pytest.importorskip("cupy", reason="cupy is optional (install with [bench])")
        import cupy
        cupy_cuda = cupy.cuda.runtime.runtimeGetVersion()
        cuda_major = cupy_cuda // 1000
        cuda_minor = (cupy_cuda % 1000) // 10
        import torch
        torch_major = int(torch.version.cuda.split(".")[0])
        assert abs(cuda_major - torch_major) <= 1, (
            f"CuPy CUDA {cuda_major}.{cuda_minor} vs PyTorch CUDA {torch.version.cuda}"
        )


class TestCrossFrameworkInterop:
    """Test that data flows correctly between frameworks at installed versions."""

    @pytest.mark.gpu
    def test_torch_to_warp_tensor(self) -> None:
        import torch
        import warp as wp
        wp.init()
        t = torch.randn(64, 3, device="cuda:0")
        w = wp.from_torch(t)
        assert w.shape == (64, 3)
        t2 = wp.to_torch(w)
        assert torch.allclose(t, t2)

    @pytest.mark.gpu
    def test_torch_to_cupy(self) -> None:
        pytest.importorskip("cupy", reason="cupy is optional (install with [bench])")
        import cupy
        import torch
        t = torch.randn(64, 3, device="cuda:0")
        c = cupy.from_dlpack(t)
        assert c.shape == (64, 3)
        t2 = torch.from_dlpack(c)
        assert torch.allclose(t, t2)

    @pytest.mark.gpu
    def test_jax_to_torch(self) -> None:
        pytest.importorskip("jax", reason="jax is optional (install with [bench])")
        import jax
        import jax.numpy as jnp
        import torch
        j = jnp.ones((64, 3))
        t = torch.from_dlpack(j)
        assert t.shape == (64, 3)
        assert torch.allclose(t, torch.ones(64, 3, device=t.device))

    @pytest.mark.gpu
    def test_warp_newton_device_agreement(self) -> None:
        """Newton models should run on the same CUDA device Warp uses."""
        import newton
        import warp as wp
        wp.init()
        builder = newton.ModelBuilder()
        builder.add_body()
        model = builder.finalize(device="cuda:0")
        assert str(model.particle_q.device) == "cuda:0"


class TestVersionSpecificFeatures:
    """Test features that depend on specific version ranges."""

    @pytest.mark.gpu
    def test_warp_autograd(self) -> None:
        """wp.Tape autograd — available since Warp 0.10+."""
        import warp as wp
        wp.init()

        @wp.kernel
        def sq(x: wp.array(dtype=float), y: wp.array(dtype=float)):
            i = wp.tid()
            y[i] = x[i] * x[i]

        x = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda:0", requires_grad=True)
        y = wp.zeros(3, dtype=float, device="cuda:0", requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(sq, dim=3, inputs=[x, y], device="cuda:0")
        tape.backward(grads={y: wp.array([1.0, 1.0, 1.0], dtype=float, device="cuda:0")})
        grad = tape.gradients[x].numpy()
        assert abs(grad[0] - 2.0) < 1e-5
        assert abs(grad[1] - 4.0) < 1e-5

    @pytest.mark.gpu
    def test_newton_replicate(self) -> None:
        """ModelBuilder.replicate() — available since Newton 1.0+."""
        import newton
        scene = newton.ModelBuilder()
        template = newton.ModelBuilder()
        template.add_body()
        scene.replicate(template, world_count=4, spacing=(0.0, 0.0, 0.0))
        model = scene.finalize(device="cuda:0")
        assert model.body_count == 4

    @pytest.mark.gpu
    def test_newton_mujoco_solver(self) -> None:
        """SolverMuJoCo backend — available since Newton 1.0+ (MuJoCo-Warp)."""
        import newton
        import newton.solvers
        template = newton.ModelBuilder()
        body = template.add_body()
        template.add_shape_sphere(body, radius=0.1)
        template.joint_q = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        template.joint_qd = [0.0] * 6
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        scene = newton.ModelBuilder()
        scene.replicate(template, world_count=1, spacing=(0.0, 0.0, 0.0))
        model = scene.finalize(device="cuda:0")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = newton.solvers.SolverMuJoCo(model)
        solver.step(state_0, state_1, control, None, 1e-3)
