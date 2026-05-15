"""Warp kernel smoke tests — exercise GPU + autograd on RTX 5090.

Smallest possible kernels:
- 1D quadratic drag F = -c * |v| * v
- autograd via wp.Tape vs analytic
- multi-launch consistency

These tests are the canary for the whole Warp pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp


@pytest.fixture(scope="module", autouse=True)
def _warp_init() -> None:
    wp.init()


@wp.kernel
def _quad_drag_1d(
    v: wp.array(dtype=wp.float32),
    c: wp.float32,
    f: wp.array(dtype=wp.float32),
) -> None:
    i = wp.tid()
    vi = v[i]
    f[i] = -c * wp.abs(vi) * vi


@pytest.mark.gpu
def test_quad_drag_forward_matches_numpy() -> None:
    n = 1024
    rng = np.random.default_rng(0)
    v_np = rng.normal(0.0, 2.0, size=n).astype(np.float32)
    c = 0.5

    v = wp.array(v_np, dtype=wp.float32, device="cuda")
    f = wp.zeros(n, dtype=wp.float32, device="cuda")

    wp.launch(_quad_drag_1d, dim=n, inputs=[v, c, f], device="cuda")
    wp.synchronize()

    f_got = f.numpy()
    f_ref = -c * np.abs(v_np) * v_np
    np.testing.assert_allclose(f_got, f_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_quad_drag_autograd_matches_analytic() -> None:
    """dF/dv = -2 c |v|, computed via wp.Tape, must match analytic."""
    n = 256
    rng = np.random.default_rng(1)
    v_np = rng.normal(0.0, 1.5, size=n).astype(np.float32)
    c = 0.3

    v = wp.array(v_np, dtype=wp.float32, device="cuda", requires_grad=True)
    f = wp.zeros(n, dtype=wp.float32, device="cuda", requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(_quad_drag_1d, dim=n, inputs=[v, c, f], device="cuda")

    # adj: arbitrary upstream gradient
    upstream = np.ones(n, dtype=np.float32)
    f.grad = wp.array(upstream, dtype=wp.float32, device="cuda")
    tape.backward()

    grad_got = v.grad.numpy()
    grad_ref = -2.0 * c * np.abs(v_np)  # d(-c|v|v)/dv = -2c|v|

    np.testing.assert_allclose(grad_got, grad_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_multi_launch_consistency() -> None:
    """Repeated launches give identical output (no nondeterminism in this kernel)."""
    n = 512
    rng = np.random.default_rng(2)
    v_np = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    v = wp.array(v_np, dtype=wp.float32, device="cuda")

    outs = []
    for _ in range(3):
        f = wp.zeros(n, dtype=wp.float32, device="cuda")
        wp.launch(_quad_drag_1d, dim=n, inputs=[v, 0.5, f], device="cuda")
        wp.synchronize()
        outs.append(f.numpy().copy())

    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[1], outs[2])


@pytest.mark.gpu
def test_device_is_rtx5090() -> None:
    devices = wp.get_devices()
    cuda_devices = [d for d in devices if d.is_cuda]
    assert len(cuda_devices) >= 1
    name = cuda_devices[0].name
    # Loose match — Blackwell / RTX 5090 / Ada / Hopper all OK for stack validation
    assert any(s in name for s in ("RTX", "Blackwell", "Hopper", "A100", "H100")), name
