"""Tests for CUDA Graph capture utility."""

from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from oceanscale.graph_capture import GraphCapture

wp.init()


@wp.kernel
def _add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    c[i] = a[i] + b[i]


@wp.kernel
def _scale(a: wp.array(dtype=float), s: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = a[i] * s


@pytest.fixture
def device() -> str:
    return "cuda:0"


@pytest.fixture
def arrays(device: str) -> tuple[wp.array, wp.array, wp.array]:
    n = 1024
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.full(n, 2.0, dtype=np.float32), device=device)
    c = wp.zeros(n, dtype=float, device=device)
    return a, b, c


def test_capture_creates_graph(arrays: tuple, device: str) -> None:
    a, b, c = arrays

    def step():
        wp.launch(_add, dim=len(a), inputs=[a, b, c], device=device)

    gc = GraphCapture(step, warmup_steps=2, device=device)
    assert gc.is_captured


def test_replay_matches_direct(arrays: tuple, device: str) -> None:
    a, b, c = arrays

    def step():
        wp.launch(_add, dim=len(a), inputs=[a, b, c], device=device)

    step()
    wp.synchronize()
    direct_result = c.numpy().copy()

    c.zero_()
    gc = GraphCapture(step, warmup_steps=2, device=device)
    c.zero_()
    gc.replay()
    wp.synchronize()
    graph_result = c.numpy()

    np.testing.assert_array_equal(direct_result, graph_result)


def test_replay_sees_inplace_writes(device: str) -> None:
    """Graph replays should see in-place array content changes."""
    n = 256
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)

    def step():
        wp.launch(_scale, dim=n, inputs=[a, 2.0, out], device=device)

    gc = GraphCapture(step, warmup_steps=2, device=device)

    gc.replay()
    wp.synchronize()
    np.testing.assert_allclose(out.numpy(), 2.0)

    wp.copy(dest=a, src=wp.array(np.full(n, 5.0, dtype=np.float32), device=device))
    gc.replay()
    wp.synchronize()
    np.testing.assert_allclose(out.numpy(), 10.0)


def test_multi_kernel_capture(device: str) -> None:
    """Capture a pipeline with multiple kernel launches."""
    n = 512
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.full(n, 3.0, dtype=np.float32), device=device)
    tmp = wp.zeros(n, dtype=float, device=device)
    out = wp.zeros(n, dtype=float, device=device)

    def pipeline():
        wp.launch(_add, dim=n, inputs=[a, b, tmp], device=device)
        wp.launch(_scale, dim=n, inputs=[tmp, 10.0, out], device=device)

    gc = GraphCapture(pipeline, warmup_steps=2, device=device)
    out.zero_()
    gc.replay()
    wp.synchronize()
    np.testing.assert_allclose(out.numpy(), 40.0)


@pytest.mark.gpu
def test_replay_performance(device: str) -> None:
    """Graph replay should be faster than direct launch."""
    n = 10000
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.ones(n, dtype=np.float32), device=device)
    c = wp.zeros(n, dtype=float, device=device)

    def step():
        wp.launch(_add, dim=n, inputs=[a, b, c], device=device)
        wp.launch(_scale, dim=n, inputs=[c, 2.0, c], device=device)

    gc = GraphCapture(step, warmup_steps=3, device=device)

    iters = 500
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    wp.synchronize()
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        gc.replay()
    wp.synchronize()
    t_graph = time.perf_counter() - t0

    assert t_graph < t_direct, f"Graph ({t_graph:.4f}s) should be faster than direct ({t_direct:.4f}s)"


@pytest.mark.gpu
def test_capture_with_newton_solver(device: str) -> None:
    """End-to-end: capture Newton solver step as CUDA graph."""
    import newton

    builder = newton.ModelBuilder()
    builder.add_body()
    builder.add_shape_box(body=0, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverSemiImplicit(model)
    s0 = model.state()
    s1 = model.state()

    dt = 0.01

    def step_and_copy_back():
        solver.step(s0, s1, None, None, dt)
        wp.copy(dest=s0.body_q, src=s1.body_q)
        wp.copy(dest=s0.body_qd, src=s1.body_qd)

    gc = GraphCapture(step_and_copy_back, warmup_steps=3, device=device)
    assert gc.is_captured

    pos_before = s0.body_q.numpy()[0, 2]
    gc.replay()
    wp.synchronize()
    pos_after = s0.body_q.numpy()[0, 2]
    assert pos_after < pos_before, f"Body should fall: {pos_after} < {pos_before}"
