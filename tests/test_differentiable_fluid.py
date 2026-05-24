"""Tests for differentiable 2D Navier-Stokes via wp.Tape."""

import numpy as np
import pytest
import warp as wp

wp.init()

from oceanscale.fluid.differentiable import DifferentiableFluidStep, warp_fluid_to_torch


@pytest.fixture
def solver():
    return DifferentiableFluidStep(grid_size=16, viscosity=0.01, device="cuda:0")


class TestTapeBasics:
    def test_tape_creation(self):
        tape = wp.Tape()
        assert tape is not None

    def test_array_requires_grad(self):
        a = wp.zeros(10, dtype=wp.float32, requires_grad=True, device="cuda:0")
        assert a.requires_grad


class TestForward:
    def test_forward_no_nan(self, solver):
        n = solver.n
        fx = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        fy = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        solver.forward(fx, fy, dt=0.01)
        vx = solver.vx.numpy()
        vy = solver.vy.numpy()
        assert np.all(np.isfinite(vx))
        assert np.all(np.isfinite(vy))

    def test_forward_with_force(self, solver):
        n = solver.n
        fx_np = np.zeros(n, dtype=np.float32)
        fx_np[n // 2] = 10.0
        fx = wp.array(fx_np, dtype=wp.float32, device="cuda:0", requires_grad=True)
        fy = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        solver.forward(fx, fy, dt=0.01)
        vx = solver.vx.numpy()
        assert np.max(np.abs(vx)) > 0.0

    def test_multi_step_stable(self, solver):
        n = solver.n
        fx = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        fy = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        for _ in range(5):
            solver.forward(fx, fy, dt=0.001)
        assert np.all(np.isfinite(solver.vx.numpy()))


class TestBackward:
    def test_backward_produces_gradients(self, solver):
        n = solver.n
        fx = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        fy = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        target_vx = wp.array(np.ones(n, dtype=np.float32) * 0.1, dtype=wp.float32, device="cuda:0")
        target_vy = wp.zeros(n, dtype=wp.float32, device="cuda:0")

        tape = wp.Tape()
        with tape:
            solver.forward(fx, fy, dt=0.01)
            loss = solver.compute_loss(target_vx, target_vy)

        tape.backward(loss)
        fx_grad = fx.grad.numpy()
        assert fx_grad is not None
        assert np.any(fx_grad != 0.0), "Force gradients should be nonzero"

    def test_loss_is_scalar(self, solver):
        n = solver.n
        fx = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        fy = wp.zeros(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        target = wp.zeros(n, dtype=wp.float32, device="cuda:0")

        tape = wp.Tape()
        with tape:
            solver.forward(fx, fy, dt=0.01)
            loss = solver.compute_loss(target, target)

        loss_val = float(loss.numpy()[0])
        assert np.isfinite(loss_val)
        assert loss.shape == (1,)


class TestOptimization:
    def test_gradient_descent_reduces_loss(self, solver):
        n = solver.n
        target_vx = np.zeros(n, dtype=np.float32)
        target_vx[n // 2 - 1: n // 2 + 2] = 0.5
        target_vy = np.zeros(n, dtype=np.float32)

        t_vx = wp.array(target_vx, dtype=wp.float32, device="cuda:0")
        t_vy = wp.array(target_vy, dtype=wp.float32, device="cuda:0")

        losses = solver.optimize_force(t_vx, t_vy, n_iters=10, lr=0.5, dt=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"


class TestTorchBridge:
    def test_torch_bridge_gradient_flow(self, solver):
        torch = pytest.importorskip("torch")
        n = solver.n
        fx_t = torch.zeros(n, device="cuda:0", requires_grad=True)
        fy_t = torch.zeros(n, device="cuda:0", requires_grad=True)

        tape = warp_fluid_to_torch(solver, fx_t, fy_t, dt=0.01)
        vx = solver.vx.numpy()
        assert np.all(np.isfinite(vx))
        assert tape is not None

    def test_torch_from_warp_zero_copy(self):
        torch = pytest.importorskip("torch")
        a = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cuda:0")
        t = wp.to_torch(a)
        assert t.shape == (3,)
        assert t.device.type == "cuda"
        np.testing.assert_array_equal(t.cpu().numpy(), [1.0, 2.0, 3.0])
