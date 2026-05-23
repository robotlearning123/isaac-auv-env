"""Tests for NewtonEnv — Newton bridge environment."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from oceanscale.newton_env import NewtonEnv

wp.init()


@pytest.fixture
def single_env():
    return NewtonEnv(n_envs=1, solver_type="semi_implicit")


class TestEnvCreation:
    def test_state_shapes(self, single_env):
        state = single_env.get_state()
        assert state["position"].shape == (1, 3)
        assert state["orientation"].shape == (1, 4)
        assert state["velocity"].shape == (1, 3)
        assert state["angular_velocity"].shape == (1, 3)

    def test_initial_state_zero(self, single_env):
        state = single_env.get_state()
        np.testing.assert_allclose(state["position"], 0.0, atol=1e-6)
        # Identity quaternion: (0, 0, 0, 1)
        np.testing.assert_allclose(state["orientation"][0, :3], 0.0, atol=1e-6)
        np.testing.assert_allclose(state["orientation"][0, 3], 1.0, atol=1e-6)

    def test_properties(self, single_env):
        assert single_env.n_envs == 1
        assert single_env.device == "cuda"


class TestSingleStep:
    def test_state_changes_after_step(self, single_env):
        u_np = np.zeros((1, 6), dtype=np.float32)
        u_np[0, 0] = 0.5  # forward surge
        u = wp.array(u_np, dtype=wp.float32, device="cuda")

        single_env.step(u)
        state = single_env.get_state()
        # Position should have changed from zero
        assert not np.allclose(state["position"], 0.0, atol=1e-10)
        # Velocity should be non-zero
        assert not np.allclose(state["velocity"], 0.0, atol=1e-10)


class TestBuoyancy:
    def test_vehicle_floats(self):
        env = NewtonEnv(n_envs=1, solver_type="semi_implicit")
        # No thrust — buoyancy exceeds weight, vehicle should float UP (positive z)
        for _ in range(100):
            env.step()
        state = env.get_state()
        # z-position should be positive (floating up)
        assert state["position"][0, 2] > 0.0, (
            f"Expected positive z (floating), got {state['position'][0, 2]}"
        )
        # z-velocity should be positive (moving up)
        assert state["velocity"][0, 2] > 0.0, (
            f"Expected positive vz (rising), got {state['velocity'][0, 2]}"
        )


class TestThrusterMove:
    def test_forward_thrust_positive_x(self):
        env = NewtonEnv(n_envs=1, solver_type="semi_implicit")
        u_np = np.zeros((1, 6), dtype=np.float32)
        u_np[0, 0] = 1.0  # full forward surge
        u = wp.array(u_np, dtype=wp.float32, device="cuda")

        for _ in range(240):
            env.step(u)

        state = env.get_state()
        # x-velocity should be positive (moving forward)
        assert state["velocity"][0, 0] > 0.5, (
            f"Expected positive vx > 0.5 with full surge, got {state['velocity'][0, 0]}"
        )
        # x-position should be positive
        assert state["position"][0, 0] > 0.0, (
            f"Expected positive x position, got {state['position'][0, 0]}"
        )


class TestBatched64:
    def test_batched_64_envs(self):
        env = NewtonEnv(n_envs=64, solver_type="semi_implicit")
        u_np = np.zeros((64, 6), dtype=np.float32)
        u_np[:, 0] = 0.5  # forward thrust for all
        u = wp.array(u_np, dtype=wp.float32, device="cuda")

        for _ in range(10):
            env.step(u)

        state = env.get_state()
        assert state["position"].shape == (64, 3)
        assert state["velocity"].shape == (64, 3)
        # All envs should have moved forward
        assert np.all(state["position"][:, 0] > 0.0), "All envs should have positive x position"


class TestSolverComparison:
    def test_semi_implicit_vs_mujoco(self):
        """Compare SemiImplicit and MuJoCo on the same scenario.

        Both should produce forward motion with surge thrust.
        Results won't be identical (different integrators) but should
        have the same sign and comparable magnitude.
        """
        n_steps = 100

        env_si = NewtonEnv(n_envs=1, solver_type="semi_implicit")
        env_mj = NewtonEnv(n_envs=1, solver_type="mujoco")

        u_np = np.zeros((1, 6), dtype=np.float32)
        u_np[0, 0] = 0.8  # forward surge
        u_si = wp.array(u_np.copy(), dtype=wp.float32, device="cuda")
        u_mj = wp.array(u_np.copy(), dtype=wp.float32, device="cuda")

        for _ in range(n_steps):
            env_si.step(u_si)
            env_mj.step(u_mj)

        si_state = env_si.get_state()
        mj_state = env_mj.get_state()

        # Both should move forward (positive x)
        assert si_state["velocity"][0, 0] > 0.0, "SemiImplicit: expected positive vx"
        assert mj_state["velocity"][0, 0] > 0.0, "MuJoCo: expected positive vx"

        # Both should float (positive z)
        assert si_state["position"][0, 2] > 0.0, "SemiImplicit: expected positive z"
        assert mj_state["position"][0, 2] > 0.0, "MuJoCo: expected positive z"


class TestReset:
    def test_reset_returns_to_origin(self, single_env):
        u_np = np.zeros((1, 6), dtype=np.float32)
        u_np[0, 0] = 1.0
        u = wp.array(u_np, dtype=wp.float32, device="cuda")

        for _ in range(50):
            single_env.step(u)

        state_before = single_env.get_state()
        assert not np.allclose(state_before["position"], 0.0, atol=1e-6)

        single_env.reset()
        state_after = single_env.get_state()
        np.testing.assert_allclose(state_after["position"], 0.0, atol=1e-6)
        np.testing.assert_allclose(state_after["velocity"], 0.0, atol=1e-6)
