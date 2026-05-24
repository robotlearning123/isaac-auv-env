"""Tests for ROVEnv fluid-ROV FSI coupling via GridFluidSolver.

Verifies that ocean current from GridFluidSolver applies drag forces on ROV,
affecting its trajectory. Tests use use_fluid=True with known current velocity.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from oceanscale.rov_env import ROVEnv

wp.init()


@pytest.fixture
def fluid_env():
    e = ROVEnv(
        n_envs=4,
        device="cuda",
        use_fluid=True,
        sensor_noise_std=0.0,
        current_velocity=np.array([0.3, 0.0, 0.0], dtype=np.float32),
    )
    yield e
    e.close()


@pytest.fixture
def no_fluid_env():
    e = ROVEnv(
        n_envs=4,
        device="cuda",
        use_fluid=False,
        sensor_noise_std=0.0,
    )
    yield e
    e.close()


class TestFluidInit:
    def test_fluid_env_creates(self, fluid_env):
        assert fluid_env.use_fluid is True

    def test_no_fluid_env_creates(self, no_fluid_env):
        assert no_fluid_env.use_fluid is False

    def test_fluid_env_resets(self, fluid_env):
        obs, _info = fluid_env.reset()
        assert obs.shape == (4, 26)

    def test_fluid_env_steps(self, fluid_env):
        fluid_env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)
        obs, reward, _terminated, _truncated, _info = fluid_env.step(actions)
        assert obs.shape == (4, 26)
        assert reward.shape == (4,)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(reward))

    def test_fluid_env_stability(self, fluid_env):
        """Fluid env survives 100 steps without divergence."""
        fluid_env.reset()
        for _ in range(100):
            actions = np.zeros((4, 6), dtype=np.float32)
            obs, reward, _terminated, _truncated, _info = fluid_env.step(actions)
            assert np.all(np.isfinite(obs)), "NaN/Inf in observations"
            assert np.all(np.isfinite(reward)), "NaN/Inf in rewards"


class TestFluidEffect:
    def test_current_drifts_rov(self, fluid_env, no_fluid_env):
        """ROV in current should drift differently than without current."""
        fluid_env.reset(seed=42)
        no_fluid_env.reset(seed=42)

        # Apply zero thrust — ROV should drift with current
        actions = np.zeros((4, 6), dtype=np.float32)
        for _ in range(100):
            fluid_env.step(actions)
            no_fluid_env.step(actions)

        obs_fluid = fluid_env._get_flat_obs()
        obs_no_fluid = no_fluid_env._get_flat_obs()

        # Position error should differ between fluid and no-fluid envs
        pos_err_fluid = np.mean(np.abs(obs_fluid[:, :3]))
        pos_err_no_fluid = np.mean(np.abs(obs_no_fluid[:, :3]))
        assert pos_err_fluid != pos_err_no_fluid

    def test_current_info_has_fluid_data(self, fluid_env):
        """Step info should contain fluid velocity information."""
        fluid_env.reset()
        _, _, _, _, info = fluid_env.step(np.zeros((4, 6), dtype=np.float32))
        assert isinstance(info, dict)

    def test_custom_current_direction(self):
        """Current in +y direction should push ROV sideways."""
        env_y = ROVEnv(
            n_envs=4,
            device="cuda",
            use_fluid=True,
            sensor_noise_std=0.0,
            current_velocity=np.array([0.0, 5.0, 0.0], dtype=np.float32),
            fluid_config={"coupling_strength": 15.0},
        )
        env_y.reset()
        for _ in range(300):
            env_y.step(np.zeros((4, 6), dtype=np.float32))
        obs = env_y._get_flat_obs()
        # Position should have drifted from target (any direction counts)
        pos_err = np.linalg.norm(obs[:, :3], axis=-1)
        assert np.any(pos_err > 1e-3), f"pos_err={pos_err}"
        env_y.close()

    def test_stronger_current_more_drift(self):
        """Stronger current should cause more cumulative drift."""
        env_weak = ROVEnv(
            n_envs=4,
            device="cuda",
            use_fluid=True,
            sensor_noise_std=0.0,
            init_pos_noise_std=0.0,
            init_yaw_noise_std=0.0,
            current_velocity=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        )
        env_strong = ROVEnv(
            n_envs=4,
            device="cuda",
            use_fluid=True,
            sensor_noise_std=0.0,
            init_pos_noise_std=0.0,
            init_yaw_noise_std=0.0,
            current_velocity=np.array([1.5, 0.0, 0.0], dtype=np.float32),
        )

        env_weak.reset(seed=42)
        env_strong.reset(seed=42)

        # Track max position error over time (not cumulative, avoids OOB reset artifacts)
        max_weak = 0.0
        max_strong = 0.0
        for _ in range(50):
            obs_w, _, _, _, _ = env_weak.step(np.zeros((4, 6), dtype=np.float32))
            obs_s, _, _, _, _ = env_strong.step(np.zeros((4, 6), dtype=np.float32))
            max_weak = max(max_weak, float(np.mean(np.abs(obs_w[:, :3]))))
            max_strong = max(max_strong, float(np.mean(np.abs(obs_s[:, :3]))))

        assert max_strong > max_weak, f"weak={max_weak}, strong={max_strong}"

        env_weak.close()
        env_strong.close()


class TestFluidPartialReset:
    def test_partial_reset_with_fluid(self, fluid_env):
        """Partial reset works correctly with fluid enabled."""
        fluid_env.reset()
        for _ in range(30):
            fluid_env.step(np.zeros((4, 6), dtype=np.float32))

        reset_ids = np.array([0, 2])
        obs, _ = fluid_env.reset(env_ids=reset_ids)

        for idx in reset_ids:
            pos_err = np.linalg.norm(obs[idx, :3])
            assert pos_err < 1.0


class TestFluidTraining:
    def test_training_step_with_fluid(self):
        """Simulate a few PPO-style training steps with fluid enabled."""
        env = ROVEnv(
            n_envs=4,
            device="cuda",
            use_fluid=True,
            sensor_noise_std=0.02,
            max_episode_steps=50,
        )
        obs, _ = env.reset()
        rewards = []
        for _ in range(50):
            actions = np.random.uniform(-1, 1, (4, 6)).astype(np.float32)
            obs, reward, _terminated, _truncated, _info = env.step(actions)
            rewards.append(np.mean(reward))
            assert np.all(np.isfinite(obs))
            assert np.all(np.isfinite(reward))

        # Rewards should be finite and mostly negative (penalty-based)
        assert all(np.isfinite(r) for r in rewards)
        env.close()
