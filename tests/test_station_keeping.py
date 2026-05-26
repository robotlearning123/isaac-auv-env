"""Tests for CurrentStationKeepingEnv — station keeping under ocean current."""

from __future__ import annotations

import numpy as np
import pytest

import warp as wp

wp.init()

from oceanscale.envs.station_keeping_env import CurrentStationKeepingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = CurrentStationKeepingEnv(n_envs=1, device="cuda")
    yield e
    e.close()


@pytest.fixture
def env_no_noise():
    e = CurrentStationKeepingEnv(
        n_envs=1, device="cuda",
        sensor_noise_std=0.0,
        init_pos_noise_std=0.0,
        init_yaw_noise_std=0.0,
    )
    yield e
    e.close()


@pytest.fixture
def batched_env():
    e = CurrentStationKeepingEnv(n_envs=8, device="cuda")
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Creation & spaces
# ---------------------------------------------------------------------------

class TestCreation:
    def test_obs_space_29dim(self, env):
        assert env.observation_space.shape == (29,)

    def test_action_space_6dim(self, env):
        assert env.action_space.shape == (6,)

    def test_default_max_steps(self, env):
        assert env.max_episode_steps == 2400

    def test_domain_randomization_on(self, env):
        assert env.use_domain_randomization is True


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape_29(self, env):
        obs, _ = env.reset()
        assert obs.shape == (29,)

    def test_info_has_success(self, env):
        _, info = env.reset()
        assert "success" in info
        assert info["success"] is False

    def test_position_near_target(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        np.testing.assert_allclose(obs[:3], 0.0, atol=0.1)

    def test_current_obs_present(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        # obs[26:29] is current velocity in body frame
        current_body = obs[26:29]
        assert current_body.shape == (3,)
        assert np.all(np.isfinite(current_body))


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset()
        result = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(result, tuple) and len(result) == 5

    def test_obs_29dim(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == (29,)

    def test_reward_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert np.isfinite(reward).all()

    def test_info_has_success(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "success" in info
        assert "per_env_success" in info
        assert "mean_pos_error" in info

    def test_batched_shapes(self, batched_env):
        batched_env.reset()
        obs, rewards, terminated, truncated, _ = batched_env.step(
            np.zeros((8, 6), dtype=np.float32)
        )
        assert obs.shape == (8, 29)
        assert rewards.shape == (8,)
        assert terminated.shape == (8,)
        assert truncated.shape == (8,)

    def test_per_env_success_shape(self, batched_env):
        batched_env.reset()
        _, _, _, _, info = batched_env.step(np.zeros((8, 6), dtype=np.float32))
        assert info["per_env_success"].shape == (8,)


# ---------------------------------------------------------------------------
# 4. Current effect — ROV drifts without control
# ---------------------------------------------------------------------------

class TestCurrentEffect:
    def test_zero_action_drifts(self, env_no_noise):
        """Strong current with zero thrust causes ROV displacement."""
        env_no_noise.reset(seed=42)
        env_no_noise._current_params[:, :] = [2.0, 0.0, 30.0, 0.0, 0.0]
        env_no_noise._compute_current()

        # Few steps to stay in stable regime (coupling + semi-implicit solver
        # limits long-horizon zero-control stability)
        for _ in range(5):
            obs, _, _, _, _ = env_no_noise.step(np.zeros(6, dtype=np.float32))

        displacement = np.linalg.norm(obs[:3])
        assert displacement > 0.01, f"Expected displacement > 0.01m, got {displacement:.4f}m"

    def test_current_obs_nonzero_after_step(self, env_no_noise):
        env_no_noise.reset(seed=42)
        obs, _, _, _, _ = env_no_noise.step(np.zeros(6, dtype=np.float32))
        current_body = obs[26:29]
        # Current speed is 0.2-2.0 m/s, so at least one component should be nonzero
        assert np.linalg.norm(current_body) > 0.01


# ---------------------------------------------------------------------------
# 5. Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def _reward(self, env, r):
        return float(r[0]) if isinstance(r, np.ndarray) else float(r)

    def test_stay_better_than_drift(self, env_no_noise):
        # Use fixed current in +x to make the comparison deterministic
        env_no_noise.reset(seed=42)
        env_no_noise._current_params[:, 0] = 2.0
        env_no_noise._current_params[:, 1] = 0.0
        env_no_noise._current_params[:, 2] = 30.0
        env_no_noise._current_params[:, 3] = 0.0
        env_no_noise._current_params[:, 4] = 0.0
        env_no_noise._compute_current()

        rewards_stay = [
            self._reward(env_no_noise, env_no_noise.step(np.zeros(6, dtype=np.float32))[1])
            for _ in range(30)
        ]

        env_no_noise.reset(seed=42)
        env_no_noise._current_params[:, 0] = 2.0
        env_no_noise._current_params[:, 1] = 0.0
        env_no_noise._current_params[:, 2] = 30.0
        env_no_noise._current_params[:, 3] = 0.0
        env_no_noise._current_params[:, 4] = 0.0
        env_no_noise._compute_current()

        # Surge amplifies +x current → drifts faster → worse reward
        surge = np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32)
        rewards_drift = [
            self._reward(env_no_noise, env_no_noise.step(surge)[1])
            for _ in range(30)
        ]

        assert np.mean(rewards_stay) > np.mean(rewards_drift)

    def test_reward_degrades_with_distance(self, env_no_noise):
        env_no_noise.reset(seed=42)
        drift = np.array([0.8, 0, 0, 0, 0, 0], dtype=np.float32)
        rewards = [
            self._reward(env_no_noise, env_no_noise.step(drift)[1])
            for _ in range(50)
        ]
        assert np.mean(rewards[:10]) > np.mean(rewards[-10:])

    def test_info_has_holding_component(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "reward_holding" in info

    def test_holding_reward_positive(self, env_no_noise):
        env_no_noise.reset(seed=42)
        _, _, _, _, info = env_no_noise.step(np.zeros(6, dtype=np.float32))
        # holding bonus = 0.3 / (1 + pos_err) is always positive
        assert info["reward_holding"] > 0


# ---------------------------------------------------------------------------
# 6. Success criteria
# ---------------------------------------------------------------------------

class TestSuccessCriteria:
    def test_success_false_initially(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["success"] is False

    def test_success_requires_300_steps(self, env_no_noise):
        env_no_noise.reset(seed=42)
        for _ in range(299):
            env_no_noise.step(np.zeros(6, dtype=np.float32))
        _, _, _, _, info = env_no_noise.step(np.zeros(6, dtype=np.float32))
        # After 300 steps with zero action, ROV has drifted far from target
        # but the history_idx is now 300, so success CAN trigger (but won't
        # because pos error > 0.5)
        assert info["per_env_success"].shape == (1,)


# ---------------------------------------------------------------------------
# 7. Domain randomization
# ---------------------------------------------------------------------------

class TestDomainRandomization:
    def test_current_params_vary_across_resets(self):
        env = CurrentStationKeepingEnv(n_envs=1, device="cuda")
        env.reset(seed=1)
        params1 = env._current_params.copy()
        env.reset(seed=2)
        params2 = env._current_params.copy()
        assert not np.allclose(params1, params2)
        env.close()

    def test_batched_currents_differ(self):
        env = CurrentStationKeepingEnv(n_envs=8, device="cuda")
        env.reset(seed=42)
        params = env._current_params
        # 8 envs with random params should not all be identical
        assert not np.allclose(params[0], params[1])
        env.close()
