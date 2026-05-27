"""Integration tests for vec_env module — OceanScaleVecEnv.

Source: oceanscale/vec_env.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import warp as wp

from oceanscale.vec_env import OceanScaleVecEnv

wp.init()


# ── OceanScaleVecEnv tests (GPU required) ─────────────────────────────


class TestOceanScaleVecEnvCreation:
    @pytest.fixture
    def env(self):
        e = OceanScaleVecEnv(n_envs=4, max_episode_steps=100)
        yield e
        e.close()

    def test_num_envs(self, env):
        assert env.num_envs == 4

    def test_single_observation_space(self, env):
        assert env.single_observation_space.shape == (19,)

    def test_batch_observation_space(self, env):
        assert env.observation_space.shape == (4, 19)

    def test_single_action_space(self, env):
        assert env.single_action_space.shape == (6,)

    def test_batch_action_space(self, env):
        assert env.action_space.shape == (4, 6)

    def test_action_bounds(self, env):
        np.testing.assert_allclose(env.single_action_space.low, -1.0)
        np.testing.assert_allclose(env.single_action_space.high, 1.0)

    def test_metadata_autoreset(self, env):
        assert env.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP


class TestOceanScaleVecEnvReset:
    @pytest.fixture
    def env(self):
        e = OceanScaleVecEnv(n_envs=4, max_episode_steps=100)
        yield e
        e.close()

    def test_reset_shapes(self, env):
        obs, info = env.reset()
        assert obs.shape == (4, 19)
        assert isinstance(info, dict)

    def test_reset_clears_step_count(self, env):
        env.reset()
        actions = np.zeros((4, 6), dtype=np.float32)
        for _ in range(5):
            env.step(actions)

        env.reset()
        np.testing.assert_array_equal(env._step_count, 0)

    def test_reset_clears_prev_action(self, env):
        env.reset()
        env.step(np.ones((4, 6), dtype=np.float32) * 0.5)

        env.reset()
        np.testing.assert_array_equal(env._prev_action, 0.0)

    def test_reset_with_seed(self, env):
        obs, _info = env.reset(seed=42)
        assert obs.shape == (4, 19)


class TestOceanScaleVecEnvStep:
    @pytest.fixture
    def env(self):
        e = OceanScaleVecEnv(n_envs=4, max_episode_steps=100)
        e.reset()
        yield e
        e.close()

    def test_step_shapes(self, env):
        obs, reward, terminated, truncated, _info = env.step(np.zeros((4, 6), dtype=np.float32))
        assert obs.shape == (4, 19)
        assert reward.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

    def test_broadcast_single_action(self, env):
        action = np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs, *_ = env.step(action)
        assert obs.shape == (4, 19)

    def test_wrong_shape_raises(self, env):
        with pytest.raises(AssertionError, match="Expected actions shape"):
            env.step(np.zeros((3, 6), dtype=np.float32))

    def test_step_count_increments(self, env):
        actions = np.zeros((4, 6), dtype=np.float32)
        env.step(actions)
        np.testing.assert_array_equal(env._step_count, 1)

        env.step(actions)
        np.testing.assert_array_equal(env._step_count, 2)

    def test_reward_nonpositive(self, env):
        _obs, reward, _, _, _ = env.step(np.zeros((4, 6), dtype=np.float32))
        assert np.all(reward <= 0.0)


class TestOceanScaleVecEnvAutoReset:
    def test_truncation_at_max_steps(self):
        max_steps = 5
        env = OceanScaleVecEnv(n_envs=2, max_episode_steps=max_steps)
        env.reset()

        actions = np.zeros((2, 6), dtype=np.float32)
        for _ in range(max_steps):
            _obs, _reward, _terminated, truncated, _info = env.step(actions)

        assert np.all(truncated)
        env.close()

    def test_auto_reset_resets_step_count(self):
        max_steps = 3
        env = OceanScaleVecEnv(n_envs=2, max_episode_steps=max_steps)
        env.reset()

        actions = np.zeros((2, 6), dtype=np.float32)
        for _ in range(max_steps):
            env.step(actions)

        np.testing.assert_array_equal(env._step_count, 0)
        env.close()

    def test_auto_reset_clears_prev_action(self):
        max_steps = 3
        env = OceanScaleVecEnv(n_envs=2, max_episode_steps=max_steps)
        env.reset()

        actions = np.ones((2, 6), dtype=np.float32) * 0.5
        for _ in range(max_steps):
            env.step(actions)

        np.testing.assert_array_equal(env._prev_action, 0.0)
        env.close()


class TestOceanScaleVecEnvCallbacks:
    def test_custom_obs_fn(self):
        called = [False]

        def my_obs(env_state, prev_action):
            called[0] = True
            return np.zeros((4, 5), dtype=np.float32)

        env = OceanScaleVecEnv(n_envs=4, max_episode_steps=100, obs_fn=my_obs)
        obs, _ = env.reset()
        assert called[0]
        assert obs.shape == (4, 5)
        env.close()

    def test_custom_reward_fn(self):
        def my_reward(obs):
            return np.full(obs.shape[0], 42.0, dtype=np.float32)

        env = OceanScaleVecEnv(n_envs=4, max_episode_steps=100, reward_fn=my_reward)
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros((4, 6), dtype=np.float32))
        np.testing.assert_allclose(reward, 42.0)
        env.close()

    def test_custom_termination_fn(self):
        def always_done(obs):
            return np.ones(obs.shape[0], dtype=bool)

        env = OceanScaleVecEnv(n_envs=2, max_episode_steps=100, termination_fn=always_done)
        env.reset()
        _, _, terminated, _truncated, _ = env.step(np.zeros((2, 6), dtype=np.float32))
        assert np.all(terminated)
        env.close()


class TestOceanScaleVecEnvMultiEnv:
    def test_different_actions_diverge(self):
        env = OceanScaleVecEnv(n_envs=2, max_episode_steps=100)
        env.reset()

        actions = np.zeros((2, 6), dtype=np.float32)
        actions[0, 0] = 1.0
        actions[1, 0] = -1.0

        for _ in range(50):
            obs, *_ = env.step(actions)

        assert obs[0, 0] != obs[1, 0], "envs with opposite actions should diverge in x"
        env.close()
