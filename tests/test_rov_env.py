"""Tests for ROVEnv — Gymnasium wrapper for GPU-batched ROV simulation.

API contract (as of current implementation):
- n_envs=1: reset() returns obs=(26,), step() returns scalar reward/terminated/truncated
- n_envs>1: reset() returns obs=(n_envs, 26), step() returns array reward/terminated/truncated

Tests cover: Gymnasium spaces, reset, step, reward shaping, partial reset,
batched independence, sensor noise, observation structure, termination.
"""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym
import warp as wp

wp.init()

from oceanscale.rov_env import ROVEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_shape(env):
    """Return expected obs shape — always (n_envs, 26) for GPU-vectorized env."""
    return (env.n_envs, 26)


def _scalar_or_array(env, val):
    """For n_envs=1, val is scalar. For n_envs>1, val is array."""
    if env.n_envs == 1:
        return np.isscalar(val) or (isinstance(val, np.generic))
    return isinstance(val, np.ndarray) and val.shape == (env.n_envs,)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = ROVEnv(n_envs=1, device="cuda")
    yield e
    e.close()


@pytest.fixture
def env_no_noise():
    e = ROVEnv(n_envs=1, device="cuda", sensor_noise_std=0.0,
               init_pos_noise_std=0.0, init_yaw_noise_std=0.0)
    yield e
    e.close()


@pytest.fixture
def batched_env():
    e = ROVEnv(n_envs=16, device="cuda")
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Gymnasium spaces
# ---------------------------------------------------------------------------

class TestGymSpaces:
    def test_is_gym_env(self, env):
        assert isinstance(env, gym.Env)

    def test_observation_space_is_box(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_action_space_is_box(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_observation_space_26dim(self, env):
        assert env.observation_space.shape == (26,)

    def test_action_space_6dim(self, env):
        assert env.action_space.shape == (6,)

    def test_action_space_bounded(self, env):
        np.testing.assert_allclose(env.action_space.low, -1.0)
        np.testing.assert_allclose(env.action_space.high, 1.0)


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == _obs_shape(env)

    def test_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_info_has_target_position(self, env):
        _, info = env.reset()
        assert "target_position" in info

    def test_position_near_target(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        # obs shape is (26,) for n_envs=1
        o = obs if obs.ndim == 1 else obs[0]
        np.testing.assert_allclose(o[:3], 0.0, atol=0.1)

    def test_velocity_zero(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        o = obs if obs.ndim == 1 else obs[0]
        np.testing.assert_allclose(o[7:13], 0.0, atol=0.5)

    def test_idempotent_no_noise(self, env_no_noise):
        obs1, _ = env_no_noise.reset(seed=42)
        obs2, _ = env_no_noise.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2, atol=1e-5)

    def test_with_seed(self, env_no_noise):
        obs1, _ = env_no_noise.reset(seed=42)
        obs2, _ = env_no_noise.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2, atol=1e-5)

    def test_with_target_position_option(self, env):
        target = np.array([1.0, 2.0, -0.5], dtype=np.float32)
        _, info = env.reset(options={"target_position": target})
        np.testing.assert_allclose(info["target_position"], target, atol=1e-6)

    def test_batched_shape(self, batched_env):
        obs, _ = batched_env.reset()
        assert obs.shape == (16, 26)

    def test_close_no_error(self, env):
        env.close()
        env.close()


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset()
        result = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(result, tuple) and len(result) == 5

    def test_obs_shape(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == _obs_shape(env)

    def test_obs_finite(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))

    def test_reward_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        if isinstance(reward, np.ndarray):
            assert np.all(np.isfinite(reward))
        else:
            assert np.isfinite(reward)

    def test_info_is_dict(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(info, dict)

    def test_batched_shapes(self, batched_env):
        batched_env.reset()
        actions = np.zeros((16, 6), dtype=np.float32)
        obs, rewards, terminateds, truncateds, _ = batched_env.step(actions)
        assert obs.shape == (16, 26)
        assert rewards.shape == (16,)
        assert terminateds.shape == (16,)
        assert truncateds.shape == (16,)

    def test_batched_broadcasts_1d(self, batched_env):
        batched_env.reset()
        obs, rewards, _, _, _ = batched_env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == (16, 26)
        assert rewards.shape == (16,)

    def test_thrust_changes_velocity(self, env_no_noise):
        env_no_noise.reset()
        obs, _, _, _, _ = env_no_noise.step(np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32))
        o = obs if obs.ndim == 1 else obs[0]
        vel_norm = np.linalg.norm(o[7:13])
        assert vel_norm > 1e-6, f"Velocity should be non-zero, vel={o[7:13]}"


# ---------------------------------------------------------------------------
# 4. Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def _reward_scalar(self, env, r):
        return float(r[0]) if isinstance(r, np.ndarray) else float(r)

    def test_stay_better_than_drift(self, env_no_noise):
        env_no_noise.reset()
        rewards_stay = [self._reward_scalar(env_no_noise, env_no_noise.step(np.zeros(6, dtype=np.float32))[1]) for _ in range(20)]

        env_no_noise.reset()
        drift = np.array([1.0, 0, 0, 0, 0, 0], dtype=np.float32)
        rewards_drift = [self._reward_scalar(env_no_noise, env_no_noise.step(drift)[1]) for _ in range(20)]

        assert np.mean(rewards_stay) > np.mean(rewards_drift)

    def test_reward_degrades_with_distance(self, env_no_noise):
        env_no_noise.reset()
        drift = np.array([0.8, 0, 0, 0, 0, 0], dtype=np.float32)
        rewards = [self._reward_scalar(env_no_noise, env_no_noise.step(drift)[1]) for _ in range(50)]
        assert np.mean(rewards[:10]) > np.mean(rewards[-10:])

    def test_info_has_components(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "reward_distance" in info
        assert "reward_action" in info

    def test_batched_rewards_finite(self, batched_env):
        batched_env.reset()
        for _ in range(20):
            actions = np.random.uniform(-1, 1, (16, 6)).astype(np.float32)
            _, rewards, _, _, _ = batched_env.step(actions)
            assert np.all(np.isfinite(rewards))


# ---------------------------------------------------------------------------
# 5. Partial reset
# ---------------------------------------------------------------------------

class TestPartialReset:
    def test_reset_envs_exists(self, batched_env):
        assert hasattr(batched_env, "reset_envs")

    def test_partial_reset_resets_selected(self, batched_env):
        batched_env.reset()
        drift = np.zeros((16, 6), dtype=np.float32)
        drift[:, 0] = 0.8
        for _ in range(30):
            batched_env.step(drift)

        reset_ids = np.array([0, 5, 10])
        obs, _ = batched_env.reset(env_ids=reset_ids)

        for idx in reset_ids:
            pos_err = np.linalg.norm(obs[idx, :3])
            assert pos_err < 1.0


# ---------------------------------------------------------------------------
# 6. Batched independence
# ---------------------------------------------------------------------------

class TestBatchedIndependence:
    def test_different_actions_different_states(self, batched_env):
        batched_env.reset()
        actions = np.zeros((16, 6), dtype=np.float32)
        actions[0, 0] = 1.0
        actions[1, 1] = -1.0
        for _ in range(100):
            batched_env.step(actions)
        obs = batched_env._get_flat_obs()
        assert not np.allclose(obs[0, :3], obs[1, :3], atol=0.01)

    def test_zero_action_envs_bounded(self, batched_env):
        batched_env.reset()
        for _ in range(30):
            batched_env.step(np.zeros((16, 6), dtype=np.float32))
        obs = batched_env._get_flat_obs()
        for i in range(16):
            assert np.linalg.norm(obs[i, :3]) < 5.0


# ---------------------------------------------------------------------------
# 7. Sensor noise
# ---------------------------------------------------------------------------

class TestSensorNoise:
    def test_noise_varies_obs(self):
        env = ROVEnv(n_envs=1, device="cuda", sensor_noise_std=0.05)
        env.reset()
        observations = []
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
            observations.append(obs.copy())
        env.close()
        obs_arr = np.array(observations).squeeze()
        assert np.any(np.abs(np.diff(obs_arr, axis=0)) > 1e-8)

    def test_no_noise_deterministic(self):
        env = ROVEnv(n_envs=1, device="cuda", sensor_noise_std=0.0)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_allclose(obs1, obs2, atol=1e-5)
        env.close()

    def test_noise_obs_finite(self):
        env = ROVEnv(n_envs=1, device="cuda", sensor_noise_std=0.1)
        env.reset()
        for _ in range(50):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
        env.close()


# ---------------------------------------------------------------------------
# 8. Observation structure (n_envs=1)
# ---------------------------------------------------------------------------

class TestObservationStructure:
    def _flat(self, obs):
        return obs if obs.ndim == 1 else obs[0]

    def test_obs_dimension(self, env):
        obs, _ = env.reset()
        o = self._flat(obs)
        assert o.shape == (26,)

    def test_position_error_first_3(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        np.testing.assert_allclose(self._flat(obs)[:3], 0.0, atol=0.1)

    def test_quaternion_3_7(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        q = self._flat(obs)[3:7]
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=0.15)

    def test_velocity_7_13(self, env_no_noise):
        obs, _ = env_no_noise.reset()
        np.testing.assert_allclose(self._flat(obs)[7:13], 0.0, atol=0.5)

    def test_prev_action_16_22(self, env_no_noise):
        env_no_noise.reset()
        env_no_noise.step(np.array([0.5, -0.3, 0.1, 0, 0, 0], dtype=np.float32))
        obs, _, _, _, _ = env_no_noise.step(np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(self._flat(obs)[16:22], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 9. Termination
# ---------------------------------------------------------------------------

class TestTermination:
    def test_max_steps_default(self, env):
        assert env.max_episode_steps == 1000

    def test_custom_max_steps(self):
        e = ROVEnv(n_envs=1, device="cuda", max_episode_steps=50)
        assert e.max_episode_steps == 50
        e.close()

    def test_terminated_false(self, env):
        env.reset()
        for _ in range(50):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            t = terminated[0] if isinstance(terminated, np.ndarray) else terminated
            assert not t


# ---------------------------------------------------------------------------
# 10. Configuration
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_custom_n_envs(self):
        e = ROVEnv(n_envs=8, device="cuda")
        assert e.n_envs == 8
        e.close()

    def test_custom_target_pos(self):
        target = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        e = ROVEnv(n_envs=1, device="cuda", target_pos=target)
        np.testing.assert_allclose(e.target_pos, target)
        e.close()

    def test_custom_sensor_noise(self):
        e = ROVEnv(n_envs=1, device="cuda", sensor_noise_std=0.5)
        assert e.sensor_noise_std == 0.5
        e.close()

    def test_custom_dt(self):
        e = ROVEnv(n_envs=1, device="cuda", dt=0.01)
        assert e.dt == pytest.approx(0.01)
        e.close()


# ---------------------------------------------------------------------------
# 11. Domain randomization
# ---------------------------------------------------------------------------

class TestDomainRandomization:
    def test_flag_default_off(self):
        e = ROVEnv(n_envs=1, device="cuda")
        assert e.use_domain_randomization is False
        e.close()

    def test_flag_enabled(self):
        e = ROVEnv(n_envs=1, device="cuda", use_domain_randomization=True)
        assert e.use_domain_randomization is True
        e.close()

    def test_custom_ranges(self):
        from oceanscale.hydro.tier1 import RandomizationRanges
        ranges = RandomizationRanges(mass=0.1, d_lin=0.05)
        e = ROVEnv(n_envs=1, device="cuda", use_domain_randomization=True, randomization_ranges=ranges)
        assert e.randomization_ranges.mass == pytest.approx(0.1)
        assert e.randomization_ranges.d_lin == pytest.approx(0.05)
        e.close()

    def test_reset_with_randomization_produces_different_coeffs(self):
        e = ROVEnv(n_envs=8, device="cuda", use_domain_randomization=True)
        e.reset()
        ma1 = e.tier1.M_A_lin.numpy().copy()
        e.reset()
        ma2 = e.tier1.M_A_lin.numpy().copy()
        # Two resets with different RNG state should produce different coefficients
        assert not np.allclose(ma1, ma2)
        e.close()

    def test_reset_without_randomization_same_coeffs(self):
        e = ROVEnv(n_envs=8, device="cuda", use_domain_randomization=False)
        e.reset()
        ma1 = e.tier1.M_A_lin.numpy().copy()
        e.reset()
        ma2 = e.tier1.M_A_lin.numpy().copy()
        np.testing.assert_array_equal(ma1, ma2)
        e.close()

    def test_step_after_randomization_finite(self):
        e = ROVEnv(n_envs=4, device="cuda", use_domain_randomization=True)
        e.reset()
        for _ in range(20):
            obs, reward, terminated, truncated, _ = e.step(
                np.zeros((4, 6), dtype=np.float32)
            )
            assert np.all(np.isfinite(obs))
            assert np.all(np.isfinite(reward))
        e.close()

    def test_thruster_gain_default_one(self):
        e = ROVEnv(n_envs=4, device="cuda")
        e.reset()
        np.testing.assert_array_equal(e._thruster_gain, 1.0)
        e.close()

    def test_thruster_gain_randomized_in_range(self):
        e = ROVEnv(n_envs=64, device="cuda", use_domain_randomization=True)
        e.reset()
        gains = e._thruster_gain.copy()
        assert np.all(gains >= 0.8), f"min gain {gains.min()} < 0.8"
        assert np.all(gains <= 1.2), f"max gain {gains.max()} > 1.2"
        assert np.std(gains) > 0.01, "gains should vary across envs"
        e.close()

    def test_thruster_gain_differs_across_resets(self):
        e = ROVEnv(n_envs=16, device="cuda", use_domain_randomization=True)
        e.reset()
        g1 = e._thruster_gain.copy()
        e.reset()
        g2 = e._thruster_gain.copy()
        assert not np.allclose(g1, g2), "thruster gains should differ across resets"
        e.close()

    def test_thruster_gain_affects_force(self):
        """Same action should produce different velocities with different gains."""
        import torch
        e = ROVEnv(n_envs=4, device="cuda", sensor_noise_std=0.0,
                    init_pos_noise_std=0.0, init_yaw_noise_std=0.0)
        e.reset()
        e._thruster_gain[:] = [0.8, 1.0, 1.0, 1.2]
        e._thruster_gain_gpu = torch.tensor(
            e._thruster_gain, device=e.device, dtype=torch.float32
        )
        action = np.array([[1.0, 0, 0, 0, 0, 0]] * 4, dtype=np.float32)
        for _ in range(50):
            obs, _, _, _, _ = e.step(action)
        # Lower gain → lower velocity
        v_low = np.linalg.norm(obs[0, 7:10])
        v_mid = np.linalg.norm(obs[1, 7:10])
        v_high = np.linalg.norm(obs[3, 7:10])
        assert v_low < v_mid < v_high, f"v_low={v_low}, v_mid={v_mid}, v_high={v_high}"
        e.close()


# ---------------------------------------------------------------------------
# 12. GPU-native torch interface
# ---------------------------------------------------------------------------

class TestTorchInterface:
    def test_reset_torch_returns_cuda_tensors(self):
        import torch
        e = ROVEnv(n_envs=4, device="cuda")
        obs, info = e.reset_torch()
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == "cuda"
        assert obs.shape == (4, 26)
        assert obs.dtype == torch.float32
        e.close()

    def test_step_torch_returns_cuda_tensors(self):
        import torch
        e = ROVEnv(n_envs=4, device="cuda")
        e.reset_torch()
        action = torch.zeros(4, 6, device="cuda", dtype=torch.float32)
        obs, reward, terminated, truncated, info = e.step_torch(action)
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == "cuda"
        assert obs.shape == (4, 26)
        assert isinstance(reward, torch.Tensor)
        assert reward.device.type == "cuda"
        assert reward.shape == (4,)
        assert isinstance(terminated, torch.Tensor)
        assert terminated.device.type == "cuda"
        assert isinstance(truncated, torch.Tensor)
        assert truncated.device.type == "cuda"
        e.close()

    def test_step_torch_matches_step(self):
        """step_torch and step produce identical values with no noise."""
        import torch
        e = ROVEnv(n_envs=4, device="cuda", sensor_noise_std=0.0,
                    init_pos_noise_std=0.0, init_yaw_noise_std=0.0)
        action_np = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
        action_t = torch.from_numpy(action_np).to("cuda")

        # Gymnasium path
        e.reset(seed=42)
        obs_np, rew_np, term_np, trunc_np, _ = e.step(action_np)

        # Torch path (same env, fresh reset with same seed)
        e.reset(seed=42)
        obs_t, rew_t, term_t, trunc_t, _ = e.step_torch(action_t)

        np.testing.assert_allclose(obs_t.cpu().numpy(), obs_np, atol=1e-5)
        np.testing.assert_allclose(rew_t.cpu().numpy(), rew_np, atol=1e-5)
        np.testing.assert_array_equal(term_t.cpu().numpy(), term_np)
        np.testing.assert_array_equal(trunc_t.cpu().numpy(), trunc_np)
        e.close()
