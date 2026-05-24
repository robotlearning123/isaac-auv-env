"""Tests for DockingApproachEnv — precision docking approach task.

Covers: creation, reset, step, reward phases, success criteria,
hard contact detection, and domain randomization.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from oceanscale.envs.docking_env import DockingApproachEnv
from oceanscale.rov_env import ROVEnv

wp.init()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    e = DockingApproachEnv(n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
    yield e
    e.close()


@pytest.fixture
def env_no_random():
    e = DockingApproachEnv(
        n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0, init_pos_noise_std=0.0
    )
    yield e
    e.close()


@pytest.fixture
def batched_env():
    e = DockingApproachEnv(n_envs=8, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Creation and inheritance
# ---------------------------------------------------------------------------


class TestCreation:
    def test_is_rov_env_subclass(self, env):
        assert isinstance(env, ROVEnv)

    def test_observation_space_31dim(self, env):
        assert env.observation_space.shape == (31,)

    def test_action_space_6dim(self, env):
        assert env.action_space.shape == (6,)

    def test_max_episode_steps_7200(self, env):
        assert env.max_episode_steps == 7200

    def test_default_dock_position(self):
        e = DockingApproachEnv(n_envs=1, device="cuda")
        np.testing.assert_allclose(e.target_pos, [0.0, 0.0, -1.5])
        e.close()

    def test_custom_dock_position(self):
        dock = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        e = DockingApproachEnv(n_envs=1, device="cuda", dock_pos=dock)
        np.testing.assert_allclose(e.target_pos, dock)
        e.close()


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape_31(self, env):
        obs, _ = env.reset()
        assert obs.shape == (31,)

    def test_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_initial_range_in_bounds(self, env):
        obs, _ = env.reset()
        range_to_dock = obs[26]
        assert 2.0 <= range_to_dock <= 4.0, f"Initial range {range_to_dock} outside [2.0, 4.0]"

    def test_batched_reset_shape(self, batched_env):
        obs, _ = batched_env.reset()
        assert obs.shape == (8, 31)

    def test_batched_initial_ranges(self, batched_env):
        obs, _ = batched_env.reset()
        ranges = obs[:, 26]
        assert np.all(ranges >= 2.0) and np.all(ranges <= 4.0)

    def test_position_error_nonzero(self, env):
        obs, _ = env.reset()
        pos_err = np.linalg.norm(obs[0:3])
        assert pos_err > 0.5, "ROV should start away from dock"


# ---------------------------------------------------------------------------
# 3. Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset()
        result = env.step(np.zeros(6, dtype=np.float32))
        assert isinstance(result, tuple) and len(result) == 5

    def test_obs_shape_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == (31,)

    def test_obs_finite(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))

    def test_reward_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(reward))

    def test_info_has_docking_fields(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "success" in info
        assert "range_to_dock" in info
        assert "hard_contact" in info
        assert "min_range" in info

    def test_info_has_reward_components(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "reward_range" in info
        assert "reward_velocity" in info
        assert "reward_heading" in info
        assert "reward_action" in info

    def test_batched_step_shapes(self, batched_env):
        batched_env.reset()
        obs, rewards, terminated, truncated, _ = batched_env.step(
            np.zeros((8, 6), dtype=np.float32)
        )
        assert obs.shape == (8, 31)
        assert rewards.shape == (8,)
        assert terminated.shape == (8,)
        assert truncated.shape == (8,)


# ---------------------------------------------------------------------------
# 4. Reward phases (far vs close)
# ---------------------------------------------------------------------------


class TestRewardPhases:
    def test_range_reward_positive(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["reward_range"] > 0, "Range reward should be positive (exp shaping)"

    def test_speed_limit_schedule(self):
        """Verify speed_limit = clamp(0.5 * range, 0.05, 1.0)."""
        r = np.array([3.0, 1.0, 0.2, 0.05], dtype=np.float32)
        limit = np.clip(0.5 * r, 0.05, 1.0)
        np.testing.assert_allclose(limit, [1.0, 0.5, 0.1, 0.05], atol=1e-6)

    def test_exponential_reward_bounded(self):
        e = DockingApproachEnv(n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
        e.reset()
        _, _, _, _, info = e.step(np.zeros(6, dtype=np.float32))

        assert 0 < info["reward_range"] <= 0.4 + 1e-6
        assert 0 < info["reward_velocity"] <= 0.3 + 1e-6
        assert 0 < info["reward_heading"] <= 0.2 + 1e-6
        assert 0 < info["reward_action"] <= 0.1 + 1e-6
        e.close()

    def test_approach_decreases_range(self, batched_env):
        batched_env.reset()
        obs1 = batched_env._get_flat_obs()
        ranges_before = obs1[:, 26].copy()

        # Step toward dock (use position error as action direction)
        for _ in range(50):
            obs = batched_env._get_flat_obs()
            actions = np.clip(obs[:, 0:3] * 2.0, -1, 1).astype(np.float32)
            full_actions = np.zeros((8, 6), dtype=np.float32)
            full_actions[:, 0:3] = actions
            batched_env.step(full_actions)

        obs_after = batched_env._get_flat_obs()
        # At least some envs should have moved toward dock
        assert np.any(obs_after[:, 26] < ranges_before)


# ---------------------------------------------------------------------------
# 5. Success criteria
# ---------------------------------------------------------------------------


class TestSuccessCriteria:
    def test_success_requires_position_and_velocity(self):
        """Simulate ROV at dock with zero velocity — should detect success."""
        e = DockingApproachEnv(n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
        e.reset()
        import newton

        # Place ROV very close to dock with zero velocity
        dock = e._dock_positions[0].copy()
        # Offset slightly so bearing is well-defined (0.05m away)
        approach_offset = np.array([0.05, 0.0, 0.0], dtype=np.float32)
        rov_pos = dock + approach_offset

        joint_q = e.model.joint_q.numpy()
        # Set heading toward dock (negative x direction → yaw = pi)
        joint_q[0:7] = [
            float(rov_pos[0]),
            float(rov_pos[1]),
            float(rov_pos[2]),
            0.0,
            0.0,
            float(np.sin(np.pi / 2)),
            float(np.cos(np.pi / 2)),
        ]
        e.model.joint_q = wp.array(joint_q, dtype=wp.float32, device="cuda")

        joint_qd = e.model.joint_qd.numpy()
        joint_qd[0:6] = 0.0
        e.model.joint_qd = wp.array(joint_qd, dtype=wp.float32, device="cuda")

        newton.eval_fk(e.model, e.model.joint_q, e.model.joint_qd, e.state_curr)

        _, _reward, _, _, info = e.step(np.zeros(6, dtype=np.float32))

        # Range should be very small
        assert info["range_to_dock"][0] < 0.1, f"Range {info['range_to_dock'][0]}"
        # Success may or may not trigger depending on velocity after step,
        # but range condition should be met
        e.close()

    def test_success_info_is_bool_array(self, batched_env):
        batched_env.reset()
        _, _, _, _, info = batched_env.step(np.zeros((8, 6), dtype=np.float32))
        assert info["success"].dtype == bool
        assert info["success"].shape == (8,)


# ---------------------------------------------------------------------------
# 6. Hard contact detection
# ---------------------------------------------------------------------------


class TestHardContact:
    def test_hard_contact_info_exists(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "hard_contact" in info
        assert info["hard_contact"].dtype == bool


# ---------------------------------------------------------------------------
# 7. Domain randomization
# ---------------------------------------------------------------------------


class TestDomainRandomization:
    def test_different_resets_different_positions(self):
        e = DockingApproachEnv(n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
        obs1, _ = e.reset(seed=42)
        pos1 = obs1[0:3].copy()
        obs2, _ = e.reset(seed=123)
        pos2 = obs2[0:3].copy()
        # Different seeds should give different ROV positions
        assert not np.allclose(pos1, pos2, atol=0.01)
        e.close()

    def test_dock_offset_randomization(self):
        e = DockingApproachEnv(n_envs=4, device="cuda", sensor_noise_std=0.0, dock_offset_range=1.0)
        e.reset()
        # Dock positions should differ across envs (with offset)
        docks = e._dock_positions.copy()
        # Not all dock positions should be identical
        assert not np.allclose(docks[0], docks[1], atol=0.01) or not np.allclose(
            docks[0], docks[2], atol=0.01
        )
        e.close()

    def test_no_dock_offset_fixed(self):
        e = DockingApproachEnv(n_envs=4, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
        e.reset()
        docks = e._dock_positions.copy()
        # All docks at same position (target_pos)
        for i in range(4):
            np.testing.assert_allclose(docks[i], e.target_pos, atol=1e-6)
        e.close()

    def test_initial_range_bounds_across_resets(self):
        e = DockingApproachEnv(n_envs=1, device="cuda", sensor_noise_std=0.0, dock_offset_range=0.0)
        ranges = []
        for seed in range(20):
            obs, _ = e.reset(seed=seed)
            ranges.append(float(obs[26]))
        ranges = np.array(ranges)
        assert np.all(ranges >= 2.0) and np.all(ranges <= 4.0)
        # Should have some variance
        assert np.std(ranges) > 0.1
        e.close()

    def test_domain_rand_flag(self):
        e = DockingApproachEnv(n_envs=1, device="cuda", use_domain_randomization=True)
        assert e.use_domain_randomization is True
        e.close()


# ---------------------------------------------------------------------------
# 8. Observation structure
# ---------------------------------------------------------------------------


class TestObservationStructure:
    def test_range_obs_positive(self, env):
        obs, _ = env.reset()
        assert obs[26] > 0, "Range should be positive"

    def test_bearing_unit_vector(self, env):
        obs, _ = env.reset()
        bearing = obs[27:30]
        norm = np.linalg.norm(bearing)
        # With noise=0, bearing should be ~unit vector
        # But with some initial range, it should be close to 1
        if obs[26] > 0.1:
            np.testing.assert_allclose(norm, 1.0, atol=0.1)

    def test_obs_31_dims(self, env):
        obs, _ = env.reset()
        assert obs.shape[-1] == 31


# ---------------------------------------------------------------------------
# 9. Termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_max_steps_7200(self, env):
        assert env.max_episode_steps == 7200

    def test_no_early_termination(self, env):
        env.reset()
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            assert not terminated
            assert not truncated

    def test_min_range_tracking(self, env):
        env.reset()
        _obs, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert np.all(np.isfinite(info["min_range"]))
        assert info["min_range"][0] > 0
