"""Tests for WaypointFollowingEnv -- sequential waypoint navigation task."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from oceanscale.envs.waypoint_env import WaypointFollowingEnv

wp.init()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    e = WaypointFollowingEnv(n_envs=1, device="cuda:0", sensor_noise_std=0.0)
    yield e
    e.close()


@pytest.fixture
def batched_env():
    e = WaypointFollowingEnv(n_envs=8, device="cuda:0", sensor_noise_std=0.0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Creation & Spaces
# ---------------------------------------------------------------------------


class TestCreation:
    def test_obs_space_29dim(self, env):
        assert env.observation_space.shape == (29,)

    def test_action_space_6dim(self, env):
        assert env.action_space.shape == (6,)

    def test_n_waypoints(self, env):
        assert env.n_waypoints == 5

    def test_max_steps_from_time_limit(self, env):
        expected = int(120.0 / env.dt)
        assert env.max_episode_steps == expected

    def test_wp_tolerance(self, env):
        assert env.wp_tolerance == 1.0

    def test_wp_bonus(self, env):
        assert env.wp_bonus == 10.0


# ---------------------------------------------------------------------------
# 2. Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (29,)

    def test_info_has_waypoints(self, env):
        _, info = env.reset()
        assert "waypoints" in info
        assert info["waypoints"].shape == (1, 5, 3)

    def test_info_has_current_wp_idx(self, env):
        _, info = env.reset()
        assert "current_wp_idx" in info
        np.testing.assert_array_equal(info["current_wp_idx"], [0])

    def test_info_has_waypoints_reached(self, env):
        _, info = env.reset()
        assert "waypoints_reached" in info
        assert not np.any(info["waypoints_reached"])

    def test_position_near_first_waypoint(self, env):
        obs, _ = env.reset()
        pos_err = np.linalg.norm(obs[:3])
        assert pos_err < 1.0

    def test_batched_obs_shape(self, batched_env):
        obs, _ = batched_env.reset()
        assert obs.shape == (8, 29)

    def test_batched_waypoints_shape(self, batched_env):
        _, info = batched_env.reset()
        assert info["waypoints"].shape == (8, 5, 3)


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
        assert obs.shape == (29,)

    def test_obs_finite(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))

    def test_reward_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(reward))

    def test_info_has_success(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "success" in info
        assert "waypoints_reached" in info
        assert "current_wp_idx" in info

    def test_batched_step_shapes(self, batched_env):
        batched_env.reset()
        obs, rewards, terminateds, truncateds, _ = batched_env.step(
            np.zeros((8, 6), dtype=np.float32)
        )
        assert obs.shape == (8, 29)
        assert rewards.shape == (8,)
        assert terminateds.shape == (8,)
        assert truncateds.shape == (8,)


# ---------------------------------------------------------------------------
# 4. Waypoint advancement
# ---------------------------------------------------------------------------


class TestWaypointAdvancement:
    def test_advance_when_at_waypoint(self, env):
        """ROV at waypoint 0 position -> current_wp_idx advances to 1."""
        env.reset()
        pos, _, _ = env._get_body_state()
        env._waypoints[0, 0] = pos[0].clone()

        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["current_wp_idx"][0] >= 1

    def test_no_advance_when_far(self, env):
        """ROV far from waypoint -> no advancement."""
        env.reset()
        env._waypoints[0, 0] = torch.tensor(
            [50.0, 50.0, -50.0], device=env.device, dtype=torch.float32
        )

        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["current_wp_idx"][0] == 0

    def test_all_waypoints_reached(self, env):
        """All waypoints at same position -> reach all 5 after stepping."""
        env.reset()
        pos, _, _ = env._get_body_state()
        for j in range(5):
            env._waypoints[0, j] = pos[0].clone()

        for _ in range(10):
            _, _, _terminated, _truncated, info = env.step(np.zeros(6, dtype=np.float32))
            if info["success"][0]:
                break

        assert info["waypoints_reached"][0] == 5
        assert info["success"][0]


# ---------------------------------------------------------------------------
# 5. Termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_success_terminates_episode(self, env):
        """All waypoints visited -> terminated = True."""
        env.reset()
        pos, _, _ = env._get_body_state()
        for j in range(5):
            env._waypoints[0, j] = pos[0].clone()

        for _ in range(10):
            _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
            if terminated:
                break

        assert terminated
        assert info["success"][0]

    def test_oob_terminates(self, env):
        """ROV >10m from current waypoint -> terminated."""
        env.reset()
        env._waypoints[0, 0] = torch.tensor(
            [50.0, 50.0, -50.0], device=env.device, dtype=torch.float32
        )

        for _ in range(10):
            _, _, terminated, _, _ = env.step(np.zeros(6, dtype=np.float32))
            if terminated:
                break

        assert terminated


# ---------------------------------------------------------------------------
# 6. Reward structure
# ---------------------------------------------------------------------------


class TestReward:
    def test_reward_positive_without_reach(self, env):
        """Exp reward is always positive even without reaching waypoint."""
        env.reset()
        env._waypoints[0, 0] = torch.tensor(
            [50.0, 0.0, -1.5], device=env.device, dtype=torch.float32
        )
        _, reward, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["current_wp_idx"][0] == 0
        assert 0.0 <= float(reward) < 1.0

    def test_bonus_positive_on_reach(self, env):
        """Reaching waypoint gives +10 bonus, making total reward positive."""
        env.reset()
        pos, _, _ = env._get_body_state()
        env._waypoints[0, 0] = pos[0].clone()
        env._waypoints[0, 1] = pos[0].clone()

        _, reward, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert info["current_wp_idx"][0] >= 1
        assert float(reward) > 5.0, f"Expected bonus reward > 5.0, got {reward}"

    def test_closer_higher_reward(self, env):
        """Closer to waypoint -> less negative reward."""
        env.reset()
        pos, _, _ = env._get_body_state()
        env._waypoints[0, 0] = pos[0] + torch.tensor(
            [0.1, 0.0, 0.0], device=env.device, dtype=torch.float32
        )
        _, reward_close, _, _, _ = env.step(np.zeros(6, dtype=np.float32))

        env.reset()
        pos2, _, _ = env._get_body_state()
        env._waypoints[0, 0] = pos2[0] + torch.tensor(
            [4.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        )
        _, reward_far, _, _, _ = env.step(np.zeros(6, dtype=np.float32))

        assert float(reward_close) > float(reward_far)
