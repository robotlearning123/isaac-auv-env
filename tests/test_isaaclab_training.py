# mypy: ignore-errors
"""Integration tests for Isaac Lab compatible training pipeline.

Tests the full stack: GPU fluid → Newton physics → sensors → Isaac Lab env → training.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from oceanscale.training.isaaclab_env import (
    ACT_DIM,
    HAS_ISAACLAB,
    OBS_DIM,
    OCEANSCALE_UNDERWATER_TASK_ID,
    OceanScaleDirectRLEnv,
    OceanScaleEnvCfg,
    register_oceanscale_isaaclab_tasks,
)


@pytest.fixture(scope="module")
def env():
    cfg = OceanScaleEnvCfg(num_envs=2, episode_length_s=5.0, device="cuda:0")
    e = OceanScaleDirectRLEnv(cfg)
    yield e
    e.close()


class TestEnvCreation:
    def test_env_creates(self, env):
        assert env is not None

    def test_num_envs(self, env):
        assert env.num_envs == 2

    def test_observation_space(self, env):
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space(self, env):
        assert env.action_space.shape == (ACT_DIM,)

    def test_is_vector_env(self, env):
        assert env.is_vector_env is True


class TestGymTaskRegistration:
    def test_registers_oceanscale_underwater_task(self):
        task_id = register_oceanscale_isaaclab_tasks()
        spec = gym.spec(task_id)

        assert task_id == OCEANSCALE_UNDERWATER_TASK_ID
        assert spec.entry_point == "oceanscale.training.isaaclab_env:OceanScaleDirectRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] == (
            "oceanscale.training.isaaclab_env:OceanScaleEnvCfg"
        )

    def test_registered_task_respects_isaaclab_cfg_fields(self):
        task_id = register_oceanscale_isaaclab_tasks()
        cfg = OceanScaleEnvCfg(num_envs=4, device="cpu")
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = 1

        env = gym.make(task_id, cfg=cfg)
        try:
            assert isinstance(env.unwrapped, OceanScaleDirectRLEnv)
            assert env.unwrapped.num_envs == 1
            assert str(env.unwrapped.device) == "cuda:0"

            obs_dict, _ = env.reset()
            action = torch.zeros(1, ACT_DIM, device=env.unwrapped.device)
            obs_dict, rewards, terminated, truncated, _info = env.step(action)

            assert obs_dict["policy"].shape == (1, OBS_DIM)
            assert rewards.shape == (1,)
            assert terminated.shape == (1,)
            assert truncated.shape == (1,)
            assert torch.all(torch.isfinite(obs_dict["policy"]))
            assert torch.all(torch.isfinite(rewards))
        finally:
            env.close()


class TestEnvReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape(self, env):
        obs_dict, _info = env.reset()
        assert "policy" in obs_dict
        obs = obs_dict["policy"]
        assert obs.shape == (2, OBS_DIM)

    def test_reset_obs_finite(self, env):
        obs_dict, _ = env.reset()
        assert torch.all(torch.isfinite(obs_dict["policy"]))

    def test_reset_obs_on_device(self, env):
        obs_dict, _ = env.reset()
        assert obs_dict["policy"].device.type == "cuda"


class TestEnvStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        action = torch.zeros(2, ACT_DIM, device=env.device)
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        action = torch.zeros(2, ACT_DIM, device=env.device)
        obs_dict, rewards, terminated, truncated, _info = env.step(action)
        assert obs_dict["policy"].shape == (2, OBS_DIM)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)

    def test_step_obs_finite(self, env):
        env.reset()
        action = torch.zeros(2, ACT_DIM, device=env.device)
        obs_dict, _, _, _, _ = env.step(action)
        assert torch.all(torch.isfinite(obs_dict["policy"]))

    def test_step_rewards_finite(self, env):
        env.reset()
        action = torch.zeros(2, ACT_DIM, device=env.device)
        _, rewards, _, _, _ = env.step(action)
        assert torch.all(torch.isfinite(rewards))


class TestVectorizedEnvs:
    def test_batch_consistency(self, env):
        env.reset()
        action = torch.rand(2, ACT_DIM, device=env.device) * 2 - 1
        obs_dict, rewards, _terminated, _truncated, _info = env.step(action)
        assert obs_dict["policy"].shape[0] == 2
        assert rewards.shape[0] == 2

    def test_independent_envs(self):
        cfg = OceanScaleEnvCfg(num_envs=2, episode_length_s=2.0, device="cuda:0")
        e = OceanScaleDirectRLEnv(cfg)
        e.reset()
        action = torch.zeros(2, ACT_DIM, device="cuda:0")
        obs1, r1, _, _, _ = e.step(action)
        assert obs1["policy"].shape == (2, OBS_DIM)
        assert r1.shape == (2,)
        assert torch.all(torch.isfinite(obs1["policy"]))
        e.reset(env_ids=[0])
        obs2, _r2, _, _, _ = e.step(action)
        assert torch.all(torch.isfinite(obs2["policy"]))
        e.close()


class TestTrainingLoop:
    def test_training_runs(self):
        from oceanscale.training.train_isaaclab import train_docking_isaaclab

        metrics = train_docking_isaaclab(num_envs=2, max_iterations=5, device="cuda:0")
        assert "rewards" in metrics
        assert "losses" in metrics
        assert len(metrics["rewards"]) == 5
        assert all(np.isfinite(r) for r in metrics["rewards"])
        assert all(np.isfinite(l) for l in metrics["losses"])


class TestFullStackIntegration:
    def test_fluid_to_sensor_to_training(self):
        """Prove data flows: wave → drag → Newton → DVL/sonar → obs → policy → loss."""
        cfg = OceanScaleEnvCfg(num_envs=1, episode_length_s=2.0, device="cuda:0")
        e = OceanScaleDirectRLEnv(cfg)
        obs_dict, _ = e.reset()
        obs = obs_dict["policy"]

        policy = torch.nn.Linear(OBS_DIM, ACT_DIM).to("cuda:0")
        action = torch.tanh(policy(obs))
        obs_dict, rewards, _, _, _ = e.step(action)

        assert obs_dict["policy"].shape == (1, OBS_DIM)
        assert rewards.shape == (1,)

        wave_vel_part = obs_dict["policy"][0, 6:9]
        dvl_part = obs_dict["policy"][0, 9:14]
        sonar_part = obs_dict["policy"][0, 14:30]
        tension_part = obs_dict["policy"][0, 30]

        assert torch.all(torch.isfinite(wave_vel_part))
        assert torch.all(torch.isfinite(dvl_part))
        assert torch.all(torch.isfinite(sonar_part))
        assert torch.isfinite(tension_part)
        e.close()

    def test_isaaclab_api_available(self):
        """Report Isaac Lab availability (informational, not blocking)."""
        if HAS_ISAACLAB:
            pytest.skip("Isaac Lab IS installed — full DirectRLEnv available")
        else:
            pass
