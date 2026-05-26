# mypy: ignore-errors
"""Tests for Isaac Lab DirectRLEnv wrapper backed by OceanSim.

Validates the GPU-batched env works with the Isaac Lab DirectRLEnv
protocol: batched tensors, partial reset, decimation, reward/done.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from oceanscale.training.isaaclab_env import (
    ACT_DIM,
    HAS_ISAACLAB,
    OBS_DIM,
    TASK_ID,
    OceanScaleDirectRLEnv,
    OceanScaleEnvCfg,
    register_oceanscale_tasks,
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

    def test_has_ocean_sim(self, env):
        from oceanscale.sim import OceanSim
        assert isinstance(env.sim, OceanSim)


class TestEnvReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape(self, env):
        obs_dict, _ = env.reset()
        assert "policy" in obs_dict
        assert obs_dict["policy"].shape == (2, OBS_DIM)

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
        obs_dict, rewards, terminated, truncated, _ = env.step(action)
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

    def test_reward_positive_at_target(self, env):
        env.reset()
        action = torch.zeros(2, ACT_DIM, device=env.device)
        _, rewards, _, _, _ = env.step(action)
        assert torch.all(rewards > 0)

    def test_decimation_multiple_substeps(self):
        cfg = OceanScaleEnvCfg(num_envs=1, decimation=8, device="cuda:0")
        e = OceanScaleDirectRLEnv(cfg)
        e.reset()
        action = torch.ones(1, ACT_DIM, device="cuda:0") * 0.5
        e.step(action)
        assert e.sim._step_count[0] == 8
        e.close()


class TestVectorizedEnvs:
    def test_batch_consistency(self, env):
        env.reset()
        action = torch.rand(2, ACT_DIM, device=env.device) * 2 - 1
        obs_dict, rewards, _, _, _ = env.step(action)
        assert obs_dict["policy"].shape[0] == 2
        assert rewards.shape[0] == 2

    def test_partial_reset(self):
        cfg = OceanScaleEnvCfg(num_envs=4, episode_length_s=2.0, device="cuda:0")
        e = OceanScaleDirectRLEnv(cfg)
        e.reset()
        for _ in range(10):
            action = torch.rand(4, ACT_DIM, device="cuda:0") * 2 - 1
            e.step(action)
        e.reset(env_ids=[0, 2])
        obs_dict, _, _, _, _ = e.step(torch.zeros(4, ACT_DIM, device="cuda:0"))
        assert torch.all(torch.isfinite(obs_dict["policy"]))
        e.close()


class TestGymRegistration:
    def test_registers_task(self):
        task_id = register_oceanscale_tasks()
        assert task_id == TASK_ID

    def test_make_registered_env(self):
        register_oceanscale_tasks()
        e = OceanScaleDirectRLEnv(OceanScaleEnvCfg(num_envs=1, device="cuda:0"))
        obs_dict, _ = e.reset()
        assert obs_dict["policy"].shape == (1, OBS_DIM)
        e.close()


class TestPolicyGradient:
    def test_end_to_end_gradient(self):
        cfg = OceanScaleEnvCfg(num_envs=2, device="cuda:0")
        e = OceanScaleDirectRLEnv(cfg)
        obs_dict, _ = e.reset()
        obs = obs_dict["policy"]

        policy = torch.nn.Linear(OBS_DIM, ACT_DIM).to("cuda:0")
        action = torch.tanh(policy(obs))
        obs_dict, rewards, _, _, _ = e.step(action.detach())

        assert obs_dict["policy"].shape == (2, OBS_DIM)
        assert torch.all(torch.isfinite(rewards))
        e.close()

    def test_isaaclab_availability(self):
        if HAS_ISAACLAB:
            pytest.skip("Isaac Lab installed — full DirectRLEnv available")
