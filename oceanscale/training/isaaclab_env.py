# mypy: ignore-errors
"""Isaac Lab compatible environment wrapper for OceanScale.

Implements the DirectRLEnv protocol so OceanScale underwater environments
can be trained with Isaac Lab's RL infrastructure (PPO, SAC, etc.).

When Isaac Lab is not installed, provides a standalone shim implementing
the same step/reset/observation protocol, enabling training with any
Gymnasium-compatible RL library while maintaining API compatibility.

Architecture:
    GPU Fluid (FFT wave) → FSI (drag) → Newton Physics (ROV body)
      → Sensors (RayDVL, RaySonar) → Observations → RL Training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np
import torch

try:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

    HAS_ISAACLAB = True
except ImportError:
    HAS_ISAACLAB = False


OBS_DIM = 33
ACT_DIM = 6


@dataclass
class OceanScaleEnvCfg:
    """Configuration mirroring Isaac Lab's DirectRLEnvCfg fields."""

    num_envs: int = 64
    episode_length_s: float = 30.0
    decimation: int = 4
    physics_dt: float = 0.005
    device: str = "cuda:0"
    seed: int = 42
    wave_height: float = 1.0
    wave_period: float = 8.0
    seabed_depth: float = -50.0
    tether_length: float = 20.0
    drag_coeff: float = 5.0


def _build_obs(step_result: dict) -> np.ndarray:
    """Flatten step result dict into a fixed-size observation vector."""
    pos = step_result["rov_position"][:3]
    vel = step_result["rov_velocity"][:6]
    wave = step_result["wave_velocity"][:3]
    dvl_ranges = step_result["dvl"]["beam_ranges"][:4]
    dvl_alt = np.array([step_result["dvl"]["altitude"]], dtype=np.float32)
    sonar = step_result["sonar"][:16]
    tension = np.array([step_result["tether_tension"]], dtype=np.float32)
    time_frac = np.array([step_result["time"] % 30.0 / 30.0], dtype=np.float32)
    obs = np.concatenate([pos, vel, wave, dvl_ranges, dvl_alt, sonar, tension, time_frac])
    return obs[:OBS_DIM].astype(np.float32)


class OceanScaleDirectRLEnv:
    """Isaac Lab DirectRLEnv-compatible wrapper for OceanScale.

    Follows the DirectRLEnv protocol:
      - step(action) -> (obs_dict, reward, terminated, truncated, info)
      - reset() -> (obs_dict, info)
      - All tensors are batched: (num_envs, dim)
      - Observations returned as dict: {"policy": tensor}

    When Isaac Lab is available, this can be registered as an Isaac Lab env.
    When not available, it works standalone with any Gymnasium-compatible trainer.
    """

    is_vector_env = True
    metadata = {"render_modes": [None]}

    def __init__(self, cfg: OceanScaleEnvCfg | None = None) -> None:
        if cfg is None:
            cfg = OceanScaleEnvCfg()
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.physics_dt = cfg.physics_dt
        self.step_dt = cfg.physics_dt * cfg.decimation
        self.max_episode_length = int(cfg.episode_length_s / self.step_dt)
        self.max_episode_length_s = cfg.episode_length_s

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.num_observations = OBS_DIM
        self.num_actions = ACT_DIM

        from oceanscale.integration import IntegratedPipeline

        self._pipelines = [
            IntegratedPipeline(
                seabed_depth=cfg.seabed_depth,
                wave_height=cfg.wave_height,
                wave_period=cfg.wave_period,
                tether_length=cfg.tether_length,
                drag_coeff=cfg.drag_coeff,
                dt=cfg.physics_dt,
                device=cfg.device,
            )
            for _ in range(cfg.num_envs)
        ]

        self._step_count = torch.zeros(cfg.num_envs, dtype=torch.int64, device=self.device)
        self._episode_rewards = torch.zeros(cfg.num_envs, dtype=torch.float32, device=self.device)

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        if env_ids is None:
            env_ids = range(self.num_envs)

        obs_list = []
        for i in env_ids:
            result = self._pipelines[i].reset()
            obs_list.append(_build_obs(result))
            self._step_count[i] = 0
            self._episode_rewards[i] = 0.0

        if len(env_ids) == self.num_envs:
            obs_np = np.stack(obs_list)
        else:
            obs_np = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
            for idx, i in enumerate(env_ids):
                obs_np[i] = obs_list[idx]

        obs_tensor = torch.from_numpy(obs_np).to(self.device)
        return {"policy": obs_tensor}, {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = action.detach().cpu().numpy()
        obs_list = []
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(self.num_envs):
            for _ in range(self.cfg.decimation):
                result = self._pipelines[i].step(
                    action=action_np[i] if action_np.ndim > 1 else action_np,
                    dt=self.physics_dt,
                )

            obs_list.append(_build_obs(result))
            rewards[i] = result["reward"]
            self._step_count[i] += 1
            self._episode_rewards[i] += result["reward"]

            pos = result["rov_position"]
            if np.linalg.norm(pos) > 100.0 or not np.all(np.isfinite(pos)):
                terminated[i] = True
            if self._step_count[i] >= self.max_episode_length:
                truncated[i] = True

        obs_tensor = torch.from_numpy(np.stack(obs_list)).to(self.device)

        done_ids = (terminated | truncated).nonzero(as_tuple=True)[0].tolist()
        if done_ids:
            self._reset_idx(done_ids)

        extras = {
            "episode_rewards": self._episode_rewards.clone(),
            "step_count": self._step_count.clone(),
        }
        return {"policy": obs_tensor}, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        for i in env_ids:
            self._pipelines[i].reset()
            self._step_count[i] = 0
            self._episode_rewards[i] = 0.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_list = []
        for i in range(self.num_envs):
            result = self._pipelines[i].step(dt=0.0001)
            obs_list.append(_build_obs(result))
        return {"policy": torch.from_numpy(np.stack(obs_list)).to(self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return self._episode_rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self._step_count >= self.max_episode_length
        return terminated, truncated

    def close(self) -> None:
        pass
