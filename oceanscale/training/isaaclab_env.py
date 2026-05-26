# mypy: ignore-errors
"""Isaac Lab DirectRLEnv wrapper for OceanScale.

GPU-batched underwater RL environment built on the unified OceanSim
orchestrator. When Isaac Lab is installed, plugs directly into its
training infrastructure (PPO, SAC, curriculum, logging). Without
Isaac Lab, works as a standalone Gymnasium env.

Architecture::

    Isaac Lab training loop
        └── OceanScaleDirectRLEnv (this file)
            └── OceanSim.step_torch()
                ├── Newton rigid-body physics
                ├── Tier1 Fossen hydrodynamics
                ├── Ocean currents / waves / water column
                └── Sensors (DVL, sonar, magnetometer)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from oceanscale.sim import OceanConfig, OceanSim, OceanSimConfig
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


def _has_isaaclab() -> bool:
    return importlib.util.find_spec("isaaclab") is not None


HAS_ISAACLAB = _has_isaaclab()

OBS_DIM = 20
ACT_DIM = 6
TASK_ID = "OceanScale-UnderwaterRobot-Direct-v0"
OCEANSCALE_UNDERWATER_TASK_ID = TASK_ID

def register_oceanscale_isaaclab_tasks() -> str:
    return register_oceanscale_tasks()


@dataclass
class OceanScaleSimCfg:
    device: str = "cuda:0"
    use_fabric: bool = False


@dataclass
class OceanScaleSceneCfg:
    num_envs: int = 64


@dataclass
class OceanScaleEnvCfg:
    """DirectRLEnv-compatible configuration."""

    num_envs: int = 64
    episode_length_s: float = 30.0
    decimation: int = 4
    physics_dt: float = 1 / 240
    device: str = "cuda:0"
    seed: int = 42

    target_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
    init_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
    init_pos_noise: float = 0.5
    oob_distance: float = 5.0

    current_speed: float = 0.0
    current_direction: float = 0.0
    wave_height: float = 0.0
    wave_period: float = 8.0
    current_drag_coeff: float = 5.0

    reward_distance_scale: float = 1.0
    reward_velocity_weight: float = 0.1
    reward_action_weight: float = 0.05

    sim: OceanScaleSimCfg = field(default_factory=OceanScaleSimCfg)
    scene: OceanScaleSceneCfg = field(default_factory=OceanScaleSceneCfg)

    def __post_init__(self) -> None:
        self.sim.device = self.device
        self.scene.num_envs = self.num_envs


def register_oceanscale_tasks() -> str:
    if TASK_ID not in gym.envs.registration.registry:
        gym.register(
            id=TASK_ID,
            entry_point="oceanscale.training.isaaclab_env:OceanScaleDirectRLEnv",
            disable_env_checker=True,
            kwargs={"cfg": OceanScaleEnvCfg()},
        )
    return TASK_ID


class OceanScaleDirectRLEnv(gym.Env):
    """GPU-batched underwater RL env backed by OceanSim.

    Follows Isaac Lab's DirectRLEnv protocol:
    - step/reset return ``{"policy": Tensor(n_envs, obs_dim)}``
    - All computation on GPU (no CPU roundtrips in hot path)
    - Supports partial reset via ``_reset_idx``
    - Decimation: multiple physics substeps per RL step
    """

    is_vector_env = True
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        cfg: OceanScaleEnvCfg | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            cfg = OceanScaleEnvCfg()
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.physics_dt = cfg.physics_dt
        self.step_dt = cfg.physics_dt * cfg.decimation
        self.max_episode_length = int(cfg.episode_length_s / self.step_dt)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )
        self.num_observations = OBS_DIM
        self.num_actions = ACT_DIM

        self.sim = OceanSim(
            OceanSimConfig(
                vehicle=BlueROV2Heavy(),
                ocean=OceanConfig(
                    current_speed=cfg.current_speed,
                    current_direction=cfg.current_direction,
                    wave_height=cfg.wave_height,
                    wave_period=cfg.wave_period,
                ),
                n_envs=cfg.num_envs,
                dt=cfg.physics_dt,
                device=cfg.device,
                init_pos=cfg.init_pos,
                current_drag_coeff=cfg.current_drag_coeff,
            )
        )

        self._target = torch.tensor(
            cfg.target_pos, dtype=torch.float32, device=self.device
        )
        self._step_count = torch.zeros(
            cfg.num_envs, dtype=torch.int64, device=self.device
        )
        self._prev_action = torch.zeros(
            cfg.num_envs, ACT_DIM, dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # DirectRLEnv protocol
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        env_ids: list[int] | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        if env_ids is None:
            self.sim.reset()
            if self.cfg.init_pos_noise > 0:
                self._randomize_init_positions(range(self.num_envs))
            self._step_count.zero_()
            self._prev_action.zero_()
        else:
            self._reset_idx(env_ids)
        return {"policy": self._get_observations()}, {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        action = action.to(device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1)
        self._prev_action.copy_(action)

        for _ in range(self.cfg.decimation):
            self.sim.step_torch(action)

        self._step_count += 1
        obs = self._get_observations()
        reward = self._compute_reward(obs, action)
        terminated, truncated = self._get_dones(obs)

        done_ids = (terminated | truncated).nonzero(as_tuple=True)[0]
        if len(done_ids) > 0:
            self._reset_idx(done_ids.tolist())

        return {"policy": obs}, reward, terminated, truncated, {}

    def _reset_idx(self, env_ids: list[int] | range) -> None:
        ids_np = np.array(list(env_ids), dtype=np.int32)
        self.sim.reset_envs(ids_np)
        if self.cfg.init_pos_noise > 0:
            self._randomize_init_positions(env_ids)
        ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        self._step_count[ids_t] = 0
        self._prev_action[ids_t] = 0.0

    def _randomize_init_positions(self, env_ids: list[int] | range) -> None:
        noise = self.cfg.init_pos_noise
        q = self.sim.model.joint_q.numpy()
        for i in env_ids:
            q[i * 7] += np.random.uniform(-noise, noise)
            q[i * 7 + 1] += np.random.uniform(-noise, noise)
            q[i * 7 + 2] += np.random.uniform(-noise, noise)
        self.sim.model.joint_q.assign(q)

    # ------------------------------------------------------------------
    # Observations (GPU-native)
    # ------------------------------------------------------------------

    def _get_observations(self) -> torch.Tensor:
        state = self.sim.observe_torch()
        pos = state["position"]
        quat = state["orientation"]
        lin_vel = state["linear_velocity"]
        ang_vel = state["angular_velocity"]
        pos_error = self._target - pos
        depth = pos[:, 2:3]
        return torch.cat(
            [pos_error, quat, lin_vel, ang_vel, depth, self._prev_action],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pos_error = obs[:, :3]
        lin_vel = obs[:, 7:10]

        dist = torch.linalg.norm(pos_error, dim=-1)
        r_dist = torch.exp(-dist / self.cfg.reward_distance_scale)
        r_vel = -self.cfg.reward_velocity_weight * torch.linalg.norm(lin_vel, dim=-1)
        r_act = -self.cfg.reward_action_weight * torch.linalg.norm(action, dim=-1)

        return r_dist + r_vel + r_act

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_error = obs[:, :3]
        dist = torch.linalg.norm(pos_error, dim=-1)

        terminated = dist > self.cfg.oob_distance
        truncated = self._step_count >= self.max_episode_length
        return terminated, truncated

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.sim.close()
