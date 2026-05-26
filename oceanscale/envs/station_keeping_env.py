"""StationKeepingEnv — Station keeping under sinusoidal ocean current disturbance.

Uses OceanSim as the physics backend (Newton + Tier1 Fossen hydrodynamics).
Applies a time-varying sinusoidal current as external disturbance on top of
the baseline simulation.

Observation: 29-dim (26 base state + 3 current velocity in body frame).
Action: 6-dim wrench command [-1, 1].
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from gymnasium import spaces

from oceanscale.sim import OceanConfig, OceanSim, OceanSimConfig
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


def _rotate_to_body(quat: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors to body frame via quaternion (xyzw)."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack(
        [
            vx - qw * t0 - qy * t2 + qz * t1,
            vy - qw * t1 - qz * t0 + qx * t2,
            vz - qw * t2 - qx * t1 + qy * t0,
        ],
        dim=-1,
    )


def _rotate_to_world(quat: torch.Tensor, vec_body: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors to world frame via quaternion (xyzw)."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_body[:, 0], vec_body[:, 1], vec_body[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack(
        [
            vx + qw * t0 + qy * t2 - qz * t1,
            vy + qw * t1 + qz * t0 - qx * t2,
            vz + qw * t2 + qx * t1 - qy * t0,
        ],
        dim=-1,
    )


class CurrentStationKeepingEnv(gym.Env[Any, Any]):
    """BlueROV2 station-keeping under varying sinusoidal ocean current.

    OceanSim handles rigid-body physics + Tier1 Fossen hydrodynamics.
    This env layers a time-varying sinusoidal current disturbance on top
    and computes station-keeping reward (position error + velocity + action).

    Current model (per env, per episode):
        V(t) = speed * [cos(dir + amp*sin(2pi*t/T)),
                         sin(dir + amp*sin(2pi*t/T)),
                         0.15*sin(0.3*2pi*t/T)]
    where speed, dir, T, amp are randomized per episode.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        n_envs: int = 64,
        device: str = "cuda",
        decimation: int = 12,
        max_episode_steps: int = 2400,
        current_coupling_strength: float = 8.0,
        target_pos: tuple[float, float, float] | None = None,
        init_pos_noise_std: float = 0.5,
        init_yaw_noise_std: float = 0.5,
        sensor_noise_std: float = 0.02,
        current_speed: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        self.decimation = decimation
        self.max_episode_steps = max_episode_steps
        self.current_coupling_strength = current_coupling_strength
        self.sensor_noise_std = sensor_noise_std
        self.init_pos_noise_std = init_pos_noise_std
        self.init_yaw_noise_std = init_yaw_noise_std

        if target_pos is None:
            target_pos = (0.0, 0.0, -1.5)
        self.target_pos = torch.tensor(target_pos, dtype=torch.float32, device=device)

        self.sim = OceanSim(
            OceanSimConfig(
                vehicle=BlueROV2Heavy(),
                ocean=OceanConfig(current_speed=current_speed),
                n_envs=n_envs,
                dt=1.0 / 240.0,
                device=device,
                init_pos=target_pos,
            )
        )
        self.dt = self.sim.dt * decimation

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
        )

        n = n_envs
        self._current_params = torch.zeros(n, 5, device=device)
        self._current_world = torch.zeros(n, 3, device=device)
        self._pos_error_history = torch.full((n, 300), float("inf"), device=device)
        self._history_idx = torch.zeros(n, dtype=torch.long, device=device)
        self._prev_action = torch.zeros(n, 6, device=device)
        self._step_count = torch.zeros(n, dtype=torch.long, device=device)
        self._episode_steps = 0

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Sinusoidal current model (GPU-native)
    # ------------------------------------------------------------------

    def _sample_current_params(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        speed = torch.tensor(
            self._rng.uniform(0.2, 2.0, n), dtype=torch.float32, device=self.device,
        )
        direction = torch.tensor(
            self._rng.uniform(0, 2 * np.pi, n), dtype=torch.float32, device=self.device,
        )
        period = torch.tensor(
            self._rng.uniform(5.0, 30.0, n), dtype=torch.float32, device=self.device,
        )
        amplitude = 0.5 * speed * torch.tensor(
            self._rng.uniform(0, 1, n), dtype=torch.float32, device=self.device,
        )
        phase = torch.tensor(
            self._rng.uniform(0, 2 * np.pi, n), dtype=torch.float32, device=self.device,
        )
        self._current_params[env_ids] = torch.stack(
            [speed, direction, period, amplitude, phase], dim=-1,
        )

    def _compute_current(self) -> torch.Tensor:
        t = self._step_count.float() * self.dt + self._current_params[:, 4]
        speed = self._current_params[:, 0]
        direction = self._current_params[:, 1]
        period = self._current_params[:, 2]
        amplitude = self._current_params[:, 3]

        phase_t = amplitude * torch.sin(2 * np.pi * t / period)
        vx = speed * torch.cos(direction + phase_t)
        vy = speed * torch.sin(direction + phase_t)
        vz = 0.15 * speed * torch.sin(0.3 * 2 * np.pi * t / period)

        self._current_world = torch.stack([vx, vy, vz], dim=-1)
        return self._current_world

    def _apply_current_force(self) -> None:
        current_world = self._compute_current()
        obs = self.sim.observe_torch()
        quat = obs["orientation"]
        body_vel_world = _rotate_to_world(quat, obs["linear_velocity"])

        v_rel = current_world - body_vel_world
        force_world = self.current_coupling_strength * v_rel
        force_body = _rotate_to_body(quat, force_world)

        body_f = self.sim.state_curr.body_f
        body_f_np = body_f.numpy()
        force_np = force_body.cpu().numpy()
        body_f_np[: self.n_envs, 0:3] += force_np
        body_f.assign(body_f_np)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> torch.Tensor:
        obs = self.sim.observe_torch()
        pos = obs["position"]
        quat = obs["orientation"]
        lin_vel = obs["linear_velocity"]
        ang_vel = obs["angular_velocity"]

        pos_err = self.target_pos.unsqueeze(0) - pos
        depth_err = (self.target_pos[2] - pos[:, 2]).unsqueeze(-1)
        roll = torch.atan2(
            2.0 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2),
        )
        pitch = torch.asin(
            2.0 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]).clamp(-1.0, 1.0),
        )
        heading_err = torch.stack([roll, pitch], dim=-1)
        current_body = _rotate_to_body(quat, self._current_world)

        noise = torch.randn_like(pos_err) * self.sensor_noise_std

        base_obs = torch.cat(
            [
                pos_err + noise[: , :3],
                quat,
                lin_vel,
                ang_vel,
                depth_err + noise[:, 0:1],
                heading_err + noise[:, :2],
                self._prev_action,
                torch.zeros(self.n_envs, 4, device=self.device),
            ],
            dim=-1,
        )
        return torch.cat([base_obs, current_body], dim=-1)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, float]]:
        obs = self.sim.observe_torch()
        pos = obs["position"]
        vel = obs["linear_velocity"]

        pos_err = (self.target_pos.unsqueeze(0) - pos).norm(dim=-1).clamp(0, 5.0)
        vel_norm = vel.norm(dim=-1).clamp(0, 5.0)
        act_norm = self._prev_action.norm(dim=-1)

        r_pos = 0.5 * torch.exp(-pos_err / 1.0)
        r_vel = 0.3 * torch.exp(-vel_norm / 0.5)
        r_act = 0.2 * torch.exp(-act_norm / 1.0)
        in_zone = (pos_err < 0.5).float()
        r_zone = 1.0 * in_zone

        reward = r_pos + r_vel + r_act + r_zone
        info = {
            "reward_distance": float(r_pos.mean()),
            "reward_action": float(r_act.mean()),
            "reward_velocity": float(r_vel.mean()),
            "reward_holding": float(r_zone.mean()),
        }
        return reward, info

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.sim.reset()
        self._episode_steps = 0
        self._step_count.zero_()
        self._prev_action.zero_()

        all_ids = torch.arange(self.n_envs, device=self.device)
        self._sample_current_params(all_ids)
        self._compute_current()
        self._pos_error_history.fill_(float("inf"))
        self._history_idx.zero_()

        obs = self._get_obs()
        info: dict[str, Any] = {"success": False}
        if self.n_envs == 1:
            return obs[0].cpu().numpy(), info
        return obs.cpu().numpy(), info

    def step(
        self,
        action: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.n_envs, -1)
        action = action.clamp(-1.0, 1.0)
        self._prev_action = action.clone()

        for _ in range(self.decimation):
            self._apply_current_force()
            self.sim.step_torch(action)
            self._step_count += 1

        self._episode_steps += 1

        reward, reward_info = self._compute_reward()

        obs = self.sim.observe_torch()
        pos = obs["position"]
        pos_err = (self.target_pos.unsqueeze(0) - pos).norm(dim=-1)

        idx = self._history_idx % 300
        self._pos_error_history.scatter_(1, idx.unsqueeze(-1), pos_err.unsqueeze(-1))
        self._history_idx += 1

        mean_errors = self._pos_error_history.mean(dim=1)
        success = (mean_errors < 0.5) & (self._history_idx >= 300)

        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        truncated = torch.full(
            (self.n_envs,), self._episode_steps >= self.max_episode_steps, device=self.device,
        )

        info: dict[str, Any] = {
            "success": bool(success.any()),
            "per_env_success": success,
            "mean_pos_error": mean_errors,
            **reward_info,
        }

        gym_obs = self._get_obs()
        if self.n_envs == 1:
            return (
                gym_obs[0].cpu().numpy(),
                float(reward[0]),
                bool(terminated[0]),
                bool(truncated[0]),
                info,
            )
        return gym_obs.cpu().numpy(), reward.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info

    def close(self) -> None:
        self.sim.close()
