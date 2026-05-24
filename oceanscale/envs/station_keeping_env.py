"""StationKeepingEnv — Station keeping under ocean current disturbance.

Extends ROVEnv with sinusoidal ocean current as external force.
Observation: 29-dim (26 base + 3 current velocity body frame).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from numpy.typing import NDArray

from oceanscale.rov_env import ROVEnv

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
StepReturn = tuple[
    FloatArray,
    FloatArray | float,
    BoolArray | bool,
    BoolArray | bool,
    dict[str, Any],
]


def _wp_numpy(array: Any) -> NDArray[Any]:
    return cast(NDArray[Any], array.numpy())


class CurrentStationKeepingEnv(ROVEnv):
    """BlueROV2 station-keeping under varying ocean current.

    Current model (per env, per episode):
        V(t) = speed * [cos(dir + amp*sin(2pi*t/T)),
                         sin(dir + amp*sin(2pi*t/T)),
                         0.15*sin(0.3*2pi*t/T)]
    where speed, dir, T, amp are randomized per episode.
    """

    def __init__(
        self,
        n_envs: int = 64,
        n_thrusters: int = 8,
        device: str = "cuda",
        current_coupling_strength: float = 8.0,
        **kwargs: Any,
    ) -> None:
        randomize_sensor_noise = "sensor_noise_std" not in kwargs
        kwargs.setdefault("max_episode_steps", 2400)
        kwargs.setdefault("use_domain_randomization", True)
        kwargs.setdefault("init_pos_noise_std", 0.5)
        kwargs.setdefault("init_yaw_noise_std", 0.5)
        super().__init__(n_envs=n_envs, n_thrusters=n_thrusters, device=device, **kwargs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),
            dtype=np.float32,
        )
        self.current_coupling_strength = current_coupling_strength

        self._current_params = np.zeros((n_envs, 5), dtype=np.float32)
        self._current_world = np.zeros((n_envs, 3), dtype=np.float32)
        self._pos_error_history = np.full((n_envs, 300), np.inf, dtype=np.float32)
        self._history_idx = np.zeros(n_envs, dtype=np.int32)
        self._randomize_sensor_noise = randomize_sensor_noise

    def _build(self) -> None:
        super()._build()
        self._current_params = np.zeros((self.n_envs, 5), dtype=np.float32)
        self._current_world = np.zeros((self.n_envs, 3), dtype=np.float32)
        self._pos_error_history = np.full((self.n_envs, 300), np.inf, dtype=np.float32)
        self._history_idx = np.zeros(self.n_envs, dtype=np.int32)
        # Sentinel: base step() calls _apply_ocean_current() when _fluid is not None
        self._fluid = cast(Any, object())

    # ------------------------------------------------------------------
    # Current model
    # ------------------------------------------------------------------

    def _sample_current_params(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        speed = self.np_random.uniform(0.2, 2.0, n).astype(np.float32)
        direction = self.np_random.uniform(0, 2 * np.pi, n).astype(np.float32)
        period = self.np_random.uniform(5.0, 30.0, n).astype(np.float32)
        amplitude = (self.np_random.uniform(0, 1, n) * 0.5 * speed).astype(np.float32)
        phase = self.np_random.uniform(0, 2 * np.pi, n).astype(np.float32)
        self._current_params[env_ids] = np.stack(
            [speed, direction, period, amplitude, phase],
            axis=-1,
        )

    def _compute_current(self) -> np.ndarray:
        t = (self._step_count * self.dt).astype(np.float32) + self._current_params[:, 4]
        speed = self._current_params[:, 0]
        direction = self._current_params[:, 1]
        period = self._current_params[:, 2]
        amplitude = self._current_params[:, 3]

        phase_t = amplitude * np.sin(2 * np.pi * t / period)
        vx = speed * np.cos(direction + phase_t)
        vy = speed * np.sin(direction + phase_t)
        vz = 0.15 * speed * np.sin(0.3 * 2 * np.pi * t / period)

        self._current_world = np.stack([vx, vy, vz], axis=-1).astype(np.float32)
        return self._current_world

    def _apply_ocean_current(self) -> None:
        current_world = self._compute_current()

        assert self.state_curr.body_qd is not None
        assert self.state_curr.body_q is not None
        assert self.state_curr.body_f is not None

        body_qd = _wp_numpy(self.state_curr.body_qd)
        body_vel_body = body_qd[:, 0:3]
        quat = _wp_numpy(self.state_curr.body_q)[:, 3:7]
        body_vel_world = self._rotate_to_world(quat, body_vel_body)

        v_rel = current_world - body_vel_world
        force_world = self.current_coupling_strength * v_rel
        force_body = self._rotate_to_body(quat, force_world)

        body_f = _wp_numpy(self.state_curr.body_f)
        body_f[:, 0:3] += force_body
        wp.copy(
            self.state_curr.body_f,
            wp.array(body_f, dtype=wp.float32, device=self.device),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reset_indices(self, env_ids: np.ndarray) -> None:
        super()._reset_indices(env_ids)
        self._sample_current_params(env_ids)
        self._compute_current()
        self._pos_error_history[env_ids] = np.inf
        self._history_idx[env_ids] = 0
        if self._randomize_sensor_noise:
            self.sensor_noise_std = float(self.np_random.uniform(0.01, 0.05))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options, env_ids=env_ids)
        if self.n_envs == 1:
            obs = obs.squeeze(0)
        info["success"] = False
        return obs, info

    def step(  # type: ignore[override]
        self,
        action: np.ndarray,
    ) -> StepReturn:
        obs, reward, terminated, truncated, info = super().step(action)

        pos, _, _ = self._get_body_state()
        pos_err = np.linalg.norm(pos - self.target_pos, axis=-1)

        idx = self._history_idx % 300
        self._pos_error_history[np.arange(self.n_envs), idx] = pos_err
        self._history_idx += 1

        mean_errors = np.sum(self._pos_error_history, axis=1) / 300.0
        success = (mean_errors < 0.5) & (self._history_idx >= 300)

        info["success"] = bool(np.any(success))
        info["per_env_success"] = success
        info["mean_pos_error"] = mean_errors
        if self.n_envs == 1:
            return (
                cast(FloatArray, obs.squeeze(0)),
                float(reward.item()),
                bool(terminated.item()),
                bool(truncated.item()),
                info,
            )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation & reward overrides
    # ------------------------------------------------------------------

    def _get_obs_for(self, env_ids: np.ndarray) -> np.ndarray:
        base_obs = super()._get_obs_for(env_ids)
        obs = np.zeros((len(env_ids), 29), dtype=np.float32)
        obs[:, :26] = base_obs
        assert self.state_curr.body_q is not None
        quat = _wp_numpy(self.state_curr.body_q)[env_ids, 3:7]
        obs[:, 26:29] = self._rotate_to_body(quat, self._current_world[env_ids])
        return obs

    def _compute_reward_with_components(
        self,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        pos, _, vel = self._get_body_state()

        pos_err = np.clip(np.linalg.norm(self.target_pos - pos, axis=-1), 0, 5.0)
        vel_norm = np.clip(np.linalg.norm(vel[:, :3], axis=-1), 0, 5.0)
        act_norm = np.linalg.norm(self._prev_action, axis=-1)

        r_pos = 0.5 * np.exp(-pos_err / 1.0)
        r_vel = 0.3 * np.exp(-vel_norm / 0.5)
        r_act = 0.2 * np.exp(-act_norm / 1.0)
        in_zone = (pos_err < 0.5).astype(np.float32)
        r_zone = 1.0 * in_zone

        reward_np = (r_pos + r_vel + r_act + r_zone).astype(np.float32)
        reward = torch.from_numpy(reward_np).to(self.device)
        info = {
            "reward_distance": float(np.mean(r_pos)),
            "reward_action": float(np.mean(r_act)),
            "reward_velocity": float(np.mean(r_vel)),
            "reward_holding": float(np.mean(r_zone)),
        }
        return reward, info
