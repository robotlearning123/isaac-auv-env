"""VecEnv wrappers for GPU-batched environments.

OceanScaleVecEnv wraps NewtonEnv as a gymnasium.vector.VectorEnv.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from oceanscale.newton_env import NewtonEnv

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
AnyArray = NDArray[Any]


class OceanScaleVecEnv(gym.vector.VectorEnv[FloatArray, FloatArray, AnyArray]):
    """Vectorized environment backed by a single GPU-batched NewtonEnv.

    Observation space (per env): 18-dim float32 vector
      [pos(3), quat(4), vel(3), ang_vel(3), prev_action(6)]  (placeholder,
      override via obs_fn if needed)

    Action space (per env): Box(-1, 1, (6,)) float32 -- normalised thruster
    commands for 6-DOF.

    Reward: negative distance to goal (default: origin).  Override via
    reward_fn callback.
    """

    def __init__(
        self,
        n_envs: int = 16,
        dt: float = 1.0 / 240.0,
        max_episode_steps: int = 1000,
        device: str = "cuda",
        obs_fn: Any | None = None,
        reward_fn: Any | None = None,
        termination_fn: Any | None = None,
    ) -> None:
        super().__init__()

        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}
        self.num_envs = n_envs
        self._n_envs = n_envs
        self._max_steps = max_episode_steps
        self._step_count = np.zeros(n_envs, dtype=np.int64)
        self._prev_action = np.zeros((n_envs, 6), dtype=np.float32)

        self._env = NewtonEnv(n_envs=n_envs, dt=dt, device=device)

        obs_dim = 19
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_envs, obs_dim), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n_envs, 6), dtype=np.float32)

        self._obs_fn = obs_fn
        self._reward_fn = reward_fn
        self._termination_fn = termination_fn

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._env.reset()
        self._step_count[:] = 0
        self._prev_action[:] = 0.0

        obs = self._get_obs()
        return obs, {}

    def step(
        self, actions: FloatArray
    ) -> tuple[FloatArray, FloatArray, BoolArray, BoolArray, dict[str, Any]]:
        import warp as wp

        actions = cast(FloatArray, np.asarray(actions, dtype=np.float32))
        if actions.shape == (6,):
            actions = cast(FloatArray, np.tile(actions, (self._n_envs, 1)))
        assert actions.shape == (self._n_envs, 6), (
            f"Expected actions shape ({self._n_envs}, 6), got {actions.shape}"
        )

        u_cmd: Any = wp.array(actions, dtype=wp.float32, device=self._env.device)
        self._env.step(u_cmd)
        self._prev_action = actions.copy()
        self._step_count += 1

        obs = self._get_obs()
        rewards = self._get_reward(obs)

        positions = obs[:, :3]
        if self._termination_fn is not None:
            terminated = cast(BoolArray, self._termination_fn(obs))
        else:
            terminated = cast(BoolArray, np.linalg.norm(positions, axis=-1) > 100.0)

        truncated = self._step_count >= self._max_steps

        ret_terminated = terminated.copy()
        ret_truncated = truncated.copy()

        done = terminated | truncated
        if np.any(done):
            done_ids = np.where(done)[0].tolist()
            self._env.reset(env_ids=done_ids)
            self._step_count[done] = 0
            self._prev_action[done] = 0.0
            fresh_obs = self._get_obs_envs(done_ids)
            obs[done_ids] = fresh_obs

        infos: dict[str, Any] = {}
        return obs, rewards, ret_terminated, ret_truncated, infos

    def _get_obs(self) -> FloatArray:
        if self._obs_fn is not None:
            return cast(FloatArray, self._obs_fn(self._env, self._prev_action))
        state = self._env.get_state()
        obs = np.concatenate(
            [
                state["position"],
                state["orientation"],
                state["velocity"],
                state["angular_velocity"],
                self._prev_action,
            ],
            axis=-1,
        )
        return cast(FloatArray, obs.astype(np.float32))

    def _get_obs_envs(self, env_ids: list[int] | np.ndarray) -> FloatArray:
        state = self._env.get_state()
        ids = np.asarray(env_ids)
        obs = np.concatenate(
            [
                state["position"][ids],
                state["orientation"][ids],
                state["velocity"][ids],
                state["angular_velocity"][ids],
                self._prev_action[ids],
            ],
            axis=-1,
        )
        return cast(FloatArray, obs.astype(np.float32))

    def _get_reward(self, obs: FloatArray) -> FloatArray:
        if self._reward_fn is not None:
            return cast(FloatArray, self._reward_fn(obs))
        positions = obs[:, :3]
        return cast(FloatArray, -np.linalg.norm(positions, axis=-1).astype(np.float32))

    def close(self, **kwargs: Any) -> None:
        pass
