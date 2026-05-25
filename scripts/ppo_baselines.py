"""PPO baselines for all 4 OceanScale tasks: hover, waypoint, docking, station_keeping.

Trains PPO for 50k env steps each, logs reward at ~5k intervals.
Uses BatchedVecEnv wrapper for GPU-batched envs.
"""

import importlib
import json
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from oceanscale.vec_env import BatchedVecEnv

OBS_CLIP = 10.0


class RewardLogger(BaseCallback):
    def __init__(self, eval_env, interval=5000):
        super().__init__()
        self.eval_env = eval_env
        self.interval = interval
        self.data = []
        self._next_log = 0

    def _on_step(self):
        if self.num_timesteps >= self._next_log:
            obs, _ = self.eval_env.reset()
            total_r = 0.0
            for _ in range(200):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_r += float(np.asarray(reward).mean())
                if np.any(terminated) or np.any(truncated):
                    break
            self.data.append((self.num_timesteps, total_r))
            print(f"  Step {self.num_timesteps:>6}: reward={total_r:.2f}")
            self._next_log += self.interval
        return True


def make_safe_vec_env(env):
    """Wrap BatchedVecEnv to sanitize NaN/Inf in observations and rewards."""
    original_step_wait = env.step_wait
    original_reset = env.reset

    def safe_step_wait():
        obs, reward, done, infos = original_step_wait()
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP)
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-100.0)
        return obs, reward, done, infos

    def safe_reset(**kwargs):
        obs = original_reset(**kwargs)
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP)
        return obs

    env.step_wait = safe_step_wait
    env.reset = safe_reset
    return env


def make_safe_gym_env(env):
    """Wrap gymnasium Env to sanitize NaN/Inf in observations and rewards."""
    original_step = env.step
    original_reset = env.reset

    def safe_step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP)
        reward = float(np.nan_to_num(np.asarray(reward), nan=0.0, posinf=0.0, neginf=-100.0).mean())
        return obs, reward, terminated, truncated, info

    def safe_reset(**kwargs):
        obs, info = original_reset(**kwargs)
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP)
        return obs, info

    env.step = safe_step
    env.reset = safe_reset
    return env


TASKS = [
    ("hover", "oceanscale.rov_env", "ROVEnv", {}),
    ("waypoint", "oceanscale.envs.waypoint_env", "WaypointFollowingEnv", {}),
    ("docking", "oceanscale.envs.docking_env", "DockingApproachEnv", {}),
    ("station_keeping", "oceanscale.envs.station_keeping_env", "CurrentStationKeepingEnv", {}),
]


def main():
    results = {}
    for name, module, cls_name, kwargs in TASKS:
        print(f"\n=== {name} ===")
        try:
            mod = importlib.import_module(module)
            EnvCls = getattr(mod, cls_name)

            train_env = make_safe_vec_env(BatchedVecEnv(EnvCls(n_envs=64, **kwargs)))
            eval_env = make_safe_gym_env(EnvCls(n_envs=1, **kwargs))

            logger = RewardLogger(eval_env, interval=5000)
            model = PPO(
                "MlpPolicy",
                train_env,
                n_steps=128,
                batch_size=512,
                verbose=0,
                device="cpu",
                learning_rate=3e-4,
            )

            t0 = time.time()
            model.learn(total_timesteps=50_000, callback=logger)
            elapsed = time.time() - t0

            results[name] = {
                "curve": logger.data,
                "final_reward": logger.data[-1][1] if logger.data else None,
                "training_time": round(elapsed, 1),
                "samples_per_s": round(50_000 / elapsed),
            }
            print(f"  Time: {elapsed:.1f}s, Samples/s: {50000/elapsed:.0f}")

            del model, train_env, eval_env
            import torch
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    with open("install_log/ppo_baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to install_log/ppo_baselines.json")


if __name__ == "__main__":
    main()
