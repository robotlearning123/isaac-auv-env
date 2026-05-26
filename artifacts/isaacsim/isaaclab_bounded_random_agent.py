#!/usr/bin/env python3
"""Bounded Isaac Lab random-agent smoke using official task launch utilities."""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of vectorized environments.")
    parser.add_argument("--steps", type=int, default=16, help="Number of random-action environment steps.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    add_launcher_args(parser)
    return setup_preset_cli(parser)


def main() -> int:
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
        obs, _ = env.reset()
        print(f"[INFO]: Reset observation type: {type(obs).__name__}", flush=True)

        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, reward, terminated, truncated, _ = env.step(actions)
            if step == 0 or step == args_cli.steps - 1:
                print(
                    "[INFO]: step="
                    f"{step + 1} reward_mean={reward.float().mean().item():.6f} "
                    f"terminated_any={bool(terminated.any())} truncated_any={bool(truncated.any())}",
                    flush=True,
                )

        env.close()
    print("[INFO]: Bounded random-agent smoke complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
