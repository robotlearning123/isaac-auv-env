#!/usr/bin/env python3
"""Verify MarineGym Hover task runs on Isaac Sim 6 + Isaac Lab 3.

Launches the MarineGymHoverTask as a DirectRLEnv, runs 32 steps of
random actions on 4 environments, and checks for finite obs/rewards.

Usage:
    /mnt/storage/isaacsim-6.0-official/venv/bin/python \
        artifacts/isaacsim/verify_marinegym_hover_isaaclab3.py
"""
from __future__ import annotations

import json
import sys
import traceback

PYTHONPATH = "."  # Repo root


def main() -> int:
    sys.path.insert(0, PYTHONPATH)

    from isaaclab.app import AppLauncher

    launcher = AppLauncher({
        "headless": True,
        "device": "cuda:0",
        "visualizer": ["none"],
        "enable_cameras": False,
    })
    simulation_app = launcher.app

    result = {
        "app_started": True,
        "env_created": False,
        "reset_done": False,
        "steps_done": 0,
        "num_envs": 4,
        "num_steps": 32,
        "obs_shape": None,
        "reward_min": None,
        "reward_max": None,
        "terminated_count": 0,
        "truncated_count": 0,
        "passed": False,
    }

    try:
        import torch
        from oceanscale.marinegym_compat.hover_task import (
            MarineGymHoverCfg,
            MarineGymHoverIsaacCfg,
            MarineGymHoverTask,
        )

        torch.manual_seed(42)

        isaac_cfg = MarineGymHoverIsaacCfg()
        isaac_cfg.num_envs = result["num_envs"]
        isaac_cfg.sim.device = "cuda:0"
        isaac_cfg.scene.num_envs = result["num_envs"]

        hover_cfg = MarineGymHoverCfg(num_envs=result["num_envs"])

        env = MarineGymHoverTask(cfg=isaac_cfg, hover_cfg=hover_cfg)
        result["env_created"] = True

        obs, _ = env.reset()
        result["reset_done"] = True
        obs_policy = obs["policy"]
        print(f"Reset obs shape: {list(obs_policy.shape)}", flush=True)

        if not torch.isfinite(obs_policy).all():
            raise RuntimeError("non-finite observation after reset")

        for step in range(result["num_steps"]):
            actions = torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0
            actions = actions * 0.35  # Scale down
            obs, reward, terminated, truncated, _ = env.step(actions)
            obs_policy = obs["policy"]

            if not torch.isfinite(obs_policy).all():
                raise RuntimeError(f"non-finite observation at step {step + 1}")
            if not torch.isfinite(reward).all():
                raise RuntimeError(f"non-finite reward at step {step + 1}")

            r_min = float(reward.min().item())
            r_max = float(reward.max().item())
            result["reward_min"] = r_min if result["reward_min"] is None else min(result["reward_min"], r_min)
            result["reward_max"] = r_max if result["reward_max"] is None else max(result["reward_max"], r_max)
            result["terminated_count"] += int(terminated.sum().item())
            result["truncated_count"] += int(truncated.sum().item())
            result["steps_done"] += 1

        result["obs_shape"] = list(obs_policy.shape)
        result["passed"] = True
        print("MARINEGYM_HOVER_VERIFICATION_PASS", flush=True)
        print(json.dumps(result, sort_keys=True), flush=True)
        env.close()

    except BaseException as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        print("MARINEGYM_HOVER_VERIFICATION_FAIL", flush=True)
        print(result["traceback"], flush=True)
        print(json.dumps(result, sort_keys=True), flush=True)
        simulation_app.close()
        return 1

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
