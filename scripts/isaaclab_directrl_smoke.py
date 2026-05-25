#!/usr/bin/env python3
"""Run a real Isaac Lab DirectRLEnv smoke and write machine-readable evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "isaacsim" / "isaaclab_directrl_smoke.json"
DEFAULT_TASK = "Isaac-Cartpole-Direct-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the installed Isaac Lab runtime can reset and step a DirectRLEnv.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fabric", action="store_true")
    parser.add_argument(
        "--package-overlay-dir",
        type=Path,
        default=None,
        help=(
            "Optional site-packages directory used to preload Python packages such as "
            "warp and newton without adding the whole directory to PYTHONPATH."
        ),
    )
    return parser.parse_args()


def _tensors_finite(value: Any) -> bool:
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tensors_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tensors_finite(item) for item in value)
    return True


def _write_result(output_json: Path | None, result: dict[str, Any]) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _preload_package_from_overlay(package_name: str, overlay_dir: Path) -> dict[str, str]:
    package_dir = overlay_dir / package_name
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"{package_name} package is missing from overlay: {package_dir}")

    overlay_path = str(overlay_dir.resolve())
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    previous_meta_path = sys.meta_path[:]
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if "omni.ext._impl.fast_importer.FastFinder" not in repr(finder)
    ]
    sys.path.insert(0, overlay_path)
    try:
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {package_name} from overlay: {init_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path = [path for path in sys.path if path != overlay_path]
        sys.meta_path = previous_meta_path

    return {
        "name": package_name,
        "version": str(getattr(module, "__version__", "unknown")),
        "file": str(getattr(module, "__file__", init_path)),
    }


def _preload_oceanscale_runtime_overlay(overlay_dir: Path | None) -> list[dict[str, str]]:
    if overlay_dir is None:
        return []
    import numpy  # noqa: F401
    import torch  # noqa: F401

    resolved = overlay_dir.resolve()
    return [
        _preload_package_from_overlay("warp", resolved),
        _preload_package_from_overlay("newton", resolved),
    ]


def run_smoke(
    *,
    task: str,
    device: str,
    num_envs: int,
    seed: int,
    use_fabric: bool,
    package_overlay_dir: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    preloaded_packages: list[dict[str, str]] = []
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    env = None
    oceanscale_task_registered = False
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        try:
            from oceanscale.training.isaaclab_env import (
                OCEANSCALE_UNDERWATER_TASK_ID,
                register_oceanscale_isaaclab_tasks,
            )
        except ImportError:
            pass
        else:
            if task == OCEANSCALE_UNDERWATER_TASK_ID:
                register_oceanscale_isaaclab_tasks()
                oceanscale_task_registered = True

        cfg = parse_env_cfg(task, device=device, num_envs=num_envs, use_fabric=use_fabric)
        if hasattr(cfg, "seed"):
            cfg.seed = seed

        preloaded_packages = _preload_oceanscale_runtime_overlay(package_overlay_dir)
        env = gym.make(task, cfg=cfg, render_mode=None)
        obs, _info = env.reset(seed=seed)
        action_dim = env.unwrapped.action_space.shape[0]
        action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        next_obs, rewards, terminated, truncated, _extras = env.step(action)

        result = {
            "ok": True,
            "api": "Isaac Lab DirectRLEnv",
            "task": task,
            "isaaclab_version": importlib.metadata.version("isaaclab"),
            "isaaclab_tasks_version": importlib.metadata.version("isaaclab_tasks"),
            "num_envs": int(env.unwrapped.num_envs),
            "device": str(env.unwrapped.device),
            "use_fabric": use_fabric,
            "package_overlay_dir": str(package_overlay_dir) if package_overlay_dir else None,
            "preloaded_packages": preloaded_packages,
            "oceanscale_task_registered": oceanscale_task_registered,
            "observation_space": str(env.unwrapped.observation_space),
            "action_space": str(env.unwrapped.action_space),
            "action_shape": list(action.shape),
            "reset_obs_finite": _tensors_finite(obs),
            "step_obs_finite": _tensors_finite(next_obs),
            "reward_shape": list(rewards.shape),
            "reward_finite": bool(torch.isfinite(rewards).all().item()),
            "terminated_shape": list(terminated.shape),
            "truncated_shape": list(truncated.shape),
            "terminated_any": bool(torch.any(terminated).item()),
            "truncated_any": bool(torch.any(truncated).item()),
        }
        _write_result(output_json, result)
        return result
    except Exception as exc:
        result = {
            "ok": False,
            "api": "Isaac Lab DirectRLEnv",
            "task": task,
            "device": device,
            "num_envs": num_envs,
            "use_fabric": use_fabric,
            "package_overlay_dir": str(package_overlay_dir) if package_overlay_dir else None,
            "preloaded_packages": preloaded_packages,
            "oceanscale_task_registered": oceanscale_task_registered,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_result(output_json, result)
        return result
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    output_json = args.output_json.resolve()
    result = run_smoke(
        task=args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        use_fabric=args.use_fabric,
        package_overlay_dir=args.package_overlay_dir,
        output_json=output_json,
    )
    if not output_json.exists():
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
