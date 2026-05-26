"""Standard benchmark tasks and metrics for OceanScale environments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from oceanscale.envs import CurrentStationKeepingEnv, DockingApproachEnv, WaypointFollowingEnv
from oceanscale.envs.docking_env import DockingApproachEnvCfg


@dataclass
class BenchmarkResult:
    task_name: str
    n_envs: int
    total_steps: int
    wall_time_s: float
    fps: float = 0.0
    mean_reward: float = 0.0
    final_reward: float = 0.0
    success_rate: float = 0.0

    def __post_init__(self) -> None:
        self.fps = (self.n_envs * self.total_steps) / self.wall_time_s if self.wall_time_s > 0 else 0.0


def run_benchmark(
    env_factory: Callable[[int, str], Any],
    n_envs: int,
    n_steps: int,
    device: str = "cuda:0",
) -> BenchmarkResult:
    """Run a benchmark: create env, step with random actions, measure throughput."""
    env = env_factory(n_envs, device)
    obs, _ = env.reset(seed=42)

    rewards: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = np.zeros((n_envs,) + env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if hasattr(reward, "cpu"):
            r = float(reward.cpu().mean())
        elif isinstance(reward, np.ndarray):
            r = float(np.mean(reward))
        else:
            r = float(reward)
        rewards.append(r)
    elapsed = time.perf_counter() - t0

    env.close()

    return BenchmarkResult(
        task_name=env.__class__.__name__,
        n_envs=n_envs,
        total_steps=n_steps,
        wall_time_s=elapsed,
        mean_reward=float(np.mean(rewards)),
        final_reward=rewards[-1],
        success_rate=float(np.mean(info.get("success", 0))) if isinstance(info.get("success"), (np.ndarray, np.generic)) else 0.0,
    )


# --- Env factories (unified interface for heterogeneous constructors) ---

def _make_station_keeping(n_envs: int, device: str) -> CurrentStationKeepingEnv:
    return CurrentStationKeepingEnv(n_envs=n_envs, device=device, max_episode_steps=10_000)


def _make_docking(n_envs: int, device: str) -> DockingApproachEnv:
    return DockingApproachEnv(DockingApproachEnvCfg(n_envs=n_envs, device=device, max_episode_steps=10_000))


def _make_waypoint(n_envs: int, device: str) -> WaypointFollowingEnv:
    return WaypointFollowingEnv(n_envs=n_envs, device=device, time_limit=120.0)


# --- Standard benchmark configs ---

STANDARD_BENCHMARKS: list[dict[str, Any]] = [
    {"name": "station_keeping_64", "factory": _make_station_keeping, "n_envs": 64, "n_steps": 1000},
    {"name": "docking_32",        "factory": _make_docking,          "n_envs": 32, "n_steps": 500},
    {"name": "waypoint_16",       "factory": _make_waypoint,         "n_envs": 16, "n_steps": 300},
]


def run_all_benchmarks(device: str = "cuda:0") -> list[BenchmarkResult]:
    """Run all standard benchmark configurations."""
    results: list[BenchmarkResult] = []
    for cfg in STANDARD_BENCHMARKS:
        print(f"Running {cfg['name']}...")
        r = run_benchmark(cfg["factory"], cfg["n_envs"], cfg["n_steps"], device)
        r.task_name = cfg["name"]
        results.append(r)
    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    header = f"{'Task':<22} {'Envs':>5} {'Steps':>6} {'Time(s)':>8} {'FPS':>10} {'MeanR':>8} {'FinalR':>8} {'Succ%':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.task_name:<22} {r.n_envs:>5} {r.total_steps:>6} "
            f"{r.wall_time_s:>8.2f} {r.fps:>10.0f} "
            f"{r.mean_reward:>8.3f} {r.final_reward:>8.3f} "
            f"{r.success_rate * 100:>5.1f}%"
        )


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_results(results)
