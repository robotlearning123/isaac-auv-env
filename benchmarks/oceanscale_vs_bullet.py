"""OceanScale (Newton+Warp GPU) vs PyBullet (CPU) throughput benchmark.

Runs both environments for N env-steps under comparable conditions.
Init noise (position/yaw randomization) is disabled (set to 0.0) so both
environments start deterministically at the hover target — matching PyBullet's
fixed reset behavior.

Reports per-env FPS, total throughput, wall-clock time, and memory.

Usage:
    # Single run
    uv run python benchmarks/oceanscale_vs_bullet.py

    # Sweep n_envs {1,4,16,64} × 3 runs for variance
    uv run python benchmarks/oceanscale_vs_bullet.py --sweep --runs 3

    # Custom
    uv run python benchmarks/oceanscale_vs_bullet.py --n-steps 50000 --n-envs 16 --runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SWEEP_N_ENVS = [1, 4, 16, 64]


def _get_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def benchmark_oceanscale(n_envs: int, n_steps: int) -> dict:
    """Run OceanScale ROVEnv benchmark with init noise disabled."""
    import tracemalloc

    tracemalloc.start()

    from oceanscale.rov_env import ROVEnv

    # Disable init noise for fair comparison with PyBullet (which has no init randomization)
    env = ROVEnv(
        n_envs=n_envs,
        device="cuda",
        init_pos_noise_std=0.0,
        init_yaw_noise_std=0.0,
    )
    _obs, _info = env.reset()

    zero_action = np.zeros((n_envs, 6), dtype=np.float32)

    for _ in range(10):
        _obs, _reward, _term, _trunc, _info = env.step(zero_action)

    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        _obs, _reward, _term, _trunc, _info = env.step(zero_action)

    elapsed = time.perf_counter() - t0
    rss_after = _get_rss_mb()

    total_env_steps = n_steps * n_envs
    throughput = total_env_steps / elapsed
    per_env_fps = throughput / n_envs
    time_1m = 1_000_000 / throughput

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    env.close()

    return {
        "engine": "OceanScale (Newton+Warp GPU)",
        "n_envs": n_envs,
        "n_steps": n_steps,
        "total_env_steps": int(total_env_steps),
        "wall_clock_s": round(elapsed, 4),
        "throughput_env_steps_per_sec": round(throughput, 1),
        "per_env_fps": round(per_env_fps, 1),
        "time_for_1M_env_steps_s": round(time_1m, 2),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 1),
    }


def benchmark_pybullet(n_steps: int) -> dict:
    """Run PyBullet BlueROV2 benchmark (single env, CPU)."""
    import sys
    import tracemalloc

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    tracemalloc.start()

    from bullet_bluerov_env import BulletBlueROV2Env

    env = BulletBlueROV2Env()
    _obs, _info = env.reset()

    zero_action = np.zeros(6, dtype=np.float32)

    for _ in range(10):
        _obs, _reward, _term, _trunc, _info = env.step(zero_action)

    rss_before = _get_rss_mb()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        _obs, _reward, _term, _trunc, _info = env.step(zero_action)

    elapsed = time.perf_counter() - t0
    rss_after = _get_rss_mb()

    throughput = n_steps / elapsed
    time_1m = 1_000_000 / throughput

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    env.close()

    return {
        "engine": "PyBullet (CPU)",
        "n_envs": 1,
        "n_steps": n_steps,
        "total_env_steps": n_steps,
        "wall_clock_s": round(elapsed, 4),
        "throughput_env_steps_per_sec": round(throughput, 1),
        "per_env_fps": round(throughput, 1),
        "time_for_1M_env_steps_s": round(time_1m, 2),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 1),
    }


def run_sweep(n_steps: int, n_runs: int, output_path: Path) -> dict:
    """Run PyBullet at n=1 and OceanScale at n in {1,4,16,64}, each for n_runs."""
    results: dict = {
        "config": {
            "mode": "sweep",
            "n_steps": n_steps,
            "oceanscale_n_envs_list": SWEEP_N_ENVS,
            "n_runs": n_runs,
            "policy": "zero (no thruster input)",
            "task": "hover at [0, 0, -1.5]",
            "dt": "1/240 s",
            "init_noise": "disabled (0.0 pos, 0.0 yaw) for fair comparison",
        },
        "runs": [],
    }

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        run_data: dict = {"run": run_idx + 1}

        # PyBullet (always n=1)
        print(f"\n  [PyBullet n=1] {n_steps:,} steps (CPU)...")
        bullet_result = benchmark_pybullet(n_steps)
        print(f"    Throughput: {bullet_result['throughput_env_steps_per_sec']:,.0f} env-steps/s")
        run_data["pybullet"] = bullet_result

        # OceanScale at each n_envs
        for n_envs in SWEEP_N_ENVS:
            total_os = n_steps * n_envs
            print(f"\n  [OceanScale n={n_envs}] {n_steps:,} steps x {n_envs} envs = {total_os:,} env-steps (GPU)...")
            os_result = benchmark_oceanscale(n_envs, n_steps)
            print(f"    Throughput: {os_result['throughput_env_steps_per_sec']:,.0f} env-steps/s  |  Per-env: {os_result['per_env_fps']:,.0f} fps")
            run_data[f"oceanscale_n{n_envs}"] = os_result

        results["runs"].append(run_data)

    # Summary statistics
    bullet_fps = [r["pybullet"]["throughput_env_steps_per_sec"] for r in results["runs"]]
    bullet_1m = [r["pybullet"]["time_for_1M_env_steps_s"] for r in results["runs"]]

    summary: dict = {
        "pybullet_throughput_mean": round(float(np.mean(bullet_fps)), 1),
        "pybullet_throughput_std": round(float(np.std(bullet_fps)), 1),
        "pybullet_1M_mean_s": round(float(np.mean(bullet_1m)), 2),
    }

    for n_envs in SWEEP_N_ENVS:
        key = f"oceanscale_n{n_envs}"
        os_fps = [r[key]["throughput_env_steps_per_sec"] for r in results["runs"]]
        os_per_env = [r[key]["per_env_fps"] for r in results["runs"]]
        os_1m = [r[key]["time_for_1M_env_steps_s"] for r in results["runs"]]

        summary[f"oceanscale_n{n_envs}_throughput_mean"] = round(float(np.mean(os_fps)), 1)
        summary[f"oceanscale_n{n_envs}_throughput_std"] = round(float(np.std(os_fps)), 1)
        summary[f"oceanscale_n{n_envs}_per_env_mean"] = round(float(np.mean(os_per_env)), 1)
        summary[f"oceanscale_n{n_envs}_per_env_std"] = round(float(np.std(os_per_env)), 1)
        summary[f"oceanscale_n{n_envs}_1M_mean_s"] = round(float(np.mean(os_1m)), 2)
        summary[f"speedup_n{n_envs}_vs_bullet"] = round(float(np.mean(os_fps)) / float(np.mean(bullet_fps)), 2)

    results["summary"] = summary

    # Print summary table
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Engine':<30} {'Throughput (mean±std)':>25} {'Per-env FPS':>15} {'1M steps (s)':>15} {'Speedup':>10}")
    print("-" * 100)
    print(
        f"{'PyBullet n=1':<30} "
        f"{summary['pybullet_throughput_mean']:>12,.0f} ± {summary['pybullet_throughput_std']:>5.0f} "
        f"{summary['pybullet_throughput_mean']:>15,.0f} "
        f"{summary['pybullet_1M_mean_s']:>15.1f} "
        f"{'1.00x':>10}"
    )
    for n_envs in SWEEP_N_ENVS:
        tp = summary[f"oceanscale_n{n_envs}_throughput_mean"]
        tp_std = summary[f"oceanscale_n{n_envs}_throughput_std"]
        pe = summary[f"oceanscale_n{n_envs}_per_env_mean"]
        t1m = summary[f"oceanscale_n{n_envs}_1M_mean_s"]
        spd = summary[f"speedup_n{n_envs}_vs_bullet"]
        label = f"OceanScale n={n_envs}"
        print(
            f"{label:<30} "
            f"{tp:>12,.0f} ± {tp_std:>5.0f} "
            f"{pe:>15,.0f} "
            f"{t1m:>15.1f} "
            f"{spd:>9.2f}x"
        )

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    return results


def run_single(n_steps: int, n_envs: int, n_runs: int, output_path: Path) -> dict:
    """Original mode: OceanScale at one n_envs vs PyBullet."""
    results: dict = {
        "config": {
            "mode": "single",
            "n_steps": n_steps,
            "n_envs_oceanscale": n_envs,
            "n_runs": n_runs,
            "init_noise": "disabled (0.0 pos, 0.0 yaw) for fair comparison",
        },
        "runs": [],
    }

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{n_runs}")
        print(f"{'='*60}")

        print(f"\n[PyBullet] Running {n_steps:,} steps (single env, CPU)...")
        bullet_result = benchmark_pybullet(n_steps)
        print(f"  Throughput: {bullet_result['throughput_env_steps_per_sec']:,.0f} env-steps/s")

        total_os = n_steps * n_envs
        print(f"\n[OceanScale] Running {n_steps:,} steps x {n_envs} envs = {total_os:,} env-steps (GPU)...")
        os_result = benchmark_oceanscale(n_envs, n_steps)
        print(f"  Throughput: {os_result['throughput_env_steps_per_sec']:,.0f} env-steps/s")

        results["runs"].append({"run": run_idx + 1, "pybullet": bullet_result, "oceanscale": os_result})

    bullet_fps = [r["pybullet"]["throughput_env_steps_per_sec"] for r in results["runs"]]
    os_fps = [r["oceanscale"]["throughput_env_steps_per_sec"] for r in results["runs"]]
    os_per_env = [r["oceanscale"]["per_env_fps"] for r in results["runs"]]
    bullet_1m = [r["pybullet"]["time_for_1M_env_steps_s"] for r in results["runs"]]
    os_1m = [r["oceanscale"]["time_for_1M_env_steps_s"] for r in results["runs"]]

    results["summary"] = {
        "pybullet_throughput_mean": round(float(np.mean(bullet_fps)), 1),
        "pybullet_throughput_std": round(float(np.std(bullet_fps)), 1),
        "oceanscale_throughput_mean": round(float(np.mean(os_fps)), 1),
        "oceanscale_throughput_std": round(float(np.std(os_fps)), 1),
        "oceanscale_per_env_fps_mean": round(float(np.mean(os_per_env)), 1),
        "oceanscale_per_env_fps_std": round(float(np.std(os_per_env)), 1),
        "throughput_speedup": round(float(np.mean(os_fps)) / float(np.mean(bullet_fps)), 2),
        "per_env_speedup": round(float(np.mean(os_per_env)) / float(np.mean(bullet_fps)), 2),
        "pybullet_1M_mean_s": round(float(np.mean(bullet_1m)), 2),
        "oceanscale_1M_mean_s": round(float(np.mean(os_1m)), 2),
    }

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="OceanScale vs PyBullet benchmark")
    parser.add_argument("--n-steps", type=int, default=100_000, help="Steps per run")
    parser.add_argument("--n-envs", type=int, default=4, help="OceanScale parallel envs (single mode)")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs for variance")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep OceanScale at n_envs in {1,4,16,64} and PyBullet at n=1",
    )
    parser.add_argument("--output", type=str, default="benchmarks/oceanscale_vs_bullet_results.json")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.sweep:
        run_sweep(args.n_steps, args.runs, output_path)
    else:
        run_single(args.n_steps, args.n_envs, args.runs, output_path)


if __name__ == "__main__":
    main()
