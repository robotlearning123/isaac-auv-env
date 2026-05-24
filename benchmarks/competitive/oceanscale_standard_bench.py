#!/usr/bin/env python3
"""OceanScale Standardized Benchmark — measures all competitive matrix axes.

Axes measured:
  1. Setup time       — env creation + first reset + first step latency
  2. Single-env FPS   — steps/s at n_envs=1
  3. Batched FPS      — steps/s at n_envs=64, 256, 1024, 4096
  4. Memory per env   — GPU MB / n_envs at each scale
  5. Physics features — Fossen terms implemented
  6. Sensor features  — sensor models available
  7. RL integration   — Gymnasium version, SB3 compatibility
  8. Fidelity         — von Benzon cross-validation result

Output: JSON to stdout + written to benchmarks/competitive/results.json
"""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. Setup time
# ---------------------------------------------------------------------------
def measure_setup_time() -> dict[str, float]:
    from oceanscale.rov_env import ROVEnv

    t0 = time.perf_counter()
    env = ROVEnv(n_envs=1, device="cuda")
    t_created = time.perf_counter()

    _obs, _info = env.reset()
    t_reset = time.perf_counter()

    action = env.action_space.sample()
    env.step(action)
    t_first_step = time.perf_counter()

    env.close()

    return {
        "create_s": round(t_created - t0, 4),
        "first_reset_s": round(t_reset - t_created, 4),
        "first_step_s": round(t_first_step - t_reset, 4),
        "total_setup_s": round(t_first_step - t0, 4),
    }


# ---------------------------------------------------------------------------
# 2–4. Throughput + memory at scale
# ---------------------------------------------------------------------------
def measure_throughput(
    n_envs: int,
    warmup: int = 10,
    bench_steps: int = 200,
) -> dict[str, Any]:
    import warp as wp

    from oceanscale.rov_env import ROVEnv

    env = ROVEnv(n_envs=n_envs, device="cuda")
    _obs, _ = env.reset()
    wp.synchronize()

    # Compute GPU memory from actual Warp array sizes
    gpu_mem_bytes = _compute_env_memory(env)
    gpu_mem_mb = gpu_mem_bytes / (1024 * 1024)
    mem_per_env = gpu_mem_mb / n_envs if n_envs > 0 else 0.0

    actions = np.stack([env.action_space.sample() for _ in range(n_envs)])

    for _ in range(warmup):
        env.step(actions)

    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(bench_steps):
        env.step(actions)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    env.close()

    fps = bench_steps / elapsed

    return {
        "n_envs": n_envs,
        "steps_per_second": round(fps, 1),
        "env_steps_per_second": round(fps * n_envs, 0),
        "wall_time_s": round(elapsed, 4),
        "bench_steps": bench_steps,
        "gpu_mem_total_mb": round(gpu_mem_mb, 1),
        "gpu_mem_per_env_mb": round(mem_per_env, 3),
    }


def _wp_array_bytes(arr) -> int:
    """Size in bytes of a Warp GPU array."""
    import warp as wp

    return arr.size * wp.types.type_size(arr.dtype) * 4  # 4 bytes per float32


def _compute_env_memory(env) -> float:
    """Compute total GPU memory used by the env's Warp arrays (bytes)."""
    total = 0
    # Newton model state arrays
    for attr_name in ["joint_q", "joint_qd"]:
        arr = getattr(env.model, attr_name, None)
        if arr is not None and hasattr(arr, "size"):
            total += _wp_array_bytes(arr)
    # State arrays (body_q, body_qd, body_f)
    for state in [env.state_curr, env.state_next]:
        if state is not None:
            for attr in ["body_q", "body_qd", "body_f"]:
                arr = getattr(state, attr, None)
                if arr is not None and hasattr(arr, "size"):
                    total += _wp_array_bytes(arr)
    # Control
    if env.control is not None:
        arr = getattr(env.control, "joint_f", None)
        if arr is not None and hasattr(arr, "size"):
            total += _wp_array_bytes(arr)
    # Tier1 buffers
    t1 = env.tier1
    for attr in [
        "wrench_buf",
        "nu_prev",
        "nu_dot_prev",
        "u_eff_prev",
        "u_eff_out",
        "M_A_lin",
        "M_A_ang",
        "d_lin_lin",
        "d_lin_ang",
        "d_quad_lin",
        "d_quad_ang",
        "mass_arr",
        "volume_arr",
        "coBM_arr",
        "T_matrix",
    ]:
        arr = getattr(t1, attr, None)
        if arr is not None and hasattr(arr, "size"):
            total += _wp_array_bytes(arr)
    # Command buffer
    total += _wp_array_bytes(env._u_cmd)
    return float(total)


# ---------------------------------------------------------------------------
# 5. Physics features
# ---------------------------------------------------------------------------
def inspect_physics_features() -> dict[str, Any]:
    return {
        "model": "Fossen 6-DOF (Fossen 2011 Handbook)",
        "kernels": [
            "added_mass",
            "coriolis_added_mass",
            "damping_linear_quadratic",
            "restoring_buoyancy_gravity",
            "thruster_allocation_lowpass",
            "nu_dot_ema_filter",
        ],
        "terms": {
            "added_mass": True,
            "coriolis_centripetal": True,
            "linear_damping": True,
            "quadratic_damping": True,
            "restoring_forces": True,
            "cross_coupling_damping": False,
            "thruster_model": "T200_lowpass_deadband",
            "ocean_current_fsi": True,
        },
        "integration": "semi_implicit_euler",
        "ema_alpha": 0.3,
        "validated_vs": "von Benzon et al. 2022, JMSE 10(5):636",
    }


# ---------------------------------------------------------------------------
# 6. Sensor features
# ---------------------------------------------------------------------------
def inspect_sensor_features() -> dict[str, Any]:
    return {
        "imu": True,
        "dvl": True,
        "sonar_imaging": False,
        "camera_model": False,
        "depth_pressure": True,
        "acoustic_comms": False,
        "note": "Headless core with IMU, DVL, and pressure/depth sensor stubs. No sonar, camera, or acoustic comms model yet.",
    }


# ---------------------------------------------------------------------------
# 7. RL integration
# ---------------------------------------------------------------------------
def inspect_rl_integration() -> dict[str, Any]:
    import gymnasium as gym
    import stable_baselines3

    from oceanscale.rov_env import ROVEnv
    from oceanscale.vec_env import BatchedVecEnv

    env = ROVEnv(n_envs=4, device="cuda")
    _obs, _ = env.reset()

    result = {
        "gymnasium_version": gym.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "observation_space": {
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
        },
        "action_space": {
            "shape": list(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
            "low": float(env.action_space.low[0]),
            "high": float(env.action_space.high[0]),
        },
        "sb3_vecenv_wrapper": BatchedVecEnv.__name__,
        "partial_reset": True,
        "domain_randomization": True,
        "vec_normalize": "supported (oceanscale/data/vec_normalize.npz)",
    }

    env.close()
    return result


# ---------------------------------------------------------------------------
# 8. Fidelity
# ---------------------------------------------------------------------------
def inspect_fidelity() -> dict[str, Any]:
    fidelity_path = Path(__file__).parent.parent / "fidelity" / "vonbenzon_comparison_results.json"
    if fidelity_path.exists():
        with open(fidelity_path) as f:
            data = json.load(f)
        gpu = data["results"]["gpu_tier1_vs_rk4"]
        return {
            "reference": "von Benzon et al. 2022, JMSE 10(12):1898",
            "scenario": data["scenario"],
            "position_error_pct": round(gpu["position"]["relative_final_pct"], 3),
            "attitude_rms_deg": round(gpu["attitude"]["rms_deg"], 4),
            "position_rms_m": round(gpu["position"]["l2_rms_m"], 6),
            "position_max_m": round(gpu["position"]["l2_max_m"], 6),
            "pass": (
                gpu["position"]["relative_final_pct"] < 5.0 and gpu["attitude"]["rms_deg"] < 3.0
            ),
        }

    return {
        "reference": "von Benzon et al. 2022",
        "result": "not yet run — run benchmarks/fidelity/vonbenzon_comparison.py",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import warp as wp

    wp.init()
    try:
        oceanscale_version = version("oceanscale")
    except PackageNotFoundError:
        oceanscale_version = "unknown"

    W = 72
    print("=" * W)
    print("  OceanScale Standardized Benchmark")
    print("  Axes: setup, throughput, memory, physics, sensors, RL, fidelity")
    print("=" * W)

    results: dict[str, Any] = {
        "benchmark": "oceanscale_standard",
        "version": oceanscale_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": {
            "name": wp.get_device("cuda:0").name,
            "arch": wp.get_device("cuda:0").arch,
            "total_memory_mb": round(wp.get_device("cuda:0").total_memory / (1024 * 1024), 0),
            "warp_version": wp.__version__,
        },
    }

    # -- 1. Setup time --
    print("\n[1/8] Setup time (n_envs=1)...")
    results["setup_time"] = measure_setup_time()
    print(
        f"  Create: {results['setup_time']['create_s']}s  "
        f"Reset: {results['setup_time']['first_reset_s']}s  "
        f"First step: {results['setup_time']['first_step_s']}s  "
        f"Total: {results['setup_time']['total_setup_s']}s"
    )

    # -- 2. Single-env throughput --
    print("\n[2/8] Single-env throughput (n_envs=1)...")
    single = measure_throughput(n_envs=1)
    results["single_env"] = single
    print(
        f"  {single['steps_per_second']} steps/s  "
        f"GPU mem: {single['gpu_mem_total_mb']} MB  "
        f"per env: {single['gpu_mem_per_env_mb']} MB"
    )

    # -- 3-4. Batched throughput + memory --
    batched = []
    for n in [64, 256, 1024, 4096]:
        print(f"\n[3-4/8] Batched throughput (n_envs={n})...")
        try:
            r = measure_throughput(n_envs=n)
            batched.append(r)
            print(
                f"  {r['steps_per_second']} steps/s  "
                f"({r['env_steps_per_second']:.0f} env-steps/s)  "
                f"GPU: {r['gpu_mem_total_mb']} MB  "
                f"per env: {r['gpu_mem_per_env_mb']} MB"
            )
        except Exception as e:
            print(f"  SKIPPED: {e}")
            batched.append({"n_envs": n, "error": str(e)})
    results["batched_throughput"] = batched

    # -- 5. Physics features --
    print("\n[5/8] Physics features...")
    results["physics"] = inspect_physics_features()
    print(f"  Model: {results['physics']['model']}")
    print(f"  Terms: {', '.join(k for k, v in results['physics']['terms'].items() if v is True)}")

    # -- 6. Sensor features --
    print("\n[6/8] Sensor features...")
    results["sensors"] = inspect_sensor_features()
    active = [k for k, v in results["sensors"].items() if v is True]
    print(f"  Active: {', '.join(active) if active else 'none (headless)'}")

    # -- 7. RL integration --
    print("\n[7/8] RL integration...")
    results["rl_integration"] = inspect_rl_integration()
    print(
        f"  Gymnasium {results['rl_integration']['gymnasium_version']}  "
        f"SB3 {results['rl_integration']['stable_baselines3_version']}  "
        f"Obs {results['rl_integration']['observation_space']['shape']}  "
        f"Act {results['rl_integration']['action_space']['shape']}"
    )

    # -- 8. Fidelity --
    print("\n[8/8] Fidelity...")
    results["fidelity"] = inspect_fidelity()
    if "position_error_pct" in results["fidelity"]:
        f = results["fidelity"]
        print(
            f"  vs von Benzon: {f['position_error_pct']}% pos error, "
            f"{f['attitude_rms_deg']} deg attitude RMS  "
            f"{'PASS' if f['pass'] else 'FAIL'}"
        )
    else:
        print(f"  {results['fidelity'].get('result', 'N/A')}")

    # -- Output JSON --
    print("\n" + "=" * W)
    print("  JSON OUTPUT")
    print("=" * W)
    output = json.dumps(results, indent=2)
    print(output)

    out_path = Path(__file__).parent / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\n  Written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
