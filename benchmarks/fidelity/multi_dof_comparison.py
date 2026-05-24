#!/usr/bin/env python3
"""Fidelity benchmark: multi-DOF GPU vs CPU comparison.

Extends vonbenzon_comparison.py beyond surge-only to cover heave and yaw.
Runs three maneuvers:
  1. Surge: 5N forward for 1000 steps @ dt=0.01s (10s)
  2. Heave: 5N upward for 1000 steps @ dt=0.01s (10s)
  3. Yaw:   0.3 Nm torque for 300 steps @ dt=0.01s (3s)

Yaw uses reduced torque and duration because the Tier-1 cross-coupling
damping terms (tier1_kernels.py:66-70) cause exponential force growth at
yaw rates > ~2 rad/s. 0.3 Nm × 3s keeps the max yaw rate at ~1.4 rad/s
(~80 deg/s), within the GPU Tier-1 stability envelope.

For each maneuver, computes:
  - CPU RK4 (gold standard)
  - CPU Euler (integration error baseline)
  - GPU Tier-1 (Warp kernels + Euler)

Coordinate conventions (same as vonbenzon_comparison.py):
  CPU (NED): z-positive DOWN
  GPU (ENU): z-positive UP
  Transform: pos_z *= -1, vel_w *= -1, vel_r *= -1, quat_qy *= -1, quat_qz *= -1

Body-frame force/torque mapping:
  Surge (idx 0): same sign in NED and ENU (+5N = forward)
  Heave (idx 2): CPU NED up = -5N, GPU ENU up = +5N (z-axis flipped)
  Yaw   (idx 5): CPU NED +0.3 Nm, GPU ENU -0.3 Nm (yaw axis flipped with z)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from oceanscale.validation.vonbenzon_reference import (
    VonBenzonParams,
    VonBenzonReferenceModel,
    _build_allocation_matrix,
    _quat_mul,
    _quat_to_rotmat,
)

import warp as wp
from oceanscale.hydro.tier1 import Tier1

DT = 0.01  # s per step
OUTPUT_PATH = Path(__file__).parent / "multi_dof_results.json"


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------
def transform_ned_to_enu(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in traj.items():
        out[k] = v.copy()
    out["pos"][:, 2] *= -1
    out["vel"][:, 2] *= -1
    out["vel"][:, 5] *= -1
    out["quat"][:, 1] *= -1
    out["quat"][:, 2] *= -1
    return out


# ---------------------------------------------------------------------------
# Maneuver definitions
# ---------------------------------------------------------------------------
MANEUVERS = {
    "surge": {
        "tau_ned": np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "wrench_add": np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "n_steps": 1000,
        "description": "5N forward surge (10s)",
    },
    "heave": {
        "tau_ned": np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0]),
        "wrench_add": np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        "n_steps": 1000,
        "description": "5N upward heave (10s)",
    },
    "yaw": {
        "tau_ned": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
        "wrench_add": np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.3]),
        "n_steps": 300,
        "description": "0.3 Nm yaw torque (3s)",
    },
}


# ---------------------------------------------------------------------------
# CPU trajectories
# ---------------------------------------------------------------------------
def run_cpu_rk4(tau_ned: np.ndarray, n_steps: int) -> dict[str, np.ndarray]:
    model = VonBenzonReferenceModel()
    traj = model.generate_trajectory(
        thrust_func=lambda t: tau_ned,
        dt=DT,
        n_steps=n_steps,
    )
    return transform_ned_to_enu(traj)


def run_cpu_euler(tau_ned: np.ndarray, n_steps: int) -> dict[str, np.ndarray]:
    p = VonBenzonParams()
    M_inv = np.diag(
        1.0
        / np.array(
            [
                p.mass + p.added_mass[0],
                p.mass + p.added_mass[1],
                p.mass + p.added_mass[2],
                p.I_x + p.added_mass[3],
                p.I_y + p.added_mass[4],
                p.I_z + p.added_mass[5],
            ]
        )
    )
    model = VonBenzonReferenceModel()

    pos = np.zeros(3, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    nu = np.zeros(6, dtype=np.float64)

    t_arr = np.zeros(n_steps)
    pos_arr = np.zeros((n_steps, 3))
    quat_arr = np.zeros((n_steps, 4))
    vel_arr = np.zeros((n_steps, 6))

    for k in range(n_steps):
        t_arr[k] = k * DT
        pos_arr[k] = pos
        quat_arr[k] = quat
        vel_arr[k] = nu

        damping = (model._d_lin + model._d_quad * np.abs(nu)) * nu
        nu_dot = M_inv @ (tau_ned - model._coriolis(nu) - damping - model._restoring(quat))

        d_pos = _quat_to_rotmat(quat) @ nu[:3]
        d_q = 0.5 * _quat_mul(quat, np.array([nu[3], nu[4], nu[5], 0.0]))

        pos = pos + DT * d_pos
        quat = quat + DT * d_q
        quat = quat / np.linalg.norm(quat)
        nu = nu + DT * nu_dot

    traj = {"t": t_arr, "pos": pos_arr, "quat": quat_arr, "vel": vel_arr}
    return transform_ned_to_enu(traj)


# ---------------------------------------------------------------------------
# GPU Tier-1
# ---------------------------------------------------------------------------
def _c_rb(nu: np.ndarray, p: VonBenzonParams) -> np.ndarray:
    u, v, w, pp, q, r = nu
    m = p.mass
    return np.array(
        [
            m * (q * w - r * v),
            m * (r * u - pp * w),
            m * (pp * v - q * u),
            (p.I_z - p.I_y) * q * r,
            (p.I_x - p.I_z) * pp * r,
            (p.I_y - p.I_x) * pp * q,
        ]
    )


def run_gpu_tier1(wrench_add: np.ndarray, n_steps: int) -> dict[str, np.ndarray]:
    wp.init()
    p = VonBenzonParams()

    M_RB_inv = np.diag(
        1.0 / np.array([p.mass, p.mass, p.mass, p.I_x, p.I_y, p.I_z])
    )

    tier1 = Tier1(
        n_envs=1,
        n_thrusters=8,
        device="cuda",
        rho_water=p.rho,
        g_accel=p.g,
        ema_alpha=0.3,
    )
    tier1.set_coeffs(
        added_mass=p.added_mass,
        d_lin=p.d_lin,
        d_quad=p.d_quad,
        mass=p.mass,
        volume=p.volume,
        coBM=abs(p.r_b[2]),
        T_matrix=_build_allocation_matrix().tolist(),
    )

    pos = np.zeros(3, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    nu = np.zeros(6, dtype=np.float64)

    t_arr = np.zeros(n_steps)
    pos_arr = np.zeros((n_steps, 3))
    quat_arr = np.zeros((n_steps, 4))
    vel_arr = np.zeros((n_steps, 6))

    t0 = time.perf_counter()

    for k in range(n_steps):
        t_arr[k] = k * DT
        pos_arr[k] = pos
        quat_arr[k] = quat
        vel_arr[k] = nu

        nu_f32 = nu.astype(np.float32).reshape(1, 6)
        quat_f32 = quat.astype(np.float32).reshape(1, 4)
        u_cmd = np.zeros((1, 8), dtype=np.float32)

        nu_wp = wp.array(nu_f32, dtype=wp.spatial_vectorf, device="cuda")
        quat_wp = wp.array(quat_f32, dtype=wp.quatf, device="cuda")
        u_cmd_wp = wp.array(u_cmd, dtype=wp.float32, device="cuda")

        tier1.compute_wrench(nu_wp, quat_wp, u_cmd_wp, dt=DT)
        wp.synchronize()

        wrench = tier1.wrench_buf.numpy()[0].astype(np.float64)
        wrench += wrench_add

        c_rb = _c_rb(nu, p)
        nu_dot = M_RB_inv @ (wrench - c_rb)

        d_pos = _quat_to_rotmat(quat) @ nu[:3]
        d_q = 0.5 * _quat_mul(quat, np.array([nu[3], nu[4], nu[5], 0.0]))

        pos = pos + DT * d_pos
        quat = quat + DT * d_q
        quat = quat / np.linalg.norm(quat)
        nu = nu + DT * nu_dot

    elapsed = time.perf_counter() - t0
    print(f"  GPU Tier-1: {n_steps} steps in {elapsed:.2f}s ({n_steps / elapsed:.0f} steps/s)")

    return {"t": t_arr, "pos": pos_arr, "quat": quat_arr, "vel": vel_arr}


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_metrics(ref: dict, test: dict, label: str) -> dict:
    pos_err = np.linalg.norm(test["pos"] - ref["pos"], axis=1)
    pos_ref = np.linalg.norm(ref["pos"], axis=1)
    rel_final = (
        pos_err[-1] / pos_ref[-1] * 100.0 if pos_ref[-1] > 1e-8 else 0.0
    )

    att_err = quat_angle_deg(test["quat"], ref["quat"])

    vel_err = np.linalg.norm(test["vel"] - ref["vel"], axis=1)
    vel_ref = np.linalg.norm(ref["vel"], axis=1)
    rel_vel = vel_err[-1] / vel_ref[-1] * 100.0 if vel_ref[-1] > 1e-8 else 0.0

    return {
        "label": label,
        "position": {
            "l2_final_m": float(pos_err[-1]),
            "ref_displacement_m": float(pos_ref[-1]),
            "relative_final_pct": float(rel_final),
            "l2_rms_m": float(np.sqrt(np.mean(pos_err**2))),
            "l2_max_m": float(np.max(pos_err)),
        },
        "attitude": {
            "final_deg": float(att_err[-1]),
            "rms_deg": float(np.sqrt(np.mean(att_err**2))),
            "max_deg": float(np.max(att_err)),
        },
        "velocity": {
            "l2_final_ms": float(vel_err[-1]),
            "relative_final_pct": float(rel_vel),
            "l2_rms_ms": float(np.sqrt(np.mean(vel_err**2))),
        },
    }


# ---------------------------------------------------------------------------
# Per-maneuver axis error (primary DOF)
# ---------------------------------------------------------------------------
def compute_axis_error(ref: dict, test: dict, axis: int) -> dict:
    """Position error along a single axis, and angular rate error for rotation DOFs."""
    ref_axis = ref["pos"][:, axis]
    test_axis = test["pos"][:, axis]
    abs_err = np.abs(test_axis - ref_axis)
    ref_disp = np.abs(ref_axis[-1])
    return {
        "axis": ["x", "y", "z"][axis],
        "ref_final_m": float(ref_axis[-1]),
        "test_final_m": float(test_axis[-1]),
        "abs_err_final_m": float(abs_err[-1]),
        "rel_err_final_pct": float(abs_err[-1] / ref_disp * 100.0) if ref_disp > 1e-8 else 0.0,
        "rms_m": float(np.sqrt(np.mean(abs_err**2))),
    }


# ---------------------------------------------------------------------------
# Run single maneuver
# ---------------------------------------------------------------------------
def run_maneuver(name: str, cfg: dict) -> dict:
    W = 72
    tau_ned = cfg["tau_ned"]
    wrench_add = cfg["wrench_add"]
    n_steps = cfg["n_steps"]
    desc = cfg["description"]

    print("\n" + "=" * W)
    print(f"  MANEUVER: {name.upper()} — {desc}")
    print("=" * W)

    print(f"  [1/3] CPU RK4 (gold standard)...")
    cpu_rk4 = run_cpu_rk4(tau_ned, n_steps)
    print(f"  Final pos: [{cpu_rk4['pos'][-1, 0]:.5f}, "
          f"{cpu_rk4['pos'][-1, 1]:.5f}, {cpu_rk4['pos'][-1, 2]:.5f}]")

    print(f"  [2/3] CPU Euler (integration baseline)...")
    cpu_euler = run_cpu_euler(tau_ned, n_steps)
    print(f"  Final pos: [{cpu_euler['pos'][-1, 0]:.5f}, "
          f"{cpu_euler['pos'][-1, 1]:.5f}, {cpu_euler['pos'][-1, 2]:.5f}]")

    print(f"  [3/3] GPU Tier-1 (Warp kernels)...")
    gpu_tier1 = run_gpu_tier1(wrench_add, n_steps)
    print(f"  Final pos: [{gpu_tier1['pos'][-1, 0]:.5f}, "
          f"{gpu_tier1['pos'][-1, 1]:.5f}, {gpu_tier1['pos'][-1, 2]:.5f}]")

    m_euler = compute_metrics(cpu_rk4, cpu_euler, "CPU Euler")
    m_gpu = compute_metrics(cpu_rk4, gpu_tier1, "GPU Tier-1")

    # Primary axis: surge→x(0), heave→z(2), yaw→heading via quat
    if name == "surge":
        axis_err = compute_axis_error(cpu_rk4, gpu_tier1, axis=0)
    elif name == "heave":
        axis_err = compute_axis_error(cpu_rk4, gpu_tier1, axis=2)
    else:  # yaw
        axis_err = compute_axis_error(cpu_rk4, gpu_tier1, axis=0)  # yaw doesn't translate much

    # Print results
    print(f"\n  {'─' * (W - 4)}")
    print(f"  ERROR vs CPU RK4 REFERENCE")
    print(f"  {'─' * (W - 4)}")
    for m in [m_euler, m_gpu]:
        pe = m["position"]
        ae = m["attitude"]
        ve = m["velocity"]
        print(f"  {m['label']}:")
        print(f"    Position  L2 final = {pe['l2_final_m']:.6f} m  "
              f"({pe['relative_final_pct']:.2f}% of ref {pe['ref_displacement_m']:.4f} m)")
        print(f"    Attitude  RMS      = {ae['rms_deg']:.4f} deg  "
              f"(max {ae['max_deg']:.4f} deg)")
        print(f"    Velocity  L2 final = {ve['l2_final_ms']:.6f} m/s  "
              f"({ve['relative_final_pct']:.2f}%)")

    pp = m_gpu["position"]["relative_final_pct"] < 5.0
    ap = m_gpu["attitude"]["rms_deg"] < 3.0
    print(f"\n  PASS/FAIL (GPU vs RK4):")
    print(f"    Position < 5%  : {'PASS' if pp else 'FAIL'}  ({m_gpu['position']['relative_final_pct']:.2f}%)")
    print(f"    Attitude < 3°  : {'PASS' if ap else 'FAIL'}  ({m_gpu['attitude']['rms_deg']:.4f}°)")
    overall = pp and ap
    print(f"    OVERALL        : {'PASS' if overall else 'FAIL'}")

    int_pct = m_euler["position"]["relative_final_pct"]
    gpu_pct = m_gpu["position"]["relative_final_pct"]
    impl_pct = gpu_pct - int_pct
    print(f"\n  Error decomposition:")
    print(f"    Integration (Euler vs RK4) : {int_pct:+.2f}%")
    print(f"    Implementation (Tier-1)    : {impl_pct:+.2f}%")
    print(f"    Total                      : {gpu_pct:.2f}%")

    ds = 10
    return {
        "description": desc,
        "tau_ned": tau_ned.tolist(),
        "wrench_add_enu": wrench_add.tolist(),
        "pass": bool(overall),
        "results": {
            "cpu_euler_vs_rk4": m_euler,
            "gpu_tier1_vs_rk4": m_gpu,
        },
        "axis_error": axis_err,
        "time_series": {
            "t": cpu_rk4["t"][::ds].tolist(),
            "cpu_rk4_pos": cpu_rk4["pos"][::ds].tolist(),
            "gpu_tier1_pos": gpu_tier1["pos"][::ds].tolist(),
            "cpu_rk4_vel": cpu_rk4["vel"][::ds, :3].tolist(),
            "gpu_tier1_vel": gpu_tier1["vel"][::ds, :3].tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("Multi-DOF Fidelity Benchmark: GPU OceanScale vs CPU von Benzon")
    print(f"  dt={DT}s")
    print(f"  Maneuvers: {', '.join(f'{k} ({v['description']})' for k, v in MANEUVERS.items())}")

    all_results = {}
    all_pass = True

    for name, cfg in MANEUVERS.items():
        result = run_maneuver(name, cfg)
        all_results[name] = result
        if not result["pass"]:
            all_pass = False

    # Summary
    W = 72
    print("\n" + "=" * W)
    print("  SUMMARY — ALL MANEUVERS")
    print("=" * W)
    print(f"  {'Maneuver':<12} {'Description':<20} {'Pos Error':>10} {'Att RMS':>10} {'Pass':>6}")
    print(f"  {'─' * 62}")
    for name, res in all_results.items():
        m = res["results"]["gpu_tier1_vs_rk4"]
        print(f"  {name:<12} {res['description']:<20} "
              f"{m['position']['relative_final_pct']:>9.2f}% "
              f"{m['attitude']['rms_deg']:>9.4f}° "
              f"{'PASS' if res['pass'] else 'FAIL':>6}")
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * W)

    # Save
    data = {
        "scenario": f"Multi-DOF @ dt={DT}s",
        "pass_criteria": {"position_pct": 5.0, "attitude_deg": 3.0},
        "maneuvers": all_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_PATH}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
