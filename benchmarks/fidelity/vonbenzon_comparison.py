#!/usr/bin/env python3
"""Fidelity benchmark: GPU OceanScale Tier-1 vs CPU von Benzon 6-DOF Fossen model.

Scenario: BlueROV2 Heavy at rest, then constant 5N forward surge for 10s.

Three trajectories are computed:
  1. CPU RK4   — von Benzon reference, RK4 integration (gold standard)
  2. CPU Euler — same Fossen equations, forward Euler (integration-error baseline)
  3. GPU Tier1 — Warp kernel force computation + forward Euler

Error decomposition:
  Integration error  = CPU Euler − CPU RK4
  Implementation err = GPU Tier1 − CPU Euler  (EMA, cross-coupling, restoring)
  Total error        = GPU Tier1 − CPU RK4

Coordinate conventions:
  - CPU (von Benzon): z-positive DOWN (NED-like), per Fossen convention
  - GPU (Newton/Warp): z-positive UP (ENU), per Newton/Warp convention
  - CPU trajectory is transformed to ENU before comparison:
      pos_z *= -1,  vel_w *= -1,  vel_r *= -1
      quat: (qx, -qy, -qz, qw) — flip pitch and yaw axes

Pass criteria:
  - Position L2 error < 5% of reference final displacement
  - Attitude error < 3 degrees RMS

Reference: von Benzon et al. 2022, JMSE 10(12):1898, DOI 10.3390/jmse10121898
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STEPS = 1000
DT = 0.01  # s  →  10 s total
SURGE_FORCE = 5.0  # N forward
OUTPUT_PATH = Path(__file__).parent / "vonbenzon_comparison_results.json"


# ---------------------------------------------------------------------------
# Coordinate convention: NED (CPU) → ENU (GPU)
# ---------------------------------------------------------------------------
def transform_ned_to_enu(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform a trajectory from NED (z-down) to ENU (z-up) convention.

    CPU model uses Fossen NED convention (vonbenzon_reference.py:154-166).
    GPU uses Newton/Warp ENU convention (tier1_kernels.py:130-131).
    Transformation: flip z in position, heave velocity, yaw rate, and pitch/yaw
    quaternion axes.
    """
    out = {}
    for k, v in traj.items():
        out[k] = v.copy()

    # Position: flip z
    out["pos"][:, 2] *= -1

    # Velocity [u, v, w, p, q, r]: flip heave (w) and yaw rate (r)
    out["vel"][:, 2] *= -1  # w
    out["vel"][:, 5] *= -1  # r

    # Quaternion (qx, qy, qz, qw): flip pitch and yaw rotation axes
    out["quat"][:, 1] *= -1  # qy (pitch)
    out["quat"][:, 2] *= -1  # qz (yaw)

    return out


# ---------------------------------------------------------------------------
# CPU reference trajectories
# ---------------------------------------------------------------------------
def run_cpu_rk4() -> dict[str, np.ndarray]:
    """Von Benzon 6-DOF Fossen model, RK4 integration. Gold standard.

    Source: oceanscale/validation/vonbenzon_reference.py:177-224
    """
    model = VonBenzonReferenceModel()
    traj = model.generate_trajectory(
        thrust_func=lambda t: np.array([SURGE_FORCE, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dt=DT,
        n_steps=N_STEPS,
    )
    return transform_ned_to_enu(traj)


def run_cpu_euler() -> dict[str, np.ndarray]:
    """Same Fossen equations, forward Euler integration.

    Isolates integration-method error from Tier-1 implementation differences.
    Combined mass matrix M = M_RB + M_A (same as von Benzon line 114-122).
    Returns trajectory in ENU convention (z-up).
    """
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

    t_arr = np.zeros(N_STEPS)
    pos_arr = np.zeros((N_STEPS, 3))
    quat_arr = np.zeros((N_STEPS, 4))
    vel_arr = np.zeros((N_STEPS, 6))

    for k in range(N_STEPS):
        t_arr[k] = k * DT
        pos_arr[k] = pos
        quat_arr[k] = quat
        vel_arr[k] = nu

        tau = np.array([SURGE_FORCE, 0.0, 0.0, 0.0, 0.0, 0.0])
        damping = (model._d_lin + model._d_quad * np.abs(nu)) * nu
        nu_dot = M_inv @ (tau - model._coriolis(nu) - damping - model._restoring(quat))

        d_pos = _quat_to_rotmat(quat) @ nu[:3]
        d_q = 0.5 * _quat_mul(quat, np.array([nu[3], nu[4], nu[5], 0.0]))

        pos = pos + DT * d_pos
        quat = quat + DT * d_q
        quat = quat / np.linalg.norm(quat)
        nu = nu + DT * nu_dot

    traj = {"t": t_arr, "pos": pos_arr, "quat": quat_arr, "vel": vel_arr}
    return transform_ned_to_enu(traj)


# ---------------------------------------------------------------------------
# GPU Tier-1 standalone
# ---------------------------------------------------------------------------
def _c_rb(nu: np.ndarray, p: VonBenzonParams) -> np.ndarray:
    """Rigid-body Coriolis C_RB(ν)·ν with r_g = 0.

    Source: oceanscale/validation/vonbenzon_reference.py:136-144.
    In production, Newton/MJWarp provides this implicitly.
    """
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


def run_gpu_tier1() -> dict[str, np.ndarray]:
    """Tier-1 Warp hydro kernels + forward Euler integration.

    Pipeline per step:
      1. tier1.compute_wrench  →  added-mass + damping + C_A + restoring on GPU
      2. Read wrench to CPU, add 5 N surge directly (bypass thruster allocation)
      3. nu_dot = M_RB_inv @ (wrench + surge - C_RB*nu)
      4. Forward Euler state update

    Key differences from CPU reference:
      - EMA filter (alpha=0.3) for nu_dot estimation  (tier1_kernels.py:238-243)
      - 4 cross-coupling damping terms                  (tier1_kernels.py:66-70)
      - Quaternion-based restoring (no Euler angles)    (tier1_kernels.py:126-137)
      - M_A applied as force, not combined mass matrix  (tier1_kernels.py:27-33)
    """
    wp.init()
    p = VonBenzonParams()

    # Rigid-body inverse mass (no added mass — that's in the wrench as a force)
    M_RB_inv = np.diag(
        1.0 / np.array([p.mass, p.mass, p.mass, p.I_x, p.I_y, p.I_z])
    )

    tier1 = Tier1(
        n_envs=1,
        n_thrusters=8,
        device="cuda",
        rho_water=p.rho,  # 1000.0 — match von Benzon, not Tier-1 default 1025
        g_accel=p.g,  # 9.82   — match von Benzon, not Tier-1 default 9.81
        ema_alpha=0.3,  # production default (tier1.py:34)
    )
    tier1.set_coeffs(
        added_mass=p.added_mass,
        d_lin=p.d_lin,
        d_quad=p.d_quad,
        mass=p.mass,
        volume=p.volume,
        coBM=abs(p.r_b[2]),  # 0.01 m — vonbenzon_reference.py:34
        T_matrix=_build_allocation_matrix().tolist(),
    )

    pos = np.zeros(3, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    nu = np.zeros(6, dtype=np.float64)

    t_arr = np.zeros(N_STEPS)
    pos_arr = np.zeros((N_STEPS, 3))
    quat_arr = np.zeros((N_STEPS, 4))
    vel_arr = np.zeros((N_STEPS, 6))

    t0 = time.perf_counter()

    for k in range(N_STEPS):
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
        # wrench layout: (fx, fy, fz, mx, my, mz) — spatial_vectorf
        wrench[0] += SURGE_FORCE  # add surge directly (bypass thruster alloc)

        c_rb = _c_rb(nu, p)
        nu_dot = M_RB_inv @ (wrench - c_rb)

        d_pos = _quat_to_rotmat(quat) @ nu[:3]
        d_q = 0.5 * _quat_mul(quat, np.array([nu[3], nu[4], nu[5], 0.0]))

        pos = pos + DT * d_pos
        quat = quat + DT * d_q
        quat = quat / np.linalg.norm(quat)
        nu = nu + DT * nu_dot

    elapsed = time.perf_counter() - t0
    print(
        f"  GPU Tier-1: {N_STEPS} steps in {elapsed:.2f}s "
        f"({N_STEPS / elapsed:.0f} steps/s)"
    )

    return {"t": t_arr, "pos": pos_arr, "quat": quat_arr, "vel": vel_arr}


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion angular distance in degrees. Handles (N,4) batch input."""
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_metrics(ref: dict, test: dict, label: str) -> dict:
    """Position (L2), attitude (quaternion angle), velocity (L2) errors."""
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
# Report
# ---------------------------------------------------------------------------
def print_report(
    cpu_rk4: dict,
    cpu_euler: dict,
    gpu_tier1: dict,
    m_euler: dict,
    m_gpu: dict,
) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  FIDELITY BENCHMARK: GPU OceanScale vs CPU von Benzon Reference")
    print("=" * W)
    print(
        f"  Scenario : BlueROV2 Heavy, {SURGE_FORCE}N forward surge, "
        f"{N_STEPS} steps @ dt={DT}s ({N_STEPS * DT:.0f}s)"
    )
    print("  CPU RK4  : von Benzon 2022, RK4 integration (gold standard)")
    print("  CPU Euler: Same equations, forward Euler (integration baseline)")
    print("  GPU Tier1: Warp kernels + forward Euler + EMA(alpha=0.3)")

    # ── Final states ──
    print("\n" + "-" * W)
    print("  FINAL STATES")
    print("-" * W)
    for label, traj in [
        ("CPU RK4 ", cpu_rk4),
        ("CPU Euler", cpu_euler),
        ("GPU Tier1", gpu_tier1),
    ]:
        fp = traj["pos"][-1]
        fv = traj["vel"][-1]
        print(
            f"  {label}  pos=[{fp[0]:+.5f}, {fp[1]:+.5f}, {fp[2]:+.5f}] m"
        )
        print(
            f"  {'':9s} vel=[{fv[0]:+.5f}, {fv[1]:+.5f}, {fv[2]:+.5f}] m/s"
        )

    # ── Errors ──
    print("\n" + "-" * W)
    print("  ERROR vs CPU RK4 REFERENCE")
    print("-" * W)
    for m in [m_euler, m_gpu]:
        pe = m["position"]
        ae = m["attitude"]
        ve = m["velocity"]
        print(f"  {m['label']}:")
        print(
            f"    Position  L2 final = {pe['l2_final_m']:.6f} m  "
            f"({pe['relative_final_pct']:.2f}% of ref {pe['ref_displacement_m']:.4f} m)"
        )
        print(
            f"    Position  L2 RMS   = {pe['l2_rms_m']:.6f} m  "
            f"(max {pe['l2_max_m']:.6f} m)"
        )
        print(
            f"    Attitude  RMS      = {ae['rms_deg']:.4f} deg  "
            f"(max {ae['max_deg']:.4f} deg, final {ae['final_deg']:.4f} deg)"
        )
        print(
            f"    Velocity  L2 final = {ve['l2_final_ms']:.6f} m/s  "
            f"({ve['relative_final_pct']:.2f}%)"
        )

    # ── Pass/fail ──
    print("\n" + "-" * W)
    print("  PASS/FAIL  (GPU Tier-1 vs CPU RK4)")
    print("-" * W)
    pp = m_gpu["position"]["relative_final_pct"] < 5.0
    ap = m_gpu["attitude"]["rms_deg"] < 3.0
    print(
        f"  Position L2 < 5%     : {'PASS' if pp else 'FAIL'}  "
        f"({m_gpu['position']['relative_final_pct']:.2f}%)"
    )
    print(
        f"  Attitude RMS < 3 deg : {'PASS' if ap else 'FAIL'}  "
        f"({m_gpu['attitude']['rms_deg']:.4f} deg)"
    )
    overall = pp and ap
    print(f"  OVERALL              : {'PASS' if overall else 'FAIL'}")

    # ── Error decomposition ──
    print("\n" + "-" * W)
    print("  ERROR DECOMPOSITION")
    print("-" * W)
    int_pct = m_euler["position"]["relative_final_pct"]
    gpu_pct = m_gpu["position"]["relative_final_pct"]
    impl_pct = gpu_pct - int_pct
    print(f"  Integration (Euler vs RK4) : {int_pct:+.2f}%")
    print(f"  Implementation (Tier-1)    : {impl_pct:+.2f}%")
    print(f"  Total (GPU vs RK4)         : {gpu_pct:.2f}%")
    print()
    print("  Known Tier-1 differences from von Benzon reference:")
    print("    [1] EMA filter for nu_dot (alpha=0.3) vs analytical")
    print("          — tier1_kernels.py:238-243")
    print("    [2] 4 cross-coupling damping terms (MarineGym)")
    print("          — tier1_kernels.py:66-70")
    print("    [3] Quaternion restoring vs Euler-angle restoring")
    print("          — tier1_kernels.py:126-137")
    print("    [4] M_A applied as force (-M_A*nu_dot_ema) vs combined mass matrix")
    print("          — tier1_kernels.py:27-33")
    print("=" * W)


def save_json(
    cpu_rk4: dict,
    gpu_tier1: dict,
    m_euler: dict,
    m_gpu: dict,
) -> None:
    ds = 10
    data = {
        "scenario": (
            f"BlueROV2 Heavy, {SURGE_FORCE}N surge, "
            f"{N_STEPS} steps @ dt={DT}s"
        ),
        "pass_criteria": {"position_pct": 5.0, "attitude_deg": 3.0},
        "results": {
            "cpu_euler_vs_rk4": m_euler,
            "gpu_tier1_vs_rk4": m_gpu,
        },
        "time_series": {
            "t": cpu_rk4["t"][::ds].tolist(),
            "cpu_rk4_pos": cpu_rk4["pos"][::ds].tolist(),
            "gpu_tier1_pos": gpu_tier1["pos"][::ds].tolist(),
            "cpu_rk4_vel_linear": cpu_rk4["vel"][::ds, :3].tolist(),
            "gpu_tier1_vel_linear": gpu_tier1["vel"][::ds, :3].tolist(),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("Fidelity Benchmark: GPU OceanScale vs CPU von Benzon")
    print(f"  {N_STEPS} steps, dt={DT}s, {SURGE_FORCE}N surge\n")

    print("[1/4] CPU von Benzon RK4 (gold standard)...")
    cpu_rk4 = run_cpu_rk4()
    print(f"  Final pos: [{cpu_rk4['pos'][-1, 0]:.5f}, "
          f"{cpu_rk4['pos'][-1, 1]:.5f}, {cpu_rk4['pos'][-1, 2]:.5f}]")

    print("\n[2/4] CPU Euler (integration error baseline)...")
    cpu_euler = run_cpu_euler()
    print(f"  Final pos: [{cpu_euler['pos'][-1, 0]:.5f}, "
          f"{cpu_euler['pos'][-1, 1]:.5f}, {cpu_euler['pos'][-1, 2]:.5f}]")

    print("\n[3/4] GPU Tier-1 standalone (Warp kernels + Euler)...")
    gpu_tier1 = run_gpu_tier1()
    print(f"  Final pos: [{gpu_tier1['pos'][-1, 0]:.5f}, "
          f"{gpu_tier1['pos'][-1, 1]:.5f}, {gpu_tier1['pos'][-1, 2]:.5f}]")

    print("\n[4/4] Computing error metrics...")
    m_euler = compute_metrics(cpu_rk4, cpu_euler, "CPU Euler")
    m_gpu = compute_metrics(cpu_rk4, gpu_tier1, "GPU Tier-1")

    print_report(cpu_rk4, cpu_euler, gpu_tier1, m_euler, m_gpu)
    save_json(cpu_rk4, gpu_tier1, m_euler, m_gpu)

    pp = m_gpu["position"]["relative_final_pct"] < 5.0
    ap = m_gpu["attitude"]["rms_deg"] < 3.0
    return 0 if (pp and ap) else 1


if __name__ == "__main__":
    sys.exit(main())
