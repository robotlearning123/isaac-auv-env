# Fossen baseline controllers ported from PythonVehicleSimulator
# (MIT, Copyright (c) 2018 Thor I. Fossen, cybergalactic)
# See THIRD_PARTY_NOTICES.md. Study reference, not vendored.
"""Fossen baseline controllers: PID pole-placement, DP control, Integral SMC.

Ported from PVS lib/control.py and lib/guidance.py.
Reference: Fossen (2021) Handbook of Marine Craft Hydrodynamics and Motion Control.
"""

from __future__ import annotations

import numpy as np


def ssa(angle: float) -> float:
    """Smallest signed angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ref_model_3(
    x_d: float,
    v_d: float,
    a_d: float,
    r: float,
    wn_d: float,
    zeta_d: float,
    v_max: float,
    dt: float,
) -> tuple[float, float, float]:
    """3rd-order reference model for smooth trajectory generation.

    Fossen 2021 §12.2. Generates smooth position, velocity, acceleration
    from a step command r.
    """
    j_d = wn_d**3 * (r - x_d) - (2 * zeta_d + 1) * wn_d**2 * v_d - (2 * zeta_d + 1) * wn_d * a_d
    x_d += dt * v_d
    v_d += dt * a_d
    a_d += dt * j_d
    if v_d > v_max:
        v_d = v_max
    elif v_d < -v_max:
        v_d = -v_max
    return x_d, v_d, a_d


def pid_pole_placement(
    e_int: float,
    e_x: float,
    e_v: float,
    x_d: float,
    v_d: float,
    a_d: float,
    m: float,
    d: float,
    k: float,
    wn_d: float,
    zeta_d: float,
    wn: float,
    zeta: float,
    r: float,
    v_max: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """SISO PID via pole placement. Fossen 2021 §13.3.

    Kp = m*wn^2 - k, Kd = 2*zeta*wn*m - d, Ki = (wn/10)*Kp
    Returns: (u, e_int, x_d, v_d, a_d)
    """
    kp = m * wn**2 - k
    kd = m * 2.0 * zeta * wn - d
    ki = (wn / 10.0) * kp
    u = -kp * e_x - kd * e_v - ki * e_int
    e_int += dt * e_x
    x_d, v_d, a_d = ref_model_3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, dt)
    return u, e_int, x_d, v_d, a_d


def dp_pole_placement(
    e_int: np.ndarray,
    M3: np.ndarray,
    D3: np.ndarray,
    eta3: np.ndarray,
    nu3: np.ndarray,
    x_d: float,
    y_d: float,
    psi_d: float,
    wn: np.ndarray,
    zeta: np.ndarray,
    eta_ref: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """MIMO nonlinear PID for DP. Fossen 2021 §13.3.

    Returns: (tau, e_int, x_d, y_d, psi_d)
    """
    M3d = np.diag(np.diag(M3))
    D3d = np.diag(np.diag(D3))
    kp = wn @ wn @ M3d
    kd = 2.0 * zeta @ wn @ M3d - D3d
    ki = (1.0 / 10.0) * wn @ kp

    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])
    R = _rzyx(0.0, 0.0, eta3[2])
    tau = -R.T @ kp @ e - kd @ nu3 - R.T @ ki @ e_int

    T_lp = 5.0 * np.array([1 / wn[0, 0], 1 / wn[1, 1], 1 / wn[2, 2]])
    x_d += dt * (eta_ref[0] - x_d) / T_lp[0]
    y_d += dt * (eta_ref[1] - y_d) / T_lp[1]
    psi_d += dt * (eta_ref[2] - psi_d) / T_lp[2]
    e_int += dt * e
    return tau, e_int, x_d, y_d, psi_d


def integral_smc(
    e_int: float,
    e_x: float,
    e_v: float,
    x_d: float,
    v_d: float,
    a_d: float,
    T_nomoto: float,
    K_nomoto: float,
    wn_d: float,
    zeta_d: float,
    K_d: float,
    K_sigma: float,
    lam: float,
    phi_b: float,
    r: float,
    v_max: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Integral SMC heading autopilot. Fossen 2021 §16.5 Eq. 16.479.

    Boundary layer phi_b for chattering reduction.
    Returns: (delta, e_int, x_d, v_d, a_d)
    """
    vr_dot = a_d - 2 * lam * e_v - lam**2 * ssa(e_x)
    vr = v_d - 2 * lam * ssa(e_x) - lam**2 * e_int
    sigma = e_v + 2 * lam * ssa(e_x) + lam**2 * e_int

    if abs(sigma / phi_b) > 1.0:
        delta = (T_nomoto * vr_dot + vr - K_d * sigma - K_sigma * np.sign(sigma)) / K_nomoto
    else:
        delta = (T_nomoto * vr_dot + vr - K_d * sigma - K_sigma * (sigma / phi_b)) / K_nomoto

    e_int += dt * ssa(e_x)
    x_d, v_d, a_d = ref_model_3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, dt)
    return delta, e_int, x_d, v_d, a_d


def _rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """SO(3) rotation matrix ZYX Euler convention."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi],
    ])
