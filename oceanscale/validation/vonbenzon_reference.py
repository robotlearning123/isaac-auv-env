"""Von Benzon 2022 reference model — CPU 6-DOF Fossen dynamics for BlueROV2 Heavy.

Ports the equations described in von Benzon et al. 2022, JMSE 10(12):1898
(DOI 10.3390/jmse10121898). Parameters from /tmp/vonbenzon_research.md
(paper Table A1). NOT validated against Simulink output — provides a
frozen-snapshot baseline for regression testing of the GPU implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class VonBenzonParams:
    """BlueROV2 Heavy parameters — von Benzon 2022 Table A1.

    Source refs point to /tmp/vonbenzon_research.md sections.
    """

    # Vehicle constants — research.md L13-26 (Table A1)
    g: float = 9.82  # m/s²
    rho: float = 1000.0  # kg/m³
    mass: float = 13.5  # kg
    volume: float = 0.0134  # m³
    I_x: float = 0.26  # kg·m²
    I_y: float = 0.23  # kg·m²
    I_z: float = 0.37  # kg·m²
    r_g: tuple[float, float, float] = (0.0, 0.0, 0.0)  # CoG at origin
    r_b: tuple[float, float, float] = (0.0, 0.0, -0.01)  # CoB — research.md L23
    # Added mass diagonal — research.md L96-114 (Table A1 §3)
    added_mass: tuple[float, ...] = (6.36, 7.12, 18.68, 0.189, 0.135, 0.222)
    # Linear damping diagonal — research.md L118-134 (Table A1 §4)
    d_lin: tuple[float, ...] = (13.7, 0.0, 33.0, 0.0, 0.8, 0.0)
    # Quadratic damping diagonal — research.md L138-154 (Table A1 §5)
    d_quad: tuple[float, ...] = (141.0, 217.0, 190.0, 1.19, 0.47, 1.5)


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ])


def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    qx, qy, qz, qw = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _build_allocation_matrix() -> np.ndarray:
    """Thruster allocation T ∈ R^{6×8} — research.md L84-91 (Eq 15, Table A1 §2)."""
    p_h = np.array([0.156, 0.111, 0.085])
    e_h = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0])
    alpha = [0, 5.05, 1.91, np.pi]
    beta = [0, np.pi / 2, 3 * np.pi / 4, np.pi]

    p_v = np.array([0.120, 0.218, 0.0])
    e_v = np.array([0.0, 0.0, -1.0])
    gamma = [0, 4.15, 1.01, np.pi]

    T = np.zeros((6, 8))
    for i in range(4):
        r = _Rz(alpha[i]) @ p_h
        e = _Rz(beta[i]) @ e_h
        T[:3, i] = e
        T[3:, i] = np.cross(r, e)
    for i in range(4):
        r = _Rz(gamma[i]) @ p_v
        T[:3, 4 + i] = e_v
        T[3:, 4 + i] = np.cross(r, e_v)
    return T


class VonBenzonReferenceModel:
    """CPU 6-DOF Fossen model porting von Benzon 2022 BlueROV2 equations.

    Equation of motion (Fossen standard form):
        M·ν̇ + C(ν)·ν + D(ν)·ν + g(η) = τ

    Solved as:
        ν̇ = M⁻¹ · (τ − C·ν − D·ν − g(η))

    State vector: [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r]  (13 elements)
    """

    def __init__(self, params: VonBenzonParams | None = None) -> None:
        self.p = params or VonBenzonParams()
        p = self.p
        M_diag = np.array([
            p.mass + p.added_mass[0],
            p.mass + p.added_mass[1],
            p.mass + p.added_mass[2],
            p.I_x + p.added_mass[3],
            p.I_y + p.added_mass[4],
            p.I_z + p.added_mass[5],
        ])
        self._M_inv = np.diag(1.0 / M_diag)
        self._W = p.mass * p.g
        self._B = p.rho * p.g * p.volume
        self._r_b = np.array(p.r_b)
        self._d_lin = np.array(p.d_lin)
        self._d_quad = np.array(p.d_quad)
        self._ma_lin = np.array(p.added_mass[:3])
        self._ma_ang = np.array(p.added_mass[3:])
        self.T_alloc = _build_allocation_matrix()

    def _coriolis(self, nu: np.ndarray) -> np.ndarray:
        """C(ν)·ν = C_RB·ν + C_A·ν."""
        u, v, w, p, q, r = nu
        m = self.p.mass
        # C_RB·ν — rigid-body Coriolis (Newton-Euler, r_g=0)
        c_rb = np.array([
            m * (q * w - r * v),
            m * (r * u - p * w),
            m * (p * v - q * u),
            (self.p.I_z - self.p.I_y) * q * r,
            (self.p.I_x - self.p.I_z) * p * r,
            (self.p.I_y - self.p.I_x) * p * q,
        ])
        # C_A·ν — added-mass Coriolis, matching tier1_coriolis_a kernel formulation
        ab_l = self._ma_lin * nu[:3]
        ab_a = self._ma_ang * nu[3:]
        c_a = np.concatenate([
            np.cross(ab_l, nu[3:]),
            np.cross(ab_l, nu[:3]) + np.cross(ab_a, nu[3:]),
        ])
        return c_rb + c_a

    def _restoring(self, q: np.ndarray) -> np.ndarray:
        """g(η) — restoring force, research.md L179-201 (Eq 12, §7)."""
        phi, theta, _ = _quat_to_euler(q)
        W, B = self._W, self._B
        xb, yb, zb = self._r_b
        return np.array([
            (W - B) * np.sin(theta),
            -(W - B) * np.cos(theta) * np.sin(phi),
            -(W - B) * np.cos(theta) * np.cos(phi),
            yb * B * np.cos(theta) * np.cos(phi) - zb * B * np.cos(theta) * np.sin(phi),
            -zb * B * np.sin(theta) - xb * B * np.cos(theta) * np.cos(phi),
            xb * B * np.cos(theta) * np.sin(phi) + yb * B * np.sin(theta),
        ])

    def _deriv(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        q = state[3:7]
        nu = state[7:13]
        d_pos = _quat_to_rotmat(q) @ nu[:3]
        d_q = 0.5 * _quat_mul(q, np.array([nu[3], nu[4], nu[5], 0.0]))
        damping = (self._d_lin + self._d_quad * np.abs(nu)) * nu
        nu_dot = self._M_inv @ (tau - self._coriolis(nu) - damping - self._restoring(q))
        return np.concatenate([d_pos, d_q, nu_dot])

    def generate_trajectory(
        self,
        thrust_func: Callable[[float], np.ndarray] | None = None,
        dt: float = 0.01,
        n_steps: int = 1000,
        initial_state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run von Benzon 6-DOF model for n_steps with RK4 integration.

        Args:
            thrust_func: callable(t) → 6-DOF body wrench [Fx..N]. None = zero.
            dt: timestep (s). Default 0.01 → 1000 steps = 10 s.
            n_steps: number of integration steps.
            initial_state: 13-element state. None = rest at origin, identity quat.

        Returns:
            dict with keys t(N,), pos(N,3), quat(N,4), vel(N,6), omega(N,3).
        """
        if thrust_func is None:
            thrust_func = lambda t: np.zeros(6)
        if initial_state is None:
            initial_state = np.zeros(13)
            initial_state[6] = 1.0  # qw = 1

        s = initial_state.astype(np.float64).copy()
        t_arr = np.zeros(n_steps)
        pos = np.zeros((n_steps, 3))
        quat = np.zeros((n_steps, 4))
        vel = np.zeros((n_steps, 6))
        omega = np.zeros((n_steps, 3))

        for k in range(n_steps):
            t = k * dt
            tau = np.asarray(thrust_func(t), dtype=np.float64)
            t_arr[k] = t
            pos[k] = s[:3]
            quat[k] = s[3:7]
            vel[k] = s[7:13]
            omega[k] = s[10:13]

            k1 = self._deriv(s, tau)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            s[3:7] /= np.linalg.norm(s[3:7])

        return {"t": t_arr, "pos": pos, "quat": quat, "vel": vel, "omega": omega}

    def load_reference_dataset(self, path: Path) -> dict[str, np.ndarray]:
        data = np.load(path)
        required = {"pose", "velocity", "time"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"reference dataset missing keys: {missing}")
        return {k: data[k] for k in required}
