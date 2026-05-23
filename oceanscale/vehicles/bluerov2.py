"""BlueROV2 Heavy vehicle parameters for Tier1 Fossen hydrodynamics.

Source: von Benzon et al. (2022), "Hydrodynamic Modeling of a
Micro Underwater Vehicle with Measured Hydrodynamic Coefficients",
JMSE 10(5), 636. DOI: 10.3390/jmse10050636

Parameters extracted from Table A1 (identified values).
Coordinate convention: z-up, x-forward (Newton/Warp).
Origin at center of mass.

NOTE: von Benzon uses rho=1000 (fresh water), g=9.82.
      Tier1 defaults to rho=1025 (salt), g=9.81.
      Hydrodynamic coefficients are independent of water density
      (they are empirical fit constants), so no conversion needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _Rz(angle: float) -> np.ndarray:
    """Rotation matrix about z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass(frozen=True)
class BlueROV2Heavy:
    """BlueROV2 Heavy config — plugs directly into Tier1.set_coeffs()."""

    # --- Body ---
    mass: float = 13.5  # kg (with battery, von Benzon Table A1)
    volume: float = 0.0134  # m³ displacement
    length: float = 0.46  # m
    width: float = 0.58  # m
    height: float = 0.38  # m

    # Inertia tensor diagonal (kg·m²) — von Benzon Table A1
    Ix: float = 0.26
    Iy: float = 0.23
    Iz: float = 0.37

    # Projected areas (m²) — von Benzon Table A1
    Au: float = 0.0877  # frontal (y-z plane)
    Av: float = 0.1131  # side (x-z plane)
    Aw: float = 0.2049  # top (x-y plane)

    # Center of buoyancy offset from CoG in body frame (m).
    # von Benzon (z-down): (0, 0, -0.01) → COB 1cm above COG.
    # Our z-up convention: COB above COG → coBM = +0.01.
    coBM: float = 0.01

    # --- Thrusters ---
    n_thrusters: int = 8  # 4 horizontal vectored + 4 vertical
    max_thrust: float = 51.5  # N per T200 at full throttle
    min_thrust: float = -40.2  # N per T200 max reverse

    # --- Hydrodynamic coefficients (Fossen 6-tuple ordering) ---
    # Added mass (kg, kg·m²) — von Benzon Table A1
    added_mass: tuple[float, float, float, float, float, float] = (
        6.36,   # X_udot  — surge
        7.12,   # Y_vdot  — sway
        18.68,  # Z_wdot  — heave
        0.189,  # K_pdot  — roll
        0.135,  # M_qdot  — pitch
        0.222,  # N_rdot  — yaw
    )

    # Linear damping (N·s/m, N·m·s/rad) — von Benzon Table A1
    # Positive values: kernel applies -(d_lin + d_quad*|v|)*v
    d_lin: tuple[float, float, float, float, float, float] = (
        13.7,  # Xu  — surge
        0.0,   # Yv  — sway (identified as zero)
        33.0,  # Zw  — heave
        0.0,   # Kp  — roll (identified as zero)
        0.8,   # Mq  — pitch
        0.0,   # Nr  — yaw (identified as zero)
    )

    # Quadratic damping (N·s²/m², N·m·s²/rad²) — von Benzon Table A1
    d_quad: tuple[float, float, float, float, float, float] = (
        141.0,  # Xu|u|  — surge
        217.0,  # Yv|v|  — sway
        190.0,  # Zw|w|  — heave
        1.19,   # Kp|p|  — roll
        0.47,   # Mq|q|  — pitch
        1.5,    # Nr|r|  — yaw
    )

    def compute_t_matrix(self) -> list[list[float]]:
        """Compute 6xn_thrusters allocation matrix from von Benzon geometry.

        Convention: z-up, x-forward, y-left (right-handed).
        Rows: [Fx, Fy, Fz, Mx, My, Mz] in body frame.
        Columns: one per thruster.

        Thruster positions from von Benzon Table A1 (converted to z-up):
          Horizontal T1-T4: Rz(alpha_i) * [0.156, 0.111, 0.085]^T
            alpha = {0, 5.05, 1.91, pi} rad
          Vertical T5-T8: Rz(gamma_i) * [0.120, 0.218, 0.0]^T
            gamma = {0, 4.15, 1.01, pi} rad

        Thruster orientations (converted to z-up):
          Horizontal: Rz(beta_i) * [1/sqrt2, -1/sqrt2, 0]^T
            beta = {0, pi/2, 3pi/4, pi} rad
          Vertical: [0, 0, +1]^T (upward in z-up)
        """
        s2 = 1.0 / np.sqrt(2.0)
        # Horizontal thruster angles (von Benzon Table A1)
        alphas = [0.0, 5.05, 1.91, np.pi]
        betas = [0.0, np.pi / 2, 3 * np.pi / 4, np.pi]
        # Vertical thruster angles (von Benzon Table A1)
        gammas = [0.0, 4.15, 1.01, np.pi]

        base_pos_h = np.array([0.156, 0.111, 0.085])
        base_dir_h = np.array([s2, -s2, 0.0])
        base_pos_v = np.array([0.120, 0.218, 0.0])
        base_dir_v = np.array([0.0, 0.0, 1.0])  # z-up: vertical = +z

        n = 4 + 4
        T = np.zeros((6, n), dtype=np.float64)

        # Horizontal thrusters T1-T4
        for i in range(4):
            pos = _Rz(alphas[i]) @ base_pos_h
            dirn = _Rz(betas[i]) @ base_dir_h
            T[:3, i] = dirn
            T[3:, i] = np.cross(pos, dirn)

        # Vertical thrusters T5-T8
        for j in range(4):
            pos = _Rz(gammas[j]) @ base_pos_v
            T[:3, 4 + j] = base_dir_v
            T[3:, 4 + j] = np.cross(pos, base_dir_v)

        return T.tolist()

    def tier1_kwargs(self) -> dict[str, Any]:
        """Return kwargs dict for Tier1 constructor."""
        return {
            "n_thrusters": self.n_thrusters,
            "max_thrust": self.max_thrust,
        }

    def set_coeffs_kwargs(self) -> dict[str, Any]:
        """Return kwargs dict for Tier1.set_coeffs()."""
        return {
            "added_mass": self.added_mass,
            "d_lin": self.d_lin,
            "d_quad": self.d_quad,
            "mass": self.mass,
            "volume": self.volume,
            "coBM": self.coBM,
            "T_matrix": self.compute_t_matrix(),
        }
