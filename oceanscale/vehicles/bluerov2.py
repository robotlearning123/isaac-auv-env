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
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _rz(angle: float) -> np.ndarray:
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
    coBM: float = 0.01  # noqa: N815

    # --- Thrusters ---
    n_thrusters: int = 8  # 4 horizontal vectored + 4 vertical
    max_thrust: float = 51.5  # N per T200 at full throttle
    min_thrust: float = -40.2  # N per T200 max reverse

    # --- Hydrodynamic coefficients (Fossen 6-tuple ordering) ---
    # Added mass (kg, kg·m²) — von Benzon Table A1
    added_mass: tuple[float, float, float, float, float, float] = (
        6.36,  # X_udot  — surge
        7.12,  # Y_vdot  — sway
        18.68,  # Z_wdot  — heave
        0.189,  # K_pdot  — roll
        0.135,  # M_qdot  — pitch
        0.222,  # N_rdot  — yaw
    )

    # Linear damping (N·s/m, N·m·s/rad) — von Benzon Table A1
    # Positive values: kernel applies -(d_lin + d_quad*|v|)*v
    d_lin: tuple[float, float, float, float, float, float] = (
        13.7,  # Xu  — surge
        0.0,  # Yv  — sway (identified as zero)
        33.0,  # Zw  — heave
        0.0,  # Kp  — roll (identified as zero)
        0.8,  # Mq  — pitch
        0.0,  # Nr  — yaw (identified as zero)
    )

    # Quadratic damping (N·s²/m², N·m·s²/rad²) — von Benzon Table A1
    d_quad: tuple[float, float, float, float, float, float] = (
        141.0,  # Xu|u|  — surge
        217.0,  # Yv|v|  — sway
        190.0,  # Zw|w|  — heave
        1.19,  # Kp|p|  — roll
        0.47,  # Mq|q|  — pitch
        1.5,  # Nr|r|  — yaw
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
            pos = _rz(alphas[i]) @ base_pos_h
            dirn = _rz(betas[i]) @ base_dir_h
            T[:3, i] = dirn
            T[3:, i] = np.cross(pos, dirn)

        # Vertical thrusters T5-T8
        for j in range(4):
            pos = _rz(gammas[j]) @ base_pos_v
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


def _marinegym_asset_dir() -> Path:
    return Path(str(files("oceanscale.assets").joinpath("bluerov_marinegym")))


def _load_marinegym_yaml() -> dict[str, Any]:
    yaml_path = _marinegym_asset_dir() / "BlueROV.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class BlueROV2MarineGym:
    """BlueROV config from MarineGym — 6 rotors, different hydro source.

    Adapted from MarineGym (https://github.com/Marine-RL/MarineGym)
    Original: marinegym/robots/assets/usd/BlueROV/BlueROV.yaml | License: MIT
    Paper: Chu et al., "MarineGym", IROS 2025
    Modifications: Loaded as OceanScale dataclass; set_coeffs_kwargs() maps
    to Tier1 Fossen format.
    """

    mass: float = 11.5
    volume: float = 0.0113459
    length: float = 0.46
    width: float = 0.58
    height: float = 0.25

    Ix: float = 0.16
    Iy: float = 0.16
    Iz: float = 0.16

    coBM: float = 0.01  # noqa: N815
    drag_coef: float = 0.3

    n_thrusters: int = 6
    max_thrust: float = 51.5
    min_thrust: float = -40.2

    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
    d_lin: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
    d_quad: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    rotor_directions: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    rotor_time_constants: tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    rotor_force_constants: tuple[float, ...] = (4.4e-7, 4.4e-7, 4.4e-7, 4.4e-7, 4.4e-7, 4.4e-7)
    rotor_moment_constants: tuple[float, ...] = (
        1.368e-9, 1.368e-9, 1.368e-9, 1.368e-9, 1.368e-9, 1.368e-9,
    )
    rotor_max_rpm: tuple[float, ...] = (3900, 3900, 3900, 3900, 3900, 3900)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path | None = None) -> BlueROV2MarineGym:
        """Load from MarineGym YAML config."""
        if yaml_path is None:
            data = _load_marinegym_yaml()
        else:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        hc = data["hydro_coef"]
        rc = data["rotor_configuration"]
        return cls(
            volume=data.get("volume", 0.0113459),
            coBM=data.get("coBM", 0.01),
            drag_coef=data.get("drag_coef", 0.3),
            n_thrusters=rc["num_rotors"],
            added_mass=tuple(hc["added_mass"]),
            d_lin=tuple(hc["linear_damping"]),
            d_quad=tuple(hc["quadratic_damping"]),
            rotor_directions=tuple(rc["directions"]),
            rotor_time_constants=tuple(rc["time_constants"]),
            rotor_force_constants=tuple(rc["force_constants"]),
            rotor_moment_constants=tuple(rc["moment_constants"]),
            rotor_max_rpm=tuple(rc["max_rotation_velocities"]),
        )

    def usd_path(self) -> Path:
        return _marinegym_asset_dir() / "BlueROV.usd"

    def set_coeffs_kwargs(self) -> dict[str, Any]:
        """Return kwargs dict for Tier1.set_coeffs()."""
        return {
            "added_mass": self.added_mass,
            "d_lin": self.d_lin,
            "d_quad": self.d_quad,
            "mass": self.mass,
            "volume": self.volume,
            "coBM": self.coBM,
        }

    def rotor_config(self) -> dict[str, Any]:
        """Return rotor configuration for T200 thruster model."""
        return {
            "num_rotors": self.n_thrusters,
            "directions": list(self.rotor_directions),
            "time_constants": list(self.rotor_time_constants),
            "force_constants": list(self.rotor_force_constants),
            "moment_constants": list(self.rotor_moment_constants),
            "max_rpm": list(self.rotor_max_rpm),
        }


def _warpauv_asset_dir() -> Path:
    return Path(str(files("oceanscale.assets").joinpath("warpauv")))


@dataclass(frozen=True)
class WarpAUV:
    """WarpAUV vehicle config from isaac-auv-env.

    Adapted from isaac-auv-env (https://github.com/warplab/isaac-auv-env)
    Original: warpauv_env.py + assets/warpauv.py by Kevin Chang and Levi Cai
    License: BSD-3-Clause
    Paper: Lofaro et al., "Learning to Swim", arXiv 2410.00120
    Modifications: Extracted vehicle parameters into OceanScale dataclass.
    """

    mass: float = 22.701
    volume: float = 0.02275
    length: float = 0.70
    width: float = 0.40
    height: float = 0.20

    Ix: float = 0.37
    Iy: float = 0.97
    Iz: float = 1.19

    coBM: float = 0.01
    n_thrusters: int = 6

    com_to_cob_offset: tuple[float, float, float] = (0.0, 0.0, 0.01)
    water_rho: float = 997.0
    water_beta: float = 0.001306
    rotor_constant: float = 0.001
    dyn_time_constant: float = 0.05

    thruster_positions: tuple[tuple[float, float, float], ...] = (
        (-0.4127, 0.1506, -0.0889),
        (-0.4127, -0.1506, -0.0889),
        (-0.303, 0.1461, -0.1587),
        (-0.303, -0.1461, -0.1587),
        (0.0585, 0.1461, -0.0540),
        (0.0585, -0.1461, -0.0540),
    )

    def usd_path(self) -> Path:
        return _warpauv_asset_dir() / "warpauv.usd"

    def inertia_diag(self) -> np.ndarray:
        return np.array([self.Ix, self.Iy, self.Iz], dtype=np.float32)

    def mujoco_drag_kwargs(self) -> dict[str, Any]:
        """Return kwargs for MuJoCoDrag constructor."""
        return {
            "inertia_diag": self.inertia_diag(),
            "mass": self.mass,
        }


def _bluerov2_heavy_gz_dir() -> Path:
    return Path(str(files("oceanscale.assets").joinpath("bluerov2_heavy_gz")))


def bluerov2_heavy_gz_mesh_paths() -> dict[str, Path]:
    # Adapted from clydemcqueen/bluerov2_gz (https://github.com/clydemcqueen/bluerov2_gz)
    # License: MIT (in package.xml) | Commit: 661264b
    # Content: BlueROV2 Heavy high-fidelity Collada visual meshes + T200 propeller meshes
    d = _bluerov2_heavy_gz_dir() / "meshes"
    return {
        "hull": d / "bluerov2_heavy.dae",
        "prop_ccw": d / "t200_ccw_prop.dae",
        "prop_cw": d / "t200_cw_prop.dae",
    }


def bluerov2_heavy_gz_sdf_path() -> Path:
    # Adapted from clydemcqueen/bluerov2_gz (https://github.com/clydemcqueen/bluerov2_gz)
    # License: MIT | Commit: 661264b
    return _bluerov2_heavy_gz_dir() / "model.sdf"


def sand_heightmap_mesh_paths() -> dict[str, Path]:
    # Adapted from clydemcqueen/bluerov2_gz (https://github.com/clydemcqueen/bluerov2_gz)
    # License: MIT | Commit: 661264b
    d = Path(str(files("oceanscale.assets").joinpath("sand_heightmap")))
    return {
        "heightmap": d / "heightmap.dae",
        "seabed": d / "sandseabed.dae",
        "texture": d / "soil_sand_0045_01.jpg",
    }
