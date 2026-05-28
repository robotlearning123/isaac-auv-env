"""Amphibious robot dog — walks on land, swims underwater.

Demonstrates OceanScale's multi-domain simulation: ground contact, partial
submersion at the water surface, and full Fossen hydrodynamics underwater.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AmphibiousRobotDog:
    """Quadruped robot dog with integrated thrusters for amphibious operation.

    Physical specs (realistic scale for a mid-size inspection robot):
    - Body: 0.8 x 0.4 x 0.3 m, 25 kg
    - 4 legs with ground-contact actuators (walking mode)
    - 8 T200-class thrusters (swimming mode)
    """

    name: str = "AmphibiousRobotDog"

    # --- Body geometry ---
    mass: float = 25.0
    volume: float = 0.025
    length: float = 0.80
    width: float = 0.40
    height: float = 0.30

    # Inertia tensor diagonal (kg*m^2)
    Ix: float = 0.42
    Iy: float = 1.10
    Iz: float = 1.25

    # Center of buoyancy offset from CoG (m), z-up
    coBM: float = 0.03

    # --- Thrusters (swimming mode) ---
    n_thrusters: int = 8
    max_thrust: float = 80.0  # N per thruster
    min_thrust: float = -60.0  # N reverse

    # --- Fossen hydrodynamic coefficients ---
    added_mass: tuple[float, ...] = (8.0, 10.0, 22.0, 0.25, 0.18, 0.30)
    d_lin: tuple[float, ...] = (15.0, 0.0, 35.0, 0.0, 1.0, 0.0)
    d_quad: tuple[float, ...] = (160.0, 240.0, 210.0, 1.5, 0.6, 1.8)

    # --- Leg actuators (walking mode) ---
    leg_max_force: float = 200.0  # N per leg
    leg_height: float = 0.15  # m, leg reach below body
    ground_friction: float = 0.7  # Coulomb friction coefficient
    ground_stiffness: float = 5000.0  # N/m, ground contact spring
    ground_damping: float = 500.0  # N·s/m, ground contact damper

    # --- Water surface ---
    water_surface_z: float = 0.0  # z-coordinate of water surface

    # --- CPG gait parameters ---
    n_cpg_oscillators: int = 6
    cpg_frequency_walk: float = 2.0  # Hz
    cpg_frequency_swim: float = 1.2  # Hz
    cpg_amplitude_walk: float = 1.0
    cpg_amplitude_swim: float = 0.8
    cpg_coupling_strength: float = 1.5
    cpg_phase_offset_walk: float = math.pi  # trot gait diagonal offset
    cpg_phase_offset_swim: float = math.pi / 3  # travelling wave delay

    # --- Contact-force-driven mode detection ---
    contact_force_threshold: float = 20.0  # N
    body_half_height: float = 0.15  # m

    def tier1_kwargs(self) -> dict:
        return {"n_thrusters": self.n_thrusters, "max_thrust": self.max_thrust}

    def set_coeffs_kwargs(self) -> dict:
        return {
            "added_mass": self.added_mass,
            "d_lin": self.d_lin,
            "d_quad": self.d_quad,
            "mass": self.mass,
            "volume": self.volume,
            "coBM": self.coBM,
            "T_matrix": self.thruster_allocation_matrix(),
        }

    def thruster_allocation_matrix(self) -> np.ndarray:
        """6×8 thruster allocation matrix.

        Layout:
        - T1-T4: horizontal vectored (45° tilt, 90° spacing)
        - T5-T8: vertical (straight up)
        """
        T = np.zeros((6, 8))

        # Horizontal thrusters — vectored at 45° tilt, 90° spacing
        base_pos_h = np.array([0.20, 0.15, 0.0])
        base_dir_h = np.array([0.707, 0.707, 0.0])  # 45° tilt: Fx=Fy
        alphas = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

        for i in range(4):
            c_a, s_a = np.cos(alphas[i]), np.sin(alphas[i])
            Rz = np.array([[c_a, -s_a, 0], [s_a, c_a, 0], [0, 0, 1]])
            pos = Rz @ base_pos_h
            dirn = Rz @ base_dir_h  # same rotation as position
            T[:3, i] = dirn
            T[3:, i] = np.cross(pos, dirn)

        # Vertical thrusters
        base_pos_v = np.array([0.15, 0.10, 0.0])
        base_dir_v = np.array([0.0, 0.0, 1.0])
        gammas = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

        for j in range(4):
            c_g, s_g = np.cos(gammas[j]), np.sin(gammas[j])
            Rz = np.array([[c_g, -s_g, 0], [s_g, c_g, 0], [0, 0, 1]])
            pos = Rz @ base_pos_v
            T[:3, 4 + j] = base_dir_v
            T[3:, 4 + j] = np.cross(pos, base_dir_v)

        return T

    def terrain_height(self, x: float) -> float:
        """Terrain profile: flat beach → underwater slope → flat seabed."""
        if x <= 2.0:
            return 0.5  # above water
        elif x <= 6.0:
            return 0.5 - 0.25 * (x - 2.0)  # slope down
        else:
            return -0.5  # seabed

    def cpg_config(self) -> CPGConfig:
        """Return CPGConfig matching this vehicle's parameters."""
        from oceanscale.controllers.cpg import CPGConfig

        return CPGConfig(
            n_oscillators=self.n_cpg_oscillators,
            frequency_walk=self.cpg_frequency_walk,
            frequency_swim=self.cpg_frequency_swim,
            amplitude_walk=self.cpg_amplitude_walk,
            amplitude_swim=self.cpg_amplitude_swim,
            coupling_strength=self.cpg_coupling_strength,
            phase_offset_walk=self.cpg_phase_offset_walk,
            phase_offset_swim=self.cpg_phase_offset_swim,
        )
