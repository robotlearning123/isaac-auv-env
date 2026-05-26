"""Three-axis magnetometer with Earth's magnetic field model.

Simplified IGRF-like model for declination/inclination based on latitude
and longitude, plus hard/soft iron distortion from the vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MagnetometerConfig:
    """Magnetometer parameters."""

    earth_field_uT: float = 50.0
    declination_deg: float = -10.0
    inclination_deg: float = 60.0
    noise_std_uT: float = 0.5
    bias_uT: np.ndarray | None = None
    hard_iron_offset: np.ndarray | None = None
    soft_iron_matrix: np.ndarray | None = None
    noise_seed: int | None = None


class Magnetometer:
    """Three-axis magnetometer sensor."""

    def __init__(self, config: MagnetometerConfig | None = None) -> None:
        cfg = config or MagnetometerConfig()
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.noise_seed)

        decl = math.radians(cfg.declination_deg)
        incl = math.radians(cfg.inclination_deg)
        h = cfg.earth_field_uT * math.cos(incl)
        self._earth_field_ned = np.array([
            h * math.cos(decl),
            h * math.sin(decl),
            cfg.earth_field_uT * math.sin(incl),
        ], dtype=np.float32)

        self._hard_iron = (
            np.asarray(cfg.hard_iron_offset, dtype=np.float32)
            if cfg.hard_iron_offset is not None
            else np.zeros(3, dtype=np.float32)
        )
        self._soft_iron = (
            np.asarray(cfg.soft_iron_matrix, dtype=np.float32).reshape(3, 3)
            if cfg.soft_iron_matrix is not None
            else np.eye(3, dtype=np.float32)
        )
        self._bias = (
            np.asarray(cfg.bias_uT, dtype=np.float32)
            if cfg.bias_uT is not None
            else np.zeros(3, dtype=np.float32)
        )

    def measure(self, orientation: np.ndarray | None = None) -> np.ndarray:
        """Return 3-axis magnetic field in body frame (uT).

        Args:
            orientation: (4,) quaternion [x, y, z, w] body→world.
                         None = identity (body = world).
        """
        if orientation is not None:
            R = self._quat_to_rot(orientation)
            body_field = R.T @ self._earth_field_ned
        else:
            body_field = self._earth_field_ned.copy()

        distorted = self._soft_iron @ body_field + self._hard_iron
        noise = self._rng.normal(0, self.cfg.noise_std_uT, 3).astype(np.float32)
        return distorted + self._bias + noise

    def heading_deg(self, orientation: np.ndarray | None = None) -> float:
        """Compute magnetic heading from magnetometer reading."""
        m = self.measure(orientation)
        return float(math.degrees(math.atan2(m[1], m[0]))) % 360.0

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
