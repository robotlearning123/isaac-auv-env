"""USBL (Ultra-Short Baseline) acoustic positioning sensor.

Computes range and bearing from the vehicle to one or more acoustic
transponders, using the sound-speed profile from ``oceanscale.acoustics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class USBLConfig:
    """USBL configuration (EvoLogics S2CR-based defaults)."""

    frequency_khz: float = 26.0
    max_range: float = 3000.0
    range_accuracy_pct: float = 0.1
    bearing_accuracy_deg: float = 0.5
    update_rate_hz: float = 1.0
    sound_speed: float = 1500.0
    noise_seed: int | None = None


@dataclass
class Transponder:
    """An acoustic transponder placed in the environment."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    transponder_id: int = 0


class USBL:
    """USBL positioning sensor computing range + bearing to transponders."""

    def __init__(self, config: USBLConfig | None = None) -> None:
        cfg = config or USBLConfig()
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.noise_seed)
        self.transponders: list[Transponder] = []

    def add_transponder(self, position: np.ndarray, transponder_id: int = 0) -> None:
        self.transponders.append(
            Transponder(position=np.asarray(position, dtype=np.float32), transponder_id=transponder_id)
        )

    def measure(
        self,
        vehicle_position: np.ndarray,
        vehicle_orientation: np.ndarray | None = None,
    ) -> list[dict]:
        """Compute range + bearing to all transponders.

        Returns list of dicts with keys: transponder_id, range_m, bearing_rad,
        elevation_rad, position, valid.
        """
        pos = np.asarray(vehicle_position, dtype=np.float32)
        results = []
        for t in self.transponders:
            diff = t.position - pos
            true_range = float(np.linalg.norm(diff))

            if true_range > self.cfg.max_range or true_range < 0.1:
                results.append({
                    "transponder_id": t.transponder_id,
                    "range_m": 0.0,
                    "bearing_rad": 0.0,
                    "elevation_rad": 0.0,
                    "position": t.position.copy(),
                    "valid": False,
                })
                continue

            range_noise = self._rng.normal(0, self.cfg.range_accuracy_pct / 100.0 * true_range)
            measured_range = true_range + range_noise

            bearing = math.atan2(float(diff[1]), float(diff[0]))
            elevation = math.atan2(float(diff[2]), float(np.linalg.norm(diff[:2])))

            bearing_noise = self._rng.normal(0, math.radians(self.cfg.bearing_accuracy_deg))
            elev_noise = self._rng.normal(0, math.radians(self.cfg.bearing_accuracy_deg))

            results.append({
                "transponder_id": t.transponder_id,
                "range_m": float(measured_range),
                "bearing_rad": float(bearing + bearing_noise),
                "elevation_rad": float(elevation + elev_noise),
                "position": t.position.copy(),
                "valid": True,
            })
        return results
