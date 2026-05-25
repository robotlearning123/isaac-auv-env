"""Variable buoyancy system for glider-type propulsion."""

from __future__ import annotations


class BuoyancyEngine:
    """Simulates an internal piston/bladder buoyancy engine.

    Changes displaced volume to vary net buoyancy. A pump moves oil
    between an internal reservoir and an external bladder at a finite rate.
    Used in underwater gliders (Slocum, Seaglider, Spray).
    """

    GRAVITY = 9.81

    def __init__(
        self,
        max_volume_change: float = 0.0005,
        pump_rate: float = 0.0001,
        hull_volume: float = 0.05,
        hull_mass: float = 51.0,
    ) -> None:
        self.max_volume_change = max_volume_change
        self.pump_rate = pump_rate
        self.hull_volume = hull_volume
        self.hull_mass = hull_mass
        self._bladder_volume = 0.0
        self._target_volume = 0.0

    @property
    def current_volume(self) -> float:
        return self._bladder_volume

    @property
    def total_displaced_volume(self) -> float:
        return self.hull_volume + self._bladder_volume

    def set_target_volume(self, target: float) -> None:
        self._target_volume = max(-self.max_volume_change, min(target, self.max_volume_change))

    def step(self, dt: float) -> None:
        diff = self._target_volume - self._bladder_volume
        max_step = self.pump_rate * dt
        if abs(diff) <= max_step:
            self._bladder_volume = self._target_volume
        else:
            if diff > 0:
                self._bladder_volume += max_step
            else:
                self._bladder_volume -= max_step
        self._bladder_volume = max(
            -self.max_volume_change, min(self._bladder_volume, self.max_volume_change)
        )

    def net_buoyancy(self, water_density: float = 1025.0) -> float:
        buoyancy = water_density * self.GRAVITY * self.total_displaced_volume
        weight = self.hull_mass * self.GRAVITY
        return buoyancy - weight

    def is_positive(self, water_density: float = 1025.0) -> bool:
        return self.net_buoyancy(water_density) > 0.0

    def reset(self) -> None:
        self._bladder_volume = 0.0
        self._target_volume = 0.0
