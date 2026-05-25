"""Realistic propeller thruster model based on Wageningen B-series (simplified)."""

from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.kernel
def _thrust_batch_kernel(
    rpms: wp.array(dtype=wp.float32),
    velocities: wp.array(dtype=wp.float32),
    diameter: wp.float32,
    rho: wp.float32,
    kt0: wp.float32,
    kq0: wp.float32,
    out_thrust: wp.array(dtype=wp.float32),
    out_torque: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    rpm = rpms[i]
    va = velocities[i]
    sign = 1.0
    if rpm < 0.0:
        sign = -1.0
        rpm = -rpm
    n = rpm / 60.0
    if n < 1.0e-6:
        out_thrust[i] = 0.0
        out_torque[i] = 0.0
        return
    j = va / (n * diameter)
    kt = kt0 * wp.max(1.0 - 0.8 * j, 0.0)
    kq = kq0 * wp.max(1.0 - 0.6 * j, 0.0)
    out_thrust[i] = sign * kt * rho * n * n * diameter * diameter * diameter * diameter
    out_torque[i] = sign * kq * rho * n * n * diameter * diameter * diameter * diameter * diameter


class PropellerThruster:
    """Realistic propeller model with thrust/torque curves.

    Simplified Wageningen B-series: Kt and Kq decrease linearly with
    advance ratio J = V_a / (n * D). Cavitation check based on depth pressure.
    """

    VAPOR_PRESSURE = 2338.0  # Pa at 20°C
    ATM_PRESSURE = 101325.0
    GRAVITY = 9.81

    def __init__(
        self,
        diameter: float = 0.076,
        max_rpm: float = 3500.0,
        kt0: float = 0.4,
        kq0: float = 0.04,
        rho: float = 1025.0,
    ) -> None:
        self.diameter = diameter
        self.max_rpm = max_rpm
        self.kt0 = kt0
        self.kq0 = kq0
        self.rho = rho

    def _advance_ratio(self, rpm: float, advance_velocity: float) -> float:
        n = abs(rpm) / 60.0
        if n < 1e-8:
            return 0.0
        return advance_velocity / (n * self.diameter)

    def _kt(self, j: float) -> float:
        return self.kt0 * max(1.0 - 0.8 * j, 0.0)

    def _kq(self, j: float) -> float:
        return self.kq0 * max(1.0 - 0.6 * j, 0.0)

    def thrust(self, rpm: float, advance_velocity: float = 0.0) -> float:
        sign = math.copysign(1.0, rpm) if rpm != 0 else 0.0
        n = abs(rpm) / 60.0
        j = self._advance_ratio(rpm, advance_velocity)
        kt = self._kt(j)
        return sign * kt * self.rho * n**2 * self.diameter**4

    def torque(self, rpm: float, advance_velocity: float = 0.0) -> float:
        sign = math.copysign(1.0, rpm) if rpm != 0 else 0.0
        n = abs(rpm) / 60.0
        j = self._advance_ratio(rpm, advance_velocity)
        kq = self._kq(j)
        return sign * kq * self.rho * n**2 * self.diameter**5

    def efficiency(self, rpm: float, advance_velocity: float) -> float:
        j = self._advance_ratio(rpm, advance_velocity)
        kt = self._kt(j)
        kq = self._kq(j)
        if kq < 1e-12 or j < 1e-12:
            return 0.0
        return j * kt / (2.0 * math.pi * kq)

    def is_cavitating(self, rpm: float, depth: float) -> bool:
        p_local = self.ATM_PRESSURE + self.rho * self.GRAVITY * abs(depth)
        n = abs(rpm) / 60.0
        tip_speed = math.pi * n * self.diameter
        if tip_speed < 1e-8:
            return False
        p_dynamic = 0.5 * self.rho * tip_speed**2
        sigma = (p_local - self.VAPOR_PRESSURE) / max(p_dynamic, 1e-8)
        return sigma < 0.5

    def thrust_gpu(
        self, rpms: wp.array, velocities: wp.array, device: str = "cuda:0"
    ) -> tuple[wp.array, wp.array]:
        n = rpms.shape[0]
        out_thrust = wp.zeros(n, dtype=wp.float32, device=device)
        out_torque = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            _thrust_batch_kernel,
            dim=n,
            inputs=[rpms, velocities, self.diameter, self.rho, self.kt0, self.kq0],
            outputs=[out_thrust, out_torque],
            device=device,
        )
        return out_thrust, out_torque
