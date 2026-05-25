"""Oscillatory flapping fin propulsion model based on Strouhal number scaling."""

from __future__ import annotations

import math


class FlappingFin:
    """Generates thrust from periodic fin oscillation via lift-based mechanism.

    Optimal propulsive efficiency occurs at Strouhal number St = f*2A/U ≈ 0.2–0.4,
    matching biological swimmers (trout, dolphins, sharks).
    """

    def __init__(
        self,
        span: float = 0.15,
        chord: float = 0.05,
        max_amplitude: float = 0.5,
        cl_alpha: float = 2.0 * math.pi,
        cd0: float = 0.02,
        rho: float = 1025.0,
    ) -> None:
        self.span = span
        self.chord = chord
        self.area = span * chord
        self.max_amplitude = max_amplitude
        self.cl_alpha = cl_alpha
        self.cd0 = cd0
        self.rho = rho

    def strouhal_number(
        self, frequency: float, amplitude: float, flow_velocity: float
    ) -> float:
        if flow_velocity < 1e-8:
            return 0.0
        return frequency * 2.0 * amplitude / flow_velocity

    def forces(
        self,
        frequency: float,
        amplitude: float,
        flow_velocity: float,
        phase: float,
    ) -> tuple[float, float]:
        """Instantaneous (thrust, lateral_force) from a flapping fin.

        Uses quasi-steady blade-element approximation:
        effective angle of attack α_eff from fin heave velocity and freestream,
        then lift → thrust projection and lateral component.
        """
        if frequency < 1e-8 or amplitude < 1e-8:
            return 0.0, 0.0

        omega = 2.0 * math.pi * frequency
        heave_vel = amplitude * omega * math.cos(phase)
        u_eff = math.sqrt(flow_velocity**2 + heave_vel**2)
        if u_eff < 1e-8:
            return 0.0, 0.0

        alpha_eff = math.atan2(heave_vel, max(flow_velocity, 1e-8))
        alpha_clamped = max(min(alpha_eff, 0.25), -0.25)

        cl = self.cl_alpha * alpha_clamped
        cd = self.cd0 + cl**2 / (math.pi * (self.span / self.chord) + 1e-8)

        q = 0.5 * self.rho * u_eff**2
        lift = cl * q * self.area
        drag = cd * q * self.area

        sa = math.sin(alpha_eff)
        ca = math.cos(alpha_eff)
        thrust = lift * sa - drag * ca
        lateral = lift * ca + drag * sa

        return thrust, lateral

    def mean_thrust(
        self, frequency: float, amplitude: float, flow_velocity: float, n_samples: int = 64
    ) -> float:
        total = 0.0
        for i in range(n_samples):
            phase = 2.0 * math.pi * i / n_samples
            t, _ = self.forces(frequency, amplitude, flow_velocity, phase)
            total += t
        return total / n_samples

    def mean_efficiency(
        self, frequency: float, amplitude: float, flow_velocity: float, n_samples: int = 64
    ) -> float:
        if flow_velocity < 1e-8:
            return 0.0
        mean_t = self.mean_thrust(frequency, amplitude, flow_velocity, n_samples)
        power_in = 0.0
        omega = 2.0 * math.pi * frequency
        for i in range(n_samples):
            phase = 2.0 * math.pi * i / n_samples
            _, lat = self.forces(frequency, amplitude, flow_velocity, phase)
            heave_vel = amplitude * omega * math.cos(phase)
            power_in += abs(lat * heave_vel)
        power_in /= n_samples
        if power_in < 1e-12:
            return 0.0
        power_out = mean_t * flow_velocity
        return max(0.0, min(power_out / power_in, 1.0))
