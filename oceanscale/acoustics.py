"""Underwater acoustic propagation: sound speed, ray tracing, transmission loss, noise."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _transmission_loss_kernel(
    ranges: wp.array(dtype=wp.float32),
    alpha_db_per_m: wp.float32,
    out_tl: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    r = ranges[i]
    if r < 1.0:
        r = 1.0
    tl = 20.0 * wp.log(r) / wp.log(10.0) + alpha_db_per_m * r
    out_tl[i] = tl


def mackenzie_sound_speed(temperature: float, salinity: float, depth: float) -> float:
    """Mackenzie (1981) equation for sound speed in seawater.

    Args:
        temperature: degrees Celsius (2-30)
        salinity: PSU (25-40)
        depth: meters (0-8000)

    Returns:
        Sound speed in m/s.
    """
    T = temperature
    S = salinity
    D = depth
    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35.0)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35.0)
        - 7.139e-13 * T * D**3
    )
    return c


def thorp_absorption(frequency_hz: float) -> float:
    """Thorp's absorption coefficient.

    Args:
        frequency_hz: frequency in Hz

    Returns:
        Absorption in dB/m.
    """
    f = frequency_hz / 1000.0  # convert to kHz
    alpha_db_km = (
        0.11 * f**2 / (1.0 + f**2)
        + 44.0 * f**2 / (4100.0 + f**2)
        + 2.75e-4 * f**2
        + 0.003
    )
    return alpha_db_km / 1000.0  # dB/m


def wenz_ambient_noise(frequency_hz: float, sea_state: int = 3, shipping: str = "moderate") -> float:
    """Ambient noise level from Wenz curves (simplified).

    Args:
        frequency_hz: frequency in Hz
        sea_state: Beaufort sea state 0-6
        shipping: "light", "moderate", "heavy"

    Returns:
        Noise spectral density in dB re 1 µPa²/Hz.
    """
    f = frequency_hz
    shipping_offset = {"light": -5.0, "moderate": 0.0, "heavy": 5.0}.get(shipping, 0.0)
    if f < 10.0:
        nl = 107.0 - 30.0 * math.log10(max(f, 0.1)) + shipping_offset
    elif f < 1000.0:
        wind_nl = 44.0 + 23.0 * sea_state - 17.0 * math.log10(f)
        ship_nl = 60.0 - 20.0 * math.log10(f / 10.0) + shipping_offset
        nl = 10.0 * math.log10(10.0 ** (wind_nl / 10.0) + 10.0 ** (ship_nl / 10.0))
    else:
        nl = 44.0 + 23.0 * sea_state - 17.0 * math.log10(f)
        thermal = -15.0 + 20.0 * math.log10(f)
        nl = 10.0 * math.log10(10.0 ** (nl / 10.0) + 10.0 ** (thermal / 10.0))
    return nl


class WaterColumn:
    """Depth-dependent ocean water properties."""

    def __init__(
        self,
        max_depth: float = 200.0,
        surface_temperature: float = 20.0,
        bottom_temperature: float = 4.0,
        salinity: float = 35.0,
        thermocline_depth: float = 50.0,
        thermocline_thickness: float = 30.0,
    ):
        self.max_depth = max_depth
        self.surface_temperature = surface_temperature
        self.bottom_temperature = bottom_temperature
        self.salinity = salinity
        self.thermocline_depth = thermocline_depth
        self.thermocline_thickness = thermocline_thickness

    def temperature_at(self, depth: float) -> float:
        """Temperature profile with thermocline."""
        d = abs(depth)
        t_range = self.surface_temperature - self.bottom_temperature
        x = (d - self.thermocline_depth) / max(self.thermocline_thickness, 1e-6)
        x = max(-20.0, min(20.0, x))
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        offset = 1.0 / (1.0 + math.exp(self.thermocline_depth / max(self.thermocline_thickness, 1e-6)))
        return self.surface_temperature - t_range * (sigmoid - offset) / (1.0 - offset)

    def density_at(self, depth: float) -> float:
        """Seawater density (UNESCO equation simplified)."""
        T = self.temperature_at(depth)
        S = self.salinity
        rho = 1025.0 + 0.2 * (S - 35.0) - 0.15 * (T - 10.0) + 0.045 * abs(depth) / 1000.0
        return rho

    def sound_speed_at(self, depth: float) -> float:
        T = self.temperature_at(depth)
        return mackenzie_sound_speed(T, self.salinity, abs(depth))

    def sound_speed_profile(self, depths: np.ndarray) -> np.ndarray:
        return np.array([self.sound_speed_at(d) for d in depths])


class AcousticPropagation:
    """Underwater acoustic propagation model."""

    def __init__(
        self,
        water_column: WaterColumn | None = None,
        frequency: float = 200e3,
        max_depth: float = 200.0,
        device: str = "cuda:0",
    ):
        self.water_column = water_column or WaterColumn(max_depth=max_depth)
        self.frequency = frequency
        self.max_depth = max_depth
        self.device = device
        self._alpha_db_m = thorp_absorption(frequency)

    def absorption_coefficient(self, frequency: float | None = None) -> float:
        f = frequency if frequency is not None else self.frequency
        return thorp_absorption(f)

    def transmission_loss(self, range_m: float, source_depth: float = 10.0, receiver_depth: float = 10.0) -> float:
        if range_m < 1.0:
            range_m = 1.0
        return 20.0 * math.log10(range_m) + self._alpha_db_m * range_m

    def transmission_loss_gpu(self, ranges: wp.array, source_depth: float = 10.0, receiver_depths: wp.array | None = None) -> wp.array:
        n = ranges.shape[0]
        out = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(
            _transmission_loss_kernel,
            dim=n,
            inputs=[ranges, self._alpha_db_m],
            outputs=[out],
            device=self.device,
        )
        return out

    def ambient_noise(self, frequency: float | None = None, sea_state: int = 3, shipping: str = "moderate") -> float:
        f = frequency if frequency is not None else self.frequency
        return wenz_ambient_noise(f, sea_state, shipping)

    def detection_range(self, source_level: float, noise_level: float | None = None, directivity_index: float = 0.0, detection_threshold: float = 10.0) -> float:
        if noise_level is None:
            noise_level = self.ambient_noise()
        max_tl = source_level - noise_level + directivity_index - detection_threshold
        if max_tl <= 0.0:
            return 0.0
        r = 1.0
        for _ in range(200):
            tl = self.transmission_loss(r)
            if tl >= max_tl:
                break
            r *= 1.1
        lo, hi = r / 1.1, r
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if self.transmission_loss(mid) < max_tl:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def sound_speed_profile(self, depths: np.ndarray) -> np.ndarray:
        return self.water_column.sound_speed_profile(depths)

    def ray_trace(self, source_depth: float, angles_deg: np.ndarray, max_range: float = 1000.0, n_steps: int = 200) -> np.ndarray:
        """Trace acoustic rays through a depth-varying sound speed profile.

        Uses Snell's law: cos(θ)/c = constant along each ray.

        Args:
            source_depth: source depth in meters (positive downward).
            angles_deg: launch angles in degrees (0=horizontal, positive=downward).
            max_range: maximum horizontal range in meters.
            n_steps: number of integration steps.

        Returns:
            (n_rays, n_steps, 2) array of (range, depth) for each ray.
        """
        angles_rad = np.deg2rad(angles_deg)
        n_rays = len(angles_rad)
        ds = max_range / n_steps
        paths = np.zeros((n_rays, n_steps, 2), dtype=np.float64)

        for i, theta0 in enumerate(angles_rad):
            c0 = self.water_column.sound_speed_at(source_depth)
            p = math.cos(theta0) / c0  # Snell invariant (ray parameter)
            going_down = math.sin(theta0) >= 0.0

            r, z = 0.0, source_depth
            for j in range(n_steps):
                paths[i, j, 0] = r
                paths[i, j, 1] = z

                c = self.water_column.sound_speed_at(z)
                cos_theta = p * c
                cos_theta = max(-1.0, min(1.0, cos_theta))
                sin_theta_abs = math.sqrt(1.0 - cos_theta**2)

                if sin_theta_abs < 1e-12:
                    dz_probe = 0.1
                    c_below = self.water_column.sound_speed_at(z + dz_probe)
                    going_down = c_below > c

                sin_theta = sin_theta_abs if going_down else -sin_theta_abs

                dr = ds * cos_theta
                dz = ds * sin_theta
                r += dr
                z += dz

                # Check turning point (where |cos_theta| would exceed 1)
                c_new = self.water_column.sound_speed_at(z)
                if abs(p * c_new) >= 1.0:
                    going_down = not going_down

                if z < 0.0:
                    z = -z
                    going_down = True

                if z > self.max_depth:
                    z = 2.0 * self.max_depth - z
                    going_down = False

                if r >= max_range:
                    paths[i, j + 1 :, 0] = r
                    paths[i, j + 1 :, 1] = z
                    break

        return paths.astype(np.float32)
# mypy: ignore-errors
