"""Realistic ocean water column properties with depth-varying profiles.

Physics references:
- Density: Fofonoff & Millard (1983) UNESCO polynomial EOS-80
- Sound speed: Mackenzie (1981) J. Acoust. Soc. Am. 70(3), 807-812
- Temperature: hyperbolic tangent thermocline model
- Light: Beer-Lambert exponential attenuation
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()

GRAVITY = 9.81


@wp.kernel
def _density_kernel(
    depths: wp.array(dtype=wp.float32),
    surface_temp: wp.float32,
    bottom_temp: wp.float32,
    thermo_depth: wp.float32,
    thermo_thick: wp.float32,
    surface_sal: wp.float32,
    bottom_sal: wp.float32,
    max_depth: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    z = wp.clamp(depths[i], 0.0, max_depth)

    frac = 0.5 * (1.0 + wp.tanh((z - thermo_depth) / thermo_thick))
    T = surface_temp + (bottom_temp - surface_temp) * frac
    S = surface_sal + (bottom_sal - surface_sal) * (z / max_depth)

    rho_w = (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T * T
        + 1.001685e-4 * T * T * T
        - 1.120083e-6 * T * T * T * T
        + 6.536332e-9 * T * T * T * T * T
    )
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T * T - 8.2467e-7 * T * T * T + 5.3875e-9 * T * T * T * T
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T * T
    C = 4.8314e-4
    out[i] = rho_w + A * S + B * S * wp.sqrt(S) + C * S * S


@wp.kernel
def _sound_speed_kernel(
    depths: wp.array(dtype=wp.float32),
    surface_temp: wp.float32,
    bottom_temp: wp.float32,
    thermo_depth: wp.float32,
    thermo_thick: wp.float32,
    surface_sal: wp.float32,
    bottom_sal: wp.float32,
    max_depth: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    z = wp.clamp(depths[i], 0.0, max_depth)

    frac = 0.5 * (1.0 + wp.tanh((z - thermo_depth) / thermo_thick))
    T = surface_temp + (bottom_temp - surface_temp) * frac
    S = surface_sal + (bottom_sal - surface_sal) * (z / max_depth)

    out[i] = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T * T
        + 2.374e-4 * T * T * T
        + 1.340 * (S - 35.0)
        + 1.630e-2 * z
        + 1.675e-7 * z * z
        - 7.139e-13 * T * z * z * z
    )


class WaterColumn:
    """Realistic ocean water column with depth-varying properties."""

    def __init__(
        self,
        surface_temperature: float = 20.0,
        bottom_temperature: float = 4.0,
        thermocline_depth: float = 50.0,
        thermocline_thickness: float = 20.0,
        surface_salinity: float = 35.0,
        bottom_salinity: float = 34.8,
        max_depth: float = 200.0,
        attenuation_coeff: float = 0.08,
        device: str = "cuda:0",
    ) -> None:
        self.surface_temperature = surface_temperature
        self.bottom_temperature = bottom_temperature
        self.thermocline_depth = thermocline_depth
        self.thermocline_thickness = thermocline_thickness
        self.surface_salinity = surface_salinity
        self.bottom_salinity = bottom_salinity
        self.max_depth = max_depth
        self.attenuation_coeff = attenuation_coeff
        self.device = device

    def _frac(self, depth: float) -> float:
        d = max(0.0, min(depth, self.max_depth))
        return 0.5 * (1.0 + math.tanh((d - self.thermocline_depth) / self.thermocline_thickness))

    def temperature(self, depth: float) -> float:
        return self.surface_temperature + (self.bottom_temperature - self.surface_temperature) * self._frac(depth)

    def salinity(self, depth: float) -> float:
        d = max(0.0, min(depth, self.max_depth))
        return self.surface_salinity + (self.bottom_salinity - self.surface_salinity) * (d / self.max_depth)

    def density(self, depth: float) -> float:
        T = self.temperature(depth)
        S = self.salinity(depth)
        rho_w = (
            999.842594
            + 6.793952e-2 * T
            - 9.095290e-3 * T**2
            + 1.001685e-4 * T**3
            - 1.120083e-6 * T**4
            + 6.536332e-9 * T**5
        )
        A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T**2 - 8.2467e-7 * T**3 + 5.3875e-9 * T**4
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
        C = 4.8314e-4
        return rho_w + A * S + B * S * math.sqrt(S) + C * S**2

    def sound_speed(self, depth: float) -> float:
        T = self.temperature(depth)
        S = self.salinity(depth)
        z = max(0.0, min(depth, self.max_depth))
        return (
            1448.96
            + 4.591 * T
            - 5.304e-2 * T**2
            + 2.374e-4 * T**3
            + 1.340 * (S - 35.0)
            + 1.630e-2 * z
            + 1.675e-7 * z**2
            - 7.139e-13 * T * z**3
        )

    def buoyancy_force(self, depth: float, displaced_volume: float) -> float:
        return self.density(depth) * GRAVITY * displaced_volume

    def light_attenuation(self, depth: float) -> float:
        d = max(0.0, depth)
        return math.exp(-self.attenuation_coeff * d)

    def get_profile(self, depths: np.ndarray) -> dict:
        n = len(depths)
        result = {
            "depth": depths.copy(),
            "temperature": np.empty(n),
            "salinity": np.empty(n),
            "density": np.empty(n),
            "sound_speed": np.empty(n),
            "light": np.empty(n),
        }
        for i, d in enumerate(depths):
            result["temperature"][i] = self.temperature(d)
            result["salinity"][i] = self.salinity(d)
            result["density"][i] = self.density(d)
            result["sound_speed"][i] = self.sound_speed(d)
            result["light"][i] = self.light_attenuation(d)
        return result

    def density_gpu(self, depths: wp.array) -> wp.array:
        out = wp.zeros(depths.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            _density_kernel,
            dim=depths.shape[0],
            inputs=[
                depths,
                float(self.surface_temperature),
                float(self.bottom_temperature),
                float(self.thermocline_depth),
                float(self.thermocline_thickness),
                float(self.surface_salinity),
                float(self.bottom_salinity),
                float(self.max_depth),
                out,
            ],
            device=self.device,
        )
        return out

    def sound_speed_gpu(self, depths: wp.array) -> wp.array:
        out = wp.zeros(depths.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(
            _sound_speed_kernel,
            dim=depths.shape[0],
            inputs=[
                depths,
                float(self.surface_temperature),
                float(self.bottom_temperature),
                float(self.thermocline_depth),
                float(self.thermocline_thickness),
                float(self.surface_salinity),
                float(self.bottom_salinity),
                float(self.max_depth),
                out,
            ],
            device=self.device,
        )
        return out
