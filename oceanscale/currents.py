# mypy: ignore-errors
"""Realistic 3D ocean current model combining tidal, Ekman, boundary layer, and turbulence."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()

M2_PERIOD = 12.42 * 3600.0  # 44712 s
S2_PERIOD = 12.00 * 3600.0  # 43200 s
KARMAN = 0.41
Z0_ROUGHNESS = 0.001  # seabed roughness length (m)


@wp.kernel
def _current_kernel(
    positions: wp.array(dtype=wp.vec3f),
    time: wp.float32,
    tidal_amp: wp.float32,
    tidal_dir_cos: wp.float32,
    tidal_dir_sin: wp.float32,
    s2_amp: wp.float32,
    ekman_u0: wp.float32,
    ekman_dir_cos: wp.float32,
    ekman_dir_sin: wp.float32,
    ekman_depth: wp.float32,
    bg_u: wp.float32,
    bg_v: wp.float32,
    seabed_depth: wp.float32,
    turb_scale: wp.float32,
    out: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    p = positions[tid]
    z = p[2]

    # --- Tidal: M2 + S2 harmonic ---
    m2_phase = 2.0 * 3.14159265 * time / 44712.0
    s2_phase = 2.0 * 3.14159265 * time / 43200.0
    tidal_mag = tidal_amp * wp.cos(m2_phase) + s2_amp * wp.cos(s2_phase)
    u_tide = tidal_mag * tidal_dir_cos
    v_tide = tidal_mag * tidal_dir_sin

    # --- Ekman spiral (z <= 0, surface at z=0) ---
    u_ek = 0.0
    v_ek = 0.0
    if z > -ekman_depth * 3.0 and ekman_u0 > 0.0:
        z_norm = z / ekman_depth
        decay = wp.exp(z_norm)
        angle = 3.14159265 / 4.0 + z_norm
        ek_u_local = ekman_u0 * decay * wp.cos(angle)
        ek_v_local = ekman_u0 * decay * wp.sin(angle)
        u_ek = ek_u_local * ekman_dir_cos - ek_v_local * ekman_dir_sin
        v_ek = ek_u_local * ekman_dir_sin + ek_v_local * ekman_dir_cos

    # --- Boundary layer (log-law near seabed) ---
    height_above_bottom = z + seabed_depth
    bl_factor = 1.0
    if height_above_bottom < 5.0 and height_above_bottom > 0.001:
        bl_factor = wp.log(height_above_bottom / 0.001) / wp.log(5.0 / 0.001)
    elif height_above_bottom <= 0.001:
        bl_factor = 0.0

    # --- Turbulence (deterministic pseudo-random from position+time) ---
    seed_x = p[0] * 3.17 + time * 0.13
    seed_y = p[1] * 2.71 + time * 0.17
    turb_u = turb_scale * wp.sin(seed_x * 7.3 + seed_y * 5.1) * wp.cos(z * 2.3 + time * 0.31)
    turb_v = turb_scale * wp.cos(seed_x * 5.7 + seed_y * 3.9) * wp.sin(z * 1.7 + time * 0.23)

    u_total = (u_tide + u_ek + bg_u + turb_u) * bl_factor
    v_total = (v_tide + v_ek + bg_v + turb_v) * bl_factor
    w_total = 0.0

    out[tid] = wp.vec3f(u_total, v_total, w_total)


class OceanCurrentField:
    """Realistic 3D ocean current model combining multiple physical components.

    Components:
    1. Tidal currents — M2 + S2 harmonic constituents
    2. Ekman spiral — wind-driven, rotates and decays with depth
    3. Background geostrophic flow — steady large-scale current
    4. Boundary layer — log-law velocity reduction near seabed
    5. Turbulent eddies — deterministic pseudo-random perturbations
    """

    def __init__(
        self,
        tidal_amplitude: float = 0.5,
        tidal_direction: float = 0.0,
        s2_amplitude: float = 0.15,
        wind_speed: float = 5.0,
        wind_direction: float = 0.0,
        ekman_depth: float = 50.0,
        background_speed: float = 0.1,
        background_direction: float = 0.0,
        seabed_depth: float = 50.0,
        turbulence_intensity: float = 0.05,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        self.seabed_depth = seabed_depth

        self._tidal_amp = tidal_amplitude
        self._tidal_dir_cos = math.cos(tidal_direction)
        self._tidal_dir_sin = math.sin(tidal_direction)
        self._s2_amp = s2_amplitude

        self._ekman_u0 = 0.0127 * wind_speed
        self._ekman_dir_cos = math.cos(wind_direction)
        self._ekman_dir_sin = math.sin(wind_direction)
        self._ekman_depth = ekman_depth

        self._bg_u = background_speed * math.cos(background_direction)
        self._bg_v = background_speed * math.sin(background_direction)

        base_speed = max(tidal_amplitude + background_speed, 0.1)
        self._turb_scale = turbulence_intensity * base_speed

    def velocity_at(self, positions: np.ndarray, time: float) -> np.ndarray:
        pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device=self.device)
        out = wp.zeros(len(positions), dtype=wp.vec3f, device=self.device)
        wp.launch(
            _current_kernel,
            dim=len(positions),
            inputs=[
                pos_wp, float(time),
                self._tidal_amp, self._tidal_dir_cos, self._tidal_dir_sin, self._s2_amp,
                self._ekman_u0, self._ekman_dir_cos, self._ekman_dir_sin, self._ekman_depth,
                self._bg_u, self._bg_v,
                self.seabed_depth, self._turb_scale,
            ],
            outputs=[out],
            device=self.device,
        )
        wp.synchronize()
        return out.numpy()

    def velocity_at_gpu(self, positions: wp.array, time: float) -> wp.array:
        n = positions.shape[0]
        out = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        wp.launch(
            _current_kernel,
            dim=n,
            inputs=[
                positions, float(time),
                self._tidal_amp, self._tidal_dir_cos, self._tidal_dir_sin, self._s2_amp,
                self._ekman_u0, self._ekman_dir_cos, self._ekman_dir_sin, self._ekman_depth,
                self._bg_u, self._bg_v,
                self.seabed_depth, self._turb_scale,
            ],
            outputs=[out],
            device=self.device,
        )
        return out

    def tidal_velocity(self, time: float) -> np.ndarray:
        m2 = self._tidal_amp * math.cos(2.0 * math.pi * time / M2_PERIOD)
        s2 = self._s2_amp * math.cos(2.0 * math.pi * time / S2_PERIOD)
        mag = m2 + s2
        return np.array([mag * self._tidal_dir_cos, mag * self._tidal_dir_sin, 0.0])

    def ekman_velocity(self, depth: float) -> np.ndarray:
        z = -abs(depth)
        z_norm = z / self._ekman_depth
        decay = math.exp(z_norm)
        angle = math.pi / 4.0 + z_norm
        u_local = self._ekman_u0 * decay * math.cos(angle)
        v_local = self._ekman_u0 * decay * math.sin(angle)
        u = u_local * self._ekman_dir_cos - v_local * self._ekman_dir_sin
        v = u_local * self._ekman_dir_sin + v_local * self._ekman_dir_cos
        return np.array([u, v, 0.0])

    def boundary_layer_factor(self, height_above_bottom: float) -> float:
        if height_above_bottom <= Z0_ROUGHNESS:
            return 0.0
        if height_above_bottom >= 5.0:
            return 1.0
        return math.log(height_above_bottom / Z0_ROUGHNESS) / math.log(5.0 / Z0_ROUGHNESS)

    @classmethod
    def uniform(cls, speed: float = 0.3, direction: float = 0.0, **kw) -> OceanCurrentField:
        """Uniform constant current."""
        return cls(tidal_amplitude=0.0, s2_amplitude=0.0, wind_speed=0.0,
                   background_speed=speed, background_direction=direction,
                   turbulence_intensity=0.0, **kw)

    @classmethod
    def tidal(cls, amplitude: float = 0.5, direction: float = 0.0, **kw) -> OceanCurrentField:
        """Tidal M2+S2 dominant current."""
        return cls(tidal_amplitude=amplitude, tidal_direction=direction,
                   wind_speed=0.0, turbulence_intensity=0.02, **kw)

    @classmethod
    def storm(cls, wind_speed: float = 15.0, wind_dir: float = 0.0, **kw) -> OceanCurrentField:
        """Storm conditions with strong wind-driven Ekman + tidal."""
        return cls(tidal_amplitude=0.8, wind_speed=wind_speed, wind_direction=wind_dir,
                   background_speed=0.3, turbulence_intensity=0.1, **kw)

    @classmethod
    def deep_sea(cls, seabed_depth: float = 200.0, **kw) -> OceanCurrentField:
        """Deep sea with weak currents and thick Ekman layer."""
        return cls(tidal_amplitude=0.1, wind_speed=3.0, background_speed=0.05,
                   seabed_depth=seabed_depth, ekman_depth=100.0,
                   turbulence_intensity=0.01, **kw)

    @classmethod
    def shear(cls, surface_speed: float = 0.5, direction: float = 0.0,
              wind_speed: float = 8.0, **kw) -> OceanCurrentField:
        """Shear current that varies with depth via wind-driven Ekman spiral."""
        return cls(tidal_amplitude=0.0, s2_amplitude=0.0,
                   wind_speed=wind_speed, wind_direction=direction,
                   background_speed=surface_speed, background_direction=direction,
                   turbulence_intensity=0.01, **kw)

    @classmethod
    def turbulent(cls, base_speed: float = 0.2, direction: float = 0.0,
                  turbulence_intensity: float = 0.15, **kw) -> OceanCurrentField:
        """Turbulent current: steady base flow with strong random perturbation."""
        return cls(tidal_amplitude=0.0, s2_amplitude=0.0, wind_speed=0.0,
                   background_speed=base_speed, background_direction=direction,
                   turbulence_intensity=turbulence_intensity, **kw)

    @classmethod
    def harbor(cls, **kw) -> OceanCurrentField:
        """Harbor conditions: strong tidal, shallow, noisy."""
        return cls(tidal_amplitude=0.8, s2_amplitude=0.3, wind_speed=5.0,
                   background_speed=0.15, seabed_depth=15.0,
                   turbulence_intensity=0.08, **kw)


CURRENT_PRESETS = {
    "uniform": OceanCurrentField.uniform,
    "shear": OceanCurrentField.shear,
    "tidal": OceanCurrentField.tidal,
    "turbulent": OceanCurrentField.turbulent,
    "storm": OceanCurrentField.storm,
    "deep_sea": OceanCurrentField.deep_sea,
    "harbor": OceanCurrentField.harbor,
}
