# Partial submersion model for amphibious locomotion
# Physics: linear buoyancy transition at free surface (standard hydrostatics)
# Related work: FARMS framework (EPFL, biorxiv 10.1101/2023.09.25.559130)
# Note: independently implemented from first principles, not ported from FARMS code
"""Partial submersion — buoyancy and drag transition at the water surface.

For amphibious robots crossing the air-water interface, each body link
experiences fluid forces proportional to its submerged volume fraction.
Fully submerged links get full buoyancy + drag; links above the surface
get none; partially submerged links interpolate linearly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def partial_submersion_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    link_heights: wp.array(dtype=wp.float32),
    link_volumes: wp.array(dtype=wp.float32),
    water_surface_z: wp.float32,
    water_density: wp.float32,
    gravity: wp.float32,
    drag_coeff: wp.float32,
    buoyancy_out: wp.array(dtype=wp.vec3f),
    drag_out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    pos = positions[i]
    vel = velocities[i]
    h = link_heights[i]
    vol = link_volumes[i]

    z_bottom = pos[2] - h * 0.5
    submerged_height = water_surface_z - z_bottom
    frac = wp.clamp(submerged_height / wp.max(h, 1.0e-6), 0.0, 1.0)

    fb_z = frac * water_density * gravity * vol
    buoyancy_out[i] = wp.vec3f(0.0, 0.0, fb_z)

    speed_sq_x = vel[0] * wp.abs(vel[0])
    speed_sq_y = vel[1] * wp.abs(vel[1])
    speed_sq_z = vel[2] * wp.abs(vel[2])
    drag_scale = -0.5 * water_density * drag_coeff * frac
    drag_out[i] = wp.vec3f(
        drag_scale * speed_sq_x,
        drag_scale * speed_sq_y,
        drag_scale * speed_sq_z,
    )


@dataclass
class PartialSubmersionConfig:
    """Configuration for partial submersion physics."""

    water_surface_z: float = 0.0
    water_density: float = 1025.0
    gravity: float = 9.81
    drag_coeff: float = 1.0


class PartialSubmersion:
    """GPU-accelerated partial submersion model for amphibious robots.

    Each body link independently transitions between air and water based on
    its vertical position relative to the water surface.
    """

    def __init__(
        self,
        config: PartialSubmersionConfig | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = config or PartialSubmersionConfig()
        self.device = device

    def compute(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        link_heights: np.ndarray,
        link_volumes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute buoyancy and drag for each link based on submersion.

        Args:
            positions: (N, 3) link center positions.
            velocities: (N, 3) link velocities.
            link_heights: (N,) link extent along z-axis.
            link_volumes: (N,) link displaced volume in m³.

        Returns:
            buoyancy: (N, 3) buoyancy force vectors (z-up).
            drag: (N, 3) quadratic drag force vectors.
        """
        n = len(positions)
        pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device=self.device)
        vel_wp = wp.array(velocities.astype(np.float32), dtype=wp.vec3f, device=self.device)
        h_wp = wp.array(link_heights.astype(np.float32), dtype=wp.float32, device=self.device)
        v_wp = wp.array(link_volumes.astype(np.float32), dtype=wp.float32, device=self.device)
        buoy = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        drag = wp.zeros(n, dtype=wp.vec3f, device=self.device)

        wp.launch(
            partial_submersion_kernel,
            dim=n,
            inputs=[
                pos_wp,
                vel_wp,
                h_wp,
                v_wp,
                wp.float32(self.cfg.water_surface_z),
                wp.float32(self.cfg.water_density),
                wp.float32(self.cfg.gravity),
                wp.float32(self.cfg.drag_coeff),
                buoy,
                drag,
            ],
            device=self.device,
        )

        return buoy.numpy(), drag.numpy()
