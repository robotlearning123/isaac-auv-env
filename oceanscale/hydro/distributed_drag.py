# Per-link distributed hydrodynamic drag for articulated underwater robots
# Physics: independent quadratic drag on each body segment (standard Morison)
# Related work: FARMS framework (EPFL, biorxiv 10.1101/2023.09.25.559130)
# Note: independently implemented from Morison equation, not ported from FARMS code
"""Per-link distributed drag for articulated underwater bodies.

For articulated robots (manipulators, snake-like, fish) where the Fossen
whole-body model is insufficient, this module computes independent
quadratic drag on each rigid link using its own cross-sectional areas
and drag coefficients. Supports ambient ocean currents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp


@wp.kernel
def distributed_drag_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    angular_velocities: wp.array(dtype=wp.vec3f),
    drag_coeffs: wp.array(dtype=wp.vec3f),
    cross_sections: wp.array(dtype=wp.vec3f),
    water_velocity: wp.vec3f,
    water_density: wp.float32,
    forces_out: wp.array(dtype=wp.vec3f),
    torques_out: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    vel = velocities[i]
    angvel = angular_velocities[i]
    cd = drag_coeffs[i]
    area = cross_sections[i]

    rel_vx = vel[0] - water_velocity[0]
    rel_vy = vel[1] - water_velocity[1]
    rel_vz = vel[2] - water_velocity[2]

    scale = -0.5 * water_density
    fx = scale * cd[0] * area[0] * rel_vx * wp.abs(rel_vx)
    fy = scale * cd[1] * area[1] * rel_vy * wp.abs(rel_vy)
    fz = scale * cd[2] * area[2] * rel_vz * wp.abs(rel_vz)
    forces_out[i] = wp.vec3f(fx, fy, fz)

    rot_scale = -0.5 * water_density * 0.1
    tx = rot_scale * cd[0] * area[0] * angvel[0] * wp.abs(angvel[0])
    ty = rot_scale * cd[1] * area[1] * angvel[1] * wp.abs(angvel[1])
    tz = rot_scale * cd[2] * area[2] * angvel[2] * wp.abs(angvel[2])
    torques_out[i] = wp.vec3f(tx, ty, tz)


@dataclass
class DistributedDragConfig:
    """Per-link drag configuration."""

    water_density: float = 1025.0
    water_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


class DistributedDrag:
    """GPU-accelerated per-link distributed drag for articulated bodies.

    Each link gets independent quadratic drag computed from its own
    cross-sectional area and drag coefficient, with ambient current
    subtracted from the relative velocity.
    """

    def __init__(
        self,
        config: DistributedDragConfig | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = config or DistributedDragConfig()
        self.device = device

    def compute(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        angular_velocities: np.ndarray,
        drag_coeffs: np.ndarray,
        cross_sections: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-link drag forces and torques.

        Args:
            positions: (N, 3) link positions.
            velocities: (N, 3) link linear velocities.
            angular_velocities: (N, 3) link angular velocities.
            drag_coeffs: (N, 3) per-link drag coefficients (cx, cy, cz).
            cross_sections: (N, 3) per-link cross-sectional areas (Ax, Ay, Az) in m².

        Returns:
            forces: (N, 3) drag force vectors.
            torques: (N, 3) drag torque vectors.
        """
        n = len(positions)
        pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device=self.device)
        vel_wp = wp.array(velocities.astype(np.float32), dtype=wp.vec3f, device=self.device)
        angvel_wp = wp.array(angular_velocities.astype(np.float32), dtype=wp.vec3f, device=self.device)
        cd_wp = wp.array(drag_coeffs.astype(np.float32), dtype=wp.vec3f, device=self.device)
        area_wp = wp.array(cross_sections.astype(np.float32), dtype=wp.vec3f, device=self.device)
        forces = wp.zeros(n, dtype=wp.vec3f, device=self.device)
        torques = wp.zeros(n, dtype=wp.vec3f, device=self.device)

        wv = self.cfg.water_velocity
        wp.launch(
            distributed_drag_kernel,
            dim=n,
            inputs=[
                pos_wp,
                vel_wp,
                angvel_wp,
                cd_wp,
                area_wp,
                wp.vec3f(float(wv[0]), float(wv[1]), float(wv[2])),
                wp.float32(self.cfg.water_density),
                forces,
                torques,
            ],
            device=self.device,
        )

        return forces.numpy(), torques.numpy()
