"""DVL sensor using wp.Mesh mesh_query_ray for physical beam simulation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from oceanscale.sensors.dvl_configs import DVLConfig


@wp.kernel
def _dvl_raycast_kernel(
    mesh_id: wp.uint64,
    origin: wp.array(dtype=wp.vec3f),
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_ranges: wp.array(dtype=wp.float32),
    out_normals: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    o = origin[0]
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, o, d, max_range)
    if query.result:
        out_ranges[tid] = query.t
        out_normals[tid] = query.normal
    else:
        out_ranges[tid] = max_range
        out_normals[tid] = wp.vec3f(0.0, 0.0, 0.0)


def _janus_beam_directions(beam_angle_deg: float) -> np.ndarray:
    """4-beam Janus configuration with z-up world convention."""
    a = math.radians(beam_angle_deg)
    sin_a = math.sin(a)
    cos_a = math.cos(a)
    return np.array([
        [sin_a, 0.0, -cos_a],
        [-sin_a, 0.0, -cos_a],
        [0.0, sin_a, -cos_a],
        [0.0, -sin_a, -cos_a],
    ], dtype=np.float32)


class RayDVL:
    """DVL sensor using mesh_query_ray for physical beam simulation.

    Simulates a 4-beam Janus DVL that casts acoustic beams toward the
    seabed mesh and measures per-beam range. Altitude is derived from
    average beam range projected to vertical.
    """

    @classmethod
    def from_config(
        cls,
        config: "DVLConfig",
        seabed_mesh: wp.Mesh,
        device: str = "cuda:0",
    ) -> "RayDVL":
        """Construct from a real-world DVL hardware config."""
        return cls(
            seabed_mesh=seabed_mesh,
            n_beams=config.n_beams,
            beam_angle=config.beam_angle_deg,
            max_range=config.max_range_m,
            device=device,
        )

    def __init__(
        self,
        seabed_mesh: wp.Mesh,
        n_beams: int = 4,
        beam_angle: float = 30.0,
        max_range: float = 200.0,
        device: str = "cuda:0",
    ) -> None:
        self.mesh = seabed_mesh
        self.n_beams = n_beams
        self.beam_angle = beam_angle
        self.max_range = max_range
        self.device = device

        if n_beams != 4:
            raise ValueError(f"RayDVL supports exactly 4 Janus beams, got {n_beams}")
        self._body_dirs = _janus_beam_directions(beam_angle)
        self._wp_dirs = wp.array(self._body_dirs, dtype=wp.vec3f, device=device)
        self._ranges = wp.zeros(n_beams, dtype=wp.float32, device=device)
        self._normals = wp.zeros(n_beams, dtype=wp.vec3f, device=device)
        self._origin = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._rotated_dirs = wp.zeros(n_beams, dtype=wp.vec3f, device=device)

    def measure(self, position: np.ndarray, orientation: np.ndarray | None = None) -> dict:
        """Cast DVL beams and return measurements.

        Args:
            position: (3,) world-frame position.
            orientation: (4,) quaternion [x, y, z, w] for body orientation.
                If None, assumes identity (beams point toward negative z).

        Returns:
            dict with keys: beam_ranges (n_beams,), altitude (float),
            normals (n_beams, 3), valid (bool).
        """
        self._origin.numpy()[0] = position
        wp.copy(self._origin, wp.array(position.reshape(1, 3).astype(np.float32), dtype=wp.vec3f, device=self.device))

        if orientation is not None:
            dirs = self._rotate_beams(orientation)
            wp.copy(self._rotated_dirs, wp.array(dirs, dtype=wp.vec3f, device=self.device))
            dirs_arg = self._rotated_dirs
        else:
            dirs_arg = self._wp_dirs

        wp.launch(
            _dvl_raycast_kernel,
            dim=self.n_beams,
            inputs=[self.mesh.id, self._origin, dirs_arg, self.max_range, self._ranges, self._normals],
            device=self.device,
        )
        wp.synchronize_device(self.device)

        ranges = self._ranges.numpy().copy()
        normals = self._normals.numpy().copy()
        cos_angle = math.cos(math.radians(self.beam_angle))
        valid_mask = ranges < self.max_range
        if valid_mask.any():
            altitude = float(np.mean(ranges[valid_mask]) * cos_angle)
        else:
            altitude = float(self.max_range)

        return {
            "beam_ranges": ranges,
            "altitude": altitude,
            "normals": normals,
            "valid": bool(valid_mask.any()),
        }

    def _rotate_beams(self, quat: np.ndarray) -> np.ndarray:
        """Rotate body-frame beam directions by quaternion [x,y,z,w]."""
        qx, qy, qz, qw = quat
        dirs = np.empty_like(self._body_dirs)
        for i, d in enumerate(self._body_dirs):
            tx = 2.0 * (qy * d[2] - qz * d[1])
            ty = 2.0 * (qz * d[0] - qx * d[2])
            tz = 2.0 * (qx * d[1] - qy * d[0])
            dirs[i, 0] = d[0] + qw * tx + qy * tz - qz * ty
            dirs[i, 1] = d[1] + qw * ty + qz * tx - qx * tz
            dirs[i, 2] = d[2] + qw * tz + qx * ty - qy * tx
        return dirs
# mypy: ignore-errors
