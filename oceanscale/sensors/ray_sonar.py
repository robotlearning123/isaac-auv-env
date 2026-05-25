"""Forward-looking sonar using wp.Mesh mesh_query_ray fan scan."""

from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.kernel
def _sonar_raycast_kernel(
    mesh_id: wp.uint64,
    origin: wp.array(dtype=wp.vec3f),
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_ranges: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    o = origin[0]
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, o, d, max_range)
    if query.result:
        out_ranges[tid] = query.t
    else:
        out_ranges[tid] = max_range


def _fan_directions(n_rays: int, fov_deg: float) -> np.ndarray:
    """Generate a horizontal fan with +x forward and z as vertical depth."""
    half_fov = math.radians(fov_deg / 2.0)
    angles = np.linspace(-half_fov, half_fov, n_rays)
    dirs = np.zeros((n_rays, 3), dtype=np.float32)
    dirs[:, 0] = np.cos(angles)
    dirs[:, 1] = np.sin(angles)
    return dirs


class RaySonar:
    """Forward-looking sonar using mesh_query_ray fan scan.

    Casts a fan of rays in the forward direction to detect obstacles.
    Returns per-ray range measurements.
    """

    def __init__(
        self,
        environment_mesh: wp.Mesh,
        n_rays: int = 64,
        fov: float = 90.0,
        max_range: float = 50.0,
        device: str = "cuda:0",
    ) -> None:
        self.mesh = environment_mesh
        self.n_rays = n_rays
        self.fov = fov
        self.max_range = max_range
        self.device = device

        self._body_dirs = _fan_directions(n_rays, fov)
        self._wp_dirs = wp.array(self._body_dirs, dtype=wp.vec3f, device=device)
        self._ranges = wp.zeros(n_rays, dtype=wp.float32, device=device)
        self._origin = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._rotated_dirs = wp.zeros(n_rays, dtype=wp.vec3f, device=device)

    def scan(self, position: np.ndarray, orientation: np.ndarray | None = None) -> np.ndarray:
        """Cast sonar rays and return range measurements.

        Args:
            position: (3,) world-frame position.
            orientation: (4,) quaternion [x, y, z, w]. If None, identity.

        Returns:
            (n_rays,) array of ranges. max_range if no hit.
        """
        wp.copy(self._origin, wp.array(position.reshape(1, 3).astype(np.float32), dtype=wp.vec3f, device=self.device))

        if orientation is not None:
            dirs = self._rotate_rays(orientation)
            wp.copy(self._rotated_dirs, wp.array(dirs, dtype=wp.vec3f, device=self.device))
            dirs_arg = self._rotated_dirs
        else:
            dirs_arg = self._wp_dirs

        wp.launch(
            _sonar_raycast_kernel,
            dim=self.n_rays,
            inputs=[self.mesh.id, self._origin, dirs_arg, self.max_range, self._ranges],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return self._ranges.numpy().copy()

    def _rotate_rays(self, quat: np.ndarray) -> np.ndarray:
        """Rotate body-frame ray directions by quaternion [x,y,z,w]."""
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
