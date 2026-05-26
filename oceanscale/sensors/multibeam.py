"""GPU multibeam echo sounder (MBES) producing bathymetric depth profiles.

Fan-shaped beam array cast perpendicular to vehicle track.  Uses
``wp.mesh_query_ray`` — no Isaac Sim dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _mbes_raycast(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_ranges: wp.array(dtype=wp.float32),
    out_intensity: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, origin, d, max_range)
    if query.result:
        out_ranges[tid] = query.t
        face_n = wp.mesh_eval_face_normal(mesh_id, query.face)
        out_intensity[tid] = wp.abs(wp.dot(wp.normalize(-d), face_n))
    else:
        out_ranges[tid] = max_range
        out_intensity[tid] = 0.0


@dataclass
class MultibeamConfig:
    """MBES parameters (Kongsberg EM-2040 defaults)."""

    n_beams: int = 256
    swath_angle_deg: float = 140.0
    max_range: float = 500.0
    frequency_khz: float = 300.0
    range_noise_std: float = 0.02
    angular_noise_std_deg: float = 0.1


class MultibeamSonar:
    """GPU multibeam echo sounder producing bathymetric profiles.

    Returns per-beam range + intensity, and optionally a point cloud.
    """

    def __init__(
        self,
        environment_mesh: wp.Mesh,
        config: MultibeamConfig | None = None,
        device: str = "cuda:0",
        seed: int | None = None,
    ) -> None:
        cfg = config or MultibeamConfig()
        self.cfg = cfg
        self.mesh = environment_mesh
        self.device = device
        self._rng = np.random.RandomState(seed)

        half_swath = math.radians(cfg.swath_angle_deg / 2)
        angles = np.linspace(-half_swath, half_swath, cfg.n_beams, dtype=np.float32)

        dirs = np.zeros((cfg.n_beams, 3), dtype=np.float32)
        for i, a in enumerate(angles):
            dirs[i] = [0.0, math.sin(a), -math.cos(a)]
        self._body_dirs = dirs
        self._wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=device)
        self._ranges = wp.zeros(cfg.n_beams, dtype=wp.float32, device=device)
        self._intensity = wp.zeros(cfg.n_beams, dtype=wp.float32, device=device)

    def scan(
        self,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Cast a single MBES swath.

        Returns dict with keys ``ranges``, ``intensity``, ``points``.
        """
        origin = wp.vec3(float(position[0]), float(position[1]), float(position[2]))

        if orientation is not None:
            dirs = self._rotate_rays(orientation)
            wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=self.device)
        else:
            wp_dirs = self._wp_dirs
            dirs = self._body_dirs

        wp.launch(
            _mbes_raycast,
            dim=self.cfg.n_beams,
            inputs=[self.mesh.id, origin, wp_dirs, self.cfg.max_range,
                    self._ranges, self._intensity],
            device=self.device,
        )

        wp.synchronize_device(self.device)
        ranges = self._ranges.numpy().copy()
        intensity = self._intensity.numpy().copy()

        if self.cfg.range_noise_std > 0:
            ranges += self._rng.normal(0, self.cfg.range_noise_std, ranges.shape).astype(np.float32)

        pos = np.asarray(position, dtype=np.float32)
        points = pos[np.newaxis, :] + dirs * ranges[:, np.newaxis]

        return {"ranges": ranges, "intensity": intensity, "points": points}

    def _rotate_rays(self, quat: np.ndarray) -> np.ndarray:
        qx, qy, qz, qw = quat
        d = self._body_dirs
        tx = 2.0 * (qy * d[:, 2] - qz * d[:, 1])
        ty = 2.0 * (qz * d[:, 0] - qx * d[:, 2])
        tz = 2.0 * (qx * d[:, 1] - qy * d[:, 0])
        out = np.empty_like(d)
        out[:, 0] = d[:, 0] + qw * tx + qy * tz - qz * ty
        out[:, 1] = d[:, 1] + qw * ty + qz * tx - qx * tz
        out[:, 2] = d[:, 2] + qw * tz + qx * ty - qy * tx
        return out
