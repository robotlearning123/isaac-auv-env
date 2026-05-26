"""GPU side-scan sonar producing port/starboard waterfall images.

Two downward-angled beams (port/starboard) cast rays perpendicular to
the vehicle track, producing a 2D waterfall image (along-track × range).
Uses ``wp.mesh_query_ray`` for ray casting — no Isaac Sim dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _sss_raycast(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_range: wp.array(dtype=wp.float32),
    out_intensity: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, origin, d, max_range)
    if query.result:
        out_range[tid] = query.t
        face_n = wp.mesh_eval_face_normal(mesh_id, query.face)
        cos_inc = wp.abs(wp.dot(wp.normalize(-d), face_n))
        out_intensity[tid] = cos_inc * wp.exp(-0.05 * query.t)
    else:
        out_range[tid] = max_range
        out_intensity[tid] = 0.0


@wp.kernel
def _sss_bin_to_image(
    ray_ranges: wp.array(dtype=wp.float32),
    ray_intensity: wp.array(dtype=wp.float32),
    n_beams_per_side: wp.int32,
    min_range: wp.float32,
    range_res: wp.float32,
    n_range_bins: wp.int32,
    image: wp.array(ndim=2, dtype=wp.float32),
):
    tid = wp.tid()
    r = ray_ranges[tid]
    inten = ray_intensity[tid]
    if inten <= 0.0:
        return
    r_bin = wp.int32((r - min_range) / range_res)
    if r_bin < 0 or r_bin >= n_range_bins:
        return
    side = tid / n_beams_per_side
    local = tid - side * n_beams_per_side
    col = n_range_bins - 1 - r_bin if side == 0 else n_range_bins + r_bin
    wp.atomic_max(image, local, col, inten)


@dataclass
class SideScanSonarConfig:
    """Side-scan sonar parameters."""

    frequency_khz: float = 400.0
    max_range: float = 50.0
    min_range: float = 1.0
    range_res: float = 0.1
    n_beams_per_side: int = 256
    tilt_angle_deg: float = 20.0
    swath_angle_deg: float = 75.0
    attenuation: float = 0.05
    noise_std: float = 0.05


class SideScanSonar:
    """GPU side-scan sonar producing port/starboard waterfall images.

    Returns a 2D image: rows = pings (along-track), columns = range (port|stbd).
    """

    def __init__(
        self,
        environment_mesh: wp.Mesh,
        config: SideScanSonarConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        cfg = config or SideScanSonarConfig()
        self.cfg = cfg
        self.mesh = environment_mesh
        self.device = device

        self.n_range_bins = int((cfg.max_range - cfg.min_range) / cfg.range_res)
        self.image_width = self.n_range_bins * 2

        tilt = math.radians(cfg.tilt_angle_deg)
        half_swath = math.radians(cfg.swath_angle_deg / 2)
        angles = np.linspace(-half_swath, half_swath, cfg.n_beams_per_side, dtype=np.float32)

        dirs: list[np.ndarray] = []
        for sign in [-1.0, 1.0]:
            for a in angles:
                dy = sign * math.cos(tilt + a)
                dz = -math.sin(tilt + a)
                dx = math.sin(a) * 0.1
                d = np.array([dx, dy, dz], dtype=np.float32)
                dirs.append(d / np.linalg.norm(d))

        self._body_dirs = np.array(dirs, dtype=np.float32)
        self.n_total_rays = len(dirs)
        self._wp_dirs = wp.array(self._body_dirs, dtype=wp.vec3f, device=device)
        self._ray_ranges = wp.zeros(self.n_total_rays, dtype=wp.float32, device=device)
        self._ray_intensity = wp.zeros(self.n_total_rays, dtype=wp.float32, device=device)
        self._ping_line = wp.zeros((cfg.n_beams_per_side, self.image_width), dtype=wp.float32, device=device)

        self._waterfall: list[np.ndarray] = []
        self._max_pings = 512

    @property
    def image_shape(self) -> tuple[int, int]:
        return (min(len(self._waterfall), self._max_pings), self.image_width)

    def ping(
        self,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cast one ping and return the waterfall image accumulated so far.

        Returns:
            (n_pings, image_width) float32 array, values in [0, 1].
        """
        origin = wp.vec3(float(position[0]), float(position[1]), float(position[2]))

        if orientation is not None:
            dirs = self._rotate_rays(orientation)
            wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=self.device)
        else:
            wp_dirs = self._wp_dirs

        wp.launch(
            _sss_raycast,
            dim=self.n_total_rays,
            inputs=[self.mesh.id, origin, wp_dirs, self.cfg.max_range,
                    self._ray_ranges, self._ray_intensity],
            device=self.device,
        )

        self._ping_line.zero_()
        wp.launch(
            _sss_bin_to_image,
            dim=self.n_total_rays,
            inputs=[self._ray_ranges, self._ray_intensity,
                    self.cfg.n_beams_per_side, self.cfg.min_range,
                    self.cfg.range_res, self.n_range_bins, self._ping_line],
            device=self.device,
        )

        wp.synchronize_device(self.device)
        line = self._ping_line.numpy()[0, :].copy()

        if self.cfg.noise_std > 0:
            line += np.random.normal(0, self.cfg.noise_std, line.shape).astype(np.float32)
            np.clip(line, 0, 1, out=line)

        self._waterfall.append(line)
        if len(self._waterfall) > self._max_pings:
            self._waterfall.pop(0)

        return np.array(self._waterfall, dtype=np.float32)

    def _rotate_rays(self, quat: np.ndarray) -> np.ndarray:
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

    def reset(self) -> None:
        self._waterfall.clear()
