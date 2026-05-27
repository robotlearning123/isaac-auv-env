"""GPU imaging sonar using Warp mesh_query_ray — Oculus M370 defaults.

Standalone sensor producing 2D sonar images (range x bearing) via ray casting
against a wp.Mesh scene. No Isaac Sim dependency.
"""

# Adapted from OceanSim (https://github.com/umfieldrobotics/OceanSim)
# Original: isaacsim/oceansim/sensors/ImagingSonarSensor.py | License: Apache-2.0
# Paper: Song et al., "OceanSim", IROS 2025
# Modifications: Standalone Warp mesh_query_ray (no Isaac Sim), OceanScale API pattern

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _sonar_fan_raycast(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_range: wp.array(dtype=wp.float32),
    out_normal_dot: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, origin, d, max_range)
    if query.result:
        out_range[tid] = query.t
        face_normal = wp.mesh_eval_face_normal(mesh_id, query.face)
        cos_inc = wp.abs(wp.dot(wp.normalize(-d), face_normal))
        out_normal_dot[tid] = cos_inc
    else:
        out_range[tid] = max_range
        out_normal_dot[tid] = 0.0


@wp.kernel
def _bin_returns(
    ray_ranges: wp.array(dtype=wp.float32),
    ray_normals: wp.array(dtype=wp.float32),
    ray_bearing_idx: wp.array(dtype=wp.int32),
    attenuation: wp.float32,
    min_range: wp.float32,
    range_res: wp.float32,
    n_range_bins: wp.int32,
    bin_sum: wp.array(ndim=2, dtype=wp.float32),
    bin_count: wp.array(ndim=2, dtype=wp.int32),
):
    tid = wp.tid()
    r = ray_ranges[tid]
    cos_theta = ray_normals[tid]
    b = ray_bearing_idx[tid]

    if cos_theta <= 0.0:
        return
    r_bin = wp.int32((r - min_range) / range_res)
    if r_bin < 0 or r_bin >= n_range_bins:
        return

    intensity = cos_theta * wp.exp(-attenuation * r)
    wp.atomic_add(bin_sum, r_bin, b, intensity)
    wp.atomic_add(bin_count, r_bin, b, 1)


@wp.kernel
def _average_bins(
    bin_sum: wp.array(ndim=2, dtype=wp.float32),
    bin_count: wp.array(ndim=2, dtype=wp.int32),
    avg: wp.array(ndim=2, dtype=wp.float32),
):
    i, j = wp.tid()
    c = bin_count[i, j]
    if c > 0:
        avg[i, j] = bin_sum[i, j] / wp.float32(c)


@wp.kernel
def _range_max(
    arr: wp.array(ndim=2, dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    wp.atomic_max(out, i, arr[i, j])


@wp.kernel
def _apply_noise_and_normalize(
    intensity: wp.array(ndim=2, dtype=wp.float32),
    range_max_val: wp.array(dtype=wp.float32),
    r_grid: wp.array(ndim=2, dtype=wp.float32),
    azi_grid: wp.array(ndim=2, dtype=wp.float32),
    max_range: wp.float32,
    seed: wp.int32,
    gau_scale: wp.float32,
    ray_scale: wp.float32,
    out_image: wp.array(ndim=2, dtype=wp.float32),
):
    i, j = wp.tid()
    val = intensity[i, j]

    rm = range_max_val[i]
    if rm > 0.0:
        val = val / rm

    state = wp.rand_init(seed, i * intensity.shape[1] + j)
    gau = gau_scale * wp.randn(state)
    val = val * (0.5 + gau)

    n1 = wp.randn(state)
    n2 = wp.randn(state)
    rayleigh = ray_scale * wp.sqrt(n1 * n1 + n2 * n2)
    range_frac = r_grid[i, j] / max_range
    val = val + range_frac * range_frac * rayleigh

    out_image[i, j] = wp.clamp(val, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Config + sensor class
# ---------------------------------------------------------------------------

@dataclass
class ImagingSonarConfig:
    """Oculus M370-series defaults."""

    min_range: float = 0.2
    max_range: float = 3.0
    range_res: float = 0.008
    hfov_deg: float = 130.0
    vfov_deg: float = 20.0
    angular_res_deg: float = 0.5
    n_elevation_rays: int = 5
    attenuation: float = 0.1
    gau_noise: float = 0.2
    ray_noise: float = 0.05


class ImagingSonar:
    """GPU forward-looking imaging sonar producing 2D range-bearing images.

    Uses ``wp.mesh_query_ray`` to cast a fan of beams, bins returns into a
    polar image (range x bearing), and applies physically-motivated noise.
    """

    def __init__(
        self,
        environment_mesh: wp.Mesh,
        config: ImagingSonarConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        cfg = config or ImagingSonarConfig()
        self.cfg = cfg
        self.mesh = environment_mesh
        self.device = device

        self.n_range_bins = int((cfg.max_range - cfg.min_range) / cfg.range_res)
        self.n_bearing_bins = int(cfg.hfov_deg / cfg.angular_res_deg)

        bearing_angles = np.linspace(
            -math.radians(cfg.hfov_deg / 2),
            math.radians(cfg.hfov_deg / 2),
            self.n_bearing_bins,
            dtype=np.float32,
        )
        elev_angles = np.linspace(
            -math.radians(cfg.vfov_deg / 2),
            math.radians(cfg.vfov_deg / 2),
            cfg.n_elevation_rays,
            dtype=np.float32,
        )

        dirs_list: list[np.ndarray] = []
        bearing_idx_list: list[int] = []
        for bi, ba in enumerate(bearing_angles):
            for ea in elev_angles:
                dx = math.cos(ba) * math.cos(ea)
                dy = math.sin(ba) * math.cos(ea)
                dz = math.sin(ea)
                dirs_list.append(np.array([dx, dy, dz], dtype=np.float32))
                bearing_idx_list.append(bi)

        self.n_total_rays = len(dirs_list)
        self._body_dirs = np.array(dirs_list, dtype=np.float32)
        self._bearing_idx_np = np.array(bearing_idx_list, dtype=np.int32)

        self._wp_dirs = wp.array(self._body_dirs, dtype=wp.vec3f, device=device)
        self._bearing_idx = wp.array(self._bearing_idx_np, dtype=wp.int32, device=device)
        self._ray_ranges = wp.zeros(self.n_total_rays, dtype=wp.float32, device=device)
        self._ray_normals = wp.zeros(self.n_total_rays, dtype=wp.float32, device=device)

        r_np = np.linspace(cfg.min_range, cfg.max_range, self.n_range_bins, dtype=np.float32)
        a_np = np.linspace(
            math.radians(90 - cfg.hfov_deg / 2),
            math.radians(90 + cfg.hfov_deg / 2),
            self.n_bearing_bins,
            dtype=np.float32,
        )
        r_grid, azi_grid = np.meshgrid(r_np, a_np, indexing="ij")
        self._r_grid = wp.array(r_grid.astype(np.float32), dtype=wp.float32, device=device)
        self._azi_grid = wp.array(azi_grid.astype(np.float32), dtype=wp.float32, device=device)

        self._bin_sum = wp.zeros((self.n_range_bins, self.n_bearing_bins), dtype=wp.float32, device=device)
        self._bin_count = wp.zeros((self.n_range_bins, self.n_bearing_bins), dtype=wp.int32, device=device)
        self._binned = wp.zeros((self.n_range_bins, self.n_bearing_bins), dtype=wp.float32, device=device)
        self._range_max = wp.zeros(self.n_range_bins, dtype=wp.float32, device=device)
        self._image = wp.zeros((self.n_range_bins, self.n_bearing_bins), dtype=wp.float32, device=device)

        self._frame_id = 0

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.n_range_bins, self.n_bearing_bins)

    def scan(
        self,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cast sonar beams and return a 2D sonar image (range x bearing).

        Args:
            position: (3,) world-frame sensor position.
            orientation: (4,) quaternion [x, y, z, w]. None = identity.

        Returns:
            (n_range_bins, n_bearing_bins) float32 array, values in [0, 1].
        """
        origin = wp.vec3(float(position[0]), float(position[1]), float(position[2]))

        if orientation is not None:
            dirs = self._rotate_rays(orientation)
            wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=self.device)
        else:
            wp_dirs = self._wp_dirs

        wp.launch(
            _sonar_fan_raycast,
            dim=self.n_total_rays,
            inputs=[self.mesh.id, origin, wp_dirs, self.cfg.max_range,
                    self._ray_ranges, self._ray_normals],
            device=self.device,
        )

        self._bin_sum.zero_()
        self._bin_count.zero_()
        self._binned.zero_()

        wp.launch(
            _bin_returns,
            dim=self.n_total_rays,
            inputs=[self._ray_ranges, self._ray_normals, self._bearing_idx,
                    self.cfg.attenuation, self.cfg.min_range, self.cfg.range_res,
                    self.n_range_bins, self._bin_sum, self._bin_count],
            device=self.device,
        )

        wp.launch(
            _average_bins,
            dim=(self.n_range_bins, self.n_bearing_bins),
            inputs=[self._bin_sum, self._bin_count, self._binned],
            device=self.device,
        )

        self._range_max.zero_()
        wp.launch(
            _range_max,
            dim=(self.n_range_bins, self.n_bearing_bins),
            inputs=[self._binned, self._range_max],
            device=self.device,
        )

        wp.launch(
            _apply_noise_and_normalize,
            dim=(self.n_range_bins, self.n_bearing_bins),
            inputs=[self._binned, self._range_max, self._r_grid, self._azi_grid,
                    self.cfg.max_range, self._frame_id,
                    self.cfg.gau_noise, self.cfg.ray_noise, self._image],
            device=self.device,
        )

        wp.synchronize_device(self.device)
        self._frame_id += 1
        return self._image.numpy().copy()

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
