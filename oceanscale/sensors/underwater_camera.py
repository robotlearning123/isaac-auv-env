"""GPU underwater camera — pinhole ray casting + Beer-Lambert attenuation.

Produces synthetic RGB + depth images via Warp ray casting against a wp.Mesh
scene, with per-pixel underwater light transport from Jerlov water presets.
Physics: Jerlov (1976) Marine Optics; Beer-Lambert I(d) = I_0 * exp(-c*d).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from oceanscale.rendering.underwater import JERLOV_PRESETS, WaterParams

# -- Warp kernels --

@wp.kernel
def _camera_raycast(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    directions: wp.array(dtype=wp.vec3f),
    max_range: wp.float32,
    out_depth: wp.array(ndim=2, dtype=wp.float32),
    out_intensity: wp.array(ndim=2, dtype=wp.float32),
    width: wp.int32,
):
    tid = wp.tid()
    d = directions[tid]
    query = wp.mesh_query_ray(mesh_id, origin, d, max_range)
    row = tid / width
    col = tid % width
    if query.result:
        out_depth[row, col] = query.t
        face_n = wp.mesh_eval_face_normal(mesh_id, query.face)
        out_intensity[row, col] = wp.abs(wp.dot(wp.normalize(-d), face_n))
    else:
        out_depth[row, col] = max_range
        out_intensity[row, col] = 0.0


@wp.kernel
def _uw_shade(
    depth: wp.array(ndim=2, dtype=wp.float32),
    intensity: wp.array(ndim=2, dtype=wp.float32),
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    out_rgb: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    d = depth[i, j]
    cos_theta = intensity[i, j]

    ambient = wp.vec3(0.15, 0.25, 0.35)
    lit = ambient * (0.5 + 0.5 * cos_theta) * 255.0

    atten = wp.vec3(
        wp.exp(-d * atten_coeff[0]),
        wp.exp(-d * atten_coeff[1]),
        wp.exp(-d * atten_coeff[2]),
    )
    direct = wp.vec3(lit[0] * atten[0], lit[1] * atten[1], lit[2] * atten[2])

    inv = wp.vec3(1.0 - atten[0], 1.0 - atten[1], 1.0 - atten[2])
    scatter = wp.vec3(
        backscatter_value[0] * 255.0 * inv[0],
        backscatter_value[1] * 255.0 * inv[1],
        backscatter_value[2] * 255.0 * inv[2],
    )
    result = wp.vec3(direct[0] + scatter[0], direct[1] + scatter[1], direct[2] + scatter[2])

    out_rgb[i, j, 0] = wp.uint8(wp.clamp(result[0], 0.0, 255.0))
    out_rgb[i, j, 1] = wp.uint8(wp.clamp(result[1], 0.0, 255.0))
    out_rgb[i, j, 2] = wp.uint8(wp.clamp(result[2], 0.0, 255.0))


# -- Config + sensor --

@dataclass
class CameraConfig:
    """Pinhole camera with underwater light transport."""

    width: int = 640
    height: int = 480
    fov_h_deg: float = 90.0
    fov_v_deg: float = 65.0
    max_range: float = 30.0
    water_type: str = "II"


class UnderwaterCamera:
    """GPU underwater camera: pinhole ray casting + Beer-Lambert shading."""

    def __init__(
        self,
        environment_mesh: wp.Mesh,
        config: CameraConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        cfg = config or CameraConfig()
        self.cfg = cfg
        self.mesh = environment_mesh
        self.device = device

        if cfg.water_type != "custom":
            if cfg.water_type not in JERLOV_PRESETS:
                raise ValueError(
                    f"Unknown water type {cfg.water_type!r}. "
                    f"Available: {sorted(JERLOV_PRESETS.keys())}"
                )
            self._water_params = JERLOV_PRESETS[cfg.water_type]
        else:
            self._water_params = WaterParams(
                backscatter_value=(0.0, 0.31, 0.24),
                atten_coeff=(0.05, 0.05, 0.05),
                backscatter_coeff=(0.05, 0.05, 0.2),
            )

        self._backscatter = wp.vec3(*self._water_params.backscatter_value)
        self._atten = wp.vec3(*self._water_params.atten_coeff)
        self._back_coeff = wp.vec3(*self._water_params.backscatter_coeff)

        fov_h = math.radians(cfg.fov_h_deg)
        fov_v = math.radians(cfg.fov_v_deg)

        u = np.linspace(-math.tan(fov_h / 2), math.tan(fov_h / 2), cfg.width, dtype=np.float32)
        v = np.linspace(-math.tan(fov_v / 2), math.tan(fov_v / 2), cfg.height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        dirs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs = (dirs / norms).reshape(-1, 3).astype(np.float32)

        self._body_dirs = dirs
        self._wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=device)
        self._depth = wp.zeros((cfg.height, cfg.width), dtype=wp.float32, device=device)
        self._intensity = wp.zeros((cfg.height, cfg.width), dtype=wp.float32, device=device)
        self._rgb = wp.zeros((cfg.height, cfg.width, 3), dtype=wp.uint8, device=device)

    def scan(
        self,
        position: np.ndarray,
        orientation: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Render one frame. Returns ``rgb`` (H,W,3) uint8 + ``depth`` (H,W) float32."""
        origin = wp.vec3(float(position[0]), float(position[1]), float(position[2]))

        if orientation is not None:
            dirs = self._rotate_rays(orientation)
            wp_dirs = wp.array(dirs, dtype=wp.vec3f, device=self.device)
        else:
            wp_dirs = self._wp_dirs

        n_pixels = self.cfg.width * self.cfg.height
        wp.launch(
            _camera_raycast,
            dim=n_pixels,
            inputs=[self.mesh.id, origin, wp_dirs, self.cfg.max_range,
                    self._depth, self._intensity, self.cfg.width],
            device=self.device,
        )

        wp.launch(
            _uw_shade,
            dim=(self.cfg.height, self.cfg.width),
            inputs=[self._depth, self._intensity,
                    self._backscatter, self._atten, self._back_coeff, self._rgb],
            device=self.device,
        )

        wp.synchronize_device(self.device)
        return {
            "rgb": self._rgb.numpy().copy(),
            "depth": self._depth.numpy().copy(),
        }

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
