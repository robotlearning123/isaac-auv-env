"""Ocean surface extraction using wp.MarchingCubes."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import warp as wp

cast(Any, wp.init)()


@wp.kernel
def _particles_to_sdf(
    positions: wp.array(dtype=wp.vec3),
    radius: float,
    origin: wp.vec3,
    inv_dx: float,
    nx: int,
    ny: int,
    nz: int,
    field: wp.array(dtype=float, ndim=3),
):
    p = wp.tid()
    pos = positions[p]
    ci = int((pos[0] - wp.extract(origin, 0)) * inv_dx)
    cj = int((pos[1] - wp.extract(origin, 1)) * inv_dx)
    ck = int((pos[2] - wp.extract(origin, 2)) * inv_dx)
    r = int(wp.ceil(radius * inv_dx)) + 1
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                gi = ci + di
                gj = cj + dj
                gk = ck + dk
                if gi < 0 or gi >= nx or gj < 0 or gj >= ny or gk < 0 or gk >= nz:
                    continue
                gx = wp.extract(origin, 0) + float(gi) / inv_dx
                gy = wp.extract(origin, 1) + float(gj) / inv_dx
                gz = wp.extract(origin, 2) + float(gk) / inv_dx
                dist = wp.length(wp.vec3(gx, gy, gz) - pos) - radius
                wp.atomic_min(field, gi, gj, gk, dist)


class SurfaceExtractor:
    """Extract fluid surface mesh from scalar field using wp.MarchingCubes."""

    def __init__(self, nx: int = 64, ny: int = 64, nz: int = 64, device: str = "cuda:0") -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.device = device
        self.mc = wp.MarchingCubes(nx=nx, ny=ny, nz=nz, device=device)

    def extract(self, field: wp.array, threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        self.mc.surface(field, threshold)
        return self.mc.verts.numpy(), self.mc.indices.numpy()

    def extract_from_particles(
        self,
        positions: wp.array,
        radius: float = 0.05,
        domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        dx = (domain_max[0] - domain_min[0]) / self.nx
        field = wp.full(
            shape=(self.nx, self.ny, self.nz),
            value=radius * 2.0,
            dtype=wp.float32,
            device=self.device,
        )
        wp.launch(
            _particles_to_sdf,
            dim=positions.shape[0],
            inputs=[
                positions,
                radius,
                wp.vec3(*domain_min),
                1.0 / dx,
                self.nx,
                self.ny,
                self.nz,
                field,
            ],
            device=self.device,
        )
        verts, indices = self.extract(field, threshold=0.0)
        if verts.size > 0:
            scale = np.array([
                (domain_max[0] - domain_min[0]) / self.nx,
                (domain_max[1] - domain_min[1]) / self.ny,
                (domain_max[2] - domain_min[2]) / self.nz,
            ])
            verts = verts * scale + np.array(domain_min)
        return verts, indices
