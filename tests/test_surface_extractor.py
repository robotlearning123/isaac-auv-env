"""Tests for MarchingCubes surface extraction."""

from __future__ import annotations

import numpy as np
import warp as wp

from oceanscale.fluid.surface_extractor import SurfaceExtractor

DEVICE = "cuda:0"


class TestSurfaceExtractorCreation:
    def test_creates(self):
        se = SurfaceExtractor(nx=16, ny=16, nz=16, device=DEVICE)
        assert se.mc is not None

    def test_dimensions(self):
        se = SurfaceExtractor(nx=32, ny=16, nz=8, device=DEVICE)
        assert se.nx == 32
        assert se.ny == 16
        assert se.nz == 8


class TestSphereSDF:
    def test_sphere_produces_vertices(self):
        nx, ny, nz = 32, 32, 32
        field = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = (i - nx / 2) / (nx / 2)
                    y = (j - ny / 2) / (ny / 2)
                    z = (k - nz / 2) / (nz / 2)
                    field[i, j, k] = x * x + y * y + z * z - 0.5 * 0.5

        field_wp = wp.array(field, dtype=wp.float32, device=DEVICE)
        se = SurfaceExtractor(nx=nx, ny=ny, nz=nz, device=DEVICE)
        verts, indices = se.extract(field_wp, threshold=0.0)

        assert verts.shape[0] > 0, "Should produce vertices for sphere SDF"
        assert indices.shape[0] > 0, "Should produce triangles for sphere SDF"
        assert indices.shape[0] % 3 == 0, "Index count should be multiple of 3"

    def test_sphere_vertex_count(self):
        nx, ny, nz = 32, 32, 32
        field = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = (i - nx / 2) / (nx / 2)
                    y = (j - ny / 2) / (ny / 2)
                    z = (k - nz / 2) / (nz / 2)
                    field[i, j, k] = x * x + y * y + z * z - 0.5 * 0.5

        field_wp = wp.array(field, dtype=wp.float32, device=DEVICE)
        se = SurfaceExtractor(nx=nx, ny=ny, nz=nz, device=DEVICE)
        verts, _indices = se.extract(field_wp, threshold=0.0)
        assert verts.shape[0] > 50, "Sphere at 32³ should produce many vertices"


class TestEmptyField:
    def test_all_positive_no_surface(self):
        field = wp.full(
            shape=(16, 16, 16), value=1.0, dtype=wp.float32, device=DEVICE
        )
        se = SurfaceExtractor(nx=16, ny=16, nz=16, device=DEVICE)
        verts, _indices = se.extract(field, threshold=0.0)
        assert verts.shape[0] == 0

    def test_all_negative_no_surface(self):
        field = wp.full(
            shape=(16, 16, 16), value=-1.0, dtype=wp.float32, device=DEVICE
        )
        se = SurfaceExtractor(nx=16, ny=16, nz=16, device=DEVICE)
        verts, _indices = se.extract(field, threshold=0.0)
        assert verts.shape[0] == 0


class TestParticleToSurface:
    def test_particles_produce_surface(self):
        n = 500
        rng = np.random.default_rng(42)
        pts = rng.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
        positions = wp.array(pts, dtype=wp.vec3, device=DEVICE)

        se = SurfaceExtractor(nx=32, ny=32, nz=32, device=DEVICE)
        verts, _indices = se.extract_from_particles(
            positions, radius=0.05,
            domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        )
        assert verts.shape[0] > 0, "Particles should produce surface mesh"

    def test_no_particles_no_surface(self):
        positions = wp.zeros(0, dtype=wp.vec3, device=DEVICE)
        se = SurfaceExtractor(nx=16, ny=16, nz=16, device=DEVICE)
        verts, _indices = se.extract_from_particles(
            positions, radius=0.05,
            domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        )
        assert verts.shape[0] == 0
