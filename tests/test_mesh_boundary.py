"""Tests for wp.Mesh-based fluid boundary (FSI coupling)."""

import numpy as np
import pytest
import warp as wp

from oceanscale.fluid.mesh_boundary import MeshBoundary, make_box_mesh

wp.init()


class TestMeshCreation:
    def test_create_from_triangle(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        mb = MeshBoundary(verts, indices)
        assert mb.mesh is not None

    def test_create_from_box(self):
        verts, indices = make_box_mesh(center=(5, 5, 5), half_extents=(1, 1, 1))
        mb = MeshBoundary(verts, indices)
        assert mb.mesh is not None
        assert len(verts) == 8
        assert len(indices) == 36

    def test_create_with_velocities(self):
        verts, indices = make_box_mesh()
        vels = np.ones_like(verts) * 0.5
        mb = MeshBoundary(verts, indices, velocities=vels)
        assert mb.mesh is not None


class TestPointInsideDetection:
    def test_inside_box(self):
        verts, indices = make_box_mesh(center=(0, 0, 0), half_extents=(1, 1, 1))
        mb = MeshBoundary(verts, indices)

        inside_pt = wp.array(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        outside_pt = wp.array(np.array([[5.0, 5.0, 5.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")

        @wp.kernel
        def check_sign(mesh_id: wp.uint64, pts: wp.array(dtype=wp.vec3), signs: wp.array(dtype=wp.float32)):
            i = wp.tid()
            q = wp.mesh_query_point_sign_normal(mesh_id, pts[i], 100.0, 1.0e-5)
            if q.result:
                signs[i] = q.sign
            else:
                signs[i] = 99.0

        signs_in = wp.zeros(1, dtype=wp.float32, device="cuda:0")
        signs_out = wp.zeros(1, dtype=wp.float32, device="cuda:0")
        wp.launch(check_sign, dim=1, inputs=[mb.mesh.id, inside_pt, signs_in])
        wp.launch(check_sign, dim=1, inputs=[mb.mesh.id, outside_pt, signs_out])
        wp.synchronize()
        assert signs_in.numpy()[0] < 0.0
        assert signs_out.numpy()[0] > 0.0


class TestBoundaryProjection:
    def test_sph_particles_pushed_out(self):
        verts, indices = make_box_mesh(center=(0.5, 0.5, 0.5), half_extents=(0.2, 0.2, 0.2))
        mb = MeshBoundary(verts, indices)

        pos = wp.array(
            np.array([[0.5, 0.5, 0.5], [0.45, 0.5, 0.5], [2.0, 2.0, 2.0]], dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        vel = wp.array(
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            dtype=wp.vec3,
            device="cuda:0",
        )
        mb.apply_boundary_sph(pos, vel, max_dist=5.0, damping=0.5)
        wp.synchronize()

        p = pos.numpy()
        assert not np.allclose(p[0], [0.5, 0.5, 0.5]), "Inside particle should have been moved"
        assert not np.allclose(p[1], [0.45, 0.5, 0.5]), "Inside particle should have been moved"
        np.testing.assert_allclose(p[2], [2.0, 2.0, 2.0], atol=1e-5, err_msg="Outside particle should be untouched")


class TestMovingBoundary:
    def test_velocity_transfer(self):
        verts, indices = make_box_mesh(center=(0.5, 0.5, 0.5), half_extents=(0.3, 0.3, 0.3))
        wall_vel = np.zeros_like(verts)
        wall_vel[:, 1] = 2.0
        mb = MeshBoundary(verts, indices, velocities=wall_vel)

        pos = wp.array(np.array([[0.5, 0.75, 0.5]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        vel = wp.array(np.array([[0.0, -5.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        mb.apply_boundary_sph(pos, vel, max_dist=5.0, damping=0.5)
        wp.synchronize()

        v = vel.numpy()[0]
        assert v[1] > -5.0, "Normal velocity into wall should be reflected/reduced"

    def test_update_positions_refits(self):
        verts, indices = make_box_mesh(center=(0, 0, 0), half_extents=(1, 1, 1))
        mb = MeshBoundary(verts, indices)

        new_verts = verts + 10.0
        mb.update_positions(new_verts)
        np.testing.assert_allclose(mb._points.numpy(), new_verts.reshape(-1, 3), atol=1e-5)

    def test_update_velocities(self):
        verts, indices = make_box_mesh()
        mb = MeshBoundary(verts, indices)

        new_vel = np.ones_like(verts) * 3.0
        mb.update_velocities(new_vel)
        np.testing.assert_allclose(mb._velocities.numpy(), new_vel.reshape(-1, 3), atol=1e-5)


class TestGridSolverIntegration:
    def test_add_boundary_mesh(self):
        from oceanscale.fluid import create_fluid_solver, FluidLevel

        solver = create_fluid_solver(FluidLevel.GRID, grid_res=16)
        verts, indices = make_box_mesh(center=(0.5, 0.5, 0.5), half_extents=(0.1, 0.1, 0.1))
        solver.add_boundary_mesh(verts, indices)

        u = solver.u.numpy()
        n = solver.nx
        mid = n // 2
        idx = mid * n * n + mid * n + mid
        assert u[idx] == 0.0, "Velocity inside mesh boundary should be zero"

    def test_grid_boundary_preserves_outside(self):
        from oceanscale.fluid import create_fluid_solver, FluidLevel

        solver = create_fluid_solver(FluidLevel.GRID, grid_res=16)
        solver.u.fill_(1.0)
        verts, indices = make_box_mesh(center=(0.5, 0.5, 0.5), half_extents=(0.05, 0.05, 0.05))
        solver.add_boundary_mesh(verts, indices)

        u = solver.u.numpy()
        assert np.any(u == 1.0), "Outside velocity should be preserved"
        n = solver.nx
        mid = n // 2
        idx = mid * n * n + mid * n + mid
        assert u[idx] != 1.0, "Inside velocity should be modified"
