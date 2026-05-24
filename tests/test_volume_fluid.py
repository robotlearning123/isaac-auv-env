"""Tests for VolumeFluidSolver (wp.Volume NanoVDB sparse grid)."""

import numpy as np
import pytest
import warp as wp

wp = pytest.importorskip("warp", reason="warp not available")

from oceanscale.fluid import FluidLevel, create_fluid_solver
from oceanscale.fluid.volume_solver import VolumeFluidSolver


class TestVolumeCreation:
    def test_solver_creates(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        assert solver is not None
        assert solver.grid_res == 16

    def test_volumes_allocated(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        assert solver.u.get_voxel_count() > 0
        assert solver.v.get_voxel_count() > 0
        assert solver.w.get_voxel_count() > 0
        assert solver.p.get_voxel_count() > 0
        assert solver.density.get_voxel_count() > 0

    def test_factory_creates_volume(self):
        solver = create_fluid_solver(FluidLevel.VOLUME, grid_res=16)
        assert isinstance(solver, VolumeFluidSolver)

    def test_factory_by_int(self):
        solver = create_fluid_solver(4, grid_res=16)
        assert isinstance(solver, VolumeFluidSolver)

    def test_has_step(self):
        solver = VolumeFluidSolver(grid_res=16)
        assert hasattr(solver, "step")

    def test_has_sample_velocity(self):
        solver = VolumeFluidSolver(grid_res=16)
        assert hasattr(solver, "sample_velocity_at")


class TestVolumeStep:
    def test_10_steps_no_nan(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        for _ in range(10):
            solver.step(dt=0.01)
        wp.synchronize()

    def test_step_with_different_dt(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        solver.step(dt=0.001)
        solver.step(dt=0.05)
        wp.synchronize()


class TestVolumeSampleVelocity:
    def test_sample_returns_correct_shape(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        positions = wp.array(
            [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
            dtype=wp.vec3,
            device="cuda:0",
        )
        vel = solver.sample_velocity_at(positions)
        assert vel.shape[0] == 2

    def test_sample_zero_field(self):
        solver = VolumeFluidSolver(grid_res=16, domain_size=1.0, device="cuda:0")
        positions = wp.array(
            [[0.5, 0.5, 0.5]],
            dtype=wp.vec3,
            device="cuda:0",
        )
        vel = solver.sample_velocity_at(positions)
        v = vel.numpy()[0]
        np.testing.assert_allclose(v, [0.0, 0.0, 0.0], atol=1e-6)


class TestVolumeVsGrid:
    def test_zero_field_consistency(self):
        from oceanscale.fluid.grid import GridFluidSolver

        grid = GridFluidSolver(grid_res=16, domain_size=1.0, device="cuda:0")
        vol = VolumeFluidSolver(grid_res=16, domain_size=1.0, device="cuda:0")

        grid.step(dt=0.01)
        vol.step(dt=0.01)

        pos = wp.array([[0.5, 0.5, 0.5]], dtype=wp.vec3, device="cuda:0")
        grid_vel = grid.sample_velocity_at(pos).numpy()[0]
        vol_vel = vol.sample_velocity_at(pos).numpy()[0]

        np.testing.assert_allclose(grid_vel, vol_vel, atol=1e-4)


class TestVolumeMemory:
    def test_voxel_count_matches_grid(self):
        solver = VolumeFluidSolver(grid_res=16, device="cuda:0")
        voxels = solver.get_voxel_count()
        assert voxels > 0
        assert voxels <= 16 * 16 * 16

    def test_voxel_count_scales(self):
        s16 = VolumeFluidSolver(grid_res=16, device="cuda:0")
        s32 = VolumeFluidSolver(grid_res=32, device="cuda:0")
        assert s32.get_voxel_count() > s16.get_voxel_count()


class TestFluidLevelEnum:
    def test_volume_level_value(self):
        assert FluidLevel.VOLUME == 4

    def test_all_levels_defined(self):
        assert len(FluidLevel) == 5
