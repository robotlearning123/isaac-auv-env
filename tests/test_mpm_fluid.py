"""Tests for NewtonMPMFluid."""

import numpy as np
import pytest
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

from oceanscale.fluid.mpm import NewtonMPMFluid

wp.init()


class TestNewtonMPMFluid:
    def test_mpm_creation(self):
        fluid = NewtonMPMFluid(n_particles=1000, voxel_size=0.1)
        assert fluid.particle_count > 0
        positions = fluid.get_particle_positions().numpy()
        assert positions.shape == (fluid.particle_count, 3)
        assert np.all(np.isfinite(positions))

    def test_mpm_step(self):
        fluid = NewtonMPMFluid(n_particles=1000, voxel_size=0.1)
        pos_before = fluid.get_particle_positions().numpy().copy()
        for _ in range(10):
            fluid.step(dt=1.0 / 240.0)
        wp.synchronize_device()
        pos_after = fluid.get_particle_positions().numpy()
        assert not np.allclose(pos_before, pos_after, atol=1e-6)

    def test_mpm_gravity(self):
        fluid = NewtonMPMFluid(n_particles=1000, voxel_size=0.1)
        mean_z_before = np.mean(fluid.get_particle_positions().numpy()[:, 2])
        for _ in range(60):
            fluid.step(dt=1.0 / 240.0)
        wp.synchronize_device()
        mean_z_after = np.mean(fluid.get_particle_positions().numpy()[:, 2])
        assert mean_z_after < mean_z_before

    def test_dam_break(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        SolverImplicitMPM.register_custom_attributes(builder)

        NewtonMPMFluid.create_dam_break(
            builder,
            domain=(0, 0, 0, 2, 2, 2),
            fluid_region=(0, 0, 0, 0.5, 0.5, 1.0),
            voxel_size=0.1,
            n_particles=1000,
        )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        model = builder.finalize()
        model.set_gravity([0.0, 0.0, -9.81])
        model.mpm.viscosity.fill_(50.0)
        model.mpm.friction.fill_(0.0)
        model.mpm.tensile_yield_ratio.fill_(1.0)
        model.mpm.young_modulus.fill_(1.0e6)
        model.mpm.yield_pressure.fill_(1.0e4)

        config = SolverImplicitMPM.Config()
        config.voxel_size = 0.1
        config.tolerance = 1.0e-4
        config.max_iterations = 100
        config.strain_basis = "P0"
        config.velocity_basis = "Q1"
        config.transfer_scheme = "apic"

        solver = SolverImplicitMPM(model, config)
        state_0 = model.state()
        state_1 = model.state()

        pos_x_before = state_0.particle_q.numpy()[:, 0].copy()
        for _ in range(100):
            solver.step(state_0, state_1, None, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        wp.synchronize_device()

        pos_x_after = state_0.particle_q.numpy()[:, 0]
        x_spread_after = np.std(pos_x_after)
        x_spread_before = np.std(pos_x_before)
        assert x_spread_after > x_spread_before

    def test_drop_splash(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        SolverImplicitMPM.register_custom_attributes(builder)

        NewtonMPMFluid.create_drop_splash(
            builder,
            drop_center=(1.0, 1.0, 2.0),
            drop_radius=0.3,
            voxel_size=0.1,
            n_particles=500,
        )

        assert builder.particle_count > 0
