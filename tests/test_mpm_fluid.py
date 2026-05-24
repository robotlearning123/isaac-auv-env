"""Tests for NewtonMPMFluid."""

import newton
import numpy as np
import warp as wp
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
        dt = 1.0 / 240.0
        n_steps = 10
        fluid = NewtonMPMFluid(n_particles=1000, voxel_size=0.1)
        pos_before = fluid.get_particle_positions().numpy().copy()
        for _ in range(n_steps):
            fluid.step(dt=dt)
        wp.synchronize_device()
        pos_after = fluid.get_particle_positions().numpy()

        mean_dz = float(np.mean(pos_after[:, 2] - pos_before[:, 2]))
        t_total = n_steps * dt
        expected_dz = 0.5 * (-9.81) * t_total**2

        assert mean_dz < 0.0
        assert abs(mean_dz) > abs(0.2 * expected_dz), (
            f"Displacement {mean_dz:.6f} too small vs free-fall {expected_dz:.6f}"
        )
        assert abs(mean_dz) < abs(5.0 * expected_dz), (
            f"Displacement {mean_dz:.6f} too large vs free-fall {expected_dz:.6f}"
        )

    def test_mpm_gravity(self):
        dt = 1.0 / 240.0
        n_steps = 20
        fluid = NewtonMPMFluid(n_particles=1000, voxel_size=0.1)
        for _ in range(n_steps):
            fluid.step(dt=dt)
        wp.synchronize_device()

        vel = fluid.get_particle_velocities().numpy()
        mean_vz = float(np.mean(vel[:, 2]))

        t_total = n_steps * dt
        expected_vz = -9.81 * t_total

        assert mean_vz < 0.0, f"Mean vz={mean_vz:.4f} should be negative"
        assert abs(mean_vz) > abs(0.2 * expected_vz), (
            f"|vz|={abs(mean_vz):.4f} too small vs free-fall {abs(expected_vz):.4f}"
        )
        assert abs(mean_vz) < abs(5.0 * expected_vz), (
            f"|vz|={abs(mean_vz):.4f} too large vs free-fall {abs(expected_vz):.4f}"
        )

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

        pos_before = state_0.particle_q.numpy().copy()
        for _ in range(20):
            solver.step(state_0, state_1, None, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        wp.synchronize_device()

        pos_after = state_0.particle_q.numpy()
        assert not np.allclose(pos_before, pos_after, atol=1e-6)
        assert np.mean(pos_after[:, 2]) < np.mean(pos_before[:, 2])

    def test_dam_break_spread(self):
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

        dt = 1.0 / 240.0
        check_at = {25, 50, 100}
        spreads: list[float] = []
        pos_x_initial = state_0.particle_q.numpy()[:, 0].copy()
        spreads.append(float(np.std(pos_x_initial)))

        for i in range(1, 101):
            solver.step(state_0, state_1, None, None, dt)
            state_0, state_1 = state_1, state_0
            if i in check_at:
                wp.synchronize_device()
                spreads.append(float(np.std(state_0.particle_q.numpy()[:, 0])))

        assert spreads[-1] > spreads[0], (
            f"Final spread {spreads[-1]:.4f} should exceed initial {spreads[0]:.4f}"
        )
        for i in range(1, len(spreads)):
            assert spreads[i] >= spreads[i - 1] - 1e-6, (
                f"Spread not monotonically increasing: {spreads}"
            )
