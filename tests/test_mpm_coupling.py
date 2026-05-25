"""Tests for MPM two-way coupling between fluid and rigid bodies."""

from __future__ import annotations

import newton
import warp as wp
from newton.solvers import SolverImplicitMPM

from oceanscale.fluid.mpm_coupling import MPMRigidCoupling

DEVICE = "cuda:0"
VOXEL_SIZE = 0.1


def _build_scene():
    """Build a minimal rigid + fluid scene for coupling tests."""
    rigid_builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    rigid_builder.default_shape_cfg.mu = 0.5

    body_idx = rigid_builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.5), q=wp.quat_identity()),
        mass=50.0,
    )
    rigid_builder.add_shape_box(body=body_idx, hx=0.3, hy=0.3, hz=0.1)
    rigid_builder.add_ground_plane()
    rigid_model = rigid_builder.finalize(device=DEVICE)

    fluid_builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    SolverImplicitMPM.register_custom_attributes(fluid_builder)

    cell_size = VOXEL_SIZE / 3.0
    mass = (cell_size**3) * 1000.0
    radius = cell_size * 0.5
    fluid_builder.add_particle_grid(
        pos=wp.vec3(-0.3, -0.3, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=6, dim_y=6, dim_z=15,
        cell_x=cell_size, cell_y=cell_size, cell_z=cell_size,
        mass=mass, jitter=radius, radius_mean=radius,
    )
    fluid_model = fluid_builder.finalize(device=DEVICE)

    mpm_cfg = SolverImplicitMPM.Config()
    mpm_cfg.voxel_size = VOXEL_SIZE
    mpm_cfg.tolerance = 1e-3
    mpm_cfg.max_iterations = 30
    mpm_cfg.strain_basis = "P0"
    mpm_cfg.velocity_basis = "Q1"
    mpm_cfg.transfer_scheme = "apic"

    mpm_solver = SolverImplicitMPM(fluid_model, mpm_cfg)
    mpm_solver.setup_collider(model=rigid_model)

    rigid_solver = newton.solvers.SolverSemiImplicit(rigid_model)

    return rigid_model, fluid_model, mpm_solver, rigid_solver, body_idx


class TestMPMCouplingCreation:
    def test_coupling_creates(self):
        rm, fm, ms, rs, _ = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)
        assert coupling is not None

    def test_collider_body_id_exists(self):
        rm, fm, ms, rs, _ = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)
        assert coupling.collider_body_id is not None

    def test_impulse_buffers_allocated(self):
        rm, fm, ms, rs, _ = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)
        assert coupling._impulses.shape[0] > 0
        assert coupling._impulse_pos.shape[0] > 0


class TestMPMCouplingStep:
    def test_step_no_crash(self):
        rm, fm, ms, rs, _ = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)

        rs0 = rm.state()
        rs1 = rm.state()
        fs = fm.state()
        fs.body_q = wp.empty_like(rs0.body_q)
        fs.body_qd = wp.empty_like(rs0.body_qd)

        newton.eval_fk(rm, rm.joint_q, rm.joint_qd, rs0)

        dt = 1.0 / 60.0
        coupling.step(rs0, rs1, fs, dt)

    def test_multi_step_stable(self):
        rm, fm, ms, rs, _ = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)

        rs0 = rm.state()
        rs1 = rm.state()
        fs = fm.state()
        fs.body_q = wp.empty_like(rs0.body_q)
        fs.body_qd = wp.empty_like(rs0.body_qd)

        newton.eval_fk(rm, rm.joint_q, rm.joint_qd, rs0)

        dt = 1.0 / 120.0
        for _ in range(10):
            coupling.step(rs0, rs1, fs, dt)
            rs0, rs1 = rs1, rs0


class TestMPMCouplingForces:
    def test_get_forces_shape(self):
        rm, fm, ms, rs, body_idx = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)
        f = coupling.get_fluid_forces_on_body(body_idx)
        assert f.shape == (6,)

    def test_rigid_body_falls(self):
        rm, fm, ms, rs, body_idx = _build_scene()
        coupling = MPMRigidCoupling(rm, fm, ms, rs, device=DEVICE)

        rs0 = rm.state()
        rs1 = rm.state()
        fs = fm.state()
        fs.body_q = wp.empty_like(rs0.body_q)
        fs.body_qd = wp.empty_like(rs0.body_qd)

        newton.eval_fk(rm, rm.joint_q, rm.joint_qd, rs0)
        initial_z = rs0.body_q.numpy()[body_idx][2]

        dt = 1.0 / 60.0
        for _ in range(30):
            coupling.step(rs0, rs1, fs, dt)
            rs0, rs1 = rs1, rs0

        final_z = rs0.body_q.numpy()[body_idx][2]
        assert final_z < initial_z, "Rigid body should fall under gravity"
