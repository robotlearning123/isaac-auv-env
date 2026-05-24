"""Tests for Newton cloth simulation (UnderwaterCloth)."""

import newton
import numpy as np
import pytest
import warp as wp

from oceanscale.cloth import UnderwaterCloth


@pytest.fixture
def cloth_model_fixed():
    """Create a cloth with fix_top=True, finalized with SolverVBD."""
    wp.init()
    builder = newton.ModelBuilder()
    cloth = UnderwaterCloth(builder, dim_x=5, dim_y=5, fix_top=True)
    builder.color()
    model = builder.finalize(device="cuda:0")
    return cloth, model


@pytest.fixture
def cloth_model_free():
    """Create a cloth with fix_top=False."""
    wp.init()
    builder = newton.ModelBuilder()
    cloth = UnderwaterCloth(builder, dim_x=5, dim_y=5, fix_top=False)
    builder.color()
    model = builder.finalize(device="cuda:0")
    return cloth, model


class TestClothCreation:
    def test_particle_count(self, cloth_model_fixed):
        cloth, _ = cloth_model_fixed
        assert cloth.particle_count == (5 + 1) * (5 + 1)

    def test_particle_indices(self, cloth_model_fixed):
        cloth, _ = cloth_model_fixed
        assert len(cloth.particle_indices) == cloth.particle_count
        assert cloth.particle_indices[0] == 0

    def test_positions_shape(self, cloth_model_fixed):
        cloth, model = cloth_model_fixed
        state = model.state()
        pos = cloth.get_positions(state)
        assert pos.shape == (36, 3)

    def test_positions_finite(self, cloth_model_fixed):
        cloth, model = cloth_model_fixed
        state = model.state()
        pos = cloth.get_positions(state)
        assert np.all(np.isfinite(pos))


class TestClothVBD:
    def test_vbd_step_no_crash(self, cloth_model_fixed):
        cloth, model = cloth_model_fixed
        solver = newton.solvers.SolverVBD(model)
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        model.collide(s0)
        contacts = model.contacts()
        solver.step(s0, s1, control, contacts, dt=1.0 / 60.0)
        pos = cloth.get_positions(s1)
        assert np.all(np.isfinite(pos))

    def test_gravity_drape_free(self, cloth_model_free):
        cloth, model = cloth_model_free
        solver = newton.solvers.SolverVBD(model)
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        init_pos = cloth.get_positions(s0)
        init_z_mean = init_pos[:, 2].mean()

        for _ in range(60):
            model.collide(s0)
            contacts = model.contacts()
            solver.step(s0, s1, control, contacts, dt=1.0 / 60.0)
            s0, s1 = s1, s0

        final_pos = cloth.get_positions(s0)
        final_z_mean = final_pos[:, 2].mean()
        assert final_z_mean < init_z_mean

    def test_fixed_top_holds(self, cloth_model_fixed):
        cloth, model = cloth_model_fixed
        solver = newton.solvers.SolverVBD(model)
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        init_pos = cloth.get_positions(s0)
        # fix_top fixes the last row: indices [dim_y*(dim_x+1) : end]
        fixed_start = 5 * (5 + 1)  # dim_y * (dim_x + 1) = 30
        top_row_init = init_pos[fixed_start:, 2]

        for _ in range(60):
            model.collide(s0)
            contacts = model.contacts()
            solver.step(s0, s1, control, contacts, dt=1.0 / 60.0)
            s0, s1 = s1, s0

        final_pos = cloth.get_positions(s0)
        top_row_final = final_pos[fixed_start:, 2]
        np.testing.assert_allclose(top_row_final, top_row_init, atol=0.01)


class TestClothDeformation:
    def test_deformation_initial_zero(self, cloth_model_fixed):
        cloth, model = cloth_model_fixed
        state = model.state()
        d = cloth.get_deformation(state)
        assert d == 0.0

    def test_deformation_after_sim(self, cloth_model_free):
        cloth, model = cloth_model_free
        solver = newton.solvers.SolverVBD(model)
        s0 = model.state()
        s1 = model.state()
        control = model.control()
        cloth.get_deformation(s0)  # set rest shape

        for _ in range(60):
            model.collide(s0)
            contacts = model.contacts()
            solver.step(s0, s1, control, contacts, dt=1.0 / 60.0)
            s0, s1 = s1, s0

        d = cloth.get_deformation(s0)
        assert d > 0.0
