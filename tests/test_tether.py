"""Tests for ROV tether simulation using Newton add_rod()."""

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton", reason="newton not installed")
import newton.solvers  # noqa: E402

from oceanscale.tether import Tether  # noqa: E402


@pytest.fixture
def vbd_sim():
    """Build a model with a vertical tether and VBD solver."""
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()

    tether = Tether(
        builder,
        anchor_pos=(0.0, 0.0, 10.0),
        attach_pos=(0.0, 0.0, 5.0),
        n_segments=10,
        radius=0.01,
        stretch_stiffness=1e5,
        stretch_damping=100.0,
        bend_stiffness=100.0,
        bend_damping=10.0,
        density=1100.0,
    )

    builder.color()
    model = builder.finalize(device="cuda:0")
    solver = newton.solvers.SolverVBD(model, iterations=5)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = model.contacts(collision_pipeline=pipeline)

    return {
        "tether": tether,
        "model": model,
        "solver": solver,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts": contacts,
    }


def _step(sim, n_frames: int = 50, substeps: int = 10) -> None:
    dt = (1.0 / 60.0) / substeps
    for _ in range(n_frames):
        for _ in range(substeps):
            sim["state_0"].clear_forces()
            sim["model"].collide(sim["state_0"], sim["contacts"])
            sim["solver"].step(
                sim["state_0"],
                sim["state_1"],
                sim["control"],
                sim["contacts"],
                dt,
            )
            sim["state_0"], sim["state_1"] = sim["state_1"], sim["state_0"]


@pytest.mark.gpu
class TestTetherCreation:
    def test_body_count(self, vbd_sim):
        assert len(vbd_sim["tether"].body_ids) == 10

    def test_joint_count(self, vbd_sim):
        assert len(vbd_sim["tether"].joint_ids) == 9

    def test_length(self, vbd_sim):
        assert abs(vbd_sim["tether"].length - 5.0) < 1e-6

    def test_n_segments(self, vbd_sim):
        assert vbd_sim["tether"].n_segments == 10


@pytest.mark.gpu
class TestTetherPositions:
    def test_shape(self, vbd_sim):
        pos = vbd_sim["tether"].get_positions(vbd_sim["state_0"])
        assert pos.shape == (10, 3)

    def test_initial_vertical(self, vbd_sim):
        pos = vbd_sim["tether"].get_positions(vbd_sim["state_0"])
        assert pos[0, 2] > pos[-1, 2]

    def test_finite(self, vbd_sim):
        pos = vbd_sim["tether"].get_positions(vbd_sim["state_0"])
        assert np.isfinite(pos).all()


@pytest.mark.gpu
class TestTetherGravitySag:
    def test_cable_sags_under_gravity(self, vbd_sim):
        _step(vbd_sim, n_frames=100)
        pos = vbd_sim["tether"].get_positions(vbd_sim["state_0"])
        assert np.isfinite(pos).all()
        z_max = pos[:, 2].max()
        z_min = pos[:, 2].min()
        assert z_max > z_min

    def test_end_to_end_distance(self, vbd_sim):
        d0 = vbd_sim["tether"].get_end_to_end_distance(vbd_sim["state_0"])
        assert d0 > 0.0
        _step(vbd_sim, n_frames=100)
        d1 = vbd_sim["tether"].get_end_to_end_distance(vbd_sim["state_0"])
        assert d1 > 0.0

    def test_tension_estimate(self, vbd_sim):
        t = vbd_sim["tether"].get_tension_estimate(vbd_sim["state_0"])
        assert t >= 0.0


@pytest.mark.gpu
class TestTetherWithROV:
    def test_rov_plus_tether_model(self):
        builder = newton.ModelBuilder(gravity=0.0)

        builder.begin_world(label="env_0")
        rov_body = builder.add_body(mass=13.5, com=(0, 0, 0))
        builder.add_shape_box(body=rov_body, hx=0.25, hy=0.15, hz=0.15)
        builder.end_world()

        tether = Tether(
            builder,
            anchor_pos=(0.0, 0.0, 0.0),
            attach_pos=(0.0, 0.0, -5.0),
            n_segments=5,
        )

        builder.color()
        model = builder.finalize(device="cuda:0")

        assert model.body_count == 1 + 5
        assert len(tether.body_ids) == 5
        assert len(tether.joint_ids) == 4

        state = model.state()
        pos = tether.get_positions(state)
        assert pos.shape == (5, 3)
        assert np.isfinite(pos).all()
