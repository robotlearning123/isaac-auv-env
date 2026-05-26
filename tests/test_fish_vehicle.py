"""Tests for TendonFish biomimetic vehicle and fishsim-validated coefficients."""

import pytest

from oceanscale.hydro.mujoco_drag import MuJoCoDragParams
from oceanscale.vehicles.fish import TendonFish


class TestTendonFish:
    def test_default_construct(self):
        fish = TendonFish()
        assert fish.name == "TendonFish"
        assert fish.mass == 1.5

    def test_fluid_coefficients(self):
        fish = TendonFish()
        assert fish.blunt_drag == pytest.approx(0.4)
        assert fish.slender_drag == pytest.approx(7.79)
        assert fish.angular_drag == pytest.approx(2.81)
        assert fish.kutta_lift == pytest.approx(3.84)
        assert fish.magnus_lift == pytest.approx(0.27)

    def test_mujoco_drag_params(self):
        fish = TendonFish()
        params = fish.mujoco_drag_params()
        assert isinstance(params, MuJoCoDragParams)
        assert params.kutta_lift == pytest.approx(3.84)
        assert params.magnus_lift == pytest.approx(0.27)

    def test_tendon_routing(self):
        fish = TendonFish()
        assert len(fish.tendon_routing) == 5
        assert all(r in (0, 1) for r in fish.tendon_routing)

    def test_center_of_mass(self):
        fish = TendonFish()
        com = fish.center_of_mass
        assert len(com) == 3
        assert com[1] == pytest.approx(0.0, abs=1e-6)

    def test_fishsim_validated_preset(self):
        p = MuJoCoDragParams.FISHSIM_VALIDATED
        assert p is not None
        assert p.fluid_density == pytest.approx(1000.0)
        assert p.kutta_lift == pytest.approx(3.84)
        assert p.magnus_lift == pytest.approx(0.27)
