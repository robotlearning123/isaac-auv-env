"""Tests for commercial AUV/ROV vehicle configurations."""

import pytest

from oceanscale.vehicles import (
    COMMERCIAL_VEHICLE_REGISTRY,
    HUGIN1000,
    LAUV,
    REMUS100,
    Girona500,
    SparusII,
)

VEHICLES = [REMUS100, Girona500, LAUV, HUGIN1000, SparusII]


@pytest.mark.parametrize("cls", VEHICLES, ids=lambda c: c.__name__)
def test_valid_mass(cls):
    v = cls()
    assert v.mass > 0


@pytest.mark.parametrize("cls", VEHICLES, ids=lambda c: c.__name__)
def test_valid_dimensions(cls):
    v = cls()
    assert v.length > 0
    d = getattr(v, "diameter", None) or getattr(v, "hull_diameter", None)
    assert d is not None and d > 0


@pytest.mark.parametrize("cls", VEHICLES, ids=lambda c: c.__name__)
def test_has_name(cls):
    v = cls()
    assert isinstance(v.name, str) and len(v.name) > 0


def test_registry_has_all():
    assert len(COMMERCIAL_VEHICLE_REGISTRY) == 5
    for key in ["remus100", "girona500", "lauv", "hugin1000", "sparus2"]:
        assert key in COMMERCIAL_VEHICLE_REGISTRY


class TestREMUS100:
    def test_prestero_weight_buoyancy(self):
        r = REMUS100()
        assert r.weight == pytest.approx(299.0, abs=1)
        assert r.buoyancy == pytest.approx(306.0, abs=1)
        assert r.buoyancy > r.weight  # positive buoyancy

    def test_prestero_dimensions(self):
        r = REMUS100()
        assert r.length == pytest.approx(1.33, abs=0.01)
        assert r.diameter == pytest.approx(0.191, abs=0.001)

    def test_added_mass_positive(self):
        r = REMUS100()
        am = r.added_mass_diag()
        assert all(a > 0 for a in am)

    def test_added_mass_values(self):
        r = REMUS100()
        assert abs(r.Xu_dot) == pytest.approx(0.93, abs=0.01)
        assert abs(r.Yv_dot) == pytest.approx(35.5, abs=0.5)
        assert abs(r.Zw_dot) == pytest.approx(35.5, abs=0.5)

    def test_quadratic_damping_negative(self):
        r = REMUS100()
        qd = r.quadratic_damping()
        assert all(d < 0 for d in qd)

    def test_inertia_symmetric(self):
        r = REMUS100()
        assert r.Iyy == pytest.approx(r.Izz, abs=0.01)  # axisymmetric

    def test_propeller_thrust(self):
        r = REMUS100()
        assert r.propeller_thrust == pytest.approx(3.86, abs=0.1)

    def test_n_fins(self):
        r = REMUS100()
        assert r.n_fins == 4


class TestLAUV:
    def test_scaled_from_remus(self):
        lauv = LAUV()
        remus = REMUS100()
        scale = (lauv.diameter / remus.diameter) ** 2
        assert lauv.added_mass[0] == pytest.approx(abs(remus.Xu_dot) * scale, rel=0.1)
