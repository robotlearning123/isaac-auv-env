"""Tests for MarineGym BlueROV asset integration."""

import pytest


def test_bluerov_usd_exists():
    from oceanscale.vehicles.bluerov2 import _marinegym_asset_dir
    usd_path = _marinegym_asset_dir() / "BlueROV.usd"
    assert usd_path.exists(), f"BlueROV USD not found at {usd_path}"
    assert usd_path.stat().st_size > 1000


def test_bluerov_usd_valid():
    from pxr import Usd

    from oceanscale.vehicles.bluerov2 import _marinegym_asset_dir
    stage = Usd.Stage.Open(str(_marinegym_asset_dir() / "BlueROV.usd"))
    assert stage is not None
    root = stage.GetPrimAtPath("/BlueROV")
    assert root.IsValid(), "/BlueROV prim not found"
    base_link = stage.GetPrimAtPath("/BlueROV/base_link")
    assert base_link.IsValid(), "/BlueROV/base_link not found"


def test_bluerov_usd_has_rotors():
    from pxr import Usd

    from oceanscale.vehicles.bluerov2 import _marinegym_asset_dir
    stage = Usd.Stage.Open(str(_marinegym_asset_dir() / "BlueROV.usd"))
    for i in range(6):
        prim = stage.GetPrimAtPath(f"/BlueROV/rotor_{i}")
        assert prim.IsValid(), f"rotor_{i} not found"


def test_hydro_yaml_loads():
    from oceanscale.vehicles.bluerov2 import _load_marinegym_yaml
    data = _load_marinegym_yaml()
    assert data["name"] == "BlueROV"
    assert len(data["hydro_coef"]["added_mass"]) == 6
    assert len(data["hydro_coef"]["linear_damping"]) == 6
    assert len(data["hydro_coef"]["quadratic_damping"]) == 6
    assert data["rotor_configuration"]["num_rotors"] == 6


def test_bluerov2_marinegym_from_yaml():
    from oceanscale.vehicles import BlueROV2MarineGym
    v = BlueROV2MarineGym.from_yaml()
    assert v.n_thrusters == 6
    assert v.volume == pytest.approx(0.0113459)
    assert v.coBM == pytest.approx(0.01)
    assert len(v.added_mass) == 6
    assert len(v.d_lin) == 6
    assert len(v.d_quad) == 6
    assert v.added_mass[0] == pytest.approx(5.5)
    assert v.d_quad[2] == pytest.approx(36.99)


def test_bluerov2_marinegym_default():
    from oceanscale.vehicles import BlueROV2MarineGym
    v = BlueROV2MarineGym()
    assert v.n_thrusters == 6
    assert len(v.rotor_directions) == 6
    assert all(abs(d) == 1.0 for d in v.rotor_directions)


def test_bluerov2_marinegym_rotor_config():
    from oceanscale.vehicles import BlueROV2MarineGym
    v = BlueROV2MarineGym.from_yaml()
    rc = v.rotor_config()
    assert rc["num_rotors"] == 6
    assert len(rc["directions"]) == 6
    assert len(rc["force_constants"]) == 6
    assert all(fc > 0 for fc in rc["force_constants"])
    assert all(rpm > 0 for rpm in rc["max_rpm"])


def test_bluerov2_marinegym_set_coeffs():
    from oceanscale.vehicles import BlueROV2MarineGym
    v = BlueROV2MarineGym.from_yaml()
    kw = v.set_coeffs_kwargs()
    assert "added_mass" in kw
    assert "d_lin" in kw
    assert "d_quad" in kw
    assert "mass" in kw
    assert "volume" in kw
    assert all(am > 0 for am in kw["added_mass"])


def test_bluerov2_marinegym_usd_path():
    from oceanscale.vehicles import BlueROV2MarineGym
    v = BlueROV2MarineGym()
    assert v.usd_path().exists()
    assert v.usd_path().suffix == ".usd"


def test_bluerov2_heavy_still_works():
    from oceanscale.vehicles import BlueROV2Heavy
    v = BlueROV2Heavy()
    assert v.n_thrusters == 8
    assert v.mass == 13.5
    kw = v.set_coeffs_kwargs()
    assert "T_matrix" in kw
    assert len(kw["T_matrix"]) == 6
    assert len(kw["T_matrix"][0]) == 8


def test_marinegym_vs_vonbenzon_params_differ():
    """MarineGym and von Benzon have different hydro params (different experiments)."""
    from oceanscale.vehicles import BlueROV2Heavy, BlueROV2MarineGym
    heavy = BlueROV2Heavy()
    mg = BlueROV2MarineGym.from_yaml()
    assert heavy.added_mass[0] != mg.added_mass[0]
    assert heavy.n_thrusters != mg.n_thrusters
