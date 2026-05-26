"""Tests for imported robot fleet: RexROV, gliders, environments, sensors."""

import numpy as np
import pytest

from oceanscale.vehicles import (
    VEHICLE_REGISTRY,
    BlueROV2Heavy,
    BlueROV2MarineGym,
    RexROV,
    SlocumGlider,
    WarpAUV,
    WaveGlider,
    WHOIHybridGlider,
)


class TestRexROV:
    def test_params(self):
        v = RexROV()
        assert v.mass == pytest.approx(1862.87)
        assert v.n_thrusters == 8
        assert len(v.added_mass) == 6

    def test_mesh_exists(self):
        assert RexROV.mesh_path().exists()
        assert RexROV.mesh_path().suffix == ".dae"

    def test_hydro_yaml(self):
        assert RexROV.hydro_yaml_path().exists()

    def test_tam(self):
        v = RexROV()
        tam = v.thruster_allocation_matrix()
        assert tam.shape == (6, 8)
        assert np.all(np.isfinite(tam))

    def test_controller_params(self):
        v = RexROV()
        p = v.controller_params("pid_traj")
        assert isinstance(p, dict)

    def test_sliding_mode_controller(self):
        v = RexROV()
        p = v.controller_params("sliding_mode")
        assert "lambda" in p or "k" in p


class TestGliders:
    def test_slocum_params(self):
        v = SlocumGlider()
        assert v.mass == pytest.approx(69.25)
        assert len(v.added_mass) == 6

    def test_slocum_mesh(self):
        assert SlocumGlider.mesh_path().exists()

    def test_slocum_sdf(self):
        assert SlocumGlider.sdf_path().exists()

    def test_wave_glider_mesh(self):
        assert WaveGlider.mesh_path().exists()

    def test_whoi_hybrid_sdf(self):
        assert WHOIHybridGlider.sdf_path().exists()


class TestVehicleRegistry:
    def test_all_registered(self):
        assert "rexrov" in VEHICLE_REGISTRY
        assert "slocum" in VEHICLE_REGISTRY
        assert "wave_glider" in VEHICLE_REGISTRY
        assert "whoi_hybrid" in VEHICLE_REGISTRY

    def test_instantiate_all(self):
        for _name, cls in VEHICLE_REGISTRY.items():
            v = cls()
            assert hasattr(v, "name")


class TestEnvironmentAssets:
    @pytest.mark.parametrize("env_name", [
        "shipwreck/herkules",
        "munkholmen",
        "bop_panel",
        "santorini",
        "pipeline",
        "shipwreck/skytteren",
        "rock",
    ])
    def test_environment_dir_exists(self, env_name):
        from importlib.resources import files
        from pathlib import Path
        p = Path(str(files("oceanscale.assets").joinpath("environments", env_name)))
        assert p.exists(), f"Environment {env_name} not found at {p}"

    def test_shipwreck_has_mesh(self):
        from importlib.resources import files
        from pathlib import Path
        p = Path(str(files("oceanscale.assets").joinpath("environments", "shipwreck", "herkules")))
        meshes = list(p.rglob("*.dae")) + list(p.rglob("*.stl"))
        assert len(meshes) > 0


class TestObjectAssets:
    @pytest.mark.parametrize("obj", [
        "sunken_ship_distorted",
        "torpedo_mk48",
        "uxo_a",
        "flight_data_recorder",
    ])
    def test_object_dir_exists(self, obj):
        from importlib.resources import files
        from pathlib import Path
        p = Path(str(files("oceanscale.assets").joinpath("objects", obj)))
        assert p.exists()


class TestSensorAssets:
    def test_dvl_configs_exist(self):
        from importlib.resources import files
        from pathlib import Path
        p = Path(str(files("oceanscale.assets").joinpath("sensors", "dvl_configs")))
        sdfs = list(p.glob("*.sdf"))
        assert len(sdfs) >= 4

    def test_dvl_mesh_exists(self):
        from importlib.resources import files
        from pathlib import Path
        p = Path(str(files("oceanscale.assets").joinpath("sensors", "dvl.dae")))
        assert p.exists()


class TestFullFleet:
    """All vehicles in OceanScale."""

    def test_total_vehicle_count(self):
        all_vehicles = [
            BlueROV2Heavy,
            BlueROV2MarineGym,
            WarpAUV,
            RexROV,
            SlocumGlider,
            WaveGlider,
            WHOIHybridGlider,
        ]
        assert len(all_vehicles) == 7
        for cls in all_vehicles:
            v = cls()
            assert v is not None
