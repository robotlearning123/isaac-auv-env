# Tests for ROS underwater resource integration into OceanScale.
# Verifies: DR configs, T200 thruster, Lee controller, RexROV controllers,
# current field presets, vehicle YAML loading, DVL configs.

import math
from pathlib import Path

import numpy as np
import pytest


# --- MarineGym DR configs ---

class TestMarineGymDR:
    def test_body_dr_defaults(self):
        from oceanscale.training.marinegym_dr import MarineGymBodyDR
        dr = MarineGymBodyDR()
        assert dr.mass_scale == (0.8, 1.2)
        assert dr.volume_scale == (0.9, 1.1)
        assert dr.added_mass_scale == (0.5, 1.0)

    def test_no_randomization(self):
        from oceanscale.training.marinegym_dr import MarineGymDRConfig
        cfg = MarineGymDRConfig.no_randomization()
        assert cfg.body.mass_scale == (1.0, 1.0)
        assert cfg.flow.max_flow_velocity == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_train_defaults(self):
        from oceanscale.training.marinegym_dr import MarineGymDRConfig
        cfg = MarineGymDRConfig.train_defaults()
        assert cfg.body.mass_scale == (0.8, 1.2)
        assert cfg.rotor.time_constants_scale == (0.8, 1.2)
        assert cfg.flow.max_flow_velocity[0] == 0.5

    def test_flow_disturbance_6dof(self):
        from oceanscale.training.marinegym_dr import FlowDisturbance
        f = FlowDisturbance()
        assert len(f.max_flow_velocity) == 6
        assert len(f.flow_noise) == 6


# --- T200 thruster ---

class TestT200:
    def test_config_defaults(self):
        from oceanscale.propulsion.t200 import T200Config
        cfg = T200Config()
        assert cfg.num_rotors == 6
        assert cfg.tau_up == 0.43
        assert cfg.deadband == 0.075

    def test_thrust_positive_rpm(self):
        from oceanscale.propulsion.t200 import T200Config
        cfg = T200Config()
        rpm = 2000.0
        thrust = 4.7368e-07 * rpm * rpm - 1.9275e-04 * rpm + 8.4452e-02
        assert thrust > 0

    def test_thrust_negative_rpm(self):
        from oceanscale.propulsion.t200 import T200Config
        cfg = T200Config()
        rpm = -2000.0
        thrust = -3.8442e-07 * rpm * rpm - 1.6186e-04 * rpm - 3.9139e-02
        assert thrust < 0


# --- Lee controller ---

class TestLeeController:
    def test_bluerov_heavy_preset(self):
        from oceanscale.controllers.lee_position import LeePositionConfig
        cfg = LeePositionConfig.bluerov_heavy()
        np.testing.assert_allclose(cfg.position_gain, [6.0, 6.0, 6.0])
        np.testing.assert_allclose(cfg.velocity_gain, [4.7, 4.7, 4.7])
        np.testing.assert_allclose(cfg.attitude_gain, [3.0, 3.0, 0.15])
        np.testing.assert_allclose(cfg.angular_rate_gain, [0.52, 0.52, 0.18])

    def test_controller_output_shape(self):
        from oceanscale.controllers.lee_position import LeePositionController, LeePositionConfig
        ctrl = LeePositionController(LeePositionConfig.bluerov_heavy())
        out = ctrl.compute(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation_quat=np.array([0, 0, 0, 1], dtype=np.float32),
            angular_velocity=np.zeros(3),
            target_position=np.array([1.0, 0.0, 0.0]),
        )
        assert out.shape == (4,)
        assert np.isfinite(out).all()


# --- RexROV controllers ---

class TestRexROVControllers:
    def test_available_controllers(self):
        from oceanscale.vehicles.fleet import RexROV
        r = RexROV()
        names = r.available_controllers()
        assert "pos_pid_control" in names
        assert "sliding_mode" in names
        assert len(names) >= 4

    def test_load_controller_raw(self):
        from oceanscale.vehicles.fleet import RexROV
        r = RexROV()
        params = r.controller_params("pos_pid_control")
        assert any("pos_p" in k for k in params)

    def test_yaml_loader_raw(self):
        from oceanscale.controllers.yaml_loader import load_rexrov_raw, rexrov_controller_configs
        configs = rexrov_controller_configs()
        assert len(configs) >= 4
        data = load_rexrov_raw("pos_pid_control")
        assert any("pos_p" in k for k in data)

    def test_yaml_loader_pid(self):
        from oceanscale.controllers.yaml_loader import load_pid_yaml
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("kp: [1,2,3,4,5,6]\nki: [0,0,0,0,0,0]\nkd: [0.1,0.1,0.1,0.1,0.1,0.1]\n")
            f.flush()
            cfg = load_pid_yaml(f.name)
        os.unlink(f.name)
        np.testing.assert_allclose(cfg.kp, [1, 2, 3, 4, 5, 6])

    def test_yaml_loader_cascaded_rexrov(self):
        from oceanscale.controllers.yaml_loader import load_pid_yaml, rexrov_controller_configs
        configs = rexrov_controller_configs()
        path = configs["pos_pid_control"]
        cfg = load_pid_yaml(path)
        assert cfg.kp[0] > 0
        assert cfg.kp[3] > 0
        assert cfg.sat is not None
        assert cfg.sat[0] > 0


# --- Current field presets ---

class TestCurrentPresets:
    def test_uniform(self):
        from oceanscale.currents import OceanCurrentField
        c = OceanCurrentField.uniform(speed=0.5)
        assert c._bg_u == pytest.approx(0.5, abs=0.01)
        assert c._bg_v == pytest.approx(0.0, abs=0.01)
        assert c._tidal_amp == 0.0

    def test_tidal(self):
        from oceanscale.currents import OceanCurrentField
        c = OceanCurrentField.tidal(amplitude=0.8)
        assert c._tidal_amp == 0.8

    def test_deep_sea(self):
        from oceanscale.currents import OceanCurrentField
        c = OceanCurrentField.deep_sea(seabed_depth=200.0)
        assert c.seabed_depth == 200.0
        assert c._tidal_amp == 0.1

    def test_harbor(self):
        from oceanscale.currents import OceanCurrentField
        c = OceanCurrentField.harbor()
        assert c._tidal_amp == 0.8
        assert c.seabed_depth == 15.0


# --- Vehicle YAML config loader ---

class TestVehicleHydroConfig:
    def test_from_rexrov_yaml(self):
        from oceanscale.vehicles.fleet import VehicleHydroConfig
        path = Path(__file__).resolve().parent.parent / "oceanscale" / "assets" / "rexrov" / "hydro_params.yaml"
        cfg = VehicleHydroConfig.from_yaml(path)
        assert cfg.name == "rexrov"
        assert cfg.mass == pytest.approx(1862.87)
        assert len(cfg.added_mass) == 6
        assert cfg.added_mass[0] == 700.0

    def test_added_mass_diagonal(self):
        from oceanscale.vehicles.fleet import VehicleHydroConfig
        path = Path(__file__).resolve().parent.parent / "oceanscale" / "assets" / "rexrov" / "hydro_params.yaml"
        cfg = VehicleHydroConfig.from_yaml(path)
        assert cfg.added_mass == (700, 1200, 3500, 500, 800, 200)


# --- DVL configs ---

class TestDVLConfigs:
    def test_dvl_config_count(self):
        from oceanscale.sensors.dvl_configs import ALL_DVL_CONFIGS
        assert len(ALL_DVL_CONFIGS) >= 9

    def test_nortek_dvl_present(self):
        from oceanscale.sensors.dvl_configs import ALL_DVL_CONFIGS
        names = [c.name for c in ALL_DVL_CONFIGS.values()]
        assert any("Nortek" in n for n in names)

    def test_dvl_config_fields(self):
        from oceanscale.sensors.dvl_configs import ALL_DVL_CONFIGS
        for cfg in ALL_DVL_CONFIGS.values():
            assert cfg.frequency_khz > 0
            assert cfg.max_range_m > 0
            assert cfg.n_beams >= 3
