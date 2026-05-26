"""Tests for T200 thruster model ported from MarineGym."""

import numpy as np
import pytest
import warp as wp

wp.init()

from oceanscale.propulsion.t200 import T200Config, T200Thruster


@pytest.fixture
def thruster():
    return T200Thruster(num_envs=1, dt=0.02, device="cuda:0")


@pytest.fixture
def batch_thruster():
    return T200Thruster(num_envs=4, dt=0.02, device="cuda:0")


class TestDeadband:
    def test_zero_command_gives_zero_thrust(self, thruster):
        cmd = wp.array([0.0] * 6, dtype=wp.float32, device="cuda:0")
        thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.allclose(vals, 0.0, atol=1e-6)

    def test_small_command_below_deadband(self, thruster):
        cmd = wp.array([0.05] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(50):
            thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.allclose(vals, 0.0, atol=1e-6), f"Expected zero thrust in deadband, got {vals}"

    def test_negative_small_command_below_deadband(self, thruster):
        cmd = wp.array([-0.05] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(50):
            thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.allclose(vals, 0.0, atol=1e-6)

    def test_above_deadband_gives_nonzero_thrust(self, thruster):
        cmd = wp.array([0.5] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(200):
            thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.all(vals > 0.1), f"Expected positive thrust, got {vals}"


class TestTimeConstantFiltering:
    def test_thrust_ramps_up_gradually(self, thruster):
        cmd = wp.array([1.0] * 6, dtype=wp.float32, device="cuda:0")
        thrusts = []
        for _ in range(100):
            thrust = thruster.step(cmd)
            thrusts.append(thrust.numpy().copy())
        first = np.abs(thrusts[0]).mean()
        last = np.abs(thrusts[-1]).mean()
        assert last > first, "Thrust should increase over time"

    def test_throttle_filter_tau(self, thruster):
        cmd = wp.array([1.0] * 6, dtype=wp.float32, device="cuda:0")
        thruster.step(cmd)
        throttle_1 = thruster.throttle.copy()
        thruster.step(cmd)
        throttle_2 = thruster.throttle.copy()
        assert np.all(throttle_2 > throttle_1), "Throttle should increase each step"
        assert np.all(throttle_2 < 1.0), "Throttle shouldn't reach target in 2 steps (tau=0.43)"

    def test_rpm_converges(self, thruster):
        cmd = wp.array([0.5] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(500):
            thruster.step(cmd)
        rpm = thruster.rpm.flatten()
        expected_target_rpm = 0.5 * 3659.9 + 345.21
        assert np.allclose(rpm, expected_target_rpm, rtol=0.05), f"RPM {rpm[0]:.0f} vs expected {expected_target_rpm:.0f}"


class TestThrustCurve:
    def test_positive_thrust_at_max(self, thruster):
        cmd = wp.array([1.0] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(1000):
            thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.all(vals > 5.0), f"Max thrust should exceed 5N, got {vals[0]:.2f}"

    def test_negative_thrust(self, thruster):
        cmd = wp.array([-1.0] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(1000):
            thrust = thruster.step(cmd)
        vals = thrust.numpy()
        assert np.all(vals < -3.0), f"Reverse thrust should be < -3N, got {vals[0]:.2f}"

    def test_thrust_sign_matches_command(self, thruster):
        pos_cmd = wp.array([0.8] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(500):
            thruster.step(pos_cmd)
        pos_thrust = thruster.step(pos_cmd).numpy()

        thruster.reset()
        neg_cmd = wp.array([-0.8] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(500):
            thruster.step(neg_cmd)
        neg_thrust = thruster.step(neg_cmd).numpy()

        assert np.all(pos_thrust > 0)
        assert np.all(neg_thrust < 0)

    def test_clamped_to_range(self, thruster):
        cmd = wp.array([2.0] * 6, dtype=wp.float32, device="cuda:0")
        for _ in range(500):
            thruster.step(cmd)
        rpm = thruster.rpm.flatten()
        assert np.all(np.abs(rpm) <= 3900.0 + 1e-3)


class TestBatchOperation:
    def test_multi_env_independent(self, batch_thruster):
        num_rotors = batch_thruster.cfg.num_rotors
        cmds = np.zeros(4 * num_rotors, dtype=np.float32)
        cmds[0:num_rotors] = 1.0
        cmds[num_rotors : 2 * num_rotors] = -1.0
        cmds[2 * num_rotors : 3 * num_rotors] = 0.0
        cmds[3 * num_rotors :] = 0.5
        cmd_wp = wp.array(cmds, dtype=wp.float32, device="cuda:0")

        for _ in range(500):
            batch_thruster.step(cmd_wp)
        thrust = batch_thruster.step(cmd_wp).numpy().reshape(4, num_rotors)

        assert np.all(thrust[0] > 0), "Env 0: full forward"
        assert np.all(thrust[1] < 0), "Env 1: full reverse"
        assert np.allclose(thrust[2], 0.0, atol=1e-6), "Env 2: zero"
        assert np.all(thrust[3] > 0), "Env 3: half forward"

    def test_reset_single_env(self, batch_thruster):
        num_rotors = batch_thruster.cfg.num_rotors
        cmd = wp.array([1.0] * (4 * num_rotors), dtype=wp.float32, device="cuda:0")
        for _ in range(100):
            batch_thruster.step(cmd)

        batch_thruster.reset(env_ids=np.array([1]))
        rpm = batch_thruster.rpm
        assert np.all(np.abs(rpm[0]) > 100), "Env 0 should still have RPM"
        assert np.allclose(rpm[1], 0.0, atol=1e-6), "Env 1 should be reset"


class TestConfig:
    def test_default_config(self):
        cfg = T200Config()
        assert cfg.num_rotors == 6
        assert cfg.max_rpm == 3900.0
        assert cfg.deadband == 0.075

    def test_from_yaml(self, tmp_path):
        yaml_content = """
name: TestROV
rotor_configuration:
  num_rotors: 4
  force_constants: [4.4e-07, 4.4e-07, 4.4e-07, 4.4e-07]
  moment_constants: [1.37e-09, 1.37e-09, 1.37e-09, 1.37e-09]
  max_rotation_velocities: [3000, 3000, 3000, 3000]
  time_constants: [0.02, 0.02, 0.02, 0.02]
  directions: [1, -1, 1, -1]
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        cfg = T200Config.from_marinegym_yaml(str(yaml_file))
        assert cfg.num_rotors == 4
        assert cfg.max_rpm == 3000.0
        assert cfg.time_constant == 0.02
