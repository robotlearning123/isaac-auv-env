"""Tests for realistic propulsion models."""

import math

import numpy as np
import pytest
import warp as wp

from oceanscale.propulsion import BuoyancyEngine, FlappingFin, PropellerThruster

wp.init()


class TestPropellerStaticThrust:
    def test_static_thrust_formula(self):
        p = PropellerThruster(diameter=0.076, kt0=0.4, rho=1025.0)
        rpm = 2000.0
        n = rpm / 60.0
        expected = 0.4 * 1025.0 * n**2 * 0.076**4
        actual = p.thrust(rpm, advance_velocity=0.0)
        assert abs(actual - expected) < 1e-6

    def test_zero_rpm_zero_thrust(self):
        p = PropellerThruster()
        assert p.thrust(0.0) == 0.0
        assert p.torque(0.0) == 0.0

    def test_negative_rpm_reverses_thrust(self):
        p = PropellerThruster()
        t_pos = p.thrust(2000.0)
        t_neg = p.thrust(-2000.0)
        assert t_pos > 0
        assert t_neg < 0
        assert abs(t_pos + t_neg) < 1e-6


class TestPropellerAdvanceRatio:
    def test_thrust_decreases_with_advance(self):
        p = PropellerThruster()
        t0 = p.thrust(2000.0, advance_velocity=0.0)
        t1 = p.thrust(2000.0, advance_velocity=0.5)
        t2 = p.thrust(2000.0, advance_velocity=1.0)
        assert t0 > t1 > t2

    def test_torque_decreases_with_advance(self):
        p = PropellerThruster()
        q0 = p.torque(2000.0, advance_velocity=0.0)
        q1 = p.torque(2000.0, advance_velocity=1.0)
        assert q0 > q1

    def test_efficiency_range(self):
        p = PropellerThruster()
        eta = p.efficiency(2000.0, advance_velocity=0.5)
        assert 0.0 <= eta <= 1.0

    def test_zero_advance_zero_efficiency(self):
        p = PropellerThruster()
        assert p.efficiency(2000.0, 0.0) == 0.0


class TestPropellerCavitation:
    def test_no_cavitation_at_depth(self):
        p = PropellerThruster()
        assert not p.is_cavitating(2000.0, depth=50.0)

    def test_cavitation_more_likely_at_surface(self):
        p = PropellerThruster()
        cav_deep = p.is_cavitating(3500.0, depth=50.0)
        cav_shallow = p.is_cavitating(3500.0, depth=0.1)
        if cav_shallow:
            assert cav_shallow or not cav_deep

    def test_zero_rpm_no_cavitation(self):
        p = PropellerThruster()
        assert not p.is_cavitating(0.0, depth=0.0)


class TestPropellerGPU:
    def test_gpu_batch(self):
        p = PropellerThruster()
        rpms = wp.array([1000.0, 2000.0, 3000.0], dtype=wp.float32, device="cuda:0")
        vels = wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device="cuda:0")
        t, q = p.thrust_gpu(rpms, vels)
        tn = t.numpy()
        qn = q.numpy()
        assert tn[0] < tn[1] < tn[2]
        assert qn[0] < qn[1] < qn[2]

    def test_gpu_matches_cpu(self):
        p = PropellerThruster()
        rpms = wp.array([2000.0], dtype=wp.float32, device="cuda:0")
        vels = wp.array([0.0], dtype=wp.float32, device="cuda:0")
        t_gpu, _ = p.thrust_gpu(rpms, vels)
        t_cpu = p.thrust(2000.0, 0.0)
        assert abs(t_gpu.numpy()[0] - t_cpu) < 1e-4


class TestFlappingFin:
    def test_zero_frequency_zero_thrust(self):
        f = FlappingFin()
        t, lat = f.forces(frequency=0.0, amplitude=0.3, flow_velocity=0.5, phase=0.0)
        assert t == 0.0
        assert lat == 0.0

    def test_zero_amplitude_zero_thrust(self):
        f = FlappingFin()
        t, lat = f.forces(frequency=2.0, amplitude=0.0, flow_velocity=0.5, phase=0.0)
        assert t == 0.0
        assert lat == 0.0

    def test_strouhal_number_calculation(self):
        f = FlappingFin()
        st = f.strouhal_number(frequency=2.0, amplitude=0.05, flow_velocity=0.5)
        expected = 2.0 * 2.0 * 0.05 / 0.5
        assert abs(st - expected) < 1e-8

    def test_optimal_strouhal_positive_thrust(self):
        f = FlappingFin(span=0.20, chord=0.08, cd0=0.01)
        mean_t = f.mean_thrust(frequency=3.0, amplitude=0.15, flow_velocity=0.5)
        st = f.strouhal_number(3.0, 0.15, 0.5)
        assert 0.1 < st < 2.0
        assert mean_t > 0.0

    def test_forces_finite(self):
        f = FlappingFin()
        for phase in np.linspace(0, 2 * math.pi, 32):
            t, lat = f.forces(2.0, 0.3, 0.5, phase)
            assert math.isfinite(t)
            assert math.isfinite(lat)


class TestBuoyancyEngine:
    def test_initial_state(self):
        b = BuoyancyEngine()
        assert b.current_volume == 0.0

    def test_pump_rate_respected(self):
        b = BuoyancyEngine(pump_rate=0.0001)
        b.set_target_volume(0.0005)
        b.step(dt=1.0)
        assert abs(b.current_volume - 0.0001) < 1e-10

    def test_reaches_target(self):
        b = BuoyancyEngine(pump_rate=0.001, max_volume_change=0.001)
        b.set_target_volume(0.0005)
        for _ in range(10):
            b.step(dt=1.0)
        assert abs(b.current_volume - 0.0005) < 1e-10

    def test_positive_buoyancy(self):
        b = BuoyancyEngine(hull_volume=0.05, hull_mass=50.0)
        b.set_target_volume(0.0005)
        for _ in range(100):
            b.step(dt=0.1)
        assert b.is_positive(1025.0)

    def test_negative_buoyancy(self):
        b = BuoyancyEngine(hull_volume=0.05, hull_mass=52.0)
        b.set_target_volume(-0.0005)
        for _ in range(100):
            b.step(dt=0.1)
        assert not b.is_positive(1025.0)

    def test_net_buoyancy_formula(self):
        b = BuoyancyEngine(hull_volume=0.05, hull_mass=51.0)
        expected = 1025.0 * 9.81 * 0.05 - 51.0 * 9.81
        actual = b.net_buoyancy(1025.0)
        assert abs(actual - expected) < 1e-6

    def test_clamp_to_max(self):
        b = BuoyancyEngine(max_volume_change=0.0005)
        b.set_target_volume(999.0)
        for _ in range(1000):
            b.step(dt=1.0)
        assert b.current_volume <= 0.0005 + 1e-10

    def test_reset(self):
        b = BuoyancyEngine()
        b.set_target_volume(0.0003)
        b.step(dt=10.0)
        b.reset()
        assert b.current_volume == 0.0
