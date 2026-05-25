"""Tests for realistic ocean current model."""

import math

import numpy as np
import pytest
import warp as wp

from oceanscale.currents import M2_PERIOD, S2_PERIOD, OceanCurrentField

wp.init()


class TestTidalCurrents:
    def test_periodic_m2(self):
        field = OceanCurrentField(tidal_amplitude=0.5, s2_amplitude=0.0, wind_speed=0.0,
                                  background_speed=0.0, turbulence_intensity=0.0)
        v0 = field.tidal_velocity(0.0)
        v_period = field.tidal_velocity(M2_PERIOD)
        np.testing.assert_allclose(v0, v_period, atol=1e-5)

    def test_tidal_amplitude(self):
        amp = 0.8
        field = OceanCurrentField(tidal_amplitude=amp, s2_amplitude=0.0, wind_speed=0.0,
                                  background_speed=0.0, turbulence_intensity=0.0)
        v = field.tidal_velocity(0.0)
        assert abs(np.linalg.norm(v)) == pytest.approx(amp, abs=1e-5)

    def test_m2_plus_s2(self):
        field = OceanCurrentField(tidal_amplitude=0.5, s2_amplitude=0.15, wind_speed=0.0,
                                  background_speed=0.0, turbulence_intensity=0.0)
        v_peak = field.tidal_velocity(0.0)
        assert abs(np.linalg.norm(v_peak)) == pytest.approx(0.65, abs=1e-5)

    def test_tidal_zero_at_quarter_period(self):
        field = OceanCurrentField(tidal_amplitude=0.5, s2_amplitude=0.0, wind_speed=0.0,
                                  background_speed=0.0, turbulence_intensity=0.0)
        v = field.tidal_velocity(M2_PERIOD / 4.0)
        assert abs(np.linalg.norm(v)) < 1e-4

    def test_tidal_direction(self):
        angle = math.pi / 3.0
        field = OceanCurrentField(tidal_amplitude=1.0, tidal_direction=angle,
                                  s2_amplitude=0.0, wind_speed=0.0,
                                  background_speed=0.0, turbulence_intensity=0.0)
        v = field.tidal_velocity(0.0)
        assert v[0] == pytest.approx(math.cos(angle), abs=1e-5)
        assert v[1] == pytest.approx(math.sin(angle), abs=1e-5)


class TestEkmanSpiral:
    @pytest.fixture
    def ekman_field(self):
        return OceanCurrentField(tidal_amplitude=0.0, s2_amplitude=0.0,
                                 wind_speed=10.0, wind_direction=0.0,
                                 ekman_depth=50.0, background_speed=0.0,
                                 turbulence_intensity=0.0)

    def test_surface_nonzero(self, ekman_field):
        v = ekman_field.ekman_velocity(0.0)
        assert np.linalg.norm(v) > 0.01

    def test_decay_with_depth(self, ekman_field):
        v_surface = np.linalg.norm(ekman_field.ekman_velocity(0.0))
        v_25m = np.linalg.norm(ekman_field.ekman_velocity(25.0))
        v_50m = np.linalg.norm(ekman_field.ekman_velocity(50.0))
        assert v_surface > v_25m > v_50m

    def test_exponential_decay(self, ekman_field):
        v0 = np.linalg.norm(ekman_field.ekman_velocity(0.0))
        v_de = np.linalg.norm(ekman_field.ekman_velocity(50.0))
        ratio = v_de / v0
        assert ratio == pytest.approx(math.exp(-1.0), abs=0.05)

    def test_rotation_with_depth(self, ekman_field):
        v_surf = ekman_field.ekman_velocity(0.0)
        v_deep = ekman_field.ekman_velocity(25.0)
        angle_surf = math.atan2(v_surf[1], v_surf[0])
        angle_deep = math.atan2(v_deep[1], v_deep[0])
        assert angle_deep != pytest.approx(angle_surf, abs=0.1)

    def test_surface_45_degrees_right(self, ekman_field):
        v = ekman_field.ekman_velocity(0.0)
        angle = math.atan2(v[1], v[0])
        assert angle == pytest.approx(math.pi / 4.0, abs=0.01)


class TestBoundaryLayer:
    def test_zero_at_bottom(self):
        field = OceanCurrentField()
        assert field.boundary_layer_factor(0.0) == 0.0

    def test_one_above_5m(self):
        field = OceanCurrentField()
        assert field.boundary_layer_factor(5.0) == pytest.approx(1.0, abs=1e-5)
        assert field.boundary_layer_factor(10.0) == pytest.approx(1.0, abs=1e-5)

    def test_log_profile(self):
        field = OceanCurrentField()
        f1 = field.boundary_layer_factor(0.1)
        f2 = field.boundary_layer_factor(1.0)
        f3 = field.boundary_layer_factor(3.0)
        assert 0.0 < f1 < f2 < f3 <= 1.0

    def test_velocity_reduced_near_seabed(self):
        field = OceanCurrentField(tidal_amplitude=0.5, s2_amplitude=0.0,
                                  wind_speed=0.0, background_speed=0.1,
                                  turbulence_intensity=0.0, seabed_depth=50.0)
        pos_mid = np.array([[50.0, 50.0, -25.0]], dtype=np.float32)
        pos_bottom = np.array([[50.0, 50.0, -49.99]], dtype=np.float32)
        v_mid = np.linalg.norm(field.velocity_at(pos_mid, 0.0))
        v_bottom = np.linalg.norm(field.velocity_at(pos_bottom, 0.0))
        assert v_bottom < v_mid


class TestTurbulence:
    def test_turbulence_bounded(self):
        field = OceanCurrentField(tidal_amplitude=0.5, turbulence_intensity=0.1,
                                  wind_speed=0.0, background_speed=0.0, s2_amplitude=0.0)
        positions = np.random.rand(1000, 3).astype(np.float32) * np.array([100, 100, 1]) - np.array([0, 0, 25])
        v = field.velocity_at(positions, 5.0)
        assert np.all(np.isfinite(v))

    def test_turbulence_varies_in_space(self):
        field = OceanCurrentField(tidal_amplitude=0.0, turbulence_intensity=0.1,
                                  wind_speed=0.0, background_speed=0.0, s2_amplitude=0.0,
                                  seabed_depth=1000.0)
        p1 = np.array([[10.0, 10.0, -5.0]], dtype=np.float32)
        p2 = np.array([[50.0, 50.0, -5.0]], dtype=np.float32)
        v1 = field.velocity_at(p1, 1.0)
        v2 = field.velocity_at(p2, 1.0)
        assert not np.allclose(v1, v2, atol=1e-6)

    def test_turbulence_varies_in_time(self):
        field = OceanCurrentField(tidal_amplitude=0.0, turbulence_intensity=0.1,
                                  wind_speed=0.0, background_speed=0.0, s2_amplitude=0.0,
                                  seabed_depth=1000.0)
        pos = np.array([[10.0, 10.0, -5.0]], dtype=np.float32)
        v1 = field.velocity_at(pos, 1.0)
        v2 = field.velocity_at(pos, 10.0)
        assert not np.allclose(v1, v2, atol=1e-6)


class TestCombinedField:
    def test_all_components_nonzero(self):
        field = OceanCurrentField(tidal_amplitude=0.3, s2_amplitude=0.1,
                                  wind_speed=5.0, background_speed=0.1,
                                  turbulence_intensity=0.05, seabed_depth=50.0)
        pos = np.array([[50.0, 50.0, -10.0]], dtype=np.float32)
        v = field.velocity_at(pos, 1000.0)
        assert np.linalg.norm(v) > 0.01

    def test_batch_query(self):
        field = OceanCurrentField()
        positions = np.random.rand(500, 3).astype(np.float32) * np.array([200, 200, 1]) - np.array([0, 0, 25])
        v = field.velocity_at(positions, 100.0)
        assert v.shape == (500, 3)
        assert np.all(np.isfinite(v))


class TestGPU:
    def test_gpu_matches_cpu(self):
        field = OceanCurrentField(tidal_amplitude=0.5, s2_amplitude=0.1,
                                  wind_speed=5.0, background_speed=0.1,
                                  turbulence_intensity=0.0, seabed_depth=50.0)
        positions = np.array([[50, 50, -10], [50, 50, -25], [50, 50, -45]], dtype=np.float32)
        v_cpu = field.velocity_at(positions, 500.0)

        pos_gpu = wp.array(positions, dtype=wp.vec3f, device="cuda:0")
        v_gpu = field.velocity_at_gpu(pos_gpu, 500.0)
        wp.synchronize()
        v_gpu_np = v_gpu.numpy()

        np.testing.assert_allclose(v_cpu, v_gpu_np, atol=1e-4)

    def test_gpu_large_batch(self):
        field = OceanCurrentField()
        positions = wp.array(
            np.random.rand(10000, 3).astype(np.float32) * np.array([200, 200, 1]) - np.array([0, 0, 25]),
            dtype=wp.vec3f, device="cuda:0",
        )
        v = field.velocity_at_gpu(positions, 100.0)
        wp.synchronize()
        vals = v.numpy()
        assert vals.shape == (10000, 3)
        assert np.all(np.isfinite(vals))
