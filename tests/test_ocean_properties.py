"""Tests for realistic ocean water column properties."""

import math

import numpy as np
import pytest
import warp as wp

from oceanscale.ocean_properties import WaterColumn

wp.init()


@pytest.fixture
def wc():
    return WaterColumn(
        surface_temperature=20.0,
        bottom_temperature=4.0,
        thermocline_depth=50.0,
        thermocline_thickness=20.0,
        surface_salinity=35.0,
        bottom_salinity=34.8,
        max_depth=200.0,
    )


class TestTemperature:
    def test_surface(self, wc):
        assert abs(wc.temperature(0.0) - 20.0) < 0.5

    def test_bottom(self, wc):
        assert abs(wc.temperature(200.0) - 4.0) < 0.5

    def test_decreases_with_depth(self, wc):
        temps = [wc.temperature(d) for d in range(0, 201, 10)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-6

    def test_thermocline_steepest(self, wc):
        gradient = []
        for d in range(5, 196, 5):
            g = abs(wc.temperature(d + 2.5) - wc.temperature(d - 2.5)) / 5.0
            gradient.append((d, g))
        steepest = max(gradient, key=lambda x: x[1])
        assert 30 < steepest[0] < 70


class TestSalinity:
    def test_surface(self, wc):
        assert abs(wc.salinity(0.0) - 35.0) < 0.01

    def test_bottom(self, wc):
        assert abs(wc.salinity(200.0) - 34.8) < 0.01

    def test_linear(self, wc):
        s_mid = wc.salinity(100.0)
        expected = 35.0 + (34.8 - 35.0) * 0.5
        assert abs(s_mid - expected) < 0.01


class TestDensity:
    def test_range(self, wc):
        for d in range(0, 201, 20):
            rho = wc.density(d)
            assert 1020.0 < rho < 1030.0, f"Density {rho} out of range at depth {d}"

    def test_increases_with_depth(self, wc):
        densities = [wc.density(d) for d in range(0, 201, 10)]
        for i in range(1, len(densities)):
            assert densities[i] >= densities[i - 1] - 0.01

    def test_surface_realistic(self, wc):
        rho = wc.density(0.0)
        assert 1024.0 < rho < 1026.0


class TestSoundSpeed:
    def test_range(self, wc):
        for d in range(0, 201, 20):
            c = wc.sound_speed(d)
            assert 1450.0 < c < 1550.0, f"Sound speed {c} out of range at depth {d}"

    def test_surface_warm(self, wc):
        c = wc.sound_speed(0.0)
        assert c > 1510.0

    def test_deep_cold(self, wc):
        c = wc.sound_speed(200.0)
        assert c < wc.sound_speed(0.0)

    def test_mackenzie_known_value(self):
        wc_simple = WaterColumn(
            surface_temperature=10.0,
            bottom_temperature=10.0,
            surface_salinity=35.0,
            bottom_salinity=35.0,
            max_depth=100.0,
        )
        c = wc_simple.sound_speed(0.0)
        expected = 1448.96 + 4.591 * 10.0 - 5.304e-2 * 100.0 + 2.374e-4 * 1000.0
        assert abs(c - expected) < 0.1


class TestBuoyancy:
    def test_positive(self, wc):
        f = wc.buoyancy_force(10.0, 0.01)
        assert f > 0.0

    def test_scales_with_volume(self, wc):
        f1 = wc.buoyancy_force(10.0, 0.01)
        f2 = wc.buoyancy_force(10.0, 0.02)
        assert abs(f2 / f1 - 2.0) < 0.01

    def test_magnitude(self, wc):
        f = wc.buoyancy_force(0.0, 1.0)
        assert 10000.0 < f < 10100.0


class TestLightAttenuation:
    def test_surface_full(self, wc):
        assert abs(wc.light_attenuation(0.0) - 1.0) < 1e-6

    def test_decays(self, wc):
        vals = [wc.light_attenuation(d) for d in range(0, 201, 10)]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i - 1]

    def test_exponential(self, wc):
        d = 50.0
        expected = math.exp(-0.08 * d)
        assert abs(wc.light_attenuation(d) - expected) < 1e-6

    def test_deep_very_dark(self, wc):
        assert wc.light_attenuation(200.0) < 1e-6


class TestProfile:
    def test_all_keys(self, wc):
        depths = np.arange(0, 201, 10, dtype=float)
        p = wc.get_profile(depths)
        assert set(p.keys()) == {"depth", "temperature", "salinity", "density", "sound_speed", "light"}

    def test_shapes(self, wc):
        depths = np.arange(0, 201, 10, dtype=float)
        p = wc.get_profile(depths)
        for k in p:
            assert len(p[k]) == len(depths)


class TestGPU:
    def test_density_gpu_matches_cpu(self, wc):
        depths_np = np.arange(0, 201, 5, dtype=np.float32)
        depths_wp = wp.array(depths_np, dtype=wp.float32, device="cuda:0")
        rho_gpu = wc.density_gpu(depths_wp).numpy()
        rho_cpu = np.array([wc.density(d) for d in depths_np])
        np.testing.assert_allclose(rho_gpu, rho_cpu, rtol=1e-4)

    def test_sound_speed_gpu_matches_cpu(self, wc):
        depths_np = np.arange(0, 201, 5, dtype=np.float32)
        depths_wp = wp.array(depths_np, dtype=wp.float32, device="cuda:0")
        c_gpu = wc.sound_speed_gpu(depths_wp).numpy()
        c_cpu = np.array([wc.sound_speed(d) for d in depths_np])
        np.testing.assert_allclose(c_gpu, c_cpu, rtol=1e-4)

    def test_density_gpu_large_batch(self, wc):
        depths_wp = wp.array(np.random.rand(100000).astype(np.float32) * 200.0, dtype=wp.float32, device="cuda:0")
        rho = wc.density_gpu(depths_wp).numpy()
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 1020.0)
        assert np.all(rho < 1030.0)

    def test_sound_speed_gpu_large_batch(self, wc):
        depths_wp = wp.array(np.random.rand(100000).astype(np.float32) * 200.0, dtype=wp.float32, device="cuda:0")
        c = wc.sound_speed_gpu(depths_wp).numpy()
        assert np.all(np.isfinite(c))
        assert np.all(c > 1450.0)
        assert np.all(c < 1550.0)
