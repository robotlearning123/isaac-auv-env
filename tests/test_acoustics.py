"""Tests for underwater acoustic propagation model."""

import math

import numpy as np
import pytest
import warp as wp

from oceanscale.acoustics import (
    AcousticPropagation,
    WaterColumn,
    mackenzie_sound_speed,
    thorp_absorption,
    wenz_ambient_noise,
)


class TestMackenzieSoundSpeed:
    def test_typical_range(self):
        c = mackenzie_sound_speed(temperature=15.0, salinity=35.0, depth=100.0)
        assert 1450.0 < c < 1550.0

    def test_increases_with_temperature(self):
        c_cold = mackenzie_sound_speed(5.0, 35.0, 100.0)
        c_warm = mackenzie_sound_speed(25.0, 35.0, 100.0)
        assert c_warm > c_cold

    def test_increases_with_depth(self):
        c_shallow = mackenzie_sound_speed(10.0, 35.0, 10.0)
        c_deep = mackenzie_sound_speed(10.0, 35.0, 4000.0)
        assert c_deep > c_shallow

    def test_increases_with_salinity(self):
        c_fresh = mackenzie_sound_speed(15.0, 30.0, 100.0)
        c_salt = mackenzie_sound_speed(15.0, 40.0, 100.0)
        assert c_salt > c_fresh


class TestThorpAbsorption:
    def test_positive(self):
        assert thorp_absorption(10e3) > 0.0

    def test_increases_with_frequency(self):
        a_low = thorp_absorption(1e3)
        a_mid = thorp_absorption(10e3)
        a_high = thorp_absorption(100e3)
        assert a_mid > a_low
        assert a_high > a_mid

    def test_units_db_per_m(self):
        a = thorp_absorption(10e3)
        assert a < 1.0  # should be small per meter


class TestWenzNoise:
    def test_reasonable_range(self):
        nl = wenz_ambient_noise(1000.0, sea_state=3)
        assert 20.0 < nl < 100.0

    def test_increases_with_sea_state(self):
        nl_calm = wenz_ambient_noise(1000.0, sea_state=0)
        nl_rough = wenz_ambient_noise(1000.0, sea_state=6)
        assert nl_rough > nl_calm

    def test_shipping_effect(self):
        nl_light = wenz_ambient_noise(100.0, shipping="light")
        nl_heavy = wenz_ambient_noise(100.0, shipping="heavy")
        assert nl_heavy > nl_light


class TestWaterColumn:
    def test_surface_temperature(self):
        wc = WaterColumn(surface_temperature=22.0)
        assert abs(wc.temperature_at(0.0) - 22.0) < 1.0

    def test_temperature_decreases_with_depth(self):
        wc = WaterColumn(surface_temperature=25.0, bottom_temperature=4.0)
        t_surface = wc.temperature_at(0.0)
        t_deep = wc.temperature_at(200.0)
        assert t_surface > t_deep

    def test_density_positive(self):
        wc = WaterColumn()
        rho = wc.density_at(50.0)
        assert 1020.0 < rho < 1035.0

    def test_sound_speed_profile_shape(self):
        wc = WaterColumn()
        depths = np.linspace(0, 200, 50)
        ssp = wc.sound_speed_profile(depths)
        assert ssp.shape == (50,)
        assert np.all((ssp > 1400.0) & (ssp < 1600.0))


class TestAcousticPropagation:
    @pytest.fixture
    def ap(self):
        return AcousticPropagation(frequency=10e3, max_depth=200.0)

    def test_transmission_loss_increases_with_range(self, ap):
        tl_near = ap.transmission_loss(100.0)
        tl_far = ap.transmission_loss(1000.0)
        assert tl_far > tl_near

    def test_spherical_spreading(self, ap):
        tl = ap.transmission_loss(100.0)
        spreading = 20.0 * math.log10(100.0)
        assert abs(tl - spreading) < 10.0  # absorption adds a bit

    def test_detection_range_positive(self, ap):
        dr = ap.detection_range(source_level=180.0)
        assert dr > 0.0

    def test_detection_range_increases_with_sl(self, ap):
        dr_low = ap.detection_range(source_level=150.0)
        dr_high = ap.detection_range(source_level=200.0)
        assert dr_high > dr_low

    def test_sound_speed_profile(self, ap):
        depths = np.linspace(0, 200, 20)
        ssp = ap.sound_speed_profile(depths)
        assert ssp.shape == (20,)
        assert np.all(np.isfinite(ssp))

    def test_gpu_transmission_loss(self, ap):
        ranges = wp.array(np.array([10, 50, 100, 500, 1000], dtype=np.float32), device="cuda:0")
        tl = ap.transmission_loss_gpu(ranges)
        vals = tl.numpy()
        assert vals.shape == (5,)
        assert np.all(np.isfinite(vals))
        assert np.all(np.diff(vals) > 0)  # TL increases with range

    def test_ambient_noise(self, ap):
        nl = ap.ambient_noise()
        assert np.isfinite(nl)


class TestRayTracing:
    @pytest.fixture
    def uniform_ap(self):
        wc = WaterColumn(surface_temperature=15.0, bottom_temperature=15.0)
        return AcousticPropagation(water_column=wc, max_depth=200.0)

    @pytest.fixture
    def gradient_ap(self):
        wc = WaterColumn(surface_temperature=25.0, bottom_temperature=4.0, thermocline_depth=50.0)
        return AcousticPropagation(water_column=wc, max_depth=200.0)

    def test_ray_trace_shape(self, uniform_ap):
        angles = np.array([-10.0, 0.0, 10.0])
        paths = uniform_ap.ray_trace(source_depth=50.0, angles_deg=angles, n_steps=100)
        assert paths.shape == (3, 100, 2)

    def test_straight_in_uniform(self, uniform_ap):
        angles = np.array([0.0])
        paths = uniform_ap.ray_trace(source_depth=100.0, angles_deg=angles, n_steps=100, max_range=500.0)
        depths = paths[0, :, 1]
        assert np.std(depths) < 5.0  # nearly constant depth

    def test_rays_bend_in_gradient(self, gradient_ap):
        angles = np.array([5.0])
        paths = gradient_ap.ray_trace(source_depth=50.0, angles_deg=angles, n_steps=200, max_range=2000.0)
        depths = paths[0, :, 1]
        assert np.std(depths) > 1.0  # depth should vary due to refraction

    def test_surface_reflection(self, uniform_ap):
        angles = np.array([-30.0])  # upward ray
        paths = uniform_ap.ray_trace(source_depth=20.0, angles_deg=angles, n_steps=200, max_range=500.0)
        depths = paths[0, :, 1]
        assert np.all(depths >= 0.0)  # never goes above surface

    def test_bottom_reflection(self, uniform_ap):
        angles = np.array([30.0])  # downward ray
        paths = uniform_ap.ray_trace(source_depth=100.0, angles_deg=angles, n_steps=200, max_range=500.0)
        depths = paths[0, :, 1]
        assert np.all(depths <= 200.0 + 1.0)  # stays within water column

    def test_sofar_channel(self, gradient_ap):
        depths = np.linspace(0, 200, 100)
        ssp = gradient_ap.sound_speed_profile(depths)
        min_c_depth = depths[np.argmin(ssp)]
        angles = np.array([-2.0, 0.0, 2.0])
        paths = gradient_ap.ray_trace(source_depth=min_c_depth, angles_deg=angles, n_steps=200, max_range=2000.0)
        for i in range(3):
            ray_depths = paths[i, :, 1]
            assert np.all(np.isfinite(ray_depths))

    def test_multiple_angles(self, gradient_ap):
        angles = np.linspace(-20.0, 20.0, 9)
        paths = gradient_ap.ray_trace(source_depth=50.0, angles_deg=angles, n_steps=100)
        assert paths.shape == (9, 100, 2)
        assert np.all(np.isfinite(paths))
