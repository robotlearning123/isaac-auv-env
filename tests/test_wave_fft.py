"""Tests for FFT-accelerated ocean wave field."""

import numpy as np
import pytest
import warp as wp

from oceanscale.fluid.wave_fft import FFT_GRID, FFTWaveField

wp.init()


@pytest.fixture
def fft_wave():
    return FFTWaveField(device="cuda:0")


class TestFFTWaveCreation:
    def test_creates(self, fft_wave):
        assert fft_wave.grid_size == FFT_GRID

    def test_tile_fft_used(self, fft_wave):
        assert fft_wave._use_tile_fft is True

    def test_surface_shape(self, fft_wave):
        fft_wave.step(0.1)
        grid = fft_wave.get_surface_grid()
        assert grid.shape == (FFT_GRID, FFT_GRID)

    def test_has_step(self, fft_wave):
        assert hasattr(fft_wave, "step")

    def test_has_velocity(self, fft_wave):
        assert hasattr(fft_wave, "get_velocity_at")


class TestFFTWaveStep:
    def test_step_advances_time(self, fft_wave):
        assert fft_wave.time == 0.0
        fft_wave.step(0.5)
        assert abs(fft_wave.time - 0.5) < 1e-6

    def test_surface_nonzero_after_step(self, fft_wave):
        fft_wave.step(1.0)
        g = fft_wave.get_surface_grid().numpy()
        assert g.std() > 0.001

    def test_surface_finite(self, fft_wave):
        fft_wave.step(1.0)
        g = fft_wave.get_surface_grid().numpy()
        assert np.all(np.isfinite(g))

    def test_surface_changes_with_time(self, fft_wave):
        fft_wave.step(0.5)
        g1 = fft_wave.get_surface_grid().numpy().copy()
        fft_wave.step(0.5)
        g2 = fft_wave.get_surface_grid().numpy()
        assert not np.allclose(g1, g2)


class TestFFTVelocity:
    def test_velocity_shape(self, fft_wave):
        fft_wave.step(0.1)
        pos = wp.array([[250.0, 250.0, -5.0]], dtype=wp.vec3, device="cuda:0")
        vel = fft_wave.get_velocity_at(pos)
        assert vel.shape[0] == 1

    def test_velocity_finite(self, fft_wave):
        fft_wave.step(1.0)
        pos = wp.array(
            [[100.0, 100.0, -5.0], [200.0, 300.0, -10.0]],
            dtype=wp.vec3, device="cuda:0",
        )
        vel = fft_wave.get_velocity_at(pos)
        assert np.all(np.isfinite(vel.numpy()))

    def test_velocity_decays_with_depth(self, fft_wave):
        fft_wave.step(1.0)
        shallow = wp.array([[250.0, 250.0, -2.0]], dtype=wp.vec3, device="cuda:0")
        deep = wp.array([[250.0, 250.0, -30.0]], dtype=wp.vec3, device="cuda:0")
        v_shallow = np.linalg.norm(fft_wave.get_velocity_at(shallow).numpy())
        v_deep = np.linalg.norm(fft_wave.get_velocity_at(deep).numpy())
        assert v_shallow > v_deep or v_shallow < 1e-8


class TestFFTPhysics:
    def test_wave_height_physical(self, fft_wave):
        fft_wave.step(1.0)
        g = fft_wave.get_surface_grid().numpy()
        hs_approx = 4.0 * g.std()
        assert hs_approx > 0.01
        assert hs_approx < 10.0

    def test_different_wave_height(self):
        wf1 = FFTWaveField(wave_height=0.5, device="cuda:0")
        wf2 = FFTWaveField(wave_height=3.0, device="cuda:0")
        wf1.step(1.0)
        wf2.step(1.0)
        std1 = wf1.get_surface_grid().numpy().std()
        std2 = wf2.get_surface_grid().numpy().std()
        assert std2 > std1
