"""Tests for OceanWaveField — Airy wave theory on Warp GPU kernels."""

import math

import numpy as np
import pytest
import warp as wp

from oceanscale.fluid.wave import OceanWaveField

wp.init()

DEVICE = "cuda"


def test_wave_field_creation():
    """Construct with default and custom parameters."""
    wf = OceanWaveField(device=DEVICE)
    assert wf.hs == 1.0
    assert wf.tp == 8.0
    assert wf.n_components == 16
    assert wf._amplitudes.shape[0] == 16

    wf2 = OceanWaveField(
        wave_height=2.5,
        wave_period=10.0,
        wave_direction=math.pi / 4,
        water_depth=30.0,
        n_components=32,
        spectrum="pierson_moskowitz",
        device=DEVICE,
    )
    assert wf2.hs == 2.5
    assert wf2.n_components == 32
    assert wf2.spectrum_type == "pierson_moskowitz"


def test_surface_elevation_range():
    """Surface elevation should stay within physical bounds (±~2*Hs)."""
    hs = 1.0
    wf = OceanWaveField(wave_height=hs, n_components=32, device=DEVICE)

    n_pts = 200
    x = wp.array(np.linspace(0, 100, n_pts, dtype=np.float32), dtype=wp.float32, device=DEVICE)
    y = wp.zeros(n_pts, dtype=wp.float32, device=DEVICE)

    eta = wf.get_surface_elevation(x, y, time=0.0)
    eta_np = eta.numpy()

    assert np.all(np.isfinite(eta_np))
    assert np.max(np.abs(eta_np)) < 3.0 * hs


def test_velocity_depth_decay():
    """Wave-induced velocity must decrease with depth (Airy theory)."""
    wf = OceanWaveField(wave_height=2.0, wave_period=8.0, water_depth=50.0, device=DEVICE)

    surface_pts = wp.array(
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3, device=DEVICE,
    )
    deep_pts = wp.array(
        np.array([[10.0, 0.0, -25.0]], dtype=np.float32),
        dtype=wp.vec3, device=DEVICE,
    )

    v_surface = wf.get_velocity_at(surface_pts, time=1.0).numpy()
    v_deep = wf.get_velocity_at(deep_pts, time=1.0).numpy()

    speed_surface = np.linalg.norm(v_surface[0])
    speed_deep = np.linalg.norm(v_deep[0])

    assert speed_deep < speed_surface, (
        f"Deep velocity ({speed_deep:.4f}) should be less than surface ({speed_surface:.4f})"
    )


def test_pressure_depth_decay():
    """Dynamic pressure from waves should also decay with depth."""
    wf = OceanWaveField(wave_height=2.0, wave_period=8.0, water_depth=50.0, device=DEVICE)

    surface_pts = wp.array(
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3, device=DEVICE,
    )
    deep_pts = wp.array(
        np.array([[10.0, 0.0, -25.0]], dtype=np.float32),
        dtype=wp.vec3, device=DEVICE,
    )

    p_surface = wf.get_pressure_at(surface_pts, time=1.0).numpy()
    p_deep = wf.get_pressure_at(deep_pts, time=1.0).numpy()

    assert abs(p_deep[0]) < abs(p_surface[0]), (
        f"Deep pressure ({abs(p_deep[0]):.2f}) should be less than surface ({abs(p_surface[0]):.2f})"
    )


def test_jonswap_vs_pm():
    """JONSWAP and Pierson-Moskowitz should produce different amplitude distributions."""
    js = OceanWaveField(spectrum="jonswap", n_components=16, device=DEVICE)
    pm = OceanWaveField(spectrum="pierson_moskowitz", n_components=16, device=DEVICE)

    a_js = js._amplitudes.numpy()
    a_pm = pm._amplitudes.numpy()

    assert not np.allclose(a_js, a_pm, atol=1e-6), "JONSWAP and PM should differ"


def test_velocity_at_gpu():
    """Batch GPU velocity computation should return finite values."""
    wf = OceanWaveField(wave_height=1.5, device=DEVICE)

    n = 1000
    rng = np.random.default_rng(123)
    pts = rng.uniform([-50, -50, -30], [50, 50, 0], size=(n, 3)).astype(np.float32)

    positions = wp.array(pts, dtype=wp.vec3, device=DEVICE)
    vel = wf.get_velocity_at(positions, time=5.0)

    vel_np = vel.numpy()
    assert vel_np.shape == (n, 3)
    assert np.all(np.isfinite(vel_np))
    assert np.any(np.abs(vel_np) > 1e-6), "Should have non-zero velocities"


def test_zero_wave_height():
    """Hs=0 means zero velocity and zero elevation everywhere."""
    wf = OceanWaveField(wave_height=0.0, device=DEVICE)

    pts = wp.array(
        np.array([[5.0, 5.0, -2.0], [10.0, 0.0, -10.0]], dtype=np.float32),
        dtype=wp.vec3, device=DEVICE,
    )
    vel = wf.get_velocity_at(pts, time=1.0).numpy()
    assert np.allclose(vel, 0.0, atol=1e-10)

    x = wp.array(np.array([0.0, 5.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
    y = wp.array(np.array([0.0, 5.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
    eta = wf.get_surface_elevation(x, y, time=1.0).numpy()
    assert np.allclose(eta, 0.0, atol=1e-10)


def test_step_advances_time():
    """step() should advance internal time."""
    wf = OceanWaveField(device=DEVICE)
    assert wf.time == 0.0
    wf.step(dt=0.1)
    assert abs(wf.time - 0.1) < 1e-9
    wf.step(dt=0.05)
    assert abs(wf.time - 0.15) < 1e-9


def test_surface_varies_with_time():
    """Surface elevation should change over time."""
    wf = OceanWaveField(wave_height=1.0, device=DEVICE)
    x = wp.array(np.array([10.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)
    y = wp.array(np.array([0.0], dtype=np.float32), dtype=wp.float32, device=DEVICE)

    eta_t0 = wf.get_surface_elevation(x, y, time=0.0).numpy()[0]
    eta_t1 = wf.get_surface_elevation(x, y, time=2.0).numpy()[0]
    assert eta_t0 != pytest.approx(eta_t1, abs=1e-6), "Surface should vary with time"
