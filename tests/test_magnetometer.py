"""Tests for magnetometer sensor."""

import math

import numpy as np

from oceanscale.sensors.magnetometer import Magnetometer, MagnetometerConfig


def test_mag_creates():
    mag = Magnetometer()
    assert mag.cfg.earth_field_uT == 50.0


def test_mag_measure_identity():
    mag = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, noise_seed=42))
    m = mag.measure()
    assert m.shape == (3,)
    field_magnitude = np.linalg.norm(m)
    assert abs(field_magnitude - 50.0) < 1.0


def test_mag_field_magnitude_preserved():
    mag = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, noise_seed=42))
    quat = np.array([0.0, 0.0, 0.383, 0.924], dtype=np.float32)  # 45° yaw
    m = mag.measure(orientation=quat)
    assert abs(np.linalg.norm(m) - 50.0) < 1.0


def test_mag_heading_north():
    mag = Magnetometer(MagnetometerConfig(declination_deg=0.0, noise_std_uT=0.0, noise_seed=42))
    heading = mag.heading_deg()
    assert abs(heading) < 5.0 or abs(heading - 360.0) < 5.0


def test_mag_heading_changes_with_yaw():
    mag = Magnetometer(MagnetometerConfig(declination_deg=0.0, noise_std_uT=0.0, noise_seed=42))
    h1 = mag.heading_deg(orientation=np.array([0, 0, 0, 1], dtype=np.float32))
    h2 = mag.heading_deg(orientation=np.array([0, 0, 0.383, 0.924], dtype=np.float32))
    assert abs(h1 - h2) > 10.0, "heading should change with rotation"


def test_mag_noise_nonzero():
    mag = Magnetometer(MagnetometerConfig(noise_std_uT=2.0, noise_seed=42))
    measurements = [mag.measure()[0] for _ in range(20)]
    assert np.std(measurements) > 0.1


def test_mag_hard_iron():
    offset = np.array([5.0, -3.0, 2.0], dtype=np.float32)
    mag_clean = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, noise_seed=42))
    mag_hi = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, hard_iron_offset=offset, noise_seed=42))
    m_clean = mag_clean.measure()
    m_hi = mag_hi.measure()
    diff = m_hi - m_clean
    np.testing.assert_allclose(diff, offset, atol=0.01)


def test_mag_soft_iron():
    scale = np.diag([1.1, 0.9, 1.0]).astype(np.float32)
    mag = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, soft_iron_matrix=scale, noise_seed=42))
    m = mag.measure()
    mag_clean = Magnetometer(MagnetometerConfig(noise_std_uT=0.0, noise_seed=42))
    m_clean = mag_clean.measure()
    assert abs(m[0] - m_clean[0] * 1.1) < 0.01
    assert abs(m[1] - m_clean[1] * 0.9) < 0.01
