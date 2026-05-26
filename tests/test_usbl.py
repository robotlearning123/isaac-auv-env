"""Tests for USBL acoustic positioning sensor."""

import math

import numpy as np

from oceanscale.sensors.usbl import USBL, USBLConfig


def test_usbl_creates():
    usbl = USBL()
    assert usbl.cfg.max_range == 3000.0


def test_usbl_add_transponder():
    usbl = USBL()
    usbl.add_transponder(np.array([10, 0, -5]), transponder_id=1)
    assert len(usbl.transponders) == 1


def test_usbl_measure_range():
    usbl = USBL(USBLConfig(range_accuracy_pct=0.0, bearing_accuracy_deg=0.0, noise_seed=42))
    usbl.add_transponder(np.array([10, 0, 0], dtype=np.float32))
    result = usbl.measure(np.array([0, 0, 0], dtype=np.float32))
    assert len(result) == 1
    assert result[0]["valid"]
    assert abs(result[0]["range_m"] - 10.0) < 0.01


def test_usbl_measure_bearing():
    usbl = USBL(USBLConfig(range_accuracy_pct=0.0, bearing_accuracy_deg=0.0, noise_seed=42))
    usbl.add_transponder(np.array([0, 10, 0], dtype=np.float32))
    result = usbl.measure(np.array([0, 0, 0], dtype=np.float32))
    assert abs(result[0]["bearing_rad"] - math.pi / 2) < 0.01


def test_usbl_out_of_range():
    usbl = USBL(USBLConfig(max_range=100.0))
    usbl.add_transponder(np.array([200, 0, 0], dtype=np.float32))
    result = usbl.measure(np.array([0, 0, 0], dtype=np.float32))
    assert not result[0]["valid"]


def test_usbl_multiple_transponders():
    usbl = USBL(USBLConfig(noise_seed=42))
    usbl.add_transponder(np.array([10, 0, 0]), transponder_id=1)
    usbl.add_transponder(np.array([0, 20, 0]), transponder_id=2)
    usbl.add_transponder(np.array([0, 0, -30]), transponder_id=3)
    result = usbl.measure(np.array([0, 0, 0]))
    assert len(result) == 3
    assert all(r["valid"] for r in result)


def test_usbl_noise_nonzero():
    usbl = USBL(USBLConfig(range_accuracy_pct=1.0, bearing_accuracy_deg=1.0, noise_seed=42))
    usbl.add_transponder(np.array([100, 0, 0], dtype=np.float32))
    measurements = [usbl.measure(np.array([0, 0, 0], dtype=np.float32))[0]["range_m"] for _ in range(10)]
    assert np.std(measurements) > 0.01, "noise should cause variation"


def test_usbl_elevation():
    usbl = USBL(USBLConfig(range_accuracy_pct=0.0, bearing_accuracy_deg=0.0, noise_seed=42))
    usbl.add_transponder(np.array([0, 0, -100], dtype=np.float32))
    result = usbl.measure(np.array([0, 0, 0], dtype=np.float32))
    assert abs(result[0]["elevation_rad"] - (-math.pi / 2)) < 0.01
