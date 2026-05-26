"""Tests for side-scan sonar sensor."""

import numpy as np
import warp as wp

wp.init()

from oceanscale.sensors.sidescan_sonar import SideScanSonar, SideScanSonarConfig


def _make_mesh(device="cuda:0"):
    verts = np.array([[-50, -50, -10], [50, -50, -10], [50, 50, -10], [-50, 50, -10]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3f, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )


def test_sss_creates():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh)
    assert sss.n_total_rays > 0
    assert sss.image_width == sss.n_range_bins * 2


def test_sss_ping_produces_image():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=20.0, n_beams_per_side=32))
    img = sss.ping(np.array([0.0, 0.0, -5.0]))
    assert img.ndim == 2
    assert img.shape[0] == 1
    assert img.shape[1] == sss.image_width


def test_sss_accumulates_pings():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=20.0, n_beams_per_side=32))
    for i in range(5):
        img = sss.ping(np.array([float(i), 0.0, -5.0]))
    assert img.shape[0] == 5


def test_sss_detects_seabed():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=20.0, n_beams_per_side=64))
    img = sss.ping(np.array([0.0, 0.0, -5.0]))
    assert img.max() > 0.0, "should detect seabed below"


def test_sss_no_detection_above():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=5.0, n_beams_per_side=32))
    img = sss.ping(np.array([0.0, 0.0, -20.0]))
    assert img.max() < 0.3, "seabed at -10, sensor at -20 looking down should not detect much"


def test_sss_reset():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=20.0, n_beams_per_side=32))
    sss.ping(np.array([0, 0, -5.0]))
    sss.ping(np.array([1, 0, -5.0]))
    sss.reset()
    img = sss.ping(np.array([2, 0, -5.0]))
    assert img.shape[0] == 1


def test_sss_with_orientation():
    mesh = _make_mesh()
    sss = SideScanSonar(mesh, SideScanSonarConfig(max_range=20.0, n_beams_per_side=32))
    quat = np.array([0, 0, 0, 1], dtype=np.float32)
    img = sss.ping(np.array([0, 0, -5.0]), orientation=quat)
    assert img.ndim == 2
