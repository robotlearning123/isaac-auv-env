"""Tests for multibeam echo sounder."""

import numpy as np
import warp as wp

wp.init()

from oceanscale.sensors.multibeam import MultibeamConfig, MultibeamSonar


def _make_mesh(device="cuda:0"):
    verts = np.array([[-50, -50, -10], [50, -50, -10], [50, 50, -10], [-50, 50, -10]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3f, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )


def test_mbes_creates():
    mesh = _make_mesh()
    mbes = MultibeamSonar(mesh)
    assert mbes.cfg.n_beams == 256


def test_mbes_scan_returns_dict():
    mesh = _make_mesh()
    mbes = MultibeamSonar(mesh, MultibeamConfig(n_beams=64, max_range=50.0))
    result = mbes.scan(np.array([0.0, 0.0, -5.0]))
    assert "ranges" in result
    assert "intensity" in result
    assert "points" in result


def test_mbes_ranges_shape():
    mesh = _make_mesh()
    cfg = MultibeamConfig(n_beams=128)
    mbes = MultibeamSonar(mesh, cfg)
    result = mbes.scan(np.array([0.0, 0.0, -5.0]))
    assert result["ranges"].shape == (128,)
    assert result["intensity"].shape == (128,)
    assert result["points"].shape == (128, 3)


def test_mbes_detects_seabed():
    mesh = _make_mesh()
    mbes = MultibeamSonar(mesh, MultibeamConfig(n_beams=64, max_range=50.0))
    result = mbes.scan(np.array([0.0, 0.0, -5.0]))
    center_beams = result["ranges"][28:36]
    assert np.all(center_beams < 10.0), "center beams should detect seabed at ~5m below"


def test_mbes_points_below_sensor():
    mesh = _make_mesh()
    mbes = MultibeamSonar(mesh, MultibeamConfig(n_beams=64, max_range=50.0))
    result = mbes.scan(np.array([0.0, 0.0, -5.0]))
    hit_mask = result["ranges"] < 50.0
    if hit_mask.any():
        hit_points = result["points"][hit_mask]
        assert np.all(hit_points[:, 2] <= -5.0), "hit points should be below sensor"


def test_mbes_with_orientation():
    mesh = _make_mesh()
    mbes = MultibeamSonar(mesh, MultibeamConfig(n_beams=32))
    quat = np.array([0, 0, 0, 1], dtype=np.float32)
    result = mbes.scan(np.array([0, 0, -5.0]), orientation=quat)
    assert result["ranges"].shape == (32,)


def test_mbes_custom_config():
    mesh = _make_mesh()
    cfg = MultibeamConfig(n_beams=512, swath_angle_deg=120.0, frequency_khz=200.0)
    mbes = MultibeamSonar(mesh, cfg)
    result = mbes.scan(np.array([0, 0, -5.0]))
    assert result["ranges"].shape == (512,)
