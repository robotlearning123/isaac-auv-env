"""Tests for the GPU underwater camera sensor."""

import numpy as np
import pytest
import warp as wp

from oceanscale.sensors.underwater_camera import CameraConfig, UnderwaterCamera

wp.init()

# Quaternion [x,y,z,w] for 180° pitch (look down from +z default).
LOOK_DOWN = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def seabed():
    verts = np.array([
        [-50, -50, -30], [50, -50, -30], [-50, 50, -30], [50, 50, -30],
    ], dtype=np.float32)
    idx = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(idx, dtype=wp.int32, device="cuda:0"),
    )


class TestCamera:
    def test_default_config(self, seabed):
        cam = UnderwaterCamera(seabed)
        assert cam.cfg.width == 640
        assert cam.cfg.height == 480

    def test_scan_keys(self, seabed):
        cam = UnderwaterCamera(seabed)
        out = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        assert "rgb" in out and "depth" in out

    def test_rgb_shape(self, seabed):
        cam = UnderwaterCamera(seabed)
        out = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        assert out["rgb"].shape == (480, 640, 3)

    def test_depth_shape(self, seabed):
        cam = UnderwaterCamera(seabed)
        out = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        assert out["depth"].shape == (480, 640)

    def test_all_finite(self, seabed):
        cam = UnderwaterCamera(seabed)
        out = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        assert np.all(np.isfinite(out["depth"]))

    def test_depth_increases_with_distance(self, seabed):
        cfg = CameraConfig(width=16, height=12)
        cam = UnderwaterCamera(seabed, cfg)
        # z=-25: 5m above seabed → small depth; z=-5: 25m above → large depth
        close = cam.scan(np.array([0, 0, -25.0]), LOOK_DOWN)
        distant = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        assert np.mean(distant["depth"]) > np.mean(close["depth"])

    def test_color_attenuates_with_depth(self, seabed):
        cfg = CameraConfig(width=16, height=12)
        cam = UnderwaterCamera(seabed, cfg)
        close = cam.scan(np.array([0, 0, -25.0]), LOOK_DOWN)
        distant = cam.scan(np.array([0, 0, -5.0]), LOOK_DOWN)
        c = close["rgb"].shape[0] // 2, close["rgb"].shape[1] // 2
        # Closer to seabed → less attenuation → brighter
        assert float(np.mean(close["rgb"][c])) > float(np.mean(distant["rgb"][c]))

    def test_fov_affects_coverage(self, seabed):
        narrow = UnderwaterCamera(seabed, CameraConfig(width=16, height=12, fov_h_deg=20, fov_v_deg=15))
        wide = UnderwaterCamera(seabed, CameraConfig(width=16, height=12, fov_h_deg=120, fov_v_deg=90))
        pos = np.array([0, 0, -5.0])
        d_n = narrow.scan(pos, LOOK_DOWN)["depth"]
        d_w = wide.scan(pos, LOOK_DOWN)["depth"]
        # Wider FOV → edge rays travel further to hit flat seabed
        assert np.mean(d_w) > np.mean(d_n)
