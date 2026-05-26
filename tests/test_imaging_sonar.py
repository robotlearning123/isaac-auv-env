"""Tests for the GPU imaging sonar sensor."""

import numpy as np
import pytest
import warp as wp

from oceanscale.sensors.imaging_sonar import ImagingSonar, ImagingSonarConfig

wp.init()

DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


def _make_box_mesh(
    center: tuple[float, float, float] = (3.0, 0.0, 0.0),
    half: float = 1.0,
    device: str = DEVICE,
) -> wp.Mesh:
    cx, cy, cz = center
    verts = np.array(
        [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ],
        dtype=np.int32,
    )
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3f, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )


class TestImagingSonarConfig:
    def test_default_oculus_m370(self):
        cfg = ImagingSonarConfig()
        assert cfg.min_range == pytest.approx(0.2)
        assert cfg.max_range == pytest.approx(3.0)
        assert cfg.hfov_deg == pytest.approx(130.0)
        assert cfg.vfov_deg == pytest.approx(20.0)

    def test_custom_config(self):
        cfg = ImagingSonarConfig(max_range=10.0, hfov_deg=90.0, angular_res_deg=1.0)
        assert cfg.max_range == 10.0
        assert cfg.hfov_deg == 90.0


class TestImagingSonarDimensions:
    def test_image_shape_matches_config(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.1, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        expected_range_bins = int((5.0 - 0.2) / 0.1)
        expected_bearing_bins = int(90.0 / 1.0)
        assert sonar.image_shape == (expected_range_bins, expected_bearing_bins)

    def test_scan_output_shape(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.1, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        image = sonar.scan(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert image.shape == sonar.image_shape

    def test_n_total_rays(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(hfov_deg=90.0, angular_res_deg=1.0, n_elevation_rays=3)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)
        assert sonar.n_total_rays == 90 * 3


class TestImagingSonarRangeBins:
    def test_range_bin_count(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(min_range=0.5, max_range=5.5, range_res=0.05)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)
        assert sonar.n_range_bins == 100

    def test_bearing_bin_count(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(hfov_deg=120.0, angular_res_deg=0.5)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)
        assert sonar.n_bearing_bins == 240


class TestImagingSonarFOV:
    def test_fov_coverage(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(hfov_deg=130.0, angular_res_deg=0.5)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)
        assert sonar.n_bearing_bins == 260


class TestImagingSonarDetection:
    def test_detects_box_target(self):
        mesh = _make_box_mesh(center=(3.0, 0.0, 0.0), half=1.0)
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.05, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        image = sonar.scan(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert image.max() > 0.0, "Sonar should detect the box target"

    def test_no_detection_behind(self):
        mesh = _make_box_mesh(center=(3.0, 0.0, 0.0), half=1.0)
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.05, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        image = sonar.scan(np.array([10.0, 0.0, 0.0], dtype=np.float32))
        assert image.max() < 0.3, "Sonar should see little behind the box"

    def test_image_values_clamped(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.05, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        image = sonar.scan(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_sequential_scans_independent(self):
        mesh = _make_box_mesh()
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.05, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        img1 = sonar.scan(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        img2 = sonar.scan(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert img1.shape == img2.shape

    def test_with_orientation(self):
        mesh = _make_box_mesh(center=(3.0, 0.0, 0.0))
        cfg = ImagingSonarConfig(max_range=5.0, range_res=0.05, hfov_deg=90.0, angular_res_deg=1.0)
        sonar = ImagingSonar(mesh, config=cfg, device=DEVICE)

        identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        image = sonar.scan(np.zeros(3, dtype=np.float32), orientation=identity_quat)
        assert image.shape == sonar.image_shape
        assert image.max() > 0.0
