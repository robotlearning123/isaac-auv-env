"""Tests for oceanscale.rendering.underwater — Jerlov water type presets and GPU rendering."""

import numpy as np
import pytest
import warp as wp

from oceanscale.rendering.underwater import (
    JERLOV_PRESETS,
    UnderwaterRenderer,
    WaterParams,
)

wp.init()
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"


class TestWaterParams:
    def test_all_presets_exist(self):
        expected = {"I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"}
        assert set(JERLOV_PRESETS.keys()) == expected

    def test_params_are_frozen(self):
        p = JERLOV_PRESETS["II"]
        with pytest.raises(AttributeError):
            p.atten_coeff = (0, 0, 0)  # type: ignore[misc]

    def test_attenuation_increases_with_turbidity(self):
        clear = JERLOV_PRESETS["I"]
        coastal = JERLOV_PRESETS["II"]
        turbid = JERLOV_PRESETS["9C"]
        assert sum(clear.atten_coeff) < sum(coastal.atten_coeff) < sum(turbid.atten_coeff)


class TestUnderwaterRenderer:
    def test_init_default(self):
        r = UnderwaterRenderer(device=DEVICE)
        assert r.water_type == "II"

    def test_init_all_presets(self):
        for key in JERLOV_PRESETS:
            r = UnderwaterRenderer(water_type=key, device=DEVICE)
            assert r.water_type == key

    def test_init_custom(self):
        p = WaterParams((0.1, 0.2, 0.3), (0.01, 0.02, 0.03), (0.04, 0.05, 0.06))
        r = UnderwaterRenderer(water_type="custom", params=p, device=DEVICE)
        assert r.params == p

    def test_init_custom_no_params_raises(self):
        with pytest.raises(ValueError, match="params required"):
            UnderwaterRenderer(water_type="custom", device=DEVICE)

    def test_init_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown water type"):
            UnderwaterRenderer(water_type="FAKE", device=DEVICE)

    def test_render_shape(self):
        r = UnderwaterRenderer(device=DEVICE)
        rgb = np.random.randint(0, 256, (64, 80, 3), dtype=np.uint8)
        depth = np.full((64, 80), 5.0, dtype=np.float32)
        out = r.render(rgb, depth)
        assert out.shape == (64, 80, 3)
        assert out.dtype == np.uint8

    def test_render_scalar_depth(self):
        r = UnderwaterRenderer(device=DEVICE)
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        out = r.render(rgb, np.float32(10.0))
        assert out.shape == (32, 32, 3)

    def test_attenuation_increases_with_depth(self):
        r = UnderwaterRenderer(water_type="II", device=DEVICE, caustic_intensity=0.0)
        rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
        shallow = r.render(rgb, np.float32(1.0))
        deep = r.render(rgb, np.float32(20.0))
        assert float(shallow.mean()) > float(deep.mean())

    def test_turbid_water_darker_than_clear(self):
        rgb = np.full((16, 16, 3), 200, dtype=np.uint8)
        depth = np.float32(10.0)
        clear = UnderwaterRenderer(water_type="I", device=DEVICE, caustic_intensity=0.0)
        turbid = UnderwaterRenderer(water_type="9C", device=DEVICE, caustic_intensity=0.0)
        out_clear = clear.render(rgb, depth)
        out_turbid = turbid.render(rgb, depth)
        assert float(out_clear.mean()) > float(out_turbid.mean())

    def test_caustics_change_output(self):
        r_no = UnderwaterRenderer(device=DEVICE, caustic_intensity=0.0)
        r_yes = UnderwaterRenderer(device=DEVICE, caustic_intensity=1.0)
        rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
        depth = np.float32(5.0)
        out_no = r_no.render(rgb, depth, time=1.0)
        out_yes = r_yes.render(rgb, depth, time=1.0)
        assert not np.array_equal(out_no, out_yes)

    def test_render_gpu_returns_warp_array(self):
        r = UnderwaterRenderer(device=DEVICE)
        rgb = wp.array(
            np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8),
            dtype=wp.uint8, device=DEVICE,
        )
        depth = wp.array(
            np.full((16, 16), 5.0, dtype=np.float32),
            dtype=wp.float32, device=DEVICE,
        )
        out = r.render_gpu(rgb, depth)
        assert isinstance(out, wp.array)
        assert out.shape == (16, 16, 3)

    def test_zero_depth_preserves_color(self):
        r = UnderwaterRenderer(water_type="I", device=DEVICE, caustic_intensity=0.0)
        rgb = np.full((8, 8, 3), 180, dtype=np.uint8)
        out = r.render(rgb, np.float32(0.0))
        np.testing.assert_allclose(out.astype(float), 180.0, atol=2.0)


class TestVideoExporterImport:
    def test_import_from_rendering(self):
        from oceanscale.rendering import VideoExporter
        assert VideoExporter is not None

    def test_import_from_rendering_underwater(self):
        from oceanscale.rendering import UnderwaterRenderer
        assert UnderwaterRenderer is not None
