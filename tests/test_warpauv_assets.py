"""Tests for WarpAUV vehicle config and USD asset."""


import numpy as np

from oceanscale.vehicles import WarpAUV


class TestWarpAUVConfig:
    def test_default_construction(self):
        v = WarpAUV()
        assert v.mass == 22.701
        assert v.n_thrusters == 6
        assert v.volume > 0

    def test_usd_path_exists(self):
        v = WarpAUV()
        p = v.usd_path()
        assert p.exists(), f"WarpAUV USD not found at {p}"
        assert p.suffix == ".usd"

    def test_inertia_diag(self):
        v = WarpAUV()
        I = v.inertia_diag()
        assert I.shape == (3,)
        assert I.dtype == np.float32
        assert np.all(I > 0)

    def test_mujoco_drag_kwargs(self):
        v = WarpAUV()
        kw = v.mujoco_drag_kwargs()
        assert "inertia_diag" in kw
        assert "mass" in kw
        assert kw["mass"] == v.mass

    def test_thruster_positions_count(self):
        v = WarpAUV()
        assert len(v.thruster_positions) == v.n_thrusters

    def test_com_to_cob_offset(self):
        v = WarpAUV()
        assert len(v.com_to_cob_offset) == 3

    def test_usd_file_nonzero(self):
        v = WarpAUV()
        p = v.usd_path()
        assert p.stat().st_size > 1_000_000, "USD file too small"
