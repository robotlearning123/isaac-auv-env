"""Tests for AdaptiveFluidDomain (warp.fem Nanogrid / AdaptiveNanogrid)."""

import numpy as np
import warp as wp
import warp.fem as fem

from oceanscale.fluid.adaptive_grid import AdaptiveFluidDomain

wp.init()


class TestNanogridBasics:
    def test_nanogrid_creation(self):
        vol = wp.Volume.allocate(
            min=[0, 0, 0], max=[8, 8, 8], voxel_size=1.0, bg_value=0.0, device="cuda:0"
        )
        grid = fem.Nanogrid(vol)
        assert grid.cell_count() > 0

    def test_nanogrid_cell_count_matches_voxels(self):
        vol = wp.Volume.allocate(
            min=[0, 0, 0], max=[4, 4, 4], voxel_size=1.0, bg_value=0.0, device="cuda:0"
        )
        grid = fem.Nanogrid(vol)
        assert grid.cell_count() == vol.get_voxel_count()

    def test_nanogrid_function_space(self):
        vol = wp.Volume.allocate(
            min=[0, 0, 0], max=[4, 4, 4], voxel_size=1.0, bg_value=0.0, device="cuda:0"
        )
        grid = fem.Nanogrid(vol)
        space = fem.make_polynomial_space(grid, degree=1)
        assert space.node_count() > grid.cell_count()


class TestAdaptiveFluidDomain:
    def test_creation(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=1.0, coarse_res=4.0, fine_radius=8.0
        )
        assert domain.cell_count > 0

    def test_cell_count_exceeds_coarse_alone(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=1.0, coarse_res=4.0, fine_radius=8.0
        )
        coarse_only = domain.coarse_cell_count
        assert domain.cell_count > coarse_only

    def test_fine_has_more_cells_per_volume(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=1.0, coarse_res=4.0, fine_radius=8.0
        )
        fine_volume = (2 * 8.0) ** 3
        coarse_volume = 32.0**3
        fine_density = domain.fine_cell_count / fine_volume
        coarse_density = domain.coarse_cell_count / coarse_volume
        assert fine_density > coarse_density

    def test_function_space(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=2.0, coarse_res=4.0, fine_radius=8.0
        )
        space = domain.create_function_space(degree=1)
        assert space.node_count() > 0

    def test_function_space_caching(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=2.0, coarse_res=4.0, fine_radius=8.0
        )
        s1 = domain.create_function_space(degree=1)
        s2 = domain.create_function_space(degree=1)
        assert s1 is s2

    def test_update_focus(self):
        domain = AdaptiveFluidDomain(
            domain_size=64.0, fine_res=2.0, coarse_res=8.0, fine_radius=8.0,
            focus=(0.0, 0.0, 0.0),
        )
        domain.update_focus(np.array([20.0, 0.0, 20.0]))
        np.testing.assert_array_almost_equal(domain.focus, [20.0, 0.0, 20.0])
        assert domain.cell_count > 0

    def test_focus_property(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=2.0, coarse_res=4.0, fine_radius=8.0,
            focus=(5.0, 0.0, 5.0),
        )
        np.testing.assert_array_almost_equal(domain.focus, [5.0, 0.0, 5.0])

    def test_grid_is_adaptive_nanogrid(self):
        domain = AdaptiveFluidDomain(
            domain_size=32.0, fine_res=2.0, coarse_res=4.0, fine_radius=8.0
        )
        assert isinstance(domain.grid, fem.AdaptiveNanogrid)
