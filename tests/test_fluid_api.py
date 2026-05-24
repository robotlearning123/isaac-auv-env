"""Test the fluid fidelity ladder API."""

import pytest

from oceanscale.fluid import FluidLevel, create_fluid_solver


def test_fluid_level_enum():
    assert FluidLevel.NONE == 0
    assert FluidLevel.GRID == 1
    assert FluidLevel.SPH == 2
    assert FluidLevel.MPM == 3


def test_create_none():
    solver = create_fluid_solver(FluidLevel.NONE)
    assert solver is None


def test_create_none_int():
    solver = create_fluid_solver(0)
    assert solver is None


@pytest.mark.skipif(
    not pytest.importorskip("warp", reason="warp not available"),
    reason="requires warp",
)
def test_create_grid():
    solver = create_fluid_solver(FluidLevel.GRID, grid_res=16)
    assert solver is not None
    assert hasattr(solver, "step")


@pytest.mark.skipif(
    not pytest.importorskip("warp", reason="warp not available"),
    reason="requires warp",
)
def test_create_sph():
    solver = create_fluid_solver(FluidLevel.SPH, n_particles=100)
    assert solver is not None
    assert hasattr(solver, "step")


def test_create_invalid():
    with pytest.raises(ValueError):
        create_fluid_solver(99)


def test_create_negative():
    with pytest.raises(ValueError):
        create_fluid_solver(-1)


def test_level_from_int():
    assert FluidLevel(2) == FluidLevel.SPH


def test_all_levels_defined():
    assert len(FluidLevel) == 4
