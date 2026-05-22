"""Test wrench→thruster allocation via T_matrix pseudoinverse."""

import numpy as np
import pytest

from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


@pytest.fixture
def t_pinv():
    v = BlueROV2Heavy()
    T = np.array(v.compute_t_matrix(), dtype=np.float64)
    return np.linalg.pinv(T)


def _allocate(t_pinv, wrench):
    return np.clip(wrench @ t_pinv.T, -1.0, 1.0).astype(np.float32)


def test_pure_surge_horizontal_only(t_pinv):
    u = _allocate(t_pinv, np.array([1, 0, 0, 0, 0, 0]))
    T = np.array(BlueROV2Heavy().compute_t_matrix())
    wrench = T @ u
    assert abs(wrench[2]) < 0.05, f"Vertical force from surge: {wrench[2]}"
    assert np.any(np.abs(u[:4]) > 0.01), "Horizontal thrusters T1-T4 should be active"


def test_pure_heave_vertical_only(t_pinv):
    u = _allocate(t_pinv, np.array([0, 0, 1, 0, 0, 0]))
    T = np.array(BlueROV2Heavy().compute_t_matrix())
    wrench = T @ u
    assert abs(wrench[0]) < 0.05, f"Surge force from heave: {wrench[0]}"
    assert abs(wrench[1]) < 0.05, f"Sway force from heave: {wrench[1]}"
    assert np.all(np.abs(u[4:8]) > 0.01), "All vertical thrusters T5-T8 should be active"


def test_pure_yaw_opposing_horizontal(t_pinv):
    u = _allocate(t_pinv, np.array([0, 0, 0, 0, 0, 1]))
    T = np.array(BlueROV2Heavy().compute_t_matrix())
    wrench = T @ u
    assert wrench[5] > 0.5, f"Yaw torque too small: {wrench[5]}"
    horiz = u[:4]
    has_pos = np.any(horiz > 0.01)
    has_neg = np.any(horiz < -0.01)
    assert has_pos and has_neg, "Opposing horizontal thrusters should have opposite signs"


def test_zero_action_all_near_zero(t_pinv):
    u = _allocate(t_pinv, np.array([0, 0, 0, 0, 0, 0]))
    assert np.allclose(u, 0.0, atol=1e-6), f"Zero action should give zero thrust: {u}"


def test_all_thrusters_used(t_pinv):
    dof_axes = np.eye(6)
    all_u = np.stack([_allocate(t_pinv, dof_axes[i]) for i in range(6)])
    for k in range(8):
        assert np.any(np.abs(all_u[:, k]) > 1e-4), f"Thruster {k} never used"
