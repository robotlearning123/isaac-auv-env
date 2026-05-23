"""Von Benzon reference trajectory regression test.

Runs the CPU 6-DOF model with a 5N forward surge step (t=2-7 s) and
compares the final position against a frozen self-generated snapshot.
NOT a parity test against Simulink or any external ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_PATH = DATA_DIR / "vonbenzon_expected.npz"


def _surge_step(t: float) -> np.ndarray:
    if 2.0 <= t < 7.0:
        return np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.zeros(6)


@pytest.fixture(scope="module")
def traj():
    from oceanscale.validation.vonbenzon_reference import VonBenzonReferenceModel

    model = VonBenzonReferenceModel()
    return model.generate_trajectory(thrust_func=_surge_step)


@pytest.mark.slow
def test_shapes(traj):
    assert traj["t"].shape == (1000,)
    assert traj["pos"].shape == (1000, 3)
    assert traj["quat"].shape == (1000, 4)
    assert traj["vel"].shape == (1000, 6)
    assert traj["omega"].shape == (1000, 3)


@pytest.mark.slow
def test_no_nan(traj):
    for key, arr in traj.items():
        assert np.all(np.isfinite(arr)), f"{key} contains NaN/Inf"


@pytest.mark.slow
def test_quaternion_unit_norm(traj):
    norms = np.linalg.norm(traj["quat"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


@pytest.mark.slow
def test_time_monotonic(traj):
    assert np.all(np.diff(traj["t"]) > 0)


@pytest.mark.slow
def test_final_position_regression(traj):
    expected = np.load(EXPECTED_PATH)
    actual_pos = traj["pos"][-1]
    expected_pos = expected["final_pos"]
    err = np.linalg.norm(actual_pos - expected_pos)
    assert err < 0.05, (
        f"position L2 error {err:.6f} m > 0.05 m\n"
        f"  actual:   {actual_pos}\n"
        f"  expected: {expected_pos}"
    )


@pytest.mark.slow
def test_surge_dominant_motion(traj):
    final = traj["pos"][-1]
    assert abs(final[0]) > abs(final[1]), "surge should dominate sway"
    assert abs(final[0]) > abs(final[2]), "surge should dominate heave"
