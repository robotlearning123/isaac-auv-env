"""Tier-1 Fossen parity tests against von Benzon 2022 reference trajectories.

How to enable these tests:
    1. Generate reference trajectories from the von Benzon 2022 Simulink
       model (or the Python port once complete).
    2. Save as ``tests/data/vonbenzon_ref/<scenario>.npz`` with keys:
       ``'pose'`` (N,6), ``'velocity'`` (N,6), ``'time'`` (N,).
    3. Remove the ``@pytest.mark.skip`` decorators.

See: REF-VONBENZON22 in REFERENCES.md §6, R13 + R23 in RISKS.md.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# Defer GPU imports — these tests may run on CPU-only CI where Warp is absent.
pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("warp"),
    reason="Warp not installed — Tier-1 tests require CUDA + Warp",
)

REF_DATA_DIR = Path(__file__).parent / "data" / "vonbenzon_ref"


@pytest.mark.skip(reason="requires von Benzon reference dataset — see R13 in RISKS.md")
def test_terminal_velocity_matches_von_benzon() -> None:
    """Compare steady-state surge velocity under constant forward thrust
    against the von Benzon 2022 reference.

    Expected tolerance: <5% relative error in terminal velocity.
    Reference: RISKS.md R13 trigger condition.
    """
    from oceanscale.hydro.tier1 import Tier1
    from oceanscale.validation.vonbenzon_reference import VonBenzonReferenceModel

    ref_model = VonBenzonReferenceModel()
    ref_path = REF_DATA_DIR / "const_thrust_surge.npz"
    ref_data = ref_model.load_reference_dataset(ref_path)

    # TODO: instantiate Tier1 with von Benzon coefficients, run same
    # thrust profile, compare terminal velocity against ref_data.

    tier1 = Tier1(n_envs=1, n_thrusters=8, device="cuda")
    # tier1.set_coeffs(added_mass=..., d_lin=..., d_quad=..., ...)
    # ... run simulation ...
    # assert np.allclose(terminal_vel, ref_terminal_vel, rtol=0.05)


@pytest.mark.skip(reason="requires von Benzon reference dataset — see R13 in RISKS.md")
def test_step_response_under_constant_thrust() -> None:
    """Compare full 6-DOF trajectory under constant thrust against von
    Benzon 2022 reference.

    Expected tolerance: <5% relative error per-DOF at each timestep,
    measured over the transient response (0–5 s).
    """
    from oceanscale.hydro.tier1 import Tier1
    from oceanscale.validation.vonbenzon_reference import VonBenzonReferenceModel

    ref_model = VonBenzonReferenceModel()
    ref_path = REF_DATA_DIR / "const_thrust_6dof.npz"
    ref_data = ref_model.load_reference_dataset(ref_path)

    tier1 = Tier1(n_envs=1, n_thrusters=8, device="cuda")
    # tier1.set_coeffs(added_mass=..., d_lin=..., d_quad=..., ...)
    # ... run simulation ...
    # assert per-DOF error within tolerance
