"""Parameter fidelity guardrails — ROVEnv vs von Benzon 2022 reference.

These tests assert that the GPU path (ROVEnv + Tier1 + BlueROV2Heavy) uses
parameters consistent with the CPU reference model (VonBenzonReferenceModel).
Known mismatches are marked xfail with a description of the divergence.

Sources:
  - von Benzon et al. 2022, JMSE 10(12):1898, Table A1
  - oceanscale/validation/vonbenzon_reference.py (CPU baseline)
  - oceanscale/vehicles/bluerov2.py (vehicle definition)
  - oceanscale/hydro/tier1.py (GPU hydro)
  - oceanscale/rov_env.py (env glue)
  - install_log/fidelity_param_analysis.md (mismatch catalog)
"""

from __future__ import annotations

import math

import pytest

from oceanscale.hydro.tier1 import DEFAULT_RHO, Tier1
from oceanscale.validation.vonbenzon_reference import VonBenzonParams
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


# ---------------------------------------------------------------------------
# Reference instances
# ---------------------------------------------------------------------------

VON_BENZON = VonBenzonParams()
BLUEROV2 = BlueROV2Heavy()

RTOL = 1e-6  # exact-match tolerance (same source data)


# ---------------------------------------------------------------------------
# 1. Inertia tensor
# ---------------------------------------------------------------------------


class TestInertiaMatchesBlueROV2:
    """ROVEnv uses a sphere shape → Newton derives inertia from geometry.

    The sphere radius is sized from I_max so I_z matches von Benzon,
    but I_x and I_y will be equal to I_z (isotropic sphere) rather than
    the correct anisotropic values [0.26, 0.23, 0.37].

    See fidelity_param_analysis.md BUG 1 (CRITICAL).
    """

    @pytest.mark.xfail(
        reason="Sphere gives isotropic I=[0.37,0.37,0.37]; von Benzon needs "
               "[0.26, 0.23, 0.37]. ROVEnv._build() sizes sphere from I_max "
               "so I_z matches but I_x,I_y are wrong.",
        strict=True,
    )
    def test_ix_matches(self) -> None:
        """Compare actual Newton sphere inertia, not the dataclass value."""
        mass = BLUEROV2.mass
        I_max = max(BLUEROV2.Ix, BLUEROV2.Iy, BLUEROV2.Iz)
        radius = (5.0 * I_max / (2.0 * mass)) ** 0.5
        I_sphere = 2.0 / 5.0 * mass * radius**2
        assert I_sphere == pytest.approx(VON_BENZON.I_x, rel=0.01)

    @pytest.mark.xfail(
        reason="Same as test_ix — isotropic sphere inflates I_y to I_max.",
        strict=True,
    )
    def test_iy_matches(self) -> None:
        """Compare actual Newton sphere inertia, not the dataclass value."""
        mass = BLUEROV2.mass
        I_max = max(BLUEROV2.Ix, BLUEROV2.Iy, BLUEROV2.Iz)
        radius = (5.0 * I_max / (2.0 * mass)) ** 0.5
        I_sphere = 2.0 / 5.0 * mass * radius**2
        assert I_sphere == pytest.approx(VON_BENZON.I_y, rel=0.01)

    def test_iz_matches(self) -> None:
        """I_z = I_max, which the sphere radius is derived from."""
        assert BLUEROV2.Iz == pytest.approx(VON_BENZON.I_z, rel=RTOL)

    def test_sphere_radius_derivation(self) -> None:
        """Verify the sphere radius formula used in rov_env.py:_build()."""
        mass = BLUEROV2.mass
        I_max = max(BLUEROV2.Ix, BLUEROV2.Iy, BLUEROV2.Iz)
        radius = (5.0 * I_max / (2.0 * mass)) ** 0.5
        I_sphere = 2.0 / 5.0 * mass * radius**2
        assert I_sphere == pytest.approx(I_max, rel=1e-10)

    def test_isotropic_inertia_spread(self) -> None:
        """Document the spread: how far off I_x and I_y are from reference."""
        I_max = max(BLUEROV2.Ix, BLUEROV2.Iy, BLUEROV2.Iz)
        ix_error = (I_max - VON_BENZON.I_x) / VON_BENZON.I_x
        iy_error = (I_max - VON_BENZON.I_y) / VON_BENZON.I_y
        # I_x is 42% too high, I_y is 61% too high
        assert ix_error > 0.3
        assert iy_error > 0.5


# ---------------------------------------------------------------------------
# 2. Water density
# ---------------------------------------------------------------------------


class TestWaterDensityConsistent:
    """Tier1 defaults to salt water (1025 kg/m³), von Benzon uses fresh (1000).

    This causes a buoyancy sign flip: von Benzon ROV sinks (+0.98 N down),
    Tier1 ROV floats up (-2.39 N net up). Qualitatively different behavior.

    See fidelity_param_analysis.md BUG 2 (HIGH).
    """

    @pytest.mark.xfail(
        reason="Tier1 defaults to salt water rho=1025; von Benzon uses fresh "
               "rho=1000. Causes buoyancy sign flip (sinks vs floats).",
        strict=True,
    )
    def test_rho_matches(self) -> None:
        assert DEFAULT_RHO == pytest.approx(VON_BENZON.rho, rel=RTOL)

    def test_rho_values_documented(self) -> None:
        """Both values are intentional — document the delta."""
        assert DEFAULT_RHO == 1025.0  # salt water
        assert VON_BENZON.rho == 1000.0  # fresh water

    def test_buoyancy_sign_flip(self) -> None:
        """With default Tier1 rho, net buoyancy direction flips vs von Benzon."""
        g_vb = VON_BENZON.g
        g_t1 = 9.81

        # von Benzon: W - B = m*g - rho*g*V
        wb_vb = VON_BENZON.mass * g_vb - VON_BENZON.rho * g_vb * VON_BENZON.volume
        # Tier1 (salt):
        wb_t1 = BLUEROV2.mass * g_t1 - DEFAULT_RHO * g_t1 * BLUEROV2.volume

        # von Benzon: slightly negative buoyant (sinks)
        assert wb_vb > 0, f"von Benzon W-B={wb_vb:.3f}N, should be positive (sinks)"
        # Tier1: positively buoyant (floats)
        assert wb_t1 < 0, f"Tier1 W-B={wb_t1:.3f}N, should be negative (floats)"

    def test_gravity_close(self) -> None:
        """Gravity values differ by 0.1% — negligible."""
        assert 9.81 == pytest.approx(VON_BENZON.g, rel=0.01)


# ---------------------------------------------------------------------------
# 3. Hydrodynamic coefficients (added mass + damping)
# ---------------------------------------------------------------------------


class TestHydroCoefficientsMatch:
    """BlueROV2Heavy coefficients should exactly match von Benzon Table A1.

    The coefficients themselves (M_A, D) are identical — they share the
    same source. The difference is in how they're *used*:
      - von Benzon: M = M_RB + M_A (inverted as one matrix)
      - Tier1: M_A as external force via EMA-filtered ν̇ estimate

    See fidelity_param_analysis.md §2a–2c (all YES) and BUG 4 (EMA lag).
    """

    # --- Added mass ---

    def test_added_mass_surge(self) -> None:
        assert BLUEROV2.added_mass[0] == pytest.approx(VON_BENZON.added_mass[0], rel=RTOL)

    def test_added_mass_sway(self) -> None:
        assert BLUEROV2.added_mass[1] == pytest.approx(VON_BENZON.added_mass[1], rel=RTOL)

    def test_added_mass_heave(self) -> None:
        assert BLUEROV2.added_mass[2] == pytest.approx(VON_BENZON.added_mass[2], rel=RTOL)

    def test_added_mass_roll(self) -> None:
        assert BLUEROV2.added_mass[3] == pytest.approx(VON_BENZON.added_mass[3], rel=RTOL)

    def test_added_mass_pitch(self) -> None:
        assert BLUEROV2.added_mass[4] == pytest.approx(VON_BENZON.added_mass[4], rel=RTOL)

    def test_added_mass_yaw(self) -> None:
        assert BLUEROV2.added_mass[5] == pytest.approx(VON_BENZON.added_mass[5], rel=RTOL)

    def test_added_mass_full_tuple(self) -> None:
        for i, (gpu, cpu) in enumerate(zip(BLUEROV2.added_mass, VON_BENZON.added_mass)):
            assert gpu == pytest.approx(cpu, rel=RTOL), f"Mismatch at DOF index {i}"

    # --- Linear damping ---

    def test_d_lin_full_tuple(self) -> None:
        for i, (gpu, cpu) in enumerate(zip(BLUEROV2.d_lin, VON_BENZON.d_lin)):
            assert gpu == pytest.approx(cpu, rel=RTOL), f"d_lin mismatch at DOF index {i}"

    # --- Quadratic damping ---

    def test_d_quad_full_tuple(self) -> None:
        for i, (gpu, cpu) in enumerate(zip(BLUEROV2.d_quad, VON_BENZON.d_quad)):
            assert gpu == pytest.approx(cpu, rel=RTOL), f"d_quad mismatch at DOF index {i}"

    # --- Scalar body params ---

    def test_mass_matches(self) -> None:
        assert BLUEROV2.mass == pytest.approx(VON_BENZON.mass, rel=RTOL)

    def test_volume_matches(self) -> None:
        assert BLUEROV2.volume == pytest.approx(VON_BENZON.volume, rel=RTOL)

    # --- Cross-check: effective mass matrix diagonal (von Benzon formulation) ---

    def test_effective_mass_diagonal(self) -> None:
        """M_eff = M_RB + M_A for each DOF. Documents the inertia bug impact."""
        m = VON_BENZON.mass
        ma = VON_BENZON.added_mass

        # Correct (von Benzon): uses actual I_x, I_y, I_z
        m_eff_correct = [
            m + ma[0],
            m + ma[1],
            m + ma[2],
            VON_BENZON.I_x + ma[3],
            VON_BENZON.I_y + ma[4],
            VON_BENZON.I_z + ma[5],
        ]

        # GPU path: sphere I_max for all angular DOFs
        I_max = max(VON_BENZON.I_x, VON_BENZON.I_y, VON_BENZON.I_z)
        m_eff_gpu = [
            m + ma[0],
            m + ma[1],
            m + ma[2],
            I_max + ma[3],
            I_max + ma[4],
            I_max + ma[5],
        ]

        # Linear DOFs match exactly
        for i in range(3):
            assert m_eff_gpu[i] == pytest.approx(m_eff_correct[i], rel=RTOL)

        # Angular DOFs: GPU is wrong (xfail above), document magnitude
        for i in range(3, 6):
            ratio = m_eff_gpu[i] / m_eff_correct[i]
            # Angular effective mass is ~1.5-2x too high
            assert ratio >= 1.0, f"DOF {i}: GPU eff mass {m_eff_gpu[i]:.3f} should exceed correct {m_eff_correct[i]:.3f}"
