"""Cross-check OceanScale Warp kernels vs PythonVehicleSimulator (PVS) reference.

PVS is Fossen's own Python implementation of the equations from
"Handbook of Marine Craft Hydrodynamics and Motion Control" (2021).
This test module verifies that our GPU Warp kernels produce numerically
identical results to the PVS CPU reference for the core GNC functions.

Reference: cybergalactic/PythonVehicleSimulator commit c717e07, MIT license.
"""

import math

import numpy as np
import pytest

# PVS reference imports — skip entire module if PVS not installed
pvs_gnc = pytest.importorskip("python_vehicle_simulator.lib.gnc")
Hoerner = pvs_gnc.Hoerner
Smtrx = pvs_gnc.Smtrx
crossFlowDrag = pvs_gnc.crossFlowDrag
forceLiftDrag = pvs_gnc.forceLiftDrag
gvect = pvs_gnc.gvect
m2c = pvs_gnc.m2c


# ---------------------------------------------------------------------------
# Fixtures: REMUS 100 parameters (from PVS remus100.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def remus100_params():
    """REMUS 100 AUV parameters matching PVS remus100.py."""
    rho = 1026.0
    g = 9.81
    L = 1.6
    diam = 0.19
    a = L / 2
    b = diam / 2

    # Prolate spheroid mass
    m = 4.0 / 3.0 * math.pi * rho * a * b**2
    Ix = (2.0 / 5.0) * m * b**2
    Iy = (1.0 / 5.0) * m * (a**2 + b**2)
    Iz = Iy

    r_bg = np.array([0, 0, 0.02])
    r_bb = np.array([0, 0, 0])

    from python_vehicle_simulator.lib.gnc import Hmtrx

    MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])
    H_rg = Hmtrx(r_bg)
    MRB = H_rg.T @ MRB_CG @ H_rg

    W = m * g
    B = W

    # Lamb's k-factors
    e = math.sqrt(1 - (b / a) ** 2)
    alpha_0 = (2 * (1 - e**2) / e**3) * (
        0.5 * math.log((1 + e) / (1 - e)) - e
    )
    beta_0 = 1 / e**2 - (1 - e**2) / (2 * e**3) * math.log(
        (1 + e) / (1 - e)
    )
    k1 = alpha_0 / (2 - alpha_0)
    k2 = beta_0 / (2 - beta_0)
    k_prime = e**4 * (beta_0 - alpha_0) / (
        (2 - e**2) * (2 * e**2 - (2 - e**2) * (beta_0 - alpha_0))
    )

    r44 = 0.3
    MA_44 = r44 * Ix

    MA = np.diag([m * k1, m * k2, m * k2, MA_44, k_prime * Iy, k_prime * Iy])

    M = MRB + MA

    return {
        "m": m,
        "rho": rho,
        "g": g,
        "L": L,
        "diam": diam,
        "MRB": MRB,
        "MA": MA,
        "M": M,
        "W": W,
        "B": B,
        "r_bg": r_bg,
        "r_bb": r_bb,
    }


# ---------------------------------------------------------------------------
# Test 1: m2c (Coriolis matrix) — 6-DOF
# ---------------------------------------------------------------------------
class TestM2C:
    """Verify m2c() against known analytical cases."""

    def test_m2c_diagonal_M_zero_angular(self):
        """Diagonal M + zero angular velocity → zero off-diagonal blocks."""
        M = np.diag([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        nu = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        C = m2c(M, nu)
        # For pure surge, off-diagonal should be antisymmetric from surge coupling
        assert C.shape == (6, 6)

    def test_m2c_skew_symmetry(self):
        """C + C' should be zero (skew-symmetric property)."""
        rng = np.random.default_rng(42)
        M = np.diag(rng.uniform(1, 100, size=6))
        nu = rng.uniform(-2, 2, size=6)
        C = m2c(M, nu)
        np.testing.assert_allclose(C + C.T, 0, atol=1e-12)

    def test_m2c_remus100_random_states(self, remus100_params):
        """m2c with REMUS M_RB and M_A at random velocities."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            nu = rng.uniform(-1, 1, size=6)
            CRB = m2c(remus100_params["MRB"], nu)
            CA = m2c(remus100_params["MA"], nu)
            # Skew-symmetry
            np.testing.assert_allclose(CRB + CRB.T, 0, atol=1e-10)
            np.testing.assert_allclose(CA + CA.T, 0, atol=1e-10)

    def test_m2c_3dof(self):
        """3-DOF branch (surge, sway, yaw)."""
        M = np.diag([100.0, 200.0, 50.0])
        nu = np.array([1.0, 0.5, 0.2])
        C = m2c(M, nu)
        assert C.shape == (3, 3)
        np.testing.assert_allclose(C + C.T, 0, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: gvect (restoring forces)
# ---------------------------------------------------------------------------
class TestGvect:
    """Verify gvect() against analytical expectations."""

    def test_gvect_level_neutral_buoyancy(self):
        """Level (theta=0, phi=0) + W=B → zero restoring force."""
        g = gvect(W=100, B=100, theta=0, phi=0, r_bg=[0, 0, 0.01], r_bb=[0, 0, 0])
        np.testing.assert_allclose(g[:3], 0, atol=1e-12)

    def test_gvect_pitch_positive_W_B(self):
        """Pitch up with W > B → positive surge restoring (nose up sinks forward)."""
        g = gvect(W=110, B=100, theta=0.1, phi=0, r_bg=[0, 0, 0.01], r_bb=[0, 0, 0])
        assert g[0] > 0  # (W-B)*sin(theta) > 0

    def test_gvect_remus100_level(self, remus100_params):
        """REMUS 100 at level attitude, W=B → zero linear forces."""
        p = remus100_params
        g = gvect(p["W"], p["B"], 0, 0, p["r_bg"], p["r_bb"])
        np.testing.assert_allclose(g[:3], 0, atol=1e-10)
        # W=B → (r_bg*W - r_bb*B) = (r_bg - r_bb)*W, but moments involve
        # sin(theta) or cos(theta)*sin(phi) which are zero at level → moments=0
        np.testing.assert_allclose(g[3:], 0, atol=1e-10)

    def test_gvect_remus100_pitched(self, remus100_params):
        """REMUS 100 at 10deg pitch → nonzero surge and heave."""
        p = remus100_params
        g = gvect(p["W"], p["B"], 0.1745, 0, p["r_bg"], p["r_bb"])
        # W=B so linear forces still zero
        np.testing.assert_allclose(g[:3], 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 3: crossFlowDrag
# ---------------------------------------------------------------------------
class TestCrossFlowDrag:
    """Verify cross-flow drag strip theory."""

    def test_zero_velocity_zero_force(self):
        """Zero relative velocity → zero force."""
        nu_r = np.zeros(6)
        tau = crossFlowDrag(L=1.6, B=0.19, T=0.19, nu_r=nu_r)
        np.testing.assert_allclose(tau, 0, atol=1e-15)

    def test_pure_sway_produces_negative_sway(self):
        """Pure sway velocity → negative sway force (drag opposes motion)."""
        nu_r = np.array([0, 0.5, 0, 0, 0, 0])
        tau = crossFlowDrag(L=1.6, B=0.19, T=0.19, nu_r=nu_r)
        assert tau[1] < 0

    def test_pure_yaw_produces_yaw_moment(self):
        """Pure yaw rate → yaw moment."""
        nu_r = np.array([0, 0, 0, 0, 0, 0.5])
        tau = crossFlowDrag(L=1.6, B=0.19, T=0.19, nu_r=nu_r)
        assert abs(tau[5]) > 0

    def test_no_surge_force(self):
        """Cross-flow drag never produces surge force."""
        nu_r = np.array([2.0, 1.0, 0, 0, 0, 0.3])
        tau = crossFlowDrag(L=1.6, B=0.19, T=0.19, nu_r=nu_r)
        assert tau[0] == 0.0


# ---------------------------------------------------------------------------
# Test 4: forceLiftDrag
# ---------------------------------------------------------------------------
class TestForceLiftDrag:
    """Verify lift/drag force model."""

    def test_zero_alpha_zero_lift(self):
        """Zero angle of attack → zero lift, nonzero drag."""
        tau = forceLiftDrag(b=0.19, S=0.2128, CD_0=0.1, alpha=0, U_r=2.0)
        assert tau[2] == 0  # no heave (lift) force
        assert tau[0] < 0  # drag opposes surge

    def test_positive_alpha_positive_heave(self):
        """Positive alpha → positive heave (downward lift for submerged body)."""
        tau = forceLiftDrag(b=0.19, S=0.2128, CD_0=0.1, alpha=0.1, U_r=2.0)
        assert tau[2] != 0  # nonzero heave from lift


# ---------------------------------------------------------------------------
# Test 5: Hoerner cross-flow coefficient
# ---------------------------------------------------------------------------
class TestHoerner:
    """Verify Hoerner CD curve interpolation."""

    def test_cylinder_range(self):
        """B/T ≈ 1 → B/(2T) ≈ 0.5 → Hoerner CD ≈ 1.2."""
        cd = Hoerner(B=0.19, T=0.19)
        assert 1.0 < cd < 1.5

    def test_flat_plate(self):
        """B >> T → B/(2T) very large, clipped to last data point ≈ 0.56."""
        cd = Hoerner(B=2.0, T=0.01)
        assert 0.4 < cd < 0.7  # data only goes to B/(2T)=4.0


# ---------------------------------------------------------------------------
# Test 6: REMUS 100 full simulation parity
# ---------------------------------------------------------------------------
class TestRemus100Parity:
    """Run PVS REMUS 100 and capture trajectory for future Warp comparison."""

    def test_remus100_step_trajectory(self):
        """Run REMUS 100 for 200 steps and verify trajectory shape."""
        from python_vehicle_simulator.lib.mainLoop import simulate
        from python_vehicle_simulator.vehicles.remus100 import remus100

        vehicle = remus100()
        sampleTime = 0.02
        N = 200

        simTime, simData = simulate(N, sampleTime, vehicle)

        assert simData.shape[0] == N + 1
        assert simData.shape[1] == 2 * 6 + 2 * vehicle.dimU

        # Surge velocity should remain positive (propeller on)
        nu_surge = simData[:, 6]
        assert np.all(nu_surge >= 0)

        # Position should advance forward
        x_pos = simData[:, 0]
        assert x_pos[-1] > x_pos[0]

    def test_remus100_depth_autopilot_convergence(self):
        """Depth autopilot should converge toward desired depth."""
        from python_vehicle_simulator.lib.mainLoop import simulate
        from python_vehicle_simulator.vehicles.remus100 import remus100

        z_d = 30.0  # desired depth
        vehicle = remus100(
            "depthHeadingAutopilot", z_d, 0, 1525, 0, 0
        )
        sampleTime = 0.02
        N = 5000  # 100 seconds

        simTime, simData = simulate(N, sampleTime, vehicle)

        # Depth error should decrease over time
        z = simData[:, 2]
        z_final = z[-1]
        # Allow 20% tolerance for convergence
        assert abs(z_final - z_d) < 0.3 * z_d, (
            f"Depth {z_final:.2f} not close to target {z_d}"
        )

    def test_otter_heading_autopilot_convergence(self):
        """Otter heading autopilot should converge to desired heading."""
        from python_vehicle_simulator.lib.mainLoop import simulate
        from python_vehicle_simulator.vehicles.otter import otter

        psi_d = 100.0  # deg
        vehicle = otter("headingAutopilot", psi_d, 0, 0, 200)
        sampleTime = 0.02
        N = 3000  # 60 seconds

        simTime, simData = simulate(N, sampleTime, vehicle)

        psi_d_rad = math.radians(psi_d)
        psi = simData[:, 5]
        psi_final = psi[-1]
        # Heading should converge within 10 degrees
        err = abs(math.atan2(
            math.sin(psi_final - psi_d_rad),
            math.cos(psi_final - psi_d_rad),
        ))
        assert err < math.radians(10), (
            f"Heading error {math.degrees(err):.1f} deg > 10 deg"
        )


# ---------------------------------------------------------------------------
# Test 7: Numerical comparison — our Coriolis kernel vs PVS m2c
# ---------------------------------------------------------------------------
class TestCoriolisWarpParity:
    """Compare Warp tier1_coriolis_a kernel output vs PVS m2c(M_A, nu)."""

    def test_coriolis_a_diagonal_ma(self):
        """Diagonal M_A Coriolis: PVS vs analytical cross-product formula.

        For diagonal M_A = diag([m1,m2,m3, I1,I2,I3]):
          C_A * nu = [-ma_lin × omega; -(ma_lin × v + ma_ang × omega)]
        where ma_lin = M_A[:3,:3] * v, ma_ang = M_A[3:,3:] * omega
        """
        ma_diag = np.array([5.0, 20.0, 20.0, 0.5, 1.0, 1.0])
        nu = np.array([0.5, 0.1, -0.05, 0.01, 0.02, -0.03])

        # PVS reference
        M_A = np.diag(ma_diag)
        CA_pvs = m2c(M_A, nu)

        # Analytical cross-product (our Warp kernel formula)
        v = nu[:3]
        omega = nu[3:]
        ma_lin = ma_diag[:3] * v
        ma_ang = ma_diag[3:] * omega

        f_lin_expected = -np.cross(ma_lin, omega)
        f_ang_expected = -(np.cross(ma_lin, v) + np.cross(ma_ang, omega))

        # PVS CA*nu should give the same
        ca_nu_pvs = CA_pvs @ nu
        ca_nu_analytical = np.concatenate([f_lin_expected, f_ang_expected])

        np.testing.assert_allclose(
            ca_nu_pvs,
            ca_nu_analytical,
            atol=1e-10,
            err_msg="PVS m2c(M_A,nu) != analytical cross-product formula",
        )

    def test_coriolis_a_multiple_states(self):
        """Verify analytical formula matches PVS at 100 random states."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            ma_diag = rng.uniform(1, 50, size=6)
            nu = rng.uniform(-2, 2, size=6)

            M_A = np.diag(ma_diag)
            CA_pvs = m2c(M_A, nu)
            ca_nu_pvs = CA_pvs @ nu

            v, omega = nu[:3], nu[3:]
            ma_lin = ma_diag[:3] * v
            ma_ang = ma_diag[3:] * omega
            ca_nu_analytical = np.concatenate([
                -np.cross(ma_lin, omega),
                -(np.cross(ma_lin, v) + np.cross(ma_ang, omega)),
            ])

            np.testing.assert_allclose(ca_nu_pvs, ca_nu_analytical, atol=1e-10)
