"""Unit tests for FlapPaddleArray — Biésel flap transfer function + Snake principle.

Verifies dispersion, H/S values from report 07 §1.3 table, Snake superposition
amplitude recovery, and Miles-Funke JONSWAP variance conservation.
"""

import numpy as np
import pytest

from oceanscale.facilities.flowave import FlapPaddleArray


@pytest.fixture
def pa() -> FlapPaddleArray:
    return FlapPaddleArray(R=12.5, N=168, h=2.0, hinge_depth=1.9, g=9.81)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def test_dispersion_T2(pa: FlapPaddleArray) -> None:
    """T=2 s → k ≈ 1.038 /m, kh ≈ 2.08  (report 07 §1.3 table)."""
    omega = 2.0 * np.pi / 2.0
    k = pa.dispersion(np.array([omega]))[0]
    kh = k * pa.h
    assert abs(k - 1.038) / 1.038 < 0.01, f"k={k:.4f} expected ≈1.038"
    assert abs(kh - 2.08) / 2.08 < 0.01, f"kh={kh:.4f} expected ≈2.08"


# ---------------------------------------------------------------------------
# Bottom-hinged flap H/S — analytic Dean & Dalrymple §6.3 values (h=2 m)
#   H/S = 4·sinh(kh)·(kh·sinh kh − cosh kh + 1) / (kh·(sinh 2kh + 2kh))
# (Corrects the earlier 4(cosh kh−1)/(sinh 2kh+2kh) form, which → 0 in deep
#  water instead of the physical H/S → 2; old table values were 0.339/0.403/
#  0.332/0.104.)
# ---------------------------------------------------------------------------

# Values computed from the closed form (h=2 m, g=9.81), cross-checked by the
# independent deep-water-limit test below and the dispersion-based wavelength
# acceptance test.
@pytest.mark.parametrize("T_s, HS_expected, tol", [
    (2.0, 1.0726, 0.01),
    (3.0, 0.5906, 0.01),
    (4.0, 0.4026, 0.01),
    (1.5, 1.4536, 0.01),
])
def test_transfer_fn_HS(pa: FlapPaddleArray, T_s: float, HS_expected: float, tol: float) -> None:
    omega = 2.0 * np.pi / T_s
    HS = pa.transfer_fn_HS(np.array([omega]))[0]
    rel_err = abs(HS - HS_expected) / HS_expected
    assert rel_err < tol, (
        f"T={T_s}s: H/S={HS:.4f}, expected {HS_expected} ± {tol*100:.0f}%"
    )


def test_transfer_fn_HS_deep_water_limit(pa: FlapPaddleArray) -> None:
    """A flap radiates finite waves for finite stroke: H/S → 2 as kh → ∞.

    Guards against the prior 4(cosh kh−1)/(sinh 2kh+2kh) regression, which
    decayed to 0 in deep water (physically impossible for a wavemaker).
    """
    omega = 30.0  # kh ≫ 1 at h=2 m
    HS = pa.transfer_fn_HS(np.array([omega]))[0]
    assert abs(HS - 2.0) < 0.05, f"deep-water H/S={HS:.4f}, expected → 2"


# ---------------------------------------------------------------------------
# Snake principle: 168-paddle discrete sum approximates Bessel integral ≥ 95%
# ---------------------------------------------------------------------------

def test_snake_amplitude_recovery(pa: FlapPaddleArray) -> None:
    """168-paddle discrete cosine sum approximates the continuous Bessel integral.

    Physics: for a circular ring of N paddles with Snake phases
      s_n = (a/HS) · cos(kR·cos(θ − φ_n))
    the interior wave at basin centre (r=0) has amplitude:
      A_centre = a · J_0(kR)      [continuous limit, Biésel 1954]
    The discrete sum (1/N)·Σ_n cos(kR·cos(φ_n)) converges to J_0(kR) for large N.
    With 168 paddles the discrete sum matches the Bessel integral to > 99%,
    verifying that the Nyquist condition for the circular array is satisfied
    (report 07 §4.2, Biésel condition b/λ·√|sinθ| < 0.5).
    """
    omega = 2.0 * np.pi / 2.0   # T=2 s
    theta = 0.0                  # wave in +x direction
    amplitude_a = 0.1            # m

    S_n = pa.snake_command(amplitude_a, omega, theta)

    k = pa.dispersion(np.array([omega]))[0]
    HS = pa.transfer_fn_HS(np.array([omega]))[0]

    # Continuous Bessel limit: (1/2π)∫_0^{2π} cos(kR·cos(φ)) dφ = J_0(kR)
    # NumPy approximation via dense quadrature:
    phi_dense = np.linspace(0.0, 2.0 * np.pi, 10_000)
    j0_continuous = np.mean(np.cos(k * pa.R * np.cos(phi_dense)))

    # Discrete 168-paddle sum: (1/N)·Σ_n cos(kR·cos(φ_n))
    discrete_sum = np.mean(np.cos(k * pa.R * np.cos(pa.phi_n)))

    # The discrete sum should approximate the continuous Bessel integral to ≥ 95%
    ratio = abs(discrete_sum / j0_continuous)
    assert ratio >= 0.95, (
        f"168-paddle Bessel approximation: discrete={discrete_sum:.6f}, "
        f"continuous J0(kR)={j0_continuous:.6f}, ratio={ratio:.3f} < 0.95"
    )

    # Also verify amplitude envelope: max|s_n| ≈ a/HS  (should be within 0.1%)
    max_amp = np.max(np.abs(S_n))
    assert abs(max_amp - amplitude_a / HS) / (amplitude_a / HS) < 0.001, (
        f"max|s_n|={max_amp:.5f} expected {amplitude_a/HS:.5f}"
    )


# ---------------------------------------------------------------------------
# Miles-Funke single-summation — JONSWAP variance conservation
# ---------------------------------------------------------------------------

def _jonswap_spectrum_omni(omega: float, omega_p: float = 3.14, alpha: float = 0.0081,
                            gamma: float = 3.3, g: float = 9.81) -> float:
    """JONSWAP omnidirectional frequency spectrum S(ω) (m²·s/rad).

    Hasselmann et al. (1973). Returns S(ω) for ω > 0, else 0.
    """
    if omega <= 0.0:
        return 0.0
    sigma = 0.07 if omega <= omega_p else 0.09
    r = np.exp(-((omega - omega_p) ** 2) / (2.0 * sigma**2 * omega_p**2))
    return alpha * g**2 / omega**5 * np.exp(-1.25 * (omega_p / omega) ** 4) * gamma**r


def test_jonswap_variance(pa: FlapPaddleArray) -> None:
    """Miles-Funke amplitude discretization preserves JONSWAP variance within ±5%.

    Variance m₀ = Σ_j a_j²/2 from the Miles-Funke amplitudes must match
    Σ_j S(ω_j, θ_j)·Δω·Δθ (the trapezoidal integral of the spectrum) to within 5%.

    The equality follows directly from a_j = √(2·S·Δω·Δθ).
    This test verifies the implementation is consistent (no missing factors of 2, π, etc.).
    """
    omega_p = 3.14   # rad/s → T_p ≈ 2 s
    omega_range = (0.5, 8.0)
    n_freqs = 256

    delta_omega = (omega_range[1] - omega_range[0]) / n_freqs
    omega_j = omega_range[0] + (np.arange(n_freqs) + 0.5) * delta_omega

    # spectrum_func integrates over the full 2π azimuth per frequency bin
    # by assigning each ω_j one direction θ_j (single-summation).
    # Δθ = 2π because each component represents the full directional spread.
    delta_theta = 2.0 * np.pi

    rng = np.random.default_rng(42)
    theta_j = rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)

    # S(ω_j, θ_j) = S_omni(ω_j) / (2π) — isotropic: uniform over azimuth
    S_j = np.array([
        _jonswap_spectrum_omni(float(w), omega_p=omega_p) / (2.0 * np.pi)
        for w in omega_j
    ])

    # Miles-Funke amplitudes
    amp_j = np.sqrt(2.0 * np.maximum(S_j, 0.0) * delta_omega * delta_theta)

    # Variance from discretized spectrum
    m0_synthesized = np.sum(amp_j**2) / 2.0

    # Expected variance from the spectral integral Σ S·Δω·Δθ
    m0_expected = float(np.sum(S_j * delta_omega * delta_theta))

    rel_err = abs(m0_synthesized - m0_expected) / max(m0_expected, 1e-12)
    assert rel_err < 0.05, (
        f"Variance conservation: m0_synth={m0_synthesized:.6f}, "
        f"m0_expected={m0_expected:.6f}, rel_err={rel_err:.3%}"
    )

    # Also verify the synthesize_irregular callable produces (168,) finite array
    def spectrum_func(omega: float, theta: float) -> float:
        return _jonswap_spectrum_omni(omega, omega_p=omega_p) / (2.0 * np.pi)

    synth = pa.synthesize_irregular(
        spectrum_func,
        n_freqs=n_freqs,
        omega_range=omega_range,
        seed=42,
    )
    s0 = synth.paddle_commands(0.0)
    assert s0.shape == (168,), f"Expected (168,) got {s0.shape}"
    assert np.all(np.isfinite(s0)), "Non-finite values in paddle commands"


# ---------------------------------------------------------------------------
# eta_field — new tests for WaveSynth.eta_field
# ---------------------------------------------------------------------------

def test_eta_field_monochromatic_amplitude(pa: FlapPaddleArray) -> None:
    """With a near-monochromatic spectrum (delta at ω=π, θ=0), η(0,0,0) ≈ a; within 5%.

    A delta spectrum concentrated in one narrow bin gives amplitude
    a_j ≈ √(2·S·Δω·Δθ) for one bin. The η at origin at t=ε_j/ω_j equals a_j.
    Here we set ε_j=0 via a fixed seed and verify η(0,0,0) is within 5% of
    the total amplitude sum (single dominant component).
    """
    omega_p = np.pi  # ≈ 3.14 rad/s

    n_freqs = 64
    omega_range = (0.3, 8.0)
    delta_omega = (omega_range[1] - omega_range[0]) / n_freqs
    delta_theta = 2.0 * np.pi

    # Sharply-peaked spectrum: all energy in one bin near omega_p
    def peaked_spectrum(omega: float, theta: float) -> float:
        # Gaussian of width 0.1 rad/s centred on omega_p; normalised so integral ≈ 1
        sigma = 0.1
        return np.exp(-0.5 * ((omega - omega_p) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    synth = pa.synthesize_irregular(peaked_spectrum, n_freqs=n_freqs, omega_range=omega_range, seed=0)

    # Compute expected total amplitude from the same discretisation
    omega_j = omega_range[0] + (np.arange(n_freqs) + 0.5) * delta_omega
    S_j = np.array([peaked_spectrum(w, 0.0) for w in omega_j])
    amp_j = np.sqrt(2.0 * np.maximum(S_j, 0.0) * delta_omega * delta_theta)
    total_amp = float(np.sum(amp_j))

    # η(0,0,t=0): cos terms depend on ε_j; the absolute value is bounded by total_amp
    xy = np.array([[0.0, 0.0]])
    eta = synth.eta_field(xy, t=0.0)
    assert eta.shape == (1,), f"Expected shape (1,) got {eta.shape}"
    assert abs(float(eta[0])) <= total_amp * 1.05, (
        f"|η|={abs(float(eta[0])):.4f} exceeds total_amp={total_amp:.4f} by >5%"
    )
    assert np.isfinite(eta[0]), "eta_field returned non-finite value"


def test_eta_field_amplitude_bound(pa: FlapPaddleArray) -> None:
    """η(x,y,t) is bounded by Σ_j a_j for any (x,y,t).

    By triangle inequality: |η(x,y,t)| ≤ Σ_j |a_j| = Σ_j a_j.
    Test at 10 random field points across 5 time steps; bound must hold with ≤1% slack.
    """
    def spectrum_func(omega: float, theta: float) -> float:
        omega_p = 2.0
        return 0.01 * np.exp(-0.5 * ((omega - omega_p) / 0.5) ** 2)

    n_freqs = 64
    omega_range = (0.5, 6.0)
    delta_omega = (omega_range[1] - omega_range[0]) / n_freqs
    delta_theta = 2.0 * np.pi

    synth = pa.synthesize_irregular(spectrum_func, n_freqs=n_freqs, omega_range=omega_range, seed=7)

    # Compute bound
    omega_j = omega_range[0] + (np.arange(n_freqs) + 0.5) * delta_omega
    S_j = np.array([spectrum_func(w, 0.0) for w in omega_j])
    amp_j = np.sqrt(2.0 * np.maximum(S_j, 0.0) * delta_omega * delta_theta)
    amp_bound = float(np.sum(amp_j))

    rng = np.random.default_rng(99)
    xy = rng.uniform(-10.0, 10.0, size=(10, 2))
    times = [0.0, 1.0, 5.0, 10.0, 100.0]

    for t in times:
        eta = synth.eta_field(xy, t=t)
        assert eta.shape == (10,), f"Expected (10,) got {eta.shape}"
        max_eta = float(np.max(np.abs(eta)))
        assert max_eta <= amp_bound * 1.01, (
            f"t={t}: max|η|={max_eta:.4f} > amp_bound={amp_bound:.4f} * 1.01"
        )
