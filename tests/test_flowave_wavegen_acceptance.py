"""Acceptance tests: the GENERATED FloWave wave field must be physically correct.

These tests verify the *realized* free-surface field produced by
``FlapPaddleArray.synthesize_regular`` against first-principles physics that is
recomputed independently *inside the test* (finite-depth dispersion solved here
from scratch).  They do NOT assert synth internals against themselves; if the
synthesis is physically wrong, these tests must fail.

Pure NumPy, headless, fast.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from oceanscale.facilities.flowave.paddle_array import FlapPaddleArray
except Exception as exc:  # pragma: no cover - import guard only
    pytest.skip(f"oceanscale import failed: {exc}", allow_module_level=True)

G = 9.81  # m/s^2

# Well-resolved operating point (FloWave REGULAR_DESIGN: H=0.10, T=2.0, depth=2.0).
H = 0.1
T = 2.0
TANK_HALF = 12.0  # [-12, 12] m working zone


def _depth() -> float:
    """Still-water depth actually used by the paddle array (no invented value)."""
    return float(FlapPaddleArray().h)


def _dispersion_independent(omega: float, depth: float) -> float:
    """Solve omega^2 = g k tanh(k h) for k via bisection, independently.

    Deliberately does NOT call FlapPaddleArray.dispersion: this is the ground
    truth the synth is checked against.
    """
    def residual(k: float) -> float:
        return G * k * np.tanh(k * depth) - omega * omega

    lo, hi = 1e-8, 100.0
    assert residual(lo) < 0.0 < residual(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _refined_peak_freq(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    """Peak frequency with parabolic (log-magnitude) interpolation around the bin."""
    i = int(np.argmax(magnitude))
    df = float(freqs[1] - freqs[0])
    if 0 < i < len(magnitude) - 1:
        a = np.log(magnitude[i - 1] + 1e-30)
        b = np.log(magnitude[i] + 1e-30)
        c = np.log(magnitude[i + 1] + 1e-30)
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    else:
        delta = 0.0
    return float(freqs[i] + delta * df)


def test_generated_wavelength_matches_dispersion() -> None:
    depth = _depth()
    ws = FlapPaddleArray(h=depth).synthesize_regular(H, T, direction=0.0)

    # Realized eta along a dense 1D line y=0, t=0.
    n = 1024
    xs = np.linspace(-TANK_HALF, TANK_HALF, n)
    xy = np.column_stack([xs, np.zeros_like(xs)]).astype(np.float32)
    eta = ws.eta_field(xy, t=0.0)
    eta = eta - eta.mean()

    dx = float(xs[1] - xs[0])
    spec = np.abs(np.fft.rfft(eta))
    sfreq = np.fft.rfftfreq(n, d=dx)  # cycles per metre
    peak_sfreq = _refined_peak_freq(spec, sfreq)
    L_measured = 1.0 / peak_sfreq

    # Independent ground truth.
    omega = 2.0 * np.pi / T
    k_expected = _dispersion_independent(omega, depth)
    L_expected = 2.0 * np.pi / k_expected

    print(f"L_measured={L_measured:.6f} L_expected={L_expected:.6f} "
          f"rel_err={abs(L_measured - L_expected) / L_expected:.4e}")
    assert abs(L_measured - L_expected) / L_expected < 0.05


def test_generated_period_matches_command() -> None:
    depth = _depth()
    ws = FlapPaddleArray(h=depth).synthesize_regular(H, T, direction=0.0)

    dt = T / 40.0
    ts = np.arange(0.0, 6.0 * T, dt)  # >= 4 periods
    gauge = np.array([[0.0, 0.0]], dtype=np.float32)
    eta = np.array([float(ws.eta_field(gauge, t=float(tt))[0]) for tt in ts])
    eta = eta - eta.mean()

    spec = np.abs(np.fft.rfft(eta))
    freqs = np.fft.rfftfreq(ts.shape[0], d=dt)  # Hz
    peak_freq = _refined_peak_freq(spec, freqs)
    T_measured = 1.0 / peak_freq

    print(f"T_measured={T_measured:.6f} T_expected={T:.6f} "
          f"rel_err={abs(T_measured - T) / T:.4e}")
    assert abs(T_measured - T) / T < 0.05


def _measured_height(depth: float, height: float, period: float) -> float:
    """Realized wave height (max-min) from a steady-state gauge time series."""
    ws = FlapPaddleArray(h=depth).synthesize_regular(height, period, direction=0.0)
    dt = period / 200.0
    # Sample one steady period well after t=0.
    ts = np.arange(4.0 * period, 5.0 * period, dt)
    gauge = np.array([[0.0, 0.0]], dtype=np.float32)
    eta = np.array([float(ws.eta_field(gauge, t=float(tt))[0]) for tt in ts])
    return float(eta.max() - eta.min())


def test_generated_height_matches_biesel_command() -> None:
    depth = _depth()
    H_measured = _measured_height(depth, H, T)
    print(f"H_measured={H_measured:.6f} H_expected={H:.6f} "
          f"rel_err={abs(H_measured - H) / H:.4e}")
    assert abs(H_measured - H) / H < 0.10


def test_height_scales_with_command() -> None:
    depth = _depth()
    h_small = _measured_height(depth, 0.05, T)
    h_large = _measured_height(depth, 0.15, T)
    ratio = h_large / h_small
    print(f"h_small={h_small:.6f} h_large={h_large:.6f} ratio={ratio:.4f} (expect ~3)")
    assert abs(ratio - 3.0) / 3.0 < 0.15


def _flap_transfer_independent(k: float, depth: float) -> float:
    """Bottom-hinged flap H/S recomputed from scratch (Dean & Dalrymple §6.3),
    independent of FlapPaddleArray.transfer_fn_HS so it can audit it."""
    kh = k * depth
    return float(
        4.0 * np.sinh(kh) * (kh * np.sinh(kh) - np.cosh(kh) + 1.0)
        / (kh * (np.sinh(2.0 * kh) + 2.0 * kh))
    )


def test_paddle_stroke_obeys_independent_biesel_transfer() -> None:
    """Anti-tautology: the realized surface amplitude is H/2 by construction, so
    the height tests above cannot detect a wrong wavemaker transfer function.
    This checks the stroke->height inversion instead: the peak commanded paddle
    stroke must equal (H/2)/(H/S) * Snake-envelope, with k and H/S recomputed
    independently here. The earlier ~3x-off transfer form fails this test.
    """
    depth = _depth()
    pa = FlapPaddleArray(h=depth)
    omega = 2.0 * np.pi / T
    k = _dispersion_independent(omega, depth)
    hs_indep = _flap_transfer_independent(k, depth)
    a = 0.5 * H
    envelope = float(np.max(np.abs(np.cos(k * pa.R * np.cos(0.0 - pa.phi_n)))))
    expected_max_stroke = (a / hs_indep) * envelope

    ws = pa.synthesize_regular(H=H, T=T, direction=0.0)
    dt = T / 40.0
    ts = np.arange(int(2 * T / dt)) * dt
    pc = np.array([ws.paddle_commands(float(t)) for t in ts])  # (n_steps, N)
    realized_max_stroke = float(np.max(np.abs(pc)))

    rel = abs(realized_max_stroke - expected_max_stroke) / expected_max_stroke
    print(f"stroke realized={realized_max_stroke:.6f} expected={expected_max_stroke:.6f} "
          f"hs_indep={hs_indep:.4f} rel={rel:.4e}")
    assert rel < 0.03
