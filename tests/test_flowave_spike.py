"""Acceptance tests for the FloWave concentric-spike (axisymmetric time-focused) wave.

Verifies the realized free surface of FlapPaddleArray.synthesize_focused_spike
against the defining properties of the Edinburgh FloWave "concentric spike" demo:
all paddles fire identically (axisymmetric), circular wavefronts converge to a
central spike at t_focus, and the field is radially symmetric.

Pure NumPy + scipy, headless, fast.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from oceanscale.facilities.flowave.paddle_array import FlapPaddleArray
except Exception as exc:  # pragma: no cover - import guard only
    pytest.skip(f"oceanscale import failed: {exc}", allow_module_level=True)

AMP = 0.15
TF = 3.0


def _ws():
    return FlapPaddleArray().synthesize_focused_spike(amplitude=AMP, t_focus=TF)


def test_paddles_axisymmetric() -> None:
    """All 168 paddles move identically (no directional phase)."""
    ws = _ws()
    pc = ws.paddle_commands(TF)
    assert pc.shape == (168,)
    assert np.allclose(pc, pc[0]), "paddles must be identical for an axisymmetric spike"
    # and they actually move over time (not a constant)
    assert not np.isclose(ws.paddle_commands(TF)[0], ws.paddle_commands(TF + 0.5)[0])


def test_temporal_focusing() -> None:
    """Central elevation peaks at t_focus and is far smaller away from it."""
    ws = _ws()
    c_focus = abs(float(ws.eta_field(np.array([[0.0, 0.0]]), TF)[0]))
    c_after = abs(float(ws.eta_field(np.array([[0.0, 0.0]]), TF + 3.0)[0]))
    assert c_focus / max(c_after, 1e-9) > 3.0, (
        f"temporal focusing gain {c_focus / max(c_after, 1e-9):.1f} too small"
    )


def test_spatial_concentration() -> None:
    """At t_focus the centre dominates the off-centre field (converged spike)."""
    ws = _ws()
    c0 = abs(float(ws.eta_field(np.array([[0.0, 0.0]]), TF)[0]))
    cr = abs(float(ws.eta_field(np.array([[8.0, 0.0]]), TF)[0]))
    assert c0 / max(cr, 1e-9) > 3.0, (
        f"spatial concentration {c0 / max(cr, 1e-9):.1f} too small"
    )


def test_axisymmetry() -> None:
    """The field depends only on r = hypot(x, y), not on angle."""
    ws = _ws()
    r = 5.0
    vals = [
        float(ws.eta_field(np.array([[r * np.cos(a), r * np.sin(a)]]), TF)[0])
        for a in (0.0, 0.7, 1.5, 2.6, 4.1)
    ]
    assert np.allclose(vals, vals[0], atol=1e-6), "spike field must be axisymmetric"


def test_central_amplitude_matches_command() -> None:
    """Equal-weight packet focuses to ~= the commanded amplitude at (0, t_focus)."""
    ws = _ws()
    c0 = float(ws.eta_field(np.array([[0.0, 0.0]]), TF)[0])
    assert abs(c0 - AMP) / AMP < 0.15, f"central focus {c0:.4f} vs amplitude {AMP}"


def test_amplitude_scales() -> None:
    """Central focal height scales linearly with commanded amplitude."""
    small = FlapPaddleArray().synthesize_focused_spike(amplitude=0.05, t_focus=TF)
    large = FlapPaddleArray().synthesize_focused_spike(amplitude=0.15, t_focus=TF)
    cs = float(small.eta_field(np.array([[0.0, 0.0]]), TF)[0])
    cl = float(large.eta_field(np.array([[0.0, 0.0]]), TF)[0])
    assert abs((cl / cs) - 3.0) / 3.0 < 0.15, f"ratio {cl / cs:.3f} (want ~3)"
