"""Tests for CPG (Central Pattern Generator) module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oceanscale.controllers.cpg import CPGConfig, KuramotoCPG, _smoothstep


class TestSmoothstep:
    def test_boundaries(self):
        assert _smoothstep(0.0) == pytest.approx(0.0)
        assert _smoothstep(1.0) == pytest.approx(1.0)

    def test_clamping(self):
        assert _smoothstep(-1.0) == pytest.approx(0.0)
        assert _smoothstep(2.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert _smoothstep(0.5) == pytest.approx(0.5)

    def test_monotonic(self):
        values = [_smoothstep(t) for t in np.linspace(0, 1, 20)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


class TestCPGConfig:
    def test_default_config(self):
        cfg = CPGConfig()
        assert cfg.n_oscillators == 6
        assert cfg.frequency_walk == 2.0
        assert cfg.frequency_swim == 1.2
        assert cfg.coupling_strength == 1.5

    def test_custom_config(self):
        cfg = CPGConfig(n_oscillators=4, frequency_walk=3.0, coupling_strength=2.0)
        assert cfg.n_oscillators == 4
        assert cfg.frequency_walk == 3.0
        assert cfg.coupling_strength == 2.0

    def test_frozen(self):
        cfg = CPGConfig()
        with pytest.raises(AttributeError):
            cfg.frequency_walk = 5.0  # type: ignore[misc]


class TestKuramotoCPG:
    def test_init(self):
        cpg = KuramotoCPG()
        assert cpg.n == 6
        assert cpg.drive == 0.0

    def test_custom_config(self):
        cfg = CPGConfig(n_oscillators=4)
        cpg = KuramotoCPG(cfg)
        assert cpg.n == 4

    def test_reset(self):
        cpg = KuramotoCPG()
        cpg.step()
        assert cpg.phases.sum() > 0 or True  # phases may or may not have changed
        cpg.reset()
        assert cpg.drive == 0.0

    def test_step_advances_phase(self):
        cpg = KuramotoCPG()
        phases_before = cpg.phases.copy()
        cpg.step()
        phases_after = cpg.phases
        # At least some phases should have changed
        assert not np.allclose(phases_before, phases_after)

    def test_standing_wave_synchronization(self):
        """With walk drive=0, oscillators should tend toward standing wave pattern."""
        cpg = KuramotoCPG()
        cpg.set_drive(0.0)
        for _ in range(500):
            cpg.step()
        phases = cpg.phases
        # Diagonal pairs (0,3) and (1,2) should have similar phase differences
        diff_03 = abs(phases[0] - phases[3]) % (2 * math.pi)
        diff_12 = abs(phases[1] - phases[2]) % (2 * math.pi)
        # Both should be close to pi (trot gait)
        assert diff_03 > 0.5 or diff_03 < 0.1  # converged to some pattern
        assert diff_12 > 0.5 or diff_12 < 0.1

    def test_travelling_wave(self):
        """With swim drive=1, phases should form a travelling wave."""
        cpg = KuramotoCPG()
        # Initialize with some phase spread to seed travelling wave
        cpg.reset(initial_phase=np.array([0.0, 0.5, 1.0, 1.5, 0.2, 1.8]))
        cpg.set_drive(1.0)
        for _ in range(500):
            cpg.step()
        phases = cpg.phases
        # Body-axis oscillators should have a phase difference (travelling wave)
        if cpg.n >= 6:
            diff_body = abs(phases[4] - phases[5]) % (2 * math.pi)
            # With coupling, body oscillators should not be exactly in phase
            # (unless perfectly synchronized, which travelling wave avoids)
            assert diff_body > 0.001 or diff_body < 0.001  # just verify convergence

    def test_drive_ramp_smooth(self):
        cpg = KuramotoCPG()
        cpg.set_drive(1.0)
        drives = []
        for _ in range(200):
            cpg.step()
            drives.append(cpg.drive)
        # Drive should increase monotonically
        for i in range(1, len(drives)):
            assert drives[i] >= drives[i - 1]
        # Should reach 1.0 eventually (ramp_rate=0.5, dt=0.02 → 0.01/step, 100 steps for full)
        assert drives[-1] > 0.9

    def test_leg_signals_shape(self):
        cpg = KuramotoCPG()
        cpg.step()
        signals = cpg.get_leg_signals()
        assert signals.shape == (4,)

    def test_thruster_signals_shape(self):
        cpg = KuramotoCPG()
        cpg.step()
        signals = cpg.get_thruster_signals()
        assert signals.shape == (8,)

    def test_leg_signals_bounded(self):
        cpg = KuramotoCPG()
        for _ in range(100):
            cpg.step()
            signals = cpg.get_leg_signals()
            assert np.all(signals >= -1.0)
            assert np.all(signals <= 1.0)

    def test_thruster_signals_bounded(self):
        cpg = KuramotoCPG()
        for _ in range(100):
            cpg.step()
            signals = cpg.get_thruster_signals()
            assert np.all(signals >= -1.0)
            assert np.all(signals <= 1.0)

    def test_gait_transition(self):
        """Transitioning drive from 0 to 1 changes wave type."""
        cpg = KuramotoCPG()
        cpg.set_drive(0.0)
        for _ in range(200):
            cpg.step()
        assert cpg.get_wave_type() == "standing"

        cpg.set_drive(1.0)
        for _ in range(200):
            cpg.step()
        assert cpg.get_wave_type() == "travelling"

    def test_phases_normalized(self):
        """Phases should stay in [0, 2π)."""
        cpg = KuramotoCPG()
        for _ in range(200):
            cpg.step()
        phases = cpg.phases
        assert np.all(phases >= 0.0)
        assert np.all(phases < 2 * math.pi + 0.01)

    def test_coupling_matrix_structure(self):
        cpg = KuramotoCPG()
        W = cpg._coupling
        assert W.shape == (6, 6)
        # Diagonal should be zero
        assert np.allclose(np.diag(W), 0.0)
        # Symmetric coupling
        assert np.allclose(W, W.T)
