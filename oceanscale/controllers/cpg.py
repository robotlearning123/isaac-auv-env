"""Central Pattern Generator (CPG) for amphibious gait control.

Implements a Kuramoto phase-oscillator network inspired by EPFL BioRob's
FARMS framework. A descending drive signal smoothly morphs the gait between
standing wave (walking trot) and travelling wave (swimming undulation).

Reference: Ijspeert et al., "From swimming to walking with a salamander robot
driven by a spinal cord model", Science 315:1416-1420, 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: smooth transition from 0 to 1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interpolate_gait_params(
    drive: float,
    freq_walk: float,
    freq_swim: float,
    amp_walk: float,
    amp_swim: float,
    phase_offset_walk: float,
    phase_offset_swim: float,
) -> tuple[float, float, float]:
    """Linearly interpolate frequency, amplitude, phase_offset by drive."""
    freq = freq_walk + drive * (freq_swim - freq_walk)
    amp = amp_walk + drive * (amp_swim - amp_walk)
    po = phase_offset_walk + drive * (phase_offset_swim - phase_offset_walk)
    return freq, amp, po


@dataclass(frozen=True)
class CPGConfig:
    """Kuramoto CPG parameters for amphibious gait generation."""

    n_oscillators: int = 6  # 4 legs (FL, FR, BL, BR) + 2 body-axis
    frequency_walk: float = 2.0  # Hz
    frequency_swim: float = 1.2  # Hz
    amplitude_walk: float = 1.0
    amplitude_swim: float = 0.8
    coupling_strength: float = 1.5
    phase_offset_walk: float = math.pi  # trot gait diagonal offset
    phase_offset_swim: float = math.pi / 3  # travelling wave delay
    drive_ramp_rate: float = 0.5  # drive change per second


class KuramotoCPG:
    """Kuramoto phase-oscillator network for amphibious gait generation.

    Oscillator indices:
        0 = FL (front-left leg)
        1 = FR (front-right leg)
        2 = BL (back-left leg)
        3 = BR (back-right leg)
        4 = body-axis fore
        5 = body-axis aft

    Drive signal: 0.0 = walking (standing wave), 1.0 = swimming (travelling wave).
    """

    def __init__(self, config: CPGConfig | None = None, dt: float = 0.02):
        self.config = config or CPGConfig()
        self.dt = dt
        self.n = self.config.n_oscillators

        self._phases = np.zeros(self.n, dtype=np.float64)
        self._omegas = np.full(self.n, self.config.frequency_walk * 2 * math.pi)
        self._coupling = self._build_coupling_matrix()
        self._drive = 0.0
        self._target_drive = 0.0

    def _build_coupling_matrix(self) -> np.ndarray:
        """Build phase coupling matrix.

        Layout (n=6):
          Legs: FL(0)-FR(1) diagonal pair, BL(2)-BR(3) diagonal pair.
          Cross-coupling between pairs for coordination.
          Body-axis (4,5): coupled to both leg pairs for wave propagation.
        """
        n = self.n
        w = np.zeros((n, n), dtype=np.float64)
        cs = self.config.coupling_strength

        # Leg diagonal pairs: FL(0)↔BR(3), FR(1)↔BL(2)
        w[0, 3] = cs
        w[3, 0] = cs
        w[1, 2] = cs
        w[2, 1] = cs

        # Same-side coupling (weaker)
        w[0, 2] = cs * 0.5
        w[2, 0] = cs * 0.5
        w[1, 3] = cs * 0.5
        w[3, 1] = cs * 0.5

        # Body-axis coupling to legs (bidirectional)
        if n >= 6:
            w[0, 4] = cs * 0.3
            w[4, 0] = cs * 0.3
            w[1, 4] = cs * 0.3
            w[4, 1] = cs * 0.3
            w[2, 5] = cs * 0.3
            w[5, 2] = cs * 0.3
            w[3, 5] = cs * 0.3
            w[5, 3] = cs * 0.3
            # Body-axis chain
            w[4, 5] = cs
            w[5, 4] = cs

        return w

    def reset(self, initial_phase: np.ndarray | None = None) -> None:
        """Reset oscillator phases."""
        if initial_phase is not None:
            self._phases = initial_phase.astype(np.float64).copy()
        else:
            self._phases = np.zeros(self.n, dtype=np.float64)
            # Initialize leg pairs with trot offset
            if self.n >= 4:
                self._phases[3] = self.config.phase_offset_walk
                self._phases[2] = self.config.phase_offset_walk * 0.5
        self._drive = 0.0
        self._target_drive = 0.0
        self._omegas = np.full(self.n, self.config.frequency_walk * 2 * math.pi)

    def set_drive(self, drive: float) -> None:
        """Set target descending drive signal [0=walk, 1=swim]."""
        self._target_drive = max(0.0, min(1.0, drive))

    def step(self) -> None:
        """Advance oscillators by one timestep.

        Kuramoto model: dθ_i/dt = ω_i + Σ_j w_ij * sin(θ_j - θ_i)
        """
        # Smooth drive ramping
        ramp_delta = self.config.drive_ramp_rate * self.dt
        if self._drive < self._target_drive:
            self._drive = min(self._drive + ramp_delta, self._target_drive)
        elif self._drive > self._target_drive:
            self._drive = max(self._drive - ramp_delta, self._target_drive)

        # Interpolate gait parameters by drive
        freq, amp, phase_offset = _interpolate_gait_params(
            self._drive,
            self.config.frequency_walk,
            self.config.frequency_swim,
            self.config.amplitude_walk,
            self.config.amplitude_swim,
            self.config.phase_offset_walk,
            self.config.phase_offset_swim,
        )

        omega_base = freq * 2.0 * math.pi

        # Phase coupling: Σ_j w_ij * sin(θ_j - θ_i)
        phase_diffs = self._phases[np.newaxis, :] - self._phases[:, np.newaxis]
        coupling = np.sum(self._coupling * np.sin(phase_diffs), axis=1)

        # Forward Euler integration
        self._phases += (omega_base + coupling) * self.dt
        # Normalize to [0, 2π)
        self._phases = self._phases % (2.0 * math.pi)

    def get_leg_signals(self) -> np.ndarray:
        """Return 4-dim per-leg activation signals in [-1, 1].

        Walking: sinusoidal oscillation with trot gait pattern.
        Swimming: attenuated (legs less active underwater).
        """
        leg_phases = self._phases[:4]
        _, amp, _ = _interpolate_gait_params(
            self._drive,
            self.config.frequency_walk,
            self.config.frequency_swim,
            self.config.amplitude_walk,
            self.config.amplitude_swim,
            self.config.phase_offset_walk,
            self.config.phase_offset_swim,
        )

        # Sinusoidal activation from phase
        signals = amp * np.sin(leg_phases)

        # Attenuate legs as swim drive increases (legs fold in water)
        swim_atten = 1.0 - 0.7 * self._drive
        signals *= swim_atten

        return np.clip(signals, -1.0, 1.0)

    def get_thruster_signals(self) -> np.ndarray:
        """Return 8-dim per-thruster activation signals in [-1, 1].

        Layout: [T1, T2, T3, T4 (horizontal), T5, T6, T7, T8 (vertical)].

        Walking: minimal thruster activation.
        Swimming: travelling wave across horizontal thrusters,
                  heave activation on vertical thrusters.
        """
        signals = np.zeros(8, dtype=np.float64)

        _, amp, phase_offset = _interpolate_gait_params(
            self._drive,
            self.config.frequency_walk,
            self.config.frequency_swim,
            self.config.amplitude_walk,
            self.config.amplitude_swim,
            self.config.phase_offset_walk,
            self.config.phase_offset_swim,
        )

        # Horizontal thrusters (T1-T4): body-axis travelling wave
        if self.n >= 6:
            # Phase from body-axis oscillators modulates horizontal thrusters
            body_fore = self._phases[4]
            body_aft = self._phases[5]

            swim_gain = self._drive * amp
            signals[0] = swim_gain * math.sin(body_fore)
            signals[1] = swim_gain * math.sin(body_fore + phase_offset * 0.5)
            signals[2] = swim_gain * math.sin(body_aft + phase_offset)
            signals[3] = swim_gain * math.sin(body_aft + phase_offset * 1.5)

        # Vertical thrusters (T5-T8): depth control signal
        swim_gain = self._drive * amp * 0.5
        signals[4] = swim_gain * math.sin(self._phases[0])
        signals[5] = swim_gain * math.sin(self._phases[1])
        signals[6] = swim_gain * math.sin(self._phases[2])
        signals[7] = swim_gain * math.sin(self._phases[3])

        return np.clip(signals, -1.0, 1.0)

    @property
    def drive(self) -> float:
        """Current drive signal value."""
        return self._drive

    @property
    def phases(self) -> np.ndarray:
        """Current oscillator phases."""
        return self._phases.copy()

    def get_wave_type(self) -> str:
        """Return current wave type based on drive."""
        if self._drive < 0.3:
            return "standing"
        elif self._drive > 0.7:
            return "travelling"
        return "transitioning"
