"""Biésel flap transfer function + Snake principle + Miles-Funke single-summation
for the FloWave-TT 168-paddle circular array.

Math reference: reports/research/07_paddle_wave_theory.md
Facility spec:  reports/research/01_flowave_facility.md §2
Architecture:   docs/virtual_flowave_architecture.md §2.2
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


class WaveSynth(NamedTuple):
    """Return value of FlapPaddleArray.synthesize_irregular.

    Both callables share the same (ω_j, θ_j, a_j, k_j, ε_j) random samples so
    paddle_commands and eta_field describe an identical physical wave field.

    Attributes
    ----------
    paddle_commands : Callable[[float], np.ndarray]
        t → s_n(t), shape (N,) per-paddle surface displacement (m).
    eta_field : Callable[[np.ndarray, float], np.ndarray]
        (xy, t) → η(x,y,t), shape (M,) surface elevation (m) for xy of shape (M, 2).
    """

    paddle_commands: Callable[[float], np.ndarray]
    eta_field: Callable[[np.ndarray, float], np.ndarray]

    def eta_grid(
        self,
        nx: int = 256,
        ny: int = 256,
        t: float = 0.0,
        extent: float = 12.0,
    ) -> np.ndarray:
        """Evaluate ``eta_field`` on a regular tank meshgrid.

        Returns ``(ny, nx)`` surface elevation over ``[-extent, extent]^2``.
        """
        xs = np.linspace(-extent, extent, nx, dtype=np.float32)
        ys = np.linspace(-extent, extent, ny, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        xy = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        return self.eta_field(xy, t).reshape(ny, nx)


class FlapPaddleArray:
    """Wave-field generator for a circular bottom-hinged flap paddle array.

    Converts a target directional wave spectrum S(ω, θ) into per-paddle
    surface displacement commands s_n(t), n ∈ {0..N-1}.

    All angles in radians; frequencies in rad/s; lengths in metres.

    Parameters
    ----------
    R : float
        Paddle ring radius (m). FloWave: 12.5 m (25 m Ø / 2).
    N : int
        Number of paddles. FloWave: 168.
    h : float
        Water depth (m). FloWave: 2.0 m.
    hinge_depth : float
        Vertical distance from SWL to hinge (m). Determines S = θ_pp × hinge_depth.
        Sonnet review fix N7: S is peak-to-peak linear displacement at the free surface.
    g : float
        Gravitational acceleration (m/s²).
    """

    def __init__(
        self,
        R: float = 12.5,
        N: int = 168,
        h: float = 2.0,
        hinge_depth: float = 1.9,
        g: float = 9.81,
    ) -> None:
        self.R = float(R)
        self.N = int(N)
        self.h = float(h)
        self.hinge_depth = float(hinge_depth)
        self.g = float(g)

        # Paddle azimuth angles φ_n = 2π·n / N, CCW from +x (architecture §2.1)
        self.phi_n: np.ndarray = 2.0 * np.pi * np.arange(N) / N  # shape (N,)

    # ------------------------------------------------------------------
    # 1. Finite-depth dispersion
    # ------------------------------------------------------------------

    def dispersion(self, omega: np.ndarray) -> np.ndarray:
        """Invert ω² = g·k·tanh(k·h) via Newton-Raphson.

        Uses the Padé initial guess from CERC-85-4 (report 07 §2) to reach
        the propagating-mode wavenumber k in O(5) iterations to 1e-10 precision.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies (rad/s), shape (J,).

        Returns
        -------
        np.ndarray
            Wavenumbers k (1/m), same shape as omega.
        """
        # TODO: Warp kernel — compile dispersion_kernel(omega[:], k[:]) with
        #       wp.launch for GPU-vectorised batch evaluation (architecture §2.2).

        omega = np.asarray(omega, dtype=np.float64)
        scalar = omega.ndim == 0
        omega = np.atleast_1d(omega)

        h = self.h
        g = self.g

        # Padé initial guess (CERC-85-4): kh ≈ x / (1 - exp(-5/2 · x^(2/5)))^(5/2), x = ω²h/g
        x = omega**2 * h / g
        # Guard against x=0
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_term = np.where(x > 0, np.exp(-2.5 * x**0.4), 0.0)
            denom = np.where(x > 0, (1.0 - exp_term) ** 2.5, 1.0)
            kh0 = np.where(x > 0, x / denom, 0.0)
        k = kh0 / h

        # Newton-Raphson: f(k) = k·tanh(k·h) - ω²/g = 0
        for _ in range(20):
            th = np.tanh(k * h)
            sech2 = 1.0 - th**2  # sech²(kh)
            f = k * th - omega**2 / g
            df = th + k * h * sech2
            dk = f / df
            k -= dk
            if np.all(np.abs(dk) < 1e-10 * np.maximum(k, 1e-12)):
                break

        k = np.abs(k)  # ensure positive
        return k[0] if scalar else k

    # ------------------------------------------------------------------
    # 2. Biésel flap transfer function
    # ------------------------------------------------------------------

    def transfer_fn_HS(self, omega: np.ndarray) -> np.ndarray:
        """Bottom-hinged flap H/S for a given angular-frequency array.

        H/S = 4·sinh(kh)·[kh·sinh(kh) − cosh(kh) + 1] / [kh·(sinh(2kh) + 2kh)]

        H is peak-to-peak far-field wave height; S is peak-to-peak surface
        displacement at the paddle face (= θ_pp × hinge_depth, per N7).
        Deep-water limit H/S → 2 as kh → ∞ (a flap radiates finite waves for
        finite stroke); the earlier 4(cosh kh−1)/(sinh 2kh+2kh) form was wrong
        (→ 0 in deep water).

        Citation: Dean & Dalrymple (1991), Water Wave Mechanics §6.3
        (bottom-hinged flap, full-draft, stroke at SWL); Galvin (1964).

        Parameters
        ----------
        omega : np.ndarray
            Angular frequencies (rad/s).

        Returns
        -------
        np.ndarray
            H/S (dimensionless), same shape as omega.
        """
        # TODO: Warp kernel — transfer_fn_kernel(k[:], h, HS[:]) on GPU.

        omega = np.asarray(omega, dtype=np.float64)
        scalar = omega.ndim == 0
        omega = np.atleast_1d(omega)

        k = self.dispersion(omega)
        kh = k * self.h

        numerator = 4.0 * np.sinh(kh) * (kh * np.sinh(kh) - np.cosh(kh) + 1.0)
        denominator = kh * (np.sinh(2.0 * kh) + 2.0 * kh)

        HS = numerator / denominator
        return float(HS[0]) if scalar else HS

    # ------------------------------------------------------------------
    # 3. Snake principle — monochromatic command
    # ------------------------------------------------------------------

    def snake_command(
        self,
        amplitude_a: float,
        omega: float,
        theta_target: float,
        eps: float = 0.0,
    ) -> np.ndarray:
        """Per-paddle peak-to-peak surface displacement for a monochromatic wave.

        Biésel (1954) Snake principle:
          s_n(t=0) = (a / (H/S)) · cos(k·R·cos(θ − φ_n) + ε)

        The caller uses the returned array as cosine amplitudes; at time t the
        full command is s_n · cos(−ω·t) = Re{ s_n · exp(−iωt) }.

        Citation: report 07 §4.2; MARINET D2.12 §5.1.

        Parameters
        ----------
        amplitude_a : float
            Target wave amplitude a (m), i.e. half peak-to-trough height H/2.
        omega : float
            Angular frequency (rad/s).
        theta_target : float
            Target wave propagation direction (rad), measured CCW from +x.
        eps : float
            Constant phase offset (rad).

        Returns
        -------
        np.ndarray
            Shape (N,) peak-to-peak surface displacement S_n (m) at t = 0.
        """
        # TODO: Warp kernel — snake_kernel(phi_n[:], k, R, theta, HS, a, eps, s_n[:]).

        k = float(self.dispersion(np.array([omega]))[0])
        HS = float(self.transfer_fn_HS(np.array([omega]))[0])

        # Phase for each paddle: k·R·cos(θ − φ_n)
        phase = k * self.R * np.cos(theta_target - self.phi_n)
        S_n = (amplitude_a / HS) * np.cos(phase + eps)
        return S_n

    # ------------------------------------------------------------------
    # 4. Miles-Funke single-summation for irregular directional spectra
    # ------------------------------------------------------------------

    def synthesize_irregular(
        self,
        spectrum_func: Callable[[float, float], float],
        n_freqs: int = 128,
        theta_func: Callable[[float], float] | None = None,
        omega_range: tuple[float, float] = (0.3, 8.0),
        seed: int = 0,
    ) -> WaveSynth:
        """Miles-Funke (1989) single-summation for a directional irregular spectrum.

        Each frequency component ω_j is assigned exactly one direction θ_j,
        preventing phase-locking (report 07 §5.2).

          η(x,y,t) = Σ_j a_j · cos(k_j·x·cosθ_j + k_j·y·sinθ_j − ω_j·t + ε_j)

          a_j = √(2·S(ω_j, θ_j)·Δω·Δθ)   [report 07 eq. §5.2]

        Paddle command for component j follows Snake principle (§4.2):
          S_{n,j} = (a_j / HS_j) · cos(k_j·R·cos(θ_j − φ_n) + ε_j)

        Total:
          s_n(t) = Σ_j S_{n,j} · cos(−ω_j·t)

        Both paddle_commands and eta_field share the SAME (ω_j, θ_j, a_j, k_j, ε_j)
        random samples, ensuring physical consistency between the paddle stroke and
        the resulting wave field.

        Citation: Miles & Funke (1989); report 07 §5.2 + §6.1.

        Parameters
        ----------
        spectrum_func : Callable[[float, float], float]
            S(ω, θ) in m²·s/rad. Called once per frequency bin.
        n_freqs : int
            Number of frequency bins J.
        theta_func : Callable[[float], float] | None
            Maps ω → θ_j (rad). If None, samples uniformly in [0, 2π).
        omega_range : tuple[float, float]
            (ω_min, ω_max) in rad/s.
        seed : int
            RNG seed for random phases ε_j and fallback θ_j sampling.

        Returns
        -------
        WaveSynth
            NamedTuple with paddle_commands and eta_field callables.
            paddle_commands: t → s_n(t), shape (N,). Pure NumPy.
            eta_field: (xy, t) → η, shape (M,) for xy of shape (M, 2).
        """
        # TODO: Warp kernel — synthesize_kernel(t, omega_j[:], k_j[:], HS_j[:],
        #       amp_j[:], theta_j[:], eps_j[:], phi_n[:], R, s_n[:]) for real-time
        #       GPU evaluation (architecture §2.2 final paragraph).

        rng = np.random.default_rng(seed)

        omega_min, omega_max = omega_range
        delta_omega = (omega_max - omega_min) / n_freqs
        # Centre frequencies
        omega_j = omega_min + (np.arange(n_freqs) + 0.5) * delta_omega  # (J,)

        # One direction per frequency (single-summation — avoids phase locking)
        delta_theta = 2.0 * np.pi  # integrating over full circle per component
        if theta_func is not None:
            theta_j = np.array([theta_func(w) for w in omega_j])
        else:
            theta_j = rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)

        # Spectral amplitudes
        S_j = np.array([spectrum_func(float(w), float(t)) for w, t in zip(omega_j, theta_j)])

        # a_j = √(2·S(ω_j,θ_j)·Δω·Δθ)  [report 07 §5.2]
        amp_j = np.sqrt(2.0 * np.maximum(S_j, 0.0) * delta_omega * delta_theta)  # (J,)

        # Random phases ε_j ∈ [0, 2π)
        eps_j = rng.uniform(0.0, 2.0 * np.pi, size=n_freqs)

        # Pre-compute wavenumbers and transfer functions for all frequencies
        k_j = self.dispersion(omega_j)  # (J,)
        HS_j = self.transfer_fn_HS(omega_j)  # (J,)

        # Pre-compute per-paddle per-frequency spatial phase: k_j · R · cos(θ_j − φ_n)
        # Shape: (J, N) — outer product of k_j and cos(θ_j − φ_n)
        dangle = theta_j[:, np.newaxis] - self.phi_n[np.newaxis, :]  # (J, N)
        spatial_phase = k_j[:, np.newaxis] * self.R * np.cos(dangle)  # (J, N)

        # Paddle amplitude per component: (a_j / HS_j) broadcast to (J, N)
        S_nj = (amp_j / HS_j)[:, np.newaxis] * np.cos(spatial_phase + eps_j[:, np.newaxis])  # (J, N)

        def _command_at_t(t: float) -> np.ndarray:
            """Returns shape (N,) paddle surface displacement at time t (m)."""
            # s_n(t) = Σ_j S_{n,j} · cos(−ω_j · t) = Σ_j S_{n,j} · cos(ω_j · t)
            # (cosine is even; sign of ωt doesn't matter for real signal)
            time_phase = np.cos(omega_j * t)  # (J,)
            return S_nj.T @ time_phase  # (N, J) · (J,) → (N,)

        def _eta_field(xy: np.ndarray, t: float) -> np.ndarray:
            """Miles-Funke η(x,y,t) at arbitrary field points.

            η(x,y,t) = Σ_j a_j · cos(k_j·x·cosθ_j + k_j·y·sinθ_j − ω_j·t + ε_j)

            Uses the SAME (ω_j, θ_j, a_j, k_j, ε_j) as paddle_commands.

            Parameters
            ----------
            xy : np.ndarray
                Shape (M, 2) with columns [x, y] in metres.
            t : float
                Time in seconds.

            Returns
            -------
            np.ndarray
                Shape (M,) surface elevation η (m).
            """
            # TODO: Warp kernel — eta_kernel(xy[:], omega_j[:], k_j[:], theta_j[:],
            #       amp_j[:], eps_j[:], t, eta[:]) for GPU batch evaluation.
            xy = np.asarray(xy, dtype=np.float64)  # (M, 2)
            x = xy[:, 0]  # (M,)
            y = xy[:, 1]  # (M,)

            # Phase at each field point for each component: k_j·x·cosθ_j + k_j·y·sinθ_j
            # Shape: (J, M)
            phase_xy = (
                k_j[:, np.newaxis] * np.cos(theta_j[:, np.newaxis]) * x[np.newaxis, :]
                + k_j[:, np.newaxis] * np.sin(theta_j[:, np.newaxis]) * y[np.newaxis, :]
            )  # (J, M)

            # Full argument: phase_xy − ω_j·t + ε_j, shape (J, M)
            full_phase = phase_xy - omega_j[:, np.newaxis] * t + eps_j[:, np.newaxis]

            # η = Σ_j a_j · cos(full_phase_j)
            return amp_j @ np.cos(full_phase)  # (J,) · (J, M) → (M,)

        return WaveSynth(paddle_commands=_command_at_t, eta_field=_eta_field)

    # ------------------------------------------------------------------
    # 5. Regular (single-frequency) Airy wave
    # ------------------------------------------------------------------

    def synthesize_regular(
        self,
        H: float,
        T: float,
        depth: float | None = None,
        direction: float = 0.0,
    ) -> WaveSynth:
        """Synthesize a single-frequency (regular) Airy wave.

        Uses the same finite-depth dispersion and Biésel-Suquet transfer the
        class already provides; the commanded paddle stroke is inverted from the
        target height H so the realised free-surface amplitude is H/2.

          η(x,y,t) = (H/2)·cos(k·x·cosβ + k·y·sinβ − ω·t)
          S        = H / (H/S)            (peak-to-peak paddle stroke)

        Paddle commands reuse the Snake-principle mapping of :meth:`snake_command`
        (evaluated at amplitude a = H/2) modulated by cos(ω·t), identical to the
        time evolution used by :meth:`synthesize_irregular`.

        Parameters
        ----------
        H : float
            Target wave height (peak-to-trough, m).
        T : float
            Wave period (s).
        depth : float | None
            Still-water depth (m). If None, uses the array's own depth ``self.h``.
        direction : float
            Wave propagation direction β (rad), CCW from +x.

        Returns
        -------
        WaveSynth
            NamedTuple with the same shape contract as synthesize_irregular:
            paddle_commands: t → s_n(t), shape (N,);
            eta_field: (xy, t) → η, shape (M,) for xy of shape (M, 2).
        """
        omega = 2.0 * np.pi / T
        omega_arr = np.array([omega])

        # Reuse the class finite-depth physics; allow a per-call depth override
        # via self.h without permanently mutating shared state.
        prev_h = self.h
        try:
            if depth is not None:
                self.h = float(depth)
            k = float(self.dispersion(omega_arr)[0])
            HS = float(self.transfer_fn_HS(omega_arr)[0])
            # Snake-principle cosine amplitudes for this monochromatic wave at a = H/2.
            S_n = self.snake_command(0.5 * H, omega, direction)
        finally:
            self.h = prev_h

        # Peak-to-peak paddle stroke that inverts Biésel-Suquet for target H.
        S = H / HS if HS != 0.0 else 0.0  # noqa: F841 — reported in WaveSynth docs/measured numbers

        kx = k * np.cos(direction)
        ky = k * np.sin(direction)

        def _command_at_t(t: float) -> np.ndarray:
            """Returns shape (N,) paddle surface displacement at time t (m)."""
            return S_n * np.cos(omega * t)

        def _eta_field(xy: np.ndarray, t: float) -> np.ndarray:
            """Airy single-component η(x,y,t), shape (M,) for xy of shape (M, 2)."""
            xy = np.asarray(xy, dtype=np.float64)
            x = xy[:, 0]
            y = xy[:, 1]
            return (0.5 * H) * np.cos(kx * x + ky * y - omega * t)

        return WaveSynth(paddle_commands=_command_at_t, eta_field=_eta_field)

    # ------------------------------------------------------------------
    # 6. Concentric spike — axisymmetric time-focused converging wave
    # ------------------------------------------------------------------

    def synthesize_focused_spike(
        self,
        amplitude: float = 0.15,
        t_focus: float = 3.0,
        omega_range: tuple[float, float] = (2.0, 6.0),
        n_freqs: int = 32,
        depth: float | None = None,
    ) -> WaveSynth:
        """Axisymmetric time-focused converging wave — the FloWave "concentric spike".

        All N paddles move IDENTICALLY (axisymmetric, no directional phase), so the
        whole ring drives inward-converging circular wavefronts.  A band of
        frequencies is phased so every component crest reaches the basin centre
        (r = 0) at the same instant t = t_focus, producing a sharp transient
        central spike ringed by concentric crests — the iconic FloWave demo.

        Interior axisymmetric (regular-at-origin) solution of the Helmholtz
        equation is the J₀ Fourier-Bessel mode (Hankel would diverge at r=0):

          η(r, t) = Σ_j a_j · J₀(k_j·r) · cos(ω_j·(t − t_focus))

        At (r=0, t=t_focus): J₀(0)=1 and cos(0)=1 for every component, so the
        packet adds in phase → a focused central elevation ≈ Σ_j a_j = amplitude.
        Away from the focus in space (J₀ decays, oscillates) or time (components
        dephase) the elevation drops sharply.

        Citation: Fourier-Bessel / axisymmetric focusing; FloWave concentric-wave
        demonstration. Biésel inversion of the paddle stroke via transfer_fn_HS.

        Parameters
        ----------
        amplitude : float
            Target focal elevation at (r=0, t=t_focus), metres.
        t_focus : float
            Time at which all components focus at the centre, seconds.
        omega_range : tuple[float, float]
            (ω_min, ω_max) frequency band of the converging packet, rad/s.
        n_freqs : int
            Number of frequency components.
        depth : float | None
            Still-water depth (m). If None, uses the array's own depth ``self.h``.

        Returns
        -------
        WaveSynth
            paddle_commands: t → s_n(t), shape (N,) (all paddles equal —
            axisymmetric); eta_field: (xy, t) → η, shape (M,), axisymmetric in
            r = hypot(x, y).
        """
        from scipy.special import j0

        omega_j = np.linspace(omega_range[0], omega_range[1], n_freqs)  # (J,)
        # Equal weights so the in-phase central sum equals the target amplitude.
        a_j = np.full(n_freqs, amplitude / n_freqs)  # (J,)

        # Reuse the class finite-depth physics; per-call depth override without
        # permanently mutating shared state.
        prev_h = self.h
        try:
            if depth is not None:
                self.h = float(depth)
            k_j = self.dispersion(omega_j)  # (J,)
            HS_j = self.transfer_fn_HS(omega_j)  # (J,)
        finally:
            self.h = prev_h

        # Paddle stroke per component (Biésel inversion). All paddles identical.
        S_j = a_j / HS_j  # (J,)

        def _command_at_t(t: float) -> np.ndarray:
            """Shape (N,) paddle surface displacement at time t (m); all equal."""
            s = float(np.sum(S_j * np.cos(omega_j * (t - t_focus))))
            return np.full(self.N, s)

        def _eta_field(xy: np.ndarray, t: float) -> np.ndarray:
            """Axisymmetric J₀ converging field η(x,y,t), shape (M,) for xy (M, 2)."""
            xy = np.asarray(xy, dtype=np.float64)
            r = np.hypot(xy[:, 0], xy[:, 1])  # (M,)
            phase = np.cos(omega_j * (t - t_focus))  # (J,)
            bessel = j0(np.outer(r, k_j))  # (M, J)
            return bessel @ (a_j * phase)  # (M,)

        return WaveSynth(paddle_commands=_command_at_t, eta_field=_eta_field)
