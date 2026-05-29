"""Wave probe (resistance gauge) for the Virtual FloWave digital twin.

Emulates the fixed wave gauges used in the physical FloWave tank to record
free-surface elevation time series at arbitrary (x, y) locations.

Pure NumPy; no USD/omni dependencies.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class WaveProbe:
    """Fixed free-surface elevation gauges at arbitrary tank locations.

    Parameters
    ----------
    gauge_xy : np.ndarray
        Shape (G, 2) float32 — fixed (x, y) gauge positions in tank coords (m).
    """

    def __init__(self, gauge_xy: np.ndarray) -> None:
        self._gauge_xy: np.ndarray = np.asarray(gauge_xy, dtype=np.float32)
        if self._gauge_xy.ndim != 2 or self._gauge_xy.shape[1] != 2:
            raise ValueError(
                f"gauge_xy must have shape (G, 2); got {self._gauge_xy.shape}"
            )
        self._times: list[float] = []
        self._eta_rows: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def gauge_xy(self) -> np.ndarray:
        """Shape (G, 2) gauge positions."""
        return self._gauge_xy

    def sample(
        self,
        eta_field: Callable[[np.ndarray, float], np.ndarray],
        t: float,
    ) -> np.ndarray:
        """Return η at all gauges for time t without recording.

        Parameters
        ----------
        eta_field : callable
            (xy: (M, 2) float32, t: float) -> np.ndarray (M,).
            Typically WaveSynth.eta_field.
        t : float
            Simulation time (s).

        Returns
        -------
        np.ndarray
            Shape (G,) surface elevation η (m) at each gauge.
        """
        return np.asarray(eta_field(self._gauge_xy, t), dtype=np.float64)

    def record(
        self,
        eta_field: Callable[[np.ndarray, float], np.ndarray],
        t: float,
    ) -> None:
        """Sample η at all gauges for time t and append to internal time series.

        Parameters
        ----------
        eta_field : callable
            (xy: (M, 2) float32, t: float) -> np.ndarray (M,).
        t : float
            Simulation time (s).
        """
        self._times.append(float(t))
        self._eta_rows.append(self.sample(eta_field, t))

    def as_dict(self) -> dict:
        """Return recorded time series as a plain dict.

        Returns
        -------
        dict with keys:
            times   : np.ndarray (T,)      simulation times (s)
            eta     : np.ndarray (T, G)    surface elevation (m)
            gauge_xy: np.ndarray (G, 2)    gauge positions (m)
            meta    : dict                  arbitrary metadata passed to save_npz
        """
        times = np.array(self._times, dtype=np.float64)
        eta = np.array(self._eta_rows, dtype=np.float64) if self._eta_rows else np.empty((0, len(self._gauge_xy)), dtype=np.float64)
        return {
            "times": times,
            "eta": eta,
            "gauge_xy": self._gauge_xy,
            "meta": getattr(self, "_meta", {}),
        }

    def save_npz(self, path: str, meta: dict) -> None:
        """Save recorded time series to a compressed .npz file.

        Parameters
        ----------
        path : str
            Output file path (will be created or overwritten).
        meta : dict
            Arbitrary metadata (e.g. {'H': 0.1, 'T': 2.0, 'depth': 2.0}).
            Stored as individual scalar arrays under key ``meta_<key>``.
        """
        self._meta = dict(meta)
        d = self.as_dict()
        meta_arrays = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
        np.savez_compressed(
            path,
            times=d["times"],
            eta=d["eta"],
            gauge_xy=d["gauge_xy"],
            **meta_arrays,
        )
