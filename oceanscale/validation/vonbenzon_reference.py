"""Von Benzon 2022 reference-trajectory model — CPU-side interface for
Tier-1 Fossen parity validation.

Source: von Benzon et al. 2022, "An open-source BlueROV2 platform for
research and education", JMSE 10(12):1898, DOI 10.3390/jmse10121898.
License: CC-BY-4.0.  Cited as REF-VONBENZON22 in REFERENCES.md §6.

This module provides the *interface* only. The full Fossen 6-DOF +
thruster + tether dynamics from the Simulink model must be ported to
Python before generate_trajectory() returns real data.

Risk context: R13 (RISKS.md) — no validated AUV reference data available
today.  R23 — von Benzon Simulink simulator designated as the fallback
reference for Tier-1 unit tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class VonBenzonReferenceModel:
    """CPU-side reference model wrapping the von Benzon 2022 Simulink
    simulator for BlueROV2 Heavy.

    Purpose: generate ground-truth trajectories (pose, velocity) for
    parity-testing `oceanscale.hydro.tier1.Tier1` Warp kernels.

    Typical usage::

        ref = VonBenzonReferenceModel()
        # Once Simulink port is complete:
        traj = ref.generate_trajectory(thruster_cmd, initial_pose, dt, n_steps)
        np.savez("tests/data/vonbenzon_ref/const_thrust_surge.npz", **traj)
    """

    # BlueROV2 Heavy parameters from von Benzon 2022 Table X.
    # Placeholder values — must be populated from the paper during the
    # Simulink-to-Python port.
    _DEFAULT_PARAMS: dict[str, Any] = {
        "mass": None,  # (needs lookup from von Benzon 2022 §3.2)
        "volume": None,  # (needs lookup from von Benzon 2022 §3.3)
        "buoyancy_center_offset": None,  # (needs lookup from von Benzon 2022 §3.3)
        "added_mass_diag": None,  # 6-tuple, (needs lookup from von Benzon 2022 §4.1)
        "damping_lin_diag": None,  # 6-tuple, (needs lookup from von Benzon 2022 §4.2)
        "damping_quad_diag": None,  # 6-tuple, (needs lookup from von Benzon 2022 §4.2)
        "inertia_tensor_diag": None,  # 6-tuple, (needs lookup from von Benzon 2022 §3.2)
        "thruster_allocation_matrix": None,  # (6, 8), (needs lookup from von Benzon 2022 §5)
        "max_thrust_N": None,  # (needs lookup from von Benzon 2022 §5)
        "tether_model": None,  # dict or None, (needs lookup from von Benzon 2022 §6)
    }

    def __init__(self, vehicle_params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self._DEFAULT_PARAMS)
        if vehicle_params is not None:
            self.params.update(vehicle_params)

    def generate_trajectory(
        self,
        thruster_cmd: np.ndarray,
        initial_pose: np.ndarray,
        dt: float,
        n_steps: int,
    ) -> dict[str, np.ndarray]:
        """Run the von Benzon 6-DOF Fossen model for *n_steps* under a
        constant thruster command.

        Args:
            thruster_cmd: per-thruster commands, shape ``(n_thrusters,)``.
                Values in [-1, 1] normalised throttle.
            initial_pose: initial pose [x, y, z, roll, pitch, yaw],
                shape ``(6,)``.  Roll/pitch/yaw in radians.
            dt: integration timestep in seconds.
            n_steps: number of timesteps to simulate.

        Returns:
            dict with keys:
                ``'pose'``     — ``(n_steps, 6)`` pose time-series.
                ``'velocity'`` — ``(n_steps, 6)`` body-frame velocity.
                ``'time'``     — ``(n_steps,)`` monotonically increasing time.

        Raises:
            NotImplementedError: the Simulink→Python port is not yet done.
        """
        # TODO: Simulink-to-Python port of von Benzon 2022 Fossen 6-DOF model.
        #
        # Steps for the port (estimated 2-3 weeks):
        #   1. Extract M_RB (rigid-body mass + inertia) from Simulink
        #      "inertia" block — von Benzon 2022 §3.2.
        #   2. Extract M_A (added-mass diagonal) from "added mass" block —
        #      von Benzon 2022 §4.1.
        #   3. Extract D_lin, D_quad (damping) from "damping" blocks —
        #      von Benzon 2022 §4.2.
        #   4. Extract C_RB + C_A (Coriolis) — von Benzon 2022 §4.3.
        #   5. Extract restoring (buoyancy – weight) — von Benzon 2022 §3.3.
        #   6. Extract thruster allocation T ∈ R^{6×8} + saturation curve —
        #      von Benzon 2022 §5.
        #   7. Port tether model (if enabled) — von Benzon 2022 §6.
        #   8. Implement RK4 integrator matching Simulink ode4 solver.
        #   9. Validate against Simulink output for ≥3 test cases
        #      (surge, yaw, combined).
        #   10. Freeze reference trajectories as .npz in
        #       tests/data/vonbenzon_ref/.
        #
        # Risk: R23 (RISKS.md) — blocks v0.1 validation gate T3.3.
        raise NotImplementedError(
            "Von Benzon 2022 Simulink→Python port not yet complete. "
            "See REF-VONBENZON22 in REFERENCES.md §6 and R23 in RISKS.md. "
            "Port checklist: oceanscale/validation/vonbenzon_reference.py "
            "generate_trajectory() TODO."
        )

    def load_reference_dataset(self, path: Path) -> dict[str, np.ndarray]:
        """Load a pre-computed reference trajectory from an ``.npz`` file.

        The file must contain arrays keyed ``'pose'`` ``(N, 6)``,
        ``'velocity'`` ``(N, 6)``, and ``'time'`` ``(N,)``.

        Args:
            path: path to the ``.npz`` reference file.

        Returns:
            dict with ``'pose'``, ``'velocity'``, ``'time'`` arrays.
        """
        data = np.load(path)
        required = {"pose", "velocity", "time"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"reference dataset missing keys: {missing}")
        return {k: data[k] for k in required}
