"""Mission definitions for the Virtual FloWave BlueROV2 hero render.

v1 ships one mission: StationKeepingMission.
v1.1 will add DockApproach and WaypointFollowing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bluerov_in_tank import BlueROV2InTank


@dataclass
class StationKeepingMission:
    """Hold a fixed position in the tank for the hero render.

    The impeller current pushes the vehicle; the PD controller on BlueROV2InTank
    maintains the setpoint.  The gentle corrective thrust produces cinematic motion
    rather than random drift.

    Parameters
    ----------
    target_position : (float, float, float)
        World-frame station-keeping target (m).  Default (0, 0, 1.0):
        basin centre, mid-depth (SWL = 2.0 m; floor = 0.0 m).
    duration : float
        Mission duration (s).  Default 30.0 s for a 30 s hero clip at 24 fps.
    """

    target_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    duration: float = 30.0

    def setup(self, vehicle: "BlueROV2InTank") -> None:
        """Push target_position to the vehicle's PD controller.

        Call once before the render loop starts.

        Parameters
        ----------
        vehicle : BlueROV2InTank
        """
        import numpy as np
        vehicle.target_position = np.array(self.target_position, dtype=np.float64)

    def is_done(self, t: float) -> bool:
        """Return True once t >= duration.

        Parameters
        ----------
        t : float
            Current simulation time (s).
        """
        return t >= self.duration
