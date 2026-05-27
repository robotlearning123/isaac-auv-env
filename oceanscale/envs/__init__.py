"""OceanScale task environments."""

from oceanscale.envs.docking_env import DockingApproachEnv
from oceanscale.envs.station_keeping_env import CurrentStationKeepingEnv
from oceanscale.envs.waypoint_env import WaypointFollowingEnv

__all__ = [
    "CurrentStationKeepingEnv",
    "DockingApproachEnv",
    "WaypointFollowingEnv",
]
