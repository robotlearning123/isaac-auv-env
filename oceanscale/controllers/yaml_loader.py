# Adapted from UUV Simulator (https://github.com/uuvsimulator/uuv_simulator)
# License: Apache-2.0
"""Load controller configurations from UUV Simulator-style YAML files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from oceanscale.controllers.lee_position import LeePositionConfig
from oceanscale.controllers.pid import PIDConfig

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_pid_yaml(path: str | Path) -> PIDConfig:
    """Load a PID controller YAML into a PIDConfig.

    Supports two formats:
    1. Flat: kp/ki/kd as 6-element arrays
    2. Cascaded (UUV Simulator): position_control/{pos_p,pos_i,pos_d} +
       position_control/{rot_p,rot_i,rot_d}
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if "kp" in data:
        return PIDConfig(
            kp=np.array(data["kp"], dtype=np.float32),
            ki=np.array(data.get("ki", [0] * 6), dtype=np.float32),
            kd=np.array(data["kd"], dtype=np.float32),
            sat=np.array(data["sat"], dtype=np.float32) if "sat" in data else None,
        )

    pos_p = data.get("position_control/pos_p", 0)
    pos_i = data.get("position_control/pos_i", 0)
    pos_d = data.get("position_control/pos_d", 0)
    rot_p = data.get("position_control/rot_p", 0)
    rot_i = data.get("position_control/rot_i", 0)
    rot_d = data.get("position_control/rot_d", 0)
    kp = np.array([pos_p, pos_p, pos_p, rot_p, rot_p, rot_p], dtype=np.float32)
    ki = np.array([pos_i, pos_i, pos_i, rot_i, rot_i, rot_i], dtype=np.float32)
    kd = np.array([pos_d, pos_d, pos_d, rot_d, rot_d, rot_d], dtype=np.float32)
    pos_sat = data.get("position_control/pos_sat")
    rot_sat = data.get("position_control/rot_sat")
    sat = None
    if pos_sat is not None or rot_sat is not None:
        sat = np.array([
            pos_sat or 1, pos_sat or 1, pos_sat or 1,
            rot_sat or 1, rot_sat or 1, rot_sat or 1,
        ], dtype=np.float32)
    return PIDConfig(kp=kp, ki=ki, kd=kd, sat=sat)


def load_lee_yaml(path: str | Path) -> LeePositionConfig:
    """Load a MarineGym-style Lee controller YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return LeePositionConfig(
        position_gain=np.array(data["position_gain"], dtype=np.float32),
        velocity_gain=np.array(data["velocity_gain"], dtype=np.float32),
        attitude_gain=np.array(data["attitude_gain"], dtype=np.float32),
        angular_rate_gain=np.array(data["angular_rate_gain"], dtype=np.float32),
    )


def load_rexrov_raw(name: str) -> dict:
    """Load raw YAML data from a RexROV controller config."""
    configs = rexrov_controller_configs()
    if name not in configs:
        raise ValueError(
            f"Unknown RexROV controller '{name}'. "
            f"Available: {sorted(configs.keys())}"
        )
    with open(configs[name]) as f:
        return yaml.safe_load(f)


def rexrov_controller_configs() -> dict[str, Path]:
    """Return available RexROV controller config paths."""
    ctrl_dir = _ASSETS_DIR / "rexrov" / "controllers"
    if not ctrl_dir.exists():
        return {}
    return {p.stem: p for p in sorted(ctrl_dir.glob("*.yaml"))}
