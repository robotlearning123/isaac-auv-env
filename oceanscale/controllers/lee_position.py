# Adapted from MarineGym (https://github.com/Marine-RL/MarineGym)
# Original: marinegym/controllers/lee_position_controller.py
# License: MIT | Paper: Lee et al., "Geometric tracking control", arXiv 1003.2005
# Modifications: Removed PyTorch/TorchRL dependency, pure numpy for standalone use
"""Lee geometric position controller for 6-DOF underwater vehicles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class LeePositionConfig:
    """Gains for the Lee geometric position controller."""

    position_gain: np.ndarray
    velocity_gain: np.ndarray
    attitude_gain: np.ndarray
    angular_rate_gain: np.ndarray

    PRESETS: ClassVar[dict[str, "LeePositionConfig"]] = {}

    @classmethod
    def bluerov_heavy(cls) -> LeePositionConfig:
        return LeePositionConfig(
            position_gain=np.array([6.0, 6.0, 6.0], dtype=np.float32),
            velocity_gain=np.array([4.7, 4.7, 4.7], dtype=np.float32),
            attitude_gain=np.array([3.0, 3.0, 0.15], dtype=np.float32),
            angular_rate_gain=np.array([0.52, 0.52, 0.18], dtype=np.float32),
        )


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float32)


class LeePositionController:
    """Lee geometric tracking controller (SO(3) error on rotation matrix)."""

    def __init__(self, config: LeePositionConfig | None = None):
        self.config = config or LeePositionConfig.bluerov_heavy()

    def compute(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation_quat: np.ndarray,
        angular_velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray = None,
        target_yaw: float = 0.0,
    ) -> np.ndarray:
        if target_velocity is None:
            target_velocity = np.zeros(3, dtype=np.float32)

        pos_error = position - target_position
        vel_error = velocity[:3] - target_velocity

        R = _quat_to_rotmat(orientation_quat)
        e3 = np.array([0, 0, 1], dtype=np.float32)

        force_des = (
            -self.config.position_gain * pos_error
            - self.config.velocity_gain * vel_error
        )

        thrust = float(np.dot(force_des, R @ e3))

        cy, sy = np.cos(target_yaw), np.sin(target_yaw)
        R_des = np.array(
            [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32
        )

        eR = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        eW = angular_velocity[:3]

        torques = (
            -self.config.attitude_gain * eR
            - self.config.angular_rate_gain * eW
        )

        return np.concatenate([[thrust], torques]).astype(np.float32)
