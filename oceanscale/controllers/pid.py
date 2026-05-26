# Adapted from UUV Simulator (https://github.com/uuvsimulator/uuv_simulator)
# Original: uuv_control/uuv_trajectory_control + uuv_control_cascaded_pids
# License: Apache-2.0 | Manhaes et al., UUV Simulator, 2016
# Modifications: Rewritten from ROS nodes to standalone numpy, added vehicle presets
"""6-DOF PID controllers with vehicle-specific presets from UUV Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class PIDConfig:
    """PID gains for 6-DOF underwater vehicle control."""

    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    sat: np.ndarray | None = None

    PRESETS: ClassVar[dict[str, PIDConfig]] = {}

    @classmethod
    def rexrov_position(cls) -> PIDConfig:
        return PIDConfig(
            kp=np.array([3300, 3300, 3300, 3300, 3300, 330], dtype=np.float32),
            kd=np.array([1100, 1100, 1100, 1100, 1100, 1100], dtype=np.float32),
            ki=np.zeros(6, dtype=np.float32),
        )

    @classmethod
    def rexrov_nonlinear(cls) -> PIDConfig:
        return PIDConfig(
            kp=np.array([500, 500, 500, 300, 300, 300], dtype=np.float32),
            kd=np.array([50, 50, 50, 20, 20, 20], dtype=np.float32),
            ki=np.array([200, 200, 200, 100, 100, 100], dtype=np.float32),
        )

    @classmethod
    def rexrov_cascaded_position(cls) -> PIDConfig:
        return PIDConfig(
            kp=np.array([0.8, 0.8, 0.8, 0.3, 0.3, 0.3], dtype=np.float32),
            ki=np.array([0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001], dtype=np.float32),
            kd=np.array([0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001], dtype=np.float32),
            sat=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32),
        )

    @classmethod
    def rexrov_cascaded_velocity(cls) -> PIDConfig:
        return PIDConfig(
            kp=np.array([10, 10, 10, 10, 10, 10], dtype=np.float32),
            ki=np.array([2, 2, 2, 2, 2, 2], dtype=np.float32),
            kd=np.zeros(6, dtype=np.float32),
            sat=np.array([20, 20, 20, 5, 5, 5], dtype=np.float32),
        )

    @classmethod
    def bluerov2_hover(cls) -> PIDConfig:
        return PIDConfig(
            kp=np.array([1.1, 1.1, 0.25, 0.0, 0.0, 0.12], dtype=np.float32),
            kd=np.array([0.2, 0.2, 0.12, 0.0, 0.0, 0.0], dtype=np.float32),
            ki=np.zeros(6, dtype=np.float32),
        )


class PIDController:
    """Standalone 6-DOF PID controller for underwater vehicles."""

    def __init__(self, config: PIDConfig, dt: float = 0.02):
        self.config = config
        self.dt = dt
        self._integral = np.zeros(6, dtype=np.float32)
        self._prev_error = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        self._integral[:] = 0.0
        self._prev_error[:] = 0.0

    def compute(self, error: np.ndarray) -> np.ndarray:
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / max(self.dt, 1e-9)
        self._prev_error = error.copy()

        output = (
            self.config.kp * error
            + self.config.ki * self._integral
            + self.config.kd * derivative
        )

        if self.config.sat is not None:
            output = np.clip(output, -self.config.sat, self.config.sat)

        return output


class CascadedPIDController:
    """Two-loop cascaded PID: outer position → inner velocity."""

    def __init__(
        self,
        pos_config: PIDConfig | None = None,
        vel_config: PIDConfig | None = None,
        dt: float = 0.02,
    ):
        self.pos_pid = PIDController(
            pos_config or PIDConfig.rexrov_cascaded_position(), dt
        )
        self.vel_pid = PIDController(
            vel_config or PIDConfig.rexrov_cascaded_velocity(), dt
        )

    def reset(self) -> None:
        self.pos_pid.reset()
        self.vel_pid.reset()

    def compute(
        self, pos_error: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        vel_target = self.pos_pid.compute(pos_error)
        vel_error = vel_target - velocity
        return self.vel_pid.compute(vel_error)
