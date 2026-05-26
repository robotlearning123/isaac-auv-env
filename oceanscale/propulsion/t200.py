"""Blue Robotics T200 thruster model ported from MarineGym.

Two-stage first-order dynamics: throttle filtering (tau ~0.43) then RPM
filtering (tau ~0.01). Piecewise-linear throttle-to-RPM map with ±0.075
deadband, piecewise-quadratic RPM-to-thrust curve from empirical data.
"""

# Adapted from MarineGym (https://github.com/Marine-RL/MarineGym)
# Original: marinegym/actuators/t200.py | License: MIT
# Paper: Chu et al., "MarineGym", IROS 2025
# Modifications: Ported from PyTorch to Warp GPU kernel, added batch support

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def _t200_step_kernel(
    cmd: wp.array(dtype=wp.float32),
    throttle_state: wp.array(dtype=wp.float32),
    rpm_state: wp.array(dtype=wp.float32),
    thrust_out: wp.array(dtype=wp.float32),
    dt: wp.float32,
    tau_up: wp.float32,
    tau_down: wp.float32,
    rpm_time_constant: wp.float32,
    force_constant_scale: wp.float32,
    max_rpm: wp.float32,
    deadband: wp.float32,
    slope_pos: wp.float32,
    intercept_pos: wp.float32,
    slope_neg: wp.float32,
    intercept_neg: wp.float32,
):
    i = wp.tid()
    target = wp.clamp(cmd[i], -1.0, 1.0)

    prev_throttle = throttle_state[i]
    tau = wp.where(target > prev_throttle, tau_up, tau_down)
    tau = wp.clamp(tau, 0.0, 1.0)
    new_throttle = prev_throttle + tau * (target - prev_throttle)
    throttle_state[i] = new_throttle

    target_rpm = 0.0
    if new_throttle > deadband:
        target_rpm = slope_pos * new_throttle + intercept_pos
    elif new_throttle < -deadband:
        target_rpm = slope_neg * new_throttle + intercept_neg

    alpha = wp.exp(-dt / rpm_time_constant)
    new_rpm = alpha * rpm_state[i] + (1.0 - alpha) * target_rpm
    new_rpm = wp.clamp(new_rpm, -max_rpm, max_rpm)
    rpm_state[i] = new_rpm

    rpm_sq = new_rpm * new_rpm
    if wp.abs(new_rpm) < 1.0:
        thrust_out[i] = 0.0
    elif new_rpm > 0.0:
        thrust = force_constant_scale * (4.7368e-07 * rpm_sq - 1.9275e-04 * new_rpm + 8.4452e-02)
        thrust_out[i] = thrust * 9.81
    else:
        thrust = force_constant_scale * (-3.8442e-07 * rpm_sq - 1.6186e-04 * new_rpm - 3.9139e-02)
        thrust_out[i] = thrust * 9.81


@dataclass(frozen=True)
class T200Config:
    """T200 thruster configuration from MarineGym BlueROV YAML."""

    num_rotors: int = 6
    force_constant: float = 4.4e-07
    moment_constant: float = 1.3678e-09
    max_rpm: float = 3900.0
    time_constant: float = 0.01
    tau_up: float = 0.43
    tau_down: float = 0.43
    deadband: float = 0.075
    slope_pos: float = 3659.9
    intercept_pos: float = 345.21
    slope_neg: float = 3494.4
    intercept_neg: float = -433.50

    @classmethod
    def from_marinegym_yaml(cls, path: str) -> T200Config:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        rc = data["rotor_configuration"]
        return cls(
            num_rotors=rc["num_rotors"],
            force_constant=rc["force_constants"][0],
            moment_constant=rc["moment_constants"][0],
            max_rpm=rc["max_rotation_velocities"][0],
            time_constant=rc["time_constants"][0],
        )


class T200Thruster:
    """GPU-batched T200 thruster bank for N environments × num_rotors."""

    def __init__(
        self,
        num_envs: int = 1,
        config: T200Config | None = None,
        dt: float = 0.02,
        device: str = "cuda:0",
    ) -> None:
        cfg = config or T200Config()
        self.cfg = cfg
        self.dt = dt
        self.device = device
        self.num_envs = num_envs
        total = num_envs * cfg.num_rotors
        self._throttle = wp.zeros(total, dtype=wp.float32, device=device)
        self._rpm = wp.zeros(total, dtype=wp.float32, device=device)
        self._thrust = wp.zeros(total, dtype=wp.float32, device=device)
        self._force_scale = cfg.force_constant / 4.4e-07

    def step(self, commands: wp.array) -> wp.array:
        """Advance thruster dynamics one timestep.

        Args:
            commands: flattened throttle commands [-1, 1], shape (num_envs * num_rotors,)

        Returns:
            Thrust forces in Newtons, same shape.
        """
        cfg = self.cfg
        wp.launch(
            _t200_step_kernel,
            dim=commands.shape[0],
            inputs=[
                commands,
                self._throttle,
                self._rpm,
                self._thrust,
                wp.float32(self.dt),
                wp.float32(cfg.tau_up),
                wp.float32(cfg.tau_down),
                wp.float32(cfg.time_constant),
                wp.float32(self._force_scale),
                wp.float32(cfg.max_rpm),
                wp.float32(cfg.deadband),
                wp.float32(cfg.slope_pos),
                wp.float32(cfg.intercept_pos),
                wp.float32(cfg.slope_neg),
                wp.float32(cfg.intercept_neg),
            ],
            device=self.device,
        )
        return self._thrust

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is None:
            self._throttle.zero_()
            self._rpm.zero_()
            self._thrust.zero_()
        else:
            t_np = self._throttle.numpy()
            r_np = self._rpm.numpy()
            s_np = self._thrust.numpy()
            for eid in env_ids:
                start = int(eid) * self.cfg.num_rotors
                end = start + self.cfg.num_rotors
                t_np[start:end] = 0.0
                r_np[start:end] = 0.0
                s_np[start:end] = 0.0
            self._throttle = wp.array(t_np, dtype=wp.float32, device=self.device)
            self._rpm = wp.array(r_np, dtype=wp.float32, device=self.device)
            self._thrust = wp.array(s_np, dtype=wp.float32, device=self.device)

    @property
    def rpm(self) -> np.ndarray:
        return self._rpm.numpy().reshape(self.num_envs, self.cfg.num_rotors)

    @property
    def throttle(self) -> np.ndarray:
        return self._throttle.numpy().reshape(self.num_envs, self.cfg.num_rotors)
