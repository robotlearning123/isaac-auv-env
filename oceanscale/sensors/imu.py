"""IMU sensor stub — returns (accel_xyz, gyro_xyz) from Newton body state."""

from __future__ import annotations

import torch


class IMUSensor:
    """Inertial Measurement Unit: accelerometer + gyroscope in body frame.

    Args:
        n_envs: Number of parallel environments.
        device: Torch device (e.g. "cuda").
        noise_std: Gaussian noise standard deviation. 0 = deterministic.
    """

    dim: int = 6  # (accel_xyz, gyro_xyz)

    def __init__(self, n_envs: int, device: str = "cuda", noise_std: float = 0.0) -> None:
        self.n_envs = n_envs
        self.device = device
        self.noise_std = noise_std

    def read(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        linear_acceleration: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (n_envs, 6) tensor: [accel_body_xyz, gyro_xyz].

        Args:
            position: (n_envs, 3) world position.
            velocity: (n_envs, 3) world linear velocity.
            quaternion: (n_envs, 4) orientation (xyzw).
            angular_velocity: (n_envs, 3) angular velocity in body frame.
            linear_acceleration: (n_envs, 3) world linear acceleration.
                If None, derived from velocity via finite difference (zeros on first call).
        """
        if linear_acceleration is not None:
            accel_world = linear_acceleration
        else:
            accel_world = torch.zeros(self.n_envs, 3, device=self.device, dtype=velocity.dtype)

        accel_body = self._rotate_to_body(quaternion, accel_world)
        gyro = angular_velocity

        out = torch.cat([accel_body, gyro], dim=-1)

        if self.noise_std > 0.0:
            out = out + torch.randn_like(out) * self.noise_std

        return out

    @staticmethod
    def _rotate_to_body(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        t0 = 2.0 * (qy * vz - qz * vy)
        t1 = 2.0 * (qz * vx - qx * vz)
        t2 = 2.0 * (qx * vy - qy * vx)
        return torch.stack(
            [
                vx - qw * t0 - qy * t2 + qz * t1,
                vy - qw * t1 - qz * t0 + qx * t2,
                vz - qw * t2 - qx * t1 + qy * t0,
            ],
            dim=-1,
        )
