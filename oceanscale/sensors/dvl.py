"""DVL sensor stub — returns body-frame velocity + noise."""

from __future__ import annotations

import torch


class DVLSensor:
    """Doppler Velocity Log: measures body-frame linear velocity.

    Real DVLs have 4 beams for bottom-lock, but this stub simply
    returns the body-frame velocity with optional Gaussian noise.

    Args:
        n_envs: Number of parallel environments.
        device: Torch device (e.g. "cuda").
        noise_std: Gaussian noise standard deviation in m/s. 0 = deterministic.
    """

    dim: int = 3  # (vx, vy, vz) in body frame

    def __init__(self, n_envs: int, device: str = "cuda", noise_std: float = 0.0) -> None:
        self.n_envs = n_envs
        self.device = device
        self.noise_std = noise_std

    def read(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        quaternion: torch.Tensor,
        angular_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (n_envs, 3) body-frame velocity tensor."""
        vel_body = self._rotate_to_body(quaternion, velocity)

        if self.noise_std > 0.0:
            vel_body = vel_body + torch.randn_like(vel_body) * self.noise_std

        return vel_body

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
