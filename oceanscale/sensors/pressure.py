"""Pressure/depth sensor stub — returns depth from body z-position + noise."""

from __future__ import annotations

import torch

_GRAVITY = 9.81
_WATER_DENSITY = 1025.0  # seawater kg/m^3


class PressureSensor:
    """Depth sensor: converts z-position to depth (positive downward).

    Convention: z=0 is water surface, negative z is underwater.
    Output depth = -z (clamped >= 0).

    Args:
        n_envs: Number of parallel environments.
        device: Torch device (e.g. "cuda").
        noise_std: Gaussian noise standard deviation in meters. 0 = deterministic.
    """

    dim: int = 1  # depth scalar

    def __init__(
        self, n_envs: int, device: str = "cuda", noise_std: float = 0.0
    ) -> None:
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
        """Return (n_envs, 1) depth tensor in meters."""
        depth = torch.clamp(-position[:, 2:3], min=0.0)

        if self.noise_std > 0.0:
            depth = depth + torch.randn_like(depth) * self.noise_std

        return depth
