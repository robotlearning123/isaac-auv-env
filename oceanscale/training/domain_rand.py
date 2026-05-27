# Adapted from isaac-auv-env (https://github.com/warplab/isaac-auv-env)
# Original: warpauv_env.py domain_randomization class by Kevin Chang and Levi Cai (MIT WarpLab)
# License: BSD-3-Clause
# Paper: Lofaro et al., "Learning to Swim", arXiv 2410.00120
# Modifications: Extracted to standalone dataclass, added more randomization targets
"""Domain randomization for underwater RL training.

Randomizes physical parameters per environment to improve sim-to-real transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainRandomizationConfig:
    """Per-environment randomization ranges for underwater vehicle training."""

    enabled: bool = True

    volume_range: tuple[float, float] = (0.0134, 0.0134)
    mass_range: tuple[float, float] = (13.5, 13.5)
    com_to_cob_offset_radius: float = 0.0
    drag_scale_range: tuple[float, float] = (0.8, 1.2)
    thruster_noise_std: float = 0.0
    current_speed_range: tuple[float, float] = (0.0, 0.5)
    water_density_range: tuple[float, float] = (1020.0, 1030.0)

    @classmethod
    def no_randomization(cls) -> DomainRandomizationConfig:
        return cls(enabled=False)

    @classmethod
    def warpauv_defaults(cls) -> DomainRandomizationConfig:
        """Defaults from isaac-auv-env WarpAUV training."""
        return cls(
            enabled=True,
            volume_range=(0.01975, 0.02575),
            mass_range=(22.7, 22.7),
            com_to_cob_offset_radius=0.05,
            drag_scale_range=(0.8, 1.2),
            thruster_noise_std=0.02,
        )

    @classmethod
    def bluerov2_defaults(cls) -> DomainRandomizationConfig:
        """Conservative defaults for BlueROV2 Heavy."""
        return cls(
            enabled=True,
            volume_range=(0.0120, 0.0148),
            mass_range=(12.5, 14.5),
            com_to_cob_offset_radius=0.02,
            drag_scale_range=(0.85, 1.15),
            thruster_noise_std=0.03,
            current_speed_range=(0.0, 0.4),
            water_density_range=(1020.0, 1030.0),
        )


def sample_randomized_params(
    config: DomainRandomizationConfig,
    n_envs: int,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Sample randomized physical parameters for n_envs environments."""
    if rng is None:
        rng = np.random.default_rng()

    if not config.enabled:
        return {
            "volume": np.full(n_envs, np.mean(config.volume_range), dtype=np.float32),
            "mass": np.full(n_envs, np.mean(config.mass_range), dtype=np.float32),
            "com_to_cob_offset": np.zeros((n_envs, 3), dtype=np.float32),
            "drag_scale": np.ones(n_envs, dtype=np.float32),
            "thruster_noise": np.zeros(n_envs, dtype=np.float32),
            "current_speed": np.zeros(n_envs, dtype=np.float32),
            "water_density": np.full(n_envs, 1025.0, dtype=np.float32),
        }

    volume = rng.uniform(*config.volume_range, size=n_envs).astype(np.float32)
    mass = rng.uniform(*config.mass_range, size=n_envs).astype(np.float32)

    directions = rng.standard_normal((n_envs, 3)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions /= norms
    radii = config.com_to_cob_offset_radius * rng.uniform(0, 1, size=(n_envs, 1)).astype(np.float32) ** (1.0 / 3.0)
    com_to_cob_offset = (directions * radii).astype(np.float32)

    drag_scale = rng.uniform(*config.drag_scale_range, size=n_envs).astype(np.float32)
    thruster_noise = (config.thruster_noise_std * rng.standard_normal(n_envs)).astype(np.float32)
    current_speed = rng.uniform(*config.current_speed_range, size=n_envs).astype(np.float32)
    water_density = rng.uniform(*config.water_density_range, size=n_envs).astype(np.float32)

    return {
        "volume": volume,
        "mass": mass,
        "com_to_cob_offset": com_to_cob_offset,
        "drag_scale": drag_scale,
        "thruster_noise": thruster_noise,
        "current_speed": current_speed,
        "water_density": water_density,
    }
