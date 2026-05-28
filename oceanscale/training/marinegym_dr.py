# Adapted from MarineGym (https://github.com/Marine-RL/MarineGym)
# MIT License, Copyright (c) 2023 Botian Xu, Tsinghua University
# Source: cfg/task/randomization.yaml, cfg/task/disturbances.yaml
# See THIRD_PARTY_NOTICES.md
"""MarineGym-style domain randomization and flow disturbance configs.

Fine-grained randomization ranges for body hydrodynamics, rotor dynamics,
ocean current disturbances, and payload variations -- ported from MarineGym's
Hydra configs to structured dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarineGymBodyDR:
    """Per-parameter DR ranges for vehicle body hydrodynamics.

    Each tuple is (min_scale, max_scale) applied multiplicatively to the
    nominal parameter value.
    """

    mass_scale: tuple[float, float] = (0.8, 1.2)
    volume_scale: tuple[float, float] = (0.9, 1.1)
    cob_scale: tuple[float, float] = (0.5, 1.5)
    inertia_scale: tuple[float, float] = (0.8, 1.2)
    added_mass_scale: tuple[float, float] = (0.5, 1.0)
    linear_damping_scale: tuple[float, float] = (0.5, 1.0)
    quadratic_damping_scale: tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class MarineGymRotorDR:
    """DR ranges for thruster/rotor parameters."""

    time_constants_scale: tuple[float, float] = (0.8, 1.2)
    force_constants_scale: tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class FlowDisturbance:
    """Ocean current flow disturbance configuration.

    max_flow_velocity and noise are 6-DOF (surge, sway, heave, roll, pitch, yaw).
    """

    max_flow_velocity: tuple[float, ...] = (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    flow_noise: tuple[float, ...] = (0.1, 0.1, 0.1, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PayloadDisturbance:
    """Payload mass attachment disturbance."""

    mass_fraction: tuple[float, float] = (0.01, 0.2)
    z_offset_range: tuple[float, float] = (-0.1, 0.1)


@dataclass(frozen=True)
class MarineGymDRConfig:
    """Complete MarineGym-style DR configuration with train/eval presets."""

    body: MarineGymBodyDR = field(default_factory=MarineGymBodyDR)
    rotor: MarineGymRotorDR = field(default_factory=MarineGymRotorDR)
    flow: FlowDisturbance = field(default_factory=FlowDisturbance)
    payload: PayloadDisturbance = field(default_factory=PayloadDisturbance)

    @classmethod
    def train_defaults(cls) -> MarineGymDRConfig:
        """MarineGym training-time DR (broader ranges)."""
        return cls(
            body=MarineGymBodyDR(),
            rotor=MarineGymRotorDR(),
            flow=FlowDisturbance(),
            payload=PayloadDisturbance(),
        )

    @classmethod
    def eval_defaults(cls) -> MarineGymDRConfig:
        """MarineGym evaluation-time DR (wider body DR for robustness test)."""
        return cls(
            body=MarineGymBodyDR(
                volume_scale=(0.8, 1.2),
            ),
            rotor=MarineGymRotorDR(),
            flow=FlowDisturbance(),
            payload=PayloadDisturbance(),
        )

    @classmethod
    def no_randomization(cls) -> MarineGymDRConfig:
        """All scales fixed at 1.0."""
        return cls(
            body=MarineGymBodyDR(
                mass_scale=(1.0, 1.0),
                volume_scale=(1.0, 1.0),
                cob_scale=(1.0, 1.0),
                inertia_scale=(1.0, 1.0),
                added_mass_scale=(1.0, 1.0),
                linear_damping_scale=(1.0, 1.0),
                quadratic_damping_scale=(1.0, 1.0),
            ),
            rotor=MarineGymRotorDR(
                time_constants_scale=(1.0, 1.0),
                force_constants_scale=(1.0, 1.0),
            ),
            flow=FlowDisturbance(
                max_flow_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                flow_noise=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            payload=PayloadDisturbance(
                mass_fraction=(0.0, 0.0),
                z_offset_range=(0.0, 0.0),
            ),
        )
