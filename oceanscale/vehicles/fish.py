# Adapted from fishsim (https://github.com/srl-ethz/fishsim)
# License: MIT (ETH Zürich, 2026)
# Paper: Michelis et al., arXiv 2602.23283 (2026)
# Content: Tendon-driven robot fish parametric model
"""Biomimetic fish robot vehicle configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oceanscale.hydro.mujoco_drag import MuJoCoDragParams


@dataclass(frozen=True)
class TendonFish:
    """Tendon-driven robot fish from ETH Zürich SRL fishsim.

    Validated via CMA-ES system identification against real hardware.
    MuJoCo fluid coefficients experimentally calibrated.
    """

    name: str = "TendonFish"
    mass: float = 1.5
    body_mass: float = 0.6
    electric_box_mass: float = 0.53
    tail_mass: float = 0.01
    acrylic_density: float = 1180.0
    pla_density: float = 1250.0

    motor_shaft_length: float = 0.12
    motor_arm_length: float = 0.0395
    tail_segment_length: float = 0.015
    center_of_mass: tuple[float, float, float] = (0.02457, 0.0, 0.00080)

    hinge_stiffness: float = 0.65
    hinge_damping: float = 0.0
    tendon_stiffness: float = 20000.0
    tendon_damping: float = 10.0
    tendon_routing: tuple[int, ...] = (0, 0, 0, 1, 1)

    fluid_shape: str = "ellipsoid"
    fluid_density: float = 1000.0
    fluid_viscosity: float = 0.0013
    fluid_coef: tuple[float, ...] = (0.4, 7.79, 2.81, 3.84, 0.27)

    @property
    def blunt_drag(self) -> float:
        return self.fluid_coef[0]

    @property
    def slender_drag(self) -> float:
        return self.fluid_coef[1]

    @property
    def angular_drag(self) -> float:
        return self.fluid_coef[2]

    @property
    def kutta_lift(self) -> float:
        return self.fluid_coef[3]

    @property
    def magnus_lift(self) -> float:
        return self.fluid_coef[4]

    def mujoco_drag_params(self) -> MuJoCoDragParams:
        from oceanscale.hydro.mujoco_drag import MuJoCoDragParams

        return MuJoCoDragParams(
            fluid_density=self.fluid_density,
            fluid_viscosity=self.fluid_viscosity,
            blunt_drag=self.blunt_drag,
            slender_drag=self.slender_drag,
            angular_drag=self.angular_drag,
            kutta_lift=self.kutta_lift,
            magnus_lift=self.magnus_lift,
        )
