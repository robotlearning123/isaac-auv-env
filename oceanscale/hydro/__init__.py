"""Tier-1 Fossen 6-DOF hydrodynamics — GPU Warp kernels for Newton physics.

Adapted from MarineGym marinegym/robots/drone/underwaterVehicle.py
(MIT-licensed, Copyright (c) 2025 Shuguang Chu, Zhejiang University,
commit ebdca1bf 2026-01-21). See THIRD_PARTY_NOTICES.md.
"""

from oceanscale.hydro.tier1 import Tier1
from oceanscale.hydro.tier1_kernels import (
    tier1_accumulate_to_body_f,
    tier1_added_mass,
    tier1_coriolis_a,
    tier1_damping,
    tier1_restoring,
    tier1_thruster_alloc,
)

__all__ = [
    "Tier1",
    "tier1_accumulate_to_body_f",
    "tier1_added_mass",
    "tier1_coriolis_a",
    "tier1_damping",
    "tier1_restoring",
    "tier1_thruster_alloc",
]
