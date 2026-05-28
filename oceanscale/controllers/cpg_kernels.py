"""Warp GPU kernels for CPG-based amphibious force blending."""

from __future__ import annotations

try:
    import warp as wp
except ImportError:
    wp = None  # type: ignore[assignment]


@wp.func
def _smoothstep(t: wp.float32) -> wp.float32:
    """Hermite smoothstep: smooth transition from 0 to 1."""
    tc = wp.clamp(t, 0.0, 1.0)
    return tc * tc * (3.0 - 2.0 * tc)


@wp.kernel
def compute_blend_factor(
    body_q: wp.array(dtype=wp.transformf),
    body_f: wp.array(dtype=wp.spatial_vectorf),
    contact_force_threshold: wp.float32,
    water_surface_z: wp.float32,
    body_half_height: wp.float32,
    blend_out: wp.array(dtype=wp.float32),
):
    """Compute per-body walk/swim blend factor from contact forces + submersion.

    blend = smoothstep(submersion) * (1.0 - smoothstep(contact / threshold))
    0.0 = pure walking, 1.0 = pure swimming.
    """
    tid = wp.tid()
    pos_z = body_q[tid][2]

    # Ground contact force from body_f (Fz from spring-damper)
    fz = body_f[tid][2]
    contact_mag = wp.abs(fz)
    contact_factor = _smoothstep(contact_mag / contact_force_threshold)

    # Submersion fraction
    body_top = pos_z + body_half_height
    body_bot = pos_z - body_half_height
    if body_bot >= water_surface_z:
        submersion = wp.float32(0.0)
    elif body_top <= water_surface_z:
        submersion = wp.float32(1.0)
    else:
        submersion = (water_surface_z - body_bot) / (body_top - body_bot)

    # Blend: high submersion + low contact → swimming
    blend = _smoothstep(submersion) * (1.0 - contact_factor)
    blend_out[tid] = blend
