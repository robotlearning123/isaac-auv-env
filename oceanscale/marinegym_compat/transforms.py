"""Rotation-matrix and coordinate-frame transform utilities.

Pure PyTorch, no Isaac Sim / torchrl dependencies.
"""

from __future__ import annotations

from torch import Tensor

from oceanscale.marinegym_compat.math import (
    euler_to_quaternion,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


def euler_to_rotation_matrix(euler: Tensor) -> Tensor:
    """Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Args:
        euler: (..., 3) tensor of Euler angles in radians.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = euler_to_quaternion(euler)
    return quaternion_to_rotation_matrix(q)


def rotation_matrix_to_euler(R: Tensor) -> Tensor:
    """Extract Euler angles (roll, pitch, yaw) from rotation matrices.

    ZYX intrinsic convention via full Shepperd method with largest-diagonal
    branch for numerical stability at 180-degree rotations.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) Euler angles in radians.
    """
    import torch

    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    w = torch.zeros(n, device=R.device, dtype=R.dtype)
    x = torch.zeros(n, device=R.device, dtype=R.dtype)
    y = torch.zeros(n, device=R.device, dtype=R.dtype)
    z = torch.zeros(n, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    m1 = trace > 0
    s = (trace[m1] + 1.0).sqrt() * 2.0
    w[m1] = 0.25 * s
    x[m1] = (R_flat[m1, 2, 1] - R_flat[m1, 1, 2]) / s
    y[m1] = (R_flat[m1, 0, 2] - R_flat[m1, 2, 0]) / s
    z[m1] = (R_flat[m1, 1, 0] - R_flat[m1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    m2 = (~m1) & (R_flat[:, 0, 0] >= R_flat[:, 1, 1]) & (R_flat[:, 0, 0] >= R_flat[:, 2, 2])
    s = (1.0 + R_flat[m2, 0, 0] - R_flat[m2, 1, 1] - R_flat[m2, 2, 2]).clamp(min=1e-10).sqrt() * 2.0
    x[m2] = 0.25 * s
    w[m2] = (R_flat[m2, 2, 1] - R_flat[m2, 1, 2]) / s
    y[m2] = (R_flat[m2, 0, 1] + R_flat[m2, 1, 0]) / s
    z[m2] = (R_flat[m2, 0, 2] + R_flat[m2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    m3 = (~m1) & (~m2) & (R_flat[:, 1, 1] >= R_flat[:, 2, 2])
    s = (1.0 + R_flat[m3, 1, 1] - R_flat[m3, 0, 0] - R_flat[m3, 2, 2]).clamp(min=1e-10).sqrt() * 2.0
    y[m3] = 0.25 * s
    w[m3] = (R_flat[m3, 0, 2] - R_flat[m3, 2, 0]) / s
    x[m3] = (R_flat[m3, 0, 1] + R_flat[m3, 1, 0]) / s
    z[m3] = (R_flat[m3, 1, 2] + R_flat[m3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    m4 = (~m1) & (~m2) & (~m3)
    s = (1.0 + R_flat[m4, 2, 2] - R_flat[m4, 0, 0] - R_flat[m4, 1, 1]).clamp(min=1e-10).sqrt() * 2.0
    z[m4] = 0.25 * s
    w[m4] = (R_flat[m4, 1, 0] - R_flat[m4, 0, 1]) / s
    x[m4] = (R_flat[m4, 0, 2] + R_flat[m4, 2, 0]) / s
    y[m4] = (R_flat[m4, 1, 2] + R_flat[m4, 2, 1]) / s

    q = torch.stack([w, x, y, z], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    q = q.reshape(*batch_shape, 4)
    return quaternion_to_euler(q)
