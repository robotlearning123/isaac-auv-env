"""MarineGym compatibility layer.

Pure-PyTorch quaternion, rotation, tensor, and sensor utilities ported from
MarineGym (MIT License, Copyright (c) 2023 Botian Xu, Tsinghua University).
No Isaac Sim / torchrl / tensordict dependencies.

Quaternion convention: (w, x, y, z) — aerospace / scalar-first.
"""

__version__ = "0.1.0a0"

from oceanscale.marinegym_compat.math import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_to_quaternion,
    normalize,
    off_diag,
    quat_axis,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    quat_mul,
    symexp,
    symlog,
)
from oceanscale.marinegym_compat.controller import (
    AttitudeController,
    ControllerBase,
    LeePositionController,
    RateController,
    compute_parameters,
)
from oceanscale.marinegym_compat.sensor import (
    FisheyeCameraCfg,
    PinholeCameraCfg,
    is_isaaclab_available,
    load_camera_cfg_from_dict,
    orientation_from_view,
    to_camel_case,
    to_isaaclab_camera_cfg,
)
from oceanscale.marinegym_compat.thruster import (
    RotorConfig,
    RotorGroupModel,
    T200Thruster,
)

__all__ = [
    "__version__",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "AttitudeController",
    "compute_parameters",
    "ControllerBase",
    "euler_to_quaternion",
    "FisheyeCameraCfg",
    "is_isaaclab_available",
    "LeePositionController",
    "load_camera_cfg_from_dict",
    "normalize",
    "off_diag",
    "orientation_from_view",
    "PinholeCameraCfg",
    "quat_axis",
    "quat_mul",
    "quat_rotate",
    "quat_rotate_inverse",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "RateController",
    "RotorConfig",
    "RotorGroupModel",
    "symexp",
    "symlog",
    "T200Thruster",
    "to_camel_case",
    "to_isaaclab_camera_cfg",
]
