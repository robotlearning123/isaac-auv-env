"""Physics-based underwater image rendering — Beer-Lambert light transport.

Applies per-pixel depth-dependent attenuation and backscatter using Warp GPU
kernels. Supports multiple Jerlov water type presets.
"""

# Adapted from OceanSim (https://github.com/umfieldrobotics/OceanSim)
# Original: isaacsim/oceansim/utils/UWrenderer_utils.py | License: Apache-2.0
# Paper: Song et al., "OceanSim", IROS 2025
# Physics ref: Jerlov, N.G. (1976). Marine Optics. Elsevier.
# Modifications: Added 10 Jerlov water type presets, caustic kernel, numpy+warp dual API

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------

@wp.func
def _vec3_exp(v: wp.vec3):
    return wp.vec3(wp.exp(v[0]), wp.exp(v[1]), wp.exp(v[2]))


@wp.func
def _vec3_mul(a: wp.vec3, b: wp.vec3):
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _uw_render(
    raw_image: wp.array(ndim=3, dtype=wp.uint8),
    depth_image: wp.array(ndim=2, dtype=wp.float32),
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    out_image: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    raw = wp.vec3(
        wp.float32(raw_image[i, j, 0]),
        wp.float32(raw_image[i, j, 1]),
        wp.float32(raw_image[i, j, 2]),
    )
    d = depth_image[i, j]
    direct = _vec3_mul(raw, _vec3_exp(-d * atten_coeff))
    scatter = _vec3_mul(
        backscatter_value * 255.0,
        wp.vec3(1.0, 1.0, 1.0) - _vec3_exp(-d * backscatter_coeff),
    )
    result = direct + scatter
    out_image[i, j, 0] = wp.uint8(wp.clamp(result[0], 0.0, 255.0))
    out_image[i, j, 1] = wp.uint8(wp.clamp(result[1], 0.0, 255.0))
    out_image[i, j, 2] = wp.uint8(wp.clamp(result[2], 0.0, 255.0))


@wp.kernel
def _caustic_overlay(
    image: wp.array(ndim=3, dtype=wp.uint8),
    time_val: wp.float32,
    intensity: wp.float32,
    out: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    x = wp.float32(j)
    y = wp.float32(i)
    c1 = wp.sin(x * 0.02 + time_val * 1.8) * wp.cos(y * 0.025 - time_val * 1.2)
    c2 = wp.sin((x + y) * 0.015 + time_val * 2.5)
    c = (c1 + c2) * 0.5 * intensity
    out[i, j, 0] = wp.uint8(wp.clamp(wp.float32(image[i, j, 0]) + c * 3.0, 0.0, 255.0))
    out[i, j, 1] = wp.uint8(wp.clamp(wp.float32(image[i, j, 1]) + c * 6.0, 0.0, 255.0))
    out[i, j, 2] = wp.uint8(wp.clamp(wp.float32(image[i, j, 2]) + c * 10.0, 0.0, 255.0))


# ---------------------------------------------------------------------------
# Jerlov water type presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaterParams:
    """Optical parameters for a water type.

    Attributes
    ----------
    backscatter_value : tuple[float, float, float]
        RGB backscatter color (normalised 0-1, multiplied by 255 in kernel).
    atten_coeff : tuple[float, float, float]
        RGB beam attenuation coefficients (m⁻¹).
    backscatter_coeff : tuple[float, float, float]
        RGB backscatter coefficients (m⁻¹).
    """

    backscatter_value: tuple[float, float, float]
    atten_coeff: tuple[float, float, float]
    backscatter_coeff: tuple[float, float, float]


JERLOV_PRESETS: dict[str, WaterParams] = {
    # Open ocean — very clear
    "I": WaterParams(
        backscatter_value=(0.00, 0.10, 0.12),
        atten_coeff=(0.020, 0.015, 0.010),
        backscatter_coeff=(0.020, 0.015, 0.010),
    ),
    "IA": WaterParams(
        backscatter_value=(0.00, 0.15, 0.15),
        atten_coeff=(0.030, 0.022, 0.015),
        backscatter_coeff=(0.030, 0.022, 0.015),
    ),
    "IB": WaterParams(
        backscatter_value=(0.00, 0.20, 0.18),
        atten_coeff=(0.040, 0.030, 0.020),
        backscatter_coeff=(0.040, 0.030, 0.020),
    ),
    # Coastal — moderate turbidity (OceanSim default)
    "II": WaterParams(
        backscatter_value=(0.00, 0.31, 0.24),
        atten_coeff=(0.050, 0.050, 0.050),
        backscatter_coeff=(0.050, 0.050, 0.200),
    ),
    "III": WaterParams(
        backscatter_value=(0.00, 0.35, 0.28),
        atten_coeff=(0.080, 0.060, 0.040),
        backscatter_coeff=(0.080, 0.060, 0.250),
    ),
    # Coastal turbid — high scattering
    "1C": WaterParams(
        backscatter_value=(0.05, 0.38, 0.30),
        atten_coeff=(0.120, 0.090, 0.060),
        backscatter_coeff=(0.100, 0.080, 0.300),
    ),
    "3C": WaterParams(
        backscatter_value=(0.08, 0.40, 0.32),
        atten_coeff=(0.180, 0.130, 0.080),
        backscatter_coeff=(0.150, 0.100, 0.350),
    ),
    "5C": WaterParams(
        backscatter_value=(0.10, 0.42, 0.34),
        atten_coeff=(0.250, 0.180, 0.100),
        backscatter_coeff=(0.200, 0.130, 0.400),
    ),
    "7C": WaterParams(
        backscatter_value=(0.12, 0.44, 0.36),
        atten_coeff=(0.350, 0.250, 0.140),
        backscatter_coeff=(0.250, 0.170, 0.450),
    ),
    "9C": WaterParams(
        backscatter_value=(0.15, 0.46, 0.38),
        atten_coeff=(0.500, 0.350, 0.200),
        backscatter_coeff=(0.300, 0.200, 0.500),
    ),
}


class UnderwaterRenderer:
    """GPU-accelerated underwater image rendering using Warp kernels.

    Parameters
    ----------
    water_type : str
        Jerlov water type key (e.g. ``"II"``, ``"I"``, ``"3C"``).
        Alternatively ``"custom"`` with explicit ``params``.
    device : str
        Warp device string.
    caustic_intensity : float
        Caustic overlay strength (0 = off, 1 = default).
    params : WaterParams | None
        Custom water parameters. Required when ``water_type="custom"``.
    """

    def __init__(
        self,
        water_type: str = "II",
        device: str = "cuda:0",
        caustic_intensity: float = 1.0,
        params: WaterParams | None = None,
    ) -> None:
        if water_type == "custom":
            if params is None:
                raise ValueError("params required when water_type='custom'")
            self._params = params
        else:
            if water_type not in JERLOV_PRESETS:
                raise ValueError(
                    f"Unknown water type {water_type!r}. "
                    f"Available: {sorted(JERLOV_PRESETS.keys())}"
                )
            self._params = JERLOV_PRESETS[water_type]

        self._water_type = water_type
        self._device = device
        self._caustic_intensity = caustic_intensity

        self._backscatter = wp.vec3(*self._params.backscatter_value)
        self._atten = wp.vec3(*self._params.atten_coeff)
        self._back_coeff = wp.vec3(*self._params.backscatter_coeff)

    @property
    def water_type(self) -> str:
        return self._water_type

    @property
    def params(self) -> WaterParams:
        return self._params

    def render_gpu(
        self,
        rgb: wp.array,
        depth: wp.array,
        out: wp.array | None = None,
        time: float = 0.0,
    ) -> wp.array:
        """Apply underwater light transport on GPU arrays.

        Parameters
        ----------
        rgb : wp.array(ndim=3, dtype=wp.uint8)
            Input image (H, W, 3).
        depth : wp.array(ndim=2, dtype=wp.float32)
            Per-pixel depth in metres (H, W).
        out : wp.array, optional
            Pre-allocated output buffer (H, W, 3). Created if ``None``.
        time : float
            Time value for animated caustics.

        Returns
        -------
        wp.array
            Rendered underwater image (H, W, 3) uint8.
        """
        h, w = rgb.shape[0], rgb.shape[1]
        if out is None:
            out = wp.zeros((h, w, 3), dtype=wp.uint8, device=self._device)

        wp.launch(
            _uw_render,
            dim=(h, w),
            inputs=[rgb, depth, self._backscatter, self._atten, self._back_coeff, out],
            device=self._device,
        )

        if self._caustic_intensity > 0.0:
            caustic_out = wp.zeros((h, w, 3), dtype=wp.uint8, device=self._device)
            wp.launch(
                _caustic_overlay,
                dim=(h, w),
                inputs=[out, wp.float32(time), wp.float32(self._caustic_intensity), caustic_out],
                device=self._device,
            )
            return caustic_out

        return out

    def render(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        """Apply underwater light transport to numpy arrays.

        Parameters
        ----------
        rgb : np.ndarray
            Input image (H, W, 3) uint8.
        depth : np.ndarray
            Per-pixel depth in metres (H, W) float32.
            Can also be a scalar broadcast to all pixels.
        time : float
            Time value for animated caustics.

        Returns
        -------
        np.ndarray
            Rendered underwater image (H, W, 3) uint8.
        """
        h, w = rgb.shape[:2]
        rgb_3ch = rgb[:, :, :3].copy()

        if np.ndim(depth) == 0 or depth.shape == ():
            depth = np.full((h, w), float(depth), dtype=np.float32)
        depth = depth.astype(np.float32)

        rgb_gpu = wp.array(rgb_3ch, dtype=wp.uint8, device=self._device)
        depth_gpu = wp.array(depth, dtype=wp.float32, device=self._device)

        result = self.render_gpu(rgb_gpu, depth_gpu, time=time)
        return result.numpy()
