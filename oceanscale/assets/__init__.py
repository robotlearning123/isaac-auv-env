"""Packaged OpenUSD assets for OceanScale demos."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import newton
import warp as wp

BLUEROV2_HEAVY_USD_PRIM = "/World/BlueROV2Heavy"
BLUEROV2_HEAVY_THRUSTER_COUNT = 8


def bluerov2_heavy_usd_path() -> Path:
    return Path(str(files("oceanscale.assets").joinpath("bluerov2_heavy.usda")))


def add_bluerov2_heavy_usd(
    builder: newton.ModelBuilder,
    *,
    xform: wp.transform | None = None,
    load_visual_shapes: bool = True,
) -> tuple[int, dict[str, Any]]:
    result = builder.add_usd(
        str(bluerov2_heavy_usd_path()),
        xform=xform,
        floating=True,
        collapse_fixed_joints=True,
        load_visual_shapes=load_visual_shapes,
        root_path=BLUEROV2_HEAVY_USD_PRIM,
    )
    body_index = int(result["path_body_map"][BLUEROV2_HEAVY_USD_PRIM])
    return body_index, result
