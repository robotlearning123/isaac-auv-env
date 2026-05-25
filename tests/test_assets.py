"""Tests for packaged OpenUSD demo assets."""

from __future__ import annotations

import numpy as np
import pytest

from oceanscale.assets import (
    BLUEROV2_HEAVY_THRUSTER_COUNT,
    BLUEROV2_HEAVY_USD_PRIM,
    bluerov2_heavy_usd_path,
)
from oceanscale.demo import UnifiedDemo
from oceanscale.vehicles import BlueROV2Heavy


def test_bluerov2_heavy_usd_asset_matches_vehicle_parameters() -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    vehicle = BlueROV2Heavy()
    asset_path = bluerov2_heavy_usd_path()
    assert asset_path.exists()

    stage = Usd.Stage.Open(str(asset_path))
    prim = stage.GetPrimAtPath(BLUEROV2_HEAVY_USD_PRIM)
    assert prim.IsValid()

    mass_api = UsdPhysics.MassAPI(prim)
    assert mass_api.GetMassAttr().Get() == pytest.approx(vehicle.mass)
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (vehicle.Ix, vehicle.Iy, vehicle.Iz)
    )
    assert prim.GetAttribute("oceanscale:assetRole").Get() == "mvp_visual_proxy"
    assert prim.GetAttribute("oceanscale:vehicle").Get() == "BlueROV2 Heavy"
    assert prim.GetAttribute("oceanscale:geometryReference").Get() == (
        "scratch OpenUSD from public dimensions and von Benzon Table A1"
    )
    assert prim.GetAttribute("oceanscale:dynamicsReference").Get() == (
        "von Benzon et al. 2022 Table A1"
    )
    assert prim.GetAttribute("oceanscale:thrusterCount").Get() == BLUEROV2_HEAVY_THRUSTER_COUNT
    assert prim.GetAttribute("oceanscale:cadDerived").Get() is False
    assert "clydemcqueen/bluerov2_gz" in prim.GetAttribute(
        "oceanscale:upgradeCandidate"
    ).Get()

    thrusters = [
        child for child in prim.GetChildren() if child.GetName() in {f"Thruster{i}" for i in range(1, 9)}
    ]
    assert len(thrusters) == BLUEROV2_HEAVY_THRUSTER_COUNT
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/HullCollision").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/ElectronicsTube").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/AcrylicSidePlateL").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/AcrylicSidePlateR").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/CameraDome").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/TetherLead").IsValid()
    assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/Materials/RobotHull").IsValid()
    for detail in (
        "SideRailL",
        "SideRailR",
        "SkidRailL",
        "SkidRailR",
        "FrameStandoffFL",
        "FrameStandoffAR",
        "ForwardSonar",
        "DvlPod",
        "DvlTransducerFL",
        "ForwardLedL",
        "ForwardLedR",
        "ManipulatorBase",
        "ManipulatorArm",
        "ManipulatorClawL",
        "ManipulatorClawR",
        "Thruster1RotorBlade",
        "Thruster4RotorBlade",
    ):
        assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/{detail}").IsValid()
    for material in ("StainlessSteel", "PropellerWhite", "SensorBlack", "ForwardLed"):
        assert stage.GetPrimAtPath(f"{BLUEROV2_HEAVY_USD_PRIM}/Materials/{material}").IsValid()


def test_unified_demo_loads_bluerov2_usd_asset() -> None:
    demo = UnifiedDemo(
        vehicle=BlueROV2Heavy(),
        asset_source="usd",
        enable_structures=False,
    )

    assert demo.asset_source == "usd"
    assert demo.asset_path is not None
    assert demo.asset_shape_count >= BLUEROV2_HEAVY_THRUSTER_COUNT + 20
    assert demo.model.body_count >= 1

    obs = demo.reset()
    stepped = demo.step(np.zeros(6, dtype=np.float32))
    assert np.all(np.isfinite(obs["rov_position"]))
    assert np.all(np.isfinite(stepped["rov_position"]))
    assert stepped["dvl"]["valid"]
