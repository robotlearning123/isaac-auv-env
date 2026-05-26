"""Render the OceanScale MVP demo in Isaac Sim with path-traced underwater visuals.

Usage:
    conda run -n isaac5 env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        python scripts/isaacsim_mvp_render.py [--steps 240] [--output /tmp/oceanscale_isaacsim_demo.mp4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ASSET_USDA = str(Path(__file__).resolve().parent.parent / "oceanscale" / "assets" / "bluerov2_heavy.usda")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Sim underwater MVP renderer")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--output", type=str, default="/tmp/oceanscale_isaacsim_demo.mp4")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def build_underwater_scene(stage, world):
    """Construct an underwater environment in the Isaac Sim stage."""
    from pxr import UsdGeom, UsdLux, UsdShade, Sdf, Gf, UsdPhysics

    # -- Seabed plane (dark sandy material) --
    seabed = UsdGeom.Mesh.Define(stage, "/World/Seabed")
    half = 20.0
    depth = -30.0
    verts = [(-half, -half, depth), (half, -half, depth), (half, half, depth), (-half, half, depth)]
    seabed.CreatePointsAttr(verts)
    seabed.CreateFaceVertexCountsAttr([4])
    seabed.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    seabed.CreateNormalsAttr([(0, 0, 1)] * 4)

    seabed_mat = UsdShade.Material.Define(stage, "/World/Seabed/Material")
    seabed_shader = UsdShade.Shader.Define(stage, "/World/Seabed/Material/Shader")
    seabed_shader.CreateIdAttr("UsdPreviewSurface")
    seabed_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.25, 0.20, 0.15))
    seabed_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    seabed_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    seabed_mat.CreateSurfaceOutput().ConnectToSource(seabed_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(seabed.GetPrim()).Bind(seabed_mat)

    # -- Water surface plane (translucent blue) --
    water = UsdGeom.Mesh.Define(stage, "/World/WaterSurface")
    water_depth = 0.0
    water.CreatePointsAttr([(-half, -half, water_depth), (half, -half, water_depth),
                            (half, half, water_depth), (-half, half, water_depth)])
    water.CreateFaceVertexCountsAttr([4])
    water.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    water.CreateNormalsAttr([(0, 0, 1)] * 4)
    water.CreateDoubleSidedAttr(True)

    water_mat = UsdShade.Material.Define(stage, "/World/WaterSurface/Material")
    water_shader = UsdShade.Shader.Define(stage, "/World/WaterSurface/Material/Shader")
    water_shader.CreateIdAttr("UsdPreviewSurface")
    water_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.15, 0.25))
    water_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
    water_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    water_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    water_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    water_mat.CreateSurfaceOutput().ConnectToSource(water_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(water.GetPrim()).Bind(water_mat)

    # -- Docking station (simple box) --
    dock = UsdGeom.Cube.Define(stage, "/World/DockStation")
    dock.CreateSizeAttr(3.0)
    xform = UsdGeom.Xformable(dock.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, -10.0))
    xform.AddScaleOp().Set(Gf.Vec3d(1.0, 0.5, 1.0))

    dock_mat = UsdShade.Material.Define(stage, "/World/DockStation/Material")
    dock_shader = UsdShade.Shader.Define(stage, "/World/DockStation/Material/Shader")
    dock_shader.CreateIdAttr("UsdPreviewSurface")
    dock_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.4, 0.4, 0.45))
    dock_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    dock_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
    dock_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    dock_mat.CreateSurfaceOutput().ConnectToSource(dock_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(dock.GetPrim()).Bind(dock_mat)

    # -- Underwater lighting --
    dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterDome")
    dome.CreateIntensityAttr(200)
    dome.CreateColorAttr(Gf.Vec3f(0.15, 0.35, 0.5))

    spot = UsdLux.SphereLight.Define(stage, "/World/CausticLight")
    spot.CreateIntensityAttr(5000)
    spot.CreateColorAttr(Gf.Vec3f(0.3, 0.6, 0.8))
    spot.CreateRadiusAttr(0.5)
    UsdGeom.Xformable(spot.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -2))

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(800)
    key.CreateColorAttr(Gf.Vec3f(0.6, 0.75, 0.85))
    key.CreateAngleAttr(2.0)
    xf = UsdGeom.Xformable(key.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3d(-45, 30, 0))

    # -- Particle dust (small spheres to suggest particulate water) --
    rng = np.random.RandomState(42)
    for i in range(30):
        pos = rng.uniform([-8, -8, -20], [8, 8, -3])
        size = rng.uniform(0.02, 0.06)
        dust = UsdGeom.Sphere.Define(stage, f"/World/Particles/dust_{i:03d}")
        dust.CreateRadiusAttr(float(size))
        UsdGeom.Xformable(dust.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))

        dust_mat = UsdShade.Material.Define(stage, f"/World/Particles/dust_{i:03d}/Mat")
        dust_sh = UsdShade.Shader.Define(stage, f"/World/Particles/dust_{i:03d}/Mat/Sh")
        dust_sh.CreateIdAttr("UsdPreviewSurface")
        dust_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.55, 0.45))
        dust_sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.4)
        dust_sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        dust_mat.CreateSurfaceOutput().ConnectToSource(dust_sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(dust.GetPrim()).Bind(dust_mat)


def load_bluerov2(stage, position=(0, 0, -10)):
    """Load the BlueROV2 USD asset into the stage."""
    from pxr import UsdGeom, Gf

    result = stage.GetRootLayer().subLayerPaths.append(ASSET_USDA)

    rov_prim = stage.GetPrimAtPath("/World/BlueROV2Heavy")
    if not rov_prim.IsValid():
        rov_prim = stage.OverridePrim("/World/BlueROV2Heavy")
        ref = rov_prim.GetReferences()
        ref.AddReference(ASSET_USDA, "/World/BlueROV2Heavy")

    xf = UsdGeom.Xformable(rov_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))

    return rov_prim


def main():
    args = parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": True,
        "width": args.width,
        "height": args.height,
        "anti_aliasing": 0,
    })

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from pxr import Gf

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    print("[1/6] Building underwater scene...")
    build_underwater_scene(stage, world)

    print("[2/6] Loading BlueROV2 Heavy USD...")
    rov_prim = load_bluerov2(stage, position=(0, 0, -10))

    print("[3/6] Setting up cameras...")
    cam_chase = Camera(
        prim_path="/World/ChaseCam",
        position=np.array([-3.0, -2.0, -8.0]),
        frequency=args.fps,
        resolution=(args.width, args.height),
        orientation=rot_utils.euler_angles_to_quats(np.array([75, 0, 25]), degrees=True),
    )

    world.reset()
    cam_chase.initialize()

    print("[4/6] Warming up renderer...")
    for _ in range(30):
        world.step(render=True)

    # Run the OceanScale physics in a separate process data flow:
    # We read trajectory from the MVP and animate the USD prim along it.
    print("[5/6] Running OceanScale physics + capturing frames...")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from oceanscale.mvp import UnderwaterRobotMVPConfig, run_underwater_robot_mvp

    result = run_underwater_robot_mvp(UnderwaterRobotMVPConfig(
        n_steps=args.steps,
        device="cuda:0",
        render_mp4=None,
    ))

    trajectory = result["trajectory"]["samples"]
    print(f"  Trajectory: {len(trajectory)} samples, mission {'COMPLETE' if result['mission']['completed'] else 'INCOMPLETE'}")

    frames = []
    from pxr import UsdGeom

    for i, sample in enumerate(trajectory):
        pos = sample["position"]
        xf = UsdGeom.Xformable(rov_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

        # Orbit camera around ROV
        angle = i * (360.0 / len(trajectory)) * 0.3
        rad = np.radians(angle)
        cam_dist = 4.0
        cam_offset = np.array([
            pos[0] + cam_dist * np.cos(rad),
            pos[1] + cam_dist * np.sin(rad),
            pos[2] + 1.5,
        ])
        cam_chase.set_world_pose(
            position=cam_offset,
            orientation=rot_utils.euler_angles_to_quats(
                np.array([80, 0, float(np.degrees(rad) + 180)]), degrees=True
            ),
        )

        for _ in range(2):
            world.step(render=True)

        frame = cam_chase.get_rgba()
        if frame is not None and frame.size > 0:
            frames.append(frame[:, :, :3].copy())

        if (i + 1) % 8 == 0:
            d = sample["distance_to_target_m"]
            print(f"  Frame {i+1}/{len(trajectory)}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) dist={d:.2f}m")

    print(f"[6/6] Encoding {len(frames)} frames to {args.output}...")

    if frames:
        import imageio.v3 as iio
        writer = iio.imopen(args.output, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=args.fps)
        for f in frames:
            h, w = f.shape[:2]
            if h % 2: f = f[:h-1]
            if w % 2: f = f[:w-1]
            writer.write_frame(f)
        writer.close()
        print(f"  Video saved: {args.output} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
    else:
        print("  WARNING: No frames captured — falling back to numpy frame export")
        np.save(args.output.replace(".mp4", "_frames.npy"), np.array(frames[:1] if frames else []))

    app.close()

    print("\n=== OceanScale Isaac Sim Demo ===")
    print(f"Vehicle: {result['vehicle']['name']} ({result['vehicle']['thrusters']} thrusters)")
    print(f"Mission: {result['mission']['type']} — {'COMPLETE' if result['mission']['completed'] else 'INCOMPLETE'}")
    print(f"Final distance: {result['metrics']['final_distance_to_target_m']:.3f} m")
    print(f"Sonar detection: {result['metrics']['sonar_detection_rate']:.0%}")
    print(f"Render: {len(frames)} frames @ {args.fps}fps")


if __name__ == "__main__":
    main()
