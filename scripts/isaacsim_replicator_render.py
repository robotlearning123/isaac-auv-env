"""Isaac Sim headless rendering using Replicator API.

Usage:
    conda run -n isaac5 env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        python scripts/isaacsim_replicator_render.py
"""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "width": 1920,
    "height": 1080,
    "anti_aliasing": 0,
})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

ASSET_USDA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "oceanscale", "assets", "bluerov2_heavy.usda",
)

stage = omni.usd.get_context().get_stage()


def build_underwater_scene():
    # -- Seabed plane at z=-30 --
    seabed = UsdGeom.Mesh.Define(stage, "/World/Seabed")
    half = 20.0
    depth = -30.0
    verts = [
        (-half, -half, depth), (half, -half, depth),
        (half, half, depth), (-half, half, depth),
    ]
    seabed.CreatePointsAttr(verts)
    seabed.CreateFaceVertexCountsAttr([4])
    seabed.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    seabed.CreateNormalsAttr([(0, 0, 1)] * 4)

    seabed_mat = UsdShade.Material.Define(stage, "/World/Seabed/Material")
    seabed_sh = UsdShade.Shader.Define(stage, "/World/Seabed/Material/Shader")
    seabed_sh.CreateIdAttr("UsdPreviewSurface")
    seabed_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.25, 0.20, 0.15))
    seabed_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    seabed_sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    seabed_mat.CreateSurfaceOutput().ConnectToSource(seabed_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(seabed.GetPrim()).Bind(seabed_mat)

    # -- Water surface at z=0 --
    water = UsdGeom.Mesh.Define(stage, "/World/WaterSurface")
    water.CreatePointsAttr([
        (-half, -half, 0), (half, -half, 0),
        (half, half, 0), (-half, half, 0),
    ])
    water.CreateFaceVertexCountsAttr([4])
    water.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    water.CreateNormalsAttr([(0, 0, 1)] * 4)
    water.CreateDoubleSidedAttr(True)

    water_mat = UsdShade.Material.Define(stage, "/World/WaterSurface/Material")
    water_sh = UsdShade.Shader.Define(stage, "/World/WaterSurface/Material/Shader")
    water_sh.CreateIdAttr("UsdPreviewSurface")
    water_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.15, 0.25))
    water_sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.08)
    water_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    water_sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    water_mat.CreateSurfaceOutput().ConnectToSource(water_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(water.GetPrim()).Bind(water_mat)

    # -- Dock station at (4, 0, -10) --
    dock = UsdGeom.Cube.Define(stage, "/World/DockStation")
    dock.CreateSizeAttr(3.0)
    xf = UsdGeom.Xformable(dock.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, -10.0))
    xf.AddScaleOp().Set(Gf.Vec3d(1.0, 0.5, 1.0))

    dock_mat = UsdShade.Material.Define(stage, "/World/DockStation/Material")
    dock_sh = UsdShade.Shader.Define(stage, "/World/DockStation/Material/Shader")
    dock_sh.CreateIdAttr("UsdPreviewSurface")
    dock_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.4, 0.4, 0.45))
    dock_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    dock_sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
    dock_sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    dock_mat.CreateSurfaceOutput().ConnectToSource(dock_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(dock.GetPrim()).Bind(dock_mat)

    # -- Underwater dome light (ambient fill) --
    dome = UsdLux.DomeLight.Define(stage, "/World/UnderwaterDome")
    dome.CreateIntensityAttr(500)
    dome.CreateColorAttr(Gf.Vec3f(0.12, 0.30, 0.45))

    # -- Key light from above (sun through water) --
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2000)
    key.CreateColorAttr(Gf.Vec3f(0.5, 0.7, 0.85))
    key.CreateAngleAttr(3.0)
    xf_key = UsdGeom.Xformable(key.GetPrim())
    xf_key.AddRotateXYZOp().Set(Gf.Vec3d(-60, 20, 0))

    # -- ROV spotlight (simulates onboard lights) --
    spot = UsdLux.SphereLight.Define(stage, "/World/ROVLight")
    spot.CreateIntensityAttr(30000)
    spot.CreateColorAttr(Gf.Vec3f(0.7, 0.85, 1.0))
    spot.CreateRadiusAttr(0.1)
    UsdGeom.Xformable(spot.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -9.5))

    # -- Fill light behind camera --
    fill = UsdLux.SphereLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(4000)
    fill.CreateColorAttr(Gf.Vec3f(0.3, 0.5, 0.7))
    fill.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-3.0, -3.0, -6.0))

    # -- Suspended particles --
    rng = np.random.RandomState(42)
    particles_scope = stage.DefinePrim("/World/Particles", "Scope")
    for i in range(20):
        pos = rng.uniform([-8, -8, -20], [8, 8, -3])
        size = float(rng.uniform(0.03, 0.08))
        dust = UsdGeom.Sphere.Define(stage, f"/World/Particles/dust_{i:03d}")
        dust.CreateRadiusAttr(size)
        UsdGeom.Xformable(dust.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        dust_mat = UsdShade.Material.Define(stage, f"/World/Particles/dust_{i:03d}/Mat")
        dust_sh = UsdShade.Shader.Define(stage, f"/World/Particles/dust_{i:03d}/Mat/Sh")
        dust_sh.CreateIdAttr("UsdPreviewSurface")
        dust_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.55, 0.45))
        dust_sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.4)
        dust_sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        dust_mat.CreateSurfaceOutput().ConnectToSource(dust_sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(dust.GetPrim()).Bind(dust_mat)


def _make_bright_material(path: str, color: tuple, metallic: float = 0.0, roughness: float = 0.5):
    """Create a bright UsdPreviewSurface material visible underwater."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _override_rov_materials(rov_prim):
    """Override all ROV child materials with bright colors for underwater visibility."""
    # Color assignments by part name pattern
    color_map = {
        "Hull": (0.9, 0.55, 0.1),        # orange hull
        "Float": (0.95, 0.8, 0.15),       # yellow floats
        "Acrylic": (0.3, 0.85, 0.9),      # cyan plates
        "Camera": (0.15, 0.15, 0.18),     # dark dome
        "Thruster": (0.2, 0.2, 0.25),     # dark gray
        "RotorBlade": (0.85, 0.85, 0.85), # white blades
        "Rail": (0.6, 0.6, 0.65),         # light gray frame
        "Skid": (0.6, 0.6, 0.65),
        "Standoff": (0.7, 0.7, 0.72),     # silver standoffs
        "Led": (1.0, 1.0, 0.85),          # bright LEDs
        "Sonar": (0.3, 0.3, 0.35),
        "Dvl": (0.3, 0.3, 0.35),
        "Manipulator": (0.75, 0.4, 0.1),  # orange arm
        "Tether": (0.8, 0.7, 0.1),        # yellow tether
        "Electronics": (0.85, 0.55, 0.1),  # orange tube
    }
    default_color = (0.9, 0.55, 0.1)  # fallback orange

    for child in rov_prim.GetAllChildren():
        if not child.IsA(UsdGeom.Gprim):
            continue
        name = child.GetName()
        color = default_color
        for pattern, c in color_map.items():
            if pattern in name:
                color = c
                break

        mat_path = child.GetPath().pathString + "/OverrideMat"
        mat = _make_bright_material(mat_path, color, metallic=0.3, roughness=0.5)
        if not child.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI.Apply(child)
        UsdShade.MaterialBindingAPI(child).Bind(mat)


def _add_rov_part(parent_path, name, prim_type, translate, color, **kwargs):
    """Add a single ROV geometry part with material."""
    path = f"{parent_path}/{name}"
    mat_path = f"{path}/Mat"
    t = Gf.Vec3d(*translate)

    if prim_type == "Cube":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(kwargs.get("size", 1.0))
        xf = UsdGeom.Xformable(prim.GetPrim())
        xf.AddTranslateOp().Set(t)
        if "scale" in kwargs:
            xf.AddScaleOp().Set(Gf.Vec3d(*kwargs["scale"]))
    elif prim_type == "Cylinder":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr(kwargs.get("radius", 0.035))
        prim.CreateHeightAttr(kwargs.get("height", 0.09))
        xf = UsdGeom.Xformable(prim.GetPrim())
        xf.AddTranslateOp().Set(t)
    elif prim_type == "Sphere":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(kwargs.get("radius", 0.05))
        xf = UsdGeom.Xformable(prim.GetPrim())
        xf.AddTranslateOp().Set(t)

    mat = _make_bright_material(mat_path, color, metallic=kwargs.get("metallic", 0.3), roughness=kwargs.get("roughness", 0.5))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)
    return prim


def load_bluerov2():
    """Build BlueROV2 Heavy directly in the scene with native prims."""
    rov = UsdGeom.Xform.Define(stage, "/World/ROV")
    rov.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
    p = "/World/ROV"

    # Hull (main body)
    _add_rov_part(p, "Hull", "Cube", (0, 0, 0), (0.9, 0.55, 0.1), scale=(0.46, 0.58, 0.38))
    # Electronics tube
    _add_rov_part(p, "ETube", "Cylinder", (0.02, 0, 0.02), (0.85, 0.5, 0.1), radius=0.165, height=0.72)
    # Top float
    _add_rov_part(p, "Float", "Cube", (0, 0, 0.28), (0.95, 0.8, 0.15), scale=(0.58, 0.24, 0.10))
    # Acrylic side plates
    _add_rov_part(p, "PlateL", "Cube", (0.02, 0.31, -0.01), (0.3, 0.85, 0.9), scale=(0.44, 0.012, 0.21))
    _add_rov_part(p, "PlateR", "Cube", (0.02, -0.31, -0.01), (0.3, 0.85, 0.9), scale=(0.44, 0.012, 0.21))
    # Camera dome
    _add_rov_part(p, "CamDome", "Sphere", (0.57, 0, -0.02), (0.15, 0.15, 0.18), radius=0.052)
    # Forward LEDs (bright)
    _add_rov_part(p, "LedL", "Sphere", (0.52, 0.13, -0.07), (1.0, 1.0, 0.85), radius=0.045, metallic=0.0, roughness=0.2)
    _add_rov_part(p, "LedR", "Sphere", (0.52, -0.13, -0.07), (1.0, 1.0, 0.85), radius=0.045, metallic=0.0, roughness=0.2)
    # Vertical thrusters (top)
    for i, (tx, ty) in enumerate([(0.156, 0.111), (0.156, -0.11), (-0.157, 0.11), (-0.156, -0.111)]):
        _add_rov_part(p, f"VThr{i}", "Cylinder", (tx, ty, 0.085), (0.2, 0.2, 0.25), radius=0.035, height=0.09)
    # Horizontal thrusters
    for i, (tx, ty) in enumerate([(0.12, 0.218), (0.12, -0.218), (-0.121, 0.218), (-0.12, -0.218)]):
        _add_rov_part(p, f"HThr{i}", "Cylinder", (tx, ty, 0), (0.2, 0.2, 0.25), radius=0.035, height=0.09)
    # Side rails
    _add_rov_part(p, "RailL", "Cylinder", (0, 0.34, -0.03), (0.6, 0.6, 0.65), radius=0.025, height=1.08)
    _add_rov_part(p, "RailR", "Cylinder", (0, -0.34, -0.03), (0.6, 0.6, 0.65), radius=0.025, height=1.08)
    # Skid rails
    _add_rov_part(p, "SkidL", "Cylinder", (0, 0.28, -0.34), (0.6, 0.6, 0.65), radius=0.022, height=0.96)
    _add_rov_part(p, "SkidR", "Cylinder", (0, -0.28, -0.34), (0.6, 0.6, 0.65), radius=0.022, height=0.96)
    # Frame standoffs
    for i, (fx, fy) in enumerate([(0.42, 0.28), (0.42, -0.28), (-0.42, 0.28), (-0.42, -0.28)]):
        _add_rov_part(p, f"Standoff{i}", "Cylinder", (fx, fy, -0.08), (0.7, 0.7, 0.72), radius=0.014, height=0.5)

    return rov.GetPrim()


def main():
    from isaacsim.core.api import World

    sys.stderr.write("[1/6] Creating World...\n")
    sys.stderr.flush()
    world = World(stage_units_in_meters=1.0)

    sys.stderr.write("[2/6] Building underwater scene...\n")
    sys.stderr.flush()
    build_underwater_scene()

    sys.stderr.write("[3/6] Loading BlueROV2 Heavy USD...\n")
    sys.stderr.flush()
    load_bluerov2()

    # Debug: check if ROV loaded
    rov_check = stage.GetPrimAtPath("/World/ROV")
    sys.stderr.write(f"  ROV prim valid: {rov_check.IsValid()}\n")
    if rov_check.IsValid():
        bb = UsdGeom.Imageable(rov_check).ComputeWorldBound(0, "default")
        sys.stderr.write(f"  ROV bbox: {bb.GetRange()}\n")
        children = [c.GetPath().pathString for c in rov_check.GetChildren()]
        sys.stderr.write(f"  ROV children: {children[:5]}\n")
    sys.stderr.flush()

    sys.stderr.write("[4/6] Resetting world + warming up renderer...\n")
    sys.stderr.flush()
    world.reset()
    for _ in range(10):
        world.step(render=True)

    sys.stderr.write("[5/6] Moving camera to underwater view...\n")
    sys.stderr.flush()

    # Use omni.kit.viewport to set camera transform reliably
    import omni.kit.viewport.utility as vp_util
    vp = vp_util.get_active_viewport()

    # Create a fresh camera prim to avoid xform conflicts with default persp
    cam_prim = UsdGeom.Camera.Define(stage, "/World/UnderwaterCam")
    cam_prim.CreateFocalLengthAttr(18.0)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))

    eye = Gf.Vec3d(-2.5, -4.0, -8.0)
    target = Gf.Vec3d(1.0, 0.0, -10.0)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    fwd = (target - eye).GetNormalized()
    right = Gf.Cross(fwd, up).GetNormalized()
    new_up = Gf.Cross(right, fwd)
    mat = Gf.Matrix4d(1)
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
    mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    cam_prim.AddTransformOp().Set(mat)

    # Set this camera as the active viewport camera
    import omni.kit.viewport.utility as vp_util
    vp = vp_util.get_active_viewport()
    if vp is not None:
        vp.set_active_camera("/World/UnderwaterCam")
        sys.stderr.write("  Camera set to /World/UnderwaterCam\n")
    else:
        sys.stderr.write("  No active viewport\n")

    for _ in range(80):
        app.update()

    sys.stderr.write("[6/6] Capturing frame via swapchain...\n")
    sys.stderr.flush()

    import omni.renderer_capture
    cap = omni.renderer_capture.acquire_renderer_capture_interface()

    cap.capture_next_frame_swapchain("/tmp/isaacsim_replicator_test.png")
    for _ in range(10):
        app.update()

    if not os.path.exists("/tmp/isaacsim_replicator_test.png"):
        cap.capture_next_frame_swapchain_to_file("/tmp/isaacsim_replicator_test.png")
        for _ in range(10):
            app.update()

    if os.path.exists("/tmp/isaacsim_replicator_test.png"):
        from PIL import Image as PILImage
        img = PILImage.open("/tmp/isaacsim_replicator_test.png")
        sys.stderr.write(f"SAVED: /tmp/isaacsim_replicator_test.png (size={img.size})\n")
    else:
        sys.stderr.write("ERROR: File not saved\n")
    sys.stderr.flush()

    os._exit(0)


main()
