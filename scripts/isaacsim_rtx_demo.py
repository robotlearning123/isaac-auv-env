"""Isaac Sim RTX underwater demo — MarineGym BlueROV mesh + OceanSim water FX.

Usage:
    conda run -n isaac5 env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        python scripts/isaacsim_rtx_demo.py
"""
import os
import sys
import glob

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1920, "height": 1080, "anti_aliasing": 0})

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

stage = omni.usd.get_context().get_stage()

BLUEROV_USD = "/home/robot/workspace/marinegym/marinegym/robots/assets/usd/BlueROV/BlueROV.usd"
OUT_DIR = "/tmp/isaacsim_rtx_frames"
N_FRAMES = 120

os.makedirs(OUT_DIR, exist_ok=True)
for f in glob.glob(f"{OUT_DIR}/*.png"):
    os.remove(f)


def _mat(path, color, metallic=0.0, roughness=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Sh")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def build_underwater_scene():
    """Build seabed, dock, lighting, and particles."""
    # Seabed
    seabed = UsdGeom.Mesh.Define(stage, "/World/Seabed")
    h = 30.0
    seabed.CreatePointsAttr([(-h, -h, -30), (h, -h, -30), (h, h, -30), (-h, h, -30)])
    seabed.CreateFaceVertexCountsAttr([4])
    seabed.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    seabed.CreateNormalsAttr([(0, 0, 1)] * 4)
    _bind(seabed.GetPrim(), _mat("/World/Seabed/Mat", (0.28, 0.22, 0.15), roughness=0.9))

    # Dock station
    dock = UsdGeom.Cube.Define(stage, "/World/Dock")
    dock.CreateSizeAttr(3.0)
    xf = UsdGeom.Xformable(dock.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(4, 0, -10))
    xf.AddScaleOp().Set(Gf.Vec3d(1, 0.5, 1))
    _bind(dock.GetPrim(), _mat("/World/Dock/Mat", (0.45, 0.45, 0.5), metallic=0.7, roughness=0.5))

    # Dock legs
    for i, (dx, dy) in enumerate([(-1.2, -0.6), (1.2, -0.6), (-1.2, 0.6), (1.2, 0.6)]):
        leg = UsdGeom.Cylinder.Define(stage, f"/World/DockLeg{i}")
        leg.CreateRadiusAttr(0.08)
        leg.CreateHeightAttr(4.0)
        UsdGeom.Xformable(leg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(4 + dx, dy, -12))
        _bind(leg.GetPrim(), _mat(f"/World/DockLeg{i}/Mat", (0.4, 0.4, 0.42), metallic=0.6))

    # Underwater dome light
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(500)
    dome.CreateColorAttr(Gf.Vec3f(0.12, 0.30, 0.45))

    # Key light (sun through water)
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(2000)
    key.CreateColorAttr(Gf.Vec3f(0.5, 0.7, 0.85))
    key.CreateAngleAttr(3.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3d(-60, 20, 0))

    # ROV spotlight (follows ROV — updated per frame)
    rov_light = UsdLux.SphereLight.Define(stage, "/World/RovLight")
    rov_light.CreateIntensityAttr(50000)
    rov_light.CreateColorAttr(Gf.Vec3f(0.7, 0.85, 1.0))
    rov_light.CreateRadiusAttr(0.08)
    UsdGeom.Xformable(rov_light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -9.0))

    # Fill light behind camera
    fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
    fill.CreateIntensityAttr(8000)
    fill.CreateColorAttr(Gf.Vec3f(0.4, 0.6, 0.8))
    fill.CreateRadiusAttr(1.5)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-4, -5, -7))

    # Particles
    rng = np.random.RandomState(42)
    stage.DefinePrim("/World/Particles", "Scope")
    for i in range(30):
        pos = rng.uniform([-10, -10, -24], [10, 10, -4])
        s = UsdGeom.Sphere.Define(stage, f"/World/Particles/p{i}")
        s.CreateRadiusAttr(float(rng.uniform(0.03, 0.08)))
        UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        _bind(s.GetPrim(), _mat(f"/World/Particles/p{i}/M", (0.6, 0.55, 0.45), roughness=0.8))


def load_bluerov():
    """Load MarineGym BlueROV USD mesh into the scene."""
    # Create a parent xform to position the ROV
    rov_parent = stage.DefinePrim("/World/ROV", "Xform")
    xf = UsdGeom.Xformable(rov_parent)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -10))

    # Add reference to the BlueROV USD
    rov_parent.GetReferences().AddReference(BLUEROV_USD, "/BlueROV")

    return rov_parent


def apply_underwater_postfx_numpy(frame, depth_hint, time, distance, step, total_steps):
    """Numpy-based underwater post-processing with HUD."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = frame.shape[:2]
    img = frame.astype(np.float32)

    # Subtle blue tint — preserve RTX lighting
    img *= np.array([0.75, 0.90, 1.0], dtype=np.float32)

    # Very light fog — let RTX do the work
    fog = np.array([10, 25, 50], dtype=np.float32)
    strength = 0.08
    img = img * (1.0 - strength) + fog * strength

    # Subtle caustic pattern
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c1 = np.sin(xx * 0.015 + time * 1.5) * np.cos(yy * 0.02 - time * 1.0)
    caustic = c1 * 0.5
    img += np.stack([caustic * 2, caustic * 4, caustic * 6], axis=-1)

    # Minimal grain
    rng = np.random.RandomState(step)
    img += rng.normal(0, 0.8, img.shape).astype(np.float32)

    # Light vignette
    cy, cx = h / 2.0, w / 2.0
    vig = 1.0 - 0.18 * np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    img *= np.clip(vig, 0.6, 1.0)[:, :, np.newaxis]

    img = np.clip(img, 0, 255).astype(np.uint8)

    # HUD
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_lg = ImageFont.load_default()
        font_sm = font_lg
        font_xs = font_lg

    hud = (120, 200, 255)
    accent = (255, 160, 60)

    draw.text((20, 14), "OceanScale", fill=hud, font=font_lg)
    draw.text((20, 40), "GPU-native ocean simulation", fill=(80, 150, 200), font=font_xs)

    mt = "BlueROV2 Heavy | Dock Approach"
    tw = draw.textbbox((0, 0), mt, font=font_sm)[2]
    draw.text((w - tw - 20, 16), mt, fill=hud, font=font_sm)

    dt = f"Distance: {distance:.2f} m"
    draw.text((20, h - 42), dt, fill=accent if distance < 0.5 else hud, font=font_sm)

    st = f"Step {step}/{total_steps}"
    tw = draw.textbbox((0, 0), st, font=font_sm)[2]
    draw.text((w - tw - 20, h - 42), st, fill=hud, font=font_sm)

    dpt = f"Depth: {10.0 + depth_hint * 0.1:.1f} m"
    tw = draw.textbbox((0, 0), dpt, font=font_xs)[2]
    draw.text(((w - tw) // 2, h - 38), dpt, fill=(80, 170, 200), font=font_xs)

    # Crosshair
    cx_i, cy_i = w // 2, h // 2
    draw.line([(cx_i - 10, cy_i), (cx_i + 10, cy_i)], fill=(100, 200, 180), width=1)
    draw.line([(cx_i, cy_i - 10), (cx_i, cy_i + 10)], fill=(100, 200, 180), width=1)

    # Title card fade-in
    if step <= 30:
        alpha = min(step / 30.0, 1.0)
        try:
            tf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
            sf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except OSError:
            tf = font_lg
            sf = font_sm
        title = "OceanScale"
        sub = "GPU-Native Ocean Simulation"
        t_w = draw.textbbox((0, 0), title, font=tf)[2]
        s_w = draw.textbbox((0, 0), sub, font=sf)[2]
        a_int = int(alpha * 220)
        draw.text(((w - t_w) // 2, h // 2 - 40), title, fill=(180, 230, 255, a_int), font=tf)
        draw.text(((w - s_w) // 2, h // 2 + 15), sub, fill=(120, 190, 230, a_int), font=sf)

    return np.array(pil_img)


def set_camera_lookat(eye, look):
    """Set camera transform using look-at."""
    fwd = (look - eye).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(fwd, up).GetNormalized()
    new_up = Gf.Cross(right, fwd)
    mat = Gf.Matrix4d(1)
    mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
    mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
    mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))

    cam_prim = stage.GetPrimAtPath("/World/Cam")
    cam_xf = UsdGeom.Xformable(cam_prim)
    cam_ops = cam_xf.GetOrderedXformOps()
    if cam_ops:
        cam_ops[0].Set(mat)
    else:
        cam_xf.AddTransformOp().Set(mat)


def main():
    from isaacsim.core.api import World
    import omni.renderer_capture

    sys.stderr.write("[1/6] Building underwater scene...\n"); sys.stderr.flush()
    world = World(stage_units_in_meters=1.0)
    build_underwater_scene()

    sys.stderr.write("[2/6] Loading MarineGym BlueROV mesh...\n"); sys.stderr.flush()
    rov = load_bluerov()

    # Verify it loaded
    rov_check = stage.GetPrimAtPath("/World/ROV")
    sys.stderr.write(f"  ROV valid: {rov_check.IsValid()}\n")
    children = [c.GetName() for c in rov_check.GetChildren()]
    sys.stderr.write(f"  Children: {children[:8]}\n")
    sys.stderr.flush()

    sys.stderr.write("[3/6] Setting up camera...\n"); sys.stderr.flush()
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))

    import omni.kit.viewport.utility as vp_util
    vp = vp_util.get_active_viewport()
    if vp:
        vp.set_active_camera("/World/Cam")

    # Initial camera position
    set_camera_lookat(Gf.Vec3d(-3, -3, -8), Gf.Vec3d(0, 0, -10))

    sys.stderr.write("[4/6] Resetting world + warming up...\n"); sys.stderr.flush()
    world.reset()
    for _ in range(60):
        app.update()

    sys.stderr.write(f"[5/6] Rendering {N_FRAMES} frames...\n"); sys.stderr.flush()
    cap = omni.renderer_capture.acquire_renderer_capture_interface()

    start = np.array([0.0, 0.0, -10.0])
    target = np.array([1.5, 0.0, -10.0])
    raw_frames_dir = f"{OUT_DIR}/raw"
    os.makedirs(raw_frames_dir, exist_ok=True)

    for frame in range(N_FRAMES):
        t = frame / max(N_FRAMES - 1, 1)

        # Smooth approach (ease-out)
        rov_pos = start + (target - start) * (1 - (1 - min(t * 1.5, 1.0)) ** 3)

        # Move ROV
        rov_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ROV"))
        ops = rov_xf.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(*rov_pos.tolist()))

        # Move ROV light to follow
        light_prim = stage.GetPrimAtPath("/World/RovLight")
        if light_prim.IsValid():
            l_xf = UsdGeom.Xformable(light_prim)
            l_ops = l_xf.GetOrderedXformOps()
            if l_ops:
                l_ops[0].Set(Gf.Vec3d(rov_pos[0] + 0.3, rov_pos[1], rov_pos[2] + 0.5))

        # Move fill light to follow camera
        fill_prim = stage.GetPrimAtPath("/World/Fill")
        if fill_prim.IsValid():
            f_xf = UsdGeom.Xformable(fill_prim)
            f_ops = f_xf.GetOrderedXformOps()
            if f_ops:
                f_ops[0].Set(Gf.Vec3d(rov_pos[0] - 4, rov_pos[1] - 5, rov_pos[2] + 3))

        # Tight chase camera — ROV always centered
        angle = -0.6 + t * 0.4
        cam_r = 1.8
        cam_x = rov_pos[0] - cam_r * np.cos(angle)
        cam_y = rov_pos[1] - cam_r * np.sin(angle)
        cam_z = rov_pos[2] + 0.6
        set_camera_lookat(
            Gf.Vec3d(cam_x, cam_y, cam_z),
            Gf.Vec3d(*rov_pos.tolist()),
        )

        # Render
        for _ in range(3):
            app.update()

        raw_path = f"{raw_frames_dir}/frame_{frame:04d}.png"
        cap.capture_next_frame_swapchain(raw_path)
        for _ in range(3):
            app.update()

        if (frame + 1) % 20 == 0:
            dist = float(np.linalg.norm(rov_pos - target))
            sys.stderr.write(f"  Frame {frame+1}/{N_FRAMES}: dist={dist:.2f}m\n")
            sys.stderr.flush()

    sys.stderr.write("[6/6] Applying underwater post-processing...\n"); sys.stderr.flush()

    from PIL import Image as PILImage

    for frame in range(N_FRAMES):
        raw_path = f"{raw_frames_dir}/frame_{frame:04d}.png"
        if not os.path.exists(raw_path):
            continue

        img = np.array(PILImage.open(raw_path))[:, :, :3]
        t = frame / max(N_FRAMES - 1, 1)
        rov_pos = start + (target - start) * (1 - (1 - min(t * 1.5, 1.0)) ** 3)
        dist = float(np.linalg.norm(rov_pos - target))
        depth = float(-rov_pos[2])

        fx = apply_underwater_postfx_numpy(
            img, depth_hint=depth, time=frame * 0.033,
            distance=dist, step=frame + 1, total_steps=N_FRAMES,
        )

        PILImage.fromarray(fx).save(f"{OUT_DIR}/frame_{frame:04d}.png")

        if (frame + 1) % 40 == 0:
            sys.stderr.write(f"  PostFX {frame+1}/{N_FRAMES}\n")
            sys.stderr.flush()

    sys.stderr.write("Done! Frames saved.\n"); sys.stderr.flush()
    os._exit(0)


main()
