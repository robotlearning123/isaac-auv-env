"""Isaac Sim RTX multi-frame video renderer — swapchain capture loop."""
import os
import sys
import glob

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1920, "height": 1080, "anti_aliasing": 0})

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

stage = omni.usd.get_context().get_stage()
OUT_DIR = "/tmp/isaacsim_frames"
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


def build_scene():
    # Seabed
    seabed = UsdGeom.Mesh.Define(stage, "/World/Seabed")
    h = 25.0
    seabed.CreatePointsAttr([(-h,-h,-30),(h,-h,-30),(h,h,-30),(-h,h,-30)])
    seabed.CreateFaceVertexCountsAttr([4])
    seabed.CreateFaceVertexIndicesAttr([0,1,2,3])
    seabed.CreateNormalsAttr([(0,0,1)]*4)
    _bind(seabed.GetPrim(), _mat("/World/Seabed/Mat", (0.28, 0.22, 0.15), roughness=0.9))

    # Dock
    dock = UsdGeom.Cube.Define(stage, "/World/Dock")
    dock.CreateSizeAttr(3.0)
    xf = UsdGeom.Xformable(dock.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(4,0,-10))
    xf.AddScaleOp().Set(Gf.Vec3d(1, 0.5, 1))
    _bind(dock.GetPrim(), _mat("/World/Dock/Mat", (0.45, 0.45, 0.5), metallic=0.7, roughness=0.5))

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(600)
    dome.CreateColorAttr(Gf.Vec3f(0.12, 0.30, 0.45))

    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(2500)
    key.CreateColorAttr(Gf.Vec3f(0.5, 0.7, 0.85))
    key.CreateAngleAttr(3.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3d(-60, 20, 0))

    rov_light = UsdLux.SphereLight.Define(stage, "/World/RovLight")
    rov_light.CreateIntensityAttr(40000)
    rov_light.CreateColorAttr(Gf.Vec3f(0.7, 0.85, 1.0))
    rov_light.CreateRadiusAttr(0.1)
    UsdGeom.Xformable(rov_light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, -9.5))

    fill = UsdLux.SphereLight.Define(stage, "/World/Fill")
    fill.CreateIntensityAttr(5000)
    fill.CreateColorAttr(Gf.Vec3f(0.3, 0.5, 0.7))
    fill.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-3, -4, -6))

    # Particles
    rng = np.random.RandomState(42)
    stage.DefinePrim("/World/Particles", "Scope")
    for i in range(25):
        pos = rng.uniform([-8,-8,-22],[8,8,-4])
        s = UsdGeom.Sphere.Define(stage, f"/World/Particles/p{i}")
        s.CreateRadiusAttr(float(rng.uniform(0.03, 0.08)))
        UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        _bind(s.GetPrim(), _mat(f"/World/Particles/p{i}/M", (0.6,0.55,0.45), roughness=0.8))


def build_rov(pos=(0, 0, -10)):
    """Build BlueROV2 from primitives with bright materials."""
    rov = stage.DefinePrim("/World/ROV", "Xform")
    xf = UsdGeom.Xformable(rov)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

    hull_mat = _mat("/World/ROV/HullMat", (0.85, 0.5, 0.1), roughness=0.4)
    float_mat = _mat("/World/ROV/FloatMat", (0.9, 0.75, 0.15), roughness=0.3)
    plate_mat = _mat("/World/ROV/PlateMat", (0.25, 0.8, 0.85), roughness=0.2)
    frame_mat = _mat("/World/ROV/FrameMat", (0.5, 0.5, 0.55), metallic=0.6, roughness=0.4)
    thr_mat = _mat("/World/ROV/ThrMat", (0.2, 0.2, 0.25), metallic=0.5, roughness=0.6)
    led_mat = _mat("/World/ROV/LedMat", (1.0, 0.95, 0.8), roughness=0.1)
    cam_mat = _mat("/World/ROV/CamMat", (0.1, 0.1, 0.12), roughness=0.3)

    # Hull
    hull = UsdGeom.Cube.Define(stage, "/World/ROV/Hull")
    hull.CreateSizeAttr(1.0)
    UsdGeom.Xformable(hull.GetPrim()).AddScaleOp().Set(Gf.Vec3d(0.46, 0.36, 0.25))
    _bind(hull.GetPrim(), hull_mat)

    # E-tube
    tube = UsdGeom.Cylinder.Define(stage, "/World/ROV/ETube")
    tube.CreateRadiusAttr(0.05)
    tube.CreateHeightAttr(0.30)
    xf_t = UsdGeom.Xformable(tube.GetPrim())
    xf_t.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.05))
    xf_t.AddRotateXYZOp().Set(Gf.Vec3d(0, 90, 0))
    _bind(tube.GetPrim(), hull_mat)

    # Float
    flt = UsdGeom.Cube.Define(stage, "/World/ROV/Float")
    flt.CreateSizeAttr(1.0)
    xf_f = UsdGeom.Xformable(flt.GetPrim())
    xf_f.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.17))
    xf_f.AddScaleOp().Set(Gf.Vec3d(0.40, 0.30, 0.06))
    _bind(flt.GetPrim(), float_mat)

    # Acrylic plates
    for i, dy in enumerate([-0.19, 0.19]):
        p = UsdGeom.Cube.Define(stage, f"/World/ROV/Plate{i}")
        p.CreateSizeAttr(1.0)
        xf_p = UsdGeom.Xformable(p.GetPrim())
        xf_p.AddTranslateOp().Set(Gf.Vec3d(0, dy, 0))
        xf_p.AddScaleOp().Set(Gf.Vec3d(0.38, 0.02, 0.22))
        _bind(p.GetPrim(), plate_mat)

    # Camera dome
    cam = UsdGeom.Sphere.Define(stage, "/World/ROV/CamDome")
    cam.CreateRadiusAttr(0.04)
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.24, 0, -0.02))
    _bind(cam.GetPrim(), cam_mat)

    # 8 thrusters
    thr_pos = [
        (0.18, 0.16, 0.10), (-0.18, 0.16, 0.10),
        (0.18, -0.16, 0.10), (-0.18, -0.16, 0.10),
        (0.15, 0.18, 0.0), (-0.15, 0.18, 0.0),
        (0.15, -0.18, 0.0), (-0.15, -0.18, 0.0),
    ]
    for i, tp in enumerate(thr_pos):
        t = UsdGeom.Cylinder.Define(stage, f"/World/ROV/Thr{i}")
        t.CreateRadiusAttr(0.03)
        t.CreateHeightAttr(0.06)
        UsdGeom.Xformable(t.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*tp))
        _bind(t.GetPrim(), thr_mat)

    # Rails
    for i, dy in enumerate([-0.20, 0.20]):
        r = UsdGeom.Cylinder.Define(stage, f"/World/ROV/Rail{i}")
        r.CreateRadiusAttr(0.012)
        r.CreateHeightAttr(0.50)
        xf_r = UsdGeom.Xformable(r.GetPrim())
        xf_r.AddTranslateOp().Set(Gf.Vec3d(0, dy, -0.10))
        xf_r.AddRotateXYZOp().Set(Gf.Vec3d(0, 90, 0))
        _bind(r.GetPrim(), frame_mat)

    # LEDs
    for i, dy in enumerate([-0.12, 0.12]):
        l = UsdGeom.Sphere.Define(stage, f"/World/ROV/Led{i}")
        l.CreateRadiusAttr(0.025)
        UsdGeom.Xformable(l.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.24, dy, 0.02))
        _bind(l.GetPrim(), led_mat)

    return rov


def main():
    from isaacsim.core.api import World
    import omni.renderer_capture

    sys.stderr.write("[1/5] Building scene...\n"); sys.stderr.flush()
    world = World(stage_units_in_meters=1.0)
    build_scene()

    sys.stderr.write("[2/5] Building BlueROV2...\n"); sys.stderr.flush()
    rov = build_rov(pos=(0, 0, -10))

    sys.stderr.write("[3/5] Setting camera...\n"); sys.stderr.flush()
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 200.0))

    import omni.kit.viewport.utility as vp_util
    vp = vp_util.get_active_viewport()
    if vp:
        vp.set_active_camera("/World/Cam")

    world.reset()
    for _ in range(20):
        app.update()

    sys.stderr.write("[4/5] Rendering frames...\n"); sys.stderr.flush()
    cap = omni.renderer_capture.acquire_renderer_capture_interface()

    # Trajectory: ROV approaches dock
    n_frames = 60
    start = np.array([0.0, 0.0, -10.0])
    target = np.array([1.5, 0.0, -10.0])

    for frame in range(n_frames):
        t = frame / max(n_frames - 1, 1)
        # Smooth approach
        rov_pos = start + (target - start) * (1 - (1 - t) ** 2)

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
                l_ops[0].Set(Gf.Vec3d(rov_pos[0]+0.3, rov_pos[1], rov_pos[2]+0.5))

        # Fixed chase camera behind ROV
        angle = -0.4 + t * 0.3
        cam_r = 4.0
        cam_x = rov_pos[0] - cam_r * np.cos(angle)
        cam_y = rov_pos[1] - cam_r * np.sin(angle)
        cam_z = rov_pos[2] + 1.5
        eye = Gf.Vec3d(cam_x, cam_y, cam_z)
        look = Gf.Vec3d(*rov_pos.tolist())
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

        for _ in range(3):
            app.update()

        out_path = f"{OUT_DIR}/frame_{frame:04d}.png"
        cap.capture_next_frame_swapchain(out_path)
        for _ in range(3):
            app.update()

        if (frame + 1) % 10 == 0:
            dist = float(np.linalg.norm(rov_pos - target))
            sys.stderr.write(f"  Frame {frame+1}/{n_frames}: dist={dist:.2f}m\n")
            sys.stderr.flush()

    sys.stderr.write(f"[5/5] Frames saved to {OUT_DIR}/\n"); sys.stderr.flush()
    os._exit(0)


main()
