"""Render OceanScale MVP demo as 3D video using Warp OpenGL headless renderer.

Usage:
    uv run python scripts/warp_mvp_render.py [--steps 240] [--output /tmp/oceanscale_3d_demo.mp4]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import warp as wp
import warp.render

wp.init()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from PIL import Image, ImageDraw, ImageFont


# -- Physics-based underwater rendering (adapted from NVIDIA OceanSim UW_render) --

@wp.func
def _vec3_exp(v: wp.vec3):
    return wp.vec3(wp.exp(v[0]), wp.exp(v[1]), wp.exp(v[2]))


@wp.func
def _vec3_mul(a: wp.vec3, b: wp.vec3):
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])


@wp.kernel
def uw_render_kernel(
    raw_image: wp.array(ndim=3, dtype=wp.uint8),
    depth_image: wp.array(ndim=2, dtype=wp.float32),
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    uw_image: wp.array(ndim=3, dtype=wp.uint8),
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
    uw_image[i, j, 0] = wp.uint8(wp.clamp(result[0], 0.0, 255.0))
    uw_image[i, j, 1] = wp.uint8(wp.clamp(result[1], 0.0, 255.0))
    uw_image[i, j, 2] = wp.uint8(wp.clamp(result[2], 0.0, 255.0))


@wp.kernel
def caustic_kernel(
    image: wp.array(ndim=3, dtype=wp.uint8),
    time_val: wp.float32,
    out: wp.array(ndim=3, dtype=wp.uint8),
):
    i, j = wp.tid()
    x = wp.float32(j)
    y = wp.float32(i)
    c1 = wp.sin(x * 0.02 + time_val * 1.8) * wp.cos(y * 0.025 - time_val * 1.2)
    c2 = wp.sin((x + y) * 0.015 + time_val * 2.5)
    c = (c1 + c2) * 0.5
    boost_r = c * 3.0
    boost_g = c * 6.0
    boost_b = c * 10.0
    out[i, j, 0] = wp.uint8(wp.clamp(wp.float32(image[i, j, 0]) + boost_r, 0.0, 255.0))
    out[i, j, 1] = wp.uint8(wp.clamp(wp.float32(image[i, j, 1]) + boost_g, 0.0, 255.0))
    out[i, j, 2] = wp.uint8(wp.clamp(wp.float32(image[i, j, 2]) + boost_b, 0.0, 255.0))


def apply_underwater_postfx(
    frame: np.ndarray,
    depth_hint: float,
    time: float,
    distance: float = 0.0,
    step: int = 0,
    total_steps: int = 240,
    depth_map: np.ndarray | None = None,
) -> np.ndarray:
    """Physics-based underwater rendering + HUD overlay."""
    h, w = frame.shape[:2]

    # GPU underwater light transport (OceanSim model)
    raw_gpu = wp.array(frame[:, :, :3], dtype=wp.uint8, device="cuda:0")
    if depth_map is not None:
        dm = depth_map.astype(np.float32)
    else:
        dm = np.full((h, w), depth_hint, dtype=np.float32)
    depth_gpu = wp.array(dm, dtype=wp.float32, device="cuda:0")
    uw_gpu = wp.zeros((h, w, 3), dtype=wp.uint8, device="cuda:0")

    # Coastal water parameters (Jerlov Type II)
    backscatter = wp.vec3(0.0, 0.31, 0.24)
    atten = wp.vec3(0.05, 0.05, 0.05)
    back_coeff = wp.vec3(0.05, 0.05, 0.2)

    wp.launch(uw_render_kernel, dim=(h, w), inputs=[
        raw_gpu, depth_gpu, backscatter, atten, back_coeff, uw_gpu,
    ], device="cuda:0")

    # Caustics
    caustic_out = wp.zeros((h, w, 3), dtype=wp.uint8, device="cuda:0")
    wp.launch(caustic_kernel, dim=(h, w), inputs=[
        uw_gpu, wp.float32(time), caustic_out,
    ], device="cuda:0")

    img = caustic_out.numpy()

    # Vignette (numpy, fast enough)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    vig = 1.0 - 0.30 * np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    vig = np.clip(vig, 0.45, 1.0)[:, :, np.newaxis]
    img = np.clip(img.astype(np.float32) * vig, 0, 255).astype(np.uint8)

    # Film grain
    rng = np.random.RandomState(step)
    noise = rng.normal(0, 1.5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # HUD overlay
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font_lg = ImageFont.load_default()
        font_sm = font_lg
        font_xs = font_lg

    hud = (120, 200, 255)
    accent = (255, 160, 60)

    draw.text((30, 20), "OceanScale", fill=hud, font=font_lg)
    draw.text((30, 54), "GPU-native ocean simulation", fill=(80, 150, 200), font=font_xs)

    mt = "BlueROV2 Heavy | Dock Approach"
    tw = draw.textbbox((0, 0), mt, font=font_sm)[2]
    draw.text((w - tw - 30, 22), mt, fill=hud, font=font_sm)

    dt = f"Distance: {distance:.2f} m"
    draw.text((30, h - 55), dt, fill=accent if distance < 0.5 else hud, font=font_sm)

    st = f"Step {step}/{total_steps}"
    tw = draw.textbbox((0, 0), st, font=font_sm)[2]
    draw.text((w - tw - 30, h - 55), st, fill=hud, font=font_sm)

    dpt = f"Depth: {10.0 + depth_hint * 0.1:.1f} m"
    tw = draw.textbbox((0, 0), dpt, font=font_xs)[2]
    draw.text(((w - tw) // 2, h - 50), dpt, fill=(80, 170, 200), font=font_xs)

    cx_i, cy_i = w // 2, h // 2
    draw.line([(cx_i - 12, cy_i), (cx_i + 12, cy_i)], fill=(100, 200, 180), width=1)
    draw.line([(cx_i, cy_i - 12), (cx_i, cy_i + 12)], fill=(100, 200, 180), width=1)

    # Title card fade-in for first 30 frames
    title_frames = 30
    if step <= title_frames:
        alpha = min(step / max(title_frames, 1), 1.0)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
            sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        except OSError:
            title_font = font_lg
            sub_font = font_sm

        title = "OceanScale"
        sub = "GPU-Native Ocean Simulation"
        t_bbox = draw.textbbox((0, 0), title, font=title_font)
        s_bbox = draw.textbbox((0, 0), sub, font=sub_font)
        t_w = t_bbox[2] - t_bbox[0]
        s_w = s_bbox[2] - s_bbox[0]

        a_int = int(alpha * 220)
        draw.text(
            ((w - t_w) // 2, h // 2 - 50), title,
            fill=(180, 230, 255, a_int), font=title_font,
        )
        draw.text(
            ((w - s_w) // 2, h // 2 + 20), sub,
            fill=(120, 190, 230, a_int), font=sub_font,
        )

    return np.array(pil_img)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--output", type=str, default="/tmp/oceanscale_3d_demo.mp4")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def build_renderer(W: int, H: int, fps: int) -> warp.render.OpenGLRenderer:
    return warp.render.OpenGLRenderer(
        fps=fps,
        screen_width=W,
        screen_height=H,
        headless=True,
        background_color=(0.04, 0.10, 0.20),
        draw_sky=False,
        draw_grid=False,
        draw_axis=False,
        show_info=False,
        camera_pos=(-3.0, -4.0, 3.0),
        camera_front=(0.4, 0.5, -0.3),
        camera_up=(0.0, 0.0, 1.0),
        near_plane=0.1,
        far_plane=100.0,
        camera_fov=55.0,
    )


def scene_pos(sim_pos: np.ndarray, origin: np.ndarray) -> tuple[float, float, float]:
    """Map simulation coords (z=-10 depth) to renderer coords (z~0)."""
    p = sim_pos - origin
    return (float(p[0]), float(p[1]), float(p[2]))


def render_static_scene(renderer: warp.render.OpenGLRenderer, dock_pos, target_pos):
    """Render static elements once on frame 0."""
    renderer.render_box(
        pos=(0, 0, -20.5), rot=(0, 0, 0, 1),
        extents=(30, 30, 0.5), name="seabed", color=(0.22, 0.17, 0.12),
    )
    renderer.render_box(
        pos=dock_pos, rot=(0, 0, 0, 1),
        extents=(1.5, 1.0, 1.5), name="dock", color=(0.55, 0.55, 0.60),
    )
    for idx, (dx, dy) in enumerate([(-1.2, -0.8), (1.2, -0.8), (-1.2, 0.8), (1.2, 0.8)]):
        renderer.render_cylinder(
            pos=(dock_pos[0] + dx, dock_pos[1] + dy, dock_pos[2]),
            rot=(0, 0, 0, 1), radius=0.06, half_height=1.5,
            name=f"leg_{idx}", color=(0.4, 0.4, 0.42),
        )
    renderer.render_sphere(
        pos=target_pos, rot=(0, 0, 0, 1),
        radius=0.12, name="target", color=(1.0, 0.3, 0.1),
    )


def render_frame(
    renderer: warp.render.OpenGLRenderer,
    frame_time: float,
    rov_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    dock_pos: tuple[float, float, float],
    is_first: bool = False,
) -> None:
    renderer.begin_frame(frame_time)

    if is_first:
        render_static_scene(renderer, dock_pos, target_pos)

    renderer.render_box(
        pos=rov_pos, rot=(0, 0, 0, 1),
        extents=(0.23, 0.29, 0.19), name="rov_body", color=(0.20, 0.50, 0.90),
    )
    renderer.render_box(
        pos=(rov_pos[0], rov_pos[1], rov_pos[2] + 0.20),
        rot=(0, 0, 0, 1), extents=(0.30, 0.35, 0.02),
        name="rov_frame", color=(0.08, 0.08, 0.08),
    )
    for idx, (dx, dy) in enumerate([(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]):
        renderer.render_cylinder(
            pos=(rov_pos[0] + dx, rov_pos[1] + dy, rov_pos[2] + 0.15),
            rot=(0, 0, 0, 1), radius=0.05, half_height=0.06,
            name=f"thr_{idx}", color=(0.5, 0.5, 0.5),
        )
    for idx, dy in enumerate([-0.15, 0.15]):
        renderer.render_sphere(
            pos=(rov_pos[0] + 0.25, rov_pos[1] + dy, rov_pos[2]),
            rot=(0, 0, 0, 1), radius=0.04,
            name=f"light_{idx}", color=(1.0, 0.95, 0.7),
        )

    renderer.end_frame()


def main():
    args = parse_args()
    W, H = args.width, args.height

    from oceanscale.mvp import UnderwaterRobotMVPConfig, compute_mission_action
    from oceanscale.demo import UnifiedDemo
    from oceanscale.vehicles import BlueROV2Heavy

    print("[1/4] Setting up OceanScale physics...")
    vehicle = BlueROV2Heavy()
    dock_sim = np.array([4.0, 0.0, -10.0], dtype=np.float32)
    target_sim = dock_sim - np.array([2.5, 0.0, 0.0], dtype=np.float32)

    demo = UnifiedDemo(
        seabed_depth=-30.0, wave_height=1.2, wave_period=8.0,
        current_speed=0.25, tether_length=25.0, dt=0.02,
        device="cuda:0", vehicle=vehicle,
        dock_position=tuple(dock_sim.tolist()),
        enable_structures=False, asset_source="usd",
    )

    origin = np.array([0.0, 0.0, -10.0], dtype=np.float32)

    print("[2/4] Setting up Warp renderer...")
    renderer = build_renderer(W, H, args.fps)
    image = wp.empty(shape=(H, W, 3), dtype=float)
    depth_buf = wp.empty(shape=(H, W, 1), dtype=float)

    print(f"[3/4] Running {args.steps}-step simulation + rendering...")
    obs = demo.reset()
    frames = []
    rng = np.random.RandomState(42)

    for step in range(args.steps):
        action = compute_mission_action(
            obs["rov_position"], obs["rov_velocity"], target_sim,
        )

        rov_scene = scene_pos(obs["rov_position"], origin)
        target_scene = scene_pos(target_sim, origin)
        dock_scene = scene_pos(dock_sim, origin)

        # Cinematic orbit camera — full 360 with height variation
        t = step / max(args.steps, 1)
        angle = t * 2.0 * np.pi
        cam_r = 6.0
        cam_x = rov_scene[0] + cam_r * np.cos(angle) - 1.5
        cam_y = rov_scene[1] + cam_r * np.sin(angle) - 1.5
        cam_z = rov_scene[2] + 2.5 + 1.0 * np.sin(t * np.pi * 2)
        look = np.array(rov_scene)
        cam_dir = look - np.array([cam_x, cam_y, cam_z])
        cam_dir = cam_dir / np.linalg.norm(cam_dir)

        renderer._camera_pos = (cam_x, cam_y, cam_z)
        renderer._camera_front = tuple(cam_dir.tolist())

        render_frame(
            renderer, step * demo.dt,
            rov_scene, target_scene, dock_scene,
            is_first=(step == 0),
        )

        renderer.get_pixels(image, split_up_tiles=False, mode="rgb")
        pixels = (np.clip(image.numpy(), 0, 1) * 255).astype(np.uint8)
        raw = np.flipud(pixels).copy()

        # Depth buffer for per-pixel underwater rendering
        try:
            renderer.get_pixels(depth_buf, split_up_tiles=False, mode="depth")
            depth_np = np.flipud(depth_buf.numpy()[:, :, 0]).copy()
            depth_np = np.clip(depth_np, 0.1, 50.0)
        except Exception:
            depth_np = None

        obs = demo.step(action * 0.4)
        dist = float(np.linalg.norm(obs["rov_position"] - target_sim))

        depth_hint = float(-obs["rov_position"][2])
        fx_frame = apply_underwater_postfx(
            raw, depth_hint=depth_hint, time=step * demo.dt,
            distance=dist, step=step + 1, total_steps=args.steps,
            depth_map=depth_np,
        )
        frames.append(fx_frame)

        if (step + 1) % 30 == 0:
            print(f"  Step {step+1}/{args.steps}: dist={dist:.2f}m")

        if dist <= 0.15 and step > 0:
            if not hasattr(main, '_completed'):
                print(f"  Mission complete at step {step+1}! (continuing render)")
                main._completed = True

    renderer.clear()

    print(f"[4/4] Encoding {len(frames)} frames to {args.output}...")
    import imageio

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"\nDone: {args.output} ({len(frames)} frames, {len(frames)/args.fps:.1f}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
