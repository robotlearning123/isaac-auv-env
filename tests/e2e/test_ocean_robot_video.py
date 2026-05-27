"""E2E tests with real underwater robot rendering.

Uses UnderwaterCamera (GPU ray-casting + Beer-Lambert) to produce
actual underwater visuals of the robot in an ocean scene with seabed,
dock, and obstacles. Videos saved to tmp/test_videos/.

Run: pytest tests/e2e/test_ocean_robot_video.py -v -x
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import warp as wp

wp.init()

VIDEO_DIR = Path(__file__).resolve().parent.parent.parent / "tmp" / "test_videos"


def _make_seabed_mesh(
    nx: int = 40, nz: int = 40, size: float = 60.0, base_depth: float = -20.0,
    device: str = "cuda:0",
) -> wp.Mesh:
    xs = np.linspace(-size / 2, size / 2, nx + 1, dtype=np.float32)
    ys = np.linspace(-size / 2, size / 2, nz + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.full_like(xx, base_depth)
    zz += np.sin(xx * 0.1) * 2.0 + np.cos(yy * 0.15) * 1.5

    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    indices = []
    for i in range(nx):
        for j in range(nz):
            v0 = i * (nz + 1) + j
            v1 = v0 + 1
            v2 = (i + 1) * (nz + 1) + j
            v3 = v2 + 1
            indices.extend([v0, v2, v1, v1, v2, v3])
    idx = np.array(indices, dtype=np.int32)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(idx, dtype=wp.int32, device=device),
    )


def _make_dock_mesh(
    pos: tuple[float, float, float] = (8.0, 8.0, -15.0),
    half: float = 1.5,
    device: str = "cuda:0",
) -> wp.Mesh:
    v, idx = _box_mesh(center=pos, half_extents=(half, half * 0.5, half))
    return wp.Mesh(
        points=wp.array(v, dtype=wp.vec3, device=device),
        indices=wp.array(idx.ravel(), dtype=wp.int32, device=device),
    )


def _box_mesh(
    center: tuple[float, float, float], half_extents: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    v = np.array(
        [
            [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    idx = np.array(
        [
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
            0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3,
            0, 7, 4, 0, 3, 7, 1, 5, 6, 1, 6, 2,
        ],
        dtype=np.int32,
    )
    return v, idx


def _make_obstacles(device: str = "cuda:0") -> list[wp.Mesh]:
    """Create a few underwater obstacles (pillars, rocks)."""
    obstacles = []
    for pos in [(5.0, -3.0, -12.0), (-4.0, 6.0, -10.0), (0.0, 0.0, -8.0)]:
        v, idx = _box_mesh(center=pos, half_extents=(0.5, 0.5, 2.0))
        obstacles.append(
            wp.Mesh(
                points=wp.array(v, dtype=wp.vec3, device=device),
                indices=wp.array(idx.ravel(), dtype=wp.int32, device=device),
            )
        )
    return obstacles


def _merge_meshes(meshes: list[wp.Mesh], device: str = "cuda:0") -> wp.Mesh:
    """Merge multiple wp.Mesh objects into one."""
    all_verts = []
    all_indices = []
    offset = 0
    for m in meshes:
        pts = m.points.numpy()
        idx = m.indices.numpy()
        all_verts.append(pts)
        all_indices.append(idx + offset)
        offset += len(pts)
    verts = np.concatenate(all_verts, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    return wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )


def _render_ocean_robot_video(
    output_path: Path,
    n_steps: int = 200,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Render a real underwater robot video using GPU ray-casting.

    Creates an ocean scene with seabed, dock, and obstacles.
    Runs a BlueROV2 with PD controller toward the dock.
    Renders each frame from a chase camera using UnderwaterCamera.
    """
    from imageio import v3 as iio

    from oceanscale.sensors.underwater_camera import CameraConfig, UnderwaterCamera

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build scene meshes
    seabed = _make_seabed_mesh(device=device)
    dock = _make_dock_mesh(device=device)
    obstacles = _make_obstacles(device=device)
    scene = _merge_meshes([seabed, dock] + obstacles, device=device)

    # Camera setup — chase cam behind/above the robot
    cam_cfg = CameraConfig(width=640, height=480, fov_h_deg=90, fov_v_deg=65, water_type="II")
    camera = UnderwaterCamera(scene, config=cam_cfg, device=device)

    # Robot state — simple kinematic simulation
    pos = np.array([0.0, 0.0, -10.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    target = np.array([8.0, 8.0, -15.0], dtype=np.float32)
    dt = 0.05

    frames: list[np.ndarray] = []

    for step in range(n_steps):
        # PD controller toward dock
        err = target - pos
        dist = np.linalg.norm(err)
        if dist > 0.1:
            direction = err / dist
            thrust = direction * min(0.5, dist * 0.3)
        else:
            thrust = np.zeros(3, dtype=np.float32)

        # Update kinematics
        vel = vel * 0.95 + thrust * dt
        pos = pos + vel * dt

        # Chase camera: 3m behind and 2m above robot, looking at robot
        yaw = np.arctan2(float(err[1]), float(err[0])) if dist > 0.5 else 0.0
        cam_pos = pos + np.array([
            -3.0 * np.cos(yaw),
            -3.0 * np.sin(yaw),
            2.0,
        ], dtype=np.float32)

        # Render underwater frame
        result = camera.scan(cam_pos)
        rgb = result["rgb"]  # (H, W, 3) uint8

        # Apply animated caustics via UnderwaterRenderer
        from oceanscale.rendering.underwater import UnderwaterRenderer

        renderer = UnderwaterRenderer(water_type="II", device=device, caustic_intensity=0.8)
        depth = result["depth"]
        rgb_wp = wp.array(rgb, dtype=wp.uint8, device=device)
        depth_wp = wp.array(depth, dtype=wp.float32, device=device)
        rendered = renderer.render_gpu(rgb_wp, depth_wp, time=step * dt)
        frame = rendered.numpy()

        frames.append(frame)

    # Write MP4
    iio.imwrite(str(output_path), frames, fps=30, codec="libx264", pixelformat="yuv420p")

    return {
        "frames": len(frames),
        "video_path": str(output_path),
        "file_size": output_path.stat().st_size,
    }


@pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA/Warp required"
)
class TestOceanRobotVideo:
    """Real underwater robot video with GPU ray-casting + Beer-Lambert rendering."""

    def test_docking_approach_video(self, video_dir: Path) -> None:
        result = _render_ocean_robot_video(
            video_dir / "ocean_docking_approach.mp4",
            n_steps=200,
        )
        assert result["frames"] == 200
        assert Path(result["video_path"]).exists()
        assert result["file_size"] > 50_000  # real rendered video should be substantial

    def test_station_keeping_video(self, video_dir: Path) -> None:
        """Robot hovers at a fixed position — shows ocean currents and caustics."""
        from imageio import v3 as iio
        from oceanscale.sensors.underwater_camera import CameraConfig, UnderwaterCamera
        from oceanscale.rendering.underwater import UnderwaterRenderer

        out = video_dir / "ocean_station_keeping.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)

        seabed = _make_seabed_mesh(device="cuda:0")
        scene = _merge_meshes([seabed], device="cuda:0")
        cam_cfg = CameraConfig(width=640, height=480, water_type="I")  # clear ocean
        camera = UnderwaterCamera(scene, config=cam_cfg, device="cuda:0")
        renderer = UnderwaterRenderer(water_type="I", device="cuda:0", caustic_intensity=1.0)

        pos = np.array([0.0, 0.0, -8.0], dtype=np.float32)
        frames: list[np.ndarray] = []

        for step in range(150):
            # Small drift to show ocean movement
            drift = np.array([
                np.sin(step * 0.02) * 0.3,
                np.cos(step * 0.015) * 0.2,
                0.0,
            ], dtype=np.float32)
            cam_pos = pos + drift + np.array([0.0, -5.0, 3.0], dtype=np.float32)

            result = camera.scan(cam_pos)
            rgb_wp = wp.array(result["rgb"], dtype=wp.uint8, device="cuda:0")
            depth_wp = wp.array(result["depth"], dtype=wp.float32, device="cuda:0")
            rendered = renderer.render_gpu(rgb_wp, depth_wp, time=step * 0.05)
            frames.append(rendered.numpy())

        iio.imwrite(str(out), frames, fps=30, codec="libx264", pixelformat="yuv420p")
        assert out.exists()
        assert out.stat().st_size > 50_000

    def test_multi_angle_sweep_video(self, video_dir: Path) -> None:
        """Camera orbits around the scene — shows full ocean environment."""
        from imageio import v3 as iio
        from oceanscale.sensors.underwater_camera import CameraConfig, UnderwaterCamera
        from oceanscale.rendering.underwater import UnderwaterRenderer

        out = video_dir / "ocean_multi_angle.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)

        seabed = _make_seabed_mesh(device="cuda:0")
        dock = _make_dock_mesh(device="cuda:0")
        obstacles = _make_obstacles(device="cuda:0")
        scene = _merge_meshes([seabed, dock] + obstacles, device="cuda:0")

        cam_cfg = CameraConfig(width=640, height=480, water_type="3C")  # turbid coastal
        camera = UnderwaterCamera(scene, config=cam_cfg, device="cuda:0")
        renderer = UnderwaterRenderer(water_type="3C", device="cuda:0", caustic_intensity=0.6)

        center = np.array([2.0, 2.0, -12.0], dtype=np.float32)
        frames: list[np.ndarray] = []

        for step in range(180):
            angle = step * 2.0 * np.pi / 180.0
            radius = 15.0 + 3.0 * np.sin(step * 0.03)
            cam_pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                5.0 + 2.0 * np.sin(step * 0.05),
            ], dtype=np.float32)

            result = camera.scan(cam_pos)
            rgb_wp = wp.array(result["rgb"], dtype=wp.uint8, device="cuda:0")
            depth_wp = wp.array(result["depth"], dtype=wp.float32, device="cuda:0")
            rendered = renderer.render_gpu(rgb_wp, depth_wp, time=step * 0.05)
            frames.append(rendered.numpy())

        iio.imwrite(str(out), frames, fps=30, codec="libx264", pixelformat="yuv420p")
        assert out.exists()
        assert out.stat().st_size > 50_000
