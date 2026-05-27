#!/usr/bin/env python3
"""Run bounded Isaac Sim 6 standalone example/test smokes."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/storage/isaacsim-6.0-official")
DEFAULT_SIM = DEFAULT_ROOT / "sources" / "IsaacSim-develop"
DEFAULT_VENV = DEFAULT_ROOT / "venv-isaacsim-strict"
ROS2_INTERNAL_JAZZY_ENV = (
    ("ROS_DISTRO", "jazzy"),
    ("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
    (
        "LD_LIBRARY_PATH",
        "{old_ld_library_path}:{venv}/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib",
    ),
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    relpath: str
    args: tuple[str, ...] = ()
    timeout_s: int = 120
    markers: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()
    expected_timeout: bool = False
    marker_grace_s: float = 5.0
    failure_markers: tuple[str, ...] = (
        "Traceback (most recent call last):",
        "[fatal]",
        "RuntimeError:",
        "AttributeError:",
        "SystemError:",
        "ModuleNotFoundError:",
        "ROS2 Bridge startup failed",
    )


CASES: tuple[SmokeCase, ...] = (
    SmokeCase(
        name="asset_importer_mjcf_test",
        relpath="source/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py",
        args=("--test",),
        timeout_s=180,
        markers=("MJCF import successful.",),
    ),
    SmokeCase(
        name="asset_importer_urdf_test",
        relpath="source/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py",
        args=("--test",),
        timeout_s=180,
        markers=("URDF import successful.",),
    ),
    SmokeCase(
        name="asset_transformer_test",
        relpath="source/standalone_examples/api/isaacsim.asset.transformer/run_asset_transformer.py",
        args=("--test",),
        timeout_s=180,
        markers=("Asset transformation completed successfully.",),
    ),
    SmokeCase(
        name="simulation_app_hello_world",
        relpath="source/standalone_examples/api/isaacsim.simulation_app/hello_world.py",
        args=("--headless",),
        timeout_s=90,
        markers=("Hello World!",),
    ),
    SmokeCase(
        name="simulation_app_fetch_results",
        relpath="source/standalone_examples/testing/isaacsim.simulation_app/test_fetch_results.py",
        timeout_s=90,
        markers=("Start test", "Fetch results", "Finish Test"),
    ),
    SmokeCase(
        name="simulation_app_ogn",
        relpath="source/standalone_examples/testing/isaacsim.simulation_app/test_ogn.py",
        timeout_s=90,
        markers=("Hello", "Goodbye"),
    ),
    SmokeCase(
        name="simulation_app_syntheticdata",
        relpath="source/standalone_examples/testing/isaacsim.simulation_app/test_syntheticdata.py",
        timeout_s=180,
        markers=("3686400",),
    ),
    SmokeCase(
        name="core_api_omnigraph_triggers",
        relpath="source/standalone_examples/api/isaacsim.core.api/omnigraph_triggers.py",
        timeout_s=180,
        markers=("Starting just the app.", "separate rendering and physics stepping."),
    ),
    SmokeCase(
        name="core_api_simulation_callbacks",
        relpath="source/standalone_examples/api/isaacsim.core.api/simulation_callbacks.py",
        timeout_s=240,
        markers=("step 59", "Current joint 2 position @ step", "Render Frame"),
    ),
    SmokeCase(
        name="core_api_add_cubes",
        relpath="source/standalone_examples/api/isaacsim.core.api/add_cubes.py",
        args=("--headless",),
        timeout_s=240,
        markers=("array(",),
    ),
    SmokeCase(
        name="core_api_add_frankas_test",
        relpath="source/standalone_examples/api/isaacsim.core.api/add_frankas.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("resetting...", "Franka 1's joint positions are:", "Franka 2's joint positions are:"),
    ),
    SmokeCase(
        name="core_api_control_robot",
        relpath="source/standalone_examples/api/isaacsim.core.api/control_robot.py",
        args=("--headless",),
        timeout_s=300,
        markers=("Reached:",),
    ),
    SmokeCase(
        name="core_api_data_logging",
        relpath="source/standalone_examples/api/isaacsim.core.api/data_logging.py",
        args=("--headless",),
        timeout_s=300,
        markers=("joint_positions", "applied_joint_positions"),
    ),
    SmokeCase(
        name="core_api_simulate_robot",
        relpath="source/standalone_examples/api/isaacsim.core.api/simulate_robot.py",
        args=("--headless",),
        timeout_s=300,
        markers=("Finished simulating for 1000 steps",),
    ),
    SmokeCase(
        name="core_api_cloth_test",
        relpath="source/standalone_examples/api/isaacsim.core.api/cloth.py",
        args=("--test", "--headless"),
        timeout_s=300,
        markers=("cloth 0 average height",),
    ),
    SmokeCase(
        name="core_api_visual_materials_test",
        relpath="source/standalone_examples/api/isaacsim.core.api/visual_materials.py",
        args=("--test", "--headless"),
        timeout_s=180,
        markers=("Finished simulating for 10000 steps",),
    ),
    SmokeCase(
        name="core_api_rigid_contact_view_test",
        relpath="source/standalone_examples/api/isaacsim.core.api/rigid_contact_view.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("Bottom box net forces:", "ground net force from GeometryPrim"),
    ),
    SmokeCase(
        name="core_api_detailed_contact_data_test",
        relpath="source/standalone_examples/api/isaacsim.core.api/detailed_contact_data.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("friction forces:", "contact forces:"),
    ),
    SmokeCase(
        name="core_api_time_stepping",
        relpath="source/standalone_examples/api/isaacsim.core.api/time_stepping.py",
        timeout_s=240,
        markers=("step physics once with a step size", "cleanup and exit"),
    ),
    SmokeCase(
        name="core_experimental_omnigraph_triggers",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/omnigraph_triggers.py",
        args=("--test",),
        timeout_s=120,
        markers=("Starting just the app.", "Separate rendering and physics stepping."),
    ),
    SmokeCase(
        name="core_experimental_simulation_callbacks",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/simulation_callbacks.py",
        args=("--test",),
        timeout_s=180,
        markers=("Step 0", "Current joint 2 position @ step", "Render Frame"),
    ),
    SmokeCase(
        name="core_experimental_add_cubes_test",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py",
        args=("--test", "--headless"),
        timeout_s=180,
        markers=("Angular velocity:", "World pose - Position:"),
    ),
    SmokeCase(
        name="core_experimental_control_robot_numpy_cpu",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/control_robot_numpy.py",
        args=("--device", "cpu", "--headless"),
        timeout_s=300,
    ),
    SmokeCase(
        name="core_experimental_control_robot_torch_cpu",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/control_robot_torch.py",
        args=("--device", "cpu", "--headless"),
        timeout_s=300,
    ),
    SmokeCase(
        name="core_experimental_control_robot_warp_cpu",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/control_robot_warp.py",
        args=("--device", "cpu", "--headless"),
        timeout_s=300,
    ),
    SmokeCase(
        name="core_experimental_deformable_stress_visualization_test",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/deformable_stress_visualization.py",
        args=("--test", "--headless"),
        timeout_s=300,
        markers=("von Mises stress range:",),
    ),
    SmokeCase(
        name="core_experimental_visual_materials_test",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/visual_materials.py",
        args=("--test", "--headless"),
        timeout_s=180,
        markers=("Finished simulating for 10000 steps",),
    ),
    SmokeCase(
        name="core_experimental_control_frankas_test",
        relpath="source/standalone_examples/api/isaacsim.core.experimental.api/control_frankas.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("DOF index:", "Franka 1 final joint positions:"),
    ),
    SmokeCase(
        name="simulation_app_load_stage_franka",
        relpath="source/standalone_examples/api/isaacsim.simulation_app/load_stage.py",
        args=("--headless", "--test", "--usd_path", "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"),
        timeout_s=240,
        markers=("Loading stage...", "Loading Complete"),
    ),
    SmokeCase(
        name="simulation_app_async_call",
        relpath="source/standalone_examples/api/isaacsim.simulation_app/async_call.py",
        args=("--headless",),
        timeout_s=240,
        markers=("Populating stage asynchronously:", " - Done populating stage", "Collected USD saved at:"),
    ),
    SmokeCase(
        name="core_cloner_clone_ants",
        relpath="source/standalone_examples/api/isaacsim.core.cloner/clone_ants.py",
        args=("--headless",),
        timeout_s=300,
    ),
    SmokeCase(
        name="motion_generation_trajectory_min_time",
        relpath="source/standalone_examples/api/isaacsim.robot_motion.experimental.motion_generation/trajectory_example.py",
        args=("--test", "--no-window"),
        timeout_s=300,
        markers=("Motion Generation API - Trajectory Following Example", "Trajectory following complete!", "Example Complete"),
    ),
    SmokeCase(
        name="motion_generation_trajectory_linear",
        relpath="source/standalone_examples/api/isaacsim.robot_motion.experimental.motion_generation/trajectory_example.py",
        args=("--test", "--no-window", "--linear"),
        timeout_s=300,
        markers=("Trajectory Following Example: LinearTrajectory", "Trajectory following complete!", "Example Complete"),
    ),
    SmokeCase(
        name="motion_generation_mobile_robot_control",
        relpath=(
            "source/standalone_examples/api/"
            "isaacsim.robot_motion.experimental.motion_generation/mobile_robot_control_example.py"
        ),
        args=("--test", "--no-window"),
        timeout_s=300,
        markers=("Motion Generation API - Differential Drive Controller Example", "Example Complete"),
    ),
    SmokeCase(
        name="motion_generation_scene_interaction",
        relpath=(
            "source/standalone_examples/api/"
            "isaacsim.robot_motion.experimental.motion_generation/scene_interaction_example.py"
        ),
        args=("--no-window",),
        timeout_s=300,
        markers=("Motion Generation API - Scene Interaction Example", "Complete workflow demonstrated successfully!"),
    ),
    SmokeCase(
        name="replicator_sdg_getting_started_01",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Output directory:", "Step 2"),
    ),
    SmokeCase(
        name="replicator_sdg_getting_started_02",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_02.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Output directory:", "Step 2", "[Annotator][Top][2]"),
    ),
    SmokeCase(
        name="replicator_sdg_getting_started_03",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_03.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Output directory:", "Step 2"),
    ),
    SmokeCase(
        name="replicator_sdg_getting_started_04",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_04.py",
        args=("--headless",),
        timeout_s=240,
        markers=("Output directory:", "Step 0;"),
    ),
    SmokeCase(
        name="replicator_sdg_getting_started_05",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_05.py",
        args=("--headless",),
        timeout_s=300,
        markers=("[SDG] Running with wait_for_render=True", "[SDG] Avg randomization:"),
    ),
    SmokeCase(
        name="replicator_multi_camera",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/multi_camera.py",
        args=("--headless",),
        timeout_s=240,
        markers=("Writing writer data to", "Writing annotator data to", "Step 4"),
    ),
    SmokeCase(
        name="replicator_custom_event_and_write",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/custom_event_and_write.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Writing data to", "Moving large cube position"),
    ),
    SmokeCase(
        name="replicator_custom_fps_writer_annotator",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/custom_fps_writer_annotator.py",
        args=("--headless",),
        timeout_s=240,
        markers=("Writer data will be written to:", "Capturing frame 5"),
    ),
    SmokeCase(
        name="replicator_simulation_get_data",
        relpath="source/standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py",
        args=("--headless",),
        timeout_s=300,
        markers=("Outputting data to", "Cube_0 stopped moving"),
    ),
    SmokeCase(
        name="benchmark_nucleus_kpis_json",
        relpath="source/standalone_examples/benchmarks/benchmark_nucleus_kpis.py",
        args=("--backend-type", "JSONFileMetrics"),
        timeout_s=300,
    ),
    SmokeCase(
        name="manipulator_ur10_pick_up_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/ur10_pick_up.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="manipulator_franka_gripper_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_gripper.py",
        args=("--test", "--headless"),
        timeout_s=180,
    ),
    SmokeCase(
        name="manipulator_franka_pick_place_cpu",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/franka/pick_place.py",
        args=("--device", "cpu", "--headless"),
        timeout_s=300,
        markers=("Starting pick-and-place execution",),
        expected_timeout=True,
        marker_grace_s=12.0,
    ),
    SmokeCase(
        name="manipulator_ur10e_gripper_control_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/gripper_control.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="manipulator_ur10e_pick_up_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/pick_up_example.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("done picking and placing",),
        expected_timeout=True,
        marker_grace_s=8.0,
    ),
    SmokeCase(
        name="manipulator_cobotta_follow_target_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/follow_target_example.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="manipulator_cobotta_gripper_control_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/gripper_control.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="manipulator_cobotta_pick_up_test",
        relpath="source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/pick_up_example.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="policy_spot_standalone_test_cpu",
        relpath="source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py",
        args=("--test", "--device", "cpu", "--headless"),
        timeout_s=300,
        markers=("Using device: cpu", "Reached:"),
    ),
    SmokeCase(
        name="policy_h1_standalone_cpu",
        relpath="source/standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py",
        args=("--num-robots", "1", "--device", "cpu", "--headless"),
        timeout_s=300,
        markers=("Number of robots: 1", "Using device: cpu"),
        expected_timeout=True,
        marker_grace_s=15.0,
    ),
    SmokeCase(
        name="wheeled_kaya_holonomic_move",
        relpath="source/standalone_examples/api/isaacsim.robot.wheeled_robots.examples/kaya_holonomic_move.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Simulation App Startup Complete",),
        expected_timeout=True,
        marker_grace_s=15.0,
    ),
    SmokeCase(
        name="wheeled_jetbot_differential_move_test",
        relpath="source/standalone_examples/api/isaacsim.robot.wheeled_robots.examples/jetbot_differential_move.py",
        args=("--test", "--headless"),
        timeout_s=240,
    ),
    SmokeCase(
        name="sensor_camera_stereoscopic_depth_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_stereoscopic_depth.py",
        args=("--test",),
        timeout_s=240,
    ),
    SmokeCase(
        name="sensor_camera_basic_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera.py",
        args=("--test", "--headless", "--disable-output"),
        timeout_s=300,
        markers=("Saving image to:",),
    ),
    SmokeCase(
        name="sensor_camera_add_depth_sensor_usd",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_add_depth_sensor.py",
        args=("--headless",),
        timeout_s=120,
    ),
    SmokeCase(
        name="sensor_camera_annotator_device",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_annotator_device.py",
        args=("--headless",),
        timeout_s=300,
        markers=("Testing: rgba", "Testing: pointcloud", "[PASS]"),
        failure_markers=(
            "Traceback (most recent call last):",
            "[fatal]",
            "RuntimeError:",
            "AttributeError:",
            "SystemError:",
            "ModuleNotFoundError:",
            "ROS2 Bridge startup failed",
            "[FAIL]",
        ),
    ),
    SmokeCase(
        name="sensor_camera_opencv_fisheye",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_fisheye.py",
        timeout_s=180,
        markers=("Saving the rendered image to:", "Saving the asset to:"),
    ),
    SmokeCase(
        name="sensor_camera_opencv_pinhole",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_opencv_pinhole.py",
        timeout_s=180,
        markers=("Saving the rendered image to:", "Saving the asset to:"),
    ),
    SmokeCase(
        name="sensor_camera_pre_isp_pipeline",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_pre_isp_pipeline.py",
        args=("--output-dir", "pre_isp_camera_pipeline_outputs"),
        timeout_s=180,
    ),
    SmokeCase(
        name="sensor_camera_ros_projection",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_ros.py",
        timeout_s=240,
        markers=("Header Frame ID:", "Saving the rendered image to:", "Saving the asset to:"),
    ),
    SmokeCase(
        name="sensor_camera_view",
        relpath="source/standalone_examples/api/isaacsim.sensors.camera/camera_view.py",
        args=("--headless",),
        timeout_s=300,
        markers=("out_dir:", "rgb_tiled_np.shape:", "depth_tiled_np.shape:"),
    ),
    SmokeCase(
        name="sensor_rtx_lidar_basic_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/create_lidar_basic.py",
        args=("--test", "--headless"),
        timeout_s=300,
    ),
    SmokeCase(
        name="sensor_rtx_radar_basic_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/create_radar_basic.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("Starting simulation - observe the radar detections",),
    ),
    SmokeCase(
        name="sensor_rtx_nonvisual_materials_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/apply_nonvisual_materials.py",
        args=("--test", "--headless"),
        timeout_s=240,
        markers=("Creating cubes with visual and non-visual materials", "Created RTX Lidar"),
    ),
    SmokeCase(
        name="sensor_rtx_lidar_config_variants_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/create_lidar_with_config_and_variants.py",
        args=("--test", "--headless"),
        timeout_s=300,
        markers=("Available Lidar Configurations",),
    ),
    SmokeCase(
        name="sensor_rtx_lidar_gmo_inspect_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/inspect_lidar_gmo.py",
        args=("--test",),
        timeout_s=300,
        markers=("Auxiliary level: FULL", "Sample Point Cloud Data"),
    ),
    SmokeCase(
        name="sensor_rtx_radar_gmo_inspect_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/inspect_radar_gmo.py",
        args=("--test",),
        timeout_s=300,
        markers=("Radar GMO Data", "Sample Point Cloud Data"),
    ),
    SmokeCase(
        name="sensor_rtx_lidar_object_ids_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/resolve_lidar_object_ids.py",
        args=("--test",),
        timeout_s=300,
        markers=("Object ID to Prim Path Mapping",),
    ),
    SmokeCase(
        name="sensor_rtx_lidar_robot_integration_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.rtx/lidar_robot_integration.py",
        args=("--test", "--headless"),
        timeout_s=300,
    ),
    SmokeCase(
        name="sensor_physics_contact_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.physics/contact_sensor.py",
        args=("--test", "--headless"),
        timeout_s=180,
        markers=("physics_step",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_physics_effort_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.physics/effort_sensor.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Sensor Time:",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_physics_imu_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.physics/imu_sensor.py",
        args=("--headless",),
        timeout_s=180,
        markers=("lin_acc",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_experimental_physics_contact_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.experimental.physics/contact_sensor.py",
        args=("--test", "--headless"),
        timeout_s=180,
        markers=("physics_step",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_experimental_physics_effort_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.experimental.physics/effort_sensor.py",
        args=("--headless",),
        timeout_s=180,
        markers=("Sensor Time:",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_experimental_physics_imu_api",
        relpath="source/standalone_examples/api/isaacsim.sensors.experimental.physics/imu_sensor.py",
        args=("--headless",),
        timeout_s=180,
        markers=("lin_acc",),
        expected_timeout=True,
    ),
    SmokeCase(
        name="sensor_physx_rotating_lidar_test",
        relpath="source/standalone_examples/api/isaacsim.sensors.physx/rotating_lidar_physX.py",
        args=("--test", "--headless"),
        timeout_s=180,
    ),
    SmokeCase(
        name="sensors_physics_contact",
        relpath="source/standalone_examples/testing/isaacsim.sensors.physics/contact_sensor_test.py",
        timeout_s=120,
        markers=("cube pose",),
    ),
    SmokeCase(
        name="sensors_experimental_physics_contact",
        relpath="source/standalone_examples/testing/isaacsim.sensors.experimental.physics/contact_sensor_test.py",
        timeout_s=120,
        markers=("cube pose",),
    ),
    SmokeCase(
        name="ros2_bridge_enable_extension_internal_fallback",
        relpath="source/standalone_examples/testing/isaacsim.ros2.bridge/enable_extension.py",
        args=("--headless",),
        timeout_s=180,
        markers=("isaacsim.ros2.bridge-5.0.0] startup",),
        extra_env=ROS2_INTERNAL_JAZZY_ENV,
    ),
    SmokeCase(
        name="ros2_bridge_clock_internal",
        relpath="source/standalone_examples/api/isaacsim.ros2.bridge/clock.py",
        timeout_s=180,
        markers=("sim time:", "manual stepped sim time:"),
        extra_env=ROS2_INTERNAL_JAZZY_ENV,
    ),
    SmokeCase(
        name="ros2_bridge_camera_tf_delay_internal",
        relpath="source/standalone_examples/testing/isaacsim.ros2.bridge/test_camera_tf_delay.py",
        args=("--test-steps", "5"),
        timeout_s=240,
        markers=("TF / CAMERA TIMESTAMP SYNC TEST", "Zero-delta steps:", "Status:         PASS"),
        extra_env=ROS2_INTERNAL_JAZZY_ENV,
        failure_markers=("[fatal]", "[error]", "RuntimeError:", "AttributeError:", "SystemError:", "ROS2 Bridge startup failed"),
    ),
    SmokeCase(
        name="testing_core_api_articulation",
        relpath="source/standalone_examples/testing/isaacsim.core.api/test_articulation.py",
        timeout_s=360,
        markers=(
            "[PASS] test_articulation_root",
            "[PASS] test_articulation_determinism",
            "[PASS] test_tensor_api_handles",
        ),
    ),
    SmokeCase(
        name="testing_core_api_time_stepping",
        relpath="source/standalone_examples/testing/isaacsim.core.api/test_time_stepping.py",
        timeout_s=360,
        markers=("Test physics update on render call", "cleanup and exit"),
    ),
)


def stop_process_group(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, sig)


def run_case(
    case: SmokeCase, sim_root: Path, python_bin: Path, venv: Path, logs_dir: Path, env: dict[str, str]
) -> dict[str, object]:
    log_path = logs_dir / f"{case.name}.log"
    case_work_dir = logs_dir / "work" / case.name
    case_work_dir.mkdir(parents=True, exist_ok=True)
    script = sim_root / case.relpath
    cmd = (str(python_bin), str(script), *case.args)
    case_env = env.copy()
    resolved_extra_env: dict[str, str] = {}
    for key, value in case.extra_env:
        resolved_extra_env[key] = value.format(venv=venv, old_ld_library_path=case_env.get("LD_LIBRARY_PATH", ""))
        case_env[key] = resolved_extra_env[key]
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=case_work_dir,
        env=case_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    timed_out = False
    marker_stopped = False
    output_chunks: list[str] = []
    deadline = start + case.timeout_s
    settle_deadline: float | None = None

    while True:
        now = time.monotonic()
        if proc.poll() is not None:
            output_chunks.append(proc.stdout.read() or "")
            break
        if settle_deadline is not None and now >= settle_deadline:
            marker_stopped = True
            stop_process_group(proc, signal.SIGINT)
            break
        if now >= deadline:
            timed_out = True
            stop_process_group(proc, signal.SIGINT)
            break

        wait_s = min(0.5, max(0.0, deadline - now))
        if settle_deadline is not None:
            wait_s = min(wait_s, max(0.0, settle_deadline - now))
        readable, _, _ = select.select([proc.stdout], [], [], wait_s)
        if readable:
            line = proc.stdout.readline()
            output_chunks.append(line)
            current_output = "".join(output_chunks)
            if case.expected_timeout and settle_deadline is None and all(
                marker in current_output for marker in case.markers
            ):
                settle_deadline = time.monotonic() + case.marker_grace_s

    if timed_out or marker_stopped:
        try:
            output_chunks.append(proc.communicate(timeout=25)[0] or "")
        except subprocess.TimeoutExpired:
            stop_process_group(proc, signal.SIGKILL)
            output_chunks.append(proc.communicate(timeout=10)[0] or "")
    else:
        proc.wait(timeout=10)

    output = "".join(output_chunks)
    elapsed = time.monotonic() - start
    exit_code = proc.returncode
    markers_found = {marker: marker in output for marker in case.markers}
    failures_found = {marker: marker in output for marker in case.failure_markers}
    stop_ok = (exit_code == 0 and not timed_out and not marker_stopped) or (
        case.expected_timeout and (timed_out or marker_stopped)
    )
    passed = stop_ok and all(markers_found.values()) and not any(failures_found.values())

    log_path.write_text("$ " + " ".join(cmd) + "\n" + output + f"EXIT:{exit_code}\n", encoding="utf-8")
    result = {
        "name": case.name,
        "passed": passed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "marker_stopped": marker_stopped,
        "expected_timeout": case.expected_timeout,
        "elapsed_s": round(elapsed, 3),
        "log_path": str(log_path),
        "work_dir": str(case_work_dir),
        "markers": markers_found,
        "failures": failures_found,
        "command": " ".join(cmd),
        "extra_env": resolved_extra_env,
    }
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        print(f"--- tail: {case.name} ---\n" + "\n".join(output.splitlines()[-30:]), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_ROOT / "logs" / "isaacsim-standalone-smokes-2026-05-25")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()

    python_bin = args.venv / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit(f"Python not found: {python_bin}")

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.logs_dir / "summary.json"
    requested = set(args.case or [])
    selected = [case for case in CASES if not requested or case.name in requested]
    if not selected:
        raise SystemExit("No smoke cases selected.")

    env = os.environ.copy()
    env.update(
        {
            "ACCEPT_EULA": "Y",
            "EXP_PATH": str(args.venv / "lib" / "python3.12" / "site-packages" / "isaacsim" / "apps"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": str(args.venv),
            "PATH": f"{args.venv / 'bin'}:{env.get('PATH', '')}",
        }
    )

    results = []
    for case in selected:
        print(f"\n=== {case.name} ===", flush=True)
        results.append(run_case(case, args.sim_root, python_bin, args.venv, args.logs_dir, env))

    summary = {
        "root": str(args.root),
        "sim_root": str(args.sim_root),
        "venv": str(args.venv),
        "logs_dir": str(args.logs_dir),
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
