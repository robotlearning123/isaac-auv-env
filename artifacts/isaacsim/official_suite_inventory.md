# Isaac Sim 6 / Isaac Lab 3 Official Suite Inventory

Isaac Sim source: /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop
Isaac Lab source: /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
Total Python entrypoints inventoried: 2099

## Runnable Classes

| Class | Count |
| --- | ---: |
| asset_or_network_dependent | 160 |
| bounded_expected_timeout | 50 |
| gui_or_patch_headless | 93 |
| headless_candidate | 1340 |
| interactive_or_device_dependent | 127 |
| long_runtime | 247 |
| optional_extra_blocked_or_long | 63 |
| requires_ros2 | 19 |

## Signals

| Signal | Count |
| --- | ---: |
| asset_conversion_or_file_asset | 350 |
| benchmark | 66 |
| gui_default | 98 |
| long_running_loop | 173 |
| mimic_or_imitation | 67 |
| newton | 360 |
| remote_or_nucleus_asset | 258 |
| replicator_sdg | 89 |
| ros2 | 19 |
| sensor_or_camera | 530 |
| teleoperation | 127 |
| test | 415 |
| training_or_rl | 221 |

## Top Categories

| Category | Count |
| --- | ---: |
| other | 1381 |
| lab_source_test | 370 |
| lab_reinforcement_learning | 38 |
| isaacsim.robot.manipulators | 33 |
| lab_tool | 19 |
| standalone_benchmark | 19 |
| lab_benchmark | 15 |
| isaacsim.replicator.examples | 14 |
| isaacsim.core.api | 13 |
| lab_imitation_learning | 12 |
| standalone_replicator | 11 |
| standalone_testing/isaacsim.simulation_app | 11 |
| isaacsim.core.experimental.api | 10 |
| lab_environment | 10 |
| isaacsim.ros2.bridge | 9 |
| isaacsim.sensors.camera | 9 |
| lab_demo/sensors | 9 |
| isaacsim.sensors.rtx | 8 |
| isaacsim.cortex.framework | 6 |
| isaacsim.simulation_app | 6 |
| standalone_testing/isaacsim.core.api | 6 |
| lab_tutorial/00_sim | 5 |
| lab_tutorial/01_assets | 5 |
| lab_tutorial/03_envs | 5 |
| lab_tutorial/04_sensors | 5 |
| isaacsim.robot.experimental.manipulators | 4 |
| standalone_testing/isaacsim.ros2.bridge | 4 |
| standalone_testing/python_sh | 4 |
| isaacsim.robot.policy.examples | 3 |
| isaacsim.robot_motion.experimental.motion_generation | 3 |
| isaacsim.sensors.experimental.physics | 3 |
| isaacsim.sensors.physics | 3 |
| standalone_testing/validation | 3 |
| isaacsim.robot.wheeled_robots.examples | 2 |
| lab_tutorial/05_controllers | 2 |
| standalone_tutorial | 2 |
| isaacsim.asset.importer.mjcf | 1 |
| isaacsim.asset.importer.urdf | 1 |
| isaacsim.asset.transformer | 1 |
| isaacsim.core.cloner | 1 |

## Examples By Runnable Class

### asset_or_network_dependent
- source/standalone_examples/api/isaacsim.core.api/simulation_callbacks.py
- source/standalone_examples/api/isaacsim.core.api/time_stepping.py
- source/standalone_examples/api/isaacsim.core.experimental.api/simulation_callbacks.py
- source/standalone_examples/api/isaacsim.replicator.examples/sdg_deformables.py
- source/standalone_examples/api/isaacsim.replicator.grasping/grasping_workflow_sdg.py
- source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/tasks/follow_target.py
- source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/tasks/pick_place.py
- source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/tasks/follow_target.py
- source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/tasks/pick_place.py
- source/standalone_examples/api/isaacsim.sensors.camera/camera_stereoscopic_depth.py
- source/standalone_examples/api/isaacsim.sensors.rtx/inspect_lidar_gmo.py
- source/standalone_examples/api/isaacsim.sensors.rtx/inspect_radar_gmo.py

### bounded_expected_timeout
- source/standalone_examples/api/isaacsim.simulation_app/constant_fps.py
- source/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py
- source/standalone_examples/replicator/infinigen/infinigen_sdg.py
- source/standalone_examples/testing/isaacsim.simulation_app/test_multiprocess.py
- scripts/demos/arl_robot_1.py
- scripts/demos/bipeds.py
- scripts/demos/hands.py
- scripts/demos/procedural_terrain.py
- scripts/demos/quadcopter.py
- scripts/demos/quadrupeds.py
- scripts/demos/sensors/cameras.py
- scripts/demos/sensors/contact_sensor.py

### gui_or_patch_headless
- source/standalone_examples/api/isaacsim.core.api/add_cubes.py
- source/standalone_examples/api/isaacsim.core.api/add_frankas.py
- source/standalone_examples/api/isaacsim.core.api/cloth.py
- source/standalone_examples/api/isaacsim.core.api/control_robot.py
- source/standalone_examples/api/isaacsim.core.api/data_logging.py
- source/standalone_examples/api/isaacsim.core.api/deformable.py
- source/standalone_examples/api/isaacsim.core.api/detailed_contact_data.py
- source/standalone_examples/api/isaacsim.core.api/rigid_contact_view.py
- source/standalone_examples/api/isaacsim.core.api/simulate_robot.py
- source/standalone_examples/api/isaacsim.core.api/visual_materials.py
- source/standalone_examples/api/isaacsim.core.cloner/clone_ants.py
- source/standalone_examples/api/isaacsim.core.experimental.api/add_cubes.py

### headless_candidate
- source/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py
- source/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py
- source/standalone_examples/api/isaacsim.asset.transformer/run_asset_transformer.py
- source/standalone_examples/api/isaacsim.core.api/omnigraph_triggers.py
- source/standalone_examples/api/isaacsim.core.experimental.api/omnigraph_triggers.py
- source/standalone_examples/api/isaacsim.cortex.framework/behaviors/franka/franka_behaviors.py
- source/standalone_examples/api/isaacsim.replicator.examples/cosmos_writer_simple.py
- source/standalone_examples/api/isaacsim.replicator.examples/custom_event_and_write.py
- source/standalone_examples/api/isaacsim.replicator.examples/multi_camera.py
- source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py
- source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_02.py
- source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_03.py

### interactive_or_device_dependent
- source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/follow_target_with_ik_experimental.py
- source/standalone_examples/api/isaacsim.robot.policy.examples/anymal_standalone.py
- scripts/demos/h1_locomotion.py
- scripts/demos/haply_teleoperation.py
- scripts/demos/pick_and_place.py
- scripts/environments/teleoperation/teleop_replay_agent.py
- scripts/environments/teleoperation/teleop_se3_agent.py
- scripts/imitation_learning/isaaclab_mimic/annotate_demos.py
- scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py
- scripts/imitation_learning/isaaclab_mimic/generate_dataset.py
- scripts/imitation_learning/locomanipulation_sdg/generate_data.py
- scripts/imitation_learning/locomanipulation_sdg/gr00t/rollout_policy.py

### long_runtime
- source/standalone_examples/benchmarks/benchmark_camera.py
- source/standalone_examples/benchmarks/benchmark_core_world.py
- source/standalone_examples/benchmarks/benchmark_nucleus_kpis.py
- source/standalone_examples/benchmarks/benchmark_physx_lidar.py
- source/standalone_examples/benchmarks/benchmark_robots_evobot.py
- source/standalone_examples/benchmarks/benchmark_robots_humanoid.py
- source/standalone_examples/benchmarks/benchmark_robots_nova_carter.py
- source/standalone_examples/benchmarks/benchmark_robots_o3dyn.py
- source/standalone_examples/benchmarks/benchmark_robots_ur10.py
- source/standalone_examples/benchmarks/benchmark_rtx_lidar.py
- source/standalone_examples/benchmarks/benchmark_rtx_radar.py
- source/standalone_examples/benchmarks/benchmark_sdg.py

### optional_extra_blocked_or_long
- scripts/imitation_learning/robomimic/play.py
- scripts/imitation_learning/robomimic/robust_eval.py
- scripts/imitation_learning/robomimic/train.py
- source/isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py
- source/isaaclab/isaaclab/envs/mimic_env_cfg.py
- source/isaaclab/test/install_ci/test_isaaclabx_i_mimic.py
- source/isaaclab_mimic/isaaclab_mimic/__init__.py
- source/isaaclab_mimic/isaaclab_mimic/datagen/__init__.py
- source/isaaclab_mimic/isaaclab_mimic/datagen/data_generator.py
- source/isaaclab_mimic/isaaclab_mimic/datagen/datagen_info.py
- source/isaaclab_mimic/isaaclab_mimic/datagen/datagen_info_pool.py
- source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py

### requires_ros2
- source/standalone_examples/api/isaacsim.ros2.bridge/camera_manual.py
- source/standalone_examples/api/isaacsim.ros2.bridge/camera_noise.py
- source/standalone_examples/api/isaacsim.ros2.bridge/camera_periodic.py
- source/standalone_examples/api/isaacsim.ros2.bridge/carter_multiple_robot_navigation.py
- source/standalone_examples/api/isaacsim.ros2.bridge/carter_stereo.py
- source/standalone_examples/api/isaacsim.ros2.bridge/clock.py
- source/standalone_examples/api/isaacsim.ros2.bridge/moveit.py
- source/standalone_examples/api/isaacsim.ros2.bridge/rtx_lidar.py
- source/standalone_examples/api/isaacsim.ros2.bridge/subscriber.py
- source/standalone_examples/benchmarks/benchmark_robots_nova_carter_ros2.py
- source/standalone_examples/benchmarks/benchmark_rtx_lidar_ros2_pcl_metadata.py
- source/standalone_examples/benchmarks/benchmark_scene_loading.py
