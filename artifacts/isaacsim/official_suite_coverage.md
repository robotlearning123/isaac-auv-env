# Isaac Sim 6 / Isaac Lab 3 Official Suite Coverage

Inventory entrypoints: 2099
Smoke summary records: 283
Matched summary records: 283

## Clean Source Coverage

| Scope | Passed | Failed | Uncovered |
| --- | ---: | ---: | ---: |
| all | 144 | 32 | 1923 |
| isaacsim | 110 | 16 | 81 |
| isaaclab | 34 | 16 | 1842 |

## Clean Failures

| Suite | Path | Runs |
| --- | --- | --- |
| isaacsim | source/standalone_examples/api/isaacsim.core.experimental.api/control_frankas.py | fail: core_experimental_control_frankas_test |
| isaacsim | source/standalone_examples/api/isaacsim.core.experimental.api/visual_materials.py | fail: core_experimental_visual_materials_test |
| isaacsim | source/standalone_examples/api/isaacsim.cortex.framework/example_command_api_main.py | fail: cortex_command_api_smoke |
| isaacsim | source/standalone_examples/api/isaacsim.cortex.framework/franka_examples_main.py | fail: cortex_franka_simple_decider_smoke |
| isaacsim | source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py | fail: policy_spot_standalone_test_cpu |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.experimental.physics/contact_sensor.py | fail: sensor_experimental_physics_contact_api |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.experimental.physics/imu_sensor.py | fail: sensor_experimental_physics_imu_api |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.rtx/create_radar_basic.py | fail: sensor_rtx_radar_basic_test |
| isaacsim | source/standalone_examples/api/omni.kit.app/app_framework.py | fail: omni_kit_app_framework; fail: omni_kit_app_framework |
| isaacsim | source/standalone_examples/replicator/mobility_gen/replay_directory.py | fail: mobility_gen_replay_empty_input |
| isaacsim | source/standalone_examples/testing/isaacsim.core.api/test_articulation.py | fail: testing_core_api_articulation |
| isaacsim | source/standalone_examples/testing/isaacsim.core.api/test_time_stepping.py | fail: testing_core_api_time_stepping |
| isaacsim | source/standalone_examples/testing/isaacsim.simulation_app/test_config.py | fail: simulation_app_config |
| isaacsim | source/standalone_examples/testing/python_sh/import_scipy.py | fail: python_sh_import_scipy; fail: python_sh_import_scipy; fail: python_sh_import_scipy |
| isaacsim | source/standalone_examples/testing/python_sh/import_torch.py | fail: python_sh_import_torch; fail: python_sh_import_torch; fail: python_sh_import_torch |
| isaacsim | source/standalone_examples/testing/validation/test_docstring_coverage.py | fail: validation_docstring_coverage_json |
| isaaclab | scripts/benchmarks/benchmark_view_comparison.py | fail: benchmark_view_comparison_tiny |
| isaaclab | scripts/benchmarks/benchmark_xform_prim_view.py | fail: benchmark_xform_prim_view_tiny |
| isaaclab | scripts/demos/bin_packing.py | fail: demo_bin_packing |
| isaaclab | scripts/demos/multi_asset.py | fail: demo_multi_asset |
| isaaclab | scripts/demos/quadcopter.py | fail: demo_quadcopter; fail: demo_quadcopter |
| isaaclab | scripts/demos/sensors/contact_sensor.py | fail: demo_sensor_contact |
| isaaclab | scripts/reinforcement_learning/rsl_rl/train.py | fail: rsl_rl_train_cartpole_direct_1_iter |
| isaaclab | scripts/tutorials/00_sim/set_rendering_mode.py | fail: tutorial_00_set_rendering_mode |
| isaaclab | scripts/tutorials/01_assets/add_new_robot.py | fail: tutorial_01_add_new_robot |
| isaaclab | scripts/tutorials/03_envs/create_cube_base_env.py | fail: tutorial_03_create_cube_base_env |
| isaaclab | scripts/tutorials/03_envs/create_quadruped_base_env.py | fail: tutorial_03_create_quadruped_base_env |
| isaaclab | scripts/tutorials/04_sensors/add_sensors_on_robot.py | fail: tutorial_04_add_sensors_on_robot |
| isaaclab | scripts/tutorials/04_sensors/run_frame_transformer.py | fail: tutorial_04_run_frame_transformer |
| isaaclab | scripts/tutorials/04_sensors/run_ray_caster_camera.py | fail: tutorial_04_run_ray_caster_camera |
| isaaclab | scripts/tutorials/04_sensors/run_usd_camera.py | fail: tutorial_04_run_usd_camera |
| isaaclab | scripts/tutorials/05_controllers/run_osc.py | fail: tutorial_05_run_osc |

## Patched Successes For Clean Failures

| Path | Patch Evidence |
| --- | --- |
| scripts/demos/quadcopter.py | fail: demo_quadcopter; pass: demo_quadcopter |
| scripts/demos/sensors/contact_sensor.py | pass: demo_sensor_contact |
| scripts/tutorials/03_envs/create_quadruped_base_env.py | pass: tutorial_03_create_quadruped_base_env |
| scripts/tutorials/04_sensors/add_sensors_on_robot.py | fail: tutorial_04_add_sensors_on_robot; fail: tutorial_04_add_sensors_on_robot; fail: tutorial_04_add_sensors_on_robot; pass: tutorial_04_add_sensors_on_robot |
| scripts/tutorials/04_sensors/run_frame_transformer.py | fail: tutorial_04_run_frame_transformer; pass: tutorial_04_run_frame_transformer |
| scripts/tutorials/04_sensors/run_ray_caster_camera.py | pass: tutorial_04_run_ray_caster_camera; pass: tutorial_04_run_ray_caster_camera |
| scripts/tutorials/04_sensors/run_usd_camera.py | fail: tutorial_04_run_usd_camera; fail: tutorial_04_run_usd_camera; fail: tutorial_04_run_usd_camera; fail: tutorial_04_run_usd_camera; pass: tutorial_04_run_usd_camera |
| scripts/tutorials/05_controllers/run_osc.py | pass: tutorial_05_run_osc |

## Next Isaac Sim Candidates

| Class | Path | Signals |
| --- | --- | --- |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.core.api/deformable.py | gui_default, long_running_loop, remote_or_nucleus_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.core.experimental.api/control_robot_jax.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset |
| headless_candidate | source/standalone_examples/api/isaacsim.cortex.framework/behaviors/franka/franka_behaviors.py | - |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/demo_ur10_conveyor_main.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/follow_example_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/follow_example_modified_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.behavior/behaviors.py | asset_conversion_or_file_asset, gui_default, replicator_sdg, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.domain_randomization/randomization_demo.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset, replicator_sdg |
| headless_candidate | source/standalone_examples/api/isaacsim.replicator.examples/simready_assets_sdg.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.experimental.manipulators/franka/multiple_tasks.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.experimental.manipulators/franka/pick_place.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.experimental.manipulators/franka/stacking.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.experimental.manipulators/universal_robots/follow_target_with_ik.py | gui_default, long_running_loop |
| headless_candidate | source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/controllers/pick_place.py | - |
| headless_candidate | source/standalone_examples/api/isaacsim.robot.manipulators/cobotta_900/controllers/rmpflow.py | asset_conversion_or_file_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/franka/follow_target_with_ik.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/franka/follow_target_with_rmpflow.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/franka/multiple_tasks.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/franka/stacking.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/franka_pick_up.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/rmpflow_supported_robots/supported_robot_follow_target_example.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/bin_filling.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/follow_target_with_ik.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/follow_target_with_rmpflow.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/multiple_tasks.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/pick_place.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/pick_place2.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/universal_robots/stacking.py | gui_default, long_running_loop |
| headless_candidate | source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/controller/ik_solver.py | asset_conversion_or_file_asset |
| headless_candidate | source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/controller/pick_place.py | - |
| headless_candidate | source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/controller/rmpflow.py | asset_conversion_or_file_asset |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example.py | gui_default, long_running_loop |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example_rmpflow.py | gui_default, long_running_loop |
| bounded_expected_timeout | source/standalone_examples/replicator/infinigen/infinigen_sdg.py | asset_conversion_or_file_asset, long_running_loop, replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/replicator/object_based_sdg/object_based_sdg_utils.py | replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/replicator/scene_based_sdg/scene_based_sdg_utils.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/testing/doc_snippets/test_snippets_async.py | asset_conversion_or_file_asset, replicator_sdg, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.robot.manipulators.examples.franka/torque_control.py | gui_default, long_running_loop, test |

## Unmatched Summary Records

| Name | Command |
| --- | --- |
