# Isaac Sim 6 / Isaac Lab 3 Official Suite Coverage

Inventory entrypoints: 2099
Smoke summary records: 198
Matched summary records: 198

## Clean Source Coverage

| Scope | Passed | Failed | Uncovered |
| --- | ---: | ---: | ---: |
| all | 96 | 19 | 1984 |
| isaacsim | 79 | 8 | 120 |
| isaaclab | 17 | 11 | 1864 |

## Clean Failures

| Suite | Path | Runs |
| --- | --- | --- |
| isaacsim | source/standalone_examples/api/isaacsim.core.experimental.api/control_frankas.py | fail: core_experimental_control_frankas_test |
| isaacsim | source/standalone_examples/api/isaacsim.core.experimental.api/visual_materials.py | fail: core_experimental_visual_materials_test |
| isaacsim | source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py | fail: policy_spot_standalone_test_cpu |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.experimental.physics/contact_sensor.py | fail: sensor_experimental_physics_contact_api |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.experimental.physics/imu_sensor.py | fail: sensor_experimental_physics_imu_api |
| isaacsim | source/standalone_examples/api/isaacsim.sensors.rtx/create_radar_basic.py | fail: sensor_rtx_radar_basic_test |
| isaacsim | source/standalone_examples/testing/isaacsim.core.api/test_articulation.py | fail: testing_core_api_articulation |
| isaacsim | source/standalone_examples/testing/isaacsim.core.api/test_time_stepping.py | fail: testing_core_api_time_stepping |
| isaaclab | scripts/demos/quadcopter.py | fail: demo_quadcopter; fail: demo_quadcopter |
| isaaclab | scripts/demos/sensors/contact_sensor.py | fail: demo_sensor_contact |
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
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/example_command_api_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/follow_example_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/follow_example_modified_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.cortex.framework/franka_examples_main.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.behavior/behaviors.py | asset_conversion_or_file_asset, gui_default, replicator_sdg, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.domain_randomization/randomization_demo.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset, replicator_sdg |
| headless_candidate | source/standalone_examples/api/isaacsim.replicator.examples/cosmos_writer_simple.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.examples/motion_blur.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset, replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/api/isaacsim.replicator.examples/simready_assets_sdg.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.examples/subscribers_and_events.py | asset_conversion_or_file_asset, gui_default, replicator_sdg |
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.replicator.experimental.domain_randomization/randomization_demo.py | asset_conversion_or_file_asset, gui_default, long_running_loop, remote_or_nucleus_asset, replicator_sdg |
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
| gui_or_patch_headless | source/standalone_examples/api/isaacsim.simulation_app/change_resolution.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset |
| bounded_expected_timeout | source/standalone_examples/api/isaacsim.simulation_app/constant_fps.py | long_running_loop |
| headless_candidate | source/standalone_examples/api/isaacsim.simulation_app/livestream.py | - |
| headless_candidate | source/standalone_examples/api/omni.kit.app/app_framework.py | asset_conversion_or_file_asset |
| bounded_expected_timeout | source/standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py | asset_conversion_or_file_asset, long_running_loop, sensor_or_camera |
| bounded_expected_timeout | source/standalone_examples/replicator/infinigen/infinigen_sdg.py | asset_conversion_or_file_asset, long_running_loop, replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/replicator/mobility_gen/replay_directory.py | replicator_sdg |
| headless_candidate | source/standalone_examples/replicator/object_based_sdg/object_based_sdg_utils.py | replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/replicator/scene_based_sdg/scene_based_sdg_utils.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera |
| headless_candidate | source/standalone_examples/testing/doc_snippets/test_snippets_async.py | asset_conversion_or_file_asset, replicator_sdg, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.core.api/test_delete_in_contact.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.core.api/test_hello_world.py | gui_default, long_running_loop, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.core.api/test_save_stage.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.core.api/test_xform_prim_view.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.cortex.framework/cortex_bringup_test.py | gui_default, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.replicator.examples/ar_capture_pipeline.py | asset_conversion_or_file_asset, gui_default, replicator_sdg, test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.robot.manipulators.examples.franka/torque_control.py | gui_default, long_running_loop, test |
| headless_candidate | source/standalone_examples/testing/isaacsim.simulation_app/test_frame_delay.py | asset_conversion_or_file_asset, replicator_sdg, sensor_or_camera, test |
| headless_candidate | source/standalone_examples/testing/isaacsim.simulation_app/test_headless_no_rendering.py | asset_conversion_or_file_asset, test |
| bounded_expected_timeout | source/standalone_examples/testing/isaacsim.simulation_app/test_multiprocess.py | long_running_loop, test |
| headless_candidate | source/standalone_examples/testing/isaacsim.simulation_app/test_ovd.py | test |
| gui_or_patch_headless | source/standalone_examples/testing/isaacsim.simulation_app/test_viewport_ready.py | asset_conversion_or_file_asset, gui_default, test |
| headless_candidate | source/standalone_examples/testing/isaacsim.test.docstring/standalone_doctest.py | test |
| headless_candidate | source/standalone_examples/testing/omni.replicator.agent/test_scripting.py | asset_conversion_or_file_asset, replicator_sdg, test |
| headless_candidate | source/standalone_examples/testing/python_sh/import_scipy.py | test |
| headless_candidate | source/standalone_examples/testing/python_sh/import_sys.py | test |
| headless_candidate | source/standalone_examples/testing/python_sh/import_torch.py | test |
| headless_candidate | source/standalone_examples/testing/python_sh/path_length.py | test |
| headless_candidate | source/standalone_examples/testing/validation/test_docstring_coverage.py | test |
| headless_candidate | source/standalone_examples/testing/validation/test_extension_count.py | test |
| gui_or_patch_headless | source/standalone_examples/tutorials/getting_started.py | gui_default |
| gui_or_patch_headless | source/standalone_examples/tutorials/getting_started_robot.py | asset_conversion_or_file_asset, gui_default, remote_or_nucleus_asset, sensor_or_camera |

## Unmatched Summary Records

| Name | Command |
| --- | --- |
