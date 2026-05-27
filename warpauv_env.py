"""
WarpAUV environment for Isaac Lab 3 / Isaac Sim 6

Ported from Isaac Lab 2.2 to Isaac Lab 3.0.

Original authors: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
ICRA 2025: "Learning to Swim: RL for 6-DOF Control of Thruster-driven AUVs"

Isaac Lab 3 migration:
- ProxyArray .torch accessor for data properties
- permanent_wrench_composer replaces set_external_force_and_torque()
- scene.rigid_objects (not scene.articulations) for RigidObject
- torch.arange replaces _ALL_INDICES (wp.array in IL3)
- String-based gym.register entry points
- obs_groups required in RslRlOnPolicyRunnerCfg
- write_root_pose_to_sim_index / write_root_velocity_to_sim_index
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from .assets.warpauv import WARPAUV_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    sample_uniform,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    random_orientation,
    quat_error_magnitude,
)
import isaaclab.utils.math as math_utils
import gymnasium as gym
import numpy as np

from isaaclab.markers import (
    CUBOID_MARKER_CFG,
    VisualizationMarkers,
    RED_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)

from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations


@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # robot
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = False

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)
    state_space: gym.spaces.Space | None = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)

    # env
    decimation = 2
    episode_length_s = 3.0
    episode_length_before_reset = None
    num_actions = 6
    num_observations = 17
    num_states = 0
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    starting_depth = 8
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000

    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.2

    # dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01]
    water_rho = 997.0
    water_beta = 0.001306
    rotor_constant = 0.1 / 100.0
    dyn_time_constant = 0.05
    volume = 0.022747843530591776
    mass = 2.2701e+01

    class domain_randomization:
        use_custom_randomization = True
        com_to_cob_offset_radius = 0.05
        volume_range = [0.019747843530591773, 0.02574784353059178]
        mass_range = [2.2701e+01, 2.2701e+01]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._debug = False

        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins
        self._step_count = 0

        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            torch.manual_seed(0)

        self.set_debug_vis(self.cfg.debug_vis)

        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.clone()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.clone()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True)

        self._init_thruster_dynamics()

        # Isaac Lab 3: _ALL_INDICES is wp.array, use torch.arange instead
        self._reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.cfg.com_to_cob_offset = torch.tensor(
                self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False
            ).reshape(1, 3).repeat(self.num_envs, 1)

        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Isaac Lab 3: RigidObject goes in rigid_objects, not articulations
        self.scene.rigid_objects["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(self._actions)
        # Isaac Lab 3: permanent_wrench_composer replaces set_external_force_and_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        # Isaac Lab 3: data properties may be ProxyArray, use .torch
        if hasattr(root_pos_w, 'torch'):
            root_pos_w = root_pos_w.torch
            root_quat_w = root_quat_w.torch
            root_lin_vel_b = root_lin_vel_b.torch
            root_ang_vel_b = root_ang_vel_b.torch

        offset_from_origin_b = quat_apply(quat_conjugate(root_quat_w), self._default_env_origins - root_pos_w)

        obs = torch.cat(
            [
                self._goal,
                offset_from_origin_b,
                root_quat_w,
                root_lin_vel_b,
                root_ang_vel_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        if hasattr(root_pos_w, 'torch'):
            root_pos_w = root_pos_w.torch
            root_quat_w = root_quat_w.torch
            root_lin_vel_b = root_lin_vel_b.torch
            root_ang_vel_b = root_ang_vel_b.torch

        offsets_from_origin = quat_apply(quat_conjugate(root_quat_w), self._default_env_origins - root_pos_w)

        return _compute_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            root_lin_vel_b,
            root_ang_vel_b,
            self.reset_terminated,
            root_pos_w,
            root_quat_w,
            self._goal,
            offsets_from_origin,
            self._completed_envs,
            self._actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._step_count += 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        if self.cfg.use_boundaries:
            root_pos_w = self._robot.data.root_pos_w
            if hasattr(root_pos_w, 'torch'):
                root_pos_w = root_pos_w.torch
            out_of_bounds = (
                (torch.abs(root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x)
                | (torch.abs(root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y)
                | (torch.abs(root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        default_state = self._robot.data.default_root_state
        if hasattr(default_state, 'torch'):
            default_state = default_state.torch

        self._default_root_state[env_ids, :] = default_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if not self.cfg.eval_mode:
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(
                len(env_ids), self.cfg.goal_spawn_radius
            )

        self._step_count = 0

        self._reset_domain(env_ids)
        self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            self._default_root_state[env_ids_to_guide, :3] = self._default_env_origins[env_ids_to_guide, :3]
            self._default_root_state[env_ids_to_guide, 3:7] = self._goal[env_ids_to_guide, 0:4]

        # Isaac Lab 3: use _index variants for per-env writes
        self._robot.write_root_pose_to_sim_index(
            root_pose=self._default_root_state[env_ids, :7], env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=self._default_root_state[env_ids, 7:], env_ids=env_ids
        )

    def _reset_goal(self, env_ids: Sequence[int]):
        self._goal[env_ids, 0:4] = random_orientation(len(env_ids), device=self.device)

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = (
                self.cfg.com_to_cob_offset[env_ids]
                + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)
            )

        if self.cfg.domain_randomization.use_custom_randomization:
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(
                vol_lower, vol_upper, self.volumes[env_ids].shape, self.device
            )

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms
        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1 / 3)
        return radii * coords

    def _compute_dynamics(self, actions):
        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motor_values = torch.clone(actions)

        # Convert PWM commands to rad/s
        motor_values[torch.abs(motor_values) < 0.08] = 0
        pos_mask = motor_values >= 0.08
        neg_mask = motor_values <= -0.08
        motor_values[pos_mask] = -139.0 * (torch.pow(motor_values[pos_mask], 2.0)) + 500 * motor_values[pos_mask] + 8.28
        motor_values[neg_mask] = 161.0 * (torch.pow(motor_values[neg_mask], 2.0)) + 517.86 * motor_values[neg_mask] - 5.72

        motor_values = self.thruster_dynamics.update(motor_values, self.episode_length_buf * self.sim.cfg.dt)
        motor_values = self.thruster_conversion.convert(motor_values)

        thruster_forces[..., 0] = 1.0
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces)
        thruster_forces = thruster_forces * motor_values.unsqueeze(-1)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        thruster_forces = torch.sum(thruster_forces, dim=-2)
        thruster_torques = torch.sum(thruster_torques, dim=-2)

        # Hydrodynamics
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_w = self._robot.data.root_ang_vel_w

        if hasattr(root_quat_w, 'torch'):
            root_quat_w = root_quat_w.torch
            root_lin_vel_w = root_lin_vel_w.torch
            root_ang_vel_w = root_ang_vel_w.torch

        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(
            root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.com_to_cob_offsets
        )

        density_forces, density_torques, viscosity_forces, viscosity_torques = (
            self.force_calculation_functions.calculate_density_and_viscosity_forces(
                root_quat_w, root_lin_vel_w, root_ang_vel_w,
                self.inertia_tensors, self.inertia_tensors_mean,
                self.cfg.water_beta, self.cfg.water_rho, self.masses,
            )
        )

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)
        else:
            for viz_name in ["goal_pos_visualizer", "goal_ang_visualizer", "goal_z_ang_visualizer",
                             "x_b_visualizer", "z_b_visualizer"]:
                if hasattr(self, viz_name):
                    getattr(self, viz_name).set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q, x, y, z, device=None):
        num_envs = q.shape[0]
        if device is None:
            device = self.device

        if isinstance(x, float):
            x = torch.zeros(num_envs, device=device) + x
        if isinstance(y, float):
            y = torch.zeros(num_envs, device=device) + y
        if isinstance(z, float):
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)

    def _debug_vis_callback(self, event):
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        if hasattr(root_pos_w, 'torch'):
            root_pos_w = root_pos_w.torch
            root_quat_w = root_quat_w.torch

        self.goal_pos_visualizer.visualize(translations=self._goal_pos_w)

        goal_quats_w = self._goal
        ang_marker_scales = torch.ones(self.num_envs, 3, device=self.device)
        self.goal_ang_visualizer.visualize(
            translations=root_pos_w, orientations=goal_quats_w, scales=ang_marker_scales
        )

        goal_z_quat = self._rotate_quat_by_euler_xyz(goal_quats_w, 0.0, -torch.pi / 2, 0.0)
        self.goal_z_ang_visualizer.visualize(
            translations=root_pos_w, orientations=goal_z_quat, scales=ang_marker_scales
        )

        x_w_marker_scales = torch.ones(self.num_envs, 3, device=self.device)
        self.x_b_visualizer.visualize(
            translations=root_pos_w, orientations=root_quat_w, scales=x_w_marker_scales
        )

        z_w_quat = self._rotate_quat_by_euler_xyz(root_quat_w, 0.0, -torch.pi / 2, 0.0)
        self.z_b_visualizer.visualize(
            translations=root_pos_w, orientations=z_w_quat, scales=x_w_marker_scales
        )


@torch.jit.script
def quat_dist(q1, q2):
    return 1 - torch.sum(q1 * q2, dim=-1) ** 2


@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    goal: torch.Tensor,
    offsets_from_origin: torch.Tensor,
    completed_envs: torch.Tensor,
    actions: torch.Tensor,
):
    rew_pos = rew_scale_pos * torch.exp(-1 * torch.norm(offsets_from_origin, dim=1) ** 2)
    rew_ang = rew_scale_ang * torch.exp(-1 * quat_error_magnitude(goal[:, :], root_quat[:, :]))
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-1 * torch.norm(ang_vel, dim=1) ** 2)
    rew_action = rew_scale_actions * torch.exp(-1 * torch.norm(actions, dim=1) ** 2)

    return rew_ang + rew_action + rew_pos + rew_ang_vel
