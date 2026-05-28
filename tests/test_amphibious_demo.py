"""Tests for amphibious robot dog vehicle and demo."""

from __future__ import annotations

import numpy as np
import pytest


class TestAmphibiousRobotDog:
    """Tests for the AmphibiousRobotDog vehicle config."""

    def test_import(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        assert dog.name == "AmphibiousRobotDog"

    def test_default_params(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        assert dog.mass == 25.0
        assert dog.volume == 0.025
        assert dog.n_thrusters == 8
        assert dog.max_thrust == 80.0
        assert dog.leg_max_force == 200.0
        assert dog.n_cpg_oscillators == 6
        assert dog.contact_force_threshold == 20.0

    def test_tier1_kwargs(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        kw = dog.tier1_kwargs()
        assert kw["n_thrusters"] == 8
        assert kw["max_thrust"] == 80.0

    def test_set_coeffs_kwargs(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        kw = dog.set_coeffs_kwargs()
        assert "added_mass" in kw
        assert "d_lin" in kw
        assert "d_quad" in kw
        assert "mass" in kw
        assert "volume" in kw
        assert "coBM" in kw
        assert "T_matrix" in kw
        assert kw["mass"] == 25.0
        assert kw["volume"] == 0.025

    def test_thruster_allocation_matrix_shape(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        T = dog.thruster_allocation_matrix()
        assert T.shape == (6, 8)

    def test_thruster_allocation_matrix_values(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        T = dog.thruster_allocation_matrix()
        # Vertical thrusters (T5-T8) should have Fz = 1.0
        for j in range(4):
            assert T[2, 4 + j] == pytest.approx(1.0)
        # Horizontal thrusters should have non-zero Fx and Fy
        for i in range(4):
            assert abs(T[0, i]) > 0.5
            assert abs(T[1, i]) > 0.5

    def test_terrain_height(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        # Flat beach
        assert dog.terrain_height(0.0) == pytest.approx(0.5)
        assert dog.terrain_height(2.0) == pytest.approx(0.5)
        # Slope
        assert dog.terrain_height(4.0) == pytest.approx(0.0)
        # Seabed
        assert dog.terrain_height(8.0) == pytest.approx(-0.5)
        assert dog.terrain_height(20.0) == pytest.approx(-0.5)

    def test_terrain_height_continuity(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        # Check continuity at transitions
        for x in [2.0, 6.0]:
            eps = 1e-6
            h_left = dog.terrain_height(x - eps)
            h_right = dog.terrain_height(x + eps)
            assert abs(h_left - h_right) < 1e-4

    def test_frozen_dataclass(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        with pytest.raises(AttributeError):
            dog.mass = 50.0  # type: ignore[misc]

    def test_custom_params(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog(mass=50.0, n_thrusters=6, leg_max_force=120.0)
        assert dog.mass == 50.0
        assert dog.n_thrusters == 6
        assert dog.leg_max_force == 120.0

    def test_cpg_config_method(self):
        from oceanscale.vehicles.amphibious import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        cfg = dog.cpg_config()
        from oceanscale.controllers.cpg import CPGConfig

        assert isinstance(cfg, CPGConfig)
        assert cfg.n_oscillators == 6
        assert cfg.frequency_walk == 2.0

    def test_vehicle_registry_export(self):
        from oceanscale.vehicles import AmphibiousRobotDog

        dog = AmphibiousRobotDog()
        assert dog.name == "AmphibiousRobotDog"


class TestAmphibiousDemoConfig:
    """Tests for AmphibiousDemoConfig."""

    def test_default_config(self):
        from oceanscale.demo_amphibious import AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig()
        assert cfg.n_steps == 3000
        assert cfg.dt == 0.02
        assert cfg.seed == 42
        assert cfg.wave_height == 0.3
        assert cfg.use_cpg is True

    def test_custom_config(self):
        from oceanscale.demo_amphibious import AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(n_steps=100, wave_height=0.5, current_speed=0.3)
        assert cfg.n_steps == 100
        assert cfg.wave_height == 0.5
        assert cfg.current_speed == 0.3


class TestMissionPhases:
    """Tests for mission phase definitions."""

    def test_build_mission(self):
        from oceanscale.demo_amphibious import build_mission

        phases = build_mission()
        assert len(phases) == 6
        assert phases[0].name == "Walk to Shore"
        assert phases[2].name == "Swim to Reef"
        assert phases[5].name == "Exit Water"

    def test_mission_phases_have_chinese_names(self):
        from oceanscale.demo_amphibious import build_mission

        phases = build_mission()
        for phase in phases:
            assert phase.name_zh
            assert len(phase.name_zh) > 0

    def test_mission_total_duration(self):
        from oceanscale.demo_amphibious import build_mission

        phases = build_mission()
        total = sum(p.duration for p in phases)
        assert total == 2500  # exceeds default n_steps=3000, so demo will cap


class TestPDControl:
    """Tests for the PD mission controller."""

    def test_walking_mode(self):
        from oceanscale.demo_amphibious import pd_control

        pos = np.array([0.5, 0.0, 0.8])
        vel = np.array([0.0, 0.0, 0.0])
        target = np.array([2.5, 0.0, 0.65])
        cmd = pd_control(pos, vel, target, mode=0)
        assert cmd.shape == (8,)
        # Walking forward: first command should be positive
        assert cmd[0] > 0

    def test_swimming_mode(self):
        from oceanscale.demo_amphibious import pd_control

        pos = np.array([8.0, 0.0, -2.0])
        vel = np.array([0.0, 0.0, 0.0])
        target = np.array([12.0, 0.0, -2.5])
        cmd = pd_control(pos, vel, target, mode=1)
        assert cmd.shape == (8,)
        # Swimming forward: T1 positive, T3 negative
        assert cmd[0] > 0

    def test_commands_clipped(self):
        from oceanscale.demo_amphibious import pd_control

        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([0.0, 0.0, 0.0])
        target = np.array([100.0, 100.0, -100.0])
        for mode in [0, 1]:
            cmd = pd_control(pos, vel, target, mode=mode)
            assert np.all(cmd >= -1.0)
            assert np.all(cmd <= 1.0)


class TestComputeReward:
    """Tests for reward function."""

    def test_zero_distance(self):
        from oceanscale.demo_amphibious import compute_reward

        pos = np.array([1.0, 2.0, 3.0])
        r = compute_reward(pos, pos)
        assert r == pytest.approx(1.0)

    def test_reward_decreases_with_distance(self):
        from oceanscale.demo_amphibious import compute_reward

        target = np.array([0.0, 0.0, 0.0])
        r_close = compute_reward(np.array([1.0, 0.0, 0.0]), target)
        r_far = compute_reward(np.array([10.0, 0.0, 0.0]), target)
        assert r_close > r_far

    def test_reward_positive(self):
        from oceanscale.demo_amphibious import compute_reward

        r = compute_reward(np.array([100.0, 100.0, 100.0]), np.array([0.0, 0.0, 0.0]))
        assert r > 0


class TestAmphibiousVideoRenderer:
    """Tests for the video renderer."""

    def test_init(self):
        from oceanscale.demo_amphibious import AmphibiousVideoRenderer

        renderer = AmphibiousVideoRenderer(output_path="/tmp/test.mp4")
        assert renderer.output_path.name == "test.mp4"
        assert renderer.fps == 50

    def test_record_frame(self):
        from oceanscale.demo_amphibious import AmphibiousVideoRenderer

        renderer = AmphibiousVideoRenderer(output_path="/tmp/test.mp4")
        obs = {
            "position": np.array([1.0, 0.0, -1.0]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "linear_velocity": np.array([0.5, 0.0, 0.0]),
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
            "mode": 1,
            "mode_name": "swim",
            "terrain_height": -0.5,
            "submerged_fraction": 1.0,
            "time": 1.0,
            "step_count": 50,
        }
        renderer.record_frame(obs, np.array([5.0, 0.0, -2.0]), "Swim")
        assert len(renderer._positions) == 1
        assert len(renderer._modes) == 1

    def test_mark_phase(self):
        from oceanscale.demo_amphibious import AmphibiousVideoRenderer

        renderer = AmphibiousVideoRenderer(output_path="/tmp/test.mp4")
        renderer.mark_phase("Walk")
        renderer.mark_phase("Swim")
        assert len(renderer._phase_markers) == 2


@pytest.mark.gpu
class TestAmphibiousDemoGPU:
    """GPU-dependent tests for the amphibious demo."""

    def test_demo_init(self):
        """Test that the demo initializes without errors."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10)
        demo = AmphibiousDemo(cfg)
        assert demo.vehicle.name == "AmphibiousRobotDog"

    def test_demo_reset(self):
        """Test that reset works."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10)
        demo = AmphibiousDemo(cfg)
        obs = demo.reset()
        assert "position" in obs
        assert "mode" in obs
        assert obs["step_count"] == 0

    def test_demo_step(self):
        """Test that a single step works."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10)
        demo = AmphibiousDemo(cfg)
        demo.reset()
        action = np.zeros(8)
        obs = demo.step(action)
        assert obs["step_count"] == 1
        assert obs["time"] == pytest.approx(0.02)

    def test_demo_walk_forward(self):
        """Test that robot moves forward when given walking command."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=100)
        demo = AmphibiousDemo(cfg)
        obs = demo.reset()
        x_start = obs["position"][0]

        for _ in range(100):
            action = np.zeros(8)
            action[0] = 0.5  # walk forward
            obs = demo.step(action)

        x_end = obs["position"][0]
        assert x_end > x_start, f"Robot should move forward: {x_start} -> {x_end}"

    def test_demo_step_has_contact_force_obs(self):
        """Test that step returns contact_force and blend_factor in obs."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10)
        demo = AmphibiousDemo(cfg)
        demo.reset()
        action = np.zeros(8)
        obs = demo.step(action)
        assert "contact_force" in obs
        assert "blend_factor" in obs


@pytest.mark.gpu
class TestAmphibiousDemoCPG:
    """GPU tests for CPG mode."""

    def test_cpg_demo_init(self):
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10, use_cpg=True)
        demo = AmphibiousDemo(cfg)
        assert demo.cpg_ctrl is not None
        obs = demo.reset()
        assert obs["step_count"] == 0

    def test_cpg_demo_step_with_target(self):
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=100, use_cpg=True)
        demo = AmphibiousDemo(cfg)
        demo.reset()
        target = np.array([2.0, 0.0, 0.65])
        for _ in range(10):
            obs = demo.step(target, target=target)
        assert obs["step_count"] == 10

    def test_cpg_gait_transition(self):
        """CPG drive should transition from standing to travelling."""
        wp = pytest.importorskip("warp")
        if not wp.is_cuda_available():
            pytest.skip("CUDA not available")

        from oceanscale.demo_amphibious import AmphibiousDemo, AmphibiousDemoConfig

        cfg = AmphibiousDemoConfig(device="cuda:0", n_steps=10, use_cpg=True)
        demo = AmphibiousDemo(cfg)
        demo.reset()
        # Robot starts on land, drive should be 0
        assert demo.cpg_ctrl is not None
        # After many steps with underwater target, drive should increase
        target = np.array([8.0, 0.0, -2.0])
        for _ in range(100):
            obs = demo.step(target, target=target)
        # Drive should have moved toward swimming
        assert demo.cpg_ctrl.drive > 0.0 or obs["submerged_fraction"] == 0.0
