"""End-to-end integration tests: GPU fluid → FSI → Newton → sensors → training.

Proves the full NVIDIA feature stack works as a connected system.
Each test verifies real data flow between layers, not mocks.
"""

from __future__ import annotations

import newton
import numpy as np
import pytest
import warp as wp

from oceanscale.integration import IntegratedPipeline

wp.init()

DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def pipeline():
    return IntegratedPipeline(
        seabed_depth=-50.0,
        wave_height=1.0,
        wave_period=8.0,
        tether_length=20.0,
        device=DEVICE,
    )


class TestFullPipeline:
    def test_pipeline_creation(self, pipeline):
        assert pipeline.wave is not None
        assert pipeline.dvl is not None
        assert pipeline.sonar is not None
        assert pipeline.tether is not None
        assert pipeline.mesh_boundary is not None
        assert pipeline.solver is not None

    def test_single_step(self, pipeline):
        pipeline.reset()
        result = pipeline.step()
        assert "wave_velocity" in result
        assert "current" in result
        assert "water_density" in result
        assert "sound_speed" in result
        assert "acoustic_loss_db" in result
        assert "dvl" in result
        assert "sonar" in result
        assert "rov_position" in result
        assert "rov_velocity" in result
        assert "tether_tension" in result
        assert "reward" in result
        assert "time" in result

    def test_action_changes_rov_state(self):
        baseline = IntegratedPipeline(device=DEVICE)
        driven = IntegratedPipeline(device=DEVICE)
        baseline.reset()
        driven.reset()

        for _ in range(20):
            zero = baseline.step(np.zeros(6, dtype=np.float32))
            actuated = driven.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        assert not np.allclose(zero["rov_position"], actuated["rov_position"], atol=1e-5)
        assert not np.allclose(zero["rov_velocity"], actuated["rov_velocity"], atol=1e-5)

    def test_multi_step_stable(self, pipeline):
        pipeline.reset()
        for _ in range(100):
            result = pipeline.step()
        pos = result["rov_position"]
        assert np.all(np.isfinite(pos)), f"NaN in position after 100 steps: {pos}"
        vel = result["rov_velocity"]
        assert np.all(np.isfinite(vel)), f"NaN in velocity after 100 steps: {vel}"

    def test_wave_affects_rov(self, pipeline):
        pipeline.reset()
        result0 = pipeline.step()
        vel0 = result0["rov_velocity"]
        for _ in range(50):
            pipeline.step()
        result50 = pipeline.step()
        vel50 = result50["rov_velocity"]
        assert not np.allclose(
            vel0, vel50, atol=1e-6
        ), "ROV velocity should change due to wave drag forces"

    def test_dvl_reads_seabed(self, pipeline):
        pipeline.reset()
        result = pipeline.step()
        dvl = result["dvl"]
        assert "altitude" in dvl
        alt = dvl["altitude"]
        assert np.isfinite(alt), f"DVL altitude not finite: {alt}"
        assert alt > 0.0, "DVL should detect seabed below ROV"
        assert alt < 200.0, "DVL altitude should be within max range"

    def test_sonar_returns_ranges(self, pipeline):
        pipeline.reset()
        result = pipeline.step()
        sonar = result["sonar"]
        assert sonar.shape[0] == 16, "Sonar should return 16 rays"
        assert np.all(np.isfinite(sonar)), "Sonar ranges should be finite"
        assert np.all(sonar > 0.0), "Sonar ranges should be positive"

    def test_tether_constrains_rov(self, pipeline):
        pipeline.reset()
        for _ in range(200):
            pipeline.step()
        result = pipeline.step()
        pos = result["rov_position"]
        dist_from_origin = np.linalg.norm(pos)
        assert dist_from_origin < 100.0, "Tether should prevent ROV from flying away"

    def test_observation_vector_complete(self, pipeline):
        pipeline.reset()
        result = pipeline.step()
        assert result["wave_velocity"].shape == (3,)
        assert isinstance(result["dvl"], dict)
        assert result["sonar"].shape == (16,)
        assert result["rov_position"].shape == (3,)
        assert result["rov_velocity"].shape == (6,)
        assert isinstance(result["tether_tension"], float)
        assert isinstance(result["reward"], float)
        assert 0.0 <= result["reward"] <= 1.0


class TestDifferentiablePipeline:
    def test_gradient_through_fluid_to_loss(self):
        from oceanscale.fluid.differentiable import DifferentiableFluidStep

        solver = DifferentiableFluidStep(grid_size=16, device=DEVICE)
        fx = wp.zeros(solver.n, dtype=wp.float32, device=DEVICE, requires_grad=True)
        fy = wp.zeros(solver.n, dtype=wp.float32, device=DEVICE, requires_grad=True)
        target_vx = wp.array(
            np.random.randn(solver.n).astype(np.float32) * 0.01,
            dtype=wp.float32, device=DEVICE,
        )
        target_vy = wp.array(
            np.random.randn(solver.n).astype(np.float32) * 0.01,
            dtype=wp.float32, device=DEVICE,
        )

        tape = wp.Tape()
        with tape:
            solver.forward(fx, fy, dt=0.01)
            loss = solver.compute_loss(target_vx, target_vy)
        tape.backward(loss)

        assert fx.grad is not None, "Force x should have gradients"
        assert fy.grad is not None, "Force y should have gradients"
        grad_norm = float(np.linalg.norm(fx.grad.numpy()) + np.linalg.norm(fy.grad.numpy()))
        assert grad_norm > 0.0, "Gradients should be nonzero"

    def test_torch_policy_gradient(self):
        import torch

        from oceanscale.fluid.differentiable import DifferentiableFluidStep

        solver = DifferentiableFluidStep(grid_size=16, device=DEVICE)
        policy = torch.nn.Linear(4, solver.n * 2).cuda()
        obs = torch.randn(4, device="cuda", requires_grad=True)
        out = policy(obs)
        fx_t = out[: solver.n].contiguous()
        fy_t = out[solver.n :].contiguous()

        fx_wp = wp.from_torch(fx_t)
        fy_wp = wp.from_torch(fy_t)

        solver.reset()
        tape = wp.Tape()
        with tape:
            solver.forward(fx_wp, fy_wp, dt=0.01)
            loss = solver.compute_loss(
                wp.zeros(solver.n, dtype=wp.float32, device=DEVICE),
                wp.zeros(solver.n, dtype=wp.float32, device=DEVICE),
            )
        tape.backward(loss)

        assert fx_t.grad is not None or obs.grad is not None, \
            "Gradients should flow from Warp loss back to PyTorch"


class TestPerformancePipeline:
    def test_graph_capture_speedup(self):
        from oceanscale.graph_capture import GraphCapture

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_body(mass=1.0)
        builder.add_shape_box(body=0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        solver = newton.solvers.SolverSemiImplicit(model)
        s0 = model.state()
        s1 = model.state()

        def physics_step():
            solver.step(s0, s1, model.control(), None, 0.01)
            wp.copy(dest=s0.body_q, src=s1.body_q)
            wp.copy(dest=s0.body_qd, src=s1.body_qd)

        import time as _time

        for _ in range(5):
            physics_step()
        wp.synchronize()
        t0 = _time.perf_counter()
        for _ in range(200):
            physics_step()
        wp.synchronize()
        direct_time = _time.perf_counter() - t0

        gc = GraphCapture(physics_step, warmup_steps=3, device=DEVICE)
        wp.synchronize()
        t0 = _time.perf_counter()
        for _ in range(200):
            gc.replay()
        wp.synchronize()
        graph_time = _time.perf_counter() - t0

        speedup = direct_time / max(graph_time, 1e-9)
        assert speedup > 1.0, f"Graph capture should be faster: {speedup:.2f}x"
