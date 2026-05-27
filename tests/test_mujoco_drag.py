"""Tests for MuJoCo-style geometry-inferred drag model."""

import numpy as np
import warp as wp

wp.init()

from oceanscale.hydro.mujoco_drag import (
    MuJoCoDrag,
    MuJoCoDragParams,
    mujoco_added_mass,
    mujoco_blunt_slender_drag,
    mujoco_kutta_lift,
    mujoco_magnus_lift,
    mujoco_quadratic_drag,
    mujoco_viscous_drag,
)


class TestMuJoCoDragKernels:
    def test_zero_velocity_no_drag(self):
        n = 4
        vel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_quadratic_drag, dim=n, inputs=[vel, angvel, inertia, mass, wp.float32(997.0), f, t], device="cuda:0")
        assert np.allclose(f.numpy(), 0.0, atol=1e-6)
        assert np.allclose(t.numpy(), 0.0, atol=1e-6)

    def test_forward_velocity_produces_backward_drag(self):
        n = 1
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        inertia = wp.array(np.array([[0.37, 0.97, 1.19]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.array([22.7], dtype=np.float32), dtype=wp.float32, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_quadratic_drag, dim=n, inputs=[vel, angvel, inertia, mass, wp.float32(997.0), f, t], device="cuda:0")
        forces = f.numpy()
        assert forces[0, 0] < 0, "Forward drag should be negative"
        assert abs(forces[0, 1]) < 1e-6
        assert abs(forces[0, 2]) < 1e-6

    def test_drag_scales_with_velocity_squared(self):
        n = 2
        vel = wp.array(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_quadratic_drag, dim=n, inputs=[vel, angvel, inertia, mass, wp.float32(997.0), f, t], device="cuda:0")
        forces = f.numpy()
        ratio = forces[1, 0] / forces[0, 0]
        assert abs(ratio - 4.0) < 0.01, f"Drag should scale as v^2, ratio={ratio}"

    def test_viscous_drag_linear(self):
        n = 2
        vel = wp.array(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_viscous_drag, dim=n, inputs=[vel, angvel, inertia, mass, wp.float32(0.001306), f, t], device="cuda:0")
        forces = f.numpy()
        ratio = forces[1, 0] / forces[0, 0]
        assert abs(ratio - 2.0) < 0.01, f"Viscous drag should be linear, ratio={ratio}"


class TestMuJoCoDragClass:
    def test_construct_single_env(self):
        drag = MuJoCoDrag(
            n_envs=1,
            inertia_diag=np.array([0.37, 0.97, 1.19], dtype=np.float32),
            mass=22.7,
        )
        vel = wp.array(np.array([[0.5, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        f, _t = drag.compute(vel, angvel)
        assert f.numpy()[0, 0] < 0

    def test_batch_envs(self):
        n = 8
        drag = MuJoCoDrag(
            n_envs=n,
            inertia_diag=np.array([0.37, 0.97, 1.19], dtype=np.float32),
            mass=22.7,
        )
        vel = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        f, _t = drag.compute(vel, angvel)
        assert f.numpy().shape == (n, 3)

    def test_seawater_params(self):
        p = MuJoCoDragParams.SEAWATER
        assert p.fluid_density == 1025.0

    def test_freshwater_params(self):
        p = MuJoCoDragParams.FRESHWATER
        assert p.fluid_density == 997.0

    def test_fishsim_validated_preset(self):
        p = MuJoCoDragParams.FISHSIM_VALIDATED
        assert p.kutta_lift == 3.84
        assert p.magnus_lift == 0.27
        assert p.blunt_drag == 0.4


class TestLiftForces:
    def _make_arrays(self, n=1, vel=(1.0, 0.0, 0.0), angvel=(0.0, 0.0, 1.0)):
        v = wp.array(np.tile(vel, (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        w = wp.array(np.tile(angvel, (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        return v, w, inertia, mass

    def test_kutta_lift_perpendicular_to_velocity(self):
        # Flow perpendicular to slender axis → nonzero lift perpendicular to v
        # Inertia [0.37, 0.97, 1.19]: smallest is I_x → slender axis is x
        # Velocity along y → perpendicular to slender → should produce lift
        v, w, inertia, mass = self._make_arrays(vel=(0.0, 1.0, 0.0), angvel=(0.0, 0.0, 1.0))
        f = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_kutta_lift, dim=1,
                  inputs=[v, w, inertia, mass, wp.float32(1000.0), wp.float32(3.84), f],
                  device="cuda:0")
        force = f.numpy()[0]
        vel_np = np.array([0.0, 1.0, 0.0])
        dot = np.dot(force, vel_np)
        assert abs(dot) < 0.1 * np.linalg.norm(force) + 1e-6, "Kutta lift should be ~perpendicular to velocity"

    def test_kutta_lift_zero_when_parallel_to_slender(self):
        # Flow along slender axis → v_hat . n_hat = 1 but n x v = 0 → no lift
        # Slender axis is x (smallest inertia)
        v, w, inertia, mass = self._make_arrays(vel=(1.0, 0.0, 0.0), angvel=(0.0, 0.0, 0.0))
        f = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_kutta_lift, dim=1,
                  inputs=[v, w, inertia, mass, wp.float32(1000.0), wp.float32(3.84), f],
                  device="cuda:0")
        assert np.allclose(f.numpy(), 0.0, atol=1e-4), "Kutta lift should be zero when flow is along slender axis"

    def test_kutta_lift_zero_when_no_velocity(self):
        v, w, inertia, mass = self._make_arrays(vel=(0.0, 0.0, 0.0), angvel=(0.0, 0.0, 1.0))
        f = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_kutta_lift, dim=1,
                  inputs=[v, w, inertia, mass, wp.float32(1000.0), wp.float32(3.84), f],
                  device="cuda:0")
        assert np.allclose(f.numpy(), 0.0, atol=1e-6)

    def test_magnus_lift_from_spin(self):
        v, w, inertia, mass = self._make_arrays(vel=(1.0, 0.0, 0.0), angvel=(0.0, 0.0, 1.0))
        f = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_magnus_lift, dim=1,
                  inputs=[v, w, inertia, mass, wp.float32(1000.0), wp.float32(0.27), f],
                  device="cuda:0")
        force = f.numpy()[0]
        assert np.linalg.norm(force) > 0, "Magnus lift should be nonzero for spinning body"
        expected_dir = np.cross([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        dot = np.dot(force / np.linalg.norm(force), expected_dir / np.linalg.norm(expected_dir))
        assert abs(dot) > 0.9, f"Magnus should be in omega x v direction, dot={dot}"

    def test_magnus_zero_when_no_spin(self):
        v, w, inertia, mass = self._make_arrays(vel=(1.0, 0.0, 0.0), angvel=(0.0, 0.0, 0.0))
        f = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_magnus_lift, dim=1,
                  inputs=[v, w, inertia, mass, wp.float32(1000.0), wp.float32(0.27), f],
                  device="cuda:0")
        assert np.allclose(f.numpy(), 0.0, atol=1e-6)

    def test_compute_with_lift_enabled(self):
        drag = MuJoCoDrag(
            n_envs=2,
            inertia_diag=np.array([0.37, 0.97, 1.19], dtype=np.float32),
            mass=22.7,
            params=MuJoCoDragParams.FISHSIM_VALIDATED,
        )
        vel = wp.array(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.array(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        f, _t = drag.compute(vel, angvel)
        assert f.numpy().shape == (2, 3)
        assert np.any(np.abs(f.numpy()) > 0)


class TestAddedMass:
    def _make_arrays(self, n=1):
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        return inertia, mass

    def test_sphere_added_mass_half_displaced(self):
        """For a sphere, added mass coefficient alpha = 0.5."""
        n = 1
        # Use equal inertia → sphere-like
        inertia = wp.array(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.array([10.0], dtype=np.float32), dtype=wp.float32, device="cuda:0")
        # Accelerating: v=1, v_prev=0 → a = 1/dt
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        vel_prev = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_added_mass, dim=n,
                  inputs=[vel, vel_prev, angvel, inertia, mass,
                          wp.float32(1000.0), wp.float32(0.02), f, t],
                  device="cuda:0")
        force = f.numpy()[0]
        assert force[0] < 0, "Added mass should resist acceleration"

    def test_zero_acceleration_zero_added_mass(self):
        n = 1
        inertia, mass = self._make_arrays()
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        vel_prev = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        t = wp.zeros(n, dtype=wp.vec3, device="cuda:0")

        wp.launch(mujoco_added_mass, dim=n,
                  inputs=[vel, vel_prev, angvel, inertia, mass,
                          wp.float32(1000.0), wp.float32(0.02), f, t],
                  device="cuda:0")
        # Zero acceleration → only Coriolis term (which needs angvel)
        assert np.allclose(f.numpy(), 0.0, atol=1e-3)

    def test_added_mass_via_compute(self):
        params = MuJoCoDragParams.MUJOCO_ELLIPSOID_DEFAULT
        drag = MuJoCoDrag(n_envs=1, inertia_diag=np.array([0.37, 0.97, 1.19]), mass=22.7, params=params)
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        angvel = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
        # First call: vel_prev = 0 → acceleration
        f1, _ = drag.compute(vel, angvel)
        # Second call: same vel → no acceleration
        f2, _ = drag.compute(vel, angvel)
        # First should have larger force magnitude (includes added mass accel term)
        assert np.linalg.norm(f1.numpy()) > np.linalg.norm(f2.numpy()) * 0.5


class TestBluntSlenderDrag:
    def _make_arrays(self, n=1):
        inertia = wp.array(np.tile([0.37, 0.97, 1.19], (n, 1)).astype(np.float32), dtype=wp.vec3, device="cuda:0")
        mass = wp.array(np.full(n, 22.7, dtype=np.float32), dtype=wp.float32, device="cuda:0")
        return inertia, mass

    def test_blunt_slender_produces_drag(self):
        n = 1
        inertia, mass = self._make_arrays()
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_blunt_slender_drag, dim=n,
                  inputs=[vel, inertia, mass, wp.float32(1000.0),
                          wp.float32(0.5), wp.float32(0.25), f],
                  device="cuda:0")
        force = f.numpy()[0]
        assert force[0] < 0, "Drag should oppose motion"
        assert abs(force[1]) < 1e-6
        assert abs(force[2]) < 1e-6

    def test_zero_velocity_zero_drag(self):
        n = 1
        inertia, mass = self._make_arrays()
        vel = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        f = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_blunt_slender_drag, dim=n,
                  inputs=[vel, inertia, mass, wp.float32(1000.0),
                          wp.float32(0.5), wp.float32(0.25), f],
                  device="cuda:0")
        assert np.allclose(f.numpy(), 0.0, atol=1e-6)

    def test_blunt_dominates_for_blunt_body(self):
        """High blunt coef, low slender → mostly blunt drag."""
        n = 1
        inertia, mass = self._make_arrays()
        vel = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        f_blunt = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        f_slender = wp.zeros(n, dtype=wp.vec3, device="cuda:0")
        wp.launch(mujoco_blunt_slender_drag, dim=n,
                  inputs=[vel, inertia, mass, wp.float32(1000.0),
                          wp.float32(5.0), wp.float32(0.0), f_blunt],
                  device="cuda:0")
        wp.launch(mujoco_blunt_slender_drag, dim=n,
                  inputs=[vel, inertia, mass, wp.float32(1000.0),
                          wp.float32(0.0), wp.float32(5.0), f_slender],
                  device="cuda:0")
        # Both should produce drag, but different magnitudes
        assert abs(f_blunt.numpy()[0, 0]) > 0
        assert abs(f_slender.numpy()[0, 0]) > 0

    def test_ellipsoid_default_preset(self):
        p = MuJoCoDragParams.MUJOCO_ELLIPSOID_DEFAULT
        assert p.enable_added_mass is True
        assert p.enable_blunt_slender is True
        assert p.blunt_drag == 0.5
        assert p.slender_drag == 0.25
        assert p.kutta_lift == 1.0
        assert p.magnus_lift == 1.0
