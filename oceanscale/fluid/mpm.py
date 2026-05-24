"""Newton MPM fluid simulation via SolverImplicitMPM."""

from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def _require_value(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"Newton MPM state missing {name}")
    return value


cast(Any, wp.init)()


class NewtonMPMFluid:
    """Newton MPM fluid simulation.

    Creates fluid particles within Newton's SolverImplicitMPM.
    Supports dam break, drop splash, and custom scenarios.
    """

    def __init__(
        self,
        n_particles: int = 10000,
        domain_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
        voxel_size: float = 0.05,
        viscosity: float = 50.0,
        density: float = 1000.0,
        device: str = "cuda",
    ) -> None:
        self.n_particles = n_particles
        self.domain_size = domain_size
        self.voxel_size = voxel_size
        self.viscosity = viscosity
        self.density = density
        self.device = device

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        SolverImplicitMPM.register_custom_attributes(builder)

        self.create_dam_break(
            builder,
            domain=(0.0, 0.0, 0.0, *domain_size),
            fluid_region=(
                0.0,
                0.0,
                0.0,
                domain_size[0] * 0.5,
                domain_size[1] * 0.5,
                domain_size[2] * 0.5,
            ),
            voxel_size=voxel_size,
            n_particles=n_particles,
            density=density,
        )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.set_gravity([0.0, 0.0, -9.81])

        mpm = cast(Any, self.model).mpm
        mpm.viscosity.fill_(viscosity)
        mpm.friction.fill_(0.0)
        mpm.tensile_yield_ratio.fill_(1.0)
        mpm.young_modulus.fill_(1.0e6)
        mpm.yield_pressure.fill_(1.0e4)

        config = SolverImplicitMPM.Config()
        config.voxel_size = voxel_size
        config.tolerance = 1.0e-4
        config.max_iterations = 100
        config.strain_basis = "P0"
        config.velocity_basis = "Q1"
        config.transfer_scheme = "apic"

        self.solver = SolverImplicitMPM(self.model, config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self._step_count = 0

    def step(self, dt: float = 1.0 / 240.0) -> None:
        """Advance fluid one timestep."""
        cast(Any, self.solver).step(self.state_0, self.state_1, None, None, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._step_count += 1

    def get_particle_positions(self) -> Any:
        """Return current particle positions (n_particles, 3)."""
        return _require_value(self.state_0.particle_q, "particle_q")

    def get_particle_velocities(self) -> Any:
        """Return current particle velocities."""
        return _require_value(self.state_0.particle_qd, "particle_qd")

    @property
    def particle_count(self) -> int:
        return self.model.particle_count

    @staticmethod
    def create_dam_break(
        builder: newton.ModelBuilder,
        domain: tuple[float, ...] = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
        fluid_region: tuple[float, ...] | None = None,
        voxel_size: float = 0.05,
        n_particles: int = 10000,
        density: float = 1000.0,
    ) -> None:
        """Add dam break particles to Newton ModelBuilder."""
        if fluid_region is None:
            fluid_region = (0.0, 0.0, 0.0, domain[3] * 0.5, domain[4] * 0.5, domain[5] * 0.5)

        side = int(np.ceil(n_particles ** (1.0 / 3.0)))
        cell_size = voxel_size / 3.0
        cell_volume = cell_size**3
        mass = cell_volume * density
        radius = cell_size * 0.5

        # Position the fluid block at the base corner of the domain
        origin_x = fluid_region[0]
        origin_y = fluid_region[1]
        origin_z = fluid_region[2]

        builder.add_particle_grid(
            pos=wp.vec3(origin_x, origin_y, origin_z + 0.5),
            rot=cast(Any, wp.quat_identity)(dtype=wp.float32),
            vel=wp.vec3(0.0),
            dim_x=side,
            dim_y=side,
            dim_z=side,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )

    @staticmethod
    def create_drop_splash(
        builder: newton.ModelBuilder,
        domain: tuple[float, ...] = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
        drop_center: tuple[float, float, float] = (1.0, 1.0, 2.0),
        drop_radius: float = 0.5,
        voxel_size: float = 0.05,
        n_particles: int = 10000,
        density: float = 1000.0,
    ) -> None:
        """Add drop splash particles to Newton ModelBuilder."""
        cell_size = voxel_size / 3.0
        cell_volume = cell_size**3
        mass = cell_volume * density
        radius = cell_size * 0.5

        sphere_r = (n_particles * cell_volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
        center = np.array(drop_center, dtype=np.float32)

        rng = np.random.default_rng(42)
        points: list[FloatArray] = []
        while len(points) < n_particles:
            batch = rng.uniform(-1, 1, (n_particles * 3, 3))
            batch = batch[np.linalg.norm(batch, axis=1) <= 1.0]
            for p in batch:
                if len(points) >= n_particles:
                    break
                points.append(cast(FloatArray, center + p * sphere_r))

        points_arr = np.array(points[:n_particles], dtype=np.float32)
        points_arr += (rng.random(points_arr.shape) - 0.5) * 2.0 * radius

        builder.add_particles(
            pos=points_arr.tolist(),
            vel=np.zeros_like(points_arr).tolist(),
            mass=[mass] * len(points_arr),
            radius=[radius] * len(points_arr),
        )
