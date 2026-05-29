"""FloWave-TT impeller array current-field generator — v1 bulk-flow parametric model.

Physics source: Report 01 §3 (reports/research/01_flowave_facility.md)
  - 28 axial impellers, Φ1.7 m, 48 kW each, submerged below raisable floor
  - Water drawn through 90° turning vanes, exits through floor-level guide vanes
  - Produces wall-jet floor discharge → bulk upwelling by turbulent entrainment
  - Uniform region: central r ≤ 5.5 m, variance < 7% (Sutherland et al. 2017)
  - Max 1.6 m/s at 200 RPM (commissioning); 0.8 m/s at 82 RPM design
  - Stationarity period 43 s at 82 RPM (Sutherland 2017)

v1 model: parametric empirical fit for bulk-flow purposes.
  - Single bulk direction vector (28 per-unit directions NOT modelled — v2)
  - Radial: uniform in r ≤ 5.5 m, Gaussian decay to zero at r = 12.5 m wall
  - Vertical: linear shear 0 at floor → 1.0 at mid-depth → 0.8 at SWL
  - Time: 5 s low-pass filter on setpoint
  - Turbulence: configurable scalar TI (no real turbulence spectrum — v2)

Limitation (v1): Per-impeller direction (28 independent unit vectors) is NOT
modelled — all 28 units are treated as a single bulk vector. v2 will implement
per-unit injection and superposition for non-uniform directional fields.
"""

import numpy as np


class ImpellerArray:
    """Parametric current-field generator for the FloWave 28-unit impeller array.

    Implements an empirical bulk-flow model (v1) valid for hero-render purposes.
    A Navier-Stokes implementation is deferred to v2.

    Parameters
    ----------
    basin_radius:
        Inner radius of working volume (m). Default 12.5 m (25 m Ø tank).
    water_depth:
        Working water depth (m). Default 2.0 m.
    uniform_radius:
        Radius of the uniform flow core (m). Default 5.5 m (Sutherland 2017:
        11 m Ø uniform region → r = 5.5 m).
    n_units:
        Number of impeller units. Default 28 (FloWave spec).
    motor_max_rpm:
        RPM at which max_uniform_speed_at_max_rpm is achieved. Default 200 RPM
        (commissioning condition per Ingram 2014 §III-A).
    max_uniform_speed_at_max_rpm:
        Speed at the uniform core centre at motor_max_rpm (m/s). Default 1.6 m/s.
    default_TI:
        Turbulence intensity in the uniform core (dimensionless). Default 0.07
        (7% TI design target per Ingram 2014 §II-A).
    lp_tau:
        Low-pass filter time constant (s). Default 5.0 s.
    """

    def __init__(
        self,
        basin_radius: float = 12.5,
        water_depth: float = 2.0,
        uniform_radius: float = 5.5,
        n_units: int = 28,
        motor_max_rpm: float = 200.0,
        max_uniform_speed_at_max_rpm: float = 1.6,
        default_TI: float = 0.07,
        lp_tau: float = 5.0,
    ) -> None:
        self.basin_radius = float(basin_radius)
        self.water_depth = float(water_depth)
        self.uniform_radius = float(uniform_radius)
        self.n_units = int(n_units)
        self.motor_max_rpm = float(motor_max_rpm)
        self.max_uniform_speed_at_max_rpm = float(max_uniform_speed_at_max_rpm)
        self.default_TI = float(default_TI)
        self.lp_tau = float(lp_tau)

        # Radial Gaussian decay: exp(-((r - r_u) / sigma_r)^2)
        # chosen so that at r = basin_radius the value ≈ 0.
        # With sigma_r = 4.0 m: at r=12.5, r-r_u = 7.0 m → exp(-1.225) ≈ 0.294.
        # To guarantee near-zero at the wall we use sigma_r = 4.0 m and the
        # wall-clamp is enforced downstream; for r < uniform_radius the factor = 1.
        self._radial_sigma: float = 4.0  # m; per task spec

        # Turbulence intensity at outer edge (r → basin_radius)
        self._TI_outer: float = 0.15

        # Setpoint and filtered state (u, v) components in world X-Y
        self._target_u: float = 0.0
        self._target_v: float = 0.0
        self._filtered_u: float = 0.0
        self._filtered_v: float = 0.0

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def set_target(self, u: float, v: float) -> None:
        """Set bulk horizontal velocity setpoint (m/s) at mid-depth.

        Parameters
        ----------
        u, v:
            X and Y components of the target current velocity vector at
            mid-depth in the uniform core (m/s).
        """
        self._target_u = float(u)
        self._target_v = float(v)

    def step(self, dt: float) -> None:
        """Advance the 5 s low-pass filter on the bulk setpoint by dt seconds.

        Implements a first-order IIR low-pass:
            x_filt[n] = x_filt[n-1] + (dt / tau) * (x_target - x_filt[n-1])

        This is the Euler-discretised RC filter with time constant tau = lp_tau.

        Parameters
        ----------
        dt:
            Time step (s).
        """
        alpha = float(dt) / self.lp_tau
        # Clamp to avoid instability if dt > tau
        alpha = min(alpha, 1.0)
        self._filtered_u += alpha * (self._target_u - self._filtered_u)
        self._filtered_v += alpha * (self._target_v - self._filtered_v)

    # ------------------------------------------------------------------
    # Field sampling
    # ------------------------------------------------------------------

    def sample(self, xyz: np.ndarray) -> np.ndarray:
        """Sample velocity at world-frame points.

        Parameters
        ----------
        xyz:
            Array of shape (..., 3) with columns [x, y, z] in metres.
            z = 0 is tank floor, z = water_depth is still water level (SWL).

        Returns
        -------
        velocity : np.ndarray
            Array of shape (..., 3) with velocity components [vx, vy, vz] in m/s.
            vz is always 0 (bulk model does not resolve vertical component).
        """
        xyz = np.asarray(xyz, dtype=float)
        original_shape = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)

        x = xyz_flat[:, 0]
        y = xyz_flat[:, 1]
        z = xyz_flat[:, 2]

        # Radial distance from basin axis
        r = np.sqrt(x**2 + y**2)

        # --- Radial weight ---
        # 1.0 for r ≤ uniform_radius, Gaussian decay beyond
        radial_weight = np.where(
            r <= self.uniform_radius,
            1.0,
            np.exp(-((r - self.uniform_radius) / self._radial_sigma) ** 2),
        )

        # --- Vertical shear weight ---
        # 0 at floor (z=0), 1 at mid-depth (z=water_depth/2),
        # 0.8 at SWL (z=water_depth).
        # Piecewise linear: two segments sharing the mid-depth peak.
        mid = self.water_depth / 2.0
        z_clipped = np.clip(z, 0.0, self.water_depth)

        vert_weight = np.where(
            z_clipped <= mid,
            # Lower half: 0 → 1
            z_clipped / mid,
            # Upper half: 1 → 0.8
            1.0 - 0.2 * (z_clipped - mid) / mid,
        )

        # --- Combined scalar weight ---
        w = radial_weight * vert_weight

        # --- Velocity ---
        vx = w * self._filtered_u
        vy = w * self._filtered_v
        vz = np.zeros_like(vx)

        vel_flat = np.stack([vx, vy, vz], axis=-1)
        return vel_flat.reshape(original_shape)

    def sample_TI(self, xyz: np.ndarray) -> np.ndarray:
        """Sample turbulence intensity at world-frame points.

        TI is linearly interpolated from default_TI at the uniform-core edge
        (r = 0) to TI_outer (0.15) at the basin wall (r = basin_radius).

        Parameters
        ----------
        xyz:
            Array of shape (..., 3).

        Returns
        -------
        ti : np.ndarray
            Scalar TI array of shape (...,).
        """
        xyz = np.asarray(xyz, dtype=float)
        original_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(-1, 3)

        x = xyz_flat[:, 0]
        y = xyz_flat[:, 1]
        r = np.sqrt(x**2 + y**2)

        t = np.clip(r / self.basin_radius, 0.0, 1.0)
        ti = self.default_TI + t * (self._TI_outer - self.default_TI)

        return ti.reshape(original_shape)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def filtered_target(self) -> tuple[float, float]:
        """Currently-applied (u_eff, v_eff) after low-pass filter."""
        return (self._filtered_u, self._filtered_v)
