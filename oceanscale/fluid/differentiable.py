# mypy: ignore-errors
"""Differentiable 2D Navier-Stokes via wp.Tape for gradient-based optimization."""

from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def _add_force(
    vx: wp.array(dtype=wp.float32),
    vy: wp.array(dtype=wp.float32),
    fx: wp.array(dtype=wp.float32),
    fy: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    i = wp.tid()
    vx[i] = vx[i] + fx[i] * dt
    vy[i] = vy[i] + fy[i] * dt


@wp.kernel
def _diffuse_2d(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    alpha: wp.float32,
    beta: wp.float32,
):
    idx = wp.tid()
    iy = idx / nx
    ix = idx - iy * nx
    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
        dst[idx] = src[idx]
        return
    c = src[idx]
    l = src[idx - 1]
    r = src[idx + 1]
    d = src[idx - nx]
    u = src[idx + nx]
    dst[idx] = (c + alpha * (l + r + d + u)) * beta


@wp.kernel
def _divergence_2d(
    vx: wp.array(dtype=wp.float32),
    vy: wp.array(dtype=wp.float32),
    div: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    inv_2dx: wp.float32,
):
    idx = wp.tid()
    iy = idx / nx
    ix = idx - iy * nx
    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
        div[idx] = 0.0
        return
    div[idx] = (vx[idx + 1] - vx[idx - 1]) * inv_2dx + (vy[idx + nx] - vy[idx - nx]) * inv_2dx


@wp.kernel
def _pressure_jacobi_2d(
    p: wp.array(dtype=wp.float32),
    p_new: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    dx2: wp.float32,
):
    idx = wp.tid()
    iy = idx / nx
    ix = idx - iy * nx
    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
        p_new[idx] = 0.0
        return
    p_new[idx] = (p[idx - 1] + p[idx + 1] + p[idx - nx] + p[idx + nx] - dx2 * rhs[idx]) * 0.25


@wp.kernel
def _project_2d(
    vx: wp.array(dtype=wp.float32),
    vy: wp.array(dtype=wp.float32),
    p: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    inv_2dx: wp.float32,
):
    idx = wp.tid()
    iy = idx / nx
    ix = idx - iy * nx
    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
        return
    vx[idx] = vx[idx] - (p[idx + 1] - p[idx - 1]) * inv_2dx
    vy[idx] = vy[idx] - (p[idx + nx] - p[idx - nx]) * inv_2dx


@wp.kernel
def _l2_loss(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    diff = a[i] - b[i]
    wp.atomic_add(loss, 0, diff * diff)


@wp.kernel
def _copy_arr(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = src[i]


class DifferentiableFluidStep:
    """Differentiable 2D Navier-Stokes via Chorin projection + wp.Tape."""

    def __init__(self, grid_size: int = 32, viscosity: float = 0.01, device: str = "cuda:0"):
        self.nx = grid_size
        self.ny = grid_size
        self.n = grid_size * grid_size
        self.dx = 1.0 / grid_size
        self.viscosity = viscosity
        self.device = device
        self._pressure_iters = 20

        self.vx = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self.vy = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self._vx_tmp = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self._vy_tmp = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self._div = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self._p = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)
        self._p_tmp = wp.zeros(self.n, dtype=wp.float32, device=device, requires_grad=True)

    def forward(self, fx: wp.array, fy: wp.array, dt: float = 0.01):
        """One Chorin projection step (differentiable). Modifies vx/vy in place."""
        nx, ny, n, dev = self.nx, self.ny, self.n, self.device
        alpha_diff = self.viscosity * dt / (self.dx * self.dx)
        beta_diff = 1.0 / (1.0 + 4.0 * alpha_diff)
        inv_2dx = 0.5 / self.dx
        dx2 = self.dx * self.dx

        # 1. Add external force
        wp.launch(_add_force, dim=n, inputs=[self.vx, self.vy, fx, fy, dt], device=dev)

        # 2. Diffuse (Jacobi iterations)
        for _ in range(4):
            wp.launch(_diffuse_2d, dim=n, inputs=[self.vx, self._vx_tmp, nx, ny, alpha_diff, beta_diff], device=dev)
            wp.launch(_copy_arr, dim=n, inputs=[self._vx_tmp, self.vx], device=dev)
            wp.launch(_diffuse_2d, dim=n, inputs=[self.vy, self._vy_tmp, nx, ny, alpha_diff, beta_diff], device=dev)
            wp.launch(_copy_arr, dim=n, inputs=[self._vy_tmp, self.vy], device=dev)

        # 3. Pressure projection
        wp.launch(_divergence_2d, dim=n, inputs=[self.vx, self.vy, self._div, nx, ny, inv_2dx], device=dev)
        self._p.zero_()
        for _ in range(self._pressure_iters):
            wp.launch(_pressure_jacobi_2d, dim=n, inputs=[self._p, self._p_tmp, self._div, nx, ny, dx2], device=dev)
            wp.launch(_copy_arr, dim=n, inputs=[self._p_tmp, self._p], device=dev)

        wp.launch(_project_2d, dim=n, inputs=[self.vx, self.vy, self._p, nx, ny, inv_2dx], device=dev)

    def compute_loss(self, target_vx: wp.array, target_vy: wp.array) -> wp.array:
        """L2 loss between current and target velocity."""
        loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
        wp.launch(_l2_loss, dim=self.n, inputs=[self.vx, target_vx, loss], device=self.device)
        wp.launch(_l2_loss, dim=self.n, inputs=[self.vy, target_vy, loss], device=self.device)
        return loss

    def reset(self):
        self.vx.zero_()
        self.vy.zero_()
        self._p.zero_()

    def optimize_force(
        self, target_vx: wp.array, target_vy: wp.array, n_iters: int = 10, lr: float = 0.01, dt: float = 0.01
    ) -> list[float]:
        """Find external force producing target velocity via gradient descent."""
        fx = wp.zeros(self.n, dtype=wp.float32, device=self.device, requires_grad=True)
        fy = wp.zeros(self.n, dtype=wp.float32, device=self.device, requires_grad=True)
        losses: list[float] = []

        for _ in range(n_iters):
            self.reset()
            tape = wp.Tape()
            with tape:
                self.forward(fx, fy, dt)
                loss = self.compute_loss(target_vx, target_vy)
            tape.backward(loss)
            loss_val = float(loss.numpy()[0])
            losses.append(loss_val)

            fx_grad = fx.grad.numpy()
            fy_grad = fy.grad.numpy()
            fx_np = fx.numpy() - lr * fx_grad
            fy_np = fy.numpy() - lr * fy_grad
            fx.assign(wp.array(fx_np, dtype=wp.float32, device=self.device))
            fy.assign(wp.array(fy_np, dtype=wp.float32, device=self.device))

            tape.zero()

        return losses


def warp_fluid_to_torch(solver: DifferentiableFluidStep, fx_torch, fy_torch, dt: float = 0.01):
    """Bridge: torch tensors → Warp fluid step → loss with gradient support."""

    fx_wp = wp.from_torch(fx_torch.contiguous().flatten())
    fy_wp = wp.from_torch(fy_torch.contiguous().flatten())

    solver.reset()
    tape = wp.Tape()
    with tape:
        solver.forward(fx_wp, fy_wp, dt)
    return tape
