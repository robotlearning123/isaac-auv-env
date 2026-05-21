"""XLB (Autodesk LBM) benchmark: D2Q9 on RTX 5090."""
import warp as wp
import time
import numpy as np

wp.init()

from xlb.velocity_set import D2Q9
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper

N = 256
vs = D2Q9(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
grid = grid_factory(N, N, vs, ComputeBackend.WARP)

stepper = IncompressibleNavierStokesStepper(
    omega=1.7,
    velocity_set=vs,
    grid=grid,
    boundary_conditions=[],
)

print(f"XLB LBM D2Q9: grid={N}x{N}, backend=Warp")

f_in = grid.create_field()
f_out = grid.create_field()

# Warmup
for _ in range(10):
    f_out = stepper(f_in)
    f_in, f_out = f_out, f_in

# Benchmark
times = []
for step in range(100):
    t0 = time.perf_counter()
    f_out = stepper(f_in)
    f_in, f_out = f_out, f_in
    wp.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = np.mean(times[5:])
print(f"XLB LBM D2Q9: avg step time={avg*1000:.2f} ms, throughput={N*N/avg/1e6:.1f} Mcell/s")
print(f"  min={np.min(times[5:])*1000:.2f} ms, max={np.max(times[5:])*1000:.2f} ms")
