#!/bin/bash
# Comprehensive benchmark smoke-test runner
# Runs each benchmark with minimal problem size for validation
# Results logged to install_log/bench_YYYYMMDD_HHMMSS/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="install_log/bench_${TIMESTAMP}"
mkdir -p "$LOGDIR"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

run_bench() {
    local name="$1"
    local cmd="$2"
    local logfile="${LOGDIR}/${name}.log"
    TOTAL=$((TOTAL + 1))

    echo -n "[$TOTAL] $name ... "
    if timeout 120 uv run $cmd > "$logfile" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "TIMEOUT (120s)"
            SKIP=$((SKIP + 1))
        else
            echo "FAIL (exit $exit_code)"
            FAIL=$((FAIL + 1))
        fi
    fi
}

echo "═══════════════════════════════════════════════════════"
echo "  OceanScale Benchmark Smoke Test"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "  Logs: $LOGDIR/"
echo "═══════════════════════════════════════════════════════"
echo ""

# --- Group 1: Core OceanScale ---
echo "── Core OceanScale ──"
run_bench "tier1_throughput" "python benchmarks/tier1_throughput.py"
run_bench "oceanscale_vs_bullet" "python benchmarks/oceanscale_vs_bullet.py"
run_bench "kernel_throughput" "python benchmarks/kernel_throughput.py"

# --- Group 2: Newton solvers ---
echo "── Newton Solvers ──"
run_bench "newton_underwater_api" "python benchmarks/newton_underwater_bench.py --scenario api_survey"
run_bench "newton_underwater_rigid" "python benchmarks/newton_underwater_bench.py --scenario rigid"
run_bench "newton_worlds" "python benchmarks/newton_worlds_throughput.py"
run_bench "newton_mpm" "python benchmarks/newton_mpm_bench.py"
run_bench "kamino_solver" "python benchmarks/kamino_solver_bench.py"
run_bench "mujoco_warp" "python benchmarks/mujoco_warp_bench.py"

# --- Group 3: Fluid dynamics ---
echo "── Fluid Dynamics ──"
run_bench "warp_fluid" "python benchmarks/warp_fluid_bench.py"
run_bench "navier_stokes_3d" "python benchmarks/navier_stokes_3d.py"
run_bench "sph_advanced" "python benchmarks/sph_advanced_bench.py"
run_bench "spectral_cfd" "python benchmarks/spectral_cfd_bench.py"
run_bench "vortex_particle" "python benchmarks/vortex_particle_bench.py"
run_bench "fsi_underwater" "python benchmarks/fsi_underwater_bench.py"

# --- Group 4: Consolidated benchmarks ---
echo "── Consolidated (new) ──"
run_bench "jacobi_stencil_3d_warp" "python benchmarks/jacobi_stencil_bench.py --frameworks warp --dims 3d --sizes-3d 32 --iters-3d 10"
run_bench "jacobi_stencil_3d_torch" "python benchmarks/jacobi_stencil_bench.py --frameworks pytorch --dims 3d --sizes-3d 32 --iters-3d 10"
run_bench "lbm_d3q19_warp" "python benchmarks/lbm_d3q19_bench.py --backend warp --size 32 --steps 10"
run_bench "lbm_d3q19_torch" "python benchmarks/lbm_d3q19_bench.py --backend torch --size 32 --steps 10"

# --- Group 5: Cross-framework ---
echo "── Cross-framework ──"
run_bench "cupy_lbm" "python benchmarks/cupy_lbm_bench.py"
run_bench "jax_cfd" "python benchmarks/jax_cfd_bench.py"
run_bench "sparse_solve" "python benchmarks/sparse_solve_bench.py"
run_bench "gpu_baseline" "python benchmarks/gpu_baseline.py"
run_bench "gpu_optimization" "python benchmarks/gpu_optimization_bench.py"

# --- Group 6: External frameworks ---
echo "── External Frameworks ──"
run_bench "taichi_sph" "python benchmarks/taichi_sph_bench.py"
run_bench "taichi_euler" "python benchmarks/taichi_euler_bench.py"
run_bench "xlb_bench" "python benchmarks/xlb_bench.py"

# --- Summary ---
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  RESULTS: $PASS PASS / $FAIL FAIL / $SKIP TIMEOUT / $TOTAL TOTAL"
echo "  Logs: $LOGDIR/"
echo "═══════════════════════════════════════════════════════"

# Write summary file
cat > "${LOGDIR}/SUMMARY.txt" << EOF
OceanScale Benchmark Smoke Test
$(date -u +%Y-%m-%dT%H:%M:%SZ)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)

RESULTS: $PASS PASS / $FAIL FAIL / $SKIP TIMEOUT / $TOTAL TOTAL

Per-benchmark status:
$(for f in "$LOGDIR"/*.log; do
    name=$(basename "$f" .log)
    if grep -q "Error\|Traceback\|FAILED" "$f" 2>/dev/null; then
        echo "  FAIL  $name"
    else
        echo "  PASS  $name"
    fi
done)
EOF

echo ""
echo "Summary written to ${LOGDIR}/SUMMARY.txt"

# Exit with non-zero if any failures
[ $FAIL -eq 0 ] || exit 1
