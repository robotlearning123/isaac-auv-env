"""Smoke tests for the benchmark suite — 2 envs, 10 steps each."""

import pytest

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("warp").is_cuda_available(),
    reason="CUDA required",
)

from oceanscale.benchmarks.suite import (
    STANDARD_BENCHMARKS,
    run_all_benchmarks,
    run_benchmark,
    _make_station_keeping,
    _make_docking,
    _make_waypoint,
)


@pytest.mark.parametrize("factory", [_make_station_keeping, _make_docking, _make_waypoint])
def test_run_benchmark_smoke(factory):
    result = run_benchmark(factory, n_envs=2, n_steps=10, device="cuda:0")
    assert result.wall_time_s > 0
    assert result.fps > 0
    assert result.total_steps == 10
    assert result.n_envs == 2


def test_print_results_smoke(capsys):
    from oceanscale.benchmarks.suite import print_results, BenchmarkResult
    fake = [BenchmarkResult(task_name="test", n_envs=2, total_steps=10, wall_time_s=0.1)]
    print_results(fake)
    out = capsys.readouterr().out
    assert "test" in out
    assert "FPS" in out
