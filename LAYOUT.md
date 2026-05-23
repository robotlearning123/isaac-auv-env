# Package Layout — `oceanscale/` (v0.1.0 target)

**Date:** 2026-05-15  
**Status:** Plan. The directory tree below is what we **will** create as we implement v0.1 → v0.5; not all files exist yet.

---

## 1. Top-level repo layout

```
oceanscale/                         # repo root (monorepo, GitHub: robotlearning123/oceanscale)
├── pyproject.toml                  # uv-managed deps + tool config
├── uv.lock                         # pinned versions, committed
├── README.md                       # project description
├── LICENSE                         # Apache-2.0
├── CHANGELOG.md                    # added at v0.1.0 tag
├── CONTRIBUTING.md                 # added when repo opens for contributors
├── .gitignore .gitattributes
├── .pre-commit-config.yaml
├── .github/                        # added when CI configured (v0.2)
│   └── workflows/
│       ├── lint.yml                # ruff + mypy
│       ├── test-cpu.yml            # pytest -m "not gpu"
│       └── test-gpu.yml            # pytest -m gpu (self-hosted RTX runner)
├── docs/                           # added at v0.3 (sphinx or mkdocs)
│
├── SURVEY.md                       # SOTA landscape (May 2026)
├── ARCHITECTURE.md                       # architecture overview
├── STACK.md                        # tech stack analysis + decision framework
├── VERSIONS.md                     # version pinning rationale
├── LAYOUT.md                       # this file
│
├── oceanscale/                     # main package (Apache-2.0)
│   ├── __init__.py                 # __version__
│   ├── _config.py                  # global constants (default dt, sound speed, etc.)
│   ├── core/                       # cross-cutting primitives
│   │   ├── __init__.py
│   │   ├── types.py                # Pose, Twist, Wrench dataclasses (Warp-array backed)
│   │   ├── frames.py               # SE(3) transforms (Warp + scipy reference)
│   │   ├── tape.py                 # Warp-tape utilities for autograd
│   │   └── logging.py              # rich/structured logging setup
│   │
│   ├── hydro/                      # § 5.1 of STACK.md — tiered fluid model
│   │   ├── __init__.py
│   │   ├── tier0_ellipsoid.py      # wrapper around MJWarp built-in fluid
│   │   ├── tier1_fossen.py         # custom Warp kernel — added-mass M_A, Coriolis C_A, damping D
│   │   ├── tier1_kernels.py        # @wp.kernel definitions (separated for cache)
│   │   ├── tier2_sph.py            # localized SPH (manipulator, thruster wake)
│   │   ├── tier3_waves.py          # Gerstner + Stable-Fluid 2D surface
│   │   ├── currents.py             # 3D velocity field sampling
│   │   ├── schemas.py              # USD schema definitions (oceanscale:Hydrodynamics)
│   │   └── injector.py             # hooks that inject forces into Newton state.body_f
│   │
│   ├── sensors/                    # § 5.2 of STACK.md
│   │   ├── __init__.py
│   │   ├── sonar.py                # BVH ray-cast kernel + speckle + reverberation
│   │   ├── sonar_kernels.py        # @wp.kernel ray launch + atomic energy accumulate
│   │   ├── vision_jmg.py           # Jaffe-McGlamery post-pass (Beer-Lambert + scatter)
│   │   ├── dvl.py                  # 4-beam Doppler analytic
│   │   ├── imu.py                  # Allan-variance noise model
│   │   ├── pressure.py             # depth + noise
│   │   ├── gps.py                  # surface-only with mask
│   │   ├── acoustic_modem.py       # BELLHOP cache + Q-D Doppler kernel
│   │   └── schemas.py              # USD schemas for all sensors
│   │
│   ├── scene/                      # USD asset authoring + scenario composition
│   │   ├── __init__.py
│   │   ├── loader.py               # Newton + USD bridge
│   │   ├── builder.py              # programmatic scenario construction
│   │   ├── instancing.py           # 8k-env GPU instancing helpers
│   │   ├── vehicles/               # canonical vehicle USD assets + Python wrappers
│   │   │   ├── bluerov2_heavy.py
│   │   │   └── bluerov2.py
│   │   └── terrain/                # bathymetry loaders
│   │       └── flat.py
│   │
│   ├── tasks/                      # § 7.2 — RL benchmark tasks (Isaac Lab compatible)
│   │   ├── __init__.py
│   │   ├── _base.py                # common ManagerBasedRLEnv-shaped base
│   │   ├── station_keeping.py      # v0.1 first task
│   │   ├── pipe_following.py       # v0.2
│   │   ├── docking.py              # v0.2
│   │   ├── bathymetric_survey.py   # v0.3
│   │   └── acoustic_marl.py        # v0.4
│   │
│   ├── randomize/                  # § 5.4 DR toolkit
│   │   ├── __init__.py
│   │   ├── inertial.py             # mass, COB-COM offset, added-mass scale
│   │   ├── actuator.py             # thruster gain, deadband, tau
│   │   ├── environment.py          # current, wave, turbidity, salinity
│   │   ├── sensor.py               # IMU bias, DVL dropout, sonar SNR, camera ISO
│   │   ├── faults.py               # thruster failure, hull damage
│   │   └── ddr.py                  # data-informed DR hooks
│   │
│   ├── rl/                         # RL frontend adapters
│   │   ├── __init__.py
│   │   ├── envs.py                 # gym.Env wrapper over Newton vectorized state
│   │   ├── obs.py                  # observation extractors
│   │   ├── actions.py              # action → thruster mapping
│   │   ├── rewards.py              # reward terms
│   │   └── adapters/               # backend bridges
│   │       ├── rl_games.py         # v0.2
│   │       ├── skrl.py             # v0.2
│   │       └── rsl_rl.py           # v0.3
│   │
│   ├── viz/                        # visualization & logging
│   │   ├── __init__.py
│   │   ├── newton_viewer.py        # default Newton render
│   │   ├── tensorboard_hooks.py    # training logs
│   │   ├── trajectory_plot.py      # matplotlib for offline eval
│   │   └── usd_recorder.py         # write USD timeline for replay
│   │
│   └── cli/                        # entry points
│       ├── __init__.py
│       ├── train.py                # `oceanscale train <task>`
│       ├── eval.py                 # `oceanscale eval <ckpt>`
│       └── bench.py                # `oceanscale bench`  — throughput micro-benchmark
│
├── tests/                          # mirrors oceanscale/ layout
│   ├── __init__.py
│   ├── conftest.py                 # GPU fixtures, scene fixtures
│   ├── test_environment.py         # current — stack validation
│   ├── core/
│   │   ├── test_types.py
│   │   ├── test_frames.py
│   │   └── test_tape.py
│   ├── hydro/
│   │   ├── test_tier1_fossen.py    # numerical match vs MarineGym PyTorch reference
│   │   ├── test_tier1_autograd.py  # wp.Tape vs torch.autograd numerical check
│   │   └── test_tier3_waves.py
│   ├── sensors/
│   │   ├── test_sonar.py
│   │   ├── test_vision_jmg.py
│   │   ├── test_dvl.py
│   │   └── test_imu.py
│   ├── scene/
│   │   └── test_loader.py
│   ├── tasks/
│   │   └── test_station_keeping.py # smoke test, 100 steps
│   ├── randomize/
│   │   └── test_inertial.py
│   └── rl/
│       └── test_envs.py
│
├── benchmarks/                     # micro/macro perf benchmarks
│   ├── __init__.py
│   ├── kernel_throughput.py        # per-kernel FPS measurement
│   ├── env_fps.py                  # end-to-end env step rate
│   └── compare_marinegym.py        # apples-to-apples vs MarineGym (when v0.2)
│
├── examples/                       # end-user-facing minimal demos
│   ├── 01_hello_warp.py            # smallest Warp kernel + autograd
│   ├── 02_hello_newton.py          # smallest Newton scene
│   ├── 03_fossen_drag.py           # Tier-1 drag demo
│   ├── 04_station_keeping_train.py # full RL training script
│   └── 05_sonar_render.py          # one-shot sonar image
│
├── assets/                         # canonical USD assets (small, committed)
│   ├── vehicles/
│   │   └── bluerov2_heavy.usd
│   └── terrain/
│       └── flat_seabed.usd
│
├── scripts/                        # one-off ops scripts (not packaged)
│   ├── port_marinegym_bluerov2.py  # URDF → MJCF → USD pipeline
│   └── fetch_assets.py             # download large assets from a CDN
│
├── configs/                        # Hydra configs
│   ├── env/
│   │   ├── station_keeping.yaml
│   │   └── _base.yaml
│   ├── agent/
│   │   ├── ppo_rl_games.yaml
│   │   └── ppo_skrl.yaml
│   └── train.yaml                  # top-level
│
└── install_log/                    # gitignored — install / verify logs
```

---

## 2. Module responsibility table

| Module | Owns | Depends on | First populated at |
|---|---|---|---|
| `oceanscale.core` | SE(3) math, Warp-tape utils, types | warp, numpy | v0.1 |
| `oceanscale.hydro` | hydrodynamics kernels (tiered) | warp, newton, core | v0.1 (Tier 0-1) → v0.3 (Tier 2-3) |
| `oceanscale.sensors` | sensor physics + USD schemas | warp, newton, core, scene | v0.2 |
| `oceanscale.scene` | USD/Newton bridge + asset wrappers | usd-core, newton | v0.1 (BlueROV2 only) |
| `oceanscale.tasks` | RL env definitions | gymnasium / Isaac Lab | v0.1 (station-keep) → ongoing |
| `oceanscale.randomize` | DR terms | core, scene | v0.1 (basic) → v0.3 (DDR) |
| `oceanscale.rl` | RL backend adapters | task-agnostic | v0.2 |
| `oceanscale.viz` | rendering + logging | tensorboard, matplotlib | v0.1 (Newton viewer) → v0.3 (USD recorder) |
| `oceanscale.cli` | CLI entry points | hydra | v0.2 |

---

## 3. Import discipline

To keep cycles out and tests fast:

```
core   <─ depended on by all
hydro  <─ depends on core only
sensors <─ depends on core + scene
scene  <─ depends on core
tasks  <─ depends on hydro + sensors + scene
randomize <─ depends on tasks
rl     <─ depends on tasks
viz    <─ depends on anything (read-only)
cli    <─ depends on everything (top of stack)
```

Forbidden: `core` importing from any other oceanscale module.

---

## 4. File-size discipline (preserves review quality)

- Any `.py` > 500 lines: must be split.
- Any `@wp.kernel` body > 80 lines: refactor into helper kernels.
- Any test file > 300 lines: split by behavior.
- One class per file when class > 100 lines.

---

## 5. v0.1.0 minimal scaffold (what we'll create *first*)

```
oceanscale/
├── __init__.py                  ✓ exists
├── core/
│   ├── __init__.py
│   ├── types.py
│   └── tape.py
├── hydro/
│   ├── __init__.py
│   ├── tier1_fossen.py
│   └── tier1_kernels.py
└── scene/
    ├── __init__.py
    └── vehicles/
        ├── __init__.py
        └── bluerov2_heavy.py
tests/
├── __init__.py                  ✓ exists
├── test_environment.py          ✓ exists
├── core/
│   └── test_types.py
└── hydro/
    └── test_tier1_fossen.py
examples/
└── 01_hello_warp.py
benchmarks/
└── kernel_throughput.py
```

Everything else gets added at the version it's listed in §2.

---

## 6. Naming conventions

- **Modules:** snake_case
- **Classes:** PascalCase
- **Functions:** snake_case, verb-first
- **Constants:** UPPER_SNAKE
- **Warp kernels:** `_kernel_*` prefix when private; otherwise descriptive verb (e.g., `fossen_damping`, `sonar_raycast`)
- **Test files:** `test_<module>.py` mirroring source layout
- **Fixtures:** in nearest `conftest.py`, named with intent (`bluerov2_state`, `gpu_device`)

---

## 7. What lives **outside** the package

- `scripts/` — one-shot operational scripts (asset porting, data fetching). Not importable, not tested.
- `examples/` — runnable demonstrations. Tested via `pytest --collect-only` parse, but main test value is `mypy --strict`.
- `benchmarks/` — perf measurement; `pytest-benchmark` integration eventually.
- `configs/` — Hydra YAML. No Python in here.
- `assets/` — USD/mesh files. Large assets fetched at runtime; committed assets are < 10 MB.

---

## 8. Documentation lives where

- **Top-level docs** (SURVEY/DESIGN/STACK/VERSIONS/LAYOUT) — strategic, project-wide
- **Module docstrings** — what each subpackage does
- **Function docstrings** — Google style, type hints in signature
- **`docs/` sphinx site** — added at v0.3 when API stabilizes
- **No README per subpackage** — module docstring is enough for now

---

## 9. Open layout questions

1. **Should `cli/` use `click` or just `argparse`?** Default: argparse (no extra dep). Reconsider when CLI gets complex.
2. **Hydra in v0.1 or v0.2?** Hydra in v0.1 — config explosion happens fast.
3. **`examples/` vs `cli/`?** examples = pedagogical, copy-edit-friendly; cli = production entry points. Distinct.
4. **Where does `Isaac Lab` env config go?** v0.2 question; likely `oceanscale.tasks.<task>.IsaacLabEnvCfg`.
5. **Vendored USD assets in git or LFS?** v0.3 question. Default: small assets in git (<5 MB), large via fetch script.
