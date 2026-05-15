# Version Pinning Plan — v0.1.0

**Date:** 2026-05-15  
**Scope:** Pinned version specification for the v0.1.0 dev environment. Plan only — no installs yet. Translates directly into `pyproject.toml` when implementation phase starts.

---

## 1. Verified facts (May 2026)

Sources checked directly against PyPI + official docs in this session:

| Package | Latest version | Date | Python support | CUDA | Notes |
|---|---|---|---|---|---|
| `warp-lang` | **1.13.0** | 2026-05-04 | 3.10 – 3.14 | wheel built with **CUDA 12.8** | `torch-cu12` extra available; CUDA 13 driver back-compat OK |
| `newton` | **1.2.0** | 2026-05-12 | 3.10 – 3.13 | bundled CUDA 12 | Renamed from `newton-physics` (1.0.0 deprecated); extras: `dev`, `examples`, `sim`, `torch-cu12`, `torch-cu13` |
| `newton-physics` | 1.0.0 | 2026-02-27 | — | — | **DEPRECATED redirect**, do not use |
| `torch` | stable 2.7.x | 2026 | 3.10+ | wheels for 11.8 / 12.6 / 12.8; **no CUDA 13** | use `cu128` for our x86_64 RTX 5090 |
| `isaacsim` (pip) | 5.1.0 | 2026 | **3.11** required for 5.1 | cu128 | Pip-installable only for 5.x; 6.0 is source-build / early dev |
| Isaac Lab 3.0 | beta on `develop` branch | 2026 | not pinned | — | **Source build only**, no pip wheel |
| `rl-games` | (unverified, last known 1.6.5) | 2026-02 | 3.10+ | — | Battle-tested PPO; use for v0.1 |
| `skrl` | (unverified, 2.0.x line) | 2026 | 3.10+ | — | Defer to v0.2 |
| `usd-core` (Pixar OpenUSD Python) | 25.11+ | 2026 | 3.10+ | — | Needed for USD asset loading |

**User's current environment (verified by `nvidia-smi` + `pip show`):**
- Driver 580.95.05 (supports up to CUDA 13)
- nvcc / CUDA Toolkit 12.8
- Python 3.13.5
- `warp-lang` 1.9.0 (one minor behind 1.13)
- `torch` 2.10.0 (newer than verified stable — likely future-tracking nightly)
- RTX 5090, 32 GB Blackwell

---

## 2. Version decisions (the pinning)

### 2.1 Python: **3.12** (not 3.13)

Reasoning:
- Isaac Lab 3.0 hasn't pinned a Python version yet; **Isaac Sim 5.1 (pip-installable today) requires 3.11**, Isaac Sim 6 expected to standardize on 3.12.
- Newton & Warp both fully support 3.10-3.13, so they're flexible.
- **3.12 is the safest common denominator across the full target ecosystem** (Newton + Warp + Isaac Lab + future Isaac Sim 6 GA).
- 3.13 is newer than your installed system Python but is also one version ahead of the documented Isaac Lab baseline — risks edge-case bugs in 3.13-specific dict/iterator changes that no one has tested.
- **Decision**: project uses 3.12 via `uv venv --python 3.12`. User's system 3.13 remains untouched.

### 2.2 CUDA Toolkit: **12.8** (do not change)

Per chat answer:
- All target frameworks ship cu128 wheels primarily.
- Driver 580 covers CUDA 12 and 13; toolkit 12.8 is what code is built against.
- CUDA 13 wheel ecosystem is not yet mature (May 2026); revisit Q4 2026.
- aarch64 (e.g., DGX Spark) uses cu130 per Isaac Lab docs — irrelevant for our x86_64 RTX 5090.

### 2.3 Warp: `>=1.13.0,<2.0`

- 1.13.0 is the current minor; 1.x semver promises API stability.
- Upper-bound pin guards against 2.x breaking changes (likely 2026 Q4 - 2027 Q1).
- User's installed 1.9.0 → upgrade to 1.13.x in the new isolated env (no conflict with system).
- Install via extras: `warp-lang[torch-cu12]` for PyTorch interop.

### 2.4 Newton: `>=1.2.0,<2.0`

- Package name is `newton` (NOT `newton-physics`, which is the deprecated redirect).
- 1.2.0 is current; 1.x semver. Upper-bound 2.0.
- Install via extras: `newton[examples,torch-cu12]` for example scenarios + torch interop.

### 2.5 PyTorch: `>=2.7.0,<3.0`

- Stable 2.7 is the documented minimum for Isaac Lab 3.0.
- Wheel selection: `--index-url https://download.pytorch.org/whl/cu128`.
- User has 2.10 in system; we'll pull whatever stable ≥2.7 cu128 wheel is current at install time.

### 2.6 RL backend: `rl_games >=1.6.0,<2.0`

- Battle-tested default for Isaac Lab.
- skrl deferred to v0.2 when JAX/Warp backend support matters.
- PureJaxRL/Stoix deferred until JAX track opens.

### 2.7 Isaac Sim / Isaac Lab: **deferred to v0.2**

- Isaac Sim 6.0 is Early Dev (source build) — too heavy for v0.1 smoke tests.
- Isaac Lab 3.0 Beta is source-build only.
- v0.1 will use **Newton standalone** (without Isaac Sim) — Newton can step physics, render via its own viewer, sufficient for kernel-level development.
- **v0.2 milestone gate**: install Isaac Sim 5.1 (pip) for first end-to-end env test; revisit 6.0 source-build at v0.3 once GA timing is clearer.

### 2.8 USD tooling: `usd-core >=25.11`

- Needed early to author USD scenes for vehicles + sensors.
- Pixar's official Python binding; no Omniverse dependency.

### 2.9 Tooling

| Tool | Pin | Why |
|---|---|---|
| `pytest` | `>=8.0,<9` | Test runner |
| `pytest-cov` | `>=5.0` | Coverage |
| `hypothesis` | `>=6.100` | Property tests for kernel correctness |
| `ruff` | `>=0.6,<1.0` | Lint + format (replaces black + flake8 + isort) |
| `mypy` | `>=1.10,<2.0` | Static type check |
| `pre-commit` | `>=3.7` | Hook runner |
| `hydra-core` | `>=1.3,<2.0` | Config management for experiments |
| `tensorboard` | `>=2.16` | Local training logs |
| `wandb` | optional | Online experiment tracking (defer to v0.2) |
| `tqdm` | `>=4.66` | Progress bars |
| `rich` | `>=13` | Pretty CLI output |
| `numpy` | `>=1.26,<3` | Array ops |
| `scipy` | `>=1.13` | Validation reference (analytic Fossen, BELLHOP wrapper) |

---

## 3. Tiered install plan

### Tier A — Core (v0.1, install at sprint 1)

```
python==3.12
warp-lang[torch-cu12] >=1.13.0,<2.0
newton[examples,torch-cu12] >=1.2.0,<2.0
torch >=2.7.0,<3 (via cu128 index)
numpy >=1.26,<3
scipy >=1.13
usd-core >=25.11
hydra-core >=1.3,<2.0
pytest, pytest-cov, hypothesis, ruff, mypy, pre-commit, tensorboard, tqdm, rich
```

### Tier B — Add at v0.2 (Isaac Lab + RL)

```
isaacsim[all,extscache] ==5.1.0 (or 6.0 source build)
rl-games >=1.6.0,<2.0
gymnasium >=0.29
```

### Tier C — Add at v0.3 (vision + extras)

```
skrl >=2.0,<3.0
rsl-rl >=5.0
wandb >=0.17
opencv-python-headless >=4.10
```

### Tier D — Reserved (v0.4+)

```
jax + jaxlib (CUDA 12.8 wheels, separate venv)  # PureJaxRL track
gsplat or splatfacto                             # 3DGS evaluation
```

---

## 4. Proposed `pyproject.toml` skeleton

```toml
[project]
name = "oceanscale"
version = "0.1.0"
description = "GPU-native fluid-structure simulation for underwater robotics"
authors = [{name = "TODO"}]
license = "Apache-2.0"
readme = "README.md"
requires-python = ">=3.12,<3.14"

dependencies = [
    "warp-lang[torch-cu12]>=1.13.0,<2.0",
    "newton[examples,torch-cu12]>=1.2.0,<2.0",
    "torch>=2.7.0,<3",
    "numpy>=1.26,<3",
    "scipy>=1.13",
    "usd-core>=25.11",
    "hydra-core>=1.3,<2.0",
    "tensorboard>=2.16",
    "tqdm>=4.66",
    "rich>=13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.6,<1.0",
    "mypy>=1.10,<2.0",
    "pre-commit>=3.7",
]
rl = [
    "rl-games>=1.6.0,<2.0",
    "gymnasium>=0.29",
]
sim = [
    # Reserved for Isaac Sim / Lab when we add at v0.2
]

[tool.uv]
index-url = "https://pypi.org/simple"
extra-index-url = ["https://download.pytorch.org/whl/cu128"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true  # warp/newton lack type stubs

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires CUDA GPU",
    "slow: takes >10s",
]
```

---

## 5. Open questions to resolve before install phase

1. **Where do we install?** `/home/robot/workspace/46-marine/.venv` (uv default) — confirm.
2. **Do we commit `uv.lock`?** Yes, for reproducibility.
3. **CI runner?** Plan: self-hosted on RTX 5090 box for now; revisit cloud GPU CI at v0.3.
4. **Python 3.12 install needed?** Probably — user has 3.13. `uv venv --python 3.12` will fetch one if missing. Trivial.
5. **Do we set `WARP_CACHE_DIR` env var?** Recommend yes to `~/.cache/warp-46marine/` to isolate from user's existing warp install.

---

## 6. Validation checklist (before any code)

When we move from plan → install, verify in order:

- [ ] `uv venv --python 3.12` creates clean venv
- [ ] `uv sync` installs Tier A without resolver errors
- [ ] `python -c "import warp; warp.init(); print(warp.get_devices())"` shows RTX 5090
- [ ] `python -c "import newton; print(newton.__version__)"` shows ≥1.2.0
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` shows True + RTX 5090
- [ ] `python -c "from pxr import Usd; print(Usd.GetVersion())"` shows ≥25.11
- [ ] `pytest --collect-only` runs without import errors

Each line above is one validation step. If any fail, debug before continuing.

---

## 7. What changes from STACK.md

`STACK.md §11` mentioned Python 3.12 + CUDA 12.4 + PyTorch 2.7. This document supersedes with verified May 2026 facts:

- ✅ Python 3.12 — confirmed
- ⚠️ CUDA toolkit 12.8 (not 12.4 — newer is fine, 12.4 baseline was conservative)
- ✅ PyTorch 2.7+ — confirmed minimum
- ➕ Newton package name is `newton`, not `newton-physics`
- ➕ Warp wheel CUDA 12 is canonical PyPI; CUDA 13 build only on aarch64 / source

I'll patch STACK.md §11 with these updates after this planning doc is approved.
