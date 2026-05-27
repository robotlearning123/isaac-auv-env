# Deep Research Report

- Generated: 2026-05-15 16:21:12
- Mode: `light`
- Elapsed: 21s
- Events: 43 (tool=3, meta=0)
- Tool: `search("Warp `

---

Below are concise research‑graded answers to **Q1–Q6**, as of *May 15 2026*, using the most current (2025–2026) documentation available. Each answer is categorized per your **ACHIEVABLE/REASONABLE/LATEST axes** and tagged for action priority.

---

## Q1 — Warp 1.13 Kernel Signatures (A, C)

**Finding:** Warp 1.13.0 APIs have *not* removed core primitives like spatial math or `wp.atomic_add`, and vector types remain supported; the release notes detail kernel, array, and AD primitives without breaking changes relevant to physics kernels. citeturn0search1turn0search4

- Warp 1.12+ added subscript‑style type hints (`wp.array[dtype]`), which is idiomatic for Python static typing and aligns with your type use. citeturn0search4  
- `wp.atomic_add` behaviour and `+=` still generate atomic adds (as documented). citeturn0search4  
- No indications of changed quaternion/spatial vector kernel semantics in v1.13. citeturn0search4

**Assessment:**  
- **LATEST:** Kernel signatures in §5 are consistent with Warp 1.13 idioms; slight tweak: prefer `wp.array[dtype]` type annotations for better mypy integration. **confident (high)**  
- **ACHIEVABLE:** Acceptable for 12‑20 h; no API incompatibility. **confident (high)**  
- **REASONABLE:** Scope stays within current Warp API. **confident (high)**

**Priority:** *MEDIUM — update code to use subscript type hints for arrays to satisfy type checkers.*

---

## Q2 — Newton 1.2 Pre‑step Hook (C)

**Finding:** Official Newton documentation and repo don’t currently expose documented high‑level callbacks or explicit pre‑force hooks in Python API. Available examples show typical simulation loop (ModelBuilder → Model → State → Solver), but no canonical `pre_step` callback API is documented. citeturn1search0turn1search1

Newton’s extensibility model does emphasize custom solvers and integration, but the specific “pre‑step force injection hook” pattern is not part of the publicly documented API surface as of v1.2. citeturn1search0

**Assessment:**  
- **LATEST:** Using undocumented or internal “pre‑step” wiring is risky; you may need to instead inject forces via the regular force buffer or create a custom Newton plugin. **confident (medium)**  
- **ACHIEVABLE:** If no native `pre_step` hook exists, you’ll need engineering to place your hydro force writes at the right stage (e.g., before solver integration). **confident (medium)**  
- **REASONABLE:** Plan assumes an API not (yet) documented; update plan to reflect force injection via known API.

**Priority:** *BLOCKER — confirm exact force injection mechanism in Newton 1.2 and update implementation target.*

---

## Q3 — Throughput Goal ≥ 100k @ 8192 (A, B)

**Finding:** Warp + Newton are designed for **massive SIMT parallelism** and high Throughput. Benchmarks for Newton/MuJoCo Warp in community posts claim **hundreds of millions of env‑steps/s on Blackwell GPUs** relative to CPU baselines. citeturn0reddit28

- Quoted 475× faster than MJX on Blackwell; implies multi‑millions of env‑steps/s feasible given vectorized kernels and GPU affinity. citeturn0reddit28  
- Custom Warp physics kernels (e.g., Navier‑Stokes AD examples) demonstrate efficient parallel updates with AD. citeturn0search16

Hydro kernels add work, but 100k/env‑steps at 8192 (~819M component updates) is modest for a Blackwell GPU.

**Assessment:**  
- **ACHIEVABLE:** ~100k env‑steps/s is consistent with Warp workloads on Blackwell, even with moderate overhead. **confident (high)**  
- **REASONABLE:** Projected 99% spare throughput after baseline suggests a realistic headroom. **confident (high)**  
- **LATEST:** Latest NV hardware and Warp design are suited to large env batches. **confident (high)**

**Priority:** *WATCH — but throughput target is defensible based on published parallels.*

---

## Q4 — 12–20 h Solo Budget (B)

**Finding:** Custom Warp kernels with full autograd and testing can be nontrivial, especially if AD edge cases arise. Warp AD examples (Navier‑Stokes) have ~300–500 lines of code and integrate complex differentiation features. citeturn0search16

No public repo shows an exact analog, but porting physics kernels with AD in Warp has required on the order of days (not hours) for experienced developers.

**Assessment:**  
- **ACHIEVABLE:** If you already have deep familiarity with Warp AD primitives and tests, 12–20 h might be plausible for initial kernels, but full autograd + exhaustive tests is a stretch. **confident (medium)**  
- **REASONABLE:** Planning thorough tests and CI may require more than one workday. **confident (medium)**  
- **LATEST:** Lean dev practices around PyTorch ↔ Warp AD comparisons are emerging and could speed validation.

**Priority:** *HIGH — consider adding slack to timeline for AD validation.*

---

## Q5 — Best Autograd Test Practices (A, C)

**Finding:** Warp provides native interoperability with PyTorch (`wp.from_torch` / `wp.to_torch`), suitable for AD comparisons. citeturn0search13

Using `torch.autograd.gradcheck` is idiomatic for verifying gradients; comparing with finite‑difference PyTorch via `functional_call` is workable but not the optimal 2026 idiom.

**Assessment:**  
- **LATEST:** Best practice is to convert Warp outputs to Torch tensors and use PyTorch’s `gradcheck` utilities when possible, reducing manual finite‑diff noise and making tests robust. **confident (high)**  
- **ACHIEVABLE:** This is implementable within your test suite without significant overhead. **confident (high)**  
- **REASONABLE:** Aligns with mainstream conventions in differentiable physics validation.

**Priority:** *MEDIUM — augment T2.7 tests with a `gradcheck` pattern.*

---

## Q6 — Missing Failure Modes (A, B)

**Finding:** Warp v1.13.0 has several **experimental features** (graph capture, texture BVH) that *could* pull in unexpected dependencies but aren’t directly relevant. citeturn0search4

Known general caveats (from Warp FAQ):  
- Synchronization costs and kernel launch overhead can sink perf if not batched efficiently. citeturn0search9  
- AD through non‑smooth functions uses subgradients; may lead to inconsistent gradients near discontinuities if not explicitly smoothed. citeturn0search9

Newton is still young; community discussions mention documentation gaps and evolving APIs. citeturn1search1

**Assessment:**  
- **LATEST:** Warp synchronization and AD quirks are real hazards for custom physics. **confident (high)**  
- **ACHIEVABLE:** If tests miss edge cases, early AD failures likely. **confident (medium)**  
- **REASONABLE:** These conditions aren’t in your failure modes list currently.

**Priority:** *HIGH — add explicit watch points for synchronization overhead and non‑smooth AD failure cases in tests.*

---

## Prioritized Action Items

1. **[BLOCKER]** Clarify Newton 1.2 force injection mechanism — update design because pre‑step hook may not exist.  
2. **[HIGH]** Extend timeline for AD validation; Warp AD kernel debugging can exceed initial budget.  
3. **[HIGH]** Add tests for Warp AD non‑smooth behavior and synchronization pitfalls.  
4. **[MEDIUM]** Use subscript array type hints for better type checking.  
5. **[MEDIUM]** Augment autograd tests with `torch.autograd.gradcheck`.

---

## Final Verdict (Last 100 Words)

- **ACHIEVABLE:** Tier‑1 hydro with tests and throughput ~100k env/s on a Blackwell GPU is probable given Warp + Newton’s design and published performance gains; main danger is AD complexity and untested injection API. **Confidence: Medium‑High**

- **REASONABLE:** Scope and velocity expectation for a v0.1 target is sensible, though autograd work and Newton integration may stretch the ∼12–20 h budget. **Confidence: Medium**

- **LATEST:** Plan mostly uses current Warp 1.13 and Newton 1.2 APIs; adjust autograd testing and force injection assumptions to match latest practices. **Confidence: High**.

---

If you want, I can also source examples from specific 2025–2026 GitHub repositories implementing Warp hydrodynamics kernels to further quantify development effort estimates.

---

## Sources

- [NVIDIA Warp Documentation — Warp 1.13.0](https://nvidia.github.io/warp/?utm_source=chatgpt.com)
- [Changelog — Warp 1.13.0](https://nvidia.github.io/warp/user_guide/changelog.html?utm_source=chatgpt.com)
- [Newton Physics Documentation](https://newton-physics.github.io/newton/stable/?utm_source=chatgpt.com)
- [Newton 1.0 is 100% open source. GPU-accelerated physics engine from NVIDIA, DeepMind, and Disney Research, now under the Linux Foundation](https://www.reddit.com/r/robotics/comments/1squlyf/newton_10_is_100_open_source_gpuaccelerated/?utm_source=chatgpt.com)
- [Build Accelerated, Differentiable Computational Physics Code for AI with NVIDIA Warp | NVIDIA Technical Blog](https://developer.nvidia.com/blog/build-accelerated-differentiable-computational-physics-code-for-ai-with-nvidia-warp/?utm_source=chatgpt.com)
- [Interoperability — Warp 1.13.0](https://nvidia.github.io/warp/user_guide/interoperability.html?utm_source=chatgpt.com)
- [FAQ — Warp 1.13.0](https://nvidia.github.io/warp/user_guide/faq.html?utm_source=chatgpt.com)
- [newton-physics/newton](https://github.com/newton-physics/newton?utm_source=chatgpt.com)