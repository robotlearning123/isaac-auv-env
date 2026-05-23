# OceanScale — Positioning

**Status:** Canonical v1.1. Last revised 2026-05-22.
**Authority:** Single source of truth for OceanScale messaging, scope, and voice. Website copy, package READMEs, fundraising deck, talks, social posts, and AI-agent prompts derive from this file. When any downstream artifact conflicts with this file, fix the artifact.
**Maintenance:** Changes happen here first, downstream follows. Edits to category / scope / anchor / hierarchy / lexicon require a new entry in the Changelog at the bottom.

This doc is brand law. The reference voice versions are in Appendix A — they are examples derived from this law, not part of it.

---

## 1. Category

**OceanScale is an infrastructure company.**

Specifically: AI-native simulation infrastructure for underwater robotics.

Not an AI lab (Wayve / 1X / Physical Intelligence pattern). Not a generic simulation tool (Gazebo / Stonefish / HoloOcean pattern). Not a foundation-model company. Any external doc names the category as infrastructure.

Voice models for tone: Lightwheel, Modal Labs, Anthropic. Compress, do not accumulate.

---

## 2. Scope

**OceanScale serves underwater robotics: AUVs, ROVs, underwater vehicles.**

We do not currently serve:
- Surface vessels / USVs / autonomous ships
- Generic marine engineering (ports, offshore wind, hydrography, climate)
- Defense / military applications (see §8)

Test: if a sentence reads naturally with "marine engineering" or "marine robotics" replacing "underwater robotics", rewrite it.

---

## 3. Anchor Sentence

**EN:** The ocean simulator for underwater robotics.
**ZH:** 面向水下机器人的海洋仿真基础设施。

Appears as: site `<title>`, homepage H1, deck cover, root `README.md` first line, social bios. Repetition is the point. Do not freshen for each surface.

---

## 4. Concept Hierarchy

| Tier | Concept (EN) | Concept (ZH) | Where it appears | Promotion threshold |
|------|--------------|--------------|------------------|---------------------|
| 1 | ocean simulator | 海洋仿真基础设施 | public default, repeat freely | always public |
| 2 | ocean world model | 海洋世界模型 | one click below the homepage; deck "architecture" slide | promote to homepage when a shipped demo shows the learned layer improving simulator fidelity or transfer, with a public benchmark against a relevant non-learned baseline |
| 3 | ocean foundation model | 海洋基础模型 | research notes / private deck depth only, with explicit distinction from tier 2 | promote outward only with (a) data scale evidence, (b) model trained at meaningful parameter scale, (c) distribution leverage |

Why this hierarchy: a simulator is a concrete product primitive. A world model is a deeper technical claim. A foundation model is a status claim. Promote each tier outward only when evidence catches up.

---

## 5. The Enemy

Field testing as the bottleneck.

**EN:** Underwater robots still learn in the most expensive test environment on earth: the real ocean.
**ZH:** 水下机器人,仍然在地球上最昂贵的测试环境中学习:真实海洋。

Canonical opener for vision pages and the deck enemy slide. Quote verbatim, do not paraphrase.

---

## 6. Derivation Rules

All downstream copy (website pages, deck slides, READMEs, talks, social posts, release notes, job pages, meta descriptions) derives from §§1–5 and §§7–8.

**May adapt:** length, grammar, CTA, surface-specific context, lead/hook, illustrative examples.

**May not change:** category (§1), scope (§2), anchor sentence (§3), tier hierarchy (§4), enemy line (§5), required lexicon (§7), banned framings (§8).

**Translation rule:** EN and ZH are both canonical. Neither is a translation of the other. Both are written for native readers and must convey the same category, scope, anchor, and enemy. Bilingual asymmetry is a release-blocking bug.

**Audience priority per surface:**

| Surface | Primary reader | Reference voice |
|---------|----------------|-----------------|
| Homepage | technical followers + investors | A3 |
| About page | investors + hires | A4 |
| Vision / `/vision` | investors | A6 |
| Root `README.md` first line | mixed | A1 (the anchor) |
| Root `README.md` opening paragraph | mixed | A3 or A4 |
| Python package README (`oceanscale/`) | developers / researchers | A4 EN |
| GitHub repo description (≤350 chars) | developers | A2 (truncated to fit) |
| Fundraising deck | investors | A1 cover, A5 architecture, A6 closing |
| Career / hiring | future hires | A4 |
| Press / blog | journalists, community | varies, A6 for narrative pieces |
| Social bio | mixed | A1 |

**Visual identity:** infrastructure-first. No ocean romance, no submarine hero fantasy, no cinematic blue abyss as the main idea. See `DESIGN.md` for visual specifics.

---

## 7. Required Lexicon

Use exactly. Where two forms are listed for ZH, the first is primary; the second is allowed only in the noted context.

| Concept | EN canonical | ZH canonical | ZH alt |
|---------|--------------|--------------|--------|
| Company | OceanScale | OceanScale | — |
| Primitive | ocean simulator | 海洋仿真基础设施 | 海洋仿真器 (short form, product/code contexts only) |
| Category | AI-native simulation infrastructure | AI 原生仿真基础设施 | — |
| Audience | underwater robotics / underwater robots | 水下机器人 | — |
| Platforms | AUVs and ROVs | AUV 与 ROV | — |
| Tier 2 | ocean world model | 海洋世界模型 | — |
| Loop verb | train, test, and validate | 训练、测试与验证 | — |
| Loop endpoint | before deployment | 部署前 | 下海之前 (narrative contexts) |
| Enemy noun | field testing / field trials | 现场试验 | 现场测试 |

---

## 8. Banned Words & Framings

### Scope drift (banned)
- `marine engineering` / `海洋工程`
- `marine robotics` as primary scope noun (allowed in body copy only after AUV/ROV context exists; never in headlines or canonical lines)
- `ocean robotics` / `海洋机器人` as primary scope noun
- `autonomous ocean systems` / `自主海洋系统`
- `autonomous marine systems`
- `offshore autonomy` as primary scope noun

### Category drift (banned)
- `ocean foundation model` as a headline noun (deck/research depth only, with explicit distinction from world model)
- `digital twin` / `数字孪生`
- `GPU + HPC` stacked (pick one; GPU is enough)
- `embodied AI` / `具身智能`
- `world's first` / `全球首个`

### Analogy fatigue (banned)
- Tesla / autonomous-driving analogy lazily applied. If used at all, lead with the *differences* (ocean economics, regulation, data sparsity).

### Reputation / legal (banned in public-facing surfaces and fundraising materials)
- Defense / military framing — `defense`, `military`, `naval`, `防务`, `国防`, `军事`, `军用`
- `naval autonomy`
- `maritime security`
- `surveillance`
- `dual-use`

### Style (reviewer judgment, not grep-enforceable)
- `mission` used in military-coded ways (mission *task* in the technical sense is fine; `mission set` / `mission profile` should be flagged)
- `next-gen` / `下一代` — limit to once per page
- `multi-agent coordination` / `多智能体协同` — fine in technical roadmap; banned from public homepage and deck cover

---

## 9. Open Positioning Questions

Tracked here. Resolve before the next revision.

### 9.1 When does underwater robotics expand to marine robotics?

Current answer: not yet. Public positioning remains underwater robotics.

Revisit when all three hold:
- the same simulator primitive clearly serves underwater and surface systems,
- at least two serious commercial conversations pull toward USVs or broader marine autonomy,
- the category can expand without reviving "marine engineering" or defense-coded language.

### 9.2 When does ocean world model promote to homepage (tier 2 → tier 1)?

Current answer: only when a shipped demo shows the learned layer improving simulator fidelity or transfer, with a public benchmark against a relevant non-learned baseline. Until then, the world model stays one click below.

### 9.3 Chinese company name

Currently English-only (`沧渊` retired). If a new Chinese name is chosen, add it to §7 and update all surfaces atomically.

---

## Changelog

- **2026-05-22 v0** — Initial draft. Anchor: "ocean simulator for underwater robotics." Six voice versions inline.
- **2026-05-22 v1** — Post cx review. Fixed EN/ZH drift in enemy line (Marine→Underwater). Softened tier-2 tech claim ("classical fluid physics cannot" removed). Generic-ized the `$300k vehicle` reference. Replaced "do not write a seventh" with a derivation rule. Cut roadmap from positioning law (moves to a future `ROADMAP.md`). Cut cross-model-review process bloat. Added §6 derivation/translation/audience/visual rules. Added §4 tier promotion thresholds. Added §9 open positioning questions. Expanded §8 banned terms. Moved v4–v6 to Appendix A as derived examples.
- **2026-05-22 v1.1** — Post second cx review. Fixed README surface mapping ambiguity (split into 4 distinct surfaces in §6). Generalized tier-2 promotion threshold beyond "learned dynamics" to any "learned layer improving simulator fidelity or transfer." Recategorized `offshore autonomy` from defense to scope drift (correct classification — offshore inspection/wind/subsea ops are civilian). Softened defense-policy scope from "banned everywhere" to "public-facing surfaces and fundraising materials" (matches original public-copy-only policy intent). A5 "marine environments" → "ocean environments." Changed status wording from "Locked" to "Canonical v1.1."

---

## Appendix A — Canonical Voice (examples derived from §§1–8)

The following six versions are reference implementations of the rules above. They are not law. The law is upstream. Surface-specific adaptations are allowed within §6.

### A1 — Title

**EN:** OceanScale — The ocean simulator for underwater robotics.
**ZH:** OceanScale — 面向水下机器人的海洋仿真基础设施。

### A2 — Tagline

**EN:** OceanScale builds AI-native simulation infrastructure where underwater robots learn, test, and validate before deployment.
**ZH:** OceanScale 构建 AI 原生仿真基础设施,让水下机器人在部署前完成学习、测试与验证。

### A3 — Short (homepage hero)

**EN:**
> Underwater robots still learn in the most expensive test environment on earth: the real ocean.
>
> OceanScale builds the simulator where they train, test, and validate before they dive.

**ZH:**
> 水下机器人,仍然在地球上最昂贵的测试环境中学习:真实海洋。
>
> OceanScale 构建海洋仿真基础设施,让水下机器人在下海之前完成训练、测试与验证。

### A4 — Standard (About / intro deck)

**EN:**
> OceanScale builds AI-native simulation infrastructure for underwater robotics.
>
> Today, underwater robots learn through field trials in the most expensive test environment on earth. OceanScale replaces that loop with a high-fidelity ocean simulator: a virtual ocean where AUVs and ROVs train, test, and validate before deployment. Beneath the simulator sits an ocean world model that learns from data and physics together.

**ZH:**
> OceanScale 构建面向水下机器人的 AI 原生仿真基础设施。
>
> 今天的水下机器人主要依赖真实海洋中的现场试验来学习,而真实海洋是地球上最昂贵、不确定性最高的测试环境。OceanScale 用高保真海洋仿真器替换这一循环:AUV 与 ROV 在虚拟海洋中完成训练、测试与验证,再下海部署。仿真器之下,是一个结合数据与物理学习海洋行为的海洋世界模型。

### A5 — Technical (deck / engineering talks)

**EN:**
> OceanScale builds AI-native simulation infrastructure for underwater robotics.
>
> The architecture has two layers: a high-fidelity ocean simulator above, and a learned ocean world model below.
>
> The simulator runs sensor-accurate ocean environments, vehicle dynamics, and task scenarios at scale on GPU. The world model captures ocean behavior learned from data and physics, complementing the simulator's first-principles core. Together they form a training loop in which underwater robots learn, test, and transfer to the real ocean, without expensive field trials as the unit of iteration.

**ZH:**
> OceanScale 构建面向水下机器人的 AI 原生仿真基础设施。
>
> 架构分两层:上层是高保真海洋仿真器,下层是学习而来的海洋世界模型。
>
> 仿真器在 GPU 上规模化运行传感器级精度的海洋环境、机器人动力学与任务场景。世界模型从数据与物理中学习海洋行为,与仿真器的第一性原理形成互补。两者组成一个训练闭环,使水下机器人能够在虚拟海洋中学习、测试、并迁移到真实海洋,而不再需要以现场试验作为迭代单位。

### A6 — Detailed (long-form vision page)

**EN:**
> **Underwater robots still learn in the most expensive test environment on earth: the real ocean.**
>
> Most of the iteration loop runs in the ocean itself. Ship time, weather windows, crew costs, and the risk of losing an expensive vehicle to a current set the pace of progress. Underwater robotics moves slowly because the test environment is slow.
>
> **OceanScale builds AI-native simulation infrastructure for underwater robotics.**
>
> The product is an ocean simulator: a GPU-native virtual ocean where AUVs and ROVs train, test, and validate before they ever touch saltwater. Beneath the simulator sits an ocean world model that learns from data and physics together.
>
> We believe the next generation of underwater robots will not be programmed against static rules and limited field tests. They will learn against a virtual ocean that is faster, cheaper, and more controllable than the real one.
>
> **OceanScale is building that ocean.**

**ZH:**
> **水下机器人,仍然在地球上最昂贵的测试环境中学习:真实海洋。**
>
> 迭代回路的大部分都跑在真实海洋里。船时、天气窗口、人员成本,以及一台昂贵机器人随时可能被洋流带走的风险,决定了水下机器人的进步速度。这个行业走得慢,因为它的测试场走得慢。
>
> **OceanScale 构建面向水下机器人的 AI 原生仿真基础设施。**
>
> 产品是一个海洋仿真器:一个 GPU 原生的虚拟海洋,AUV 与 ROV 在接触海水之前完成训练、测试与验证。仿真器之下,是一个结合数据与物理学习海洋行为的海洋世界模型。
>
> 我们相信,下一代水下机器人不会再依赖静态规则与有限的现场测试。它们会在一个比真实海洋更快、更便宜、更可控的虚拟海洋中学习。
>
> **OceanScale 正在构建那片海洋。**
