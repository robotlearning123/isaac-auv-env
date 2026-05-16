---
title: OceanScale (沧渊) Landing Page v0.1 — Design Spec
status: approved
date: 2026-05-15
author: Claude Opus 4.7 + @wangcongrobot
sub_project: F (Landing page) of OceanScale startup work
implementation_status: pending
---

# OceanScale (沧渊) Landing Page v0.1 — Design Spec

## 0. Context

OceanScale (沧渊) is an **AI-native simulation infrastructure for marine robotics**. This document specifies the v0.1 landing page — the first public-facing artifact for the project.

**Reference / inspiration:** Lightwheel AI (Beijing-based, Newton physics partner, "vertical infra on NVIDIA" thesis, focused on humanoid robotics). OceanScale targets the equivalent role for underwater robotics.

**Positioning (locked 2026-05-15):**
> AI-native simulation infrastructure accelerating the future of underwater robotics. GPU-native simulation + agent training platform for marine robotics research, industry, and defense.

**Big framing (user 2026-05-15):** "Fully utilise the latest AI/agent/robot research, speedup ocean/marine robotics research/application, for the next future road to future underwater robotics research, large." → Platform play, not narrow vertical.

This landing page is **sub-project F of 7** in the OceanScale startup work:
- A. Un-park strategy memory (DONE 2026-05-15)
- B. Ocean review (code + STACK + SURVEY + RISKS audit) — pending
- C. POSITIONING.md (对外版定位文档) — pending
- D. BUSINESS.md (TAM / GTM / 团队 / 里程碑) — pending
- E. Pitch artifacts (one-pager + pitch outline) — pending
- **F. Landing page (this doc)** — design approved, implementation pending
- G. Hosting + domain (Cloudflare Pages + oceanscale.cn) — pending

## 1. Goals & Non-Goals

### Goals (v0.1)
- Convey OceanScale's positioning to first visitors (researchers, industry, design partners, VCs)
- Showcase the BlueROV2 11,435 FPS technical proof point
- Capture design-partner inquiries via email
- Bilingual (zh default, en at `/en/`) for Beijing-based audience + international research/GitHub community
- Performant: Lighthouse Performance ≥ 90, Accessibility ≥ 95, SEO ≥ 95
- Deployable to Cloudflare Pages with one command

### Non-Goals (v0.1)
- ❌ Blog / documentation site (later sub-project)
- ❌ User authentication / dashboard
- ❌ Newsletter signup database (mailto: link suffices)
- ❌ Dynamic demos / WebGL interactive sim (static videos only)
- ❌ Public GitHub link (repo is **private** — "Contact" replaces it in CTAs)
- ❌ ICP 备案 (only required if oceanscale.cn is registered + hosted via 国内云)

## 2. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | **Astro 4** | Zero-JS by default, Islands Architecture, native i18n routing. Best-in-class for content-heavy marketing sites. |
| Styling | **Tailwind CSS 3** | Utility-first, no custom CSS bloat, design tokens via config |
| Adapter | **@astrojs/cloudflare** | Native Cloudflare Pages deployment, edge functions if ever needed |
| Hosting | **Cloudflare Pages** | Free tier, edge CDN, China-accessible, Astro/Cloudflare same-team integration |
| Package mgr | **pnpm** | 3x faster than npm, disk-efficient |
| Fonts | Inter + Noto Sans SC + JetBrains Mono | en + zh + mono via Google Fonts CDN |
| Build target | Static HTML | No SSR; portable to any host |

**Why Astro over Next.js:** Next.js hydrates everything → ~100KB JS even for static pages. Astro ships near-0KB by default. Every KB of JS is TTI cost.

**Why Cloudflare Pages:** 中国可达性 (CF 在国内有节点,Vercel/Netlify 偶发不稳),Astro 与 CF 同生态,域名/DNS/CDN 一站式。

## 3. File Architecture

```
46-marine/website/
├── README.md                    # 本地开发 + 部署说明
├── astro.config.mjs             # i18n + Cloudflare adapter
├── tailwind.config.mjs          # 品牌色板 + 字体
├── package.json
├── tsconfig.json
├── wrangler.toml                # Cloudflare Pages 配置
├── .gitignore                   # node_modules/, dist/, .astro/, .wrangler/
├── public/
│   ├── favicon.svg
│   ├── og-image.png             # 1200x630 社交分享卡
│   ├── videos/
│   │   └── bluerov2-demo.mp4    # 11,435 FPS 训练实拍
│   └── images/
│       └── logo.svg
└── src/
    ├── pages/
    │   ├── index.astro          # / (中文首页)
    │   └── en/index.astro       # /en/ (英文首页)
    ├── layouts/
    │   └── BaseLayout.astro     # 共享 <head> + SEO meta + Nav + Footer
    ├── components/
    │   ├── Nav.astro            # 含语言切换
    │   ├── Hero.astro
    │   ├── Why.astro            # 5-problem 列表
    │   ├── What.astro
    │   ├── Demos.astro
    │   ├── Benchmarks.astro
    │   ├── Contact.astro
    │   └── Footer.astro
    ├── content/                 # 文案 (与代码严格分离)
    │   ├── config.ts            # content collection schema
    │   ├── zh/
    │   │   ├── hero.md
    │   │   ├── why.md
    │   │   ├── what.md
    │   │   ├── demos.md
    │   │   ├── benchmarks.md
    │   │   └── contact.md
    │   └── en/
    │       └── ...              (相同 6 文件)
    ├── i18n/
    │   ├── ui.ts                # 导航/按钮 UI 字符串
    │   └── utils.ts             # locale 检测 + cross-locale URL helper
    └── styles/
        └── global.css           # Tailwind directives + 自定义 base
```

**Key principle:** 文案 (`content/`) 与组件 (`components/`) 严格分离 — 修改 hero copy 只编辑 `content/zh/hero.md`,不动 .astro 代码。便于以后合伙人/marketing 改文案。

## 4. Content Strategy

### Tone
- 严肃技术 + 行业级气场 (不口语化,不卖萌)
- 数字密度高 (具体 FPS、具体硬件、具体节省)
- 中文不直译英文 — 各自符合该语言的科技语境

### Hero

**zh:**
- **H1:** 加速水下机器人的下一个十年
- 副: GPU-native 仿真 + 智能体训练平台,服务海洋机器人的研究、产业与防务
- CTAs: 「查看演示」 + 「联系我们」

**en:**
- **H1:** Accelerating the next decade of underwater robotics
- Sub: GPU-native simulation + agent training infrastructure for marine robotics research, industry, and defense
- CTAs: "See demos" + "Get in touch"

### Why — multi-problem framing (locked decision)

5 个问题,**真实出海测试昂贵** 放第 1 位:

1. **真实海上测试昂贵** (Real ocean trials are prohibitively expensive)
   - 出海一天: ¥10,000–¥100,000+ (船时 + 人员 + 设备折旧)
   - 设备损坏率高 (海水腐蚀、压力、缠绕、丢失)
   - 环境不可重复 (洋流、能见度、生物干扰)
   - 实验进度受天气/季节限制
   - **结论:** 没有高保真仿真,海洋机器人迭代速度被自然界锁死

2. **CFD 太慢 — RL 训练不可达** (CFD too slow for RL training)
   - 高保真 CFD: 1 秒物理需 1–100 小时计算
   - RL 训练所需 1M+ episodes 数量级不可达
   - 即使 GPU 加速的 CFD 也无法接近 real-time × 千倍

3. **现有水下仿真器非 GPU-native** (Legacy sims waste modern GPUs)
   - Gazebo / UWSim / HoloOcean: 单线程 CPU,几十–几百 FPS
   - 无法充分利用 RTX 5090 / H100 算力
   - 多智能体 / 多场景并行能力差

4. **Sim-to-real gap** (Fluid fidelity gap)
   - 流体动力学保真度不足 → 策略迁移失败
   - 缺少有原理的 sim-to-real 校准方法
   - 传感器仿真 (sonar / underwater camera) 也常被简化

5. **缺少 AI-native 接口** (Missing AI-native interfaces)
   - gym/gymnasium 适配缺失或质量低
   - 多智能体场景支持差
   - 多模态海洋数据集 (sonar/optical/IMU) 不公开

### What — 能力 (实数 + 实测)

- **11,435 FPS** BlueROV2 单环境训练 (RTX 5090, 单卡, 已实测 — 来源 `install_log/56_r11_commit.txt`)
- **90 M env-steps/s** Fossen 多环境 throughput @ 8192 parallel envs (RTX 5090, 900× 目标 — 来源 commit `5cc3935` T2.8 benchmark, W2 完成时实测)
- **Newton 物理** (Anthropic + NVIDIA + Lightwheel + Apple 联合维护) + **Isaac Sim 6** 渲染
- 自研 **Fossen 6-DoF + SPH 流体 kernel**
- 即插即用 **gym/gymnasium** + RL 库 (rl_games, stable-baselines3, RSL-RL)
- **多智能体协同** + multi-robot scene
- 真实海洋资产库 (船/AUV/ROV/传感器) + procedural ocean scene generation

### Demos
- 视频 1: BlueROV2 RL 训练 (11,435 FPS, 实测数据)
- 视频 2: (待录) 多 AUV 编队 — 标 "Coming v0.3"
- 视频 3: (待录) SPH 流体可视化 — 标 "Coming v0.3"

v0.1 上线时只有视频 1 是真实数据。

### Benchmarks (对比表)

| 指标 | OceanScale | UWSim | HoloOcean | Gazebo |
|---|---|---|---|---|
| Single-env FPS (RTX 5090) | **11,435** | ~200 | ~500 | ~100 |
| Multi-env throughput @ 8192 envs | **90 M env-steps/s** | ❌ | ❌ | ❌ |
| Multi-env parallel scaling | ✅ 8192+ | ❌ | 部分 | 部分 |
| GPU-native physics | ✅ Newton | ❌ | ❌ | ❌ |
| RL gym interface | ✅ 原生 | 第三方 | 第三方 | 第三方 |
| SPH fluid kernel | ✅ | ❌ | ❌ | ❌ |
| Fossen 6-DoF | ✅ | ✅ | 部分 | ❌ |
| ROS 2 integration | ⏳ v0.3 | ✅ | ✅ | ✅ |

**注:** 数据需在上线前由 sub-project B (review) 核对,避免 inflated numbers。 其他工具的 FPS 来自其文档,需各自源头确认。

### Contact

- 业务咨询: business@oceanscale.cn (邮箱待注册)
- 技术咨询: tech@oceanscale.cn
- Design partner 申请: design-partners@oceanscale.cn
- **代码访问:** 仅向 design partners 开放,通过 业务邮箱 申请。GitHub repo 私有 (decision 2026-05-15)
- 无公开 GitHub link (stealth mode)

## 5. Visual Design Tokens (Lightwheel-style + 深海意象)

### 色板
- Background: 深海蓝-黑渐变 `#000510` → `#0A1628`
- Primary (CTA, headings): 海波青 `#00C6FF`
- Accent (高光动效, sparingly): 极光绿 `#39FF14`
- Text primary: `#E8EEF2`
- Text secondary: `#A0B0C0`
- Border / divider: `#1F3047`

### 字体
- Display + body (en): **Inter** (Google Fonts CDN, subset Latin)
- Display + body (zh): **Noto Sans SC** (Google Fonts CDN, subset CJK)
- Mono (FPS 数字, code): **JetBrains Mono**

### 动效 (轻量,无重型库)
- Hero 视频自动播放 + 缓慢 zoom + parallax (CSS transforms)
- 滚动揭示 (原生 Intersection Observer)
- FPS 数字 count-up (原生 requestAnimationFrame)
- 按钮 hover: `translateY(-2px)` + glow shadow

### Layout / spacing
- Mobile-first responsive
- Max content width: 1280px
- Section padding: 6rem vertical (desktop) / 3rem (mobile)
- Component gap base: 1.5rem

## 6. i18n Strategy

### 路由配置

```js
// astro.config.mjs
export default defineConfig({
  i18n: {
    defaultLocale: 'zh',
    locales: ['zh', 'en'],
    routing: { prefixDefaultLocale: false },
  },
  output: 'static',
  adapter: cloudflare(),
});
```

- `/` → 中文首页 (无前缀,SEO 友好,中文用户为主)
- `/en/` → 英文首页
- 顶部右上角 「中 / EN」 切换 (保持 path 对应)

### 内容对称
- zh/en sections 数必须一致 (6 each)
- 中文不是英文逐字翻译 — 各自符合本语言的科技语境

## 7. Deployment

### Workflow
```bash
cd 46-marine/website
pnpm install                      # 安装依赖
pnpm dev                          # 本地 http://localhost:4321
pnpm build                        # 输出到 dist/
wrangler pages deploy dist/ \
  --project-name=oceanscale-web   # 部署到 Cloudflare Pages
```

### 国内可达性
- Cloudflare 国内有节点 (近年改善)
- 备选: 阿里云 OSS+CDN (国内 100% 快,海外慢)
- 推荐: 先 CF,如果国内访客慢再加阿里镜像

### 域名 (sub-project G)
- 候选: `oceanscale.cn` / `oceanscale.ai` / 双域名
- ICP 备案: 仅 `oceanscale.cn` 走国内云时需要
- v0.1 先用 Cloudflare Pages 临时域名 `oceanscale-web.pages.dev`

## 8. Testing / Verify Before Ship

### Lighthouse (上线前必过)
- Performance ≥ 90
- Accessibility ≥ 95
- Best Practices ≥ 90
- SEO ≥ 95

### Manual checklist
- [ ] zh ↔ en 跳转保持 path 对应
- [ ] 移动端首屏 (Chrome DevTools Slow 4G) < 2s
- [ ] 视频 lazy load + poster image fallback
- [ ] Noto Sans SC fallback 工作
- [ ] OG card 在 WhatsApp / Twitter / Slack / WeChat 预览
- [ ] favicon 跨浏览器
- [ ] 全部 CTA mailto: 打开邮件客户端
- [ ] WCAG AA 文字对比度

## 9. 工时估计

| 阶段 | 估计 | 备注 |
|---|---|---|
| 1. Scaffold (folder + Astro init + Tailwind + i18n) | 30 min | `pnpm create astro@latest` + config |
| 2. BaseLayout + Nav + Footer + Lang switcher | 1 hr | 骨架 |
| 3. 6 sections 组件 (静态版,占位文案) | 2 hr | |
| 4. 中英文案 + Markdown content collections | 2 hr | 5-problem Why 是重头 |
| 5. 视觉抛光 (色彩、间距、动效) | 1.5 hr | Lightwheel-style |
| 6. Cloudflare Pages 部署 + Lighthouse 优化 | 30 min | wrangler pages deploy |
| **合计 v0.1** | **~7.5 hr (1-2 天)** | |

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Benchmark 数据被 over-claim | 信誉损失 (高) | sub-project B (review) 上线前核对每个数字 |
| 中英文案不对称 → SEO 损失 | 中 | i18n schema 锁定 6 sections,内容审计前置 |
| Cloudflare 国内访问慢 | 中 | 监测国内 latency,必要时加阿里云镜像 |
| 域名未注册 → 不能上线生产 | 高 | 用 `*.pages.dev` 临时域名先发布,域名注册并行 |
| Lightwheel 视觉抄袭风险 | 低 | "Lightwheel-style" 是品类参考,不照抄具体元素 |
| 私有 repo 与 "open" 形象冲突 | 低 | 文案明确 "design partner only" 定位 |
| 出海测试昂贵的说法被海洋研究者反驳 | 低-中 | 引用 NOAA / 海洋所公开数据支撑 |

## 11. Future (out of v0.1 scope)

- **F.2 Docs site** — 触发: sub-project C 完成后
- **F.3 Blog** — 触发: 有 3+ 篇内容
- **F.4 Interactive WebGL demo** — 触发: viewer 准备好
- **F.5 Press kit** — 触发: 准备 launch
- **F.6 Customer dashboard** — 触发: 有 design partners 在用

## 12. Approval & Next Steps

- Design approved by @wangcongrobot 2026-05-15 (this session)
- Locked: brand, tech stack, sections, visual ref, i18n, hosting, GitHub privacy, multi-problem Why
- **Next:** spec self-review (inline check below) → user re-review → invoke `superpowers:writing-plans` skill to produce step-by-step implementation plan

---

## Appendix A. Spec Self-Review (per brainstorming skill)

| Check | Result |
|---|---|
| Placeholder scan (TBD / TODO / vague) | ✅ Pass — all sections have concrete content; "Coming v0.3" is intentional |
| Internal consistency | ✅ Pass — architecture matches feature descriptions, sections count consistent (6) |
| Scope check | ✅ Single implementation plan size (~7.5 hr), not requiring further decomposition |
| Ambiguity check | ✅ Pass — Hero CTA explicitly "Contact" not GitHub; Why has 5 numbered problems |

## Appendix B. Decisions Locked in This Session (2026-05-15)

1. Folder: `46-marine/website/`
2. Brand: OceanScale (en) / 沧渊 (zh)
3. Tech: Astro 4 + Tailwind + Cloudflare Pages
4. i18n: cn/en, zh default no prefix, en at /en/
5. Sections: Hero + Why + What + Demos + Benchmarks + Contact (6)
6. Visual ref: Lightwheel-style
7. GitHub repo: PRIVATE, no public link in CTAs
8. Why structure: multi-problem (5), ocean test cost #1
9. Hero CTAs: "查看演示" / "See demos" + "联系我们" / "Get in touch"
10. Hosting: Cloudflare Pages first, optional 阿里云镜像 later
11. Benchmarks: single-env 11,435 FPS (BlueROV2) + multi-env 90 M env-steps/s @ 8192 envs (Fossen kernel, T2.8 benchmark 2026-05-15, commit `5cc3935`)
