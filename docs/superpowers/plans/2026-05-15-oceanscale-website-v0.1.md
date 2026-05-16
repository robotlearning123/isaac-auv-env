# OceanScale (沧渊) Landing Page v0.1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and deploy the OceanScale (沧渊) bilingual marketing landing page to Cloudflare Pages, per design spec `docs/superpowers/specs/2026-05-15-oceanscale-landing-page-design.md`.

**Architecture:** Astro 4 static site, Tailwind for styling, native i18n routing (zh default at `/`, en at `/en/`), content stored as Markdown separated from components, Cloudflare Pages for hosting.

**Tech Stack:** Astro 4, Tailwind CSS 3, `@astrojs/tailwind`, `@astrojs/cloudflare`, pnpm, wrangler CLI.

**Branch:** `feat/oceanscale-website-v0.1.0` (already created)

---

## Pre-flight (manual, one-time)

Before starting tasks, verify on the developer machine:

```bash
pnpm --version          # Expected: ≥ 8.0.0 (else: npm install -g pnpm)
node --version          # Expected: ≥ 18.0.0
pnpm dlx wrangler --version  # Should print version (or installs wrangler)
```

**Manual prerequisites (cannot be automated):**
1. Cloudflare account at dash.cloudflare.com (free tier OK)
2. BlueROV2 demo video MP4 — if absent, Task 8 uses a placeholder

---

## File Structure (locked from spec §3)

```
46-marine/website/
├── README.md, astro.config.mjs, tailwind.config.mjs, package.json, tsconfig.json
├── wrangler.toml, .gitignore
├── public/{favicon.svg, og-image.png, videos/, images/}
└── src/
    ├── pages/index.astro              (zh, default /)
    ├── pages/en/index.astro           (en, /en/)
    ├── layouts/BaseLayout.astro
    ├── components/{Nav,Hero,Why,What,Demos,Benchmarks,Contact,Footer}.astro
    ├── content/{config.ts, zh/*.md, en/*.md}
    ├── i18n/{ui.ts, utils.ts}
    └── styles/global.css
```

---

## Task 1: Scaffold Astro 4 minimal project

**Files:**
- Create: `46-marine/website/` (entire folder via `pnpm create astro`)
- Create: `46-marine/website/.gitignore` (append)

- [ ] **Step 1: Scaffold**

```bash
cd /home/robot/workspace/46-marine
pnpm create astro@latest website --template minimal --typescript strict --no-install --no-git --skip-houston
```

Expected: Folder `website/` with `astro.config.mjs`, `package.json`, `tsconfig.json`, `src/pages/index.astro`. No errors.

- [ ] **Step 2: Install deps**

```bash
cd /home/robot/workspace/46-marine/website
pnpm install
```

Expected: ~50-80 packages installed. No errors.

- [ ] **Step 3: Smoke-test dev server**

```bash
pnpm dev &
sleep 3
curl -sI http://localhost:4321/ | head -1
kill %1
```

Expected: `HTTP/1.1 200 OK`.

- [ ] **Step 4: Append .gitignore**

Append to `website/.gitignore`:

```
.wrangler/
*.log
.DS_Store
```

- [ ] **Step 5: Commit**

```bash
cd /home/robot/workspace/46-marine
git add website/
git commit -m "feat(website): scaffold Astro 4 minimal project"
```

---

## Task 2: Add Tailwind + Cloudflare integrations

**Files:**
- Modify: `website/astro.config.mjs` (integrations added by `astro add`)
- Modify: `website/package.json` (deps added)
- Create: `website/tailwind.config.mjs` (created by `astro add tailwind`)
- Create: `website/src/styles/global.css`

- [ ] **Step 1: Add Tailwind integration**

```bash
cd /home/robot/workspace/46-marine/website
pnpm dlx astro add tailwind --yes
```

Expected: `astro.config.mjs` includes `import tailwind from '@astrojs/tailwind'` + `integrations: [tailwind()]`.

- [ ] **Step 2: Add Cloudflare adapter**

```bash
pnpm dlx astro add cloudflare --yes
```

Expected: `astro.config.mjs` includes `adapter: cloudflare()`. **Note:** keep `output: 'static'` (don't switch to SSR).

- [ ] **Step 3: Replace `tailwind.config.mjs` with brand tokens**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-deep': '#000510',
        'bg-deep-2': '#0A1628',
        'primary': '#00C6FF',
        'accent': '#39FF14',
        'text-primary': '#E8EEF2',
        'text-secondary': '#A0B0C0',
        'border-deep': '#1F3047',
      },
      fontFamily: {
        sans: ['Inter', 'Noto Sans SC', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      maxWidth: { 'content': '1280px' },
    },
  },
  plugins: [],
};
```

- [ ] **Step 4: Create `src/styles/global.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html { @apply text-text-primary; }
  body {
    @apply min-h-screen font-sans;
    background: linear-gradient(180deg, #000510 0%, #0A1628 100%);
    background-attachment: fixed;
  }
  code, pre { @apply font-mono; }
}

@layer components {
  .btn-primary {
    @apply inline-block px-6 py-3 bg-primary text-bg-deep font-semibold rounded-md
           transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg
           hover:shadow-primary/30;
  }
  .btn-secondary {
    @apply inline-block px-6 py-3 border border-primary text-primary font-semibold rounded-md
           transition-all duration-200 hover:bg-primary/10;
  }
  .section { @apply max-w-content mx-auto px-6 py-24 md:py-32; }
}
```

- [ ] **Step 5: Verify build**

```bash
cd /home/robot/workspace/46-marine/website
pnpm build
```

Expected: Build succeeds, `dist/` created.

- [ ] **Step 6: Commit**

```bash
cd /home/robot/workspace/46-marine
git add website/
git commit -m "feat(website): add Tailwind + Cloudflare adapter + brand tokens"
```

---

## Task 3: Configure i18n routing

**Files:**
- Modify: `website/astro.config.mjs`
- Create: `website/src/pages/en/index.astro` (placeholder)
- Modify: `website/src/pages/index.astro` (placeholder)

- [ ] **Step 1: Add i18n config to `astro.config.mjs`**

Edit `astro.config.mjs` — ensure config includes:

```js
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  output: 'static',
  adapter: cloudflare({ imageService: 'compile' }),
  integrations: [tailwind({ applyBaseStyles: false })],
  i18n: {
    defaultLocale: 'zh',
    locales: ['zh', 'en'],
    routing: { prefixDefaultLocale: false },
  },
});
```

**Note:** `applyBaseStyles: false` so our `global.css` controls base styles (avoid Tailwind preflight conflicts with our gradient bg).

- [ ] **Step 2: Replace `src/pages/index.astro` placeholder**

```astro
---
import '../styles/global.css';
---
<html lang="zh">
  <head>
    <meta charset="UTF-8" />
    <title>OceanScale 沧渊 — Coming soon</title>
  </head>
  <body>
    <h1 class="text-4xl p-12">沧渊 — 中文首页占位</h1>
  </body>
</html>
```

- [ ] **Step 3: Create `src/pages/en/index.astro` placeholder**

```astro
---
import '../../styles/global.css';
---
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>OceanScale — Coming soon</title>
  </head>
  <body>
    <h1 class="text-4xl p-12">OceanScale — English placeholder</h1>
  </body>
</html>
```

- [ ] **Step 4: Verify both routes**

```bash
cd /home/robot/workspace/46-marine/website
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "中文首页占位"
curl -s http://localhost:4321/en/ | grep "English placeholder"
kill %1
```

Expected: Both grep matches print non-empty lines.

- [ ] **Step 5: Commit**

```bash
cd /home/robot/workspace/46-marine
git add website/
git commit -m "feat(website): configure i18n routing zh/en with placeholders"
```

---

## Task 4: BaseLayout + Nav + Footer + i18n UI strings

**Files:**
- Create: `website/src/i18n/ui.ts`
- Create: `website/src/i18n/utils.ts`
- Create: `website/src/layouts/BaseLayout.astro`
- Create: `website/src/components/Nav.astro`
- Create: `website/src/components/Footer.astro`

- [ ] **Step 1: Create `src/i18n/ui.ts`**

```ts
export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'zh';

export const ui = {
  zh: {
    'nav.what': '能力',
    'nav.demos': '演示',
    'nav.benchmarks': '基准',
    'nav.contact': '联系',
    'cta.demos': '查看演示',
    'cta.contact': '联系我们',
    'lang.other': 'EN',
    'footer.copyright': '© 2026 OceanScale (沧渊). 保留所有权利。',
  },
  en: {
    'nav.what': 'Platform',
    'nav.demos': 'Demos',
    'nav.benchmarks': 'Benchmarks',
    'nav.contact': 'Contact',
    'cta.demos': 'See demos',
    'cta.contact': 'Get in touch',
    'lang.other': '中',
    'footer.copyright': '© 2026 OceanScale. All rights reserved.',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
```

- [ ] **Step 2: Create `src/i18n/utils.ts`**

```ts
import { ui, defaultLang, type Lang, type UIKey } from './ui';

export function getLangFromUrl(url: URL): Lang {
  const [, lang] = url.pathname.split('/');
  if (lang in ui) return lang as Lang;
  return defaultLang;
}

export function useTranslations(lang: Lang) {
  return function t(key: UIKey): string {
    return ui[lang][key] ?? ui[defaultLang][key];
  };
}

export function getOtherLangUrl(currentLang: Lang, pathname: string): string {
  if (currentLang === 'zh') return '/en' + (pathname === '/' ? '/' : pathname);
  return pathname.replace(/^\/en/, '') || '/';
}
```

- [ ] **Step 3: Create `src/components/Nav.astro`**

```astro
---
import { getLangFromUrl, useTranslations, getOtherLangUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const t = useTranslations(lang);
const otherUrl = getOtherLangUrl(lang, Astro.url.pathname);
---
<nav class="sticky top-0 z-50 backdrop-blur-md bg-bg-deep/80 border-b border-border-deep">
  <div class="max-w-content mx-auto px-6 py-4 flex items-center justify-between">
    <a href={lang === 'zh' ? '/' : '/en/'} class="text-xl font-bold text-primary">
      {lang === 'zh' ? 'OceanScale 沧渊' : 'OceanScale'}
    </a>
    <div class="flex items-center gap-8 text-sm">
      <a href="#what" class="text-text-secondary hover:text-primary transition">{t('nav.what')}</a>
      <a href="#demos" class="text-text-secondary hover:text-primary transition">{t('nav.demos')}</a>
      <a href="#benchmarks" class="text-text-secondary hover:text-primary transition">{t('nav.benchmarks')}</a>
      <a href="#contact" class="text-text-secondary hover:text-primary transition">{t('nav.contact')}</a>
      <a href={otherUrl} class="border border-border-deep px-3 py-1 rounded text-text-primary hover:border-primary hover:text-primary transition">
        {t('lang.other')}
      </a>
    </div>
  </div>
</nav>
```

- [ ] **Step 4: Create `src/components/Footer.astro`**

```astro
---
import { getLangFromUrl, useTranslations } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const t = useTranslations(lang);
---
<footer class="border-t border-border-deep mt-24">
  <div class="max-w-content mx-auto px-6 py-12 text-center text-text-secondary text-sm">
    {t('footer.copyright')}
  </div>
</footer>
```

- [ ] **Step 5: Create `src/layouts/BaseLayout.astro`**

```astro
---
import '../styles/global.css';
import Nav from '../components/Nav.astro';
import Footer from '../components/Footer.astro';
import { getLangFromUrl } from '../i18n/utils';

interface Props { title: string; description: string; }
const { title, description } = Astro.props;
const lang = getLangFromUrl(Astro.url);
---
<!DOCTYPE html>
<html lang={lang}>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <meta name="description" content={description} />
    <meta property="og:title" content={title} />
    <meta property="og:description" content={description} />
    <meta property="og:image" content="/og-image.png" />
    <meta property="og:type" content="website" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;600;700&display=swap" rel="stylesheet" />
  </head>
  <body>
    <Nav />
    <main><slot /></main>
    <Footer />
  </body>
</html>
```

- [ ] **Step 6: Update both `index.astro` pages to use BaseLayout**

`src/pages/index.astro`:
```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
---
<BaseLayout title="OceanScale 沧渊 — 加速水下机器人的下一个十年" description="AI-native 仿真基础设施,服务海洋机器人研究、产业与防务">
  <div class="section">
    <h1 class="text-5xl font-bold text-text-primary">沧渊 — 内容构建中</h1>
  </div>
</BaseLayout>
```

`src/pages/en/index.astro`:
```astro
---
import BaseLayout from '../../layouts/BaseLayout.astro';
---
<BaseLayout title="OceanScale — Accelerating the next decade of underwater robotics" description="AI-native simulation infrastructure for marine robotics research, industry, and defense">
  <div class="section">
    <h1 class="text-5xl font-bold text-text-primary">OceanScale — content under construction</h1>
  </div>
</BaseLayout>
```

- [ ] **Step 7: Verify nav + lang switcher work**

```bash
cd /home/robot/workspace/46-marine/website
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "OceanScale 沧渊"
curl -s http://localhost:4321/en/ | grep "OceanScale —"
kill %1
```

Expected: Both match. Manually open both URLs in browser and click 中/EN button → verify routing.

- [ ] **Step 8: Commit**

```bash
cd /home/robot/workspace/46-marine
git add website/
git commit -m "feat(website): BaseLayout + Nav + Footer + i18n UI strings"
```

---

## Task 5: Hero component + content

**Files:**
- Create: `website/src/components/Hero.astro`
- Create: `website/src/content/config.ts`
- Create: `website/src/content/zh/hero.md`
- Create: `website/src/content/en/hero.md`
- Modify: both `index.astro` pages

- [ ] **Step 1: Create content collection schema `src/content/config.ts`**

```ts
import { defineCollection, z } from 'astro:content';

const sectionSchema = z.object({
  h1: z.string(),
  sub: z.string(),
});

const zh = defineCollection({ type: 'content', schema: sectionSchema });
const en = defineCollection({ type: 'content', schema: sectionSchema });

export const collections = { zh, en };
```

- [ ] **Step 2: Create `src/content/zh/hero.md`**

```markdown
---
h1: "加速水下机器人的下一个十年"
sub: "GPU-native 仿真 + 智能体训练平台,服务海洋机器人的研究、产业与防务"
---
```

- [ ] **Step 3: Create `src/content/en/hero.md`**

```markdown
---
h1: "Accelerating the next decade of underwater robotics"
sub: "GPU-native simulation + agent training infrastructure for marine robotics research, industry, and defense"
---
```

- [ ] **Step 4: Create `src/components/Hero.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl, useTranslations } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const t = useTranslations(lang);
const hero = await getEntry(lang, 'hero');
const { h1, sub } = hero.data;
---
<section class="section text-center pt-32">
  <h1 class="text-5xl md:text-7xl font-bold tracking-tight text-text-primary mb-6">
    {h1}
  </h1>
  <p class="text-xl md:text-2xl text-text-secondary max-w-3xl mx-auto mb-10">
    {sub}
  </p>
  <div class="flex flex-wrap items-center justify-center gap-4">
    <a href="#demos" class="btn-primary">{t('cta.demos')}</a>
    <a href="#contact" class="btn-secondary">{t('cta.contact')}</a>
  </div>
</section>
```

- [ ] **Step 5: Wire Hero into pages**

`src/pages/index.astro`:
```astro
---
import BaseLayout from '../layouts/BaseLayout.astro';
import Hero from '../components/Hero.astro';
---
<BaseLayout title="OceanScale 沧渊 — 加速水下机器人的下一个十年" description="AI-native 仿真基础设施">
  <Hero />
</BaseLayout>
```

`src/pages/en/index.astro` — symmetric, import paths `../../`.

- [ ] **Step 6: Verify**

```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "加速水下机器人的下一个十年"
curl -s http://localhost:4321/en/ | grep "Accelerating the next decade"
kill %1
```

Expected: Both match.

- [ ] **Step 7: Commit**

```bash
git add website/
git commit -m "feat(website): Hero section + content collections schema"
```

---

## Task 6: Why component (5-problem list)

**Files:**
- Create: `website/src/components/Why.astro`
- Create: `website/src/content/zh/why.md`
- Create: `website/src/content/en/why.md`
- Modify: both `index.astro` pages
- Modify: `src/content/config.ts` (add `why` schema with `problems` array)

- [ ] **Step 1: Extend content schema for Why**

Update `src/content/config.ts`:
```ts
import { defineCollection, z } from 'astro:content';

const problemItem = z.object({ title: z.string(), body: z.string() });
const sectionSchema = z.object({
  title: z.string().optional(),
  h1: z.string().optional(),
  sub: z.string().optional(),
  problems: z.array(problemItem).optional(),
});

const zh = defineCollection({ type: 'content', schema: sectionSchema });
const en = defineCollection({ type: 'content', schema: sectionSchema });
export const collections = { zh, en };
```

- [ ] **Step 2: Create `src/content/zh/why.md`** — content from spec §4 (Why subsection)

```markdown
---
title: "为什么需要 OceanScale"
problems:
  - title: "真实海上测试昂贵"
    body: "出海一天 ¥10,000–¥100,000+(船时 + 人员 + 设备折旧),设备损坏率高(海水腐蚀、压力、缠绕、丢失),环境不可重复(洋流、能见度、生物干扰),进度受天气/季节限制。没有高保真仿真,海洋机器人迭代速度被自然界锁死。"
  - title: "CFD 太慢 — RL 训练不可达"
    body: "高保真 CFD: 1 秒物理需 1–100 小时计算。RL 训练所需 1M+ episodes 数量级不可达。 即使 GPU 加速的 CFD 也无法接近 real-time × 千倍。"
  - title: "现有水下仿真器非 GPU-native"
    body: "Gazebo / UWSim / HoloOcean: 单线程 CPU,几十–几百 FPS。无法充分利用 RTX 5090 / H100 算力。多智能体 / 多场景并行能力差。"
  - title: "Sim-to-real gap"
    body: "流体动力学保真度不足 → 策略迁移失败。缺少有原理的 sim-to-real 校准方法。传感器仿真 (sonar / underwater camera) 也常被简化。"
  - title: "缺少 AI-native 接口"
    body: "gym/gymnasium 适配缺失或质量低。多智能体场景支持差。多模态海洋数据集 (sonar/optical/IMU) 不公开。"
---
```

- [ ] **Step 3: Create `src/content/en/why.md`** — same 5 problems in English

```markdown
---
title: "Why OceanScale exists"
problems:
  - title: "Real ocean trials are prohibitively expensive"
    body: "A day at sea: ¥10,000–¥100,000+ (ship time + crew + equipment depreciation). High equipment loss rate (corrosion, pressure, entanglement, loss). Non-reproducible conditions (currents, visibility, marine life). Progress gated by weather and season. Without high-fidelity simulation, marine robotics iteration speed is locked to nature."
  - title: "CFD is too slow for RL training"
    body: "High-fidelity CFD: 1 second of physics takes 1–100 hours to compute. The 1M+ episodes RL training demands is unreachable. Even GPU-accelerated CFD cannot approach real-time × 1000x."
  - title: "Legacy underwater sims waste modern GPUs"
    body: "Gazebo / UWSim / HoloOcean: single-threaded CPU, tens to hundreds of FPS. Cannot fully utilize RTX 5090 / H100 compute. Multi-agent / multi-scene parallelism is poor."
  - title: "The sim-to-real gap"
    body: "Insufficient fluid dynamics fidelity → policy transfer fails. No principled sim-to-real calibration method. Sensor simulation (sonar, underwater camera) is often oversimplified."
  - title: "Missing AI-native interfaces"
    body: "Gym/gymnasium adapters missing or low quality. Poor multi-agent scene support. Multimodal marine datasets (sonar/optical/IMU) are not openly available."
---
```

- [ ] **Step 4: Create `src/components/Why.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const why = await getEntry(lang, 'why');
const { title, problems } = why.data;
---
<section id="why" class="section">
  <h2 class="text-3xl md:text-5xl font-bold text-text-primary mb-12">{title}</h2>
  <ol class="space-y-8 list-none">
    {problems?.map((p, i) => (
      <li class="border-l-2 border-primary pl-6 py-2">
        <div class="flex items-baseline gap-4 mb-2">
          <span class="font-mono text-primary text-sm">{String(i + 1).padStart(2, '0')}</span>
          <h3 class="text-xl md:text-2xl font-semibold text-text-primary">{p.title}</h3>
        </div>
        <p class="text-text-secondary leading-relaxed">{p.body}</p>
      </li>
    ))}
  </ol>
</section>
```

- [ ] **Step 5: Wire into both pages**

In `src/pages/index.astro` and `src/pages/en/index.astro`, import `Why` and place after `<Hero />`.

- [ ] **Step 6: Verify**

```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep -c "border-l-2"  # expects 5
curl -s http://localhost:4321/ | grep "真实海上测试昂贵"
curl -s http://localhost:4321/en/ | grep "Real ocean trials"
kill %1
```

Expected: First grep prints `5`, others match.

- [ ] **Step 7: Commit**

```bash
git add website/
git commit -m "feat(website): Why section with 5-problem multi-pain framing"
```

---

## Task 7: What component (capabilities)

**Files:**
- Create: `website/src/components/What.astro`
- Create: `website/src/content/zh/what.md`
- Create: `website/src/content/en/what.md`
- Modify: `src/content/config.ts` (extend schema with `capabilities` array)
- Modify: both `index.astro`

- [ ] **Step 1: Extend schema with `capabilities`**

Append to `sectionSchema` in `src/content/config.ts`:
```ts
capabilities: z.array(z.object({
  emoji: z.string().optional(),
  title: z.string(),
  body: z.string(),
})).optional(),
```

- [ ] **Step 2: Create `src/content/zh/what.md`** (content from spec §4 What subsection)

```markdown
---
title: "OceanScale 的能力"
capabilities:
  - emoji: "⚡"
    title: "11,435 FPS BlueROV2 单环境训练"
    body: "RTX 5090 单卡实测,完整 RL 训练管线,已验证。"
  - emoji: "🚀"
    title: "90 M env-steps/s 多环境吞吐"
    body: "8192 并行环境 (Fossen kernel),RTX 5090,达到 900× 目标。"
  - emoji: "🌊"
    title: "Newton 物理 + Isaac Sim 6"
    body: "Anthropic + NVIDIA + Lightwheel + Apple 联合维护的 Newton 物理引擎,Isaac Sim 6 高保真渲染。"
  - emoji: "🧮"
    title: "自研 Fossen 6-DoF + SPH 流体"
    body: "海洋机器人专属 kernel,涵盖 6 自由度刚体与 SPH 流体仿真。"
  - emoji: "🤖"
    title: "即插即用 RL 接口"
    body: "原生 gym/gymnasium,支持 rl_games / stable-baselines3 / RSL-RL 等主流 RL 库。"
  - emoji: "🌍"
    title: "多智能体 + 真实海洋场景"
    body: "多智能体协同,真实海洋资产库 (船/AUV/ROV/传感器),procedural ocean scene 生成。"
---
```

- [ ] **Step 3: Create `src/content/en/what.md`** (mirror in English)

```markdown
---
title: "What OceanScale delivers"
capabilities:
  - emoji: "⚡"
    title: "11,435 FPS single-env BlueROV2 training"
    body: "Measured on RTX 5090 single card. Complete RL training pipeline. Verified."
  - emoji: "🚀"
    title: "90 M env-steps/s multi-env throughput"
    body: "8192 parallel envs (Fossen kernel) on RTX 5090. Hits 900× target."
  - emoji: "🌊"
    title: "Newton physics + Isaac Sim 6"
    body: "Newton physics engine (jointly maintained by Anthropic + NVIDIA + Lightwheel + Apple) with Isaac Sim 6 rendering."
  - emoji: "🧮"
    title: "Custom Fossen 6-DoF + SPH fluids"
    body: "Marine-robotics-specific kernels: 6-DoF rigid body and SPH fluid simulation."
  - emoji: "🤖"
    title: "Drop-in RL interfaces"
    body: "Native gym/gymnasium. Compatible with rl_games, stable-baselines3, RSL-RL, and other major RL libraries."
  - emoji: "🌍"
    title: "Multi-agent + real ocean scenes"
    body: "Multi-agent coordination, real-world marine asset library (ships, AUVs, ROVs, sensors), procedural ocean scene generation."
---
```

- [ ] **Step 4: Create `src/components/What.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const what = await getEntry(lang, 'what');
const { title, capabilities } = what.data;
---
<section id="what" class="section bg-bg-deep-2/50 rounded-2xl mt-12">
  <h2 class="text-3xl md:text-5xl font-bold text-text-primary mb-12">{title}</h2>
  <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
    {capabilities?.map(c => (
      <div class="p-6 bg-bg-deep/60 border border-border-deep rounded-xl hover:border-primary/50 transition">
        <div class="text-3xl mb-4">{c.emoji}</div>
        <h3 class="text-lg font-semibold text-text-primary mb-2">{c.title}</h3>
        <p class="text-text-secondary text-sm leading-relaxed">{c.body}</p>
      </div>
    ))}
  </div>
</section>
```

- [ ] **Step 5: Wire into pages**

Import + place `<What />` after `<Why />` in both pages.

- [ ] **Step 6: Verify**

```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "90 M env-steps/s"
curl -s http://localhost:4321/en/ | grep "8192 parallel envs"
kill %1
```

Expected: Both match.

- [ ] **Step 7: Commit**

```bash
git add website/
git commit -m "feat(website): What section showcasing 6 capabilities incl. 11,435 FPS + 90M env-steps/s"
```

---

## Task 8: Demos component

**Files:**
- Create: `website/src/components/Demos.astro`
- Create: `website/src/content/zh/demos.md`
- Create: `website/src/content/en/demos.md`
- Create: `website/public/videos/bluerov2-demo.mp4` (if available — else use placeholder note)
- Modify: schema (`demos` array)

- [ ] **Step 1: Extend schema with `demos`**

Append to `sectionSchema`:
```ts
demos: z.array(z.object({
  title: z.string(),
  description: z.string(),
  video: z.string().optional(),
  status: z.enum(['available', 'coming']).default('available'),
  comingLabel: z.string().optional(),
})).optional(),
```

- [ ] **Step 2: Create `src/content/zh/demos.md`**

```markdown
---
title: "演示"
demos:
  - title: "BlueROV2 RL 训练 (11,435 FPS)"
    description: "RTX 5090 单卡完整训练管线实测。"
    video: "/videos/bluerov2-demo.mp4"
    status: "available"
  - title: "多 AUV 编队 (Coming v0.3)"
    description: "多智能体协同导航与避障。"
    status: "coming"
    comingLabel: "Coming v0.3"
  - title: "SPH 流体可视化 (Coming v0.3)"
    description: "实时流体场可视化,GPU-native SPH 内核。"
    status: "coming"
    comingLabel: "Coming v0.3"
---
```

- [ ] **Step 3: Create `src/content/en/demos.md`**

```markdown
---
title: "Demos"
demos:
  - title: "BlueROV2 RL training (11,435 FPS)"
    description: "Full training pipeline on RTX 5090 single card, measured."
    video: "/videos/bluerov2-demo.mp4"
    status: "available"
  - title: "Multi-AUV formation (Coming v0.3)"
    description: "Multi-agent navigation and collision avoidance."
    status: "coming"
    comingLabel: "Coming v0.3"
  - title: "SPH fluid visualization (Coming v0.3)"
    description: "Real-time fluid field rendering, GPU-native SPH kernel."
    status: "coming"
    comingLabel: "Coming v0.3"
---
```

- [ ] **Step 4: Create `src/components/Demos.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const demos = await getEntry(lang, 'demos');
const { title, demos: items } = demos.data;
---
<section id="demos" class="section">
  <h2 class="text-3xl md:text-5xl font-bold text-text-primary mb-12">{title}</h2>
  <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
    {items?.map(d => (
      <div class="rounded-xl overflow-hidden border border-border-deep bg-bg-deep/60">
        <div class="aspect-video bg-bg-deep flex items-center justify-center relative">
          {d.video && d.status === 'available' ? (
            <video class="w-full h-full object-cover" autoplay muted loop playsinline preload="metadata">
              <source src={d.video} type="video/mp4" />
            </video>
          ) : (
            <span class="text-text-secondary text-sm">{d.comingLabel}</span>
          )}
        </div>
        <div class="p-4">
          <h3 class="font-semibold text-text-primary mb-1">{d.title}</h3>
          <p class="text-sm text-text-secondary">{d.description}</p>
        </div>
      </div>
    ))}
  </div>
</section>
```

- [ ] **Step 5: Handle bluerov2-demo.mp4**

```bash
mkdir -p /home/robot/workspace/46-marine/website/public/videos
```

If the actual video file is available, copy it:
```bash
cp /path/to/bluerov2-demo.mp4 /home/robot/workspace/46-marine/website/public/videos/
```

If NOT available, create a 1-second placeholder so the `<video>` tag doesn't 404:
```bash
# Use ffmpeg to make a 1s black mp4 (install ffmpeg if missing)
ffmpeg -f lavfi -i color=c=black:s=1280x720:d=1 -vcodec libx264 -pix_fmt yuv420p /home/robot/workspace/46-marine/website/public/videos/bluerov2-demo.mp4
```

If ffmpeg missing, mark this as TODO in `website/README.md` and proceed.

- [ ] **Step 6: Wire into pages + verify**

Place `<Demos />` after `<What />`. Then:
```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "BlueROV2 RL 训练"
curl -s http://localhost:4321/en/ | grep "Multi-AUV formation"
kill %1
```

- [ ] **Step 7: Commit**

```bash
git add website/
git commit -m "feat(website): Demos section with video + Coming v0.3 placeholders"
```

---

## Task 9: Benchmarks component (comparison table)

**Files:**
- Create: `website/src/components/Benchmarks.astro`
- Create: `website/src/content/zh/benchmarks.md`
- Create: `website/src/content/en/benchmarks.md`
- Modify: schema (`benchmarks` table)
- Modify: both `index.astro`

- [ ] **Step 1: Extend schema for benchmarks**

```ts
benchmarks: z.object({
  headers: z.array(z.string()),
  rows: z.array(z.array(z.string())),
  caveat: z.string().optional(),
}).optional(),
```

- [ ] **Step 2: Create `src/content/zh/benchmarks.md`**

```markdown
---
title: "性能基准"
benchmarks:
  headers: ["指标", "OceanScale", "UWSim", "HoloOcean", "Gazebo"]
  rows:
    - ["Single-env FPS (RTX 5090)", "**11,435**", "~200", "~500", "~100"]
    - ["Multi-env throughput @ 8192 envs", "**90 M env-steps/s**", "❌", "❌", "❌"]
    - ["Multi-env parallel scaling", "✅ 8192+", "❌", "部分", "部分"]
    - ["GPU-native physics", "✅ Newton", "❌", "❌", "❌"]
    - ["RL gym 接口", "✅ 原生", "第三方", "第三方", "第三方"]
    - ["SPH 流体 kernel", "✅", "❌", "❌", "❌"]
    - ["Fossen 6-DoF", "✅", "✅", "部分", "❌"]
    - ["ROS 2 集成", "⏳ v0.3", "✅", "✅", "✅"]
  caveat: "OceanScale 数据来自 W2 实测 (2026-05-15)。其他工具数据来自各自官方文档,实际值取决于场景与硬件配置。"
---
```

- [ ] **Step 3: Create `src/content/en/benchmarks.md`**

```markdown
---
title: "Benchmarks"
benchmarks:
  headers: ["Metric", "OceanScale", "UWSim", "HoloOcean", "Gazebo"]
  rows:
    - ["Single-env FPS (RTX 5090)", "**11,435**", "~200", "~500", "~100"]
    - ["Multi-env throughput @ 8192 envs", "**90 M env-steps/s**", "❌", "❌", "❌"]
    - ["Multi-env parallel scaling", "✅ 8192+", "❌", "Partial", "Partial"]
    - ["GPU-native physics", "✅ Newton", "❌", "❌", "❌"]
    - ["RL gym interface", "✅ Native", "3rd-party", "3rd-party", "3rd-party"]
    - ["SPH fluid kernel", "✅", "❌", "❌", "❌"]
    - ["Fossen 6-DoF", "✅", "✅", "Partial", "❌"]
    - ["ROS 2 integration", "⏳ v0.3", "✅", "✅", "✅"]
  caveat: "OceanScale figures measured by W2 benchmark (2026-05-15). Other tools' figures sourced from public documentation; actual values depend on scene and hardware."
---
```

- [ ] **Step 4: Create `src/components/Benchmarks.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const data = await getEntry(lang, 'benchmarks');
const { title, benchmarks } = data.data;
---
<section id="benchmarks" class="section">
  <h2 class="text-3xl md:text-5xl font-bold text-text-primary mb-12">{title}</h2>
  <div class="overflow-x-auto rounded-xl border border-border-deep">
    <table class="w-full text-left text-sm">
      <thead class="bg-bg-deep-2/80">
        <tr>
          {benchmarks?.headers.map((h, i) => (
            <th class={`px-4 py-3 font-semibold ${i === 1 ? 'text-primary' : 'text-text-primary'}`}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {benchmarks?.rows.map(row => (
          <tr class="border-t border-border-deep">
            {row.map((cell, i) => (
              <td class={`px-4 py-3 ${i === 1 ? 'text-primary font-mono' : 'text-text-secondary'}`} set:html={cell.replace(/\*\*(.+?)\*\*/g, '<strong class="text-primary">$1</strong>')}></td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
  {benchmarks?.caveat && <p class="mt-4 text-xs text-text-secondary italic">{benchmarks.caveat}</p>}
</section>
```

- [ ] **Step 5: Wire + verify**

Place `<Benchmarks />` after `<Demos />`. Then:
```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "90 M env-steps/s"
curl -s http://localhost:4321/en/ | grep "Multi-env throughput"
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add website/
git commit -m "feat(website): Benchmarks comparison table with 11,435 FPS + 90M env-steps/s"
```

---

## Task 10: Contact component

**Files:**
- Create: `website/src/components/Contact.astro`
- Create: `website/src/content/zh/contact.md`
- Create: `website/src/content/en/contact.md`
- Modify: schema (`contact` block)

- [ ] **Step 1: Extend schema**

```ts
contact: z.object({
  intro: z.string(),
  business: z.string(),
  tech: z.string(),
  designPartners: z.string(),
  codeAccess: z.string(),
}).optional(),
```

- [ ] **Step 2: Create `src/content/zh/contact.md`**

```markdown
---
title: "联系我们"
contact:
  intro: "欢迎研究机构、产业伙伴、潜在 design partners 联系。代码访问仅向 design partners 开放,通过 业务邮箱 申请。"
  business: "business@oceanscale.cn"
  tech: "tech@oceanscale.cn"
  designPartners: "design-partners@oceanscale.cn"
  codeAccess: "代码访问: 仅向 design partners 开放 (GitHub repo 私有)"
---
```

- [ ] **Step 3: Create `src/content/en/contact.md`**

```markdown
---
title: "Get in touch"
contact:
  intro: "We welcome research institutions, industry partners, and prospective design partners. Code access is for design partners only — apply via business email."
  business: "business@oceanscale.cn"
  tech: "tech@oceanscale.cn"
  designPartners: "design-partners@oceanscale.cn"
  codeAccess: "Code access: design partners only (GitHub repo private)"
---
```

- [ ] **Step 4: Create `src/components/Contact.astro`**

```astro
---
import { getEntry } from 'astro:content';
import { getLangFromUrl } from '../i18n/utils';
const lang = getLangFromUrl(Astro.url);
const data = await getEntry(lang, 'contact');
const { title, contact } = data.data;
---
<section id="contact" class="section text-center">
  <h2 class="text-3xl md:text-5xl font-bold text-text-primary mb-6">{title}</h2>
  <p class="text-text-secondary max-w-2xl mx-auto mb-10">{contact?.intro}</p>
  <div class="flex flex-col sm:flex-row gap-4 justify-center max-w-2xl mx-auto">
    <a href={`mailto:${contact?.business}`} class="btn-primary">{contact?.business}</a>
    <a href={`mailto:${contact?.designPartners}`} class="btn-secondary">{contact?.designPartners}</a>
  </div>
  <p class="mt-8 text-xs text-text-secondary italic">{contact?.codeAccess}</p>
</section>
```

- [ ] **Step 5: Wire + verify**

```bash
pnpm dev &
sleep 3
curl -s http://localhost:4321/ | grep "business@oceanscale.cn"
kill %1
```

- [ ] **Step 6: Commit**

```bash
git add website/
git commit -m "feat(website): Contact section with mailto + design-partner gating"
```

---

## Task 11: Polish — animations, responsive, accessibility

**Files:**
- Create: `website/src/scripts/reveal.ts`
- Create: `website/src/scripts/countup.ts`
- Modify: `Benchmarks.astro` (add count-up data attribute)
- Modify: `BaseLayout.astro` (inject scripts)
- Create: `website/public/og-image.png` (1200x630 placeholder OK)
- Create: `website/public/favicon.svg`

- [ ] **Step 1: Create scroll-reveal script `src/scripts/reveal.ts`**

```ts
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('opacity-100', 'translate-y-0');
      e.target.classList.remove('opacity-0', 'translate-y-4');
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('section').forEach(s => {
  s.classList.add('opacity-0', 'translate-y-4', 'transition-all', 'duration-700');
  observer.observe(s);
});
```

- [ ] **Step 2: Create count-up script `src/scripts/countup.ts`**

```ts
function countUp(el: HTMLElement, end: number, duration = 1500) {
  const start = performance.now();
  const tick = (now: number) => {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.floor(end * eased).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const el = e.target as HTMLElement;
      const target = parseInt(el.dataset.countup || '0', 10);
      countUp(el, target);
      observer.unobserve(el);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('[data-countup]').forEach(el => observer.observe(el));
```

- [ ] **Step 3: Inject scripts into `BaseLayout.astro`**

Add before closing `</body>`:
```astro
<script>import '../scripts/reveal';</script>
<script>import '../scripts/countup';</script>
```

- [ ] **Step 4: Add favicon `public/favicon.svg`**

Simple wave-shaped SVG:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" fill="#000510"/>
  <path d="M2 20 Q 8 14, 16 20 T 30 20" stroke="#00C6FF" stroke-width="3" fill="none"/>
  <path d="M2 24 Q 8 18, 16 24 T 30 24" stroke="#00C6FF" stroke-width="2" fill="none" opacity="0.6"/>
</svg>
```

- [ ] **Step 5: Create placeholder `og-image.png`** — 1200x630 PNG

```bash
# If ImageMagick installed:
convert -size 1200x630 xc:"#000510" -fill "#00C6FF" -gravity center -pointsize 80 -annotate +0+0 "OceanScale\n沧渊" /home/robot/workspace/46-marine/website/public/og-image.png
```

If ImageMagick missing, mark as TODO in README and place a 1x1 transparent PNG.

- [ ] **Step 6: Test responsive in browser**

Open Chrome DevTools, toggle device toolbar, test at 375px (mobile), 768px (tablet), 1280px (desktop). Verify:
- Nav collapses sensibly
- Hero text scales
- What/Demos grids reflow

Note any issues. Fix in `What.astro` / `Demos.astro` if needed.

- [ ] **Step 7: Commit**

```bash
git add website/
git commit -m "feat(website): polish — scroll reveal, count-up, favicon, og-image"
```

---

## Task 12: Lighthouse audit + fix

- [ ] **Step 1: Build production bundle**

```bash
cd /home/robot/workspace/46-marine/website
pnpm build
```

Expected: `dist/` ready, no errors.

- [ ] **Step 2: Serve dist/ locally**

```bash
pnpm dlx serve dist -p 4321 &
sleep 2
```

- [ ] **Step 3: Run Lighthouse via CLI**

```bash
pnpm dlx lighthouse http://localhost:4321/ \
  --only-categories=performance,accessibility,best-practices,seo \
  --output=json --output-path=lighthouse-zh.json --chrome-flags="--headless"
pnpm dlx lighthouse http://localhost:4321/en/ \
  --only-categories=performance,accessibility,best-practices,seo \
  --output=json --output-path=lighthouse-en.json --chrome-flags="--headless"

# Print scores
for f in lighthouse-zh.json lighthouse-en.json; do
  echo "$f:"
  node -e "const r=require('./$f'); for (const k of ['performance','accessibility','best-practices','seo']) console.log('  ' + k + ': ' + Math.round(r.categories[k].score*100));"
done
kill %1
```

Expected scores: Performance ≥ 90, Accessibility ≥ 95, Best Practices ≥ 90, SEO ≥ 95.

- [ ] **Step 4: Fix any below-target metrics**

Common fixes:
- Performance low → check for un-optimized images (use `astro:assets`), missing `loading="lazy"`
- Accessibility low → add `alt` attrs, increase contrast, fix landmark structure
- SEO low → add `meta description` (already in BaseLayout), `lang` attr (already there)

Iterate until all targets met. Each fix → run Lighthouse again.

- [ ] **Step 5: Commit Lighthouse reports + fixes**

```bash
mv lighthouse-zh.json lighthouse-en.json /home/robot/workspace/46-marine/website/
git add website/
git commit -m "test(website): Lighthouse audit reports + fixes"
```

Or if no fixes needed, add reports only:
```bash
git add website/lighthouse-*.json
git commit -m "test(website): Lighthouse audit reports (all targets met)"
```

---

## Task 13: Deploy to Cloudflare Pages

**Files:**
- Create: `website/wrangler.toml`
- Create: `website/README.md` (deploy instructions)

- [ ] **Step 1: Create `website/wrangler.toml`**

```toml
name = "oceanscale-web"
compatibility_date = "2026-01-01"
pages_build_output_dir = "dist"
```

- [ ] **Step 2: Cloudflare auth (manual, one-time)**

```bash
cd /home/robot/workspace/46-marine/website
pnpm dlx wrangler login
```

This opens a browser tab. User must approve the wrangler CLI access. After approval, `~/.config/.wrangler/` stores creds.

**If this fails or no browser available, fallback:** Set `CLOUDFLARE_API_TOKEN` env var (Pages:Edit permission, generated from dash.cloudflare.com → API Tokens).

- [ ] **Step 3: Build for production**

```bash
pnpm build
```

Expected: `dist/` ready.

- [ ] **Step 4: Deploy to Cloudflare Pages**

```bash
pnpm dlx wrangler pages deploy dist --project-name=oceanscale-web --commit-dirty=true
```

Expected: Output ends with `Deployment complete! Take a peek over at https://<hash>.oceanscale-web.pages.dev`.

Record the live URL.

- [ ] **Step 5: Verify live URL**

```bash
LIVE_URL="<the URL from previous step>"
curl -sI "$LIVE_URL" | head -1
curl -s "$LIVE_URL" | grep "加速水下机器人的下一个十年"
curl -s "$LIVE_URL/en/" | grep "Accelerating the next decade"
```

Expected: All match.

- [ ] **Step 6: Manual browser check**

Open `https://oceanscale-web.pages.dev/` and `https://oceanscale-web.pages.dev/en/` in browser. Verify visually:
- Hero looks right (gradient, fonts, CTAs)
- 5 problems show in Why
- 6 capabilities show in What
- Demos video plays (or placeholder)
- Benchmarks table renders, 90M number highlighted
- Lang switcher works

- [ ] **Step 7: Create `website/README.md`**

```markdown
# OceanScale (沧渊) Website

Marketing landing page for OceanScale.

## Local development
```bash
pnpm install
pnpm dev   # http://localhost:4321
```

## Build
```bash
pnpm build
```

## Deploy
```bash
pnpm dlx wrangler pages deploy dist --project-name=oceanscale-web
```

Live URL: https://oceanscale-web.pages.dev
```

- [ ] **Step 8: Final commit**

```bash
git add website/
git commit -m "feat(website): deploy v0.1.0 to Cloudflare Pages — oceanscale-web.pages.dev"
```

---

## Acceptance Criteria

- [ ] `https://oceanscale-web.pages.dev/` returns 200 and shows zh content
- [ ] `https://oceanscale-web.pages.dev/en/` returns 200 and shows en content
- [ ] Lang switcher in nav works both directions
- [ ] Hero CTAs scroll-to-section work
- [ ] Why section shows 5 problems (with 出海贵 #1)
- [ ] What section shows 6 capabilities incl. 11,435 FPS + 90M env-steps/s
- [ ] Benchmarks table includes 90M row, highlighted
- [ ] Contact section shows mailto links, no public GitHub link
- [ ] Lighthouse: Performance ≥ 90, Accessibility ≥ 95, Best Practices ≥ 90, SEO ≥ 95
- [ ] Mobile (375px) layout works
- [ ] Build emits `dist/` with no errors

---

## Self-Review

### Spec coverage check
| Spec section | Implemented in task |
|---|---|
| §1 Goals / Non-Goals | Implicit across all tasks |
| §2 Tech Stack (Astro, Tailwind, CF) | Tasks 1, 2 |
| §3 File Architecture | All tasks (file paths follow spec) |
| §4 Content Strategy — Hero | Task 5 |
| §4 Content Strategy — Why (5 problems) | Task 6 |
| §4 Content Strategy — What | Task 7 |
| §4 Content Strategy — Demos | Task 8 |
| §4 Content Strategy — Benchmarks (incl. 90M) | Task 9 |
| §4 Content Strategy — Contact | Task 10 |
| §5 Visual tokens | Task 2 (tailwind config) |
| §6 i18n | Tasks 3, 4 |
| §7 Deployment | Task 13 |
| §8 Testing checklist | Tasks 11, 12 |
| §11 Future (out of v0.1) | Intentionally excluded — non-goals |

All spec sections covered.

### Placeholder scan
- No "TBD" / "TODO" outside designated v0.3 placeholders in demos (intentional, surfaced as "Coming v0.3" labels)
- All code blocks contain complete implementations

### Type consistency
- `Lang` type defined in `i18n/ui.ts`, used consistently
- Content collection schema extends with each task — all `z.optional()` so earlier markdown files keep validating
- Component prop interfaces consistent (`{title, ...}` pattern)

### Risk callouts
- **Video file dependency:** Task 8 has fallback for missing bluerov2-demo.mp4
- **OG image dependency:** Task 11 has fallback for missing ImageMagick
- **Cloudflare auth:** Task 13 requires manual browser interaction — cannot be fully automated

---

## Execution Mode (next step)

Plan complete and ready to commit. After commit, choose:

**1. Subagent-Driven** — Fresh subagent per task, two-stage review between tasks. Best for iterative correctness, slowest wall-clock.

**2. Inline Execution** — Run tasks 1-13 sequentially in this session via `superpowers:executing-plans`. Fastest, you review at checkpoints.

Recommendation: **Inline Execution** — tasks are small, mostly mechanical, and the wrangler-login step needs user-side browser interaction anyway. Subagent dispatch overhead doesn't pay off here.
