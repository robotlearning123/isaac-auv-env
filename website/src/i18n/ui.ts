export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'zh';

export const ui = {
  zh: {
    'nav.what': '能力',
    'nav.benchmarks': '基准',
    'nav.contact': '联系',
    'cta.demos': '查看基准',
    'cta.contact': '联系',
    'lang.other': 'EN',
    'footer.tagline': '面向水下机器人的海洋仿真基础设施。',
    'footer.copyright': '© 2026 OceanScale',
    'hero.sonarLabel': '基准 · RTX 5090',
  },
  en: {
    'nav.what': 'Platform',
    'nav.benchmarks': 'Benchmarks',
    'nav.contact': 'Contact',
    'cta.demos': 'Benchmarks',
    'cta.contact': 'Contact',
    'lang.other': '中',
    'footer.tagline': 'The ocean simulator for underwater robotics.',
    'footer.copyright': '© 2026 OceanScale',
    'hero.sonarLabel': 'BENCHMARK · RTX 5090',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
