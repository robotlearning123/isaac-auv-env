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
    'footer.tagline': '加速水下机器人的下一个十年',
    'footer.copyright': '© 2026 OceanScale (沧渊) · Beijing',
    'hero.sonarLabel': '声纳 · 在线',
  },
  en: {
    'nav.what': 'Platform',
    'nav.demos': 'Demos',
    'nav.benchmarks': 'Benchmarks',
    'nav.contact': 'Contact',
    'cta.demos': 'See demos',
    'cta.contact': 'Get in touch',
    'lang.other': '中',
    'footer.tagline': 'Accelerating the next decade of underwater robotics',
    'footer.copyright': '© 2026 OceanScale · Beijing',
    'hero.sonarLabel': 'SONAR · ACTIVE',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
