export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'en';

export const ui = {
  zh: {
    'nav.vision': '愿景',
    'nav.product': '产品',
    'nav.benchmarks': '基准',
    'nav.usecases': '应用',
    'nav.roadmap': '路线图',
    'nav.contact': '联系',
    'lang.other': 'EN',
    'footer.tagline': '面向水下机器人的海洋仿真基础设施。',
    'footer.copyright': '© 2026 OceanScale',
  },
  en: {
    'nav.vision': 'Vision',
    'nav.product': 'Product',
    'nav.benchmarks': 'Benchmarks',
    'nav.usecases': 'Use Cases',
    'nav.roadmap': 'Roadmap',
    'nav.contact': 'Contact',
    'lang.other': '中',
    'footer.tagline': 'The ocean simulator for underwater robotics.',
    'footer.copyright': '© 2026 OceanScale',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
