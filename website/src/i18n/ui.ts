export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'en';

export const ui = {
  zh: {
    'nav.vision': '挑战',
    'nav.solution': '虚拟海洋',
    'nav.validation': '验证',
    'nav.usecases': '场景',
    'nav.contact': '联系',
    'lang.other': 'EN',
    'footer.tagline': '面向水下机器人的海洋仿真基础设施',
    'footer.copyright': '© 2026 OceanScale',
  },
  en: {
    'nav.vision': 'Challenge',
    'nav.solution': 'Virtual Ocean',
    'nav.validation': 'Validation',
    'nav.usecases': 'Missions',
    'nav.contact': 'Contact',
    'lang.other': '中',
    'footer.tagline': 'The ocean simulator for underwater robotics',
    'footer.copyright': '© 2026 OceanScale',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
