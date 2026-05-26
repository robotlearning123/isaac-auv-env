export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'en';

export const ui = {
  zh: {
    'nav.vision': '挑战',
    'nav.virtualOcean': '虚拟海洋',
    'nav.usecases': '应用',
    'nav.roadmap': '路线图',
    'nav.contact': '联系',
    'lang.other': 'EN',
    'footer.tagline': '水下机器人的虚拟海洋',
    'footer.copyright': '© 2026 OceanScale',
  },
  en: {
    'nav.vision': 'Challenge',
    'nav.virtualOcean': 'Virtual Ocean',
    'nav.usecases': 'Use Cases',
    'nav.roadmap': 'Roadmap',
    'nav.contact': 'Contact',
    'lang.other': '中',
    'footer.tagline': 'A virtual ocean for underwater robots',
    'footer.copyright': '© 2026 OceanScale',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
