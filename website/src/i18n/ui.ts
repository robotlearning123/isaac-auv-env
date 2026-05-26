export const languages = { zh: '中', en: 'EN' } as const;
export const defaultLang = 'en';

export const ui = {
  zh: {
    'nav.vision': '愿景',
    'nav.virtualOcean': 'Virtual Ocean',
    'nav.usecases': '应用',
    'nav.roadmap': '路线图',
    'nav.contact': '联系',
    'lang.other': 'EN',
    'footer.tagline': '水下机器人训练的虚拟海洋',
    'footer.copyright': '© 2026 OceanScale',
  },
  en: {
    'nav.vision': 'Vision',
    'nav.virtualOcean': 'Virtual Ocean',
    'nav.usecases': 'Use Cases',
    'nav.roadmap': 'Roadmap',
    'nav.contact': 'Contact',
    'lang.other': '中',
    'footer.tagline': 'A virtual ocean for robot training',
    'footer.copyright': '© 2026 OceanScale',
  },
} as const;

export type Lang = keyof typeof ui;
export type UIKey = keyof typeof ui.zh;
