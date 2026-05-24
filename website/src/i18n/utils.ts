import { ui, defaultLang, type Lang, type UIKey } from './ui';

export function getLangFromUrl(url: URL): Lang {
  const [, lang] = url.pathname.split('/');
  if (lang === 'zh') return 'zh';
  return defaultLang;
}

export function useTranslations(lang: Lang) {
  return function t(key: UIKey): string {
    return ui[lang][key] ?? ui[defaultLang][key];
  };
}

export function getOtherLangUrl(currentLang: Lang, pathname: string): string {
  if (currentLang === 'en') {
    return '/zh' + (pathname === '/' ? '/' : pathname);
  }
  return pathname.replace(/^\/zh/, '') || '/';
}
