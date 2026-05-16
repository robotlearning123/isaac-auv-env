import { ui, defaultLang, type Lang, type UIKey } from './ui';

export function getLangFromUrl(url: URL): Lang {
  const [, lang] = url.pathname.split('/');
  if (lang === 'en') return 'en';
  return defaultLang;
}

export function useTranslations(lang: Lang) {
  return function t(key: UIKey): string {
    return ui[lang][key] ?? ui[defaultLang][key];
  };
}

export function getOtherLangUrl(currentLang: Lang, pathname: string): string {
  if (currentLang === 'zh') {
    return '/en' + (pathname === '/' ? '/' : pathname);
  }
  return pathname.replace(/^\/en/, '') || '/';
}
