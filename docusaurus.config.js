const config = {
  title: 'LLM 101',
  tagline: 'A visual, practical guide to large language models.',
  favicon: 'img/favicon.svg',
  url: 'https://core-computings.github.io',
  baseUrl: '/llm-101/',
  organizationName: 'core-computings',
  projectName: 'llm-101',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  i18n: { defaultLocale: 'en', locales: ['en'] },
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          remarkPlugins: [require('remark-math').default || require('remark-math')],
          rehypePlugins: [require('rehype-katex').default || require('rehype-katex')],
          showLastUpdateTime: false,
          breadcrumbs: true,
        },
        blog: false,
        pages: false,
        theme: { customCss: require.resolve('./src/css/custom.css') },
      },
    ],
  ],
  themeConfig: {
    image: 'img/llm-101-social.svg',
    navbar: {
      title: '',
      logo: { alt: 'LLM 101', src: 'img/logo.svg' },
      items: [{ href: 'https://github.com/core-computings/llm-101', label: 'GitHub ↗', position: 'right' }],
    },
    prism: { additionalLanguages: ['python', 'bash', 'json'] },
    colorMode: { defaultMode: 'light', disableSwitch: false, respectPrefersColorScheme: false },
  },
};

module.exports = config;
