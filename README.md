# LLM 101

LLM 101 is a practical, visual tutorial for understanding large language models.

## Links

- **Repository:** [github.com/core-computings/llm-101](https://github.com/core-computings/llm-101)
- **Website:** [core-computings.github.io/llm-101](https://core-computings.github.io/llm-101/)

## Contents

The site is built with [Docusaurus](https://docusaurus.io/) and published through GitHub Pages. The navigation is defined in `sidebars.js` and the visual style lives in `src/css/custom.css`.

## Write a chapter

Create or edit a Markdown file in `docs/`, then push your changes:

```bash
npm install
npm run start
```

When changes are pushed to `main`, GitHub Actions builds and publishes the website automatically.

## Visitor analytics

The site is ready to use [Cloudflare Web Analytics](https://developers.cloudflare.com/web-analytics/). It tracks page views and privacy-first anonymous visitor statistics, including client-side navigation between Docusaurus pages.

To enable it:

1. In the Cloudflare dashboard, open **Web Analytics** and add the hostname `core-computings.github.io`.
2. Copy the site token from the JavaScript snippet Cloudflare provides.
3. In this GitHub repository, open **Settings → Secrets and variables → Actions** and create a repository secret named `CLOUDFLARE_ANALYTICS_TOKEN`. Paste the site token as its value.
4. Push a commit to `main`, or run the **Deploy documentation to GitHub Pages** workflow manually.

The token is injected only while GitHub Actions builds the site; it is not stored in the repository. Visit the Cloudflare Web Analytics dashboard after deployment to view page views and visitors. Data may take a few minutes to appear.

## Public page-view counter

The footer can show one cumulative **Page views** number. It uses the small Cloudflare Worker in `workers/page-view-counter/` and stores the count in a Durable Object; no visitor identity is recorded.

1. Install or run Wrangler, then deploy the Worker:

   ```bash
   cd workers/page-view-counter
   npx wrangler login
   npx wrangler deploy
   ```

2. Copy the deployed Worker URL, such as `https://llm-101-page-views.<your-subdomain>.workers.dev`.
3. In the GitHub repository, open **Settings → Secrets and variables → Actions → Variables** and create `PAGE_VIEW_COUNTER_URL` with that URL as its value.
4. Deploy the documentation site again. The footer will begin showing `Page views: <number>` and will increment once for each page navigation.

The Worker only accepts browser requests from `https://core-computings.github.io` (and local development on port 3000). The public count is an engagement indicator, not an auditable anti-fraud metric: a determined third party can still imitate browser requests.
