const JSON_HEADERS = { 'content-type': 'application/json; charset=UTF-8' };

function getCorsHeaders(request, env) {
  const origin = request.headers.get('Origin');
  const allowedOrigins = [env.SITE_ORIGIN, 'http://localhost:3000'];

  if (!origin || !allowedOrigins.includes(origin)) return null;

  return {
    'access-control-allow-origin': origin,
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-max-age': '86400',
    vary: 'Origin',
  };
}

function json(body, status, corsHeaders) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...JSON_HEADERS, ...corsHeaders },
  });
}

export class PageViewCounter {
  constructor(ctx) {
    this.ctx = ctx;
  }

  async get() {
    return (await this.ctx.storage.get('pageViews')) ?? 0;
  }

  async increment() {
    const pageViews = (await this.get()) + 1;
    await this.ctx.storage.put('pageViews', pageViews);
    return pageViews;
  }
}

export default {
  async fetch(request, env) {
    const corsHeaders = getCorsHeaders(request, env);
    if (!corsHeaders) return new Response('Forbidden', { status: 403 });
    if (request.method === 'OPTIONS') return new Response(null, { headers: corsHeaders });

    const path = new URL(request.url).pathname;
    const counter = env.PAGE_VIEW_COUNTER.getByName('llm-101');

    if (request.method === 'GET' && path === '/count') {
      return json({ pageViews: await counter.get() }, 200, corsHeaders);
    }

    if (request.method === 'POST' && path === '/view') {
      return json({ pageViews: await counter.increment() }, 200, corsHeaders);
    }

    return json({ error: 'Not found' }, 404, corsHeaders);
  },
};
