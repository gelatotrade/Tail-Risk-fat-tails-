// Serve a built site under https://tailrisk.test/ through Playwright request routing
// (no listening socket).  CDN assets are fetched by Node and cached on disk, because
// Playwright interception otherwise breaks cross-origin module loading inside workers.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, extname, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const CACHE = join(here, '.cdn-cache');
export const ORIGIN = 'https://tailrisk.test';
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.zip': 'application/zip', '.png': 'image/png', '.gif': 'image/gif', '.svg': 'image/svg+xml',
  '.csv': 'text/csv', '.txt': 'text/plain' };

async function loadPlaywright() {
  for (const spec of [process.env.PLAYWRIGHT_MODULE, 'playwright', '/opt/node22/lib/node_modules/playwright/index.mjs']) {
    if (!spec) continue;
    try { return await import(spec); } catch {}
  }
  throw new Error('Playwright not found: npm i -D playwright or set PLAYWRIGHT_MODULE');
}

export async function openSite(root, { viewport = { width: 1400, height: 900 }, colorScheme = 'light', log = true } = {}) {
  const { chromium } = await loadPlaywright();
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY } : undefined;
  const browser = await chromium.launch({ proxy });
  const context = await browser.newContext({ viewport, colorScheme, acceptDownloads: true });
  await context.route(ORIGIN + '/**', async (route) => {
    let p = decodeURIComponent(new URL(route.request().url()).pathname);
    if (p.endsWith('/')) p += 'index.html';
    const file = normalize(join(root, p));
    if (!file.startsWith(normalize(root))) return route.fulfill({ status: 403 });
    try { await route.fulfill({ status: 200, body: await readFile(file), headers: { 'content-type': TYPES[extname(file)] || 'application/octet-stream' } }); }
    catch { await route.fulfill({ status: 404, body: 'not found' }); }
  });
  for (const host of ['https://cdn.jsdelivr.net/**', 'https://cdnjs.cloudflare.com/**']) {
    await context.route(host, async (route) => {
      const url = route.request().url();
      const key = join(CACHE, createHash('sha1').update(url).digest('hex'));
      try {
        const meta = JSON.parse(await readFile(key + '.json', 'utf8'));
        return route.fulfill({ status: 200, body: await readFile(key), headers: meta.headers });
      } catch {}
      try {
        const resp = await route.fetch();
        const body = await resp.body();
        const headers = { 'content-type': resp.headers()['content-type'] || 'application/octet-stream', 'access-control-allow-origin': '*' };
        if (resp.status() === 200) { await mkdir(CACHE, { recursive: true }); await writeFile(key, body); await writeFile(key + '.json', JSON.stringify({ url, headers })); }
        return route.fulfill({ status: resp.status(), body, headers });
      } catch (e) { console.log('[cdn-fail]', url, e.message); return route.abort(); }
    });
  }
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => { errors.push(e.message); if (log) console.log('[pageerror]', e.message); });
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    if (/WebGL|GL Driver|swiftshader/i.test(t)) return;
    errors.push(t); if (log) console.log('[console.error]', t);
  });
  return { browser, context, page, errors };
}
