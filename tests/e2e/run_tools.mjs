// Run web tools in a real browser (Pyodide) and report status, timing and errors.
//
//   python scripts/build_site.py --out _site
//   node tests/e2e/run_tools.mjs _site [tool-id ...] [--shots dir] [--dataset 'syn:levy_jump:1500:7']
//
// Without tool ids every tool in the catalogue is run.  Exit code 1 if any tool fails.
import { openSite, ORIGIN } from './serve.mjs';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const args = process.argv.slice(2);
const root = resolve(args.shift() || '_site');
let shots = null, dataset = null;
const ids = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--shots') shots = resolve(args[++i]);
  else if (args[i] === '--dataset') dataset = args[++i];
  else ids.push(args[i]);
}
const catalogue = JSON.parse(await readFile(join(root, 'py', 'catalogue.json'), 'utf8'));
// ids may carry parameters: 'chaos?rp_scale=day' runs the tool with that query string
const wanted = ids.map((raw) => { const [id, qs] = raw.split('?'); return { id, qs: qs || '' }; });
const tools = wanted.length
  ? wanted.map((w) => ({ ...catalogue.tools.find((t) => t.id === w.id), qs: w.qs })).filter((t) => t.id)
  : catalogue.tools.map((t) => ({ ...t, qs: '' }));
const unknown = wanted.map((w) => w.id).filter((id) => !catalogue.tools.some((t) => t.id === id));
if (unknown.length) { console.error('unknown tool ids:', unknown.join(', ')); process.exit(2); }
if (shots) await mkdir(shots, { recursive: true });

const { browser, page, errors } = await openSite(root, { log: false });
await page.goto(`${ORIGIN}/index.html#/`);
await page.waitForSelector('.hero-chart svg', { timeout: 60000 });
const report = [];
for (const t of tools) {
  errors.length = 0;
  const qp = new URLSearchParams(t.qs);
  if (dataset && t.needs.length) qp.set('ds', dataset);
  const q = String(qp) ? `?${qp}` : '';
  const t0 = Date.now();
  await page.goto(`${ORIGIN}/index.html#/tool/${t.id}${q}`);
  const timeout = t.cost === 'slow' ? 600000 : 240000;
  let status = 'ok', detail = '';
  try {
    await page.waitForSelector('.result-bar, .error-card', { timeout });
    const err = await page.$('.error-card');
    if (err) { status = 'FAIL'; detail = (await err.innerText()).slice(0, 1500); }
    else {
      // let Plotly finish; check every chart actually rendered
      await page.waitForTimeout(800);
      const empty = await page.$$eval('.chart-card .plot', (els) => els.filter((e) => !e.querySelector('.main-svg')).length);
      if (empty) { status = 'FAIL'; detail = `${empty} chart(s) did not render`; }
    }
  } catch (e) { status = 'FAIL'; detail = 'timeout: ' + e.message.split('\n')[0]; }
  const seconds = (Date.now() - t0) / 1000;
  const pyErrors = errors.filter((e) => !/favicon/.test(e));
  if (status === 'ok' && pyErrors.length) { status = 'FAIL'; detail = pyErrors.join(' | ').slice(0, 1500); }
  const meta = await page.$eval('.result-bar .meta', (e) => e.innerText).catch(() => '');
  if (shots) {
    // lazy images far below the fold would otherwise be empty in a full-page screenshot
    await page.$$eval('img[loading="lazy"]', (imgs) => imgs.forEach((i) => { i.loading = 'eager'; }));
    await page.waitForFunction(() => [...document.images].every((i) => i.complete), null, { timeout: 15000 }).catch(() => {});
    const name = t.qs ? `${t.id}__${t.qs.replace(/[^\w=-]+/g, '_')}` : t.id;
    await page.screenshot({ path: join(shots, `${name}.png`), fullPage: true });
  }
  const label = t.qs ? `${t.id}?${t.qs}` : t.id;
  report.push({ id: label, status, seconds, meta, detail });
  console.log(`${status === 'ok' ? 'ok  ' : 'FAIL'} ${label.padEnd(34)} ${seconds.toFixed(1).padStart(6)} s  ${status === 'ok' ? meta : detail}`);
}
await browser.close();
const failed = report.filter((r) => r.status !== 'ok');
console.log(`\n${report.length - failed.length}/${report.length} tools ok in the browser`);
process.exit(failed.length ? 1 : 0);
