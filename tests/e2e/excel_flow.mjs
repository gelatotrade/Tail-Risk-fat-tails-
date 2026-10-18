// End-to-end check of the Excel workflow: sample download -> upload -> mapping -> run -> xlsx export,
// plus pasting German-formatted numbers copied from Excel.
//   node tests/e2e/excel_flow.mjs _site [outdir]
import { openSite, ORIGIN } from './serve.mjs';
import { resolve, join } from 'node:path';
import { mkdir } from 'node:fs/promises';

const root = resolve(process.argv[2] || '_site');
const outDir = resolve(process.argv[3] || join(root, '..', 'e2e-downloads'));
await mkdir(outDir, { recursive: true });
const { browser, page, errors } = await openSite(root, { log: true });
const step = (m) => console.log('·', m);
const expect = (cond, msg) => { if (!cond) { console.log('FAIL', msg); process.exitCode = 1; } else step(msg); };

await page.goto(`${ORIGIN}/index.html#/tool/fat-tails`);
await page.waitForSelector('.result-bar', { timeout: 240000 });
step('tool ran on default dataset');

// 1) sample download
await page.click('[data-act="change"]');
await page.click('[data-tab="file"]');
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('[data-act="sample"]')]);
const sample = join(outDir, dl.suggestedFilename());
await dl.saveAs(sample);
expect(sample.endsWith('.xlsx'), `sample downloaded: ${dl.suggestedFilename()}`);

// 2) upload it again and check the automatic mapping
await page.setInputFiles('dialog input[type=file]', sample);
await page.waitForSelector('dialog [data-k="value"]');
const mapping = await page.evaluate(() => {
  const sel = (k) => { const s = document.querySelector(`dialog [data-k="${k}"]`); return s.options[s.selectedIndex].text; };
  return { date: sel('date'), value: sel('value'), kind: sel('kind'), vix: sel('vix'), check: document.querySelector('dialog [data-slot="check"]').textContent };
});
expect(mapping.date === 'Datum' && mapping.value === 'Schlusskurs' && mapping.vix === 'VIX', `columns detected: ${JSON.stringify(mapping)}`);
expect(/Kurse/.test(mapping.kind), 'value kind detected as prices');
expect(/absteigend/.test(mapping.check), 'descending order detected and reversed');
await page.click('dialog [data-act="apply"]');
await page.waitForFunction(() => document.querySelector('.ds-summary .name')?.textContent.includes('.xlsx'), null, { timeout: 30000 });
await page.waitForSelector('.result-bar .meta', { timeout: 240000 });
await page.waitForFunction(() => document.querySelector('.result-bar .meta')?.textContent.includes('.xlsx'), null, { timeout: 240000 });
const meta = await page.$eval('.result-bar .meta', (e) => e.innerText);
expect(/50[23] Renditen/.test(meta), `ran on uploaded data: ${meta}`);

// 3) Excel export of the result
const [xl] = await Promise.all([page.waitForEvent('download'), page.click('[data-act="xlsx"]')]);
const exported = join(outDir, xl.suggestedFilename());
await xl.saveAs(exported);
expect(exported.endsWith('.xlsx'), `result exported: ${xl.suggestedFilename()}`);

// 4) paste German-formatted data straight from Excel
await page.click('[data-act="change"]');
await page.click('[data-tab="paste"]');
let tsv = 'Datum\tKurs\n';
let p = 1000;
for (let i = 0; i < 300; i++) {
  const d = new Date(Date.UTC(2023, 0, 2 + i));
  p *= 1 + Math.sin(i * 1.7) * 0.01;
  tsv += `${String(d.getUTCDate()).padStart(2, '0')}.${String(d.getUTCMonth() + 1).padStart(2, '0')}.${d.getUTCFullYear()}\t${p.toLocaleString('de-DE', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}\n`;
}
await page.fill('dialog textarea', tsv);
await page.waitForSelector('dialog [data-k="value"]', { timeout: 5000 });
const check = await page.$eval('dialog [data-slot="check"]', (e) => e.textContent);
expect(/300 gültige Zeilen/.test(check), `pasted rows parsed: ${check}`);
await page.click('dialog [data-act="apply"]');
await page.waitForFunction(() => document.querySelector('.result-bar .meta')?.textContent.includes('299 Renditen'), null, { timeout: 240000 });
step('ran on pasted data (299 returns)');

const real = errors.filter((e) => !/favicon/.test(e));
expect(real.length === 0, `no console errors (${real.length})`);
await browser.close();
console.log(process.exitCode ? 'EXCEL FLOW FAILED' : 'EXCEL FLOW OK', '→', exported);
