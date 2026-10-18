// Navigation while computing: leaving and re-entering a running tool must reattach (not restart),
// and "Cancel" must stop a long run and leave a working engine behind.
//   node tests/e2e/nav_flow.mjs _site
import { openSite, ORIGIN } from './serve.mjs';
import { resolve } from 'node:path';

const root = resolve(process.argv[2] || '_site');
const { browser, page, errors } = await openSite(root, { log: true });
const expect = (c, m) => { console.log(c ? '·' : 'FAIL', m); if (!c) process.exitCode = 1; };
const runs = [];
await page.exposeFunction('__countRun', (t) => runs.push(t));
await page.addInitScript(() => {
  const orig = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (msg, ...rest) { if (msg?.type === 'run') window.__countRun(msg.payload.tool); return orig.call(this, msg, ...rest); };
});

await page.goto(`${ORIGIN}/index.html#/tool/fat-tails`);
await page.waitForSelector('.result-bar', { timeout: 240000 });

// leave a running tool and come back
await page.goto(`${ORIGIN}/index.html#/tool/script-readme-figures`);
await page.waitForSelector('.results .boot-card .progress', { timeout: 30000 });
await page.goto(`${ORIGIN}/index.html#/galerie`); // the German alias of #/gallery must keep working
await page.waitForTimeout(1500);
await page.goto(`${ORIGIN}/index.html#/tool/script-readme-figures`);
await page.waitForSelector('.result-bar, .error-card', { timeout: 180000 });
expect(!(await page.$('.error-card')), 're-entered tool shows its result');
expect(runs.filter((t) => t === 'script-readme-figures').length === 1, `computed exactly once (runs: ${JSON.stringify(runs)})`);

// cancel a long run
await page.goto(`${ORIGIN}/index.html#/tool/script-crisis-examples`);
await page.waitForSelector('.results .boot-card .progress', { timeout: 30000 });
await page.waitForTimeout(2000);
await page.click('.panel .btn.primary');
await page.waitForSelector('.toast', { timeout: 10000 });
expect(/cancelled/i.test(await page.$eval('.toast', (e) => e.textContent)), 'cancel acknowledged');
await page.waitForFunction(() => document.querySelector('#engine-pill')?.dataset.state === 'ready', null, { timeout: 120000 });
expect(true, 'engine rebooted after cancel');
await page.goto(`${ORIGIN}/index.html#/tool/fat-tails?bins=60`);
await page.waitForSelector('.result-bar', { timeout: 120000 });
expect(true, 'engine computes again after cancel');
expect(errors.length === 0, `no console errors (${errors.join(' | ')})`);
await browser.close();
console.log(process.exitCode ? 'NAV FLOW FAILED' : 'NAV FLOW OK');
