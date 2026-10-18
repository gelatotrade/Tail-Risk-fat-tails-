// Tail Risk Lab - application shell: routing, views, dataset picker, tool runner.
import { Engine, CancelledError } from './engine.js';
import * as D from './data.js';
import { renderResult, renderError, statusBadge, h, openLightbox, tableHtml } from './render.js';
import { buildForm, paramsFromQuery, queryFromParams } from './form.js';
import { exportXlsx } from './export.js';
import { heroChart, normTail2 } from './hero.js';
import { icon } from './icons.js';
import { fmt, fmtDate, escapeHtml, richText } from './format.js';
import { rerenderAll } from './charts.js';

const BUILD = '__BUILD__';
const BASE = new URL('../../', import.meta.url).href;
const REPO = 'https://github.com/gelatotrade/Tail-Risk-fat-tails-';
const engine = new Engine({ base: BASE, build: BUILD });
const view = document.getElementById('view');

const app = {
  catalogue: null, market: null, version: null, gallery: null,
  choice: loadChoice(), dataset: null,
  params: new Map(),   // toolId -> values
  results: new Map(),  // toolId -> {key, result, stdout}
  run: null,           // currently running job
  view: 0,             // increments on every navigation; stale views must not act
};

// --------------------------------------------------------------------------- persistence
function loadChoice() {
  try { const c = JSON.parse(localStorage.getItem('trl.choice')); if (c && c.kind !== 'upload') return c; } catch {}
  return { ...D.DEFAULT_CHOICE };
}
function saveChoice(c) { try { if (c.kind !== 'upload') localStorage.setItem('trl.choice', JSON.stringify(c)); } catch {} }

// --------------------------------------------------------------------------- theme
function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = t === 'dark' || (!t && matchMedia('(prefers-color-scheme: dark)').matches);
  document.getElementById('theme-btn').innerHTML = icon(dark ? 'sun' : 'moon') + '<span class="sr-only">Toggle colour scheme</span>';
  rerenderAll();
  if (location.hash.replace(/^#\/?/, '') === '') heroRefresh();
}
function initTheme() {
  let t = null; try { t = localStorage.getItem('trl.theme'); } catch {}
  applyTheme(t);
  document.getElementById('theme-btn').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme === 'dark' || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const next = dark ? 'light' : 'dark';
    try { localStorage.setItem('trl.theme', next); } catch {}
    applyTheme(next);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!document.documentElement.dataset.theme) applyTheme(null); });
}

// --------------------------------------------------------------------------- engine pill
function initPill() {
  const pill = document.getElementById('engine-pill');
  const txt = pill.querySelector('.txt');
  const update = () => {
    pill.dataset.state = engine.state;
    const s = engine.statusInfo;
    const label = { idle: 'Python ready to start', booting: `Loading Python … ${Math.round(s.pct || 0)}%`, ready: 'Python running in your browser', busy: 'Python computing …', error: 'Python error' }[engine.state];
    txt.textContent = label;
    pill.title = engine.state === 'ready' ? `Pyodide ${'314.0.7'} · started in ${engine.bootSeconds?.toFixed(1)} s` : s.text;
  };
  engine.addEventListener('status', update);
  engine.addEventListener('state', update);
  update();
}

const toast = (msg) => {
  const t = h(`<div class="toast" role="status">${escapeHtml(msg)}</div>`);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
};

// --------------------------------------------------------------------------- routing
function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, qs] = raw.split('?');
  return { parts: path.split('/').filter(Boolean), query: new URLSearchParams(qs || '') };
}

function setNav(active) {
  document.querySelectorAll('.nav a').forEach((a) => {
    if (a.dataset.route === active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  document.querySelector('.nav').classList.remove('open');
  if (active !== 'tools' || !location.hash.startsWith('#/tool/')) document.title = 'Tail Risk Lab · Fat tails, live in your browser';
}

async function route() {
  const { parts, query } = parseHash();
  const [first, second] = parts;
  app.view += 1;
  if (app.run && !app.run.done) app.run.detach = true; // keeps running; result is cached for re-entry
  window.scrollTo({ top: 0 });
  if (!first) { setNav('home'); return renderHome(); }
  // German paths of the first release keep working for links that were already shared
  const alias = { werkzeuge: 'tools', galerie: 'gallery', methodik: 'methodology' };
  if (alias[first]) { history.replaceState(null, '', `#/${alias[first]}`); return route(); }
  if (first === 'tools') { setNav('tools'); return renderCatalogue(); }
  if (first === 'tool' && second) { setNav('tools'); return renderTool(second, query); }
  if (first === 'gallery') { setNav('gallery'); return renderGallery(); }
  if (first === 'methodology') { setNav('method'); return renderMethod(); }
  setNav('');
  view.innerHTML = `<div class="wrap empty"><h1>Page not found</h1><p><a href="#/">Back to the home page</a></p></div>`;
}

const toolHref = (id, q) => `#/tool/${id}${q && String(q) ? '?' + q : ''}`;
const groupTitle = (gid) => app.catalogue.groups.find((g) => g.id === gid)?.title || gid;

function toolCard(t) {
  const sim = t.needs.length === 0;
  return `<a class="card tool-card" href="${toolHref(t.id)}" data-tool="${t.id}">
    <h3><span class="ic">${icon(t.group)}</span><span>${escapeHtml(t.title)}</span></h3>
    <p>${escapeHtml(t.summary)}</p>
    <div class="meta">${sim ? '<span class="chip">Simulation</span>' : '<span class="chip accent">uses your data</span>'}
      ${t.needs.includes('vix') ? '<span class="chip">needs VIX</span>' : ''}
      ${t.cost === 'slow' ? '<span class="chip">compute-intensive</span>' : t.cost === 'fast' ? '<span class="chip">live</span>' : ''}</div></a>`;
}

function catalogueHtml() {
  const groups = app.catalogue.groups.map((g) => {
    const tools = app.catalogue.tools.filter((t) => t.group === g.id);
    if (!tools.length) return '';
    return `<div class="tool-group" data-group="${g.id}"><div class="group-title">${icon(g.id, 'width="15" height="15"')} ${escapeHtml(g.title)}</div><div class="tool-grid">${tools.map(toolCard).join('')}</div></div>`;
  }).join('');
  return `<div class="catalogue-bar">
      <label class="search">${icon('search')}<input type="search" placeholder="Search, e.g. VaR or VIX" aria-label="Search tools"></label>
      <div class="seg" role="tablist" aria-label="Filter">
        <button type="button" data-f="all" aria-selected="true">All</button>
        <button type="button" data-f="data" aria-selected="false">Uses your data</button>
        <button type="button" data-f="sim" aria-selected="false">Simulations</button>
      </div></div>${groups}<p class="empty" data-slot="none" hidden>No tool matches this search.</p>`;
}

function wireCatalogue(root) {
  const input = root.querySelector('.catalogue-bar input');
  if (!input) return;
  let filter = 'all';
  const norm = (s) => s.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  const index = new Map(app.catalogue.tools.map((t) => [t.id, norm([t.title, t.summary, t.description || '', t.covers.join(' '), groupTitle(t.group)].join(' '))]));
  const apply = () => {
    const words = norm(input.value).split(/\s+/).filter(Boolean);
    let shown = 0;
    root.querySelectorAll('.tool-group').forEach((g) => {
      let any = false;
      g.querySelectorAll('.tool-card').forEach((c) => {
        const t = app.catalogue.tools.find((x) => x.id === c.dataset.tool);
        const okF = filter === 'all' || (filter === 'data' ? t.needs.length > 0 : t.needs.length === 0);
        const okW = words.every((w) => index.get(t.id).includes(w));
        c.hidden = !(okF && okW); any ||= !c.hidden; shown += !c.hidden;
      });
      g.hidden = !any;
    });
    root.querySelector('[data-slot="none"]').hidden = shown > 0;
  };
  input.addEventListener('input', apply);
  root.querySelectorAll('.catalogue-bar [data-f]').forEach((b) => b.addEventListener('click', () => {
    filter = b.dataset.f;
    root.querySelectorAll('.catalogue-bar [data-f]').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    apply();
  }));
}

function prewarmOnIntent(root) {
  // Start downloading Python as soon as someone shows intent - not for readers who just browse.
  const warm = () => { if (engine.state === 'idle') engine.ready().catch(() => {}); };
  root.querySelectorAll('[data-tool], [data-warm]').forEach((el) => {
    el.addEventListener('pointerenter', warm, { once: true });
    el.addEventListener('focus', warm, { once: true });
    el.addEventListener('touchstart', warm, { once: true, passive: true });
  });
}

// --------------------------------------------------------------------------- home
let heroRefresh = () => {};
async function renderHome() {
  const sp = app.market.datasets.find((d) => d.id === 'sp500') || app.market.datasets[0];
  const pulse = D.marketPulse(sp);
  const n4 = pulse.beyond[4], exp4 = normTail2(4) * pulse.returns.length;
  const term = pulse.vix && pulse.vix3m ? pulse.vix / pulse.vix3m : null;
  const termStatus = term == null ? null : term > 1 ? 'critical' : term > 0.95 ? 'serious' : term > 0.9 ? 'warning' : 'good';
  const termLabel = term == null ? '' : term > 1 ? 'Backwardation – panic' : term > 0.9 ? 'Flat – heightened nervousness' : 'Contango – normal';
  const vixStatus = pulse.vix == null ? null : pulse.vix >= 30 ? 'critical' : pulse.vix >= 22 ? 'serious' : pulse.vix >= 17 ? 'warning' : 'good';
  const vixLabel = { good: 'Calm', warning: 'Slightly elevated', serious: 'Elevated', critical: 'Stress' }[vixStatus];

  view.innerHTML = `
  <section class="hero"><div class="wrap hero-grid">
    <div>
      <span class="eyebrow">${icon('cpu', 'width="14" height="14"')} Python framework, live in your browser</span>
      <h1>Crises live in the <em>tails</em> of the distribution.</h1>
      <p class="lead">The complete tail-risk framework – Lévy flights, Fokker-Planck, Tsallis statistics, extreme value theory, phase transitions and an early-warning system – runs directly in your browser. Use current market data or your own Excel series.</p>
      <div class="ctas">
        <a class="btn primary lg" href="#/tools" data-warm>${icon('play')} Open the tools</a>
        <a class="btn lg" href="${toolHref('fat-tails')}" data-warm>${icon('excel')} Use your own Excel data</a>
      </div>
      <div class="trust">
        <span>${icon('shield')} Your data stays on your computer</span>
        <span>${icon('check')} No installation</span>
        <span>${icon('check')} Export to Excel</span>
      </div>
    </div>
    <div class="card hero-chart">
      <div class="head"><h3>S&P 500 · daily returns over the last ${fmt(pulse.years, 'num', 0)} years</h3><span class="muted small">log scale · as of ${fmtDate(pulse.lastDate)}</span></div>
      <div id="hero-plot"></div>
      <div class="legend"><span><i style="background:var(--s1);opacity:.6"></i>Observed</span><span><i style="background:var(--critical)"></i>Beyond 3σ</span><span><i style="background:var(--ink);opacity:.7"></i>Normal distribution</span></div>
      <div class="hero-callout">
        <div><div class="v">${fmt(n4, 'int')}</div><div class="l">days beyond 4σ observed</div></div>
        <div><div class="v">${fmt(exp4, 'num', 2)}</div><div class="l">expected under the Gaussian assumption</div></div>
        <div><div class="v">${fmt(exp4 > 0 ? n4 / exp4 : null, 'x')}</div><div class="l">underestimation by the normal distribution</div></div>
      </div>
    </div>
  </div></section>

  <section class="wrap" aria-label="Market conditions">
    <div class="section-head" style="margin-bottom:10px"><div><h2 style="font-size:18px">Market pulse</h2><p class="small">Calculated from this site's FRED data · as of ${fmtDate(pulse.lastDate)}</p></div>
      <a class="btn sm" href="${toolHref('fat-tails')}" data-warm>Full analysis ${icon('arrow')}</a></div>
    <div class="market-strip">
      <div class="card tile"><div class="l">S&P 500</div><div class="v">${fmt(pulse.last, 'num', 0)}</div><div class="d">${pulse.change >= 0 ? '+' : ''}${fmt(pulse.change, 'pct', 2)} on the day · ${fmt(pulse.drawdown, 'pct', 1)} from the high</div></div>
      <div class="card tile"><div class="l">VIX</div><div class="v">${fmt(pulse.vix, 'num', 1)}</div><div class="d">${statusBadge(vixStatus, vixLabel)}</div></div>
      <div class="card tile"><div class="l">VIX term structure (VIX / VIX3M)</div><div class="v">${fmt(term, 'num', 2)}</div><div class="d">${statusBadge(termStatus, termLabel)}</div></div>
      <div class="card tile"><div class="l">Realised volatility (20 days, annualised)</div><div class="v">${fmt(pulse.vol20, 'pct', 1)}</div><div class="d">Long-term: ${fmt(pulse.volAll, 'pct', 1)}</div></div>
    </div>
  </section>

  <section class="section wrap">
    <div class="section-head"><div><h2>How it works</h2><p>No server is involved in the calculations. The original Python code of this repository runs as WebAssembly in your browser.</p></div></div>
    <div class="steps">
      <div class="card"><div class="n">1</div><h3>Choose your data</h3><p>S&P 500, NASDAQ and VIX from FRED, synthetic scenarios – or your own Excel or CSV file, including copy and paste from Excel.</p></div>
      <div class="card"><div class="n">2</div><h3>Run a tool</h3><p>On first use, Python loads with NumPy, SciPy and Matplotlib (a one-off download of about 30 MB, then served from the cache).</p></div>
      <div class="card"><div class="n">3</div><h3>Understand and export</h3><p>Interactive charts, the framework's original figures and every number as an Excel file – or as a link to share.</p></div>
    </div>
  </section>

  <section class="section wrap" id="tools">
    <div class="section-head"><div><h2>${app.catalogue.tools.length} tools – every feature of the framework</h2><p>From distribution analysis and physics simulations to the early-warning system and the repository's original scripts.</p></div></div>
    ${catalogueHtml()}
  </section>

  <section class="section wrap">
    <div class="section-head"><div><h2>Gallery</h2><p>The figures from the repository – each one can be recalculated live with the tools.</p></div><a class="btn" href="#/gallery">View all ${icon('arrow')}</a></div>
    <div class="gallery" style="columns:3 260px">${(app.gallery?.sections || []).flatMap((s) => s.items).filter((i) => i.teaser).slice(0, 3).map(galleryItem).join('')}</div>
  </section>`;
  const plot = view.querySelector('#hero-plot');
  heroRefresh = () => { if (plot.isConnected) plot.innerHTML = heroChart(pulse); };
  heroRefresh();
  wireGallery(view);
  wireCatalogue(view);
  prewarmOnIntent(view);
}

function renderCatalogue() {
  view.innerHTML = `<section class="section wrap">
    <div class="section-head"><div><h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Tools</h1>
    <p>Every tool calls the framework's original Python code. Tools marked ‘uses your data’ work on the selected dataset; simulations generate their own paths.</p></div></div>
    ${catalogueHtml()}</section>`;
  wireCatalogue(view);
  prewarmOnIntent(view);
  view.querySelector('.catalogue-bar input')?.focus({ preventScroll: true });
}

// --------------------------------------------------------------------------- gallery & method
function galleryItem(it) {
  const src = `gallery/${it.file}`;
  return `<figure class="card"><img src="${src}" alt="${escapeHtml(it.title)}" loading="lazy" data-cap="${escapeHtml(it.title + ' – ' + (it.caption || ''))}">
    <figcaption><b>${escapeHtml(it.title)}</b>${escapeHtml(it.caption || '')}${it.tool ? ` <a href="${toolHref(it.tool)}">Recalculate live →</a>` : ''}</figcaption></figure>`;
}
function wireGallery(root) {
  root.querySelectorAll('.gallery img').forEach((img) => img.addEventListener('click', () => openLightbox(img.src, img.dataset.cap)));
}
function renderGallery() {
  const secs = app.gallery?.sections || [];
  view.innerHTML = `<section class="section wrap"><div class="section-head"><div><h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Gallery</h1>
    <p>Every figure produced by the repository's scripts. Under ‘Original scripts’ you can regenerate them in your browser.</p></div></div>
    ${secs.map((s) => `<div class="group-title">${escapeHtml(s.title)}</div><div class="gallery">${s.items.map(galleryItem).join('')}</div>`).join('')}</section>`;
  wireGallery(view);
}

function renderMethod() {
  view.innerHTML = `<section class="section wrap"><article class="prose">
  <h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Methodology and technology</h1>
  <p>Classic risk models – modern portfolio theory, Black-Scholes, most VaR systems – assume normally distributed returns. Real markets have fat tails: extreme days occur orders of magnitude more often. This framework transfers tools from statistical physics to risk measurement.</p>
  <h2>Physics → finance</h2>
  ${tableHtml(['Physical concept', 'Application in risk management'], [
    ['Lévy flights (particle paths with jumps)', 'Price jumps, black-swan events; stability index α < 2 = fat tails'],
    ['Fokker-Planck equation (probability flow)', 'How the return distribution evolves over time; forecasting tail probabilities'],
    ['Tsallis entropy (non-extensive thermodynamics)', 'Markets far from equilibrium; q > 1 produces power-law tails'],
    ['Phase transitions (Ising model, magnetism)', 'Crashes as critical transitions; herding = coupling J'],
    ['Ornstein-Uhlenbeck process (spring)', 'Mean reversion of volatility, with jumps'],
    ['Extreme value theory (GPD, peaks over threshold)', 'Statistically sound estimation of VaR/ES deep in the tail'],
  ], ['text', 'text'])}
  <h2>Early-warning system</h2>
  <p>The composite score weights five indicators that proved their worth in the 2018, 2020, 2022 and 2025 crashes:</p>
  <div class="formula">score = 0.35·GEX + 0.25·TDEX + 0.20·VIX term structure + 0.10·DIX + 0.10·SMFI</div>
  <p style="margin-top:12px">0–25 Normal · 25–50 Elevated · 50–75 High · 75–100 Extreme. GEX, TailDex, DIX and Smart Money Flow rely on options and dark-pool data that are not freely available. Like the framework itself, the tools derive them from returns and the VIX, or let you enter the readings directly.</p>
  <h2>How the calculations run</h2>
  <p>The site is fully static and served by GitHub Pages. When you first open a tool, your browser loads <a href="https://pyodide.org" rel="noopener">Pyodide</a> – CPython 3.14 compiled to WebAssembly – with NumPy, SciPy and Matplotlib. The repository's <code>src/</code> package is then imported unmodified. A thin layer (<code>webtools/</code>) turns form values into function calls and results into charts. All calculations run in a Web Worker, so the page stays responsive.</p>
  <p><b>Data protection:</b> Excel or CSV files you upload are processed locally in your browser only and are never transmitted.</p>
  <h2>Corrections made before going live</h2>
  <p>Before the framework went public, every function was executed and checked against textbook formulas, SciPy and numerical integration. <b>More than 90 bugs in 19 modules</b> were fixed, each backed by regression tests (more than 350 tests that fail before the fix and pass after it). The author's modelling decisions were left unchanged. The figures in the README were produced with the original code, so individual values may differ.</p>
  ${tableHtml(['Area', 'What was corrected'], [
    ['Expected Shortfall', 'Historical and Cornish-Fisher ES averaged the gains instead of the losses beyond VaR; the Student\'s t ES ignored the scale.'],
    ['Drawdown', 'Log returns were compounded as if they were simple returns.'],
    ['Extreme value theory', 'Return periods were counted in 20-day blocks: the ‘10-year return level’ was in fact a 200-year level. GPD probabilities were conditional instead of unconditional, and the GEV VaR was negative.'],
    ['Lévy distribution', 'Tail probability too small by a factor of 2; the quantile fit is now the genuine McCulloch method; multi-day risk scales with h^(1/α) instead of √h.'],
    ['Tsallis statistics', 'Variance, spread of the random draws, distribution function and tail asymptote corrected; the moment fit for q ran backwards.'],
    ['Fokker-Planck', 'The backward instead of the forward equation was solved (drift with the wrong sign); the fractional operator was anti-diffusive and unstable.'],
    ['Fat-tailed distributions', 'Mirrored variance-gamma density, wrong normalisation of the generalised hyperbolic, wrong NIG samples and fits.'],
    ['Markov regimes', 'Incorrect M-step in the EM algorithm; now Hamilton filter, Kim smoother and Baum-Welch – and about 50 times faster, which is what makes the dashboards feasible in the browser.'],
    ['Risk ratios', 'The Sortino ratio averaged the downside over losing days only, overstating the downside deviation.'],
    ['Early-warning system', 'Contango and backwardation were swapped; DIX and SMFI were down-weighted twice in the composite. The simulated DIX looks five days into the future – this site therefore always computes in the new causal mode without look-ahead.'],
    ['Compatibility', 'Crashes under NumPy 2 (np.trapz, np.math) and SciPy 1.18 (Kolmogorov-Smirnov test) fixed – the prerequisite for everything running in Pyodide.'],
  ], ['text', 'text'])}
  <h2>Data sources</h2>
  <p>Market data: ${escapeHtml(app.market.source)} – series SP500, NASDAQCOM, VIXCLS, VXVCLS. Refreshed with every deployment and automatically on every business day (last update ${fmtDate(app.market.updated)}). Under its licence terms, FRED provides the S&P 500 series for the last ten years only.</p>
  <h2>Disclaimer</h2>
  <p class="disclaimer">This site is intended for research and education. It is not investment advice and not a recommendation to buy or sell any financial instrument. Models simplify reality; results based on historical data are no guarantee of future behaviour.</p>
  <p class="muted small">Build ${escapeHtml(app.version?.build || 'dev')} · Commit <a href="${REPO}/commit/${escapeHtml(app.version?.commit || '')}">${escapeHtml(app.version?.commit || '–')}</a> · ${escapeHtml(fmtDate(app.version?.built))}</p>
  </article></section>`;
}

// --------------------------------------------------------------------------- dataset
async function resolveDataset() {
  app.dataset = await D.resolve(app.choice, BASE);
  return app.dataset;
}

function datasetCard(spec) {
  const ds = app.dataset;
  const needsVix = spec.needs.includes('vix') || spec.needs.includes('vix3m');
  const hasVix = ds && (ds.arrays ? !!ds.arrays.vix : true); // synthetic datasets carry a derived VIX
  const card = h(`<section class="card"><h2>Dataset <button class="btn sm" type="button" data-act="change">Change</button></h2>
    <div class="ds-summary"><div class="name">${escapeHtml(ds.label)}</div><div class="muted">${escapeHtml(ds.detail)}</div>
    <div class="row" data-slot="stats"></div></div>
    ${needsVix && !hasVix ? `<div class="note warning" style="margin-top:10px">${icon('warn')}<div>This tool needs a VIX series. <button class="btn sm" data-act="sp500" type="button" style="margin-top:6px">Use S&P 500 with VIX</button></div></div>` : ''}
    </section>`);
  card.querySelector('[data-act="change"]').addEventListener('click', () => openDatasetDialog());
  card.querySelector('[data-act="sp500"]')?.addEventListener('click', () => applyChoice({ kind: 'market', id: 'sp500', preset: 'all' }));
  return card;
}

function fillDatasetStats(card, summary) {
  const slot = card?.querySelector('[data-slot="stats"]');
  if (!slot || !summary) return;
  const w = summary.worst;
  slot.innerHTML = `<span class="chip">Volatility p.a. ${fmt(summary.ann_vol, 'pct', 1)}</span>
    <span class="chip">Worst day ${fmt(w.value, 'pct', 1)}${typeof w.date === 'string' ? ' · ' + fmtDate(w.date) : ''}</span>
    ${summary.has.vix ? '<span class="chip">VIX ✓</span>' : ''}`;
}

async function applyChoice(choice) {
  app.choice = choice;
  saveChoice(choice);
  await resolveDataset();
  const { parts, query } = parseHash();
  if (parts[0] === 'tool') {
    if (app.dataset.shareable) query.set('ds', D.encodeChoice(choice)); else query.delete('ds');
    history.replaceState(null, '', toolHref(parts[1], query));
    renderTool(parts[1], query, { autorun: true });
  }
}

function openDatasetDialog() {
  const dlg = h(`<dialog class="sheet" aria-labelledby="ds-title">
    <div class="sheet-head"><h2 id="ds-title">Choose a dataset</h2><button class="icon-btn" data-act="close" aria-label="Close">${icon('close')}</button></div>
    <div class="tabs" role="tablist">
      <button role="tab" data-tab="market">Market data</button><button role="tab" data-tab="synthetic">Synthetic</button>
      <button role="tab" data-tab="crisis">Crisis scenario</button><button role="tab" data-tab="file">Excel / CSV file</button>
      <button role="tab" data-tab="paste">Paste from Excel</button></div>
    <div class="sheet-body"></div>
    <div class="sheet-foot"><span class="form-error" data-slot="err" style="margin-right:auto"></span><button class="btn" data-act="close">Cancel</button><button class="btn primary" data-act="apply">Apply</button></div></dialog>`);
  document.body.appendChild(dlg);
  const body = dlg.querySelector('.sheet-body'), err = dlg.querySelector('[data-slot="err"]');
  let pending = null; // () => choice | Promise<choice>
  const close = () => { dlg.close(); dlg.remove(); };
  dlg.querySelectorAll('[data-act="close"]').forEach((b) => b.addEventListener('click', close));
  dlg.addEventListener('cancel', (e) => { e.preventDefault(); close(); });
  dlg.querySelector('[data-act="apply"]').addEventListener('click', async () => {
    err.textContent = '';
    try { const c = await pending?.(); if (!c) return; close(); applyChoice(c); } catch (e) { err.textContent = e.message; }
  });

  const tabs = {
    market() {
      const cur = app.choice.kind === 'market' ? app.choice : { id: 'sp500', preset: 'all' };
      let sel = cur.id, preset = cur.from ? 'custom' : cur.preset || 'all';
      const dsById = (id) => app.market.datasets.find((d) => d.id === id);
      body.innerHTML = `<div class="options">${app.market.datasets.map((d) => `<button type="button" class="option" data-id="${d.id}" aria-pressed="${d.id === sel}"><b>${escapeHtml(d.name)}</b><span>${escapeHtml(d.blurb)}</span><span class="muted">${fmtDate(d.dates[0])} – ${fmtDate(d.dates.at(-1))}</span></button>`).join('')}</div>
        <div><div class="help" style="margin-bottom:6px">Period</div><div class="presets"></div></div>
        <div class="row-2"><label class="field"><span class="lbl"><b>From</b></span><input type="date" data-k="from"></label><label class="field"><span class="lbl"><b>To</b></span><input type="date" data-k="to"></label></div>
        <p class="privacy">${icon('database')}<span>Source: ${escapeHtml(app.market.source)} · as of ${fmtDate(app.market.updated)}</span></p>`;
      const from = body.querySelector('[data-k="from"]'), to = body.querySelector('[data-k="to"]');
      const sync = () => {
        const d = dsById(sel);
        from.min = to.min = d.dates[0]; from.max = to.max = d.dates.at(-1);
        body.querySelector('.presets').innerHTML = D.PRESETS.map((p) => {
          const r = D.presetRange(p.id, d.dates);
          return `<button type="button" class="btn sm ${p.id === preset ? 'primary' : ''}" data-p="${p.id}" ${r ? '' : 'disabled title="Not covered by this dataset"'}>${escapeHtml(p.label)}</button>`;
        }).join('');
        if (preset !== 'custom') { const r = D.presetRange(preset, d.dates) || D.presetRange('all', d.dates); from.value = r.from; to.value = r.to; }
        body.querySelectorAll('[data-p]').forEach((b) => b.addEventListener('click', () => { preset = b.dataset.p; sync(); }));
      };
      if (cur.from) { from.value = cur.from; to.value = cur.to; }
      body.querySelectorAll('.option').forEach((b) => b.addEventListener('click', () => {
        sel = b.dataset.id; body.querySelectorAll('.option').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
        if (preset === 'custom') preset = 'all';
        sync();
      }));
      [from, to].forEach((i) => i.addEventListener('change', () => { preset = 'custom'; body.querySelectorAll('[data-p]').forEach((b) => b.classList.remove('primary')); }));
      sync();
      pending = () => {
        if (preset !== 'custom') return { kind: 'market', id: sel, preset };
        if (!from.value || !to.value || from.value >= to.value) throw new Error('Please choose a valid period.');
        return { kind: 'market', id: sel, from: from.value, to: to.value };
      };
    },
    synthetic() {
      const cur = app.choice.kind === 'synthetic' ? app.choice : { model: 'levy_jump', n: 2520, seed: 42 };
      let model = cur.model;
      body.innerHTML = `<div class="options">${D.SYNTHETIC_MODELS.map((m) => `<button type="button" class="option" data-m="${m.value}" aria-pressed="${m.value === model}"><b>${escapeHtml(m.label)}</b><span>${escapeHtml(m.hint)}</span></button>`).join('')}</div>
        <div class="row-2"><label class="field"><span class="lbl"><b>Number of trading days</b></span><input type="number" min="60" max="20000" step="10" value="${cur.n}" data-k="n"></label>
        <label class="field"><span class="lbl"><b>Random seed</b></span><input type="number" min="0" max="999999" value="${cur.seed}" data-k="seed"><span class="help">The same seed gives an identical series – ideal for sharing.</span></label></div>
        <p class="privacy">${icon('cpu')}<span>Generated with <code>generate_synthetic_returns</code> from <code>src/utils/data_loader.py</code>; a matching VIX is derived with <code>generate_vix_from_returns</code>.</span></p>`;
      body.querySelectorAll('.option').forEach((b) => b.addEventListener('click', () => { model = b.dataset.m; body.querySelectorAll('.option').forEach((o) => o.setAttribute('aria-pressed', String(o === b))); }));
      pending = () => ({ kind: 'synthetic', model, n: Number(body.querySelector('[data-k="n"]').value), seed: Number(body.querySelector('[data-k="seed"]').value) });
    },
    crisis() {
      const seed = app.choice.kind === 'crisis' ? app.choice.seed : 42;
      body.innerHTML = `<p>An artificial market in three phases: 500 calm, complacent days, a 100-day crash with a volatility spike and fat tails, and 400 days of nervous recovery. Ideal for checking whether early-warning signals fire <i>before</i> the crash.</p>
        <label class="field" style="max-width:260px"><span class="lbl"><b>Random seed</b></span><input type="number" min="0" max="999999" value="${seed}" data-k="seed"><span class="help">The crisis is random: seed 42 gives −34% over the crisis window (similar to COVID 2020), seed 50 gives −59%; some seeds already recover during the crash.</span></label>
        <p class="privacy">${icon('cpu')}<span>Generated with <code>generate_crisis_scenario</code> from <code>src/utils/data_loader.py</code>.</span></p>`;
      pending = () => ({ kind: 'crisis', seed: Number(body.querySelector('[data-k="seed"]').value) });
    },
    file() { importTab(body, 'file', (fn) => { pending = fn; }); },
    paste() { importTab(body, 'paste', (fn) => { pending = fn; }); },
  };
  const select = (name) => {
    dlg.querySelectorAll('[role="tab"]').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    err.textContent = '';
    tabs[name]();
  };
  dlg.querySelectorAll('[role="tab"]').forEach((t) => t.addEventListener('click', () => select(t.dataset.tab)));
  select(app.choice.kind === 'synthetic' ? 'synthetic' : app.choice.kind === 'crisis' ? 'crisis' : app.choice.kind === 'upload' ? 'file' : 'market');
  dlg.showModal();
}

function importTab(body, mode, setPending) {
  let table = null, fileName = mode === 'paste' ? 'Pasted data' : null, sheets = null;
  body.innerHTML = `${mode === 'file'
    ? `<label class="dropzone" tabindex="0">${icon('upload')}<div><b>Drop an Excel or CSV file here</b> or click to choose one</div><div class="help">.xlsx, .xls, .ods, .csv, .txt – e.g. a price export from your banking system</div><input type="file" accept=".xlsx,.xlsm,.xls,.ods,.csv,.txt,.tsv" hidden></label>
      <p class="help" style="margin:0">Not sure about the format? <button type="button" class="btn ghost sm" data-act="sample">${icon('download')} Download a sample file (.xlsx)</button></p>`
    : `<label class="field"><span class="lbl"><b>Select the columns in Excel (e.g. date and closing price), copy them and paste them here</b></span><textarea placeholder="Date&#9;Close&#10;02/01/2025&#9;5,868.55&#10;03/01/2025&#9;5,942.47&#10;…"></textarea></label>`}
    <div data-slot="sheet"></div><div data-slot="map"></div>
    <p class="privacy">${icon('shield')}<span>Your data is processed only here in your browser and is never uploaded. English and German number formats (1,234.56 or 1.234,56), newest-first sorting and dates such as 31/12/2024, 31.12.2024 or 2024-12-31 are detected automatically.</span></p>`;
  const mapSlot = body.querySelector('[data-slot="map"]'), sheetSlot = body.querySelector('[data-slot="sheet"]');
  setPending(() => { throw new Error(mode === 'file' ? 'Please choose a file first.' : 'Please paste your data first.'); });

  const showMapping = () => {
    const opts = (sel, allowNone) => (allowNone ? `<option value="-1">– none –</option>` : '') + table.header.map((c, i) => `<option value="${i}" ${i === sel ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('');
    mapSlot.innerHTML = `<div class="preview">${tableHtml(table.header, table.body.slice(0, 8).map((r) => table.header.map((_, j) => (r[j] instanceof Date ? fmtDate(D.parseDate(r[j])) : r[j] ?? ''))), table.header.map(() => 'text'))}</div>
      <div class="row-2" style="margin-top:12px">
        <label class="field"><span class="lbl"><b>Date column</b></span><select data-k="date">${opts(table.dateCol, true)}</select></label>
        <label class="field"><span class="lbl"><b>Value column</b></span><select data-k="value">${opts(table.valueCol, false)}</select></label>
        <label class="field"><span class="lbl"><b>The values are …</b></span><select data-k="kind">${Object.entries(D.KIND_LABELS).map(([k, l]) => `<option value="${k}" ${k === table.kind ? 'selected' : ''}>${l}</option>`).join('')}</select><span class="help">Detected automatically – please check.</span></label>
        <label class="field"><span class="lbl"><b>VIX column (optional)</b></span><select data-k="vix">${opts(table.vixCol, true)}</select><span class="help">For the early-warning tools, e.g. VSTOXX.</span></label>
      </div><p class="help" data-slot="check"></p>`;
    const val = (k) => Number(mapSlot.querySelector(`[data-k="${k}"]`).value);
    const check = () => {
      const up = D.buildUpload(fileName, table, { dateCol: val('date'), valueCol: val('value'), vixCol: val('vix'), kind: mapSlot.querySelector('[data-k="kind"]').value });
      mapSlot.querySelector('[data-slot="check"]').textContent = `${up.n.toLocaleString('en-GB')} valid rows${up.dropped ? `, ${up.dropped.toLocaleString('en-GB')} skipped` : ''}${up.reversed ? ' · sorted newest first, will be reversed' : ''}.`;
      return up;
    };
    mapSlot.querySelectorAll('select').forEach((s) => s.addEventListener('change', check));
    check();
    setPending(() => {
      const up = check();
      if (up.n < 61) throw new Error(`Too few valid rows (${up.n}). At least 61 are needed.`);
      return { kind: 'upload', key: D.registerUpload(up) };
    });
  };
  const useSheet = (rows) => {
    table = D.analyzeTable(rows);
    if (!table.cols.some((c) => c.numeric)) { mapSlot.innerHTML = `<p class="form-error">No numeric column found.</p>`; return; }
    showMapping();
  };
  const loadSheets = (res) => {
    fileName = res.name; sheets = res.sheets;
    if (!sheets.length) { mapSlot.innerHTML = `<p class="form-error">The file contains no tabular data.</p>`; return; }
    sheetSlot.innerHTML = sheets.length > 1 ? `<label class="field" style="max-width:320px"><span class="lbl"><b>Worksheet</b></span><select>${sheets.map((s, i) => `<option value="${i}">${escapeHtml(s.name)}</option>`).join('')}</select></label>` : `<p class="help">${icon('excel', 'width="14" height="14"')} ${escapeHtml(res.name)}</p>`;
    sheetSlot.querySelector('select')?.addEventListener('change', (e) => useSheet(sheets[Number(e.target.value)].rows));
    useSheet(sheets[0].rows);
  };
  body.querySelector('[data-act="sample"]')?.addEventListener('click', async (e) => {
    const btn = e.currentTarget; // currentTarget is null after the first await
    btn.disabled = true;
    try { await downloadSample(); } catch (err) { toast(err.message); } finally { btn.disabled = false; }
  });
  if (mode === 'file') {
    const dz = body.querySelector('.dropzone'), input = dz.querySelector('input');
    const take = async (file) => {
      if (!file) return;
      mapSlot.innerHTML = '<div class="skeleton" style="height:120px"></div>';
      try { loadSheets(await D.readFile(file)); } catch (e) { mapSlot.innerHTML = `<p class="form-error">${escapeHtml(e.message)}</p>`; }
    };
    input.addEventListener('change', () => take(input.files[0]));
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); take(e.dataTransfer.files[0]); });
  } else {
    const ta = body.querySelector('textarea');
    let timer;
    ta.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { const rows = D.parseDelimited(ta.value); if (rows.length > 1) useSheet(rows); }, 250); });
  }
}

async function downloadSample() {
  // A realistic template: what a price export from a banking system typically looks like (newest first).
  const XLSX = await D.loadSheetJS();
  const sp = app.market.datasets.find((d) => d.id === 'sp500');
  const n = sp.dates.length, from = Math.max(0, n - 504);
  const rows = [['Date', 'Close', 'VIX']];
  for (let i = n - 1; i >= from; i--) {
    const [y, m, d] = sp.dates[i].split('-').map(Number);
    rows.push([new Date(y, m - 1, d), sp.prices[i], sp.vix?.[i] ?? null]);
  }
  const ws = XLSX.utils.aoa_to_sheet(rows, { cellDates: true });
  for (let r = 1; r < rows.length; r++) {
    const a = ws[XLSX.utils.encode_cell({ r, c: 0 })]; if (a) a.z = 'dd mmm yyyy';
    const b = ws[XLSX.utils.encode_cell({ r, c: 1 })]; if (b) b.z = '#,##0.00';
    const c = ws[XLSX.utils.encode_cell({ r, c: 2 })]; if (c) c.z = '0.00';
  }
  ws['!cols'] = [{ wch: 12 }, { wch: 14 }, { wch: 8 }];
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Prices');
  XLSX.writeFile(wb, 'tail-risk-sample-sp500.xlsx');
}

// --------------------------------------------------------------------------- tool view
const BOOT_STEPS = [['runtime', 'Python runtime (WebAssembly)'], ['packages', 'NumPy, SciPy, Matplotlib'], ['framework', 'Importing the tail-risk framework'], ['ready', 'Ready']];

function bootCard() {
  const card = h(`<section class="card boot-card" aria-live="polite"><h3>Starting Python in your browser</h3>
    <p class="muted small" style="margin:0">A one-off download of about 30 MB. After that, everything comes from the browser cache and starts within seconds.</p>
    <div class="progress"><i></i></div><ul class="boot-steps"></ul><p class="small muted" data-slot="txt" style="margin:0"></p></section>`);
  const update = () => {
    const s = engine.statusInfo, idx = BOOT_STEPS.findIndex(([k]) => k === s.stage);
    card.querySelector('.progress > i').style.width = `${Math.max(3, s.pct || 0)}%`;
    card.querySelector('.boot-steps').innerHTML = BOOT_STEPS.map(([k, l], i) => `<li class="${i < idx ? 'done' : i === idx ? 'active' : ''}">${icon(i < idx ? 'check' : 'circle')} ${l}</li>`).join('');
    card.querySelector('[data-slot="txt"]').textContent = s.text || '';
  };
  engine.addEventListener('status', update);
  update();
  card.dispose = () => engine.removeEventListener('status', update);
  return card;
}

function progressCard(text, { console: withConsole = false } = {}) {
  const card = h(`<section class="card boot-card" aria-live="polite"><h3 data-slot="t">${escapeHtml(text)}</h3><div class="progress indet"><i></i></div>
    <p class="small muted" data-slot="clock" style="margin:0"></p>${withConsole ? '<div class="console" data-slot="log" hidden></div>' : ''}</section>`);
  const t0 = performance.now(), clock = card.querySelector('[data-slot="clock"]');
  const timer = setInterval(() => {
    if (!card.isConnected) return clearInterval(timer);
    const s = Math.round((performance.now() - t0) / 1000);
    clock.textContent = s >= 3 ? `Running in your browser for ${s} s …` : '';
  }, 1000);
  card.log = (m) => {
    const log = card.querySelector('[data-slot="log"]');
    if (!log) return;
    log.hidden = false;
    const line = document.createElement('div');
    line.textContent = m.text;
    if (m.stream === 'stderr') line.className = 'err';
    log.appendChild(line);
    if (log.childElementCount > 400) log.firstElementChild.remove();
    log.scrollTop = log.scrollHeight;
  };
  return card;
}

async function renderTool(id, query, { autorun } = {}) {
  const spec = app.catalogue.tools.find((t) => t.id === id);
  if (!spec) { view.innerHTML = `<div class="wrap empty"><h1>Tool not found</h1><p><a href="#/tools">All tools</a></p></div>`; return; }
  const dsParam = D.decodeChoice(query.get('ds'));
  if (dsParam && D.encodeChoice(dsParam) !== D.encodeChoice(app.choice)) { app.choice = dsParam; saveChoice(dsParam); }
  await resolveDataset();
  const hasQueryParams = spec.params.some((p) => query.has(p.name));
  let values = hasQueryParams ? paramsFromQuery(spec, query) : (app.params.get(id) || paramsFromQuery(spec, query));
  app.params.set(id, values);
  document.title = `${spec.title} · Tail Risk Lab`;

  view.innerHTML = `<div class="wrap">
    <header class="tool-head"><div>
      <nav class="crumbs" aria-label="Breadcrumbs"><a href="#/tools">Tools</a> <span>›</span> <span>${escapeHtml(groupTitle(spec.group))}</span></nav>
      <h1>${escapeHtml(spec.title)}</h1><p class="summary">${escapeHtml(spec.summary)}</p></div></header>
    <div class="tool-layout"><aside class="panel" aria-label="Settings"></aside><main class="results" id="results" aria-live="polite"></main></div></div>`;
  const panel = view.querySelector('.panel'), results = view.querySelector('#results');
  // A form or timer of a view the user already left (e.g. a 'change' fired by the blur of a removed
  // input, or a pending live update) must neither rewrite the URL nor cancel the new view's run.
  app.view += 1;
  if (app.run && !app.run.done) app.run.detach = true;
  const myView = app.view;
  const current = () => app.view === myView;

  let dsCard = null;
  if (spec.needs.length) { dsCard = datasetCard(spec); panel.appendChild(dsCard); }
  const paramsCard = h(`<section class="card"><h2>Parameters</h2></section>`);
  const runBtn = h(`<button class="btn primary block lg" type="button">${icon('play')} Calculate</button>`);
  const runHint = h(`<div class="hint"></div>`);
  const form = buildForm(spec, values, (state, { commit }) => {
    if (!current()) return;
    values = state; app.params.set(id, values);
    const q = queryFromParams(spec, values);
    if (app.dataset.shareable && spec.needs.length) q.set('ds', D.encodeChoice(app.choice));
    history.replaceState(null, '', toolHref(id, q));
    markDirty();
    if (commit && spec.cost === 'fast') { clearTimeout(liveTimer); liveTimer = setTimeout(() => run(), 350); }
  });
  let liveTimer;
  if (spec.params.length) paramsCard.appendChild(form); else paramsCard.appendChild(h('<p class="muted small" style="margin:0">Nothing to set – just calculate.</p>'));
  const runRow = h('<div class="run-row" style="margin-top:16px"></div>');
  runRow.append(runBtn, runHint);
  paramsCard.appendChild(runRow);
  panel.appendChild(paramsCard);
  panel.appendChild(h(`<section class="card"><h2>About this tool</h2><div class="desc">${richText(spec.description || spec.summary)}</div>
    <details class="more" style="margin-top:10px"><summary>Python code used</summary><div class="covers">${spec.covers.map((c) => `<code>${escapeHtml(c)}</code>`).join('')}</div>
    <p class="small" style="margin:8px 0 0"><a href="${REPO}/blob/main/webtools/tools/${spec.id.replace(/-/g, '_')}.py" rel="noopener">View the adapter on GitHub</a></p></details></section>`));

  runHint.textContent = spec.cost === 'fast' ? 'Updates automatically when you change a setting.' : spec.cost === 'slow' ? 'Compute-intensive – may take up to a minute.' : '';
  const markDirty = () => { runBtn.innerHTML = `${icon('play')} ${app.results.has(id) ? 'Recalculate' : 'Calculate'}`; };

  const resultKey = () => JSON.stringify([id, spec.needs.length ? app.dataset.key : '-', values]);

  // toggle: the button (reads "Cancel" while this tool runs) cancels; otherwise a new run replaces the old.
  const run = async ({ toggle = false } = {}) => {
    if (!current()) return;
    const prev = app.run && !app.run.done ? app.run : null;
    if (prev) {
      if (toggle && prev.toolId === id && !prev.detach) { engine.cancel(); return; }
      if (prev.cost === 'fast') {
        // a fast run finishes within a second or two: let it, then compute with the latest values
        // (a cancel would reboot Python, ~20 s)
        prev.detach = true;
        await prev.promise;
        if (!current() || (app.run && app.run !== prev && !app.run.done)) return;
      } else {
        engine.cancel();
      }
    }
    const job = { toolId: id, cost: spec.cost, key: resultKey(), done: false, stdout: [], detach: false };
    let finish;
    job.promise = new Promise((r) => { finish = r; });
    app.run = job;
    runBtn.innerHTML = `${icon('stop')} Cancel`;
    results.classList.add('stale');
    const firstRun = !results.querySelector('.verdict, .metric, .chart-card, .figure-card, .table-card');
    let waiting = null;
    const showWaiting = (card) => { waiting?.dispose?.(); waiting?.remove(); waiting = card; if (firstRun) { results.innerHTML = ''; } results.prepend(card); };
    try {
      if (engine.state !== 'ready' && engine.state !== 'busy') showWaiting(bootCard());
      await engine.ready();
      if (spec.needs.length) {
        if (engine._dataset?.key !== app.dataset.key) {
          showWaiting(progressCard('Passing the dataset to Python …'));
          const res = await engine.setDataset(app.dataset.key, app.dataset.descriptor, app.dataset.arrays);
          if (!res.ok) throw Object.assign(new Error(res.error), res);
          app.datasetSummary = res.summary;
        }
        fillDatasetStats(dsCard, app.datasetSummary);
      }
      const prog = progressCard(`${spec.title}: calculating …`, { console: spec.group === 'scripts' });
      showWaiting(prog);
      const t0 = performance.now();
      const res = await engine.run(id, values, {
        preload: spec.preload,
        onProgress: (m) => { prog.querySelector('[data-slot="t"]').textContent = m.text; if (m.pct != null) { const bar = prog.querySelector('.progress'); bar.classList.remove('indet'); bar.firstElementChild.style.width = `${m.pct}%`; } },
        onStdout: (m) => { job.stdout.push(m); prog.log(m); },
      });
      res.wall = (performance.now() - t0) / 1000;
      if (res.ok) app.results.set(id, { key: job.key, result: res, stdout: job.stdout });
      if (job.detach) return;
      waiting?.remove();
      if (!res.ok) throw Object.assign(new Error(res.error), res);
      showResult(res, job.stdout);
    } catch (err) {
      if (job.detach) return;
      waiting?.dispose?.(); waiting?.remove();
      results.classList.remove('stale');
      if (err instanceof CancelledError) { toast('Calculation cancelled.'); if (!results.children.length) results.appendChild(h('<div class="empty">Cancelled. Press ‘Calculate’ to start again.</div>')); }
      else renderError(results, { error: err.message, traceback: err.traceback, kind: err.kind }, { onRetry: () => run() });
    } finally {
      job.done = true;
      finish();
      if (!job.detach) markDirty();
    }
  };
  runBtn.addEventListener('click', () => run({ toggle: true }));

  const showResult = (res, stdout) => {
    results.classList.remove('stale');
    const body = h('<div class="results"></div>');
    renderResult(body, res);
    const bar = h(`<div class="result-bar"><div class="meta">${icon('cpu', 'width="13" height="13" style="vertical-align:-2px"')} Calculated in ${fmt(res.seconds, 'num', 2)} s in your browser · ${escapeHtml(res.dataset?.name && spec.needs.length ? res.dataset.name : 'Simulation')}${res.dataset?.n && spec.needs.length ? ` · ${fmt(res.dataset.n, 'int')} returns` : ''}</div>
      <div class="actions"><button class="btn sm" data-act="xlsx">${icon('excel')} Excel</button><button class="btn sm" data-act="share">${icon('link')} Link</button><button class="btn sm" data-act="print">${icon('print')} Print / PDF</button></div></div>`);
    bar.querySelector('[data-act="xlsx"]').addEventListener('click', async (e) => {
      const btn = e.currentTarget; // currentTarget is null after the first await
      btn.disabled = true;
      try { await exportXlsx({ spec, result: res, dataset: spec.needs.length ? app.dataset : { label: 'Simulation', detail: '' }, params: values }); }
      catch (err) { toast(err.message); } finally { btn.disabled = false; }
    });
    bar.querySelector('[data-act="share"]').addEventListener('click', async () => {
      const url = location.href;
      try { await navigator.clipboard.writeText(url); toast(app.dataset.shareable || !spec.needs.length ? 'Link copied – it includes the dataset and the parameters.' : 'Link copied. Note: your own files are not included in the link.'); }
      catch { prompt('Copy the link:', url); }
    });
    bar.querySelector('[data-act="print"]').addEventListener('click', () => window.print());
    results.innerHTML = '';
    results.appendChild(h(`<div class="print-only"><h2>${escapeHtml(spec.title)}</h2><p>${escapeHtml(app.dataset?.label || '')} · ${escapeHtml(app.dataset?.detail || '')} · ${fmtDate(D.parseDate(new Date()))}, ${new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}</p></div>`));
    results.appendChild(bar);
    results.append(...body.childNodes);
    if (stdout?.length) {
      const text = stdout.map((m) => m.stream === 'stderr' ? `<span class="err">${escapeHtml(m.text)}</span>` : escapeHtml(m.text)).join('\n');
      results.appendChild(h(`<section class="card table-card"><details ${spec.group === 'scripts' ? 'open' : ''}><summary><b>Python output</b> <span class="muted small">(stdout, ${stdout.length.toLocaleString('en-GB')} ${stdout.length === 1 ? 'line' : 'lines'})</span></summary><div class="console">${text}</div></details></section>`));
    }
  };

  const cached = app.results.get(id);
  const pending = app.run && !app.run.done && app.run.toolId === id && app.run.key === resultKey() ? app.run : null;
  if (cached && cached.key === resultKey()) showResult(cached.result, cached.stdout);
  else if (pending) {
    // came back while this exact computation is still running: wait for it instead of restarting
    const card = progressCard(`${spec.title}: calculating …`, { console: spec.group === 'scripts' });
    pending.stdout.forEach((m) => card.log(m));
    results.appendChild(card);
    runBtn.innerHTML = `${icon('stop')} Cancel`;
    pending.promise.then(() => {
      if (!card.isConnected) return;
      const done = app.results.get(id);
      if (done && done.key === resultKey()) showResult(done.result, done.stdout); else card.remove();
      markDirty();
    });
    pending.detach = true; // its own closure renders into the old DOM; this view takes over
    return;
  } else if (autorun !== false) run();
  markDirty();
}

// --------------------------------------------------------------------------- boot
async function main() {
  initTheme();
  initPill();
  document.querySelector('.menu-btn').addEventListener('click', () => document.querySelector('.nav').classList.toggle('open'));
  try {
    const [catalogue, market, version, gallery] = await Promise.all([
      fetch(new URL(`py/catalogue.json?v=${BUILD}`, BASE)).then((r) => r.json()),
      D.loadMarket(BASE),
      fetch(new URL(`version.json?v=${BUILD}`, BASE)).then((r) => r.json()).catch(() => null),
      fetch(new URL(`data/gallery.json?v=${BUILD}`, BASE)).then((r) => r.json()).catch(() => null),
    ]);
    Object.assign(app, { catalogue, market, version, gallery });
    document.getElementById('build-info').textContent = `Build ${version?.build || 'dev'} · Market data ${fmtDate(market.updated)}`;
  } catch (err) {
    view.innerHTML = `<div class="wrap empty"><h1>The page could not be loaded</h1><p>${escapeHtml(err.message)}</p></div>`;
    return;
  }
  window.addEventListener('hashchange', route);
  route();
}

main();
export { app, engine };
