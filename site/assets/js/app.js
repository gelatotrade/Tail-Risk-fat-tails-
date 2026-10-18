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
  document.getElementById('theme-btn').innerHTML = icon(dark ? 'sun' : 'moon') + '<span class="sr-only">Farbschema wechseln</span>';
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
    const label = { idle: 'Python bereit zum Start', booting: `Python lädt … ${Math.round(s.pct || 0)} %`, ready: 'Python läuft im Browser', busy: 'Python rechnet …', error: 'Python-Fehler' }[engine.state];
    txt.textContent = label;
    pill.title = engine.state === 'ready' ? `Pyodide ${'314.0.7'} · gestartet in ${engine.bootSeconds?.toFixed(1)} s` : s.text;
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
  if (active !== 'tools' || !location.hash.startsWith('#/tool/')) document.title = 'Tail Risk Lab · Fette Ränder live im Browser';
}

async function route() {
  const { parts, query } = parseHash();
  const [first, second] = parts;
  app.view += 1;
  if (app.run && !app.run.done) app.run.detach = true; // keeps running; result is cached for re-entry
  window.scrollTo({ top: 0 });
  if (!first) { setNav('home'); return renderHome(); }
  if (first === 'werkzeuge') { setNav('tools'); return renderCatalogue(); }
  if (first === 'tool' && second) { setNav('tools'); return renderTool(second, query); }
  if (first === 'galerie') { setNav('gallery'); return renderGallery(); }
  if (first === 'methodik') { setNav('method'); return renderMethod(); }
  setNav('');
  view.innerHTML = `<div class="wrap empty"><h1>Seite nicht gefunden</h1><p><a href="#/">Zur Startseite</a></p></div>`;
}

const toolHref = (id, q) => `#/tool/${id}${q && String(q) ? '?' + q : ''}`;
const groupTitle = (gid) => app.catalogue.groups.find((g) => g.id === gid)?.title || gid;

function toolCard(t) {
  const sim = t.needs.length === 0;
  return `<a class="card tool-card" href="${toolHref(t.id)}" data-tool="${t.id}">
    <h3><span class="ic">${icon(t.group)}</span><span>${escapeHtml(t.title)}</span></h3>
    <p>${escapeHtml(t.summary)}</p>
    <div class="meta">${sim ? '<span class="chip">Simulation</span>' : '<span class="chip accent">nutzt deine Daten</span>'}
      ${t.needs.includes('vix') ? '<span class="chip">braucht VIX</span>' : ''}
      ${t.cost === 'slow' ? '<span class="chip">rechenintensiv</span>' : t.cost === 'fast' ? '<span class="chip">live</span>' : ''}</div></a>`;
}

function catalogueHtml() {
  const groups = app.catalogue.groups.map((g) => {
    const tools = app.catalogue.tools.filter((t) => t.group === g.id);
    if (!tools.length) return '';
    return `<div class="tool-group" data-group="${g.id}"><div class="group-title">${icon(g.id, 'width="15" height="15"')} ${escapeHtml(g.title)}</div><div class="tool-grid">${tools.map(toolCard).join('')}</div></div>`;
  }).join('');
  return `<div class="catalogue-bar">
      <label class="search">${icon('search')}<input type="search" placeholder="Werkzeug suchen – z. B. VaR, Lévy, VIX, Excel …" aria-label="Werkzeuge durchsuchen"></label>
      <div class="seg" role="tablist" aria-label="Filter">
        <button type="button" data-f="all" aria-selected="true">Alle</button>
        <button type="button" data-f="data" aria-selected="false">Nutzt deine Daten</button>
        <button type="button" data-f="sim" aria-selected="false">Simulationen</button>
      </div></div>${groups}<p class="empty" data-slot="none" hidden>Kein Werkzeug passt zu dieser Suche.</p>`;
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
  const termLabel = term == null ? '' : term > 1 ? 'Backwardation – Panik' : term > 0.9 ? 'Flach – erhöhte Nervosität' : 'Contango – normal';
  const vixStatus = pulse.vix == null ? null : pulse.vix >= 30 ? 'critical' : pulse.vix >= 22 ? 'serious' : pulse.vix >= 17 ? 'warning' : 'good';
  const vixLabel = { good: 'Ruhig', warning: 'Leicht erhöht', serious: 'Erhöht', critical: 'Stress' }[vixStatus];

  view.innerHTML = `
  <section class="hero"><div class="wrap hero-grid">
    <div>
      <span class="eyebrow">${icon('cpu', 'width="14" height="14"')} Python-Framework, live im Browser</span>
      <h1>Krisen leben in den <em>Rändern</em> der Verteilung.</h1>
      <p class="lead">Das komplette Tail-Risk-Framework – Lévy-Flüge, Fokker-Planck, Tsallis-Statistik, Extremwerttheorie, Phasenübergänge und Frühwarnsystem – läuft hier direkt in deinem Browser. Mit aktuellen Marktdaten oder deinen eigenen Excel-Reihen.</p>
      <div class="ctas">
        <a class="btn primary lg" href="#/werkzeuge" data-warm>${icon('play')} Werkzeuge öffnen</a>
        <a class="btn lg" href="${toolHref('fat-tails')}" data-warm>${icon('excel')} Mit eigenen Excel-Daten</a>
      </div>
      <div class="trust">
        <span>${icon('shield')} Daten bleiben auf deinem Rechner</span>
        <span>${icon('check')} Keine Installation</span>
        <span>${icon('check')} Export nach Excel</span>
      </div>
    </div>
    <div class="card hero-chart">
      <div class="head"><h3>S&P 500 · Tagesrenditen der letzten ${fmt(pulse.years, 'num', 0)} Jahre</h3><span class="muted small">log. Skala · Stand ${fmtDate(pulse.lastDate)}</span></div>
      <div id="hero-plot"></div>
      <div class="legend"><span><i style="background:var(--s1);opacity:.6"></i>Beobachtet</span><span><i style="background:var(--critical)"></i>Jenseits von 3 σ</span><span><i style="background:var(--ink);opacity:.7"></i>Normalverteilung</span></div>
      <div class="hero-callout">
        <div><div class="v">${fmt(n4, 'int')}</div><div class="l">Tage jenseits von 4 σ beobachtet</div></div>
        <div><div class="v">${fmt(exp4, 'num', 2)}</div><div class="l">von der Gauß-Annahme erwartet</div></div>
        <div><div class="v">${fmt(exp4 > 0 ? n4 / exp4 : null, 'x')}</div><div class="l">Unterschätzung durch die Normalverteilung</div></div>
      </div>
    </div>
  </div></section>

  <section class="wrap" aria-label="Marktlage">
    <div class="section-head" style="margin-bottom:10px"><div><h2 style="font-size:18px">Marktpuls</h2><p class="small">Aus den FRED-Daten dieser Seite berechnet · Stand ${fmtDate(pulse.lastDate)}</p></div>
      <a class="btn sm" href="${toolHref('fat-tails')}" data-warm>Volle Analyse ${icon('arrow')}</a></div>
    <div class="market-strip">
      <div class="card tile"><div class="l">S&P 500</div><div class="v">${fmt(pulse.last, 'num', 0)}</div><div class="d">${pulse.change >= 0 ? '+' : ''}${fmt(pulse.change, 'pct', 2)} zum Vortag · ${fmt(pulse.drawdown, 'pct', 1)} vom Hoch</div></div>
      <div class="card tile"><div class="l">VIX</div><div class="v">${fmt(pulse.vix, 'num', 1)}</div><div class="d">${statusBadge(vixStatus, vixLabel)}</div></div>
      <div class="card tile"><div class="l">VIX-Laufzeitstruktur (VIX / VIX3M)</div><div class="v">${fmt(term, 'num', 2)}</div><div class="d">${statusBadge(termStatus, termLabel)}</div></div>
      <div class="card tile"><div class="l">Realisierte Volatilität (20 Tage, p. a.)</div><div class="v">${fmt(pulse.vol20, 'pct', 1)}</div><div class="d">Langfristig: ${fmt(pulse.volAll, 'pct', 1)}</div></div>
    </div>
  </section>

  <section class="section wrap">
    <div class="section-head"><div><h2>So funktioniert es</h2><p>Kein Server rechnet mit. Der originale Python-Code aus diesem Repository wird als WebAssembly in deinem Browser ausgeführt.</p></div></div>
    <div class="steps">
      <div class="card"><div class="n">1</div><h3>Daten wählen</h3><p>S&P 500, NASDAQ und VIX von FRED, synthetische Szenarien – oder deine eigene Excel- bzw. CSV-Datei, auch per Copy &amp; Paste aus Excel.</p></div>
      <div class="card"><div class="n">2</div><h3>Werkzeug starten</h3><p>Beim ersten Start lädt Python mit NumPy, SciPy und Matplotlib (einmalig rund 30 MB, danach aus dem Cache).</p></div>
      <div class="card"><div class="n">3</div><h3>Verstehen &amp; exportieren</h3><p>Interaktive Diagramme, die Original-Grafiken des Frameworks und alle Zahlen als Excel-Datei – oder als Link zum Teilen.</p></div>
    </div>
  </section>

  <section class="section wrap" id="tools">
    <div class="section-head"><div><h2>${app.catalogue.tools.length} Werkzeuge – jedes Feature des Frameworks</h2><p>Von der Verteilungsanalyse über Physik-Simulationen bis zum Frühwarnsystem und den Original-Skripten des Repositorys.</p></div></div>
    ${catalogueHtml()}
  </section>

  <section class="section wrap">
    <div class="section-head"><div><h2>Galerie</h2><p>Die Abbildungen aus dem Repository – jede davon lässt sich über die Werkzeuge live neu berechnen.</p></div><a class="btn" href="#/galerie">Alle ansehen ${icon('arrow')}</a></div>
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
    <div class="section-head"><div><h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Werkzeuge</h1>
    <p>Jedes Werkzeug ruft den originalen Python-Code des Frameworks auf. Werkzeuge mit „nutzt deine Daten“ rechnen auf dem gewählten Datensatz, Simulationen erzeugen ihre eigenen Pfade.</p></div></div>
    ${catalogueHtml()}</section>`;
  wireCatalogue(view);
  prewarmOnIntent(view);
  view.querySelector('.catalogue-bar input')?.focus({ preventScroll: true });
}

// --------------------------------------------------------------------------- gallery & method
function galleryItem(it) {
  const src = `gallery/${it.file}`;
  return `<figure class="card"><img src="${src}" alt="${escapeHtml(it.title)}" loading="lazy" data-cap="${escapeHtml(it.title + ' – ' + (it.caption || ''))}">
    <figcaption><b>${escapeHtml(it.title)}</b>${escapeHtml(it.caption || '')}${it.tool ? ` <a href="${toolHref(it.tool)}">Live berechnen →</a>` : ''}</figcaption></figure>`;
}
function wireGallery(root) {
  root.querySelectorAll('.gallery img').forEach((img) => img.addEventListener('click', () => openLightbox(img.src, img.dataset.cap)));
}
function renderGallery() {
  const secs = app.gallery?.sections || [];
  view.innerHTML = `<section class="section wrap"><div class="section-head"><div><h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Galerie</h1>
    <p>Alle Abbildungen, die die Skripte des Repositorys erzeugen. Unter „Original-Skripte“ lassen sie sich im Browser neu erzeugen.</p></div></div>
    ${secs.map((s) => `<div class="group-title">${escapeHtml(s.title)}</div><div class="gallery">${s.items.map(galleryItem).join('')}</div>`).join('')}</section>`;
  wireGallery(view);
}

function renderMethod() {
  view.innerHTML = `<section class="section wrap"><article class="prose">
  <h1 style="font-size:clamp(26px,3vw,36px);letter-spacing:-.03em">Methodik &amp; Technik</h1>
  <p>Klassische Risikomodelle – Portfoliotheorie, Black-Scholes, die meisten VaR-Systeme – unterstellen normalverteilte Renditen. Reale Märkte haben fette Ränder: Extreme Tage sind um Größenordnungen häufiger. Dieses Framework übersetzt Werkzeuge aus der statistischen Physik in die Risikomessung.</p>
  <h2>Physik → Finanzen</h2>
  ${tableHtml(['Physikalisches Konzept', 'Anwendung im Risikomanagement'], [
    ['Lévy-Flüge (Teilchenpfade mit Sprüngen)', 'Kurssprünge, Black-Swan-Ereignisse; Stabilitätsindex α < 2 = fette Ränder'],
    ['Fokker-Planck-Gleichung (Wahrscheinlichkeitsfluss)', 'Zeitliche Entwicklung der Renditeverteilung, Prognose der Tail-Wahrscheinlichkeit'],
    ['Tsallis-Entropie (nicht-extensive Thermodynamik)', 'Märkte fern vom Gleichgewicht; q > 1 erzeugt Potenzgesetz-Ränder'],
    ['Phasenübergänge (Ising-Modell, Magnetismus)', 'Crashs als kritische Übergänge; Herdenverhalten = Kopplung J'],
    ['Ornstein-Uhlenbeck-Prozess (Feder)', 'Mean Reversion der Volatilität mit Sprüngen'],
    ['Extremwerttheorie (GPD, Peaks over Threshold)', 'Statistisch saubere Schätzung von VaR/ES weit im Rand'],
  ], ['text', 'text'])}
  <h2>Frühwarnsystem</h2>
  <p>Der Composite-Score gewichtet fünf Indikatoren, die sich in den Crashs 2018, 2020, 2022 und 2025 bewährt haben:</p>
  <div class="formula">score = 0,35·GEX + 0,25·TDEX + 0,20·VIX-Laufzeitstruktur + 0,10·DIX + 0,10·SMFI</div>
  <p style="margin-top:12px">0–25 Normal · 25–50 Erhöht · 50–75 Hoch · 75–100 Extrem. GEX, TailDex, DIX und Smart-Money-Flow beruhen auf Options- und Dark-Pool-Daten, die nicht frei verfügbar sind. Die Werkzeuge leiten sie – wie das Framework selbst – aus Renditen und VIX ab oder lassen dich die Werte direkt eingeben.</p>
  <h2>Wie diese Seite rechnet</h2>
  <p>Die Seite ist rein statisch und wird über GitHub Pages ausgeliefert. Beim ersten Werkzeugstart lädt der Browser <a href="https://pyodide.org" rel="noopener">Pyodide</a> – CPython 3.14 als WebAssembly – mit NumPy, SciPy und Matplotlib. Anschließend wird das Paket <code>src/</code> dieses Repositorys unverändert importiert. Eine dünne Schicht (<code>webtools/</code>) übersetzt Formularwerte in Funktionsaufrufe und Ergebnisse in Diagramme. Alle Berechnungen laufen in einem Web Worker, die Oberfläche bleibt dabei bedienbar.</p>
  <p><b>Datenschutz:</b> Hochgeladene Excel- oder CSV-Dateien werden ausschließlich lokal im Browser verarbeitet und nie übertragen.</p>
  <h2>Korrekturen beim Live-Schalten</h2>
  <p>Bevor das Framework öffentlich rechnet, wurde jede Funktion ausgeführt und gegen Lehrbuchformeln, SciPy und numerische Integration geprüft. Dabei wurden <b>über 90 Fehler in 19 Modulen</b> behoben, jeder abgesichert durch Regressionstests (über 350 Tests, die vor der Korrektur fehlschlagen und danach bestehen). Modellierungsentscheidungen des Autors blieben unverändert. Die Abbildungen im README entstanden mit dem ursprünglichen Code, einzelne Werte können daher abweichen.</p>
  ${tableHtml(['Bereich', 'Was korrigiert wurde'], [
    ['Expected Shortfall', 'Historischer und Cornish-Fisher-ES mittelten über die Gewinne statt über die Verluste jenseits des VaR; der Student-t-ES ignorierte die Skalierung.'],
    ['Drawdown', 'Log-Renditen wurden wie einfache Renditen aufgezinst.'],
    ['Extremwerttheorie', 'Wiederkehrperioden zählten in 20-Tage-Blöcken: das „10-Jahres-Niveau“ war tatsächlich ein 200-Jahres-Niveau. GPD-Wahrscheinlichkeiten waren bedingt statt unbedingt, der GEV-VaR negativ.'],
    ['Lévy-Verteilung', 'Tail-Wahrscheinlichkeit um Faktor 2 zu klein; der Quantil-Fit ist jetzt das echte McCulloch-Verfahren; Mehrtages-Risiko skaliert mit h^(1/α) statt √h.'],
    ['Tsallis-Statistik', 'Varianz, Stichprobenbreite, Verteilungsfunktion und Tail-Asymptote korrigiert; der Momenten-Fit für q lief rückwärts.'],
    ['Fokker-Planck', 'Die Rückwärts- statt der Vorwärtsgleichung wurde gelöst (Drift mit falschem Vorzeichen); der fraktionale Operator war anti-diffusiv und instabil.'],
    ['Fat-Tail-Verteilungen', 'Gespiegelte Variance-Gamma-Dichte, falsche Normierung der Generalized Hyperbolic, falsche NIG-Stichproben und -Fits.'],
    ['Markov-Regime', 'Falscher M-Schritt im EM-Algorithmus; jetzt Hamilton-Filter, Kim-Glätter und Baum-Welch – und rund 50-mal schneller, was die Dashboards im Browser erst möglich macht.'],
    ['Frühwarnsystem', 'Contango und Backwardation waren vertauscht; DIX und SMFI wurden im Composite doppelt heruntergewichtet. Der simulierte DIX blickt 5 Tage in die Zukunft – diese Seite rechnet daher stets im neuen kausalen Modus ohne Look-ahead.'],
    ['Kompatibilität', 'Abstürze unter NumPy 2 (np.trapz, np.math) und SciPy 1.18 (Kolmogorov-Smirnov-Test) behoben – Voraussetzung dafür, dass alles in Pyodide läuft.'],
  ], ['text', 'text'])}
  <h2>Datenquellen</h2>
  <p>Marktdaten: ${escapeHtml(app.market.source)} – Reihen SP500, NASDAQCOM, VIXCLS, VXVCLS. Aktualisiert bei jedem Deployment und werktäglich automatisch (zuletzt ${fmtDate(app.market.updated)}). Die S&P-500-Reihe ist bei FRED lizenzbedingt auf die letzten zehn Jahre begrenzt.</p>
  <h2>Hinweis</h2>
  <p class="disclaimer">Diese Seite dient Forschung und Ausbildung. Sie ist keine Anlageberatung und keine Empfehlung zum Kauf oder Verkauf von Finanzinstrumenten. Modelle vereinfachen die Wirklichkeit; Ergebnisse auf historischen Daten garantieren kein künftiges Verhalten.</p>
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
  const card = h(`<section class="card"><h2>Datensatz <button class="btn sm" type="button" data-act="change">Ändern</button></h2>
    <div class="ds-summary"><div class="name">${escapeHtml(ds.label)}</div><div class="muted">${escapeHtml(ds.detail)}</div>
    <div class="row" data-slot="stats"></div></div>
    ${needsVix && !hasVix ? `<div class="note warning" style="margin-top:10px">${icon('warn')}<div>Dieses Werkzeug braucht eine VIX-Reihe. <button class="btn sm" data-act="sp500" type="button" style="margin-top:6px">S&P 500 mit VIX verwenden</button></div></div>` : ''}
    </section>`);
  card.querySelector('[data-act="change"]').addEventListener('click', () => openDatasetDialog());
  card.querySelector('[data-act="sp500"]')?.addEventListener('click', () => applyChoice({ kind: 'market', id: 'sp500', preset: 'all' }));
  return card;
}

function fillDatasetStats(card, summary) {
  const slot = card?.querySelector('[data-slot="stats"]');
  if (!slot || !summary) return;
  const w = summary.worst;
  slot.innerHTML = `<span class="chip">Vol. p. a. ${fmt(summary.ann_vol, 'pct', 1)}</span>
    <span class="chip">Schlechtester Tag ${fmt(w.value, 'pct', 1)}${typeof w.date === 'string' ? ' · ' + fmtDate(w.date) : ''}</span>
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
    <div class="sheet-head"><h2 id="ds-title">Datensatz wählen</h2><button class="icon-btn" data-act="close" aria-label="Schließen">${icon('close')}</button></div>
    <div class="tabs" role="tablist">
      <button role="tab" data-tab="market">Marktdaten</button><button role="tab" data-tab="synthetic">Synthetisch</button>
      <button role="tab" data-tab="crisis">Krisenszenario</button><button role="tab" data-tab="file">Excel / CSV-Datei</button>
      <button role="tab" data-tab="paste">Aus Excel einfügen</button></div>
    <div class="sheet-body"></div>
    <div class="sheet-foot"><span class="form-error" data-slot="err" style="margin-right:auto"></span><button class="btn" data-act="close">Abbrechen</button><button class="btn primary" data-act="apply">Übernehmen</button></div></dialog>`);
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
        <div><div class="help" style="margin-bottom:6px">Zeitraum</div><div class="presets"></div></div>
        <div class="row-2"><label class="field"><span class="lbl"><b>Von</b></span><input type="date" data-k="from"></label><label class="field"><span class="lbl"><b>Bis</b></span><input type="date" data-k="to"></label></div>
        <p class="privacy">${icon('database')}<span>Quelle: ${escapeHtml(app.market.source)} · Stand ${fmtDate(app.market.updated)}</span></p>`;
      const from = body.querySelector('[data-k="from"]'), to = body.querySelector('[data-k="to"]');
      const sync = () => {
        const d = dsById(sel);
        from.min = to.min = d.dates[0]; from.max = to.max = d.dates.at(-1);
        body.querySelector('.presets').innerHTML = D.PRESETS.map((p) => {
          const r = D.presetRange(p.id, d.dates);
          return `<button type="button" class="btn sm ${p.id === preset ? 'primary' : ''}" data-p="${p.id}" ${r ? '' : 'disabled title="Nicht im Datensatz enthalten"'}>${escapeHtml(p.label)}</button>`;
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
        if (!from.value || !to.value || from.value >= to.value) throw new Error('Bitte einen gültigen Zeitraum wählen.');
        return { kind: 'market', id: sel, from: from.value, to: to.value };
      };
    },
    synthetic() {
      const cur = app.choice.kind === 'synthetic' ? app.choice : { model: 'levy_jump', n: 2520, seed: 42 };
      let model = cur.model;
      body.innerHTML = `<div class="options">${D.SYNTHETIC_MODELS.map((m) => `<button type="button" class="option" data-m="${m.value}" aria-pressed="${m.value === model}"><b>${escapeHtml(m.label)}</b><span>${escapeHtml(m.hint)}</span></button>`).join('')}</div>
        <div class="row-2"><label class="field"><span class="lbl"><b>Anzahl Handelstage</b></span><input type="number" min="60" max="20000" step="10" value="${cur.n}" data-k="n"></label>
        <label class="field"><span class="lbl"><b>Zufalls-Seed</b></span><input type="number" min="0" max="999999" value="${cur.seed}" data-k="seed"><span class="help">Gleicher Seed = identische Reihe, ideal zum Teilen.</span></label></div>
        <p class="privacy">${icon('cpu')}<span>Erzeugt mit <code>generate_synthetic_returns</code> aus <code>src/utils/data_loader.py</code>; ein passender VIX wird mit <code>generate_vix_from_returns</code> abgeleitet.</span></p>`;
      body.querySelectorAll('.option').forEach((b) => b.addEventListener('click', () => { model = b.dataset.m; body.querySelectorAll('.option').forEach((o) => o.setAttribute('aria-pressed', String(o === b))); }));
      pending = () => ({ kind: 'synthetic', model, n: Number(body.querySelector('[data-k="n"]').value), seed: Number(body.querySelector('[data-k="seed"]').value) });
    },
    crisis() {
      const seed = app.choice.kind === 'crisis' ? app.choice.seed : 42;
      body.innerHTML = `<p>Ein künstlicher Markt mit drei Phasen: 500 ruhige Tage mit Selbstzufriedenheit, 100 Tage Crash mit Volatilitätsspitze und fetten Rändern, 400 Tage nervöse Erholung. Ideal, um zu sehen, ob Frühwarnsignale <i>vor</i> dem Crash anschlagen.</p>
        <label class="field" style="max-width:260px"><span class="lbl"><b>Zufalls-Seed</b></span><input type="number" min="0" max="999999" value="${seed}" data-k="seed"><span class="help">Die Krise ist zufällig: Seed 42 ergibt −34 % im Krisenfenster (ähnlich COVID 2020), Seed 50 −59 %; manche Seeds erholen sich schon im Crash.</span></label>
        <p class="privacy">${icon('cpu')}<span>Erzeugt mit <code>generate_crisis_scenario</code> aus <code>src/utils/data_loader.py</code>.</span></p>`;
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
  let table = null, fileName = mode === 'paste' ? 'Eingefügte Daten' : null, sheets = null;
  body.innerHTML = `${mode === 'file'
    ? `<label class="dropzone" tabindex="0">${icon('upload')}<div><b>Excel- oder CSV-Datei hierher ziehen</b> oder klicken zum Auswählen</div><div class="help">.xlsx, .xls, .ods, .csv, .txt – z. B. ein Kursexport aus deinem Banksystem</div><input type="file" accept=".xlsx,.xlsm,.xls,.ods,.csv,.txt,.tsv" hidden></label>
      <p class="help" style="margin:0">Unsicher beim Format? <button type="button" class="btn ghost sm" data-act="sample">${icon('download')} Beispieldatei (.xlsx) herunterladen</button></p>`
    : `<label class="field"><span class="lbl"><b>In Excel Spalten markieren (z. B. Datum und Schlusskurs), kopieren und hier einfügen</b></span><textarea placeholder="Datum&#9;Schlusskurs&#10;02.01.2025&#9;5.868,55&#10;03.01.2025&#9;5.942,47&#10;…"></textarea></label>`}
    <div data-slot="sheet"></div><div data-slot="map"></div>
    <p class="privacy">${icon('shield')}<span>Deine Daten werden nur hier im Browser verarbeitet und nicht hochgeladen. Deutsche Zahlenformate (1.234,56), absteigende Sortierung und Datumsformate wie 31.12.2024 werden automatisch erkannt.</span></p>`;
  const mapSlot = body.querySelector('[data-slot="map"]'), sheetSlot = body.querySelector('[data-slot="sheet"]');
  setPending(() => { throw new Error(mode === 'file' ? 'Bitte zuerst eine Datei auswählen.' : 'Bitte zuerst Daten einfügen.'); });

  const showMapping = () => {
    const opts = (sel, allowNone) => (allowNone ? `<option value="-1">– keine –</option>` : '') + table.header.map((c, i) => `<option value="${i}" ${i === sel ? 'selected' : ''}>${escapeHtml(c)}</option>`).join('');
    mapSlot.innerHTML = `<div class="preview">${tableHtml(table.header, table.body.slice(0, 8).map((r) => table.header.map((_, j) => (r[j] instanceof Date ? r[j].toLocaleDateString('de-DE') : r[j] ?? ''))), table.header.map(() => 'text'))}</div>
      <div class="row-2" style="margin-top:12px">
        <label class="field"><span class="lbl"><b>Datumsspalte</b></span><select data-k="date">${opts(table.dateCol, true)}</select></label>
        <label class="field"><span class="lbl"><b>Wertespalte</b></span><select data-k="value">${opts(table.valueCol, false)}</select></label>
        <label class="field"><span class="lbl"><b>Die Werte sind …</b></span><select data-k="kind">${Object.entries(D.KIND_LABELS).map(([k, l]) => `<option value="${k}" ${k === table.kind ? 'selected' : ''}>${l}</option>`).join('')}</select><span class="help">Automatisch erkannt – bitte prüfen.</span></label>
        <label class="field"><span class="lbl"><b>VIX-Spalte (optional)</b></span><select data-k="vix">${opts(table.vixCol, true)}</select><span class="help">Für Frühwarn-Werkzeuge, z. B. VSTOXX.</span></label>
      </div><p class="help" data-slot="check"></p>`;
    const val = (k) => Number(mapSlot.querySelector(`[data-k="${k}"]`).value);
    const check = () => {
      const up = D.buildUpload(fileName, table, { dateCol: val('date'), valueCol: val('value'), vixCol: val('vix'), kind: mapSlot.querySelector('[data-k="kind"]').value });
      mapSlot.querySelector('[data-slot="check"]').textContent = `${up.n.toLocaleString('de-DE')} gültige Zeilen${up.dropped ? `, ${up.dropped} übersprungen` : ''}${up.reversed ? ' · absteigend sortiert erkannt, wird umgedreht' : ''}.`;
      return up;
    };
    mapSlot.querySelectorAll('select').forEach((s) => s.addEventListener('change', check));
    check();
    setPending(() => {
      const up = check();
      if (up.n < 61) throw new Error(`Zu wenige gültige Zeilen (${up.n}). Mindestens 61 nötig.`);
      return { kind: 'upload', key: D.registerUpload(up) };
    });
  };
  const useSheet = (rows) => {
    table = D.analyzeTable(rows);
    if (!table.cols.some((c) => c.numeric)) { mapSlot.innerHTML = `<p class="form-error">Keine Zahlenspalte gefunden.</p>`; return; }
    showMapping();
  };
  const loadSheets = (res) => {
    fileName = res.name; sheets = res.sheets;
    if (!sheets.length) { mapSlot.innerHTML = `<p class="form-error">Die Datei enthält keine Tabellendaten.</p>`; return; }
    sheetSlot.innerHTML = sheets.length > 1 ? `<label class="field" style="max-width:320px"><span class="lbl"><b>Tabellenblatt</b></span><select>${sheets.map((s, i) => `<option value="${i}">${escapeHtml(s.name)}</option>`).join('')}</select></label>` : `<p class="help">${icon('excel', 'width="14" height="14"')} ${escapeHtml(res.name)}</p>`;
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
  // A realistic template: what a price export from a bank system typically looks like (newest first).
  const XLSX = await D.loadSheetJS();
  const sp = app.market.datasets.find((d) => d.id === 'sp500');
  const n = sp.dates.length, from = Math.max(0, n - 504);
  const rows = [['Datum', 'Schlusskurs', 'VIX']];
  for (let i = n - 1; i >= from; i--) {
    const [y, m, d] = sp.dates[i].split('-').map(Number);
    rows.push([new Date(y, m - 1, d), sp.prices[i], sp.vix?.[i] ?? null]);
  }
  const ws = XLSX.utils.aoa_to_sheet(rows, { cellDates: true });
  for (let r = 1; r < rows.length; r++) {
    const a = ws[XLSX.utils.encode_cell({ r, c: 0 })]; if (a) a.z = 'dd.mm.yyyy';
    const b = ws[XLSX.utils.encode_cell({ r, c: 1 })]; if (b) b.z = '#,##0.00';
    const c = ws[XLSX.utils.encode_cell({ r, c: 2 })]; if (c) c.z = '0.00';
  }
  ws['!cols'] = [{ wch: 12 }, { wch: 14 }, { wch: 8 }];
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Kurse');
  XLSX.writeFile(wb, 'tail-risk-beispiel-sp500.xlsx');
}

// --------------------------------------------------------------------------- tool view
const BOOT_STEPS = [['runtime', 'Python-Laufzeit (WebAssembly)'], ['packages', 'NumPy, SciPy, Matplotlib'], ['framework', 'Tail-Risk-Framework importieren'], ['ready', 'Bereit']];

function bootCard() {
  const card = h(`<section class="card boot-card" aria-live="polite"><h3>Python wird im Browser gestartet</h3>
    <p class="muted small" style="margin:0">Einmalig rund 30 MB. Danach kommt alles aus dem Browser-Cache und startet in wenigen Sekunden.</p>
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
    clock.textContent = s >= 3 ? `Läuft seit ${s} s im Browser …` : '';
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
  if (!spec) { view.innerHTML = `<div class="wrap empty"><h1>Werkzeug nicht gefunden</h1><p><a href="#/werkzeuge">Alle Werkzeuge</a></p></div>`; return; }
  const dsParam = D.decodeChoice(query.get('ds'));
  if (dsParam && D.encodeChoice(dsParam) !== D.encodeChoice(app.choice)) { app.choice = dsParam; saveChoice(dsParam); }
  await resolveDataset();
  const hasQueryParams = spec.params.some((p) => query.has(p.name));
  let values = hasQueryParams ? paramsFromQuery(spec, query) : (app.params.get(id) || paramsFromQuery(spec, query));
  app.params.set(id, values);
  document.title = `${spec.title} · Tail Risk Lab`;

  view.innerHTML = `<div class="wrap">
    <header class="tool-head"><div>
      <nav class="crumbs" aria-label="Brotkrumen"><a href="#/werkzeuge">Werkzeuge</a> <span>›</span> <span>${escapeHtml(groupTitle(spec.group))}</span></nav>
      <h1>${escapeHtml(spec.title)}</h1><p class="summary">${escapeHtml(spec.summary)}</p></div></header>
    <div class="tool-layout"><aside class="panel" aria-label="Einstellungen"></aside><main class="results" id="results" aria-live="polite"></main></div></div>`;
  const panel = view.querySelector('.panel'), results = view.querySelector('#results');
  // A form or timer of a view the user already left (e.g. a 'change' fired by the blur of a removed
  // input, or a pending live update) must neither rewrite the URL nor cancel the new view's run.
  app.view += 1;
  if (app.run && !app.run.done) app.run.detach = true;
  const myView = app.view;
  const current = () => app.view === myView;

  let dsCard = null;
  if (spec.needs.length) { dsCard = datasetCard(spec); panel.appendChild(dsCard); }
  const paramsCard = h(`<section class="card"><h2>Parameter</h2></section>`);
  const runBtn = h(`<button class="btn primary block lg" type="button">${icon('play')} Berechnen</button>`);
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
  if (spec.params.length) paramsCard.appendChild(form); else paramsCard.appendChild(h('<p class="muted small" style="margin:0">Keine Parameter – einfach berechnen.</p>'));
  const runRow = h('<div class="run-row" style="margin-top:16px"></div>');
  runRow.append(runBtn, runHint);
  paramsCard.appendChild(runRow);
  panel.appendChild(paramsCard);
  panel.appendChild(h(`<section class="card"><h2>Worum es geht</h2><div class="desc">${richText(spec.description || spec.summary)}</div>
    <details class="more" style="margin-top:10px"><summary>Verwendeter Python-Code</summary><div class="covers">${spec.covers.map((c) => `<code>${escapeHtml(c)}</code>`).join('')}</div>
    <p class="small" style="margin:8px 0 0"><a href="${REPO}/blob/main/webtools/tools/${spec.id.replace(/-/g, '_')}.py" rel="noopener">Adapter auf GitHub ansehen</a></p></details></section>`));

  runHint.textContent = spec.cost === 'fast' ? 'Aktualisiert sich bei Änderungen automatisch.' : spec.cost === 'slow' ? 'Rechenintensiv – kann bis zu einer Minute dauern.' : '';
  const markDirty = () => { runBtn.innerHTML = `${icon('play')} ${app.results.has(id) ? 'Neu berechnen' : 'Berechnen'}`; };

  const resultKey = () => JSON.stringify([id, spec.needs.length ? app.dataset.key : '-', values]);

  // toggle: the button (reads "Abbrechen" while this tool runs) cancels; otherwise a new run replaces the old.
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
    runBtn.innerHTML = `${icon('stop')} Abbrechen`;
    results.classList.add('stale');
    const firstRun = !results.querySelector('.verdict, .metric, .chart-card, .figure-card, .table-card');
    let waiting = null;
    const showWaiting = (card) => { waiting?.dispose?.(); waiting?.remove(); waiting = card; if (firstRun) { results.innerHTML = ''; } results.prepend(card); };
    try {
      if (engine.state !== 'ready' && engine.state !== 'busy') showWaiting(bootCard());
      await engine.ready();
      if (spec.needs.length) {
        if (engine._dataset?.key !== app.dataset.key) {
          showWaiting(progressCard('Datensatz wird an Python übergeben …'));
          const res = await engine.setDataset(app.dataset.key, app.dataset.descriptor, app.dataset.arrays);
          if (!res.ok) throw Object.assign(new Error(res.error), res);
          app.datasetSummary = res.summary;
        }
        fillDatasetStats(dsCard, app.datasetSummary);
      }
      const prog = progressCard(`${spec.title} wird berechnet …`, { console: spec.group === 'scripts' });
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
      if (err instanceof CancelledError) { toast('Berechnung abgebrochen.'); if (!results.children.length) results.appendChild(h('<div class="empty">Abgebrochen. Klicke auf „Berechnen“, um neu zu starten.</div>')); }
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
    const bar = h(`<div class="result-bar"><div class="meta">${icon('cpu', 'width="13" height="13" style="vertical-align:-2px"')} In ${fmt(res.seconds, 'num', 2)} s im Browser berechnet · ${escapeHtml(res.dataset?.name && spec.needs.length ? res.dataset.name : 'Simulation')}${res.dataset?.n && spec.needs.length ? ` · ${fmt(res.dataset.n, 'int')} Renditen` : ''}</div>
      <div class="actions"><button class="btn sm" data-act="xlsx">${icon('excel')} Excel</button><button class="btn sm" data-act="share">${icon('link')} Link</button><button class="btn sm" data-act="print">${icon('print')} Drucken / PDF</button></div></div>`);
    bar.querySelector('[data-act="xlsx"]').addEventListener('click', async (e) => {
      const btn = e.currentTarget; // currentTarget is null after the first await
      btn.disabled = true;
      try { await exportXlsx({ spec, result: res, dataset: spec.needs.length ? app.dataset : { label: 'Simulation', detail: '' }, params: values }); }
      catch (err) { toast(err.message); } finally { btn.disabled = false; }
    });
    bar.querySelector('[data-act="share"]').addEventListener('click', async () => {
      const url = location.href;
      try { await navigator.clipboard.writeText(url); toast(app.dataset.shareable || !spec.needs.length ? 'Link kopiert – er enthält Datensatz und Parameter.' : 'Link kopiert. Hinweis: Eigene Dateien sind nicht im Link enthalten.'); }
      catch { prompt('Link kopieren:', url); }
    });
    bar.querySelector('[data-act="print"]').addEventListener('click', () => window.print());
    results.innerHTML = '';
    results.appendChild(h(`<div class="print-only"><h2>${escapeHtml(spec.title)}</h2><p>${escapeHtml(app.dataset?.label || '')} · ${escapeHtml(app.dataset?.detail || '')} · ${new Date().toLocaleString('de-DE')}</p></div>`));
    results.appendChild(bar);
    results.append(...body.childNodes);
    if (stdout?.length) {
      const text = stdout.map((m) => m.stream === 'stderr' ? `<span class="err">${escapeHtml(m.text)}</span>` : escapeHtml(m.text)).join('\n');
      results.appendChild(h(`<section class="card table-card"><details ${spec.group === 'scripts' ? 'open' : ''}><summary><b>Python-Ausgabe</b> <span class="muted small">(stdout, ${stdout.length} Zeilen)</span></summary><div class="console">${text}</div></details></section>`));
    }
  };

  const cached = app.results.get(id);
  const pending = app.run && !app.run.done && app.run.toolId === id && app.run.key === resultKey() ? app.run : null;
  if (cached && cached.key === resultKey()) showResult(cached.result, cached.stdout);
  else if (pending) {
    // came back while this exact computation is still running: wait for it instead of restarting
    const card = progressCard(`${spec.title} wird berechnet …`, { console: spec.group === 'scripts' });
    pending.stdout.forEach((m) => card.log(m));
    results.appendChild(card);
    runBtn.innerHTML = `${icon('stop')} Abbrechen`;
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
    document.getElementById('build-info').textContent = `Build ${version?.build || 'dev'} · Marktdaten ${fmtDate(market.updated)}`;
  } catch (err) {
    view.innerHTML = `<div class="wrap empty"><h1>Die Seite konnte nicht geladen werden</h1><p>${escapeHtml(err.message)}</p></div>`;
    return;
  }
  window.addEventListener('hashchange', route);
  route();
}

main();
export { app, engine };
