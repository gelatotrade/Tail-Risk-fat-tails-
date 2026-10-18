// Datasets: FRED market snapshot, synthetic generators, and the user's own Excel/CSV data.
// A "choice" is small serialisable state; resolve(choice) turns it into what the worker needs.

const SHEETJS_URL = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js';

export const SYNTHETIC_MODELS = [
  { value: 'levy_jump', label: 'Gauß + Lévy-Sprünge', hint: 'Realistisch: ruhige Tage mit seltenen, heftigen Sprüngen.' },
  { value: 'garch', label: 'GARCH(1,1)', hint: 'Volatilitäts-Cluster: unruhige Phasen folgen aufeinander.' },
  { value: 'student_t', label: 'Student-t (ν = 4)', hint: 'Klassische fette Ränder, symmetrisch.' },
  { value: 'levy', label: 'Lévy-stabil (α = 1,7)', hint: 'Sehr fette Ränder, unendliche Varianz.' },
  { value: 'regime_switching', label: 'Regimewechsel', hint: 'Wechsel zwischen ruhigem und Stress-Regime.' },
  { value: 'gaussian', label: 'Normalverteilung', hint: 'Referenz ohne fette Ränder – zum Vergleich.' },
];

export const PRESETS = [
  { id: 'all', label: 'Gesamter Zeitraum' },
  { id: '1y', label: '1 Jahr', years: 1 },
  { id: '3y', label: '3 Jahre', years: 3 },
  { id: '5y', label: '5 Jahre', years: 5 },
  { id: 'volmageddon', label: 'Volmageddon 2018', from: '2017-07-01', to: '2018-06-30' },
  { id: 'covid', label: 'COVID-Crash 2020', from: '2019-07-01', to: '2020-08-31' },
  { id: 'bear2022', label: 'Bärenmarkt 2022', from: '2021-07-01', to: '2023-03-31' },
  { id: 'tariff', label: 'Zoll-Crash 2025', from: '2024-10-01', to: '2025-08-31' },
  { id: 'dotcom', label: 'Dotcom 2000–02', from: '1999-01-01', to: '2003-06-30' },
  { id: 'gfc', label: 'Finanzkrise 2008', from: '2007-01-01', to: '2009-12-31' },
];

export const DEFAULT_CHOICE = { kind: 'market', id: 'sp500', preset: 'all' };

let marketP = null;
export function loadMarket(base) {
  marketP ??= fetch(new URL('data/market.json?v=__BUILD__', base)).then((r) => {
    if (!r.ok) throw new Error(`Marktdaten: HTTP ${r.status}`);
    return r.json();
  });
  return marketP;
}

function addYears(iso, years) {
  const d = new Date(iso + 'T00:00:00Z');
  d.setUTCFullYear(d.getUTCFullYear() + years);
  return d.toISOString().slice(0, 10);
}

/** Date window of a preset for a dataset, or null if the dataset does not cover it. */
export function presetRange(preset, dates) {
  const first = dates[0], last = dates[dates.length - 1];
  const p = PRESETS.find((x) => x.id === preset) || PRESETS[0];
  if (p.id === 'all') return { from: first, to: last };
  if (p.years) return { from: addYears(last, -p.years) < first ? first : addYears(last, -p.years), to: last };
  if (p.to < first || p.from > last) return null;
  return { from: p.from < first ? first : p.from, to: p.to > last ? last : p.to };
}

function lowerBound(arr, v) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } return lo; }

export function sliceMarket(ds, from, to) {
  const i0 = lowerBound(ds.dates, from);
  const i1 = lowerBound(ds.dates, to + '￿');
  const pick = (a) => (a ? a.slice(i0, i1) : undefined);
  return { dates: pick(ds.dates), prices: pick(ds.prices), vix: pick(ds.vix), vix3m: pick(ds.vix3m) };
}

let uploads = new Map(); // key -> {name, arrays, info}
export function registerUpload(upload) {
  const key = `upload:${upload.hash}`;
  uploads.set(key, upload);
  return key;
}

/** Turn a choice into {key, descriptor, arrays, label, detail, shareable}. */
export async function resolve(choice, base) {
  if (choice.kind === 'synthetic') {
    const m = SYNTHETIC_MODELS.find((x) => x.value === choice.model) || SYNTHETIC_MODELS[0];
    const n = Math.max(60, Math.min(20000, Number(choice.n) || 2520)), seed = Number(choice.seed) || 0;
    return { key: `syn:${m.value}:${n}:${seed}`, descriptor: { source: 'synthetic', model: m.value, n, seed },
      arrays: null, label: `Synthetisch · ${m.label}`, detail: `${n.toLocaleString('de-DE')} Tage · Seed ${seed}`, shareable: true };
  }
  if (choice.kind === 'crisis') {
    const seed = Number.isFinite(Number(choice.seed)) && choice.seed !== '' ? Number(choice.seed) : 42;
    return { key: `crisis:${seed}`, descriptor: { source: 'crisis', seed }, arrays: null,
      label: 'Synthetisches Krisenszenario', detail: `1.000 Tage: ruhig → Crash → Erholung · Seed ${seed}`, shareable: true };
  }
  if (choice.kind === 'upload') {
    const up = uploads.get(choice.key);
    if (!up) return resolve(DEFAULT_CHOICE, base);
    return { key: choice.key, descriptor: { source: 'upload', name: up.name, returns_unit: up.returnsUnit || 'log' },
      arrays: up.arrays, label: up.name, detail: up.detail, shareable: false };
  }
  const market = await loadMarket(base);
  const ds = market.datasets.find((d) => d.id === choice.id) || market.datasets[0];
  let range = choice.from && choice.to ? { from: choice.from, to: choice.to } : presetRange(choice.preset || 'all', ds.dates);
  if (!range) range = presetRange('all', ds.dates);
  const arrays = sliceMarket(ds, range.from, range.to);
  const hasVix = !!arrays.vix;
  const presetLabel = PRESETS.find((p) => p.id === choice.preset && !choice.from)?.label;
  return {
    key: `market:${ds.id}:${range.from}:${range.to}`,
    descriptor: { source: 'market', name: ds.name + (presetLabel && choice.preset !== 'all' ? ` · ${presetLabel}` : '') },
    arrays: { dates: arrays.dates, prices: arrays.prices, ...(hasVix ? { vix: arrays.vix, vix3m: arrays.vix3m.map((v) => (v == null ? NaN : v)) } : {}) },
    label: ds.name + (presetLabel && choice.preset !== 'all' ? ` · ${presetLabel}` : ''),
    detail: `${fmtD(range.from)} – ${fmtD(range.to)} · ${(arrays.dates.length - 1).toLocaleString('de-DE')} Renditen${hasVix ? ' · mit VIX' : ''}`,
    range, shareable: true, source: market.source, updated: market.updated,
  };
}

const fmtD = (iso) => { const [y, m, d] = iso.split('-'); return `${d}.${m}.${y}`; };

// --------------------------------------------------------------------------- URL encoding
export function encodeChoice(c) {
  if (c.kind === 'synthetic') return `syn:${c.model}:${c.n}:${c.seed}`;
  if (c.kind === 'crisis') return `crisis:${c.seed}`;
  if (c.kind === 'market') return c.from && c.to ? `mkt:${c.id}:${c.from}:${c.to}` : `mkt:${c.id}:${c.preset || 'all'}`;
  return null;
}
export function decodeChoice(s) {
  if (!s) return null;
  const p = s.split(':');
  if (p[0] === 'syn') return { kind: 'synthetic', model: p[1], n: Number(p[2]), seed: Number(p[3]) };
  if (p[0] === 'crisis') return { kind: 'crisis', seed: Number(p[1]) };
  if (p[0] === 'mkt') return p.length >= 4 ? { kind: 'market', id: p[1], from: p[2], to: p[3] } : { kind: 'market', id: p[1], preset: p[2] || 'all' };
  return null;
}

// --------------------------------------------------------------------------- Excel / CSV import
let sheetP = null;
function loadSheetJS() {
  if (window.XLSX) return Promise.resolve(window.XLSX);
  sheetP ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = SHEETJS_URL; s.async = true;
    s.onload = () => resolve(window.XLSX);
    s.onerror = () => { sheetP = null; reject(new Error('Excel-Bibliothek (SheetJS) konnte nicht geladen werden.')); };
    document.head.appendChild(s);
  });
  return sheetP;
}
export { loadSheetJS };

/** Parse a number written the German or English way: "1.234,56", "1,234.56", "-0,8 %", "12%". */
export function parseNumber(v) {
  if (typeof v === 'number') return Number.isFinite(v) ? v : NaN;
  if (v == null) return NaN;
  let s = String(v).trim().replace(/ |\s/g, '').replace(/[€$]/g, '').replace(/−/g, '-');
  if (!s) return NaN;
  const pct = s.endsWith('%');
  if (pct) s = s.slice(0, -1);
  const lastComma = s.lastIndexOf(','), lastDot = s.lastIndexOf('.');
  if (lastComma > -1 && lastDot > -1) {
    s = lastComma > lastDot ? s.replace(/\./g, '').replace(',', '.') : s.replace(/,/g, '');
  } else if (lastComma > -1) {
    s = (s.match(/,/g).length > 1) ? s.replace(/,/g, '') : s.replace(',', '.');
  } else if ((s.match(/\./g) || []).length > 1) {
    s = s.replace(/\./g, '');
  }
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(s)) return NaN;
  return Number(s);
}

/** Parse a date cell: Date, ISO, dd.mm.yyyy, mm/dd/yyyy, Excel serial. Returns ISO or null. */
export function parseDate(v, allowSerial = false) {
  if (v instanceof Date && !isNaN(v)) {
    const d = new Date(Date.UTC(v.getFullYear(), v.getMonth(), v.getDate()));
    return d.toISOString().slice(0, 10);
  }
  if (typeof v === 'number') {
    if (allowSerial && v > 20000 && v < 80000) return new Date(Math.round((v - 25569) * 86400000)).toISOString().slice(0, 10);
    return null;
  }
  const s = String(v ?? '').trim();
  let m;
  if ((m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/))) return iso(m[1], m[2], m[3]);
  if ((m = s.match(/^(\d{1,2})\.(\d{1,2})\.(\d{2,4})/))) return iso(m[3].length === 2 ? '20' + m[3] : m[3], m[2], m[1]);
  if ((m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})/))) return iso(m[3], m[1], m[2]);
  return null;
}
function iso(y, m, d) {
  const mm = String(m).padStart(2, '0'), dd = String(d).padStart(2, '0');
  if (+mm < 1 || +mm > 12 || +dd < 1 || +dd > 31) return null;
  return `${y}-${mm}-${dd}`;
}

/** Split pasted text (Excel copies as TSV) or CSV into a 2D array. */
export function parseDelimited(text) {
  const lines = String(text).replace(/\r/g, '').split('\n').filter((l) => l.trim() !== '');
  if (!lines.length) return [];
  const sample = lines.slice(0, 20).join('\n');
  const delim = sample.includes('\t') ? '\t' : (sample.match(/;/g) || []).length >= lines.slice(0, 20).length ? ';' : ',';
  return lines.map((l) => splitLine(l, delim));
}
function splitLine(line, delim) {
  const out = []; let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
    else if (c === delim && !q) { out.push(cur); cur = ''; }
    else cur += c;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

export async function readFile(file) {
  const name = file.name || 'Datei';
  if (/\.(xlsx|xlsm|xls|ods)$/i.test(name)) {
    const XLSX = await loadSheetJS();
    const wb = XLSX.read(await file.arrayBuffer(), { type: 'array', cellDates: true });
    const sheets = wb.SheetNames.map((sn) => ({ name: sn, rows: XLSX.utils.sheet_to_json(wb.Sheets[sn], { header: 1, raw: true, defval: null }) }))
      .filter((s) => s.rows.length > 1);
    return { name, sheets };
  }
  const text = await file.text();
  return { name, sheets: [{ name: 'Tabelle', rows: parseDelimited(text) }] };
}

/** Inspect a 2D array: header row, column kinds, sensible default mapping. */
export function analyzeTable(rows) {
  rows = rows.filter((r) => r && r.some((c) => c !== null && c !== ''));
  const width = Math.max(...rows.slice(0, 50).map((r) => r.length));
  const isHeader = rows[0] && rows[0].filter((c) => c !== null && c !== '').every((c) => typeof c === 'string' && Number.isNaN(parseNumber(c)) && !parseDate(c));
  const header = isHeader ? rows[0].map((c, i) => String(c ?? `Spalte ${i + 1}`)) : Array.from({ length: width }, (_, i) => `Spalte ${String.fromCharCode(65 + (i % 26))}`);
  const body = isHeader ? rows.slice(1) : rows;
  const cols = header.map((h, j) => {
    const vals = body.slice(0, 400).map((r) => r[j]).filter((v) => v !== null && v !== '' && v !== undefined);
    const dateHint = /dat|date|tag|zeit|time/i.test(h);
    const nDate = vals.filter((v) => parseDate(v, dateHint)).length;
    const nums = vals.map(parseNumber).filter(Number.isFinite);
    return { index: j, name: h, date: vals.length > 0 && nDate / vals.length > 0.8, numeric: vals.length > 0 && nums.length / vals.length > 0.8, sample: nums.slice(0, 200) };
  });
  const dateCol = cols.find((c) => c.date);
  const numeric = cols.filter((c) => c.numeric && c !== dateCol);
  const byName = (re) => numeric.find((c) => re.test(c.name));
  const valueCol = byName(/schluss|close|kurs|price|preis|adj|wert|nav|index/i) || byName(/rendite|return|perf/i) || numeric[numeric.length - 1];
  const vixCol = byName(/vix|vola/i);
  return { header, body, cols, dateCol: dateCol?.index ?? -1, valueCol: valueCol?.index ?? -1, vixCol: vixCol && vixCol !== valueCol ? vixCol.index : -1,
    kind: valueCol ? guessKind(valueCol.sample, valueCol.name) : 'prices' };
}

export function guessKind(sample, name = '') {
  if (/rendite|return|perf|%/i.test(name)) {
    const big = sample.filter((v) => Math.abs(v) > 1).length;
    return big > sample.length * 0.05 ? 'percent' : 'simple';
  }
  if (!sample.length) return 'prices';
  const allPos = sample.every((v) => v > 0);
  const sorted = [...sample].map(Math.abs).sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  if (allPos && med > 2) return 'prices';
  if (med < 0.2) return 'simple';
  return 'percent';
}

export const KIND_LABELS = { prices: 'Kurse / Indexstände', percent: 'Renditen in %', simple: 'Renditen dezimal (0,01 = 1 %)', log: 'Log-Renditen dezimal' };

/** Build upload arrays from a mapping. Sorts by date ascending, drops unparseable rows. */
export function buildUpload(name, table, { dateCol, valueCol, vixCol, kind }) {
  const recs = [];
  let dropped = 0;
  const dateHint = dateCol >= 0 && /dat|date|tag|zeit|time/i.test(table.header[dateCol] || '');
  for (const r of table.body) {
    const v = parseNumber(r[valueCol]);
    const d = dateCol >= 0 ? parseDate(r[dateCol], dateHint || typeof r[dateCol] === 'number') : null;
    if (!Number.isFinite(v) || (dateCol >= 0 && !d)) { dropped++; continue; }
    recs.push({ d, v, x: vixCol >= 0 ? parseNumber(r[vixCol]) : NaN });
  }
  let reversed = false;
  if (dateCol >= 0 && recs.length > 1 && recs[0].d > recs[recs.length - 1].d) { recs.reverse(); reversed = true; }
  if (dateCol >= 0) recs.sort((a, b) => (a.d < b.d ? -1 : a.d > b.d ? 1 : 0));
  const arrays = {};
  if (dateCol >= 0) arrays.dates = recs.map((r) => r.d);
  if (kind === 'prices') arrays.prices = recs.map((r) => r.v); else arrays.returns = recs.map((r) => r.v);
  if (vixCol >= 0) arrays.vix = recs.map((r) => r.x);
  const hash = hashString(name + JSON.stringify(arrays).slice(0, 20000) + recs.length + kind);
  const detail = `${recs.length.toLocaleString('de-DE')} Zeilen${arrays.dates ? ` · ${fmtD(arrays.dates[0])} – ${fmtD(arrays.dates[arrays.dates.length - 1])}` : ''} · ${KIND_LABELS[kind]}`;
  return { name, arrays, hash, detail, dropped, reversed, n: recs.length, returnsUnit: kind === 'percent' ? 'percent' : kind === 'simple' ? 'simple' : 'log' };
}

function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36);
}

// --------------------------------------------------------------------------- quick JS-side stats (hero, no Python needed)
export function logReturns(prices) {
  const r = [];
  for (let i = 1; i < prices.length; i++) r.push(Math.log(prices[i] / prices[i - 1]));
  return r;
}
export function marketPulse(ds) {
  const p = ds.prices, n = p.length, r = logReturns(p);
  const mean = r.reduce((a, b) => a + b, 0) / r.length;
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (r.length - 1));
  const last20 = r.slice(-20), m20 = last20.reduce((a, b) => a + b, 0) / 20;
  const vol20 = Math.sqrt(last20.reduce((a, b) => a + (b - m20) ** 2, 0) / 19) * Math.sqrt(252);
  let peak = -Infinity; for (const x of p) peak = Math.max(peak, x);
  // Same convention as TailRiskMetrics.tail_ratio in src/: |r| beyond k population standard deviations
  const sdPop = sd * Math.sqrt((r.length - 1) / r.length);
  const k = (s) => r.filter((x) => Math.abs(x) > s * sdPop).length;
  return {
    last: p[n - 1], lastDate: ds.dates[n - 1], change: p[n - 1] / p[n - 2] - 1, drawdown: p[n - 1] / peak - 1,
    vol20, volAll: sd * Math.sqrt(252), vix: ds.vix?.[n - 1], vix3m: ds.vix3m?.[n - 1],
    returns: r, mean, sd, beyond: { 3: k(3), 4: k(4), 5: k(5) }, years: r.length / 252,
  };
}
