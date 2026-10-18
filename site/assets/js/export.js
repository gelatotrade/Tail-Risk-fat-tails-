// Excel export of a tool result (SheetJS).  Percent values keep full precision with a % number format.
import { loadSheetJS, parseDate } from './data.js';
import { chartRows } from './charts.js';
import { fmtDate } from './format.js';

const XL_FMT = { pct: '0.00%', int: '#,##0', num: '#,##0.0000', x: '0.0"×"', bp: '0', money: '#,##0.00' };
// Excel allows 31 characters per sheet name: longer titles are cut at a word boundary ('…'),
// the full title stays in the overview's list of contents.
const fit = (s, max) => {
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1), space = cut.lastIndexOf(' ');
  return `${(space > max / 2 ? cut.slice(0, space) : cut).replace(/[\s,;·–-]+$/, '')}…`;
};
const sheetName = (s, used) => {
  const base = String(s || 'Sheet').replace(/[\\/?*[\]:]/g, '').replace(/\s+/g, ' ').trim() || 'Sheet';
  let name = fit(base, 31), i = 2;
  while (used.has(name.toLowerCase())) { const n = ` (${i++})`; name = fit(base, 31 - n.length) + n; }
  used.add(name.toLowerCase());
  return name;
};

function sheetFrom(XLSX, header, rows, formats) {
  const aoa = [header, ...rows.map((r) => r.map((v) => (v === null || v === undefined ? '' : v)))];
  const ws = XLSX.utils.aoa_to_sheet(aoa);
  if (formats) {
    rows.forEach((r, i) => r.forEach((v, j) => {
      const [code, d] = String(formats[j] ?? '').split(':');
      const z = code === 'pct' && d !== undefined ? (Number(d) > 0 ? `0.${'0'.repeat(Number(d))}%` : '0%') : XL_FMT[code];
      if (z && typeof v === 'number') { const ref = XLSX.utils.encode_cell({ r: i + 1, c: j }); if (ws[ref]) ws[ref].z = z; }
    }));
  }
  ws['!cols'] = header.map((h, j) => ({ wch: Math.min(48, Math.max(10, String(h).length + 2, ...rows.slice(0, 50).map((r) => String(r[j] ?? '').length + 1))) }));
  return ws;
}

export async function exportXlsx({ spec, result, dataset, params }) {
  const XLSX = await loadSheetJS();
  const wb = XLSX.utils.book_new();
  const used = new Set(['overview']);
  const sheets = [], contents = [];
  const add = (title, kind, ws) => { const name = sheetName(title, used); sheets.push([name, ws]); contents.push([name, `${kind}: ${title}`]); return ws; };
  if (result.metrics?.length) {
    const ws = add('Key metrics', 'Table', sheetFrom(XLSX, ['Metric', 'Value', 'Note'], result.metrics.map((m) => [m.label, m.value, m.hint || ''])));
    result.metrics.forEach((m, i) => { const ref = XLSX.utils.encode_cell({ r: i + 1, c: 1 }); if (ws[ref] && XL_FMT[m.fmt] && typeof m.value === 'number') ws[ref].z = XL_FMT[m.fmt]; });
  }
  for (const t of result.tables || []) add(t.title, 'Table', sheetFrom(XLSX, t.columns, t.rows, t.formats));
  for (const c of result.charts || []) {
    const { cols, rows, fmts } = chartRows(c);
    if (rows.length) add(c.title, 'Chart data', sheetFrom(XLSX, cols, rows, fmts));
  }
  const now = new Date();
  const overview = [
    ['Tool', spec.title],
    ['Description', spec.summary],
    ['Dataset', dataset?.label || result.dataset?.name || '–'],
    ['Period / size', dataset?.detail || ''],
    ['Calculated on', `${fmtDate(parseDate(now))}, ${now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`],
    ['Computation time (s)', Number(result.seconds?.toFixed(3))],
    ['Source', `Tail Risk Lab – ${location.href}`],
    [],
    ['Parameter', 'Value'],
    ...spec.params.map((p) => [p.label, params[p.name] ?? p.default]),
  ];
  if (result.verdict) overview.push([], ['Assessment', `${result.verdict.title}${result.verdict.text ? '. ' + result.verdict.text : ''}`]);
  if (contents.length) overview.push([], ['Sheet', 'Contents'], ...contents);
  XLSX.utils.book_append_sheet(wb, sheetFrom(XLSX, ['Field', 'Value'], overview), 'Overview');
  for (const [name, ws] of sheets) XLSX.utils.book_append_sheet(wb, ws, name);
  const date = new Date().toISOString().slice(0, 10);
  XLSX.writeFile(wb, `tail-risk-${spec.id}-${date}.xlsx`, { compression: true });
}
