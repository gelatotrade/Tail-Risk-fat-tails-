// Excel export of a tool result (SheetJS).  Percent values keep full precision with a % number format.
import { loadSheetJS } from './data.js';
import { chartRows } from './charts.js';

const XL_FMT = { pct: '0.00%', int: '#,##0', num: '#,##0.0000', x: '0.0"×"', bp: '0', money: '#,##0.00' };
const sheetName = (s, used) => {
  let base = String(s || 'Blatt').replace(/[\\/?*[\]:]/g, '').replace(/\s+/g, ' ').slice(0, 28).trim() || 'Blatt';
  let name = base, i = 2;
  while (used.has(name)) name = `${base.slice(0, 25)} ${i++}`;
  used.add(name);
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
  const used = new Set();
  const overview = [
    ['Werkzeug', spec.title],
    ['Beschreibung', spec.summary],
    ['Datensatz', dataset?.label || result.dataset?.name || '–'],
    ['Zeitraum / Umfang', dataset?.detail || ''],
    ['Berechnet am', new Date().toLocaleString('de-DE')],
    ['Rechenzeit (s)', Number(result.seconds?.toFixed(3))],
    ['Quelle', `Tail Risk Lab – ${location.href}`],
    [],
    ['Parameter', 'Wert'],
    ...spec.params.map((p) => [p.label, params[p.name] ?? p.default]),
  ];
  if (result.verdict) overview.push([], ['Einschätzung', `${result.verdict.title}${result.verdict.text ? ' – ' + result.verdict.text : ''}`]);
  XLSX.utils.book_append_sheet(wb, sheetFrom(XLSX, ['Feld', 'Wert'], overview), sheetName('Übersicht', used));
  if (result.metrics?.length) {
    const ws = sheetFrom(XLSX, ['Kennzahl', 'Wert', 'Hinweis'], result.metrics.map((m) => [m.label, m.value, m.hint || '']));
    XLSX.utils.book_append_sheet(wb, ws, sheetName('Kennzahlen', used));
    result.metrics.forEach((m, i) => { const ref = XLSX.utils.encode_cell({ r: i + 1, c: 1 }); if (ws[ref] && XL_FMT[m.fmt] && typeof m.value === 'number') ws[ref].z = XL_FMT[m.fmt]; });
  }
  for (const t of result.tables || []) XLSX.utils.book_append_sheet(wb, sheetFrom(XLSX, t.columns, t.rows, t.formats), sheetName(t.title, used));
  for (const c of result.charts || []) {
    const { cols, rows, fmts } = chartRows(c);
    if (rows.length) XLSX.utils.book_append_sheet(wb, sheetFrom(XLSX, cols, rows, fmts), sheetName(`Daten ${c.title}`, used));
  }
  const date = new Date().toISOString().slice(0, 10);
  XLSX.writeFile(wb, `tail-risk-${spec.id}-${date}.xlsx`, { compression: true });
}
