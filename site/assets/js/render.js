// Turn a tool result (webtools Result.to_dict) into DOM.
import { fmt, escapeHtml, richText } from './format.js';
import { icon, STATUS_ICON, STATUS_LABEL } from './icons.js';
import { renderChart, chartRows } from './charts.js';

const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };

export function statusBadge(status, label) {
  if (!status) return '';
  return `<span class="status ${status}">${icon(STATUS_ICON[status])}${escapeHtml(label || STATUS_LABEL[status])}</span>`;
}

function tableHtml(columns, rows, formats, { limit = 2000 } = {}) {
  const f = formats || columns.map(() => null);
  const head = columns.map((c, i) => `<th scope="col"${i > 0 && f[i] === 'text' ? ' class="t"' : ''}>${escapeHtml(c)}</th>`).join('');
  const body = rows.slice(0, limit).map((r) => `<tr>${r.map((v, i) => {
    const code = f[i] || (typeof v === 'number' ? 'num' : 'text');
    const text = code === 'text' || typeof v === 'string' ? escapeHtml(v ?? '–') : fmt(v, code, code === 'num' ? undefined : 2);
    return `<td class="${code === 'text' ? 't' : ''}">${text}</td>`;
  }).join('')}</tr>`).join('');
  const more = rows.length > limit ? `<p class="table-note">${rows.length - limit} weitere Zeilen im Excel-Export.</p>` : '';
  return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${more}`;
}

function chartCard(spec) {
  const card = h(`<section class="card chart-card">
    <div class="head"><div><h3>${escapeHtml(spec.title)}</h3>${spec.subtitle ? `<p class="sub">${escapeHtml(spec.subtitle)}</p>` : ''}</div>
      <div class="tools">
        <button class="btn ghost sm" data-act="table" title="Als Tabelle anzeigen" aria-pressed="false">${icon('table')}<span class="sr-only">Tabelle</span></button>
        <button class="btn ghost sm" data-act="png" title="Als PNG speichern">${icon('download')}<span class="sr-only">PNG</span></button>
      </div></div>
    <div class="plot" role="img" aria-label="${escapeHtml(spec.title)}" style="height:${spec.kind === 'gauge' ? 250 : spec.height || 340}px"></div>
    <div class="table-wrap" hidden></div></section>`);
  const plot = card.querySelector('.plot');
  card.querySelector('[data-act="table"]').addEventListener('click', (e) => {
    const btn = e.currentTarget, wrap = card.querySelector('.table-wrap');
    const on = btn.getAttribute('aria-pressed') !== 'true';
    btn.setAttribute('aria-pressed', String(on));
    if (on && !wrap.dataset.filled) {
      const { cols, rows, fmts } = chartRows(spec);
      wrap.innerHTML = tableHtml(cols, rows, fmts, { limit: 500 });
      wrap.dataset.filled = '1';
    }
    wrap.hidden = !on;
  });
  card.querySelector('[data-act="png"]').addEventListener('click', () => window.Plotly?.downloadImage(plot, { format: 'png', scale: 2, filename: slug(spec.title) }));
  queueMicrotask(() => renderChart(plot, spec).catch((err) => { plot.innerHTML = `<p class="empty">${escapeHtml(err.message)}</p>`; }));
  return card;
}

const slug = (s) => String(s || 'export').toLowerCase().normalize('NFKD').replace(/[^\w]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);

export function openLightbox(src, caption) {
  const lb = h(`<div class="lightbox" role="dialog" aria-label="Grafik vergrößert"><div><img src="${src}" alt="${escapeHtml(caption)}"><div class="cap">${escapeHtml(caption)}</div></div></div>`);
  const close = () => { lb.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  lb.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(lb);
}

function figureCard(fig, i) {
  const src = `data:${fig.mime || 'image/png'};base64,${fig.png}`;
  const caption = fig.caption || `Abbildung ${i + 1}`;
  const card = h(`<figure class="card figure-card" style="margin:0">
    <div class="head" style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start">
      <div><h3>${escapeHtml(caption)}</h3><span class="badge">${icon('image', 'width="13" height="13"')} Original-Grafik des Frameworks (Matplotlib, im Browser gerendert)</span></div>
      <a class="btn ghost sm" download="${slug(caption)}.png" href="${src}" title="PNG speichern">${icon('download')}<span class="sr-only">PNG</span></a></div>
    <img src="${src}" alt="${escapeHtml(caption)}" loading="lazy"></figure>`);
  card.querySelector('img').addEventListener('click', () => openLightbox(src, caption));
  return card;
}

export function renderResult(root, result) {
  root.innerHTML = '';
  if (result.verdict) {
    const v = result.verdict;
    root.appendChild(h(`<section class="verdict ${v.level}" role="status">${icon(STATUS_ICON[v.level])}
      <div><div class="lvl">${escapeHtml(STATUS_LABEL[v.level])}</div><h3>${escapeHtml(v.title)}</h3>${v.text ? `<p>${escapeHtml(v.text)}</p>` : ''}</div></section>`));
  }
  if (result.metrics?.length) {
    const grid = h('<section class="metrics" aria-label="Kennzahlen"></section>');
    for (const m of result.metrics) {
      grid.appendChild(h(`<div class="metric"><div class="l">${escapeHtml(m.label)}</div>
        <div class="v">${escapeHtml(fmt(m.value, m.fmt, m.digits, m.unit))}</div>
        ${m.status ? statusBadge(m.status) : ''}${m.hint ? `<div class="h">${escapeHtml(m.hint)}</div>` : ''}</div>`));
    }
    root.appendChild(grid);
  }
  if (result.notes?.length) {
    const notes = h('<section class="notes"></section>');
    for (const n of result.notes) notes.appendChild(h(`<div class="note ${n.level}">${icon(n.level === 'warning' ? 'warn' : n.level === 'success' ? 'good' : 'info')}<div>${richText(n.text)}</div></div>`));
    root.appendChild(notes);
  }
  // charts: wide ones (3D, tall) full width; regular xy charts pair up
  const charts = result.charts || [];
  let pair = null;
  for (const spec of charts) {
    const wide = spec.kind !== 'xy' && spec.kind !== 'gauge' && spec.kind !== 'heatmap' || (spec.height || 340) > 380 || charts.length === 1;
    if (wide) { pair = null; root.appendChild(chartCard(spec)); continue; }
    if (!pair) { pair = h('<div class="grid-2"></div>'); root.appendChild(pair); }
    pair.appendChild(chartCard(spec));
    if (pair.children.length === 2) pair = null;
  }
  if (pair && pair.children.length === 1) pair.classList.remove('grid-2');
  for (const t of result.tables || []) {
    root.appendChild(h(`<section class="card table-card"><h3>${escapeHtml(t.title)}</h3>${tableHtml(t.columns, t.rows, t.formats)}${t.note ? `<p class="table-note">${escapeHtml(t.note)}</p>` : ''}</section>`));
  }
  (result.figures || []).forEach((f, i) => root.appendChild(figureCard(f, i)));
  for (const d of result.downloads || []) {
    const blob = new Blob([d.content], { type: d.mime });
    const a = h(`<a class="btn sm" download="${escapeHtml(d.filename)}">${icon('download')} ${escapeHtml(d.filename)}</a>`);
    a.href = URL.createObjectURL(blob);
    const wrap = h('<div></div>'); wrap.appendChild(a); root.appendChild(wrap);
  }
}

export function renderError(root, err, { onRetry } = {}) {
  const user = err.kind === 'user';
  root.innerHTML = '';
  const card = h(`<section class="card error-card" role="alert"><h3>${user ? 'Das hat so nicht geklappt' : 'Interner Fehler im Python-Code'}</h3>
    <p style="margin-top:6px">${escapeHtml(err.error || err.message || String(err))}</p>
    ${err.traceback ? `<details class="more"><summary>Technische Details</summary><pre>${escapeHtml(err.traceback)}</pre></details>` : ''}
    ${onRetry ? '<button class="btn sm" data-act="retry">Erneut versuchen</button>' : ''}</section>`);
  card.querySelector('[data-act="retry"]')?.addEventListener('click', onRetry);
  root.appendChild(card);
}

export { tableHtml, h };
