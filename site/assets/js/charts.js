// Render webtools chart specs (see webtools/charts.py) with Plotly, themed from CSS tokens.
import { fmt, escapeHtml } from './format.js';

const PLOTLY_URL = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.1.0/plotly.min.js';
let plotlyP = null;
export function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  plotlyP ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = PLOTLY_URL;
    s.async = true;
    s.onload = () => resolve(window.Plotly);
    s.onerror = () => { plotlyP = null; reject(new Error('Diagrammbibliothek (Plotly) konnte nicht geladen werden.')); };
    document.head.appendChild(s);
  });
  return plotlyP;
}

function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    surface: v('--surface'), ink: v('--ink'), ink2: v('--ink-2'), muted: v('--muted'), grid: v('--grid'), axis: v('--axis'),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v(`--s${i}`)),
    status: { good: v('--good'), warning: v('--warning'), serious: v('--serious'), critical: v('--critical') },
    swan: v('--swan'), font: v('--font') || 'system-ui',
  };
}

function color(c, t) {
  if (typeof c === 'number') return t.series[(Math.max(1, c) - 1) % 8];
  if (c === 'muted') return t.muted;
  if (c === 'ink') return t.ink2;
  if (t.status[c]) return t.status[c];
  return c || t.series[0];
}

function withAlpha(hex, a) {
  const h = hex.replace('#', '');
  if (h.length !== 6) return hex;
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function tickformat(ax) {
  if (ax.type === 'date') return undefined;
  if (ax.fmt === 'pct') return `.${Math.max(0, (ax.digits ?? 1))}%`;
  if (ax.fmt === 'int') return ',d';
  return undefined;
}

function hoverFmt(ax) {
  if (ax.type === 'date') return '%d.%m.%Y';
  if (ax.fmt === 'pct') return `.${Math.max(2, (ax.digits ?? 2))}%`;
  if (ax.fmt === 'int') return ',d';
  return '.4~g';
}

function axisLayout(ax, t, extra = {}) {
  const type = ax.type === 'category' ? 'category' : ax.type || 'linear';
  return {
    title: ax.label ? { text: ax.label, font: { size: 12, color: t.ink2 }, standoff: 8 } : undefined,
    type, tickformat: tickformat(ax), hoverformat: hoverFmt(ax),
    range: ax.range && type === 'log' ? ax.range.map((r) => Math.log10(r)) : ax.range,
    exponentformat: type === 'log' ? 'power' : 'B', dtick: type === 'log' ? 1 : undefined,
    gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickcolor: t.axis,
    tickfont: { size: 11, color: t.muted }, automargin: true, showline: true, zeroline: false,
    ...extra,
  };
}

function baseLayout(spec, t) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: t.font, color: t.ink2, size: 12 },
    margin: { l: 8, r: 12, t: 8, b: 8 }, height: spec.height || 340,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: -0.16, yanchor: 'top', font: { size: 12, color: t.ink2 }, bgcolor: 'rgba(0,0,0,0)' },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink, family: t.font, size: 12 } },
    separators: ',.',
  };
}

function xyFigure(spec, t) {
  const data = spec.traces.map((tr) => {
    const c = color(tr.color, t);
    const common = { name: tr.name, x: tr.x, y: tr.y, hovertemplate: undefined };
    if (tr.type === 'bar') {
      return { ...common, type: 'bar', marker: { color: tr.color === 'muted' ? t.muted : withAlpha(c, 0.55), line: { width: 0 } } };
    }
    if (tr.type === 'scatter') {
      const many = (tr.x || []).length > 300;
      return { ...common, type: many ? 'scattergl' : 'scatter', mode: 'markers', text: tr.text || undefined,
        marker: { color: c, size: tr.size || 6, opacity: many ? 0.75 : 1, line: many ? { width: 0 } : { color: t.surface, width: 1.5 } } };
    }
    const line = { color: c, width: tr.width || 2, dash: tr.dash ? 'dash' : 'solid', shape: tr.type === 'step' ? 'hv' : 'linear' };
    const n = (tr.x || []).length;
    return { ...common, type: n > 4000 ? 'scattergl' : 'scatter', mode: 'lines', line, text: tr.text || undefined,
      fill: tr.fill ? 'tozeroy' : undefined, fillcolor: tr.fill ? withAlpha(c, 0.14) : undefined };
  });
  const layout = baseLayout(spec, t);
  layout.xaxis = axisLayout(spec.x, t);
  layout.yaxis = axisLayout(spec.y, t);
  layout.barmode = spec.barmode === 'stack' ? 'stack' : 'overlay';
  layout.bargap = 0.04;
  layout.hovermode = spec.x.type === 'date' ? 'x unified' : 'closest';
  layout.showlegend = data.length > 1;
  const shapes = [], annotations = [];
  for (const b of spec.bands || []) {
    shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: b.x0, x1: b.x1, y0: 0, y1: 1, fillcolor: withAlpha(color(b.color, t), 0.10), line: { width: 0 }, layer: 'below' });
    if (b.label) annotations.push({ x: b.x0, xref: 'x', y: 1, yref: 'paper', text: escapeHtml(b.label), showarrow: false, xanchor: 'left', yanchor: 'top', font: { size: 11, color: t.ink2 }, bgcolor: withAlpha(t.surface.length === 7 ? t.surface : '#ffffff', 0.7) });
  }
  for (const h of spec.hlines || []) {
    shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: h.y, y1: h.y, line: { color: color(h.color, t), width: 1.2, dash: h.dash ? 'dot' : 'solid' } });
    if (h.label) annotations.push({ xref: 'paper', x: 1, y: h.y, text: escapeHtml(h.label), showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11, color: t.ink2 } });
  }
  for (const v of spec.vlines || []) {
    shapes.push({ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: v.x, x1: v.x, line: { color: color(v.color, t), width: 1.2, dash: v.dash ? 'dot' : 'solid' } });
    if (v.label) annotations.push({ x: v.x, yref: 'paper', y: 1, text: escapeHtml(v.label), showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { size: 11, color: t.ink2 } });
  }
  layout.shapes = shapes;
  layout.annotations = annotations;
  return { data, layout };
}

function riskScale(t, scale) {
  if (scale === 'diverging') return [[0, t.series[0]], [0.5, '#9a9993'], [1, t.status.critical]];
  if (scale === 'sequential') return [[0, '#cde2fb'], [0.5, '#3987e5'], [1, '#0d366b']];
  // calm plateau -> stress ramp -> crash cliff -> black swan
  return [[0, t.status.good], [0.3, t.status.warning], [0.55, t.status.serious], [0.8, t.status.critical], [1, t.swan]];
}

function scene(spec, t) {
  const ax = (a) => ({ ...axisLayout(a, t), backgroundcolor: 'rgba(0,0,0,0)', showbackground: false, title: { text: a.label || '', font: { size: 12, color: t.ink2 } } });
  return { xaxis: ax(spec.x), yaxis: ax(spec.y), zaxis: ax(spec.z), camera: { eye: { x: 1.55, y: -1.55, z: 0.95 } }, aspectmode: 'manual', aspectratio: { x: 1.2, y: 1.2, z: 0.75 } };
}

function surfaceFigure(spec, t) {
  const data = [{
    type: 'surface', x: spec.xs, y: spec.ys, z: spec.zs, colorscale: riskScale(t, spec.scale), name: spec.z.label || 'z',
    colorbar: { thickness: 12, len: 0.7, tickformat: tickformat(spec.z), tickfont: { color: t.muted, size: 11 }, outlinewidth: 0 },
    contours: { z: { show: true, usecolormap: true, highlightcolor: t.ink, project: { z: true } } },
    hovertemplate: `${escapeHtml(spec.x.label)}: %{x:${hoverFmt(spec.x)}}<br>${escapeHtml(spec.y.label)}: %{y:${hoverFmt(spec.y)}}<br>${escapeHtml(spec.z.label)}: %{z:${hoverFmt(spec.z)}}<extra></extra>`,
    lighting: { ambient: 0.75, diffuse: 0.6, specular: 0.15, roughness: 0.6 },
  }];
  for (const p of [...(spec.paths || []), ...(spec.points || [])]) {
    const c = color(p.color, t);
    data.push({ type: 'scatter3d', name: p.name, x: p.x, y: p.y, z: p.z, text: p.text || undefined,
      mode: p.mode === 'markers' ? 'markers' : p.mode === 'lines+markers' ? 'lines+markers' : 'lines',
      line: { color: c, width: 6 }, marker: { color: c, size: p.size || 5, line: { color: t.surface, width: 1 } } });
  }
  const layout = baseLayout(spec, t);
  layout.scene = scene(spec, t);
  layout.margin = { l: 0, r: 0, t: 0, b: 0 };
  layout.showlegend = data.length > 1;
  layout.legend = { ...layout.legend, y: 0, yanchor: 'bottom' };
  return { data, layout };
}

function scatter3dFigure(spec, t) {
  const data = spec.traces.map((p) => {
    const c = color(p.color, t);
    return { type: 'scatter3d', name: p.name, x: p.x, y: p.y, z: p.z, text: p.text || undefined,
      mode: p.mode || 'lines', line: { color: c, width: 3 }, marker: { color: c, size: p.size || 3 } };
  });
  const layout = baseLayout(spec, t);
  layout.scene = scene(spec, t);
  layout.margin = { l: 0, r: 0, t: 0, b: 0 };
  layout.showlegend = data.length > 1;
  return { data, layout };
}

function heatmapFigure(spec, t) {
  const data = [{ type: 'heatmap', x: spec.xs, y: spec.ys, z: spec.zs, colorscale: riskScale(t, spec.scale), xgap: 2, ygap: 2,
    colorbar: { thickness: 12, tickfont: { color: t.muted, size: 11 }, tickformat: tickformat(spec.z), outlinewidth: 0 },
    hovertemplate: `%{y} · %{x}<br>${escapeHtml(spec.z.label || 'Wert')}: %{z:${hoverFmt(spec.z)}}<extra></extra>`,
    zmid: spec.scale === 'diverging' ? 0 : undefined }];
  const layout = baseLayout(spec, t);
  layout.xaxis = axisLayout(spec.x, t, { showgrid: false, showline: false });
  layout.yaxis = axisLayout(spec.y, t, { showgrid: false, showline: false, autorange: 'reversed' });
  layout.showlegend = false;
  return { data, layout };
}

function gaugeFigure(spec, t) {
  const [a, b, c] = spec.thresholds;
  const steps = [[0, a, 'good'], [a, b, 'warning'], [b, c, 'serious'], [c, spec.max, 'critical']].map(([x0, x1, s]) => ({ range: [x0, x1], color: withAlpha(t.status[s], 0.35) }));
  const data = [{ type: 'indicator', mode: 'gauge+number', value: spec.value,
    number: { font: { size: 44, color: t.ink }, valueformat: '.0f', suffix: ` / ${spec.max}` },
    gauge: { axis: { range: [0, spec.max], tickcolor: t.axis, tickfont: { color: t.muted } }, bar: { color: t.ink, thickness: 0.22 }, bgcolor: 'rgba(0,0,0,0)', borderwidth: 0, steps } }];
  const layout = baseLayout({ height: 250 }, t);
  layout.margin = { l: 24, r: 24, t: 10, b: 0 };
  return { data, layout };
}

const BUILDERS = { xy: xyFigure, surface: surfaceFigure, scatter3d: scatter3dFigure, heatmap: heatmapFigure, gauge: gaugeFigure };

export async function renderChart(el, spec) {
  const Plotly = await loadPlotly();
  const t = tokens();
  const { data, layout } = (BUILDERS[spec.kind] || xyFigure)(spec, t);
  const config = { responsive: true, displaylogo: false, locale: 'de',
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
    toImageButtonOptions: { format: 'png', filename: (spec.title || 'diagramm').toLowerCase().replace(/[^a-z0-9äöüß]+/gi, '-'), scale: 2 } };
  await Plotly.react(el, data, layout, config);
  el.__spec = spec;
}

export function rerenderAll(root = document) {
  if (!window.Plotly) return;
  root.querySelectorAll('.plot').forEach((el) => { if (el.__spec) renderChart(el, el.__spec); });
}

/** Plain table twin of a chart (accessibility + Excel export). */
export function chartRows(spec) {
  if (spec.kind === 'xy') {
    const cols = ['Reihe', spec.x.label || (spec.x.type === 'date' ? 'Datum' : 'x'), spec.y.label || 'y'];
    const rows = [];
    for (const tr of spec.traces) tr.x.forEach((x, i) => rows.push([tr.name, x, tr.y[i]]));
    return { cols, rows, fmts: ['text', spec.x.type === 'date' ? 'text' : spec.x.fmt, spec.y.fmt] };
  }
  if (spec.kind === 'surface' || spec.kind === 'heatmap') {
    const cols = [spec.y.label || 'y', ...spec.xs.map((x) => (typeof x === 'number' ? fmt(x, spec.x.fmt, 3) : String(x)))];
    return { cols, rows: spec.ys.map((y, i) => [y, ...spec.zs[i]]), fmts: [spec.y.fmt || 'num', ...spec.xs.map(() => spec.z.fmt || 'num')] };
  }
  if (spec.kind === 'scatter3d') {
    const rows = [];
    for (const tr of spec.traces) tr.x.forEach((x, i) => rows.push([tr.name, x, tr.y[i], tr.z[i]]));
    return { cols: ['Reihe', spec.x.label || 'x', spec.y.label || 'y', spec.z.label || 'z'], rows, fmts: ['text', 'num', 'num', 'num'] };
  }
  if (spec.kind === 'gauge') return { cols: ['Wert', 'Maximum'], rows: [[spec.value, spec.max]], fmts: ['num', 'num'] };
  return { cols: [], rows: [], fmts: [] };
}
