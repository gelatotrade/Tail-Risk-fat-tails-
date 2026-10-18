// Landing visual: the real S&P 500 return distribution vs the Gaussian, log scale, pure SVG.
import { fmt } from './format.js';

// Abramowitz-Stegun erf, |error| < 1.5e-7
function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
export const normTail2 = (k) => 1 - erf(k / Math.SQRT2); // P(|Z| > k)

export function heroChart(pulse, { width = 560, height = 300 } = {}) {
  const z = pulse.returns.map((r) => (r - pulse.mean) / pulse.sd);
  const zMin = Math.min(...z), zMax = Math.max(...z);
  const lo = Math.min(-6, Math.floor(zMin) - 0.5), hi = Math.max(6, Math.ceil(zMax) + 0.5), bins = Math.round((hi - lo) * 5), bw = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of z) { const i = Math.floor((v - lo) / bw); if (i >= 0 && i < bins) counts[i]++; }
  const dens = counts.map((c) => c / (z.length * bw));
  const pad = { l: 40, r: 12, t: 12, b: 30 };
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
  const yMin = -5, yMax = 0; // log10 density
  const X = (v) => pad.l + ((v - lo) / (hi - lo)) * W;
  const Y = (d) => pad.t + (1 - (Math.max(yMin, Math.log10(Math.max(d, 1e-9))) - yMin) / (yMax - yMin)) * H;
  const bars = dens.map((d, i) => {
    if (d <= 0) return '';
    const x0 = lo + i * bw, mid = x0 + bw / 2;
    const tail = Math.abs(mid) > 3;
    const y = Y(d);
    return `<rect x="${X(x0) + 0.5}" y="${y}" width="${Math.max(0.5, X(x0 + bw) - X(x0) - 1)}" height="${pad.t + H - y}" rx="1" fill="${tail ? 'var(--critical)' : 'var(--s1)'}" fill-opacity="${tail ? 0.9 : 0.5}"/>`;
  }).join('');
  let gauss = '';
  for (let i = 0; i <= 200; i++) {
    const v = lo + (i / 200) * (hi - lo);
    const d = Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI);
    if (Math.log10(d) < yMin) continue;
    gauss += `${gauss ? 'L' : 'M'}${X(v).toFixed(1)},${Y(d).toFixed(1)}`;
  }
  const grid = [];
  for (let e = yMin; e <= yMax; e++) {
    grid.push(`<line x1="${pad.l}" x2="${pad.l + W}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}" stroke="var(--grid)"/>`);
    grid.push(`<text x="${pad.l - 6}" y="${Y(10 ** e) + 4}" text-anchor="end" font-size="10.5" fill="var(--muted)">10${sup(e)}</text>`);
  }
  for (let s = Math.ceil(lo / 2) * 2; s <= hi; s += 2) {
    grid.push(`<text x="${X(s)}" y="${pad.t + H + 18}" text-anchor="middle" font-size="10.5" fill="var(--muted)">${s > 0 ? '+' : ''}${s}σ</text>`);
  }
  const worst = Math.min(...z);
  const worstLbl = `Schlechtester Tag: ${fmt(Math.min(...pulse.returns), 'pct', 1)} (${fmt(Math.abs(worst), 'num', 1)} σ)`;
  return `<svg class="plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Verteilung der S&P-500-Tagesrenditen auf logarithmischer Skala im Vergleich zur Normalverteilung">
    ${grid.join('')}
    <line x1="${pad.l}" x2="${pad.l + W}" y1="${pad.t + H}" y2="${pad.t + H}" stroke="var(--axis)"/>
    ${bars}
    <path d="${gauss}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-dasharray="5 4" opacity=".75"/>
    <line x1="${X(worst)}" x2="${X(worst)}" y1="${pad.t + 18}" y2="${pad.t + H}" stroke="var(--critical)" stroke-width="1"/>
    <text x="${X(worst) + 6}" y="${pad.t + 28}" font-size="11.5" fill="var(--critical-ink)" font-weight="600">${worstLbl}</text>
  </svg>`;
}
const sup = (e) => String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
