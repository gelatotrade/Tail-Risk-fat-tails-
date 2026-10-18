// German number formatting for the result codes used by webtools (pct, num, int, x, bp, money, text).
const cache = new Map();
function nf(digits, opts = {}) {
  const key = digits + JSON.stringify(opts);
  if (!cache.has(key)) cache.set(key, new Intl.NumberFormat('de-DE', { minimumFractionDigits: digits, maximumFractionDigits: digits, ...opts }));
  return cache.get(key);
}

export function fmt(value, code = 'num', digits = undefined, unit = '') {
  if (value === null || value === undefined || value === '') return '–';
  if (typeof value === 'string' || code === 'text') return String(value) + (unit ? ` ${unit}` : '');
  if (typeof value === 'boolean') return value ? 'ja' : 'nein';
  if (!Number.isFinite(value)) return '–';
  const u = unit ? ` ${unit}` : '';
  switch (code) {
    case 'pct': {
      const p = value * 100;
      if (p !== 0 && Math.abs(p) < 0.01) return `${sig(p, 2)} %`;
      return `${nf(digits ?? 2).format(p)} %`;
    }
    case 'bp': return `${nf(0).format(value * 10000)} bp`;
    case 'int': return nf(0).format(Math.round(value)) + u;
    case 'x': return `${nf(value >= 100 ? 0 : value >= 10 ? 1 : 2).format(value)}×`;
    case 'money': return nf(digits ?? 2).format(value) + (u || ' €');
    default: {
      const a = Math.abs(value);
      if (a !== 0 && (a < 1e-4 || a >= 1e12)) return value.toExponential(2).replace('.', ',') + u;
      if (digits == null) {
        // automatic: fewer decimals for bigger numbers, significant digits for small ones
        if (a >= 1000) return nf(0).format(value) + u;
        if (a >= 100) return nf(1).format(value) + u;
        if (a >= 1) return nf(2).format(value) + u;
        return sig(value, 3) + u;
      }
      if (a >= 1e5 && digits > 0) return nf(0).format(value) + u;
      return nf(digits).format(value) + u;
    }
  }
}

function sig(v, n) {
  return new Intl.NumberFormat('de-DE', { maximumSignificantDigits: n }).format(v);
}

export const fmtInt = (v) => fmt(v, 'int');
export function fmtDate(iso) {
  if (!iso) return '–';
  const [y, m, d] = String(iso).slice(0, 10).split('-');
  return d && m ? `${d}.${m}.${y}` : String(iso);
}
export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
/** Inline variant of richText (no paragraphs): **bold**, *italic*, `code`. Input is escaped first. */
export function inlineText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:!?)]|$)/g, '$1<i>$2</i>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

/** Minimal text -> HTML: paragraphs on blank lines, **bold**, `code`. Input is escaped first. */
export function richText(text) {
  return String(text || '')
    .split(/\n\s*\n/)
    .map((para) => `<p>${escapeHtml(para.trim())
      .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
      .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s.,;:!?)]|$)/g, '$1<i>$2</i>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>')}</p>`)
    .join('');
}
