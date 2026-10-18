// English (en-GB) number formatting for the result codes used by webtools (pct, num, int, x, bp, money, text).
const cache = new Map();
function nf(digits, opts = {}) {
  const key = digits + JSON.stringify(opts);
  if (!cache.has(key)) cache.set(key, new Intl.NumberFormat('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits, ...opts }));
  return cache.get(key);
}

export function fmt(value, code = 'num', digits = undefined, unit = '') {
  if (value === null || value === undefined || value === '') return '–';
  if (typeof value === 'string' || code === 'text') return withUnit(String(value), unit);
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (!Number.isFinite(value)) return '–';
  const u = (text) => withUnit(minus(text), unit, value);
  switch (code) {
    case 'pct': {
      const p = value * 100;
      if (p !== 0 && Math.abs(p) < 0.01) return `${minus(sig(p, 2))}%`;
      return `${minus(nf(digits ?? 2).format(p))}%`;
    }
    case 'bp': return `${minus(nf(0).format(value * 10000))} bp`;
    case 'int': return u(nf(0).format(Math.round(value)));
    case 'x': return `${minus(nf(value >= 100 ? 0 : value >= 10 ? 1 : 2).format(value))}×`;
    case 'money': return unit ? u(nf(digits ?? 2).format(value)) : `${minus(nf(digits ?? 2).format(value))} EUR`;
    default: {
      const a = Math.abs(value);
      if (a !== 0 && (a < 1e-4 || a >= 1e12)) return u(value.toExponential(2));
      if (digits == null) {
        // automatic: fewer decimals for bigger numbers, significant digits for small ones
        if (a >= 1000) return u(nf(0).format(value));
        if (a >= 100) return u(nf(1).format(value));
        if (a >= 1) return u(nf(2).format(value));
        return u(sig(value, 3));
      }
      if (a >= 1e5 && digits > 0) return u(nf(0).format(value));
      return u(nf(digits).format(value));
    }
  }
}

/**
 * A number with its unit, set the English way: '%' and 'σ' attach to the number (2.5%, 3σ), '$bn'
 * wraps it ($4.2bn, −$2.2bn) and word units take a space and the singular for exactly one
 * (1 trading day, 20 trading days).
 */
export function withUnit(text, unit, value) {
  if (!unit) return text;
  if (unit === '%' || unit === 'σ') return text + unit;
  if (unit === '$bn') return /^[-−]/.test(text) ? `−$${text.slice(1)}bn` : `$${text}bn`;
  if (Number(value) === 1 && /^[a-z][a-z ]*s$/i.test(unit)) return `${text} ${unit.slice(0, -1)}`;
  return `${text} ${unit}`;
}

// Negative numbers get the typographic minus (−12.8%), as in the texts the Python tools write.
const minus = (s) => s.replace(/-/g, '−');

function sig(v, n) {
  return new Intl.NumberFormat('en-GB', { maximumSignificantDigits: n }).format(v);
}

export const fmtInt = (v) => fmt(v, 'int');
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** ISO date -> '15 Oct 2026' (unambiguous in international business use). */
export function fmtDate(iso) {
  if (!iso) return '–';
  const [y, m, d] = String(iso).slice(0, 10).split('-');
  return d && m ? `${Number(d)} ${MONTHS[Number(m) - 1]} ${y}` : String(iso);
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
