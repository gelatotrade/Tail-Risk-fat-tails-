// Parameter forms generated from a tool SPEC.
import { escapeHtml } from './format.js';

const dec = (step) => { const s = String(step ?? 1); return s.includes('.') ? s.split('.')[1].length : 0; };
const show = (v, step) => Number(v).toLocaleString('de-DE', { minimumFractionDigits: dec(step), maximumFractionDigits: dec(step) });

export function buildForm(spec, values, onChange) {
  const form = document.createElement('form');
  form.className = 'form';
  form.noValidate = true;
  form.addEventListener('submit', (e) => e.preventDefault());
  const state = { ...values };
  const emit = (name, v, commit) => { state[name] = v; onChange?.(state, { name, commit }); };

  for (const p of spec.params) {
    const id = `p-${spec.id}-${p.name}`;
    const v = state[p.name] ?? p.default;
    let el;
    if (p.type === 'bool') {
      el = document.createElement('div');
      el.innerHTML = `<label class="switch"><input type="checkbox" id="${id}" ${v ? 'checked' : ''}> <span>${escapeHtml(p.label)}</span></label>${p.help ? `<div class="help">${escapeHtml(p.help)}</div>` : ''}`;
      el.querySelector('input').addEventListener('change', (e) => emit(p.name, e.target.checked, true));
    } else if (p.type === 'select') {
      el = document.createElement('label');
      el.className = 'field';
      el.innerHTML = `<span class="lbl"><b>${escapeHtml(p.label)}</b></span><select id="${id}">${p.options.map((o) => `<option value="${escapeHtml(String(o.value))}" ${String(o.value) === String(v) ? 'selected' : ''}>${escapeHtml(o.label)}</option>`).join('')}</select>${p.help ? `<span class="help">${escapeHtml(p.help)}</span>` : ''}`;
      el.querySelector('select').addEventListener('change', (e) => {
        const opt = p.options.find((o) => String(o.value) === e.target.value);
        emit(p.name, opt ? opt.value : e.target.value, true);
      });
    } else if (p.type === 'int' || p.type === 'float') {
      const step = p.step ?? (p.type === 'int' ? 1 : (p.max - p.min) / 100);
      el = document.createElement('label');
      el.className = 'field';
      el.innerHTML = `<span class="lbl"><b>${escapeHtml(p.label)}</b><output for="${id}">${show(v, step)}${p.unit ? ' ' + escapeHtml(p.unit) : ''}</output></span>
        <div class="range-row"><input type="range" id="${id}" min="${p.min}" max="${p.max}" step="${step}" value="${v}">
        <input type="number" min="${p.min}" max="${p.max}" step="${step}" value="${v}" aria-label="${escapeHtml(p.label)} (Zahl)"></div>
        ${p.help ? `<span class="help">${escapeHtml(p.help)}</span>` : ''}`;
      const range = el.querySelector('input[type=range]'), num = el.querySelector('input[type=number]'), out = el.querySelector('output');
      const set = (raw, commit) => {
        let x = Number(raw);
        if (!Number.isFinite(x)) return;
        x = Math.min(p.max, Math.max(p.min, x));
        if (p.type === 'int') x = Math.round(x);
        range.value = x; num.value = x; out.textContent = show(x, step) + (p.unit ? ' ' + p.unit : '');
        emit(p.name, x, commit);
      };
      range.addEventListener('input', () => set(range.value, false));
      range.addEventListener('change', () => set(range.value, true));
      num.addEventListener('change', () => set(num.value, true));
    } else {
      el = document.createElement('label');
      el.className = 'field';
      el.innerHTML = `<span class="lbl"><b>${escapeHtml(p.label)}</b></span><input type="text" id="${id}" value="${escapeHtml(String(v))}" maxlength="${p.maxlength || 200}">${p.help ? `<span class="help">${escapeHtml(p.help)}</span>` : ''}`;
      el.querySelector('input').addEventListener('change', (e) => emit(p.name, e.target.value, true));
    }
    form.appendChild(el);
  }
  form.values = () => ({ ...state });
  return form;
}

export function defaults(spec) {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
}

/** Read params from URLSearchParams, typed by the spec. */
export function paramsFromQuery(spec, q) {
  const out = defaults(spec);
  for (const p of spec.params) {
    if (!q.has(p.name)) continue;
    const raw = q.get(p.name);
    if (p.type === 'int' || p.type === 'float') { const x = Number(raw); if (Number.isFinite(x)) out[p.name] = Math.min(p.max, Math.max(p.min, x)); }
    else if (p.type === 'bool') out[p.name] = raw === '1' || raw === 'true';
    else if (p.type === 'select') { const o = p.options.find((o) => String(o.value) === raw); if (o) out[p.name] = o.value; }
    else out[p.name] = raw;
  }
  return out;
}

export function queryFromParams(spec, values) {
  const q = new URLSearchParams();
  for (const p of spec.params) {
    const v = values[p.name];
    if (v === undefined || String(v) === String(p.default)) continue;
    q.set(p.name, p.type === 'bool' ? (v ? '1' : '0') : String(v));
  }
  return q;
}
