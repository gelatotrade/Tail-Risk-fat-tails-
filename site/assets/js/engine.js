// Main-thread client for the Python worker: promise RPC, boot status, cancellation.
//
//   const engine = new Engine({ base, build });
//   engine.addEventListener('status', e => ...);   // {stage, text, pct}
//   await engine.ready();                           // boots once
//   await engine.setDataset(key, descriptor, arrays);
//   const result = await engine.run(toolId, params, { onProgress, onStdout });
//   engine.cancel();                                // kills a long run, reboots silently

export class CancelledError extends Error {
  constructor() { super('Abgebrochen'); this.name = 'CancelledError'; }
}

export class Engine extends EventTarget {
  constructor({ base, build }) {
    super();
    this.base = base;
    this.build = build;
    this.state = 'idle'; // idle | booting | ready | busy | error
    this.statusInfo = { stage: 'idle', text: 'Python-Engine noch nicht gestartet', pct: 0 };
    this.catalogue = null;
    this.bootSeconds = null;
    this._nextId = 1;
    this._pending = new Map();
    this._ready = null;
    this._dataset = null; // {key, descriptor, arrays} - replayed after a restart
    this._busy = 0;
  }

  _setState(state) {
    this.state = state;
    this.dispatchEvent(new CustomEvent('state', { detail: state }));
  }

  _emitStatus(info) {
    this.statusInfo = info;
    this.dispatchEvent(new CustomEvent('status', { detail: info }));
  }

  _spawn() {
    this.worker = new Worker(new URL(`worker.js?v=${this.build}`, import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this._onMessage(e.data);
    this.worker.onerror = (e) => {
      const msg = e.message || 'Worker-Fehler';
      for (const [, p] of this._pending) p.reject(new Error(msg));
      this._pending.clear();
      this._setState('error');
      this._emitStatus({ stage: 'error', text: `Python-Engine konnte nicht starten: ${msg}`, pct: 0 });
    };
  }

  _onMessage(msg) {
    if (msg.type === 'status') return this._emitStatus(msg);
    const pending = this._pending.get(msg.id);
    if (msg.type === 'progress') return pending?.onProgress?.(msg);
    if (msg.type === 'stdout') return (pending?.onStdout || ((m) => this.dispatchEvent(new CustomEvent('stdout', { detail: m }))))(msg);
    if (!pending) return;
    this._pending.delete(msg.id);
    if (msg.type === 'result') pending.resolve(msg.result);
    else pending.reject(Object.assign(new Error(msg.error), { stack: msg.stack }));
  }

  _call(type, payload, extra = {}) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject, ...extra });
      this.worker.postMessage({ id, type, payload });
    });
  }

  ready() {
    if (this._ready) return this._ready;
    this._setState('booting');
    this._spawn();
    this._ready = this._call('boot', { base: this.base, build: this.build }).then(
      async (info) => {
        this.catalogue = info.catalogue;
        this.bootSeconds = info.seconds;
        if (this._dataset) {
          const { key, descriptor, arrays } = this._dataset;
          await this._call('dataset', { key, descriptor, arrays });
        }
        this._setState('ready');
        return info;
      },
      (err) => {
        this._ready = null;
        this._setState('error');
        this._emitStatus({ stage: 'error', text: `Python-Engine konnte nicht starten: ${err.message}`, pct: 0 });
        throw err;
      },
    );
    return this._ready;
  }

  async setDataset(key, descriptor, arrays) {
    await this.ready();
    const res = await this._call('dataset', { key, descriptor, arrays });
    if (res.ok) this._dataset = { key, descriptor, arrays };
    return res;
  }

  async run(tool, params, { onProgress, onStdout, preload } = {}) {
    await this.ready();
    this._busy += 1;
    this._setState('busy');
    try {
      return await this._call('run', { tool, params, preload }, { onProgress, onStdout });
    } finally {
      this._busy -= 1;
      if (this._busy === 0 && this.state === 'busy') this._setState('ready');
    }
  }

  /** Stop whatever is running.  Pyodide cannot be interrupted without
   *  cross-origin isolation (not available on GitHub Pages), so the worker is
   *  terminated and transparently rebooted; the active dataset is replayed. */
  cancel() {
    if (!this.worker) return;
    this.worker.terminate();
    for (const [, p] of this._pending) p.reject(new CancelledError());
    this._pending.clear();
    this._busy = 0;
    this._ready = null;
    this.ready().catch(() => {});
  }
}
