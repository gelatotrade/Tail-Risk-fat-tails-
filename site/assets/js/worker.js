// Python engine: Pyodide in a dedicated module worker.
// The page talks to it through engine.js; every request carries an id and gets
// exactly one {type:'result'|'error'} reply, plus any number of progress/stdout events.

const PYODIDE_VERSION = '314.0.7';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const CORE_PACKAGES = ['numpy', 'scipy', 'matplotlib'];

let py = null;
let api = null;
let currentId = null;

const post = (msg) => self.postMessage(msg);
const status = (stage, text, pct) => post({ type: 'status', stage, text, pct });

async function boot({ base, build }) {
  const t0 = performance.now();
  status('runtime', 'Python-Laufzeit (WebAssembly) wird geladen …', 4);
  const bundleP = fetch(new URL(`py/bundle.zip?v=${build}`, base)).then((r) => {
    if (!r.ok) throw new Error(`bundle.zip: HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  py.setStdout({ batched: (text) => post({ type: 'stdout', id: currentId, text }) });
  py.setStderr({ batched: (text) => post({ type: 'stdout', id: currentId, text, stream: 'stderr' }) });

  status('packages', 'Wissenschaftliche Pakete: NumPy, SciPy, Matplotlib …', 22);
  let loaded = 0;
  await py.loadPackage(CORE_PACKAGES, {
    messageCallback: (m) => {
      if (m.startsWith('Loaded')) return;
      loaded += 1;
      status('packages', m.replace('Loading', 'Lade'), Math.min(60, 22 + loaded * 3));
    },
    errorCallback: (m) => post({ type: 'stdout', id: null, text: m, stream: 'stderr' }),
  });

  status('framework', 'Tail-Risk-Framework wird entpackt …', 64);
  py.unpackArchive(await bundleP, 'zip', { extractDir: '/home/pyodide/app' });
  py.runPython(`
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
os.chdir('/home/pyodide/app')
sys.path.insert(0, '/home/pyodide/app')
import matplotlib
matplotlib.use('Agg')
`);
  status('framework', 'Module werden importiert (Lévy, Fokker-Planck, Tsallis, EVT …) …', 72);
  api = py.pyimport('webtools.worker_api');
  const catalogue = JSON.parse(api.warmup());
  status('ready', 'Bereit', 100);
  return { catalogue, seconds: (performance.now() - t0) / 1000, pyodide: PYODIDE_VERSION };
}

function progressFor(id) {
  return (message, fraction) => post({ type: 'progress', id, text: String(message), pct: fraction < 0 ? null : fraction * 100 });
}

async function ensurePackagesFor(code) {
  // Scripts may import pandas etc.; Pyodide resolves those from its package index.
  await py.loadPackagesFromImports(code, {
    messageCallback: (m) => post({ type: 'progress', id: currentId, text: m.replace('Loading', 'Lade'), pct: null }),
  });
}

const handlers = {
  boot,
  async dataset({ key, descriptor, arrays }) {
    const pyArrays = arrays ? py.toPy(arrays) : null;
    try {
      return JSON.parse(api.set_dataset(key, JSON.stringify(descriptor), pyArrays));
    } finally {
      pyArrays?.destroy?.();
    }
  },
  async run({ tool, params, preload }, id) {
    if (preload) await ensurePackagesFor(preload);
    const cb = progressFor(id);
    return JSON.parse(api.run(tool, JSON.stringify(params || {}), cb));
  },
};

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  currentId = id;
  try {
    if (!handlers[type]) throw new Error(`unknown request ${type}`);
    if (type !== 'boot' && !api) throw new Error('Python engine not ready');
    const result = await handlers[type](payload || {}, id);
    post({ type: 'result', id, result });
  } catch (err) {
    post({ type: 'error', id, error: String(err && err.message ? err.message : err), stack: err?.stack });
  } finally {
    currentId = null;
  }
};
