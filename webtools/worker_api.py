"""
The tiny Python surface the browser's Web Worker talks to.

All calls exchange JSON *strings* so nothing depends on Pyodide's proxy
conversion rules, and the same functions are callable from CPython tests.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import core, data as data_mod

_DATASETS: Dict[str, core.Dataset] = {}
_ACTIVE: Optional[str] = None


def _dump(obj: Any) -> str:
    return json.dumps(core.to_json(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def warmup() -> str:
    """Import the heavy framework once (fonts, scipy.stats) so the first tool run is fast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (builds the font cache)
    import src  # noqa: F401
    import src.physics, src.models, src.analysis, src.visualization  # noqa: F401,E401

    return catalogue()


def catalogue() -> str:
    return _dump(core.catalogue())


def summarize(ds: core.Dataset) -> dict:
    r = ds.returns
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    worst = int(np.argmin(r))
    best = int(np.argmax(r))
    return {
        "name": ds.name, "n": ds.n,
        "start": ds.dates[0] if ds.dates else None, "end": ds.dates[-1] if ds.dates else None,
        "has": {k: getattr(ds, k) is not None for k in core.DATA_KEYS},
        "ann_vol": ann_vol,
        "mean_daily": float(np.mean(r)),
        "worst": {"value": float(r[worst]), "date": ds.dates[worst] if ds.dates else worst},
        "best": {"value": float(r[best]), "date": ds.dates[best] if ds.dates else best},
        "meta": ds.meta,
    }


def set_dataset(key: str, descriptor_json: str, arrays: Any = None) -> str:
    """Build (or reuse) a dataset and make it active.  Returns a JSON summary or an error."""
    global _ACTIVE
    try:
        if key not in _DATASETS:
            descriptor = json.loads(descriptor_json)
            if hasattr(arrays, "to_py"):
                arrays = arrays.to_py()
            _DATASETS[key] = data_mod.build(descriptor, arrays)
            # keep memory bounded: at most 6 datasets cached
            while len(_DATASETS) > 6:
                _DATASETS.pop(next(iter(_DATASETS)))
        _ACTIVE = key
        return _dump({"ok": True, "summary": summarize(_DATASETS[key])})
    except core.ToolError as exc:
        return _dump({"ok": False, "error": str(exc), "kind": "user"})
    except Exception as exc:  # noqa: BLE001
        import traceback

        return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}", "kind": "internal",
                      "traceback": traceback.format_exc(limit=6)})


def run(tool_id: str, params_json: str, progress: Optional[Callable[[str, float], None]] = None) -> str:
    params = json.loads(params_json) if params_json else {}
    dataset = _DATASETS.get(_ACTIVE) if _ACTIVE else None
    return _dump(core.run_tool(tool_id, params, dataset, progress=progress))
