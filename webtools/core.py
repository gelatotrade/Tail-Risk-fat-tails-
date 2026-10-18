"""
Tool contract, parameter coercion and JSON serialisation.

A tool module provides::

    SPEC = {
        "id": "risk-metrics",              # kebab-case, unique, = module name with '-' for '_'
        "title": "VaR & Expected Shortfall",
        "group": "risk",                   # see GROUPS
        "summary": "One sentence: what the user learns.",
        "description": "Longer text (paragraphs separated by a blank line).",
        "needs": ["returns"],              # subset of DATA_KEYS; [] for pure simulators
        "params": [ {...}, ... ],          # see PARAM_TYPES
        "covers": ["src.models.risk_metrics.TailRiskMetrics", ...],
        "cost": "fast" | "medium" | "slow",
    }

    def run(params: dict, data: Dataset, out: Result) -> None: ...

``run`` fills ``out`` through its builder methods (``metric``, ``table``,
``chart``, ``figure``, ``note``, ``download``).  Everything handed to the
builder is converted to plain JSON here, so tools never think about
serialisation, NaN or numpy scalars.
"""

from __future__ import annotations

import base64
import importlib
import io
import math
import pkgutil
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

GROUPS = {
    "data": "Data and distributions",
    "risk": "Risk metrics",
    "physics": "Physics models",
    "regime": "Regimes and phase transitions",
    "warning": "Early-warning system",
    "surface": "3D risk surfaces",
    "dashboard": "Dashboards and reports",
    "scripts": "Original scripts",
}

DATA_KEYS = ("returns", "prices", "vix", "vix3m", "dates")
PARAM_TYPES = ("int", "float", "bool", "select", "text")
COSTS = ("fast", "medium", "slow")


class ToolError(Exception):
    """An error whose message is meant for the end user (plain English, actionable)."""


# --------------------------------------------------------------------------- #
# Data handed to tools
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """The active dataset.  Arrays are aligned: ``returns[i]`` belongs to ``dates[i]``."""

    name: str = ""
    returns: Optional[np.ndarray] = None
    prices: Optional[np.ndarray] = None
    vix: Optional[np.ndarray] = None
    vix3m: Optional[np.ndarray] = None
    dates: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            labels = {"returns": "returns", "prices": "prices", "vix": "VIX", "vix3m": "VIX3M", "dates": "dates"}
            raise ToolError(
                "This tool needs data the active dataset does not contain: "
                + ", ".join(labels.get(k, k) for k in missing)
                + ". Choose a dataset that includes these series (e.g. S&P 500 with VIX)."
            )

    @property
    def n(self) -> int:
        return 0 if self.returns is None else int(len(self.returns))


# --------------------------------------------------------------------------- #
# JSON conversion
# --------------------------------------------------------------------------- #
def to_json(obj: Any) -> Any:
    """Recursively convert numpy / python objects to strict JSON (NaN/Inf -> None)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fc":
            arr = np.asarray(obj, dtype=float)
            return [to_json(v) for v in arr.tolist()] if arr.ndim > 1 else [
                (v if math.isfinite(v) else None) for v in arr.tolist()
            ]
        return to_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()[:10] if hasattr(obj, "year") else obj.isoformat()
    if hasattr(obj, "item"):
        return to_json(obj.item())
    return str(obj)


def figure_to_png(fig, dpi: int = 110) -> str:
    """Render a matplotlib figure to a base64 PNG and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Result builder
# --------------------------------------------------------------------------- #
FORMATS = ("pct", "num", "int", "text", "bp", "x", "money")
STATUSES = ("good", "warning", "serious", "critical", None)


class Result:
    """Collects everything a tool wants to show.  Methods return ``self`` for chaining."""

    def __init__(self, progress: Optional[Callable[[str, float], None]] = None):
        self.metrics: List[dict] = []
        self.tables: List[dict] = []
        self.charts: List[dict] = []
        self.figures: List[dict] = []
        self.notes: List[dict] = []
        self.downloads: List[dict] = []
        self.verdict: Optional[dict] = None
        self._progress = progress

    # -- progress ---------------------------------------------------------- #
    def progress(self, message: str, fraction: Optional[float] = None) -> None:
        if self._progress is not None:
            self._progress(message, -1.0 if fraction is None else float(fraction))

    # -- content ----------------------------------------------------------- #
    def metric(self, label: str, value: Any, fmt: str = "num", *, digits: int = 2,
               hint: str = "", status: Optional[str] = None, unit: str = "", badge: str = "") -> "Result":
        """A headline number.  ``fmt='pct'`` expects a fraction (0.0123 -> 1,23 %).

        ``status`` adds a coloured badge (default labels: Normal / Elevated / High / Critical);
        ``badge`` overrides its text, e.g. ``badge="Fat tails"``.
        """
        assert fmt in FORMATS, fmt
        assert status in STATUSES, status
        self.metrics.append({"label": label, "value": to_json(value), "fmt": fmt, "digits": digits,
                             "hint": hint, "status": status, "unit": unit, "badge": badge})
        return self

    def set_verdict(self, level: str, title: str, text: str = "") -> "Result":
        """One overall reading (shown as a status banner).  level in good/warning/serious/critical."""
        assert level in ("good", "warning", "serious", "critical"), level
        self.verdict = {"level": level, "title": title, "text": text}
        return self

    def table(self, title: str, columns: List[str], rows: List[List[Any]], *,
              formats: Optional[List[str]] = None, note: str = "") -> "Result":
        """A table.  ``formats`` (per column) uses the metric codes, optionally with digits: ``"pct:0"``."""
        if formats is not None:
            assert len(formats) == len(columns)
            for f in formats:
                assert f is None or str(f).split(":")[0] in FORMATS, f
        self.tables.append({"title": title, "columns": list(columns), "rows": to_json(rows),
                            "formats": formats, "note": note})
        return self

    def chart(self, spec: dict) -> "Result":
        """An interactive chart built with :mod:`webtools.charts`."""
        self.charts.append(to_json(spec))
        return self

    def figure(self, fig, caption: str = "", *, dpi: int = 110) -> "Result":
        """A matplotlib figure from the original framework (rendered to PNG)."""
        self.figures.append({"caption": caption, "png": figure_to_png(fig, dpi=dpi)})
        return self

    def image(self, png_bytes: bytes, caption: str = "", mime: str = "image/png") -> "Result":
        self.figures.append({"caption": caption, "png": base64.b64encode(png_bytes).decode("ascii"), "mime": mime})
        return self

    def note(self, text: str, level: str = "info") -> "Result":
        assert level in ("info", "warning", "success"), level
        self.notes.append({"text": text, "level": level})
        return self

    def download(self, filename: str, content: str, mime: str = "text/plain") -> "Result":
        self.downloads.append({"filename": filename, "content": content, "mime": mime})
        return self

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "tables": self.tables, "charts": self.charts,
                "figures": self.figures, "notes": self.notes, "downloads": self.downloads,
                "verdict": self.verdict}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Any] = {}


def _load_registry() -> Dict[str, Any]:
    if _REGISTRY:
        return _REGISTRY
    from . import tools as tools_pkg

    for info in sorted(pkgutil.iter_modules(tools_pkg.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{tools_pkg.__name__}.{info.name}")
        spec = getattr(module, "SPEC", None)
        if spec is None:
            continue
        validate_spec(spec, info.name)
        _REGISTRY[spec["id"]] = module
    return _REGISTRY


def validate_spec(spec: dict, module_name: str = "?") -> None:
    """Fail loudly at import time if a SPEC is malformed (also exercised by tests)."""
    where = f"tool '{module_name}'"
    for key in ("id", "title", "group", "summary", "needs", "params", "covers", "cost"):
        if key not in spec:
            raise ValueError(f"{where}: SPEC is missing '{key}'")
    if spec["id"].replace("-", "_") != module_name and module_name != "?":
        raise ValueError(f"{where}: id '{spec['id']}' must match module name")
    if spec["group"] not in GROUPS:
        raise ValueError(f"{where}: unknown group '{spec['group']}'")
    if spec["cost"] not in COSTS:
        raise ValueError(f"{where}: unknown cost '{spec['cost']}'")
    for need in spec["needs"]:
        if need not in DATA_KEYS:
            raise ValueError(f"{where}: unknown data need '{need}'")
    names = set()
    for p in spec["params"]:
        for key in ("name", "type", "default", "label"):
            if key not in p:
                raise ValueError(f"{where}: param {p} is missing '{key}'")
        if p["type"] not in PARAM_TYPES:
            raise ValueError(f"{where}: param '{p['name']}' has unknown type '{p['type']}'")
        if p["name"] in names:
            raise ValueError(f"{where}: duplicate param '{p['name']}'")
        for dep, allowed in (p.get("show_if") or {}).items():
            if dep not in [q["name"] for q in spec["params"]] or not isinstance(allowed, list):
                raise ValueError(f"{where}: show_if of '{p['name']}' must map an existing param to a list")
        names.add(p["name"])
        if p["type"] == "select":
            values = [o["value"] for o in p.get("options", [])]
            if p["default"] not in values:
                raise ValueError(f"{where}: default of '{p['name']}' not among options")
        if p["type"] in ("int", "float"):
            lo, hi = p.get("min"), p.get("max")
            if lo is None or hi is None:
                raise ValueError(f"{where}: numeric param '{p['name']}' needs min and max")
            if not (lo <= p["default"] <= hi):
                raise ValueError(f"{where}: default of '{p['name']}' outside [{lo}, {hi}]")


def coerce_params(spec: dict, raw: Optional[dict]) -> dict:
    """Return a complete, typed, clamped parameter dict (unknown keys are dropped)."""
    raw = raw or {}
    out = {}
    for p in spec["params"]:
        name, kind, default = p["name"], p["type"], p["default"]
        value = raw.get(name, default)
        try:
            if kind == "int":
                value = int(round(float(value)))
            elif kind == "float":
                value = float(value)
                if not math.isfinite(value):
                    value = default
            elif kind == "bool":
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif kind == "select":
                allowed = [o["value"] for o in p["options"]]
                if value not in allowed:
                    # values may arrive as strings from HTML forms
                    match = [a for a in allowed if str(a) == str(value)]
                    value = match[0] if match else default
            elif kind == "text":
                value = str(value)[: p.get("maxlength", 200)]
        except (TypeError, ValueError):
            value = default
        if kind in ("int", "float"):
            value = min(max(value, p["min"]), p["max"])
            value = int(value) if kind == "int" else float(value)
        out[name] = value
    return out


def tool_specs() -> List[dict]:
    """All tool specs (JSON), in catalogue order."""
    specs = [to_json(m.SPEC) for m in _load_registry().values()]
    group_order = list(GROUPS)
    specs.sort(key=lambda s: (group_order.index(s["group"]), s.get("order", 50), s["title"]))
    return specs


def catalogue() -> dict:
    return {"groups": [{"id": k, "title": v} for k, v in GROUPS.items()], "tools": tool_specs()}


def run_tool(tool_id: str, params: Optional[dict] = None, data: Optional[Dataset] = None,
             progress: Optional[Callable[[str, float], None]] = None) -> dict:
    """Run one tool and return a JSON-ready dict.  Never raises: errors become ``error``."""
    registry = _load_registry()
    started = time.perf_counter()
    if tool_id not in registry:
        return {"ok": False, "error": f"Unknown tool: {tool_id}", "kind": "user"}
    module = registry[tool_id]
    spec = module.SPEC
    clean = coerce_params(spec, params)
    data = data or Dataset()
    out = Result(progress=progress)
    try:
        data.require(*[k for k in spec["needs"]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                module.run(clean, data, out)
    except ToolError as exc:
        return {"ok": False, "error": str(exc), "kind": "user", "params": clean,
                "seconds": time.perf_counter() - started}
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI with traceback
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "kind": "internal",
                "traceback": traceback.format_exc(limit=8), "params": clean,
                "seconds": time.perf_counter() - started}
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:  # pragma: no cover
            pass
    result = out.to_dict()
    result.update({"ok": True, "tool": tool_id, "params": clean, "seconds": time.perf_counter() - started,
                   "dataset": {"name": data.name, "n": data.n,
                               "start": data.dates[0] if data.dates else None,
                               "end": data.dates[-1] if data.dates else None}})
    return result
