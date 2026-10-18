"""
Declarative, renderer-agnostic chart specs.

Tools describe *what* to plot; the browser (``site/assets/js/charts.js``)
decides *how* (Plotly, themed light/dark with the site's palette).  Colours
are referenced by role, never by hex:

* ``color=1..8``                 categorical slot (fixed order, never cycled)
* ``color="good"|"warning"|"serious"|"critical"``  status (only when it *means* risk state)
* ``color="muted"``              reference / baseline series (e.g. the Gaussian)

Long series are thinned with :func:`decimate`, which keeps the minimum and
maximum of every bucket - for tail risk the extremes *are* the signal.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

MAX_POINTS = 3000


def decimate(x: Sequence, y: Sequence, max_points: int = MAX_POINTS):
    """Thin (x, y) to about ``max_points`` while preserving each bucket's min and max."""
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n <= max_points:
        return list(x), y_arr
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    keep: List[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg = y_arr[lo:hi]
        if np.all(np.isnan(seg)):
            keep.append(lo)
            continue
        i_min = lo + int(np.nanargmin(seg))
        i_max = lo + int(np.nanargmax(seg))
        keep.extend(sorted({i_min, i_max}))
    keep_arr = np.array(sorted(set(keep)))
    x_list = list(x)
    return [x_list[i] for i in keep_arr], y_arr[keep_arr]


def axis(label: str = "", *, type: str = "linear", fmt: str = "num", digits: int = 2,
         range: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Axis description.  type: linear | log | date | category.  fmt: num | pct | int."""
    out = {"label": label, "type": type, "fmt": fmt, "digits": digits}
    if range is not None:
        out["range"] = [float(range[0]), float(range[1])]
    return out


def trace(name: str, x: Sequence, y: Sequence, *, type: str = "line", color: Any = 1,
          dash: bool = False, width: float = 2, fill: bool = False, decimate_to: Optional[int] = MAX_POINTS,
          text: Optional[Sequence[str]] = None, size: Optional[float] = None,
          yaxis_fmt: Optional[str] = None, error: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """One series.  type: line | step | bar | scatter | area.

    ``error`` adds symmetric error bars (same units as y), e.g. ±2 standard errors.
    """
    if type in ("line", "step", "area") and decimate_to and len(y) > decimate_to and error is None:
        x, y = decimate(x, y, decimate_to)
    return {"name": name, "type": type, "x": list(x), "y": np.asarray(y, dtype=float),
            "color": color, "dash": dash, "width": width, "fill": fill or type == "area",
            "text": list(text) if text is not None else None, "size": size,
            "error": np.asarray(error, dtype=float) if error is not None else None}


def envelope(name: str, x: Sequence, lower: Sequence, upper: Sequence, *, color: Any = 1,
             decimate_to: Optional[int] = MAX_POINTS) -> Dict[str, Any]:
    """A shaded band between two curves (confidence or percentile fan)."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    x = list(x)
    if decimate_to and len(x) > decimate_to:
        idx = np.linspace(0, len(x) - 1, decimate_to).round().astype(int)
        x, lo, hi = [x[i] for i in idx], lo[idx], hi[idx]
    return {"name": name, "type": "envelope", "x": x, "y": hi, "y0": lo, "color": color,
            "dash": False, "width": 0, "fill": True, "text": None, "size": None}


def band(x0: Any, x1: Any, label: str = "", color: Any = "critical") -> Dict[str, Any]:
    """Shaded x-range (crisis window, warning period)."""
    return {"x0": x0, "x1": x1, "label": label, "color": color}


def hline(y: float, label: str = "", color: Any = "muted", dash: bool = True) -> Dict[str, Any]:
    return {"y": float(y), "label": label, "color": color, "dash": dash}


def vline(x: Any, label: str = "", color: Any = "muted", dash: bool = True) -> Dict[str, Any]:
    return {"x": x, "label": label, "color": color, "dash": dash}


def xy(title: str, traces: Iterable[dict], *, x: Optional[dict] = None, y: Optional[dict] = None,
       subtitle: str = "", bands: Iterable[dict] = (), hlines: Iterable[dict] = (),
       vlines: Iterable[dict] = (), height: int = 340, stacked: bool = False,
       barmode: str = "group") -> Dict[str, Any]:
    """Cartesian chart (lines, bars, scatter).  One y-axis only - never dual axes."""
    return {"kind": "xy", "title": title, "subtitle": subtitle, "x": x or axis(), "y": y or axis(),
            "traces": list(traces), "bands": list(bands), "hlines": list(hlines), "vlines": list(vlines),
            "height": height, "barmode": "stack" if stacked else barmode}


def histogram(values: Sequence[float], bins: int = 80, density: bool = True,
              range: Optional[Sequence[float]] = None):
    """Pre-binned histogram -> (centers, heights).  Binning in Python keeps payloads small."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    heights, edges = np.histogram(v, bins=bins, density=density, range=range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, heights


def surface(title: str, xs: Sequence[float], ys: Sequence[float], zs, *, x: Optional[dict] = None,
            y: Optional[dict] = None, z: Optional[dict] = None, subtitle: str = "",
            paths: Iterable[dict] = (), points: Iterable[dict] = (), scale: str = "risk",
            height: int = 560, camera: Optional[dict] = None) -> Dict[str, Any]:
    """3D surface z[i][j] = f(xs[j], ys[i]) with optional 3D paths/points on top.

    scale: ``risk`` (calm -> black swan ramp) | ``sequential`` | ``diverging``.
    """
    return {"kind": "surface", "title": title, "subtitle": subtitle, "xs": list(xs), "ys": list(ys),
            "zs": np.asarray(zs, dtype=float), "x": x or axis(), "y": y or axis(), "z": z or axis(),
            "paths": list(paths), "points": list(points), "scale": scale, "height": height,
            "camera": camera}


def path3d(name: str, x: Sequence, y: Sequence, z: Sequence, *, color: Any = 1,
           mode: str = "lines", size: float = 4, text: Optional[Sequence[str]] = None,
           opacity: float = 1.0, width: Optional[float] = None) -> Dict[str, Any]:
    """A 3D path or point cloud.  Lower ``opacity`` keeps dense clouds from hiding highlighted paths."""
    return {"name": name, "x": list(map(float, x)), "y": list(map(float, y)), "z": list(map(float, z)),
            "color": color, "mode": mode, "size": size, "text": list(text) if text is not None else None,
            "opacity": float(opacity), "width": width}


def scatter3d(title: str, traces: Iterable[dict], *, x: Optional[dict] = None, y: Optional[dict] = None,
              z: Optional[dict] = None, subtitle: str = "", height: int = 560,
              camera: Optional[dict] = None) -> Dict[str, Any]:
    """3D lines / point clouds (Lévy flights, phase-space trajectories).

    ``camera`` (Plotly scene camera, e.g. ``{"eye": {"x": 1.6, "y": -1.6, "z": 1.2}}``) overrides the default view.
    """
    return {"kind": "scatter3d", "title": title, "subtitle": subtitle, "traces": list(traces),
            "x": x or axis(), "y": y or axis(), "z": z or axis(), "height": height, "camera": camera}


def heatmap(title: str, xs: Sequence, ys: Sequence, zs, *, x: Optional[dict] = None,
            y: Optional[dict] = None, z: Optional[dict] = None, scale: str = "sequential",
            subtitle: str = "", height: int = 380, annotate: bool = False, gap: int = 2) -> Dict[str, Any]:
    """Matrix view (correlations, warning matrices).  scale: sequential | diverging | risk.

    ``annotate=True`` prints every cell value (formatted with the z axis' fmt/digits) -
    use it for small matrices such as transition or correlation matrices.
    ``gap=0`` draws dense matrices (recurrence plots) without cell gaps.
    ``z=axis(..., range=(lo, hi))`` fixes the colour scale, so several heatmaps can share it.
    """
    return {"kind": "heatmap", "title": title, "subtitle": subtitle, "xs": list(xs), "ys": list(ys),
            "zs": np.asarray(zs, dtype=float), "x": x or axis(type="category"),
            "y": y or axis(type="category"), "z": z or axis(), "scale": scale, "height": height,
            "annotate": bool(annotate), "gap": int(gap)}


def gauge(title: str, value: float, *, max_value: float = 100.0,
          thresholds: Sequence[float] = (25, 50, 75), labels: Sequence[str] = ("Normal", "Elevated", "High", "Extreme"),
          subtitle: str = "") -> Dict[str, Any]:
    """A 0..max score with status bands (composite early-warning score)."""
    return {"kind": "gauge", "title": title, "subtitle": subtitle, "value": float(value),
            "max": float(max_value), "thresholds": list(map(float, thresholds)), "labels": list(labels)}
