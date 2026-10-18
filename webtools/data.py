"""
Turn a dataset description from the browser into a :class:`~webtools.core.Dataset`.

Sources
-------
``market``     arrays supplied by the page (FRED snapshot: S&P 500, NASDAQ, VIX, VIX3M)
``upload``     arrays parsed from the user's Excel/CSV file or clipboard paste
``synthetic``  generated here with ``src.utils.data_loader.generate_synthetic_returns``
``crisis``     generated here with ``src.utils.data_loader.generate_crisis_scenario``

Prices always become *daily log returns*; ``dates[i]`` is the date of
``returns[i]`` (the later of the two prices).  VIX / VIX3M are aligned to the
same dates.  Rows with missing values are dropped, never interpolated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .core import Dataset, ToolError

MIN_OBS = 60
SYNTHETIC_MODELS = {
    "levy_jump": "Gauß + Lévy-Sprünge (realistisch)",
    "garch": "GARCH(1,1) – Volatilitäts-Cluster",
    "student_t": "Student-t (ν = 4)",
    "levy": "Lévy-stabil (α = 1,7)",
    "regime_switching": "Regimewechsel (ruhig ↔ Stress)",
    "gaussian": "Normalverteilung (Referenz)",
}


def _arr(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    if hasattr(values, "to_py"):  # Pyodide JsProxy
        values = values.to_py()
    out = np.asarray(values, dtype=float).ravel()
    return out if out.size else None


def _dates(values: Any):
    if values is None:
        return None
    if hasattr(values, "to_py"):
        values = values.to_py()
    values = [str(v)[:10] for v in values]
    return values or None


def from_arrays(name: str, *, prices=None, returns=None, dates=None, vix=None, vix3m=None,
                returns_unit: str = "log", meta: Optional[Dict[str, Any]] = None) -> Dataset:
    """Build a clean, aligned dataset from raw columns.

    ``returns_unit``: ``log`` (decimal log returns), ``simple`` (decimal simple
    returns) or ``percent`` (simple returns in %).
    """
    prices, returns, vix, vix3m = _arr(prices), _arr(returns), _arr(vix), _arr(vix3m)
    dates = _dates(dates)
    if prices is None and returns is None:
        raise ToolError("Der Datensatz enthält weder Kurse noch Renditen.")

    if returns is None:
        n = len(prices)
        cols = {"p": prices, "d": dates, "v": vix, "v3": vix3m}
        keep = np.isfinite(prices) & (prices > 0)
        for key in ("v", "v3"):
            if cols[key] is not None and len(cols[key]) != n:
                raise ToolError("VIX-Reihe und Kursreihe haben unterschiedliche Längen.")
        prices_c = prices[keep]
        idx = np.flatnonzero(keep)
        if len(prices_c) < MIN_OBS + 1:
            raise ToolError(f"Zu wenige gültige Kurse ({len(prices_c)}). Mindestens {MIN_OBS + 1} nötig.")
        returns = np.diff(np.log(prices_c))
        ridx = idx[1:]  # a return belongs to the later price
        out_dates = [dates[i] for i in ridx] if dates is not None else None
        out_vix = vix[ridx] if vix is not None else None
        out_vix3m = vix3m[ridx] if vix3m is not None else None
        out_prices = prices_c[1:]
    else:
        if returns_unit == "percent":
            returns = np.log1p(returns / 100.0)
        elif returns_unit == "simple":
            returns = np.log1p(returns)
        out_dates, out_vix, out_vix3m, out_prices = dates, vix, vix3m, None

    # drop rows where the return itself is missing; keep VIX gaps as NaN-free by dropping too
    mask = np.isfinite(returns)
    for series in (out_vix, out_vix3m):
        if series is not None:
            if len(series) != len(returns):
                raise ToolError("VIX-Reihe und Renditereihe haben unterschiedliche Längen.")
    if out_vix is not None and np.isfinite(out_vix).mean() > 0.9:
        mask &= np.isfinite(out_vix)
    else:
        out_vix = None
    if out_vix3m is not None and np.isfinite(out_vix3m).mean() > 0.9:
        mask &= np.isfinite(out_vix3m)
    else:
        out_vix3m = None

    def pick(a):
        return None if a is None else a[mask]

    returns = returns[mask]
    if len(returns) < MIN_OBS:
        raise ToolError(f"Zu wenige gültige Beobachtungen ({len(returns)}). Mindestens {MIN_OBS} nötig.")
    if np.nanmax(np.abs(returns)) > 1.5:
        raise ToolError("Die Renditen wirken unplausibel groß (> 150 % an einem Tag). "
                        "Prüfe, ob die Spalte Kurse statt Renditen enthält oder Prozentwerte sind.")
    return Dataset(
        name=name,
        returns=returns,
        prices=pick(out_prices),
        vix=pick(out_vix),
        vix3m=pick(out_vix3m),
        dates=[d for d, m in zip(out_dates, mask) if m] if out_dates is not None else None,
        meta=dict(meta or {}),
    )


def synthetic(model: str = "levy_jump", n: int = 2520, seed: int = 42) -> Dataset:
    from src.utils.data_loader import generate_synthetic_returns, generate_vix_from_returns

    if model not in SYNTHETIC_MODELS:
        raise ToolError(f"Unbekanntes Modell: {model}")
    n = int(min(max(n, MIN_OBS), 20000))
    returns = np.asarray(generate_synthetic_returns(n, model, seed=int(seed)), dtype=float)
    vix = np.asarray(generate_vix_from_returns(returns), dtype=float)
    return Dataset(name=f"Synthetisch · {SYNTHETIC_MODELS[model]}", returns=returns, vix=vix,
                   meta={"source": "synthetic", "model": model, "seed": int(seed)})


def crisis(seed: int = 43, n_pre: int = 500, n_crisis: int = 100, n_post: int = 400) -> Dataset:
    from src.utils.data_loader import generate_crisis_scenario, generate_vix_from_returns

    returns, meta = generate_crisis_scenario(n_pre=int(n_pre), n_crisis=int(n_crisis), n_post=int(n_post), seed=int(seed))
    returns = np.asarray(returns, dtype=float)
    vix = np.asarray(generate_vix_from_returns(returns), dtype=float)
    meta = {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in dict(meta).items()}
    meta["source"] = "crisis"
    return Dataset(name="Synthetisches Krisenszenario", returns=returns, vix=vix, meta=meta)


def build(descriptor: Dict[str, Any], arrays: Optional[Dict[str, Any]] = None) -> Dataset:
    """Entry point used by the worker: ``descriptor`` is small JSON, ``arrays`` are columns."""
    if hasattr(descriptor, "to_py"):
        descriptor = descriptor.to_py()
    if hasattr(arrays, "to_py"):
        arrays = arrays.to_py()
    arrays = arrays or {}
    source = descriptor.get("source")
    if source == "synthetic":
        return synthetic(descriptor.get("model", "levy_jump"), descriptor.get("n", 2520), descriptor.get("seed", 42))
    if source == "crisis":
        return crisis(descriptor.get("seed", 43))
    if source in ("market", "upload"):
        return from_arrays(
            descriptor.get("name", "Eigene Daten"),
            prices=arrays.get("prices"), returns=arrays.get("returns"), dates=arrays.get("dates"),
            vix=arrays.get("vix"), vix3m=arrays.get("vix3m"),
            returns_unit=descriptor.get("returns_unit", "log"),
            meta={"source": source, **{k: v for k, v in descriptor.items() if k not in ("source",)}},
        )
    raise ToolError(f"Unbekannte Datenquelle: {source}")


def index_axis(data: Dataset):
    """x values + axis type for time-series charts: real dates if present, else trading-day index."""
    from .charts import axis

    if data.dates:
        return list(data.dates), axis("", type="date")
    return list(range(data.n)), axis("Handelstag", type="linear", fmt="int")
