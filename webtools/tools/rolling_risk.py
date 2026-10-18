"""Rolling VaR/ES backtest, rolling skewness/kurtosis/tail index and the 3D risk phase space."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import RollingRiskMetrics, compute_risk_3d_coordinates

from .. import charts as C
from ..core import ToolError
from ..data import index_axis, fmt_date

LEVELS = [{"value": v, "label": l} for v, l in ((0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"))]

SPEC = {
    "id": "rolling-risk",
    "title": "Rolling risk and VaR backtest",
    "group": "risk",
    "order": 50,
    "summary": "VaR, Expected Shortfall, skewness, kurtosis and tail index in a rolling window – with an out-of-sample "
               "backtest and the market's path through the 3D risk space.",
    "description": (
        "`RollingRiskMetrics` estimates each metric only from the last N trading days **before** the day in "
        "question. The rolling VaR is therefore a genuine forecast: if a day's return falls below −VaR, that is a "
        "**VaR breach**. At 99%, this should happen on about 1% of days; the **Kupiec test** checks whether the "
        "observed rate is consistent with that. The **cumulative breaches** chart shows when they occur: steep steps "
        "after calm phases mean that the window reacts too slowly to volatility clusters.\n\n"
        "**Excess kurtosis** (0 = Gaussian) and **skewness** show how the shape of the distribution changes; the "
        "**tail index α** (Hill or Pickands, from the largest 10% of absolute daily returns in both tails) measures "
        "the tail directly: α < 2 would mean infinite variance, α < 4 infinite kurtosis. In short windows all three "
        "are noisy – longer windows smooth them but react later.\n\n"
        "The **3D risk space** (`compute_risk_3d_coordinates`) scales three axes to 0–1: VaR level, ES/VaR ratio "
        "(how deep losses go beyond the VaR) and kurtosis. The scaling uses the whole series; the most dangerous "
        "phases lie at the top right rear. Losses are one-day losses expressed as positive decimals."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 250, "min": 20, "max": 500, "step": 5,
         "label": "Window (trading days)",
         "help": "250 ≈ one year (Basel standard for the 99% VaR); 60 ≈ three months: reacts faster but "
                 "barely captures rare losses."},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "VaR confidence level", "options": LEVELS},
        {"name": "tail_method", "type": "select", "default": "hill", "label": "Tail index estimator",
         "options": [{"value": "hill", "label": "Hill"}, {"value": "pickands", "label": "Pickands"}]},
    ],
    "covers": [
        "src.models.risk_metrics.RollingRiskMetrics",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_var",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_expected_shortfall",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_kurtosis",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_skewness",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_tail_index",
        "src.models.risk_metrics.RollingRiskMetrics.compute_all",
        "src.models.risk_metrics.compute_risk_3d_coordinates",
    ],
    "cost": "fast",
}



def _num(x, digits=2):
    return f"{x:.{digits}f}"


def _fmt_date(s):
    return fmt_date(s, "day {}")


def _kupiec(breaches, n, p):
    """Kupiec proportion-of-failures likelihood-ratio test."""
    if n == 0:
        return np.nan
    x = int(breaches)
    phat = x / n

    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)

    lr = -2 * (ll(p) - ll(phat))
    return float(stats.chi2.sf(max(lr, 0.0), 1))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    w, conf, method = int(params["window"]), float(params["confidence"]), params["tail_method"]
    if n < w + 30:
        raise ToolError(f"The series ({n:,} days) is too short for a window of {w} days: the backtest needs at "
                        f"least 30 days after the first window. Reduce the window to at most "
                        f"{max(n - 30, 0):,} days.")
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    rr = RollingRiskMetrics(r, window=w)
    allm = rr.compute_all()
    var = allm["var_99"] if conf == 0.99 else rr.rolling_var(conf)
    es = allm["es_99"] if conf == 0.99 else rr.rolling_expected_shortfall(conf)
    kurt, skew = allm["kurtosis"], allm["skewness"]
    alpha = allm["tail_index"] if method == "hill" else rr.rolling_tail_index(method)
    X, Y, Z = compute_risk_3d_coordinates(r, window=w)

    valid = np.isfinite(var)
    breach = valid & (r < -np.where(valid, var, 0.0))
    n_valid, n_breach = int(valid.sum()), int(breach.sum())
    rate = n_breach / n_valid if n_valid else np.nan
    p_kup = _kupiec(n_breach, n_valid, 1 - conf)
    x, xa = index_axis(data)
    i_max = int(np.nanargmax(var))
    last_alpha = alpha[np.isfinite(alpha)][-1] if np.any(np.isfinite(alpha)) else np.nan

    out.metric(f"{lvl} VaR today", var[-1], "pct", hint=f"Forecast for the last day from the {w} days before it",
               status="serious" if var[-1] > 1.5 * np.nanmedian(var) else None)
    out.metric(f"{lvl} Expected Shortfall today", es[-1], "pct")
    out.metric(f"{lvl} VaR · median", np.nanmedian(var), "pct", hint=f"Peak {_num(100 * var[i_max])}% on "
               f"{_fmt_date(x[i_max]) if data.dates else _fmt_date(i_max)}")
    out.metric("VaR breaches", rate, "pct", hint=f"{n_breach:,} of {n_valid:,} days · expected {_num(100 * (1 - conf), 1)}%",
               status="critical" if rate > 2.5 * (1 - conf) else "serious" if rate > 1.5 * (1 - conf) else "good")
    p_txt = "p < 0.001" if p_kup < 0.001 else f"p = {_num(p_kup, 3)}"
    out.metric("Kupiec test", "rejected" if p_kup < 0.05 else "passed", "text",
               hint=f"{p_txt} · rejected: breach rate inconsistent with the confidence level",
               status="warning" if p_kup < 0.05 else "good")
    out.metric("Excess kurtosis today", kurt[-1], "num", digits=1)
    out.metric("Skewness today", skew[-1], "num")
    out.metric(f"Tail index α today ({'Hill' if method == 'hill' else 'Pickands'})", last_alpha, "num",
               hint="Infinite variance below 2, infinite kurtosis below 4",
               status="critical" if last_alpha < 2 else "serious" if last_alpha < 3 else None)

    # --- chart 1: VaR / ES band against realised returns
    bx = [xi for xi, b in zip(x, breach) if b]
    out.chart(C.xy(
        f"Rolling {lvl} VaR and Expected Shortfall vs actual returns",
        [C.trace("Daily return", x, r, type="scatter", color="muted", size=2.5, decimate_to=None),
         C.trace(f"−VaR ({lvl})", x, -var, color=1, width=2),
         C.trace(f"−ES ({lvl})", x, -es, color=2, width=1.5),
         C.trace("VaR breach", bx, r[breach], type="scatter", color="critical", size=6, decimate_to=None)],
        x=xa, y=C.axis("Daily return", fmt="pct", digits=0),
        subtitle=f"Red dots: days below −VaR ({n_breach:,}; expected ≈ {round(n_valid * (1 - conf)):,}).",
    ))

    # --- chart 2: shape of the distribution
    out.chart(C.xy(
        f"Shape of the distribution in the {w}-day window",
        [C.trace("Excess kurtosis", x, kurt, color=1), C.trace("Skewness", x, skew, color=2)],
        x=xa, y=C.axis("Dimensionless"),
        hlines=[C.hline(0, "Gaussian")],
        subtitle="Kurtosis spikes mark crash windows; skewness then usually turns clearly negative.",
    ))

    # --- chart 3: tail index
    out.chart(C.xy(
        f"Rolling tail index α ({'Hill' if method == 'hill' else 'Pickands'})",
        [C.trace("α", x, alpha, color=1)],
        x=xa, y=C.axis("Tail index α"),
        hlines=[C.hline(2, "α = 2: infinite variance below", color="critical"),
                C.hline(4, "α = 4: finite kurtosis above")],
        subtitle="From the largest 10% of absolute daily returns in the window (gains and losses). Smaller = "
                 "fatter tail; very noisy in short windows.",
    ))

    # --- chart 4: cumulative breaches against the binomial expectation (classic backtest view)
    vi = np.flatnonzero(valid)
    if vi.size:
        p_b = 1 - conf
        cum = np.cumsum(breach[vi]).astype(float)
        expected = p_b * np.arange(1, vi.size + 1)
        half = 1.96 * np.sqrt(expected * (1 - p_b))
        xv = [x[i] for i in vi]
        out.chart(C.xy(
            "Cumulative VaR breaches vs expectation",
            [C.envelope("95% range if the VaR is correct", xv, np.maximum(expected - half, 0), expected + half,
                        color="muted"),
             C.trace("Expected", xv, expected, color="muted", dash=True),
             C.trace("Actual", xv, cum, type="step", color=1)],
            x=xa, y=C.axis("Breaches (count)", fmt="int"),
            subtitle="If the line runs above the band, the VaR understates risk; steep steps are breach "
                     "clusters.",
        ))

    # --- chart 5: 3D risk phase space
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    idx = np.flatnonzero(ok)
    if idx.size > 10:
        step = max(1, idx.size // 3000)
        hist = idx[::step]
        labels = [_fmt_date(x[i]) if data.dates else f"Day {i}" for i in range(n)]
        score = X + Y + Z
        hot = idx[score[idx] >= np.quantile(score[idx], 0.97)]
        recent = idx[-min(w, idx.size):]
        out.chart(C.scatter3d(
            "Path through the 3D risk space",
            [C.path3d("All days", X[hist], Y[hist], Z[hist], color="muted", mode="markers", size=1.5,
                      text=[labels[i] for i in hist]),
             C.path3d("Riskiest 3% of days", X[hot], Y[hot], Z[hot], color="critical", mode="markers", size=3,
                      text=[labels[i] for i in hot]),
             C.path3d(f"Last {len(recent)} days", X[recent], Y[recent], Z[recent], color=1, mode="lines+markers",
                      size=2.5, text=[labels[i] for i in recent]),
             C.path3d("Today", [X[idx[-1]]], [Y[idx[-1]]], [Z[idx[-1]]], color=2, mode="markers", size=7,
                      text=[labels[idx[-1]]])],
            x=C.axis("VaR level (0–1)"), y=C.axis("ES ÷ VaR (0–1)"), z=C.axis("Kurtosis (0–1)"),
            subtitle="All axes min-max scaled over the whole series. Drag to rotate.",
        ))

    beyond = w * (1 - conf)
    if beyond < 2:
        out.note(f"In the {w}-day window, on average only {_num(beyond, 1)} days lie beyond the {lvl} VaR. Historical "
                 "VaR is then little more than the worst day in the window and systematically understates risk – "
                 "use a longer window or a lower confidence level.", "warning")

    # --- compact table: calmest vs most stressed windows
    rows = []
    for name, sel in (("Calmest 25% of days", var <= np.nanquantile(var, 0.25)),
                      ("Most stressed 25% of days", var >= np.nanquantile(var, 0.75))):
        sel = sel & valid
        nxt = np.flatnonzero(sel)
        rows.append([name, np.nanmean(var[sel]), np.nanmean(es[sel]), np.nanmean(kurt[sel]), np.nanmean(skew[sel]),
                     np.nanmean(alpha[sel]), float(np.mean(breach[nxt])) if nxt.size else None])
    out.table("Calm vs stressed phases", ["Phase (by VaR)", f"Mean {lvl} VaR", f"Mean {lvl} ES",
                                          "Mean excess kurtosis", "Mean skewness", "Mean tail index", "Breach rate"],
              rows, formats=["text", "pct", "pct", "num", "num", "num", "pct"],
              note="Breach rate: share of days in this phase on which the return fell below −VaR.")
