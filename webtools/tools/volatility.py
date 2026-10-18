"""Estimating volatility: rolling window vs EWMA (RiskMetrics) vs realised, with a VaR backtest."""

import numpy as np
from scipy import stats

from src.utils.numerical import exponential_weighted_mean, rolling_window, safe_divide
from src.utils.statistics import compute_moments

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "volatility",
    "title": "Volatility: rolling, EWMA, realised",
    "group": "risk",
    "order": 40,
    "summary": "Rolling window versus RiskMetrics EWMA: how quickly a volatility estimate reacts to stress, "
               "measured against subsequently realised volatility and a VaR backtest.",
    "description": (
        "Volatility cannot be observed, only estimated. The **rolling window** (`rolling_window`) gives equal "
        "weight to the last n days and abruptly forgets everything before them. The **EWMA** "
        "(`exponential_weighted_mean` on squared returns, RiskMetrics standard λ = 0.94) down-weights each older "
        "day by the factor λ: it reacts immediately to a shock and lets it fade out slowly. Both are compared with "
        "the **realised** volatility of the following days – the value a forecast should have hit.\n\n"
        "How to read the results: the time series shows which estimator reacts faster to crises. The table "
        "measures the forecast error (RMSE) and counts **VaR breaches**: days on which the loss exceeded the "
        "Gaussian VaR based on the previous day's volatility. At 99% they should occur on 1% of days; markedly more "
        "means the model underestimates the risk. Dividing returns by the previous day's volatility "
        "(`safe_divide`) usually reduces the kurtosis: part of the fat tails is volatility clustering – the rest "
        "remains.\n\n"
        "Parameters: the **window** sets the memory of the rolling estimator, **λ** that of the EWMA (closer to 1 "
        "= slower), and the **horizon** the number of days over which realised volatility is measured."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 21, "min": 5, "max": 250, "step": 1,
         "label": "Rolling window", "unit": "days", "help": "21 trading days ≈ one month."},
        {"name": "lam", "type": "float", "default": 0.94, "min": 0.80, "max": 0.995, "step": 0.005,
         "label": "EWMA decay factor λ", "help": "RiskMetrics: 0.94 for daily data. Half-life ≈ ln 0.5 / ln λ days."},
        {"name": "horizon", "type": "int", "default": 21, "min": 5, "max": 63, "step": 1,
         "label": "Realised volatility horizon", "unit": "days"},
        {"name": "level", "type": "select", "default": "0.99", "label": "VaR confidence level (backtest)",
         "options": [{"value": "0.99", "label": "99%"}, {"value": "0.975", "label": "97.5%"},
                     {"value": "0.95", "label": "95%"}]},
    ],
    "covers": [
        "src.utils.numerical.rolling_window",
        "src.utils.numerical.exponential_weighted_mean",
        "src.utils.numerical.safe_divide",
        "src.utils.statistics.compute_moments",
    ],
    "cost": "fast",
}
ANN = np.sqrt(252)


def _num(x, digits=2):
    return f"{x:.{digits}f}"


def _lvl(p):
    """0.99 -> '99%', 0.975 -> '97.5%'."""
    return f"{p * 100:.1f}".rstrip("0").rstrip(".") + "%"


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    w, h, lam = int(params["window"]), int(params["horizon"]), float(params["lam"])
    level = float(params["level"])
    if n < w + h + 20:  # short series: shorten the horizon first, then the window
        w0, h0 = w, h
        h = max(5, min(h, n - w - 20))
        w = max(5, min(w, n - h - 20))
        if n < w + h + 20:
            raise ToolError("A window and a horizon of 5 days each require at least 30 observations. "
                            "Choose a longer period or another dataset.")
        changed = [f"{name} to {new} days (from {old})"
                   for name, new, old in (("the window", w, w0), ("the horizon", h, h0)) if new != old]
        out.note(f"The series has only {n} observations. To leave at least 20 days for the comparison, the tool "
                 f"shortens {' and '.join(changed)}.", "info")
    x, xa = index_axis(data)

    # estimates known at the close of day t
    roll = np.full(n, np.nan)
    roll[w - 1:] = rolling_window(r, w).std(axis=1)
    ewma = np.sqrt(exponential_weighted_mean(r ** 2, alpha=1.0 - lam))
    ewma[0] = np.nan  # a single squared return is no estimate yet
    # what actually happened over the next h days (ex post)
    realised = np.full(n, np.nan)
    realised[:n - h] = np.sqrt(rolling_window(r ** 2, h).mean(axis=1))[1:]
    long_run = float(np.std(r))

    # forecasts of day t are judged against days t+1..t+h
    valid = np.isfinite(roll) & np.isfinite(realised) & np.isfinite(ewma)
    q = stats.norm.ppf(level)
    prev = lambda s: np.concatenate([[np.nan], s[:-1]])  # noqa: E731 - estimate from the day before
    rows, z_kurt, breaches_ewma = [], {}, None
    for key, name, est in (("roll", f"Rolling ({w} days)", roll), ("ewma", f"EWMA (λ = {_num(lam, 3)})", ewma),
                           ("const", "Constant (full-sample σ, benchmark)", np.full(n, long_run))):
        e, rv = est[valid] * ANN, realised[valid] * ANN
        sig_prev = prev(est)
        z = safe_divide(r, sig_prev, fill=np.nan)
        ok = np.isfinite(z)
        z_kurt[key] = float(compute_moments(z[ok])["kurtosis"])
        breach = ok & (r < -q * sig_prev)
        rate = float(breach.sum() / ok.sum())
        pval = float(stats.binomtest(int(breach.sum()), int(ok.sum()), 1 - level).pvalue)
        if key == "ewma":
            breaches_ewma = breach
        rmse = float(np.sqrt(np.mean((e - rv) ** 2))) if key != "const" else None
        rows.append([name, float(np.mean(e)), rmse, float(np.mean(e - rv)) if key != "const" else None,
                     float(np.corrcoef(e, rv)[0, 1]) if key != "const" else None, rate, pval, z_kurt[key]])
    raw_kurt = float(compute_moments(r)["kurtosis"])

    ratio = float(safe_divide(ewma[-1], long_run, fill=np.nan))
    out.metric("Volatility today (EWMA)", ewma[-1] * ANN, "pct", digits=1, hint="Annualised",
               status="critical" if ratio > 2.5 else "serious" if ratio > 1.6 else "warning" if ratio > 1.2 else None)
    out.metric(f"Volatility today ({w} days)", roll[-1] * ANN, "pct", digits=1, hint="Annualised")
    out.metric("Long-run volatility", long_run * ANN, "pct", digits=1, hint=f"Over all {n:,} days")
    out.metric("EWMA ÷ long-run", ratio, "x", hint="Above 1: more turbulent than average")
    out.metric(f"{_lvl(level)} VaR breaches (EWMA)", rows[1][5], "pct", digits=2,
               hint=f"Target {_lvl(1 - level)}" + (" · binomial test: too many" if rows[1][6] < 0.05 and
                                                 rows[1][5] > 1 - level else ""),
               status="serious" if rows[1][6] < 0.05 and rows[1][5] > 1 - level else None)
    out.metric("Kurtosis after volatility scaling", z_kurt["ewma"], "num", digits=1,
               hint=f"Before: {_num(raw_kurt, 1)} – the rest is due to genuine jumps" if z_kurt["ewma"] < raw_kurt
               else f"Before: {_num(raw_kurt, 1)} – volatility clusters do not explain the fat tails here")

    out.chart(C.xy(
        "Volatility over time (annualised)",
        [C.trace(f"Realised over the next {h} days", x, realised * ANN, color="muted", width=1, dash=True),
         C.trace(f"Rolling ({w} days)", x, roll * ANN, color=1, width=1.3),
         C.trace(f"EWMA (λ = {_num(lam, 3)})", x, ewma * ANN, color=2, width=1.3)],
        x=xa, y=C.axis("Volatility p.a.", fmt="pct", digits=0),
        hlines=[C.hline(long_run * ANN, "long-run")],
        subtitle="Dashed: what actually happened next. The EWMA jumps immediately on a shock; the window does so only "
                 "once enough crisis days fall inside it – and it drops abruptly when they leave.",
    ))

    lags = np.arange(0, min(251, n))
    wt_e = (1 - lam) * lam ** lags
    wt_r = np.where(lags < w, 1.0 / w, 0.0)
    half = np.log(0.5) / np.log(lam)
    out.chart(C.xy(
        "Weight of each past day",
        [C.trace(f"Rolling ({w} days)", lags, wt_r, type="step", color=1),
         C.trace(f"EWMA (λ = {_num(lam, 3)})", lags, wt_e, color=2)],
        x=C.axis("Days back", fmt="int"), y=C.axis("Weight", fmt="pct", digits=1),
        vlines=[C.vline(half, f"EWMA half-life {_num(half, 1)} days")],
        subtitle="The window forgets everything after exactly n days; the EWMA forgets gradually.",
    ))

    zmax = np.nanmax(np.abs(safe_divide(r, prev(np.full(n, long_run)), fill=np.nan)))
    for est in (roll, ewma):
        zz = safe_divide(r, prev(est), fill=np.nan)
        zmax = max(zmax, float(np.nanmax(np.abs(zz[np.isfinite(zz)])))) if np.isfinite(zz).any() else zmax
    lim = float(np.clip(np.ceil(zmax + 0.5), 5, 15))
    edges = (-lim, lim)
    traces, floor = [], np.inf
    for key, name, col, scale in (("const", "Return ÷ full-sample σ", "muted", np.full(n, long_run)),
                                  ("roll", "÷ previous day's rolling σ", 1, roll),
                                  ("ewma", "÷ previous day's EWMA σ", 2, ewma)):
        z = safe_divide(r, prev(scale), fill=np.nan)
        z = z[np.isfinite(z)]
        centers, dens = C.histogram(z, bins=int(8 * lim) + 1, density=True, range=edges)
        floor = min(floor, float(np.min(dens[dens > 0])))
        traces.append(C.trace(f"{name} (kurtosis {_num(z_kurt[key], 1)})", centers, np.where(dens > 0, dens, np.nan),
                              color=col, width=1.4))
    grid = np.linspace(*edges, 300)
    pdf = stats.norm.pdf(grid)
    traces.append(C.trace("Normal distribution", grid, np.where(pdf >= floor / 4, pdf, np.nan), color="ink", width=1,
                          dash=True))
    out.chart(C.xy(
        "Standardised returns – log scale",
        traces, x=C.axis("Return in standard deviations"), y=C.axis("Density", type="log", range=(floor / 4, 1.0)),
        subtitle="Divided by the previous day's volatility, the tails move closer to the Gaussian curve – but not "
                 "all the way.",
    ))

    var_line = -q * prev(ewma)
    idx = np.flatnonzero(breaches_ewma)
    out.chart(C.xy(
        f"{_lvl(level)} VaR backtest (EWMA)",
        [C.trace("Daily return", x, r, color=1, width=1),
         C.trace("−VaR from the previous day's EWMA volatility", x, var_line, color=2, width=1.2),
         C.trace("Breach", [x[i] for i in idx], r[idx], type="scatter", color="critical", size=6, decimate_to=None)],
        x=xa, y=C.axis("Return", fmt="pct", digits=1),
        subtitle=f"{len(idx):,} breaches over {int(np.isfinite(var_line).sum()):,} days – "
                 f"{(1 - level) * np.isfinite(var_line).sum():,.0f} expected.",
    ))

    out.table(f"Forecast accuracy against realised volatility over the next {h} days",
              ["Estimator", "Mean volatility p.a.", "RMSE", "Bias", "Correlation", "VaR breaches", "p-value",
               "Kurtosis after scaling"], rows,
              formats=["text", "pct", "pct", "pct", "num", "pct", "num", "num"],
              note=f"RMSE and bias in annualised volatility points. VaR = normal quantile {_num(q, 3)} × the previous "
                   "day's σ; p-value: binomial test against the target rate (below 0.05 = the model does not fit). The "
                   "constant estimator knows the whole series and serves only as a benchmark.")
