"""Rolling Tsallis entropy and q: do they work as a leading indicator of volatility?"""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisEntropyAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tsallis-entropy",
    "title": "Tsallis entropy over time",
    "group": "regime",
    "order": 50,
    "summary": "Rolling Tsallis entropy and Tsallis index q alongside realised volatility and the VIX – with an "
               "honest test of whether entropy anticipates future turbulence.",
    "description": (
        "`TsallisEntropyAnalyzer.rolling_entropy` first estimates the index **q** in each window from the Hill "
        "tail index of the largest |returns| and then computes the Tsallis entropy S_q = (1 − Σpᵢ^q)/(q − 1) of "
        "the histogram of the window's returns. The idea from physics: rising entropy signals growing disorder "
        "and is supposed to precede turbulence.\n\n"
        "The tool tests this claim on your data: it compares entropy (and q) with the **future** realised "
        "volatility over the following days, against the simplest benchmark – past volatility. Spearman's rank "
        "correlation is robust to outliers. The quintile chart shows whether the days with the highest entropy "
        "really were followed by the most turbulent weeks.\n\n"
        "Worth knowing: the histogram has a fixed number of bins spanning the window's range. A single crash day "
        "stretches the range and squeezes all other returns into a few bins – entropy then tends to **fall**. "
        "Entropy measures the shape of the distribution, not its width."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 30, "max": 250, "step": 5,
         "label": "Window (trading days)"},
        {"name": "n_bins", "type": "int", "default": 25, "min": 10, "max": 100, "step": 5,
         "label": "Histogram bins"},
        {"name": "horizon", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Lead test: volatility over the next … days"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 15000, "step": 250,
         "label": "Period (most recent trading days)", "help": "Limits computation time for very long series."},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.rolling_entropy",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.estimate_optimal_q",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.compute_tsallis_entropy",
    ],
    "cost": "medium",
}


def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma."""
    return f"{x:,.{digits}f}".replace("-", "−")


def _rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(stats.spearmanr(a[ok], b[ok])[0])


def _quintile_means(signal, target):
    ok = np.isfinite(signal) & np.isfinite(target)
    s, t = signal[ok], target[ok]
    if len(s) < 50:
        return [None] * 5
    edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    group = np.searchsorted(edges, s, side="right")
    return [float(np.mean(t[group == g])) if np.any(group == g) else None for g in range(5)]


def run(params, data, out):
    w, bins, h = params["window"], params["n_bins"], params["horizon"]
    m = min(params["last_n"], data.n)
    r = np.asarray(data.returns[-m:], dtype=float)
    if data.n < 30 + 5 + 60:
        raise ToolError(f"With {data.n:,} returns the series is too short: the window, the lead test and enough "
                        "comparison days require at least 95 trading days. Choose a longer dataset.")
    if m < w + h + 60:
        raise ToolError(f"For a window of {w} days and a {h}-day lead the series is too short ({m:,} days). "
                        "Reduce the window or the lead" + (" or extend the period." if m < data.n else "."))
    x = list(data.dates[-m:]) if data.dates else list(range(data.n - m, data.n))
    xa = C.axis("", type="date") if data.dates else C.axis("Trading day", fmt="int")
    vix = np.asarray(data.vix[-m:], dtype=float) / 100.0 if data.vix is not None else None
    vix_name = "VIX" if data.meta.get("source") not in ("synthetic", "crisis") else "VIX (derived from returns)"

    analyzer = TsallisEntropyAnalyzer()
    out.progress("Rolling Tsallis entropy …", 0.1)
    S = np.asarray(analyzer.rolling_entropy(r, window=w, n_bins=bins), dtype=float)
    S[:w] = np.nan
    q = np.full(m, np.nan)
    for i in range(w, m):
        q[i] = analyzer.estimate_optimal_q(r[i - w:i])

    win = np.lib.stride_tricks.sliding_window_view(r, w)          # win[j] = r[j : j+w]
    past = np.full(m, np.nan)
    past[w:] = np.std(win[:-1], axis=1) * np.sqrt(252)              # uses r[i-w : i], like the entropy
    fwd = np.lib.stride_tricks.sliding_window_view(r, h)
    future = np.full(m, np.nan)
    future[: m - h + 1] = np.std(fwd, axis=1) * np.sqrt(252)        # r[i : i+h], strictly after the window
    future[:w] = np.nan

    # --- headline numbers
    valid = np.isfinite(S)
    s_now, q_now = float(S[-1]), float(q[-1])
    pct_s = float(np.mean(S[valid] <= s_now))
    pct_q = float(np.mean(q[valid] <= q_now))
    rho_s, rho_q, rho_p = _rho(S, future), _rho(q, future), _rho(past, future)
    out.metric("Tsallis entropy today", s_now, "num", hint=f"higher than on {_num(100 * pct_s, 0)}% of days")
    out.metric("q today (Hill)", q_now, "num", hint=f"higher than on {_num(100 * pct_q, 0)}% of days")
    out.metric("Realised volatility today", float(past[-1]), "pct", digits=1,
               hint=f"{w} days up to yesterday, annualised – matching the entropy window")
    out.metric(f"Rank correlation entropy → {h}-day vol", rho_s, "num",
               hint="Spearman ρ with future volatility")
    out.metric(f"For comparison: vol → {h}-day vol", rho_p, "num",
               hint="past volatility as the simplest predictor")

    # --- charts over time
    med = float(np.nanmedian(S))
    out.chart(C.xy(
        "Rolling Tsallis entropy S_q",
        [C.trace("S_q", x, S, color=1, width=1.3)], x=xa, y=C.axis("S_q"),
        hlines=[C.hline(med, "Median")],
        subtitle=f"Window {w} days, {bins} bins, q per window from the Hill estimator.",
    ))
    out.chart(C.xy(
        "Rolling Tsallis index q",
        [C.trace("q (Hill)", x, q, color=2, width=1.3)], x=xa, y=C.axis("q"),
        hlines=[C.hline(1.0, "Gaussian (q = 1)"), C.hline(5 / 3, "Variance → ∞ (q = 5/3)")],
        subtitle="The Hill estimate from a single window is very noisy; jumps in q reflect individual extreme days in the window.",
    ))
    vol_traces = [C.trace(f"Realised volatility ({w} days)", x, past, color=1, width=1.5)]
    if vix is not None:
        vol_traces.append(C.trace(vix_name, x, vix, color=2, width=1.2))
    out.chart(C.xy(
        "Realised volatility" + (" and VIX" if vix is not None else ""), vol_traces, x=xa,
        y=C.axis("Annualised volatility", fmt="pct", digits=0),
        subtitle="For reference: when was the market genuinely turbulent?",
    ))

    # --- does it lead? quintile chart
    labels = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    traces = [C.trace("Entropy S_q", labels, _quintile_means(S, future), color=1, width=2.5),
              C.trace("q (Hill)", labels, _quintile_means(q, future), color=2, width=2.5),
              C.trace("Past volatility", labels, _quintile_means(past, future), color=3, width=2.5)]
    out.chart(C.xy(
        f"Volatility over the next {h} days by signal quintile", traces,
        x=C.axis("Signal quintile on day t", type="category"),
        y=C.axis(f"Avg. realised volatility t … t+{h}", fmt="pct", digits=0),
        subtitle="For each line, all days are sorted by the signal into five groups of equal size. A good "
                 "early-warning signal rises markedly from Q1 to Q5.",
    ))

    rows = [["Entropy S_q", rho_s, _rho(S, past)], ["q (Hill)", rho_q, _rho(q, past)],
            [f"Past volatility ({w} days)", rho_p, 1.0]]
    cols = ["Signal on day t", f"With vol t … t+{h}", f"With vol t−{w} … t"]
    if vix is not None:
        for row, sig in zip(rows, (S, q, past)):
            row.append(_rho(sig, vix))
        cols.append("With VIX")
    out.table("Rank correlations (Spearman)", cols,
              rows, formats=["text"] + ["num"] * (len(cols) - 1),
              note=f"{int(np.sum(np.isfinite(S) & np.isfinite(future))):,} days. The first correlation column is the actual "
                   "lead test: only data up to t−1 enter the signal, and the volatility that follows lies genuinely "
                   "in the future.")

    if rho_s is not None and rho_p is not None:
        if rho_s > 0.2 and rho_s >= rho_p - 0.05:
            out.note(f"Here entropy does anticipate turbulence: ρ = {_num(rho_s)}, as good as past volatility "
                     f"(ρ = {_num(rho_p)}).", "success")
        elif rho_p > 0.2:
            out.note(f"On this dataset entropy gives little advance warning: ρ = {_num(rho_s)} with future volatility, whereas "
                     f"plain past volatility reaches ρ = {_num(rho_p)}.", "warning")
        else:
            out.note(f"Future volatility is hardly predictable here: neither entropy (ρ = {_num(rho_s)}) nor past "
                     f"volatility (ρ = {_num(rho_p)}) has any meaningful lead.", "info")
    if m < data.n:
        out.note(f"Analysing the most recent {m:,} of {data.n:,} trading days (parameter ‘Period’).", "info")
