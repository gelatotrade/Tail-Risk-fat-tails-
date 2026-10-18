"""Volatility regimes: Markov switching, realised-vol regimes and regime-conditional tail risk."""

import numpy as np

from src.models.regime_detection import (
    MarkovRegimeSwitching,
    RegimeAwareTailRisk,
    VolatilityClusteringDetector,
    compute_regime_coordinates,
)
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..core import ToolError
from ..data import index_axis, fmt_date

LEVELS = [{"value": v, "label": l} for v, l in ((0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"))]
STATE_NAMES = {2: ["Calm", "Stress"], 3: ["Calm", "Elevated", "Stress"]}
STATE_COLORS = {2: ["good", "critical"], 3: ["good", "warning", "critical"]}
VOL_NAMES = {"LOW_VOL": "Low", "MEDIUM_VOL": "Medium", "HIGH_VOL": "High"}
DAYS = 252
SPEC = {
    "id": "regimes",
    "title": "Volatility regimes (Markov)",
    "group": "regime",
    "order": 10,
    "summary": "Detects calm and stressed market phases with a Markov regime-switching model and shows how "
               "strongly VaR and Expected Shortfall depend on the regime.",
    "description": (
        "The **Markov regime-switching model** (`MarkovRegimeSwitching`) assumes that the market jumps between a "
        "small number of states, each with its own volatility. Which state prevails is unobserved: the EM "
        "algorithm learns the means, volatilities and the **transition matrix** from the full series, and with "
        "these parameters the **Hamilton filter** estimates each day's state probabilities from the returns up to "
        "that day only. The diagonal of the transition matrix measures persistence: p = 0.98 means that a regime "
        "lasts 1/(1 − p) = 50 days on average.\n\n"
        "Alongside this sits a simple classification based on realised 20-day volatility "
        "(`detect_vol_regime`): above the chosen percentile is high, below the complementary percentile (100 minus the chosen one) is low. "
        "`RegimeAwareTailRisk` uses it to compute VaR per regime – under stress it is often twice the average. "
        "The **Ljung-Box test** on squared returns checks whether volatility clusters at all.\n\n"
        "How to read the charts: the colour of the price line shows the most likely Markov regime; further charts "
        "show the stress probability and realised volatility with the boundaries of the volatility regimes. The "
        "**3D regime space** (`compute_regime_coordinates`) shows volatility level, persistence and switching "
        "tendency; high volatility combined with low persistence is the most unstable configuration."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "n_states", "type": "select", "default": 3, "label": "Markov states",
         "options": [{"value": 2, "label": "2 (calm / stress)"}, {"value": 3, "label": "3 (calm / elevated / stress)"}]},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "VaR confidence level", "options": LEVELS},
        {"name": "vol_percentile", "type": "int", "default": 75, "min": 55, "max": 95, "step": 5,
         "label": "High-volatility percentile", "help": "75: the most volatile 25% of days count as ‘high’."},
        {"name": "forecast_days", "type": "int", "default": 20, "min": 1, "max": 120, "step": 1,
         "label": "Forecast horizon (days)"},
        {"name": "coord_window", "type": "int", "default": 30, "min": 8, "max": 120, "step": 2,
         "label": "Window for the 3D regime space",
         "help": "Trading days per estimate of volatility and switching tendency."},
        {"name": "max_years", "type": "int", "default": 15, "min": 2, "max": 60, "step": 1,
         "label": "Period (most recent years)",
         "help": "Long series are cut to the most recent years to limit computation time."},
    ],
    "covers": [
        "src.models.regime_detection.MarkovRegimeSwitching",
        "src.models.regime_detection.MarkovRegimeSwitching.fit",
        "src.models.regime_detection.MarkovRegimeSwitching.current_regime",
        "src.models.regime_detection.MarkovRegimeSwitching.regime_probability",
        "src.models.regime_detection.MarkovRegimeSwitching.forecast_regime",
        "src.models.regime_detection.VolatilityClusteringDetector",
        "src.models.regime_detection.VolatilityClusteringDetector.ljung_box_test",
        "src.models.regime_detection.VolatilityClusteringDetector.rolling_realized_volatility",
        "src.models.regime_detection.VolatilityClusteringDetector.detect_vol_regime",
        "src.models.regime_detection.RegimeAwareTailRisk",
        "src.models.regime_detection.RegimeAwareTailRisk.current_regime",
        "src.models.regime_detection.RegimeAwareTailRisk.regime_conditional_var",
        "src.models.regime_detection.RegimeAwareTailRisk.regime_transition_risk",
        "src.models.regime_detection.compute_regime_coordinates",
    ],
    "cost": "medium",
}


def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma."""
    return f"{x:,.{digits}f}"


def _ordinal(k):
    k = int(k)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _fmt_date(s):
    return fmt_date(s, "Day {}")


def _pct0(p):
    return "below 1%" if p < 0.01 else f"{_num(100 * p, 0)}%"


def _days(d):
    return "1 day" if round(d) == 1 else f"{d:,.0f} days"


def _var_es(x, conf):
    if len(x) < 20:
        return None, None
    tm = TailRiskMetrics(x)
    return float(tm.var_historical(conf)), float(tm.expected_shortfall(conf))


def _segments(x, y, state, k):
    """One trace per state; each segment repeats its first point of the next state so lines join."""
    out = []
    for s in range(k):
        mask = state == s
        joined = mask.copy()
        joined[1:] |= mask[:-1]  # extend each run by one point to meet the next segment
        out.append(np.where(joined, y, np.nan))
    return out


def run(params, data, out):
    r_all = np.asarray(data.returns, dtype=float)
    k, conf = int(params["n_states"]), float(params["confidence"])
    keep = min(len(r_all), int(params["max_years"]) * DAYS)
    off = len(r_all) - keep
    r = r_all[off:]
    n = len(r)
    if n < 250:
        raise ToolError(f"Regime detection needs at least 250 days of data (available: {n:,}).")
    if np.std(r) <= 0:
        raise ToolError("The returns are constant – there are no volatility regimes.")
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])
    names, colors = STATE_NAMES[k], STATE_COLORS[k]

    out.progress("Detecting volatility regimes (EM algorithm) …", 0.1)
    ra = RegimeAwareTailRisk(r, vol_percentile=float(params["vol_percentile"]), markov_states=k)
    msm: MarkovRegimeSwitching = ra.markov
    if msm.filtered_probs is None:
        raise ToolError("The Markov model did not converge. Choose a different period.")

    probs = np.asarray(msm.filtered_probs, dtype=float)
    state = np.argmax(probs, axis=1)
    P = np.asarray(msm.transition_matrix, dtype=float)
    stds, means = np.asarray(msm.state_stds, dtype=float), np.asarray(msm.state_means, dtype=float)
    cur = int(msm.current_regime())
    cur_p = np.asarray(msm.regime_probability(), dtype=float)
    fc = np.asarray(msm.forecast_regime(int(params["forecast_days"])), dtype=float)

    det = VolatilityClusteringDetector(r)
    lb = det.ljung_box_test(20)
    rv = det.rolling_realized_volatility(20)
    cond = ra.regime_conditional_var(conf)
    trans = ra.regime_transition_risk()
    simple = ra.current_regime()
    X, Y, Z = compute_regime_coordinates(r, window=int(params["coord_window"]))

    dates = data.dates[off:] if data.dates else None
    x = dates if dates else list(range(n))
    xa = index_axis(data)[1]
    stay = 1 / np.maximum(1 - np.diag(P), 1e-9)
    var_by_state = [_var_es(r[state == s], conf) for s in range(k)]
    var_all = _var_es(r, conf)

    persistent = bool(lb["significant"]) and float(np.min(np.diag(P))) > 0.5
    status = {"good": "good", "warning": "warning", "critical": "serious"}
    out.metric("Current regime (Markov)", names[cur], "text", hint=f"Probability {_num(100 * cur_p[cur], 0)}%",
               status=status[colors[cur]] if persistent else None)
    out.metric("P(stress regime) today", cur_p[-1], "pct", digits=1 if cur_p[-1] >= 0.01 else 2)
    out.metric("Volatility in the stress regime", stds[-1] * np.sqrt(DAYS), "pct", digits=1,
               hint=f"calm: {_num(100 * stds[0] * np.sqrt(DAYS), 1)}% p.a.")
    out.metric("Avg. duration of the stress regime", stay[-1], "num", digits=0,
               unit="day" if round(stay[-1]) == 1 else "days", hint=f"calm: {_days(stay[0])}")
    if var_by_state[-1][0] is not None and var_all[0]:
        out.metric(f"{lvl} VaR under stress ÷ overall", var_by_state[-1][0] / var_all[0], "x",
                   hint=f"{_num(100 * var_by_state[-1][0])}% vs {_num(100 * var_all[0])}% · from "
                        f"{int(np.sum(state == k - 1)):,} stress days")
    out.metric("Realised volatility today (20 days)", rv[-1], "pct", digits=1,
               hint=f"Volatility regime: {VOL_NAMES.get(simple['simple_regime'], simple['simple_regime']).lower()}")
    p_lb = float(lb["p_value"])
    out.metric("Volatility clustering", "yes" if lb["significant"] else "no", "text",
               hint="Ljung-Box on squared returns · " + ("p < 0.001" if p_lb < 0.001 else f"p = {_num(p_lb, 3)}"))

    # --- chart 1: price coloured by regime
    if data.prices is not None:
        level, ylab = np.asarray(data.prices, dtype=float)[off:], "Price"
    else:
        level, ylab = np.exp(np.cumsum(r)), "Wealth (start = 1)"
    segs = _segments(x, level, state, k)
    out.chart(C.xy(
        "Price path by most likely Markov regime",
        [C.trace(f"{names[s]} regime", x, segs[s], color=colors[s], width=1.8) for s in range(k) if np.any(state == s)],
        x=xa, y=C.axis(ylab, type="log" if level.max() / max(level.min(), 1e-12) > 8 else "linear"),
        subtitle="Filtered states: each day uses only the returns up to that day; the model parameters are "
                 "estimated from the full series.",
    ))

    # --- chart 2: probability of the stress regime
    out.chart(C.xy(
        "Probability of the stress regime",
        [C.trace("P(stress)", x, probs[:, -1], type="area", color="critical", width=1)],
        x=xa, y=C.axis("Probability", fmt="pct", digits=0, range=(0, 1)),
        hlines=[C.hline(0.5, "50%")],
        subtitle="Filtered probability of the most volatile state; above 50% it is the most likely one.",
    ))

    # --- chart 3: realised volatility and the simple vol regimes
    vp = float(params["vol_percentile"])
    lo_thr, hi_thr = float(np.nanpercentile(rv, 100 - vp)), float(np.nanpercentile(rv, vp))
    out.chart(C.xy(
        "Realised 20-day volatility and volatility regimes",
        [C.trace("Realised volatility p.a.", x, rv, color=1, width=1.5)],
        x=xa, y=C.axis("Volatility p.a.", fmt="pct", digits=0),
        hlines=[C.hline(hi_thr, f"high: above {_ordinal(vp)} percentile"),
                C.hline(lo_thr, f"low: below {_ordinal(100 - vp)} percentile")],
        subtitle="Basis of the table ‘VaR by volatility regime’. The percentiles are taken from the full series.",
        height=260 + 30 * k,
    ))

    # --- chart 4: transition matrix
    lower = [s.lower() for s in names]
    out.chart(C.heatmap(
        "Transition matrix: P(tomorrow | today)", [f"to {s}" for s in lower], [f"from {s}" for s in lower], P,
        z=C.axis("Probability", fmt="pct", digits=1, range=(0, 1)), scale="sequential", annotate=True,
        height=260 + 30 * k,
        subtitle="Diagonal = persistence. Avg. duration: " + ", ".join(f"{s} {_days(d)}" for s, d in zip(lower, stay)),
    ))

    # --- chart 5: 3D regime space coloured by Markov state
    start = 2 * int(params["coord_window"])
    if n - start > 20:
        idx = np.arange(start, n)
        step = max(1, len(idx) // 3000)
        lab = [_fmt_date(v) if dates else f"Day {v}" for v in x]
        traces = []
        for s in range(k):
            sel = idx[state[idx] == s][::step]
            if sel.size:
                traces.append(C.path3d(f"{names[s]} regime", X[sel], Y[sel], Z[sel], color=colors[s], mode="markers",
                                       size=2.5, text=[lab[i] for i in sel]))
        traces.append(C.path3d("Today", [X[-1]], [Y[-1]], [Z[-1]], color=1, mode="markers", size=8, text=[lab[-1]]))
        out.chart(C.scatter3d(
            "3D regime space", traces,
            x=C.axis("Volatility level (0–1)"), y=C.axis("Persistence (0–1)"), z=C.axis("Switching tendency (0–1)"),
            subtitle="High volatility, low persistence and a high switching tendency = unstable stress.",
        ))

    # --- tables
    rows = []
    for s in range(k):
        share = float(np.mean(state == s))
        v_s, e_s = var_by_state[s]
        rows.append([names[s], stds[s] * np.sqrt(DAYS), means[s], P[s, s], round(float(stay[s])), share, v_s, e_s,
                     cur_p[s]])
    out.table("Markov states", ["Regime", "Volatility p.a.", "Avg. daily return", "Persistence p",
                                "Avg. duration (days)", "Share of days", f"{lvl} VaR", f"{lvl} ES", "P today"], rows,
              formats=["text", "pct", "pct", "pct", "int", "pct", "pct", "pct", "pct"],
              note="Historical VaR and ES from the days on which the regime was the most likely one "
                   "(– = fewer than 20 such days).")

    vrows = []
    for key in ("LOW_VOL", "MEDIUM_VOL", "HIGH_VOL", "UNCONDITIONAL"):
        c = cond[key]
        sub = r if key == "UNCONDITIONAL" else r[ra.regimes == ["LOW_VOL", "MEDIUM_VOL", "HIGH_VOL"].index(key)]
        vrows.append([VOL_NAMES.get(key, "All days"), c["n_observations"], c["var"], _var_es(sub, conf)[1]])
    out.table(f"{lvl} VaR by volatility regime (realised 20-day volatility)",
              ["Volatility regime", "Days", f"{lvl} VaR", f"{lvl} ES"], vrows, formats=["text", "int", "pct", "pct"],
              note=f"High = above the {_ordinal(params['vol_percentile'])} percentile, low = below the "
                   f"{_ordinal(100 - int(params['vol_percentile']))} percentile of realised volatility.")

    hs = [h for h in (1, 5, 10, 20, 60, 120) if h <= len(fc)]
    if len(fc) not in hs:
        hs.append(len(fc))
    out.table("Regime forecast", ["In … days"] + [f"P({s.lower()})" for s in names],
              [[f"{h}"] + list(fc[h - 1]) for h in hs], formats=["text"] + ["pct"] * k,
              note="Projected from the transition matrix; over long horizons it converges to the long-run "
                   "distribution.")

    out.table("After a volatility-regime switch (following 5 days)",
              ["Switch", "Count", "Avg. daily return", "Volatility (daily)"],
              [["To higher volatility", trans["n_up_transitions"], trans["post_up_mean_return"],
                trans["post_up_volatility"]],
               ["To lower volatility", trans["n_down_transitions"], trans["post_down_mean_return"],
                trans["post_down_volatility"]]],
              formats=["text", "int", "pct", "pct"])
    if off > 0:
        out.note(f"Analysing the most recent {params['max_years']} years ({n:,} of {len(r_all):,} days).", "info")

    if not persistent:
        out.set_verdict("good", "No pronounced volatility regimes",
                        f"The Ljung-Box test finds {'no' if not lb['significant'] else 'only weak'} "
                        f"clustering, and at least one state lasts only {_days(float(np.min(stay)))} on average. "
                        "Here the model separates normal days from jumps rather than identifying genuine market "
                        "phases.")
        return

    p_stress = float(cur_p[-1])
    v_s = var_by_state[-1][0]
    stress_txt = (f" In the stress regime, the {lvl} VaR is {_num(100 * v_s)}%, compared with {_num(100 * var_all[0])}% "
                  "overall." if v_s is not None and var_all[0] else "")
    if p_stress > 0.5:
        out.set_verdict("serious", f"The market is in the stress regime (P = {_num(100 * p_stress, 0)}%)",
                        f"Volatility is around {_num(100 * stds[-1] * np.sqrt(DAYS), 0)}% p.a.; stress phases "
                        f"last {_days(stay[-1])} on average." + stress_txt)
    elif cur != 0:
        out.set_verdict("warning", f"Elevated volatility (‘{names[cur]}’ regime)",
                        f"Stress probability {_pct0(p_stress)}." + stress_txt)
    else:
        out.set_verdict("good", f"Calm regime (P = {_num(100 * cur_p[0], 0)}%)",
                        f"Calm phases last {_days(stay[0])} on average. The stress probability is "
                        f"{_pct0(p_stress)}." + stress_txt)
