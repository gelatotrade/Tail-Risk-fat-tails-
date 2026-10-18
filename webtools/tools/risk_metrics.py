"""TailRiskMetrics: VaR four ways, Expected Shortfall, drawdown, Omega/Sortino/Calmar, tail dependence."""

import numpy as np

from src.models.extreme_value import GeneralizedParetoDistribution
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..data import index_axis, fmt_date

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"), (0.995, "99.5%"), (0.999, "99.9%"))]

SPEC = {
    "id": "risk-metrics",
    "title": "VaR and Expected Shortfall",
    "group": "risk",
    "order": 10,
    "summary": "Value at Risk four ways, Expected Shortfall, maximum drawdown and the Omega, Sortino and Calmar "
               "ratios – and how far the Gaussian assumption understates risk.",
    "description": (
        "**Value at Risk** (VaR) is the one-day loss that is not exceeded at the chosen confidence level. "
        "**Expected Shortfall** (ES) is the average loss on the days when the VaR is breached – the more honest "
        "measure when tails are fat. Four VaR methods: **historical** (quantile of the data), **Gaussian** "
        "(mean and volatility only), **Cornish-Fisher** (Gaussian quantile adjusted for skewness and kurtosis) "
        "and **EVT** (generalised Pareto distribution above a threshold).\n\n"
        "How to read the charts: the **VaR curve** shows the confidence level at which the methods diverge – "
        "the further right, the rarer the event. The **underwater chart** shows the distance from the previous "
        "wealth peak (wealth = exp(Σ log returns)). **Tail dependence** measures how often two extremes "
        "coincide on the same day – with VIX data: a crash day and a VIX spike; without VIX: a loss today and "
        "yesterday (volatility clustering).\n\n"
        "All losses are one-day losses expressed as positive decimals. Omega and Sortino refer to the target "
        "return; Omega > 1 means that gains above the target outweigh losses below it. Sortino divides the "
        "excess return by the downside deviation below the target return (days above it count as zero deviation)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Confidence level", "options": LEVELS},
        {"name": "evt_threshold", "type": "float", "default": 0.95, "min": 0.85, "max": 0.99, "step": 0.005,
         "label": "EVT threshold (quantile)", "help": "The GPD fit uses the losses above this quantile."},
        {"name": "target_annual", "type": "float", "default": 0.0, "min": -0.2, "max": 0.3, "step": 0.01,
         "label": "Target return p.a.", "help": "For Omega and Sortino; 0.05 = 5% per year."},
        {"name": "dependence_quantile", "type": "float", "default": 0.95, "min": 0.8, "max": 0.99, "step": 0.01,
         "label": "Tail dependence quantile"},
    ],
    "covers": [
        "src.models.risk_metrics.TailRiskMetrics",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.var_parametric_gaussian",
        "src.models.risk_metrics.TailRiskMetrics.var_cornish_fisher",
        "src.models.risk_metrics.TailRiskMetrics.var_evt",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
        "src.models.risk_metrics.TailRiskMetrics.tail_ratio",
        "src.models.risk_metrics.TailRiskMetrics.maximum_drawdown",
        "src.models.risk_metrics.TailRiskMetrics.omega_ratio",
        "src.models.risk_metrics.TailRiskMetrics.sortino_ratio",
        "src.models.risk_metrics.TailRiskMetrics.calmar_ratio",
        "src.models.risk_metrics.TailRiskMetrics.tail_dependence",
    ],
    "cost": "fast",
}

TABLE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)



def _num(x, digits=2):
    return f"{x:.{digits}f}"


def _label(c):
    return next(o["label"] for o in LEVELS if o["value"] == c)


def _date(data, i):
    return data.dates[i] if data.dates else f"Day {i}"




def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    conf, q_evt = float(params["confidence"]), float(params["evt_threshold"])
    target = float(params["target_annual"]) / 252.0
    evt_ok = n * (1 - q_evt) >= 10  # the GPD needs at least 10 losses above the threshold
    lvl = _label(params["confidence"])

    trm = TailRiskMetrics(r)
    table = {}
    for c in TABLE_LEVELS:
        table[c] = {
            "hist": float(trm.var_historical(c)), "gauss": float(trm.var_parametric_gaussian(c)),
            "cf": float(trm.var_cornish_fisher(c)),
            "evt": float(trm.var_evt(c, threshold_quantile=q_evt)) if evt_ok else np.nan,
            "es_hist": float(trm.expected_shortfall(c, "historical")),
            "es_gauss": float(trm.expected_shortfall(c, "gaussian")),
            # no observed loss beyond the CF-VaR: the src falls back to the VaR itself - show "–" instead
            "es_cf": (float(trm.expected_shortfall(c, "cornish_fisher"))
                      if np.any(r < -trm.var_cornish_fisher(c)) else None),
        }
    v = table[conf]
    dd = trm.maximum_drawdown()
    calmar, sortino, omega = float(trm.calmar_ratio()), float(trm.sortino_ratio(target)), float(trm.omega_ratio(target))

    i0, i1 = int(dd["max_dd_start"]), int(dd["max_dd_end"])
    rec = dd["recovery_time"]
    rec_txt = "not yet recovered" if not np.isfinite(rec) else f"recovered after {int(rec):,} days"
    out.metric(f"{lvl} VaR · historical", v["hist"], "pct", hint=f"ES {_num(100 * v['es_hist'])}%")
    out.metric(f"{lvl} VaR · Gaussian", v["gauss"], "pct", hint=f"ES {_num(100 * v['es_gauss'])}%")
    out.metric(f"{lvl} VaR · Cornish-Fisher", v["cf"], "pct",
               hint=f"ES {_num(100 * v['es_cf'])}%" if v["es_cf"] is not None else "no loss observed beyond it")
    if evt_ok:
        out.metric(f"{lvl} VaR · EVT (GPD)", v["evt"], "pct", hint=f"Threshold: {_num(100 * q_evt, 1)}% quantile")
    out.metric(f"{lvl} Expected Shortfall", v["es_hist"], "pct", hint="historical: average loss beyond the VaR",
               status="serious" if v["es_hist"] > 1.5 * v["es_gauss"] else "warning" if v["es_hist"] > 1.15 * v["es_gauss"] else None)
    out.metric("Maximum drawdown", dd["max_drawdown"], "pct", digits=1,
               hint=f"{fmt_date(_date(data, i0))} → {fmt_date(_date(data, i1))} · {rec_txt}",
               status="critical" if dd["max_drawdown"] > 0.4 else "serious" if dd["max_drawdown"] > 0.2 else None)
    out.metric("Calmar ratio", calmar, "num", hint="return p.a. ÷ max. drawdown")
    out.metric("Sortino ratio", sortino, "num", hint="annualised; risk = downside deviation below the target return")
    out.metric("Omega ratio", omega, "num", hint="gains ÷ losses relative to the target return")

    # --- chart 1: VaR across confidence levels (rarer events to the right)
    tails = np.geomspace(0.1, 0.001, 60)
    if evt_ok:
        gpd, st = GeneralizedParetoDistribution.fit_pot(-r, quantile=q_evt)  # the fit var_evt uses
        p_u = st["exceedance_probability"]
    curves = {"hist": [], "gauss": [], "cf": [], "evt": []}
    for a in tails:
        c = 1 - a
        curves["hist"].append(trm.var_historical(c))
        curves["gauss"].append(trm.var_parametric_gaussian(c))
        curves["cf"].append(trm.var_cornish_fisher(c))
        curves["evt"].append(gpd.var(c, p_u) if evt_ok and a < p_u else np.nan)
    sane = [curves[key][-1] for key in ("hist", "gauss", "evt") if np.isfinite(curves[key][-1])]
    y_top = 1.4 * max(sane)  # Cornish-Fisher may overshoot far beyond; it then leaves the chart
    out.chart(C.xy(
        "VaR by confidence level: where the methods diverge",
        [C.trace("Historical", tails, curves["hist"], color=4, type="step"),
         C.trace("Gaussian", tails, curves["gauss"], color="muted", dash=True),
         C.trace("Cornish-Fisher", tails, curves["cf"], color=3)]
        + ([C.trace("EVT (GPD)", tails, curves["evt"], color=1, width=2.5)] if evt_ok else []),
        x=C.axis("Exceedance probability 1 − c (per day)", type="log", fmt="pct", digits=1, range=(0.1, 0.001)),
        y=C.axis("VaR (one-day loss)", fmt="pct", digits=1,
                 range=(min(0.0, float(np.nanmin(curves["hist"] + curves["gauss"]))), y_top)),
        subtitle="Left: frequent losses (90% VaR); right: rare ones (99.9%). The Gaussian VaR falls short for rare events.",
        vlines=[C.vline(1 - conf, f"selected: {lvl}")],
    ))

    # --- chart 2: underwater curve
    x, xa = index_axis(data)
    ddser = np.asarray(dd["drawdown_series"], dtype=float)
    end_rec = i1 + int(rec) if np.isfinite(rec) else n - 1
    out.chart(C.xy(
        "Underwater chart: distance from the previous peak",
        [C.trace("Drawdown", x, ddser, type="area", color=1, width=1)],
        x=xa, y=C.axis("Drawdown", fmt="pct", digits=0),
        bands=[C.band(x[i0], x[i1], f"Max. drawdown −{_num(100 * dd['max_drawdown'], 1)}%", color="critical"),
               C.band(x[i1], x[min(end_rec, n - 1)], "", color="warning")],
        subtitle="Red: path into the maximum drawdown; yellow: " + (f"recovery to the previous peak ({int(rec):,} days)."
                                                                   if np.isfinite(rec) else "no recovery since."),
    ))

    # --- tail dependence: VIX changes if available, else yesterday's return
    qd = float(params["dependence_quantile"])
    if data.vix is not None and np.all(np.asarray(data.vix) > 0):
        dv = np.diff(np.log(np.asarray(data.vix, dtype=float)))
        a, b = r[1:], -dv  # a VIX spike is the "loss" of the second series
        partner, ylab, yser = "VIX change", "VIX change (log, daily)", dv
        explain = "crash day and VIX spike on the same day"
    else:
        a, b = r[1:], r[:-1]
        partner, ylab, yser = "previous-day return", "Previous-day return", r[:-1]
        explain = "loss today and yesterday (clustering)"
    tdep = trm.__class__(a).tail_dependence(b, quantile=qd)
    lo_a, lo_b = np.quantile(a, 1 - qd), np.quantile(b, 1 - qd)
    joint = (a < lo_a) & (b < lo_b)
    out.chart(C.xy(
        f"Tail dependence: daily return vs {partner}",
        [C.trace("All days", a[~joint], yser[~joint], type="scatter", color=1, size=4, decimate_to=None),
         C.trace(f"Both in the {_num(100 * (1 - qd), 0)}% tail", a[joint], yser[joint], type="scatter", color="critical",
                 size=7, decimate_to=None)],
        x=C.axis("Daily return", fmt="pct", digits=0), y=C.axis(ylab, fmt="pct", digits=0),
        subtitle=f"Red: {explain}. Under independence this would be {_num(100 * (1 - qd), 0)}% of the tail days.",
    ))

    # --- tables
    keys = ["hist", "gauss", "cf"] + (["evt"] if evt_ok else []) + ["es_hist", "es_gauss", "es_cf"]
    names = {"hist": "Historical VaR", "gauss": "Gaussian VaR", "cf": "Cornish-Fisher VaR", "evt": "EVT VaR",
             "es_hist": "Historical ES", "es_gauss": "Gaussian ES", "es_cf": "Cornish-Fisher ES"}
    out.table("VaR and Expected Shortfall by method", ["Confidence"] + [names[k] for k in keys],
              [[_label(c)] + [t[k] for k in keys] for c, t in table.items()],
              formats=["text"] + ["pct"] * len(keys),
              note="One-day losses. Cornish-Fisher ES = average of the observed losses beyond the Cornish-Fisher VaR "
                   "(– = no such loss observed)."
              + ("" if evt_ok else f" No EVT VaR: fewer than 10 losses lie above the {_num(100 * q_evt, 1)}% "
                                   "quantile – lower the EVT threshold or use a longer series."))

    tr_rows = []
    for k in (2, 3, 4, 5):
        t = trm.tail_ratio(k)
        asym = t["asymmetry"] if np.isfinite(t["asymmetry"]) else ("∞" if t["left_tail_prob"] > 0 else "–")
        tr_rows.append([f"{k}σ", t["left_tail_prob"], t["right_tail_prob"], t["gaussian_tail_prob"],
                        t["left_ratio"], t["right_ratio"], asym])
    out.table("Tail frequencies vs the normal distribution", ["Threshold", "P(loss > kσ)", "P(gain > kσ)",
                                                             "Gaussian (per tail)", "Left-tail factor",
                                                             "Right-tail factor", "Left/right asymmetry"],
              tr_rows, formats=["text", "pct", "pct", "pct", "x", "x", "num"],
              note="Asymmetry = P(loss > kσ) ÷ P(gain > kσ); > 1 means crashes are more frequent than rallies "
                   "of the same size.")

    out.table(f"Tail dependence (quantile {_num(qd)})", ["Metric", "Value"], [
        ["Lower tail dependence (both extremely bad)", tdep["lower_tail_dependence"]],
        ["Upper tail dependence (both extremely good)", tdep["upper_tail_dependence"]],
        ["Correlation (Pearson)", tdep["standard_correlation"]],
        ["Asymmetry (lower − upper)", tdep["tail_asymmetry"]],
    ], formats=["text", "num"],
        note=f"Partner series: {partner}" + (" (multiplied by −1 so that a VIX spike is the ‘bad’ tail)."
                                             if partner == "VIX change" else ".")
        + f" Independence would give ≈ {_num(1 - qd, 2)}, full dependence 1.")

    tail_days = n * (1 - conf)
    if tail_days < 10:
        out.note(f"At {lvl}, on average only {_num(tail_days, 1)} days lie beyond the VaR. Historical VaR and "
                 "Expected Shortfall then rest on a handful of losses and fluctuate strongly; the EVT VaR "
                 "extrapolates from the whole tail and is more robust here.", "info")

    cf_vals = [table[c]["cf"] for c in TABLE_LEVELS]
    if any(b2 < a2 for a2, b2 in zip(cf_vals, cf_vals[1:])):
        out.note("Cornish-Fisher is not monotonic here: at very high kurtosis the series expansion breaks down "
                 "and a higher confidence level yields a smaller VaR. Rely on the EVT or historical VaR instead.",
                 "warning")

    best_alt = np.nanmax([v["hist"], v["evt"]])
    best_name = "EVT" if evt_ok and v["evt"] >= v["hist"] else "historical"
    if v["cf"] > 1.5 * best_alt:
        out.note(f"Cornish-Fisher overshoots here (excess kurtosis {_num(float(trm.kurtosis), 1)}): the adjustment "
                 "for skewness and kurtosis is only intended for moderately fat tails. The historical and EVT VaR are "
                 "more reliable.", "warning")

    fat = best_alt
    r_var = fat / v["gauss"] if v["gauss"] > 0 else np.nan
    r_es = v["es_hist"] / v["es_gauss"] if v["es_gauss"] > 0 else np.nan
    worst = np.nanmax([r_var, r_es])
    if np.isfinite(worst):
        level = "serious" if worst > 1.25 else "warning" if worst > 1.05 else "good"
        pct = _num(100 * (worst - 1), 0)
        if level == "good":
            out.set_verdict("good", "Gaussian and fat-tailed risk estimates are close",
                            f"At {lvl}, the normal distribution is a reasonable approximation here – check higher "
                            "confidence levels in the VaR curve.")
        else:
            facts = (f"{lvl} VaR ({best_name}): {_num(100 * fat)}% vs Gaussian {_num(100 * v['gauss'])}% "
                     f"({'+' if r_var >= 1 else '−'}{_num(100 * abs(r_var - 1), 0)}%). Expected Shortfall: "
                     f"{_num(100 * v['es_hist'])}% vs {_num(100 * v['es_gauss'])}% ({'+' if r_es >= 1 else '−'}{_num(100 * abs(r_es - 1), 0)}%).")
            if r_var >= r_es:
                out.set_verdict(level, f"The Gaussian VaR understates risk by {pct}%", facts)
            else:
                out.set_verdict(level, f"The Gaussian model understates Expected Shortfall by {pct}%",
                                facts + " Beyond the VaR, losses run much deeper than the normal distribution "
                                        "expects – exactly what a VaR figure alone conceals.")
