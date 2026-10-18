"""The framework's all-in-one tail-risk dashboard (TailRiskDashboard) plus its text report, made interactive."""

import re

import numpy as np
from scipy import stats

from src.visualization.dashboard import TailRiskDashboard, create_summary_report

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tail-risk-dashboard",
    "title": "Integrated tail-risk dashboard",
    "group": "dashboard",
    "order": 10,
    "summary": "Everything at a glance: VaR, Expected Shortfall, tail index, volatility regime and the framework's "
               "composite risk index – with a text report and the original dashboard on demand.",
    "description": (
        "`TailRiskDashboard` computes, in a single pass, everything the framework knows about the tails of your "
        "returns: rolling VaR and Expected Shortfall (99%, 60 days), skewness, kurtosis and the Hill tail index, "
        "the three physics coordinate systems (Lévy, Tsallis, phase transition) and the current volatility regime "
        "(simple volatility percentiles and a three-state Markov model). The key series are shown interactively; "
        "the original figure (`create_full_dashboard`) with all 14 panels can be switched on.\n\n"
        "The **composite risk index** is the dashboard's own formula: (λ + 0.5·q + χ) / 2.5 from the jump "
        "intensity λ = |r|/σ (Lévy), Tsallis q and the susceptibility χ, then scaled to 0–100 over the history "
        "shown. Above 75 means EXTREME, above 50 ELEVATED, above 25 MODERATE. Because λ measures today's daily "
        "move, the index jumps from day to day – so read the 20-day average as well.\n\n"
        "The **text report** (`create_summary_report`) summarises moments, tail index, VaR/ES and the 3σ "
        "comparison with the normal distribution; you can download it in its original form. **Period** limits the "
        "calculation to the most recent trading days (the Markov model is the most expensive part).\n\n"
        "With 60 days, only one value lies beyond the 99% VaR: the rolling Expected Shortfall is then exactly the "
        "largest one-day loss in the window – a crude but honest measure."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Period", "unit": "days",
         "help": "The most recent trading days of the dataset. 2,520 days ≈ 10 years. Longer periods take considerably more computing time."},
        {"name": "figure", "type": "bool", "default": False, "label": "Draw the original dashboard (Matplotlib)",
         "help": "The 14-panel figure from create_full_dashboard – adds about 6 seconds in the browser."},
    ],
    "covers": [
        "src.visualization.dashboard.TailRiskDashboard",
        "src.visualization.dashboard.TailRiskDashboard._compute_metrics",
        "src.visualization.dashboard.TailRiskDashboard._generate_synthetic_vix",
        "src.visualization.dashboard.TailRiskDashboard.create_full_dashboard",
        "src.visualization.dashboard.create_summary_report",
    ],
    "cost": "medium",
}

CLASSES = [(75, "EXTREME", "critical"), (50, "ELEVATED", "serious"), (25, "MODERATE", "warning"), (-1, "LOW", "good")]
REGIMES = {"LOW_VOL": "Low volatility", "MEDIUM_VOL": "Medium volatility", "HIGH_VOL": "High volatility",
           "UNKNOWN": "unknown", "UNCONDITIONAL": "Overall (unconditional)"}
# MEDIUM_VOL is the middle half of the volatility history - normal, so it gets no warning colour
REGIME_STATUS = {"LOW_VOL": "good", "MEDIUM_VOL": None, "HIGH_VOL": "serious"}

# create_summary_report: the report's labels -> the site's wording (sentence case, explicit thresholds)
REPORT_SECTIONS = {"BASIC STATISTICS": "Basic statistics", "TAIL RISK METRICS": "Tail risk",
                   "TAIL PROBABILITY COMPARISON (3σ)": "3σ comparison (both tails)",
                   "INTERPRETATION": "Report assessment"}
REPORT_LABELS = {"Mean Return": "Mean daily return", "Standard Deviation": "Standard deviation (daily)",
                 "Skewness": "Skewness", "Excess Kurtosis": "Excess kurtosis", "Tail Index (α)": "Tail index α (Hill)",
                 "VaR (99%)": "99% VaR (historical)", "ES (99%)": "99% Expected Shortfall",
                 "ES/VaR Ratio": "ES / VaR ratio", "Gaussian Probability": "Normal distribution: P(|r| > 3σ)",
                 "Empirical Probability": "Observed: P(|r| > 3σ)", "Fat Tail Ratio": "Factor observed / Gaussian"}
REPORT_REMARKS = {"(Fat tails!)": "fat\u00a0tails", "(Normal)": "within\u00a0normal\u00a0range",
                  "(High tail risk)": "high\u00a0tail\u00a0risk"}
REPORT_LINES = {
    "Distribution has significant fat tails": "The distribution has clearly fat tails (excess kurtosis > 3)",
    "Distribution close to normal": "Distribution close to normal (excess kurtosis ≤ 3)",
    "Negative skewness indicates crash risk": "Negative skewness (< −0.5): crashes more frequent than rallies",
    "Symmetric distribution": "Largely symmetric (skewness ≥ −0.5)",
    "Tail events are significantly more likely than Gaussian predicts":
        "3σ days more than twice as frequent as the normal distribution predicts",
    "Tail probabilities close to Gaussian": "3σ days about as frequent as under a normal distribution",
}


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _report_rows(report):
    """Turn the box-drawn text of create_summary_report into table rows with the site's wording."""
    rows, section = [], ""
    for line in report.splitlines():
        if not line.startswith("║"):
            continue
        body = line.strip("║").strip()
        if not body or set(body) <= set("─ "):
            continue
        if body in REPORT_SECTIONS:
            section = REPORT_SECTIONS[body]
            continue
        if body.startswith(("⚠", "✓")):
            text = body.lstrip("⚠️✓ ").strip()
            rows.append([section, REPORT_LINES.get(text, text), "", "warning" if body.startswith("⚠") else "ok"])
            continue
        if ":" not in body:
            continue
        label, value = (s.strip() for s in body.split(":", 1))
        m = re.match(r"(-?[\d.]+)\s*(%|x)?\s*(\(.*\))?", value)
        if not m:
            continue
        num, unit, remark = m.groups()
        digits = len(num.split(".")[1]) if "." in num else 0
        shown = _num(float(num), digits) + ("%" if unit == "%" else "×" if unit == "x" else "")
        rows.append([section, REPORT_LABELS.get(label, label), shown, REPORT_REMARKS.get(remark or "", "")])
    return rows


def _last(a, default=np.nan):
    a = np.asarray(a, dtype=float)
    return float(a[-1]) if len(a) and np.isfinite(a[-1]) else default


def run(params, data, out):
    n_all = data.n
    k = int(min(params["last_n"], n_all))
    if k < 150:
        raise ToolError(f"The dashboard needs at least 150 daily returns; the dataset has {n_all}.")
    r = np.asarray(data.returns[-k:], dtype=float)
    vix = np.asarray(data.vix[-k:], dtype=float) if data.vix is not None else None
    dates = data.dates[-k:] if data.dates else None
    x = list(dates) if dates else list(range(n_all - k + 1, n_all + 1))
    xa = C.axis("", type="date") if dates else C.axis("Trading day", fmt="int")

    out.progress("Physics coordinates, rolling metrics and regimes …", 0.1)
    dash = TailRiskDashboard(r, vix=vix)      # __init__ runs _compute_metrics()
    report = create_summary_report(r)
    rm = dash.rolling_metrics

    # composite exactly as in create_full_dashboard (panels 1.4 and 4.2)
    risk_level = (dash.levy_coords[2] + 0.5 * dash.tsallis_coords[0] + dash.phase_coords[0]) / 2.5
    composite = (risk_level - np.nanmin(risk_level)) / (np.nanmax(risk_level) - np.nanmin(risk_level) + 1e-10) * 100
    comp_now = float(composite[-1])
    comp_pct = float(np.mean(composite <= comp_now))      # the 0-100 scale is min-max, not a percentile
    comp_avg = np.convolve(composite, np.ones(20) / 20, mode="full")[:k]
    comp_avg[:19] = np.nan
    cls, cls_status = next((name, st) for lim, name, st in CLASSES if comp_now > lim)

    var99, es99 = _last(rm["var_99"]), _last(rm["es_99"])
    alpha_now, kurt_now = _last(rm["tail_index"]), _last(rm["kurtosis"])
    reg = dash.current_regime
    simple = reg.get("simple_regime", "UNKNOWN")
    markov = reg.get("markov_regime")
    probs = reg.get("regime_probabilities", {})

    # --- metrics ------------------------------------------------------------------------
    out.metric("Risk level (composite)", cls, "text", status=cls_status,
               hint=f"index today {_num(comp_now, 0)} of 100 (percentile rank {_num(100 * comp_pct, 0)}%) · "
                    f"20-day average {_num(float(comp_avg[-1]), 0)}")
    out.metric("99% VaR (60 days)", var99, "pct", digits=2, hint="historical, one-day loss",
               status="serious" if var99 > 0.03 else "warning" if var99 > 0.02 else None)
    out.metric("99% Expected Shortfall", es99, "pct", digits=2,
               hint=(f"avg. loss beyond VaR (with 60 days: the largest one-day loss) · "
                     f"ES/VaR = {_num(es99 / var99)}") if var99 > 0 else "")
    out.metric("Tail index α (60 days)", alpha_now, "num", digits=2, hint="Hill · < 3: fat tails",
               status="warning" if alpha_now < 3 else None, badge="Fat tails" if alpha_now < 3 else "")
    out.metric("Excess kurtosis (60 days)", kurt_now, "num", digits=1, hint="normal distribution = 0")
    out.metric("Volatility regime", REGIMES.get(simple, simple), "text", status=REGIME_STATUS.get(simple),
               hint="simple: 20-day volatility · low below the 25th, high above the 75th percentile")
    if markov:
        key = {"LOW_VOL": "low_vol", "MEDIUM_VOL": "medium_vol", "HIGH_VOL": "high_vol"}.get(markov)
        p = probs.get(key)
        out.metric("Markov regime", REGIMES.get(markov, markov), "text", status=REGIME_STATUS.get(markov),
                   hint=f"probability {_num(100 * float(p), 0)}% (three-state Markov model)" if p is not None else "")
    synth = dash._generate_synthetic_vix() if vix is not None else np.asarray(dash.vix, dtype=float)
    out.metric("Synthetic VIX", _last(synth), "num", digits=1,
               hint="20-day volatility p.a. × 100" + (f" · actual VIX today {_num(float(vix[-1]), 1)}"
                                                       if vix is not None else ""))

    # --- interactive key charts ---------------------------------------------------------
    out.chart(C.xy(
        "VaR and Expected Shortfall (99%, rolling 60 days)",
        [C.envelope("Gap ES – VaR", x, rm["var_99"], rm["es_99"], color=2),
         C.trace("99% VaR", x, rm["var_99"], color=1, width=1.5),
         C.trace("99% Expected Shortfall", x, rm["es_99"], color=2, width=1.5)],
        x=xa, y=C.axis("One-day loss", fmt="pct", digits=1),
        subtitle="ES is the average of the losses beyond VaR – the wider the gap, the fatter the tail. With 60 days "
                 "it is exactly the largest one-day loss in the window. The value on a given date comes from the "
                 "60 days before it.",
    ))
    tail = np.asarray(rm["tail_index"], dtype=float)
    out.chart(C.xy(
        "Tail index α (Hill, rolling 60 days)",
        [C.trace("α", x, tail, color=1, width=1.5)],
        x=xa, y=C.axis("α", range=(0, 6)),
        hlines=[C.hline(2.0, "Lévy limit α = 2"), C.hline(3.0, "Cubic law α = 3", color="warning")],
        subtitle="Below 3, the tails are in the range typical of equity crashes; below 2, the variance would be "
                 "infinite.",
    ))
    out.chart(C.xy(
        "Composite risk index of the dashboard (0–100)",
        [C.trace("Composite (daily value)", x, composite, color=1, width=1),
         C.trace("20-day average", x, comp_avg, color=2, width=2)],
        x=xa, y=C.axis("Index", range=(0, 100)),
        hlines=[C.hline(25, "Moderate"), C.hline(50, "Elevated", color="serious"),
                C.hline(75, "Extreme", color="critical")],
        subtitle="(λ + 0.5·q + χ) / 2.5, scaled to the history shown. The daily value mainly tracks today's |r|/σ.",
    ))
    sd = float(np.std(r))
    sig = [2, 3, 4, 5]
    gauss = [2 * stats.norm.sf(s) for s in sig]                      # as in panel 4.4: both tails
    emp = [float(np.mean(np.abs(r) > s * sd)) for s in sig]
    floor = min(g for g in gauss) / 3
    out.chart(C.xy(
        "Extreme days: normal distribution vs observed (both tails)",
        [C.trace("Normal distribution", [f"{s}σ" for s in sig], gauss, type="bar", color="muted"),
         C.trace("Observed", [f"{s}σ" for s in sig], [e if e > 0 else None for e in emp], type="bar", color=1)],
        x=C.axis("Threshold", type="category"), y=C.axis("Frequency", type="log", fmt="pct", range=(floor, 0.2)),
        subtitle="Log scale: each tick is a factor of 10. A missing bar means the event never occurred.",
    ))

    # --- tables ---------------------------------------------------------------------
    rows = _report_rows(report)
    if rows:
        out.table("Tail-risk report (create_summary_report)", ["Section", "Metric", "Value", "Assessment"], rows,
                  formats=["text", "text", "text", "text"],
                  note=f"Across all {k:,} days shown. Thresholds of the original report: excess kurtosis > 3, "
                       "skewness < −0.5, factor > 2, α < 3, ES/VaR > 1.5.")
    rv = dash.regime_var or {}
    if rv:
        out.table("99% VaR by volatility regime", ["Regime", "99% VaR", "Observations"],
                  [[REGIMES.get(key, key), v.get("var"), v.get("n_observations")] for key, v in rv.items()],
                  formats=["text", "pct", "int"],
                  note="Historical VaR, using only the days of the respective regime (RegimeAwareTailRisk).")
    if probs:
        out.table("Markov model: regime probabilities today", ["Regime", "Probability"],
                  [[REGIMES.get(key.upper(), key), float(v)] for key, v in probs.items()], formats=["text", "pct"])
    elif simple == "UNKNOWN":
        out.note("The regime analysis failed for this dataset; the regime fields remain empty.", "warning")
    out.download("tail_risk_report.txt", report.strip("\n") + "\n", "text/plain;charset=utf-8")

    # --- verdict: the dashboard's own risk class ----------------------------------------
    texts = {
        "EXTREME": "The composite index is in the top quarter of its historical range – today's move, Tsallis q "
                   "and the susceptibility are all high at the same time.",
        "ELEVATED": "The composite index is above 50: several physics coordinates show elevated stress.",
        "MODERATE": "The composite index is between 25 and 50 – slightly elevated, but without an extreme signal.",
        "LOW": "The composite index is in the bottom quarter of its historical range.",
    }
    tail_txt = (f" The tail index of {_num(alpha_now)} indicates fat tails." if alpha_now < 3
                else f" The tail index of {_num(alpha_now)} is within the normal range.")
    out.set_verdict(cls_status, f"Risk level {cls} (composite {_num(comp_now, 0)})", texts[cls] + tail_txt)

    if params["figure"]:
        out.progress("Drawing the original dashboard …", 0.7)
        fig = dash.create_full_dashboard()      # no save_path: nothing is written
        out.figure(fig, "Original dashboard: create_full_dashboard()", dpi=56)
