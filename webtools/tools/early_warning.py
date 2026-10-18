"""Early-warning system: the five-indicator composite score over time (causal, real VIX/VIX3M curve when available)."""

import numpy as np

from src.analysis.early_warning import (
    CompositeEarlyWarningSystem,
    DarkIndex,
    SmartMoneyFlowIndex,
    TailDex,
    VIXTermStructure,
    analyze_crash_risk,
    compute_early_warning_coordinates,
)
from src.utils.data_loader import generate_vix_from_returns
from src.utils.numerical import rolling_window

from .. import charts as C
from ..data import index_axis, fmt_date
from ._ews_common import (
    ACTION, DIVERGENCE, DIX_LEVEL, GEX_DESCRIPTION, GEX_REGIME, KEYS, LEVEL, NAMES, STRUCTURE, TDEX_LEVEL,
    WARNING_LEVEL, bn, gex_snapshot, num,
)

SPEC = {
    "id": "early-warning",
    "title": "Early-warning composite score",
    "group": "warning",
    "order": 10,
    "summary": "The composite score of GEX, TailDex, VIX curve, DIX and SMFI over time, with warning levels, a "
               "gauge for the latest day and a review of what followed each warning.",
    "description": (
        "The framework's early-warning system condenses five indicators into a score from 0 to 100: "
        "**Net Gamma Exposure** (35%), **TailDex** (25%), **VIX curve** (20%), **Dark Index** (10%) "
        "and **Smart Money Flow** (10%). From 50 the situation counts as elevated, from 75 as high and from 90 "
        "as extreme.\n\n"
        "**Important:** GEX, TailDex, DIX and SMFI rely on proprietary options and dark-pool data that are not "
        "available here. The framework **models** them from your returns and the VIX (`simulate_*`), so the "
        "paths are illustrative, not actual measurements. The VIX curve is based on actual data when the dataset contains VIX "
        "and VIX3M (S&P 500): the actual VIX/VIX3M ratio is then used (> 1 = inverted = stress); otherwise the "
        "framework simulates a futures curve from the VIX. The calculation is causal: each daily value uses "
        "only data up to that day, with no look-ahead.\n\n"
        "How to read the results: the gauge shows the latest day, the time series shows the composite with the "
        "level boundaries, and the heatmap shows which indicator fired when. ‘Outcomes by warning level’ tests "
        "whether high levels actually preceded losses. The **seed** controls the model noise, the **divergence "
        "window** sets the days used for the divergence checks and the **evaluation horizon** sets how far "
        "ahead the outcomes are measured."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "term_source", "type": "select", "default": "auto", "label": "VIX curve",
         "options": [{"value": "auto", "label": "Actual (VIX/VIX3M) if in the dataset"},
                     {"value": "model", "label": "Always modelled (simulated from the VIX)"}],
         "help": "For comparison: how does the score change when the model replaces the actual curve?"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Model random seed",
         "help": "GEX, TDEX, DIX and SMFI contain model noise. Same seed = same series."},
        {"name": "window", "type": "int", "default": 20, "min": 10, "max": 120, "step": 5,
         "label": "Divergence window", "unit": "days",
         "help": "Most recent days used for the divergence checks (price vs TDEX, DIX, SMFI) and the curve inversion."},
        {"name": "horizon", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Evaluation horizon", "unit": "days",
         "help": "Number of trading days after a warning signal over which the price change is evaluated."},
        {"name": "show_3d", "type": "bool", "default": True, "label": "Show 3D phase space",
         "help": "GEX risk × TDEX risk × composite (runs the modelling a second time)."},
    ],
    "covers": [
        "src.analysis.early_warning.analyze_crash_risk",
        "src.analysis.early_warning.compute_early_warning_coordinates",
        "src.analysis.early_warning.CompositeEarlyWarningSystem",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.simulate_all_indicators",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.compute_composite",
        "src.analysis.early_warning.NetGammaExposure.simulate_gex",
        "src.analysis.early_warning.NetGammaExposure.get_regime_description",
        "src.analysis.early_warning.TailDex.simulate_tdex",
        "src.analysis.early_warning.TailDex.detect_divergence",
        "src.analysis.early_warning.VIXTermStructure.simulate_term_structure",
        "src.analysis.early_warning.VIXTermStructure.calculate_term_structure",
        "src.analysis.early_warning.VIXTermStructure.detect_inversion_event",
        "src.analysis.early_warning.DarkIndex.simulate_dix",
        "src.analysis.early_warning.DarkIndex.detect_divergence",
        "src.analysis.early_warning.SmartMoneyFlowIndex.simulate_smfi",
        "src.analysis.early_warning.SmartMoneyFlowIndex.detect_divergence",
        "src.utils.data_loader.generate_vix_from_returns",
        "src.utils.numerical.rolling_window",
    ],
    "cost": "medium",
}

WEIGHT_LABEL = {"gex": "GEX · 35%", "tdex": "TDEX · 25%", "vix_term": "VIX curve · 20%",
                "dix": "DIX · 10%", "smfi": "SMFI · 10%"}
SHORT = {"gex": "GEX", "tdex": "TDEX", "vix_term": "VIX curve", "dix": "DIX", "smfi": "SMFI"}


def _level_of(score, th):
    if score >= th["extreme"]:
        return "EXTREME"
    if score >= th["high"]:
        return "HIGH"
    if score >= th["elevated"]:
        return "ELEVATED"
    return "NORMAL"


def _episodes(mask, gap=5):
    """Runs of True in ``mask`` as (start, end) index pairs; runs at most ``gap`` days apart are merged."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            runs.append((int(start), int(prev)))
            start = i
        prev = i
    runs.append((int(start), int(prev)))
    return runs


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    x, xa = index_axis(data)
    label = (lambda i: fmt_date(x[i])) if data.dates else (lambda i: f"Day {i}")
    win = int(min(params["window"], n))
    h = int(min(params["horizon"], n // 3))

    vix_real = data.vix is not None
    vix = np.asarray(data.vix, dtype=float) if vix_real else np.asarray(generate_vix_from_returns(r), dtype=float)
    use_real_curve = params["term_source"] == "auto" and vix_real and data.vix3m is not None
    vix3m = np.asarray(data.vix3m, dtype=float) if use_real_curve else None
    prices = np.asarray(data.prices, dtype=float) if data.prices is not None else 100 * np.exp(np.cumsum(r))

    out.progress("Modelling the five indicators…", 0.1)
    np.random.seed(int(params["seed"]))
    res = analyze_crash_risk(r, vix, verbose=False, causal=True, vix3m=vix3m)
    ts = res["time_series"]
    comp = np.asarray(ts["composite_score"], dtype=float)
    scores = {k: np.asarray(ts[k + "_score"], dtype=float) for k in KEYS}

    ews = CompositeEarlyWarningSystem()
    th = ews.thresholds
    last = ews.compute_composite(*(float(scores[k][-1]) for k in KEYS))
    level = last["level"]
    status, level_txt = LEVEL[level]
    action_txt = ACTION.get(last["action"], last["action"])
    ratio = np.asarray(ts["vix_m1"], dtype=float) / np.asarray(ts["vix_m2"], dtype=float)
    ratio_name = "VIX/VIX3M" if use_real_curve else "M1/M2 (model)"

    # ---- headline ------------------------------------------------------------------------------------
    i_max = int(np.argmax(comp))
    out.metric("Composite (latest day)", comp[-1], "num", digits=0, unit="/ 100", status=status,
               hint=f"{level_txt} · {action_txt}")
    out.metric(ratio_name, ratio[-1], "num", digits=3,
               hint="> 1 = inverted (backwardation, stress)" if use_real_curve
               else "simulated futures curve; > 1 = inverted",
               status="serious" if ratio[-1] > 1 else None)
    out.metric("Highest composite", comp[i_max], "num", digits=0, unit="/ 100", status=LEVEL[_level_of(comp[i_max], th)][0],
               hint=f"on {label(i_max)}")
    out.metric("Share of days ≥ 50", float(np.mean(comp >= th["elevated"])), "pct", digits=1,
               hint="Elevated or higher")
    out.metric("Share of days ≥ 75", float(np.mean(comp >= th["high"])), "pct", digits=1,
               hint="High or Extreme")

    out.note("**Modelled indicators:** GEX, TailDex, DIX and SMFI require proprietary options and dark-pool "
             "data. The framework models them here from returns and the VIX, so the paths are illustrative, "
             "not actual market measurements.", "warning")
    if use_real_curve:
        curve = "**Actual VIX curve:** VIX/VIX3M ratio from the dataset (30-day vs 3-month implied volatility)."
    elif vix_real and data.vix3m is not None:
        curve = "**Modelled VIX curve** (as selected): M1/M2 are simulated from the VIX; VIX3M is not used."
    else:
        curve = "**Modelled VIX curve:** the dataset has no VIX3M, so M1/M2 are simulated from the VIX."
    if not vix_real:
        curve += " The VIX is missing as well and is derived from the returns (`generate_vix_from_returns`)."
    elif data.meta.get("source") in ("synthetic", "crisis"):
        curve += " The VIX of this synthetic dataset is itself derived from the returns."
    out.note(curve + " All values are causal: each day uses only data up to that day (`causal=True`).", "info")

    # ---- gauge + contributions --------------------------------------------------------------------
    out.chart(C.gauge("Composite on the latest day", float(comp[-1]), thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Elevated", "High", "Extreme"),
                      subtitle=f"{label(n - 1)} · {level_txt} · {action_txt}"))
    cats = [SHORT[k] for k in KEYS]
    out.chart(C.xy(
        "Composition of the score",
        [C.trace("Indicator score (0–100)", cats, [last["components"][k]["score"] for k in KEYS],
                 type="bar", color=1),
         C.trace("Contribution to composite (points)", cats, [last["components"][k]["contribution"] for k in KEYS],
                 type="bar", color=2)],
        x=C.axis("", type="category"), y=C.axis("Points", range=(0, 100)),
        subtitle="Contribution = score × weight (GEX 35%, TDEX 25%, VIX curve 20%, DIX and SMFI 10% each).",
    ))

    # ---- composite + price with warning phases ----------------------------------------------------
    out.chart(C.xy(
        "Composite score over time",
        [C.trace("Composite", x, comp, color=1, width=1.2)],
        x=xa, y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Elevated", "warning"), C.hline(th["high"], "High", "serious"),
                C.hline(th["extreme"], "Extreme", "critical")],
        subtitle="Dashed lines = the framework's level boundaries (50 / 75 / 90).",
    ))
    episodes = _episodes(comp >= th["high"])

    def _segment(mask):
        """Prices on the flagged days (plus the following day, so single days still draw a segment)."""
        m = np.asarray(mask, dtype=bool)
        m = m | np.concatenate([[False], m[:-1]])
        return np.where(m, prices, np.nan)

    flagged = [(f"Composite ≥ {th['elevated']} (Elevated)", comp >= th["elevated"], "warning", 2.0),
               (f"Composite ≥ {th['high']} (High, Extreme)", comp >= th["high"], "serious", 2.4)]
    out.chart(C.xy(
        "Price with warning phases",
        [C.trace("Price" if data.prices is not None else "Index from returns (base 100)", x, prices, color="muted",
                 width=1.2)]
        + [C.trace(name, x, _segment(m), color=col, width=wd) for name, m, col, wd in flagged if m.any()],
        x=xa, y=C.axis("Price", type="log" if prices.max() > 8 * prices.min() else "linear"),
        subtitle="Coloured: days at a warning level. Did the warnings come before the sell-offs, or only during them?",
    ))

    # ---- which indicator fired when ---------------------------------------------------------------
    step = max(1, int(np.ceil(n / 80)))
    starts = np.arange(0, n, step)
    rows = [WEIGHT_LABEL[k] for k in KEYS] + ["Composite"]
    z = [[float(np.mean(s[a:a + step])) for a in starts] for s in [scores[k] for k in KEYS] + [comp]]
    out.chart(C.heatmap(
        "When each indicator fired", [x[a] for a in starts], rows, z,
        x=xa if data.dates else C.axis("Trading day", fmt="int"), y=C.axis("", type="category"),
        z=C.axis("Score", range=(0, 100)), scale="risk",
        subtitle=f"Mean score per {step} trading days (0 = calm, 100 = maximum warning).", height=400,
    ))

    if params["show_3d"]:
        out.progress("Phase space…", 0.7)
        np.random.seed(int(params["seed"]))  # same seed -> identical modelled series
        X, Y, Z = compute_early_warning_coordinates(r, vix, causal=True, vix3m=vix3m)
        sub = slice(None, None, max(1, int(np.ceil(n / 3000))))
        Xs, Ys, Zs = X[sub] * 100, Y[sub] * 100, Z[sub] * 100
        lv = np.array([_level_of(v, th) for v in Zs])
        traces = []
        for key in ("NORMAL", "ELEVATED", "HIGH", "EXTREME"):
            m = lv == key
            if m.any():
                traces.append(C.path3d(LEVEL[key][1], Xs[m], Ys[m], Zs[m], color=LEVEL[key][0], mode="markers",
                                       size=2.5))
        traces.append(C.path3d("Latest day", [X[-1] * 100], [Y[-1] * 100], [Z[-1] * 100], color="ink",
                               mode="markers", size=8))
        out.chart(C.scatter3d(
            "Early-warning phase space", traces,
            x=C.axis("GEX risk (score)"), y=C.axis("TDEX risk (score)"), z=C.axis("Composite"),
            subtitle="Each point is one day. Calm phases sit bottom left, stress top right.",
        ))

    # ---- latest day table ---------------------------------------------------------------------------
    g, td, dx, sm = (float(ts[k][-1]) for k in ("gex", "tdex", "dix", "smfi"))
    gres, gobj = gex_snapshot(g, float(prices[-1]), 0.5 * float(prices[-1]))  # regime only (no flip in the series)
    gdesc = gobj.get_regime_description(gres["regime"])
    tres = TailDex().calculate_tdex(np.array([td / 100]), np.array([-0.10]))
    vts = VIXTermStructure()
    m1, m2 = np.asarray(ts["vix_m1"], dtype=float), np.asarray(ts["vix_m2"], dtype=float)
    for i in range(n - win, n):
        cur = vts.calculate_term_structure(float(vix[i]), float(m1[i]), float(m2[i]))
    inv = vts.detect_inversion_event(window=win)
    dres = DarkIndex().calculate_dix(dx, 100.0)
    smfi = np.asarray(ts["smfi"], dtype=float)
    sm_trend = float(smfi[-1] - smfi[-21]) if n > 21 else 0.0
    base = "Modelled from returns"
    table_rows = [
        [NAMES["gex"], bn(g, 2, sign=True), f"{GEX_REGIME[gres['regime']]}: "
         f"{GEX_DESCRIPTION.get(gdesc, gdesc)[:1].lower()}{GEX_DESCRIPTION.get(gdesc, gdesc)[1:]}", base],
        [NAMES["tdex"], num(td, 1), TDEX_LEVEL[tres["level"]], "Modelled from returns and VIX"],
        [NAMES["vix_term"], f"{ratio_name} {num(ratio[-1], 3)}", STRUCTURE[cur["structure"]],
         "Actual: VIX/VIX3M" if use_real_curve else "Modelled from VIX"],
        [NAMES["dix"], f"{num(dx, 1)}%", DIX_LEVEL[dres["level"]], base],
        [NAMES["smfi"], num(sm, 1, sign=True), f"20-day trend {num(sm_trend, 1, sign=True)}", base],
    ]
    for row, k in zip(table_rows, KEYS):
        c = last["components"][k]
        row[3:3] = [c["score"], c["weight"], c["contribution"]]
    table_rows.append(["Composite", "", f"{level_txt} · {action_txt}", last["composite_score"], 1.0,
                       last["composite_score"], ""])
    out.table(f"Latest day ({label(n - 1)}): the five indicators",
              ["Indicator", "Value", "Classification", "Score", "Weight", "Contribution", "Source"], table_rows,
              formats=["text", "text", "text", "num", "pct:0", "num", "text"],
              note="Classification follows the framework's snapshot rules (`calculate_*`); score and contribution "
                   "(points 0–100) follow the time-series rules of `simulate_all_indicators`, so classification "
                   "and score can diverge.")

    # ---- divergences (last `win` days) ------------------------------------------------------------
    div = [("Price vs TailDex", TailDex().detect_divergence(prices, ts["tdex"], window=win),
            "Price and TDEX rising together: smart money buying protection into strength"),
           ("Price vs Dark Index", DarkIndex().detect_divergence(prices, ts["dix"], window=win),
            "Price rising, DIX falling: distribution · the reverse: accumulation"),
           ("Price vs Smart Money Flow", SmartMoneyFlowIndex().detect_divergence(prices, ts["smfi"], window=win),
            "Price at a high, SMFI falling: professionals selling into the rally")]
    drows = [[name, "Yes" if d.get("divergence") else "No", DIVERGENCE.get(d.get("type", "NONE"), "None"),
              WARNING_LEVEL.get(d.get("warning_level", "NEUTRAL"), "–"), meaning] for name, d, meaning in div]
    drows.append([f"VIX curve inverts ({ratio_name} rises above 1)", "Yes" if inv.get("inversion_event") else "No",
                  STRUCTURE[cur["structure"]], "High" if inv.get("inversion_event") else "Low",
                  f"{ratio_name} latest {num(inv.get('current_ratio', ratio[-1]), 3)}, "
                  f"range {num(inv.get('min_ratio', ratio[-1]), 3)}–{num(inv.get('max_ratio', ratio[-1]), 3)}"])
    fired = [name for name, d, _ in div if d.get("divergence")] + (["VIX curve"] if inv.get("inversion_event") else [])
    out.set_verdict(
        status, f"Latest reading: {level_txt} – {comp[-1]:.0f} out of 100",
        f"Framework recommendation: {action_txt}. Divergence checks over the last {win} days: "
        + (f"{len(fired)} of 4 triggered ({', '.join(fired)})." if fired else "none triggered.")
        + " Modelled from returns and the VIX – not a live signal from options or dark-pool data.")
    out.table(f"Divergence checks over the last {win} days", ["Check", "Signal", "Type", "Warning level", "Meaning"],
              drows, formats=["text", "text", "text", "text", "text"],
              note="TDEX, DIX and SMFI are modelled, so a divergence here illustrates the framework's logic, "
                   "not actual positioning.")

    # ---- honest look forward: what followed each level? ---------------------------------------------
    c = np.concatenate([[0.0], np.cumsum(r)])
    fwd = c[1 + h:] - c[1:n + 1 - h]                                # log return over the next h days
    worst = rolling_window(c[2:], h).min(axis=1) - c[1:n + 1 - h]   # deepest point within the next h days
    lv_all = np.array([_level_of(v, th) for v in comp[:n - h]])
    erows = []
    for key in ("NORMAL", "ELEVATED", "HIGH", "EXTREME"):
        m = lv_all == key
        if m.any():
            erows.append([LEVEL[key][1], int(m.sum()), float(m.mean()), float(np.expm1(np.mean(fwd[m]))),
                          float(np.mean(worst[m] <= np.log(0.95))), float(np.expm1(worst[m].min()))])
    erows.append(["All days", int(n - h), 1.0, float(np.expm1(np.mean(fwd))), float(np.mean(worst <= np.log(0.95))),
                  float(np.expm1(worst.min()))])
    out.table(f"Outcomes by warning level (next {h} trading days)",
              ["Level on day t", "Days", "Share", f"Mean return, {h}d", "Share with decline ≥ 5%", "Largest decline"],
              erows, formats=["text", "int", "pct", "pct", "pct", "pct"],
              note="Measured from the following day. If the high levels do not show clearly worse outcomes than "
                   "‘All days’, the modelled score had no lead here: it reacts rather than warns.")

    if episodes:
        top = sorted(episodes, key=lambda e: comp[e[0]:e[1] + 1].max(), reverse=True)[:8]
        prow = []
        for a, b in sorted(top):
            peak = a + int(np.argmax(comp[a:b + 1]))
            prow.append([label(a), label(b), b - a + 1, float(comp[peak]), float(prices[b] / prices[max(a - 1, 0)] - 1),
                         float(prices[b + h] / prices[b] - 1) if b + h < n else None])
        out.table("Strongest warning phases (composite ≥ 75)",
                  ["Start", "End", "Days", "Peak", "Price change in phase", f"Price change {h}d after"], prow,
                  formats=["text", "text", "int", "num", "pct", "pct"],
                  note="Phases less than 6 days apart are merged. ‘Price change in phase’ runs from the previous "
                       "day's close to the last day of the phase; blank = fewer than "
                       f"{h} trading days have passed since.")
