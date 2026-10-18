"""Early-warning dashboard: the five indicators panel by panel, the status table and the framework's dashboard figure."""

import numpy as np

from src.analysis.early_warning import CompositeEarlyWarningSystem
from src.utils.data_loader import generate_vix_from_returns
from src.visualization.early_warning_dashboard import (
    EarlyWarning3DVisualization,
    EarlyWarningDashboard,
    create_indicator_summary_table,
)

from .. import charts as C
from ..data import index_axis, fmt_date
from ._ews_common import ACTION, KEYS, LEVEL, assessment_text, num

SPEC = {
    "id": "ews-dashboard",
    "title": "Early-warning dashboard: the five indicators",
    "group": "warning",
    "order": 15,
    "summary": "GEX, TailDex, VIX curve, Dark Index and Smart Money Flow individually with their warning "
               "thresholds, and how much each indicator contributes to the composite score.",
    "description": (
        "The dashboard breaks the early-warning system's composite score down into its five building blocks. "
        "**Net Gamma Exposure (GEX):** when options dealers are net short gamma (GEX < 0), they sell into "
        "falling prices and amplify moves. **TailDex (TDEX):** the price of crash protection; elevated above "
        "15, extreme above 20. **VIX curve:** when short-dated VIX trades above long-dated VIX (ratio > 1), the "
        "curve is inverted – a sign of acute fear. **Dark Index (DIX):** the share of hidden buying; weak below "
        "43%, distribution below 40%. **Smart Money Flow (SMFI):** when it falls, professionals are exiting.\n\n"
        "**Important:** GEX, TDEX, DIX and SMFI rely on options and dark-pool data that are not available "
        "here. The framework *models* them from your returns and the VIX; the paths are illustrative, not "
        "actual measurements. If the VIX is missing, it is estimated from the returns. If the dataset contains "
        "VIX3M (S&P 500), the VIX curve is based on actual data. The calculation is causal: each day uses only data up to that "
        "day.\n\n"
        "How to read it: the gauge and contribution bars show the latest day. Each indicator has a sub-score "
        "from 0 (unremarkable) to 100 (maximum); the composite is their weighted sum. GEX reaches 50 when it "
        "changes sign and TDEX at 15; DIX and SMFI rise from 0 once they cross their warning threshold (43% and "
        "−5 over 20 days respectively). The panels below show the paths with the framework's thresholds; for "
        "the SMFI they show the 20-day change on which it is scored. The 3D phase space shows the trajectory of "
        "GEX risk, TDEX risk and the composite, followed by the status table and the original figures. The "
        "**seed** controls the model noise."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "days", "type": "int", "default": 252, "min": 60, "max": 2520, "step": 21,
         "label": "Display period", "unit": "days",
         "help": "Most recent trading days shown in the panels and figures. The full history is always modelled."},
        {"name": "term_source", "type": "select", "default": "auto", "label": "VIX curve",
         "options": [{"value": "auto", "label": "Actual (VIX/VIX3M) if in the dataset"},
                     {"value": "model", "label": "Always modelled (simulated from the VIX)"}]},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Model random seed",
         "help": "GEX, TDEX, DIX and SMFI contain model noise. Same seed = same series."},
        {"name": "show_figures", "type": "bool", "default": True, "label": "Show original figures (Matplotlib)",
         "help": "Render the framework's 11-panel dashboard and 3D phase space as images."},
    ],
    "covers": [
        "src.visualization.early_warning_dashboard.EarlyWarningDashboard",
        "src.visualization.early_warning_dashboard.EarlyWarningDashboard.create_dashboard",
        "src.visualization.early_warning_dashboard.EarlyWarning3DVisualization",
        "src.visualization.early_warning_dashboard.EarlyWarning3DVisualization.create_3d_phase_space",
        "src.visualization.early_warning_dashboard.create_indicator_summary_table",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.simulate_all_indicators",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.compute_composite",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.get_risk_assessment",
        "src.utils.data_loader.generate_vix_from_returns",
    ],
    "cost": "medium",
}

SHORT = {"gex": "GEX", "tdex": "TDEX", "vix_term": "VIX curve", "dix": "DIX", "smfi": "SMFI"}
# rows of create_indicator_summary_table -> (indicator key, display name)
TABLE_ROWS = {"1. GEX (B$)": ("gex", "Net Gamma (GEX), $bn"), "2. TDEX": ("tdex", "TailDex (TDEX)"),
              "3. VIX M1/M2": ("vix_term", "VIX curve (M1/M2)"), "4. DIX (%)": ("dix", "Dark Index (DIX), %"),
              "5. SMFI": ("smfi", "Smart Money Flow (SMFI)"), "COMPOSITE SCORE": (None, "Composite score")}
SIGNAL = {"NEGATIVE": "Negative – dealers amplify moves", "POSITIVE": "Positive – dealers dampen moves",
          "EXTREME": "Extreme", "ELEVATED": "Elevated", "NORMAL": "Normal",
          "INVERTED": "Inverted (backwardation)", "CONTANGO": "Contango (normal)",
          "DISTRIBUTION": "Distribution", "WEAK": "Weak",
          "SELLING": "Smart money selling", "BUYING": "Smart money buying",
          "EXTREME RISK": "Extreme", "HIGH RISK": "High"}


def _parse_summary(text):
    """Rows of the framework's box-drawing status table: (label, value, weight, signal)."""
    rows = []
    for line in text.splitlines():
        cells = [c.strip(" ║") for c in line.split("│")]
        if len(cells) != 4 or cells[0] == "Indicator":
            continue
        try:
            value = float(cells[1])
            weight = float(cells[2].rstrip("%")) / 100
        except ValueError:
            continue
        rows.append((cells[0], value, weight, cells[3]))
    return rows


def _runs(mask, gap=2):
    """Index runs where ``mask`` holds; runs at most ``gap`` days apart are merged."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


def _sub_status(score):
    """Status of one indicator's 0-100 sub-score (quarters: normal / elevated / high / critical)."""
    return "critical" if score >= 75 else "serious" if score >= 50 else "warning" if score >= 25 else "good"


def _split_sign(values):
    v = np.asarray(values, dtype=float)
    return np.where(v >= 0, v, np.nan), np.where(v < 0, v, np.nan)


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    m = int(min(params["days"], n))
    x_all, xa = index_axis(data)
    x = x_all[n - m:]
    label = (lambda i: fmt_date(x[i])) if data.dates else (lambda i: f"Day {n - m + i}")

    vix_real = data.vix is not None
    vix = np.asarray(data.vix, dtype=float) if vix_real else np.asarray(generate_vix_from_returns(r), dtype=float)
    use_real_curve = params["term_source"] == "auto" and vix_real and data.vix3m is not None
    vix3m = np.asarray(data.vix3m, dtype=float) if use_real_curve else None
    real_prices = data.prices is not None
    prices = np.asarray(data.prices, dtype=float) if real_prices else 100.0 * np.exp(np.cumsum(r))

    out.progress("Modelling the five indicators…", 0.1)
    ews = CompositeEarlyWarningSystem()
    np.random.seed(int(params["seed"]))
    full = ews.simulate_all_indicators(r, prices, vix, causal=True, vix3m=vix3m)
    ind = {k: np.asarray(v, dtype=float)[n - m:] for k, v in full.items()}
    p = prices[n - m:]
    comp = ind["composite_score"]
    th = ews.thresholds
    last = ews.compute_composite(*(float(ind[k + "_score"][-1]) for k in KEYS))
    level = last["level"]
    status, level_txt = LEVEL[level]
    action_txt = ACTION.get(last["action"], last["action"])
    ratio = ind["vix_m1"] / ind["vix_m2"]
    ratio_name = "VIX/VIX3M" if use_real_curve else "M1/M2 (model)"
    score = {k: ind[k + "_score"] for k in KEYS}
    g, td, dx = (float(ind[k][-1]) for k in ("gex", "tdex", "dix"))
    # the SMFI level is a running sum since the first day of the data set; its sub-score rates the 20-day change
    smfi_full = np.asarray(full["smfi"], dtype=float)
    smfi_chg = np.full(n, np.nan)
    smfi_chg[21:] = smfi_full[21:] - smfi_full[1:-20]
    smfi_chg = smfi_chg[n - m:]

    # ---- headline: the last day, indicator by indicator ------------------------------------------------
    def sub(k):
        return f"sub-score {score[k][-1]:.0f} / 100"

    out.metric("Composite (latest day)", comp[-1], "num", digits=0, unit="/ 100", status=status,
               hint=f"{level_txt} · {action_txt}")
    out.metric("Net Gamma (GEX)", g, "num", digits=1, unit="$bn", status=_sub_status(score["gex"][-1]),
               hint=("negative: dealers amplify moves" if g < 0 else "positive: dealers dampen moves")
               + " · " + sub("gex"))
    out.metric("TailDex (TDEX)", td, "num", digits=1, status=_sub_status(score["tdex"][-1]),
               hint="elevated above 15, extreme above 20 · " + sub("tdex"))
    out.metric(f"VIX curve {ratio_name}", ratio[-1], "num", digits=3, status=_sub_status(score["vix_term"][-1]),
               hint=("> 1: inverted, acute fear" if ratio[-1] > 1 else "≤ 1: contango, normal") + " · "
               + sub("vix_term"))
    out.metric("Dark Index (DIX)", dx / 100, "pct", digits=1, status=_sub_status(score["dix"][-1]),
               hint="weak below 43%, distribution below 40% · " + sub("dix"))
    out.metric("Smart Money Flow (SMFI), 20-day change", smfi_chg[-1], "num", digits=1,
               status=_sub_status(score["smfi"][-1]),
               hint="warning below −5, sub-score 100 at −15 or below · " + sub("smfi"))

    i_max = int(np.argmax(comp))
    driver = max(KEYS, key=lambda k: last["components"][k]["contribution"])
    d_pts = last["components"][driver]["contribution"]
    out.metric("Highest composite in period", comp[i_max], "num", digits=0, unit="/ 100",
               status=LEVEL[ews.compute_composite(*(float(score[k][i_max]) for k in KEYS))["level"]][0],
               hint=f"on {label(i_max)}")
    out.metric("Share of days ≥ 50 (Elevated)", float(np.mean(comp >= th["elevated"])), "pct", digits=1,
               hint=f"over the last {num(m, 0)} trading days")
    out.metric("Share of days ≥ 75 (High)", float(np.mean(comp >= th["high"])), "pct", digits=1,
               hint=f"over the last {num(m, 0)} trading days")
    out.metric("Main driver today", SHORT[driver] if d_pts > 0 else "–", "text",
               hint=(f"{num(d_pts, 1)} of {num(float(comp[-1]), 1)} points" if d_pts > 0
                     else "no indicator is firing"))
    out.note("**Modelled indicators:** GEX, TailDex, DIX and SMFI require proprietary options and dark-pool "
             "data. The framework models them here from returns and the VIX, so the paths are illustrative, "
             "not actual market measurements.", "warning")
    src_notes = []
    if use_real_curve:
        src_notes.append("**Actual VIX curve:** VIX/VIX3M from the dataset.")
    elif vix_real and data.vix3m is not None:
        src_notes.append("**Modelled VIX curve** (as selected): M1/M2 simulated from the VIX.")
    else:
        src_notes.append("**Modelled VIX curve:** no VIX3M in the dataset, so M1/M2 are simulated from the VIX.")
    if not vix_real:
        src_notes.append("The VIX itself is missing and is estimated from the returns (`generate_vix_from_returns`).")
    if not real_prices:
        src_notes.append("No prices in the dataset: the price is an index built from the returns (base 100).")
    src_notes.append(f"Modelled causally over all {num(n, 0)} days (`causal=True`); the last {num(m, 0)} are shown.")
    out.note(" ".join(src_notes), "info")

    # ---- today: gauge + contributions -----------------------------------------------------------------------
    out.chart(C.gauge("Composite on the latest day", float(comp[-1]),
                      thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Elevated", "High", "Extreme"),
                      subtitle=f"{label(m - 1)} · {level_txt} · {action_txt}"))
    cats = [f"{SHORT[k]} · {ews.weights[k]:.0%}" for k in KEYS]
    out.chart(C.xy(
        "Indicator contributions on the latest day",
        [C.trace("Contribution to composite", cats, [last["components"][k]["contribution"] for k in KEYS],
                 type="bar", color=1),
         C.trace("Maximum possible (weight × 100)", cats, [100 * ews.weights[k] for k in KEYS],
                 type="scatter", color="muted", size=9, decimate_to=None)],
        x=C.axis("", type="category"), y=C.axis("Points", range=(0, 40)),
        subtitle="Contribution = sub-score × weight. The sum of all contributions is the composite score.",
    ))

    # ---- price with warning phases + composite history --------------------------------------------------
    bands = []
    for key, col in (("elevated", "warning"), ("high", "serious"), ("extreme", "critical")):
        for a, b in _runs(comp >= th[key])[:60]:
            bands.append(C.band(x[a], x[min(b + 1, m - 1)], "", col))
    out.chart(C.xy(
        "Price with warning phases",
        [C.trace("Price" if real_prices else "Index from returns (base 100)", x, p, color=1, width=1.5)],
        x=xa, y=C.axis("Price", type="log" if p.max() > 8 * p.min() else "linear"), bands=bands,
        subtitle=f"Shaded: days with composite ≥ {th['elevated']:g} (elevated), ≥ {th['high']:g} (high) and "
                 f"≥ {th['extreme']:g} (extreme) – the higher the level, the stronger the shading.",
    ))
    out.chart(C.xy(
        "Composite score",
        [C.trace("Composite", x, comp, color=1, width=1.5)],
        x=xa, y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Elevated", "warning"), C.hline(th["high"], "High", "serious"),
                C.hline(th["extreme"], "Extreme", "critical")],
        subtitle="The framework's level boundaries: 50 / 75 / 90.",
    ))

    # ---- the five panels ----------------------------------------------------------------------------------
    pos, neg = _split_sign(ind["gex"])
    out.chart(C.xy(
        f"1 · Net Gamma Exposure (GEX) – {ews.weights['gex']:.0%} weight",
        [C.trace("GEX ≥ 0: dampening", x, pos, type="bar", color=1),
         C.trace("GEX < 0: amplifying", x, neg, type="bar", color="serious")],
        x=xa, y=C.axis("$bn"),
        hlines=[C.hline(-5, "Sub-score 100", "critical")],
        subtitle="Below 0, dealers sell into falling prices. At −$5bn or below, the sub-score is 100.",
        height=300,
    ))
    out.chart(C.xy(
        f"2 · TailDex (TDEX) – {ews.weights['tdex']:.0%} weight",
        [C.trace("TDEX", x, ind["tdex"], color=2, width=1.5)],
        x=xa, y=C.axis("TDEX"),
        hlines=[C.hline(15, "Elevated", "warning"), C.hline(20, "Extreme", "critical")],
        subtitle="Price of crash protection. Above 20 the sub-score is 100.", height=300,
    ))
    out.chart(C.xy(
        f"3 · VIX curve – {ews.weights['vix_term']:.0%} weight",
        [C.trace(ratio_name, x, ratio, color=3, width=1.5)],
        x=xa, y=C.axis("Ratio short / long", digits=3),
        hlines=[C.hline(1.0, "Inversion", "serious")],
        subtitle=("Actual ratio of VIX to VIX3M." if use_real_curve else "Futures curve simulated from the VIX.")
        + " Above 1 the curve is inverted.", height=300,
    ))
    out.chart(C.xy(
        f"4 · Dark Index (DIX) – {ews.weights['dix']:.0%} weight",
        [C.trace("DIX", x, ind["dix"] / 100, color=4, width=1.5)],
        x=xa, y=C.axis("Share", fmt="pct", digits=0),
        hlines=[C.hline(0.43, "Weak", "warning"), C.hline(0.40, "Distribution", "serious")],
        subtitle="Share of hidden buying. Below these lines, institutions are distributing rather than buying.",
        height=300,
    ))
    out.chart(C.xy(
        f"5 · Smart Money Flow (SMFI) – {ews.weights['smfi']:.0%} weight",
        [C.trace("SMFI change over 20 days", x, smfi_chg, color=5, width=1.5)],
        x=xa, y=C.axis("Points over 20 days"),
        hlines=[C.hline(-5, "Warning", "warning"), C.hline(-15, "Sub-score 100", "critical")],
        subtitle="The change is scored, not the level: if the SMFI falls by more than 5 points within 20 days, "
                 "the sub-score rises; at −15 or below it is 100.", height=300,
    ))
    step = max(1, int(np.ceil(m / 260)))
    starts = np.arange(0, m, step)
    out.chart(C.xy(
        "Composite score breakdown",
        [C.trace(f"{SHORT[k]} · {ews.weights[k]:.0%}", [x[a] for a in starts],
                 [float(np.mean(score[k][a:a + step])) * ews.weights[k] for a in starts], type="bar", color=i + 1)
         for i, k in enumerate(KEYS)],
        x=xa, y=C.axis("Points", range=(0, 100)), stacked=True,
        subtitle="Contribution per indicator, stacked; the top edge is the composite score"
        + (f" (mean per {step} days)." if step > 1 else "."), height=300,
    ))

    # ---- 3D phase space (EarlyWarning3DVisualization as interactive trajectory) -------------------------
    gx, ty = score["gex"], score["tdex"]
    lv = np.array([("EXTREME" if v >= th["extreme"] else "HIGH" if v >= th["high"]
                    else "ELEVATED" if v >= th["elevated"] else "NORMAL") for v in comp])
    traces = [C.path3d("Trajectory", gx, ty, comp, color="muted", mode="lines")]
    for key in ("NORMAL", "ELEVATED", "HIGH", "EXTREME"):
        sel = lv == key
        if sel.any():
            traces.append(C.path3d(LEVEL[key][1], gx[sel], ty[sel], comp[sel], color=LEVEL[key][0], mode="markers",
                                   size=3, text=[label(i) for i in np.flatnonzero(sel)]))
    traces.append(C.path3d("Danger plane (composite 75)", [0, 100, 100, 0, 0], [0, 0, 100, 100, 0], [75] * 5,
                           color="critical", mode="lines"))
    traces.append(C.path3d("Latest day", [gx[-1]], [ty[-1]], [comp[-1]], color="ink", mode="markers", size=8,
                           text=[label(m - 1)]))
    out.chart(C.scatter3d(
        "Early-warning phase space", traces,
        x=C.axis("GEX risk (sub-score)"), y=C.axis("TDEX risk (sub-score)"), z=C.axis("Composite"),
        subtitle="Each point is one day. Points above the red plane (75) are at the High or Extreme level.",
    ))

    # ---- the framework's status table ----------------------------------------------------------------
    text = create_indicator_summary_table(ind)
    rows = []
    for name, value, weight, signal in _parse_summary(text):
        key, name_txt = TABLE_ROWS.get(name, (None, name))
        if key == "vix_term" and use_real_curve:
            name_txt = "VIX curve (VIX/VIX3M)"
        w_txt = f"{weight:.0%}"
        if key is None:
            rows.append([name_txt, num(value, 1), w_txt, SIGNAL.get(signal, signal), float(comp[-1]), float(comp[-1])])
        else:
            digits = 3 if key == "vix_term" else 2 if key == "gex" else 1
            rows.append([name_txt, num(value, digits), w_txt, SIGNAL.get(signal, signal),
                         last["components"][key]["score"], last["components"][key]["contribution"]])
    out.table(f"Status on {label(m - 1)} (framework table)",
              ["Indicator", "Value", "Weight", "Signal", "Sub-score", "Contribution"], rows,
              formats=["text", "text", "text", "text", "num", "num"],
              note="Signals from `create_indicator_summary_table`, supplemented by sub-score and contribution "
                   "(points). The original table is available below as a text file.")
    out.download("early_warning_status.txt", text.strip("\n") + "\n")

    assessment = assessment_text(ews.get_risk_assessment(float(comp[-1])))
    out.set_verdict(status, f"Latest reading: {level_txt} – {comp[-1]:.0f} out of 100",
                    f"{assessment} Modelled from returns and the VIX, not a live signal from options or "
                    "dark-pool data.")

    # ---- the framework's figures --------------------------------------------------------------------------
    if params["show_figures"]:
        out.progress("Drawing the original dashboard…", 0.6)
        dates64 = np.array(x, dtype="datetime64[D]") if data.dates else None
        title = f"Early Warning System Dashboard · {data.name or 'Dataset'} · {label(0)} – {label(m - 1)}"
        fig = EarlyWarningDashboard(figsize=(16, 13)).create_dashboard(ind, p, dates=dates64, title=title)
        out.figure(fig, "The framework's early-warning dashboard (EarlyWarningDashboard)", dpi=80)
        out.progress("3D phase space…", 0.85)
        fig = EarlyWarning3DVisualization().create_3d_phase_space(ind["gex_score"], ind["tdex_score"], comp,
                                                                   figsize=(10, 8))
        out.figure(fig, "The framework's 3D phase space (EarlyWarning3DVisualization)", dpi=90)
