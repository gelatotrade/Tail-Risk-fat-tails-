"""Early-warning calculator: today's readings of the five indicators -> scores, composite, level and nearby thresholds."""

import numpy as np

from src.analysis.early_warning import (
    CompositeEarlyWarningSystem,
    DarkIndex,
    SmartMoneyFlowIndex,
    TailDex,
    VIXTermStructure,
)

from .. import charts as C
from ..core import ToolError
from ._ews_common import (
    ACTION, DIX_LEVEL, DIX_SIGNAL, GEX_DESCRIPTION, GEX_REGIME, GEX_SIGNAL, KEYS, LEVEL, NAMES, SMFI_SIGNAL,
    STRUCTURE, STRUCTURE_SIGNAL, TDEX_LEVEL, TDEX_SIGNAL, assessment_text, bn, gex_snapshot, num, ordinal,
    status_of,
)

SPEC = {
    "id": "early-warning-calculator",
    "title": "Early-warning calculator",
    "group": "warning",
    "order": 20,
    "summary": "Enter today's readings for GEX, TailDex, VIX curve, DIX and SMFI to get the warning scores, the "
               "composite, the level with a recommended action and the distance to the next threshold.",
    "description": (
        "Enter the current readings of the five early-warning indicators here; the defaults are the values from "
        "the README (January 2026). The framework converts them with its snapshot methods (`calculate_gex`, "
        "`calculate_tdex`, `calculate_term_structure`, `calculate_dix`, `calculate_smfi`) into warning scores "
        "from 0 to 100, weights them 35 / 25 / 20 / 10 / 10% into the composite and maps it to a level with a "
        "recommended action (elevated from 50, high from 75, extreme from 90).\n\n"
        "The inputs: **net GEX** in $bn and the **gamma flip**, the index level below which dealers amplify "
        "moves instead of dampening them; the **TailDex** as an index level; the **VIX curve** from spot and "
        "futures (M1 above M2 = inverted = acute fear); the **DIX** in per cent; and the **SMFI** from the "
        "previous day's value and the price change in the first half hour and the last hour of trading.\n\n"
        "‘Distance to the next thresholds’ shows, for each indicator, the value at which its next level begins "
        "and the value that on its own would lift the composite by one warning level. The what-if chart varies "
        "one indicator and holds the others fixed."
    ),
    "needs": [],
    "params": [
        {"name": "spot", "type": "float", "default": 6888.0, "min": 500.0, "max": 20000.0, "step": 1.0,
         "label": "S&P 500 (spot)", "unit": "pts"},
        {"name": "gex", "type": "float", "default": 4.2, "min": -20.0, "max": 25.0, "step": 0.1,
         "label": "Net Gamma Exposure", "unit": "$bn",
         "help": "Positive: dealers dampen moves. Negative: dealers amplify them."},
        {"name": "flip", "type": "float", "default": 6700.0, "min": 500.0, "max": 20000.0, "step": 5.0,
         "label": "Gamma flip level", "unit": "pts",
         "help": "Index level at which the gamma profile changes sign. The README gives no value; "
                 "6,700 (2.7% below spot) is an assumption."},
        {"name": "tdex", "type": "float", "default": 7.8, "min": 1.0, "max": 60.0, "step": 0.1,
         "label": "TailDex (TDEX)", "help": "CBOE TailDex index level; framework thresholds 5 / 10 / 15 / 20."},
        {"name": "vix", "type": "float", "default": 14.5, "min": 5.0, "max": 90.0, "step": 0.1, "label": "VIX spot",
         "help": "The README gives no VIX level; 14.5 is an assumption consistent with the 10.4% volatility "
                 "stated there."},
        {"name": "m1", "type": "float", "default": 15.7, "min": 5.0, "max": 90.0, "step": 0.05,
         "label": "VIX futures, front month (M1)"},
        {"name": "m2", "type": "float", "default": 18.05, "min": 5.0, "max": 90.0, "step": 0.05,
         "label": "VIX futures, 2nd month (M2)",
         "help": "README: M1/M2 = 0.87 (normal contango). The individual levels of M1, M2 and M4 are assumptions."},
        {"name": "m4", "type": "float", "default": 19.6, "min": 0.0, "max": 90.0, "step": 0.05,
         "label": "VIX futures, 4th month (M4)", "help": "0 = not available."},
        {"name": "dix", "type": "float", "default": 46.2, "min": 25.0, "max": 65.0, "step": 0.1,
         "label": "Dark Index (DIX)", "unit": "%",
         "help": "Short share of dark-pool volume; high = institutions are buying."},
        {"name": "smfi_prev", "type": "float", "default": 3.0, "min": -100.0, "max": 100.0, "step": 0.1,
         "label": "Previous day's SMFI"},
        {"name": "first30", "type": "float", "default": 0.1, "min": -5.0, "max": 5.0, "step": 0.05,
         "label": "Price change, first 30 min", "unit": "%",
         "help": "From the open to 30 minutes later (mostly retail investors)."},
        {"name": "last60", "type": "float", "default": 0.3, "min": -5.0, "max": 5.0, "step": 0.05,
         "label": "Price change, last hour", "unit": "%",
         "help": "SMFI = previous day − first 30 min + last hour (README: +3.2)."},
        {"name": "what_if", "type": "select", "default": "gex", "label": "What-if variable",
         "options": [{"value": "gex", "label": "Net GEX"}, {"value": "tdex", "label": "TailDex"},
                     {"value": "ratio", "label": "VIX curve (M1/M2)"}, {"value": "dix", "label": "DIX"},
                     {"value": "last60", "label": "SMFI: last hour"}]},
    ],
    "covers": [
        "src.analysis.early_warning.NetGammaExposure",
        "src.analysis.early_warning.NetGammaExposure.calculate_gex",
        "src.analysis.early_warning.NetGammaExposure.get_regime_description",
        "src.analysis.early_warning.TailDex",
        "src.analysis.early_warning.TailDex.calculate_tdex",
        "src.analysis.early_warning.VIXTermStructure",
        "src.analysis.early_warning.VIXTermStructure.calculate_term_structure",
        "src.analysis.early_warning.DarkIndex",
        "src.analysis.early_warning.DarkIndex.calculate_dix",
        "src.analysis.early_warning.SmartMoneyFlowIndex",
        "src.analysis.early_warning.SmartMoneyFlowIndex.calculate_smfi",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.compute_composite",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.get_risk_assessment",
    ],
    "cost": "fast",
}

# indicator -> (varied input, direction of higher risk, end of the search range, unit, label)
SCAN = {
    "gex": ("gex", -1, -20.0, "$bn", "Net GEX"),
    "tdex": ("tdex", +1, 60.0, "", "TDEX"),
    "vix_term": ("ratio", +1, 1.5, "", "M1/M2"),
    "dix": ("dix", -1, 25.0, "%", "DIX"),
    "smfi": ("last60", -1, -5.0, "%", "last-hour change"),
}
WHAT_IF = {"gex": ("gex", -15.0, 15.0, "Net GEX ($bn)"), "tdex": ("tdex", 2.0, 40.0, "TailDex"),
           "ratio": ("vix_term", 0.75, 1.3, "VIX curve M1/M2"), "dix": ("dix", 30.0, 56.0, "DIX (%)"),
           "last60": ("smfi", -3.0, 3.0, "Last-hour price change (%)")}


def _calc(key, p):
    """One framework snapshot (calculate_*) for the readings ``p``."""
    if key == "gex":
        return gex_snapshot(p["gex"], p["spot"], p["flip"])[0]
    if key == "tdex":
        return TailDex().calculate_tdex(np.array([p["tdex"] / 100.0]), np.array([-0.10]))
    if key == "vix_term":
        return VIXTermStructure().calculate_term_structure(p["vix"], p["m1"], p["m2"], p["m4"] if p["m4"] > 0 else None)
    if key == "dix":
        return DarkIndex().calculate_dix(p["dix"], 100.0)
    return SmartMoneyFlowIndex().calculate_smfi(100.0, 100.0 * (1 + p["first30"] / 100), 100.0,
                                                100.0 * (1 + p["last60"] / 100), prev_smfi=p["smfi_prev"])


def evaluate(p):
    """All five calculate_* snapshots plus the composite for one set of readings ``p``."""
    res = {k: _calc(k, p) for k in KEYS}
    ews = CompositeEarlyWarningSystem()
    res["composite"] = ews.compute_composite(*(res[k]["warning_score"] for k in KEYS))
    res["ews"] = ews
    res["gex_obj"] = gex_snapshot(p["gex"], p["spot"], p["flip"])[1]
    return res


class _Vary:
    """Change one reading, recompute only that indicator and the composite (the others stay fixed)."""

    def __init__(self, p, res, key, pname):
        self.p, self.key, self.pname = p, key, pname
        self.scores = {k: res[k]["warning_score"] for k in KEYS}
        self.ews = CompositeEarlyWarningSystem()

    def __call__(self, value):
        r = _calc(self.key, _with(self.p, self.pname, float(value)))
        scores = dict(self.scores, **{self.key: r["warning_score"]})
        return r, self.ews.compute_composite(*(scores[k] for k in KEYS))


def _first_change(f, now, limit, n=300):
    """First value from ``now`` towards ``limit`` where ``f`` changes (grid + bisection), else None."""
    base = f(now)
    prev = now
    for v in np.linspace(now, limit, n)[1:]:
        if f(v) != base:
            a, b = prev, float(v)
            for _ in range(30):
                mid = 0.5 * (a + b)
                a, b = (a, mid) if f(mid) != base else (mid, b)
            return b, f(b)
        prev = float(v)
    return None


def _state_code(key, r):
    """The categorical reading of one indicator snapshot (regime / level / structure / signal)."""
    return r[{"gex": "regime", "tdex": "level", "vix_term": "structure", "dix": "level", "smfi": "signal"}[key]]


def _state_label(key, code):
    return {"gex": GEX_REGIME, "tdex": TDEX_LEVEL, "vix_term": STRUCTURE, "dix": DIX_LEVEL, "smfi": SMFI_SIGNAL}[key].get(code, code)


def _with(p, name, value):
    q = dict(p)
    if name == "ratio":
        q["m1"] = value * p["m2"]
    else:
        q[name] = value
    return q


def run(params, data, out):
    p = dict(params)
    if p["m2"] <= 0 or p["m1"] <= 0:
        raise ToolError("The VIX futures M1 and M2 must be greater than zero.")
    res = evaluate(p)
    comp = res["composite"]
    level = comp["level"]
    status, level_txt = LEVEL[level]
    action_txt = ACTION.get(comp["action"], comp["action"])
    score = float(comp["composite_score"])
    ratio = float(res["vix_term"]["ratio_m1_m2"])
    th = res["ews"].thresholds

    out.set_verdict(status, f"{level_txt}: composite {num(score, 0)} out of 100 – {action_txt}",
                    "The framework's reading for this level: "
                    + assessment_text(res["ews"].get_risk_assessment(score)))
    out.metric("Composite score", score, "num", digits=0, unit="/ 100", status=status,
               hint=f"{level_txt} · {action_txt}")
    out.metric("Net GEX", p["gex"], "num", digits=1, unit="$bn", hint=GEX_REGIME[res["gex"]["regime"]],
               status=status_of(res["gex"]["warning_score"]))
    out.metric("Distance to gamma flip", res["gex"]["distance_to_flip"] / 100, "pct", digits=1,
               hint="Spot above the flip" if res["gex"]["distance_to_flip"] >= 0 else "Spot below the flip: negative gamma")
    out.metric("TailDex", p["tdex"], "num", digits=1, hint=f"{TDEX_LEVEL[res['tdex']['level']]} · "
               f"{ordinal(res['tdex']['percentile'])} percentile", status=status_of(res["tdex"]["warning_score"]))
    out.metric("VIX curve M1/M2", ratio, "num", digits=3, hint=STRUCTURE[res["vix_term"]["structure"]],
               status=status_of(res["vix_term"]["warning_score"]))
    out.metric("DIX", p["dix"] / 100, "pct", digits=1, hint=DIX_LEVEL[res["dix"]["level"]],
               status=status_of(res["dix"]["warning_score"]))
    out.metric("SMFI", res["smfi"]["smfi"], "num", digits=1, hint=SMFI_SIGNAL[res["smfi"]["signal"]],
               status=status_of(res["smfi"]["warning_score"]))

    # ---- gauge + score bars --------------------------------------------------------------------------
    out.chart(C.gauge("Composite score", score, thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Elevated", "High", "Extreme"), subtitle=f"{level_txt} · {action_txt}"))
    cats = ["GEX", "TDEX", "VIX curve", "DIX", "SMFI"]
    out.chart(C.xy(
        "Warning score by indicator",
        [C.trace("Score (0–100)", cats, [comp["components"][k]["score"] for k in KEYS], type="bar", color=1),
         C.trace("Contribution to composite", cats, [comp["components"][k]["contribution"] for k in KEYS], type="bar",
                 color=2)],
        x=C.axis("", type="category"), y=C.axis("Points", range=(0, 100)),
        subtitle="Contribution = score × weight (35 / 25 / 20 / 10 / 10%); together they make up the composite.",
    ))

    # ---- what-if ---------------------------------------------------------------------------------------
    name = p["what_if"]
    key, lo, hi, xlabel = WHAT_IF[name]
    cur = ratio if name == "ratio" else p[name]
    grid = np.linspace(min(lo, cur), max(hi, cur), 241)
    vary = _Vary(p, res, key, "ratio" if name == "ratio" else name)
    sims = [vary(v) for v in grid]
    out.chart(C.xy(
        f"What-if: {xlabel.split(' (')[0]}",
        [C.trace("Composite", grid, [c["composite_score"] for _, c in sims], color=1),
         C.trace(f"{NAMES[key].split(' (')[0]} score", grid, [r["warning_score"] for r, _ in sims], color=2,
                 dash=True)],
        x=C.axis(xlabel, digits=3 if name == "ratio" else 1), y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Elevated", "warning"), C.hline(th["high"], "High", "serious"),
                C.hline(th["extreme"], "Extreme", "critical")],
        vlines=[C.vline(cur, "today", "ink")],
        subtitle="All other inputs stay fixed. Choose the variable under ‘What-if variable’.",
    ))

    # ---- VIX curve ---------------------------------------------------------------------------------------
    months = [0, 1, 2] + ([4] if p["m4"] > 0 else [])
    levels = [p["vix"], p["m1"], p["m2"]] + ([p["m4"]] if p["m4"] > 0 else [])
    inverted = bool(res["vix_term"]["inverted"])
    out.chart(C.xy(
        "VIX term structure",
        [C.trace("VIX curve", months, levels, color="serious" if inverted else 1),
         C.trace("Spot and futures", months, levels, type="scatter", color="serious" if inverted else 1, size=9)],
        x=C.axis("Maturity (months, 0 = spot)", fmt="int"), y=C.axis("VIX points"),
        subtitle=("Downward-sloping curve (backwardation): near-term fear exceeds longer-term fear." if inverted
                  else "Upward-sloping curve (contango): little near-term fear – the normal case."),
    ))

    # ---- table: the five indicators -----------------------------------------------------------------
    g, t, v, d, s = (res[k] for k in KEYS)
    gdesc = res["gex_obj"].get_regime_description(g["regime"])
    inputs = [f"{bn(p['gex'], 1, sign=True)} · flip {num(p['flip'], 0)}", num(p["tdex"], 1),
              f"M1/M2 {num(ratio, 3)}", f"{num(p['dix'], 1)}%", num(s["smfi"], 1, sign=True)]
    states = [(GEX_REGIME[g["regime"]], GEX_SIGNAL[g["signal"]]), (TDEX_LEVEL[t["level"]], TDEX_SIGNAL[t["signal"]]),
              (STRUCTURE[v["structure"]], STRUCTURE_SIGNAL[v["signal"]]), (DIX_LEVEL[d["level"]], DIX_SIGNAL[d["signal"]]),
              ({"UP": "Rising trend", "DOWN": "Falling trend"}.get(s["trend"], "Trend n/a (no history)"),
               SMFI_SIGNAL[s["signal"]])]
    rows = [[NAMES[k], inputs[i], states[i][0], states[i][1], comp["components"][k]["score"],
             comp["components"][k]["weight"], comp["components"][k]["contribution"]] for i, k in enumerate(KEYS)]
    rows.append(["Composite", "", level_txt, action_txt, score, 1.0, score])
    out.table("The five indicators", ["Indicator", "Input", "Level", "Signal", "Score", "Weight", "Contribution"], rows,
              formats=["text", "text", "text", "text", "num", "pct:0", "num"],
              note=f"GEX regime: {GEX_DESCRIPTION.get(gdesc, gdesc)}")

    # ---- table: how close are the next thresholds? ----------------------------------------------------
    next_level = {"NORMAL": "ELEVATED", "ELEVATED": "HIGH", "HIGH": "EXTREME"}.get(level)
    trows = []
    for k in KEYS:
        pname, direction, limit, unit, qlabel = SCAN[k]
        now = ratio if pname == "ratio" else p[pname]
        vary = _Vary(p, res, k, pname)
        digits = 3 if pname == "ratio" else 2
        fmt_v = (lambda x, sign=pname in ("gex", "last60"):
                 bn(x, digits, sign=sign) if unit == "$bn" else f"{num(x, digits, sign=sign)}{unit}")
        name = NAMES[k] + (" – last hour" if k == "smfi" else "")
        if (limit - now) * direction <= 0:
            trows.append([name, fmt_v(now), "–", "Edge of the search range", "–", "–"])
            continue
        hit_state = _first_change(lambda v: _state_code(k, vary(v)[0]), now, limit)
        if next_level:
            hit_comp = _first_change(lambda v: vary(v)[1]["level"], now, limit)
            if hit_comp:
                comp_txt = f"{LEVEL[hit_comp[1]][1]} at {qlabel} {fmt_v(hit_comp[0])}"
            else:
                comp_txt = f"Not reachable on its own (max. {num(vary(limit)[1]['composite_score'], 0)} points)"
        else:
            comp_txt = "Already at the highest level"
        trows.append([
            name, fmt_v(now),
            _state_label(k, hit_state[1]) if hit_state else "No further level",
            fmt_v(hit_state[0]) if hit_state else "–",
            fmt_v(hit_state[0] - now, True) if hit_state else "–",
            comp_txt,
        ])
    out.table("Distance to the next thresholds",
              ["Indicator", "Today", "Next level (towards risk)", "Threshold", "Distance",
               f"Composite moves to {LEVEL[next_level][1] if next_level else '–'} (this indicator only)"],
              trows, formats=["text"] * 6,
              note="The search runs in the direction of higher risk: GEX, DIX and the SMFI last-hour change falling, TDEX "
                   "and M1/M2 rising (M2 held fixed). For the SMFI, the last-hour price change is varied; all other "
                   "inputs stay fixed.")

    # ---- table: details --------------------------------------------------------------------------------
    drows = [
        ["Distance spot → gamma flip", g["distance_to_flip"] / 100, "pct",
         "Below 1% the flip contributes the full 30 points, from 3% none"],
        ["TDEX percentile", t["percentile"] / 100, "pct",
         "Without its own history, measured against the framework's normal distribution N(8, 3)"],
        ["TDEX z-score", t["z_score"], "num", "(TDEX − 8) / 3"],
        ["VIX spot / M1", v["ratio_spot_m1"], "num", "> 1: spot above the front-month future – acute panic"],
        ["M1 / M4", v["ratio_m1_m4"], "num", "Long end of the curve" if v["ratio_m1_m4"] is not None else "No M4 given"],
        ["Roll yield p.a.", v["roll_yield"], "pct", "(M2 − M1) / M1 × 12: return on a short-volatility carry trade"],
        ["DIX percentile", d["percentile"] / 100, "pct", "Without history, the framework assumes 50%"],
        ["SMFI: first 30 min / last hour", f"{num(s['first_30min'], 2, sign=True)}% / {num(s['last_60min'], 2, sign=True)}%",
         "text", "Warning only when SMFI < −5 and the final hour is weak"],
    ]
    def show(label, value, kind):
        if value is None:
            return "–"
        if kind == "pct":
            return f"{num(value * 100, 1, sign=label.startswith(('Roll', 'Distance')))}%"
        return num(value, 3) if kind == "num" else value

    out.table("Calculation details", ["Quantity", "Value", "Explanation"],
              [[a, show(a, b, f), c] for a, b, f, c in drows], formats=["text", "text", "text"])

    out.note("The defaults are the README readings of 3 Jan 2026. For that date the README quotes sub-scores of "
             "18 / 22 / 15 / 25 / 20 and a composite of 20, which are illustrative values. Here the framework's "
             "formulas do the calculation, so the score comes out differently.", "info")
