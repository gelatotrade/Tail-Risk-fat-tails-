"""Frühwarn-Rechner: today's readings of the five indicators -> scores, composite, level and nearby thresholds."""

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
    STRUCTURE, STRUCTURE_SIGNAL, TDEX_LEVEL, TDEX_SIGNAL, assessment_de, de, gex_snapshot, status_of,
)

SPEC = {
    "id": "early-warning-calculator",
    "title": "Frühwarn-Rechner",
    "group": "warning",
    "order": 20,
    "summary": "Gib die heutigen Werte von GEX, TailDex, VIX-Kurve, DIX und SMFI ein: Warnscores, Komposit, "
               "Stufe mit Handlungsempfehlung – und wie weit die nächste Schwelle entfernt ist.",
    "description": (
        "Hier gibst du die aktuellen Ablesungen der fünf Frühwarnindikatoren ein – vorbelegt mit dem Stand aus "
        "dem README (Januar 2026). Das Framework rechnet sie mit seinen Snapshot-Methoden (`calculate_gex`, "
        "`calculate_tdex`, `calculate_term_structure`, `calculate_dix`, `calculate_smfi`) in Warnscores von "
        "0 bis 100 um, gewichtet sie mit 35 / 25 / 20 / 10 / 10 % zum Komposit und ordnet ihn einer Stufe mit "
        "Handlungsempfehlung zu (ab 50 erhöht, ab 75 hoch, ab 90 extrem).\n\n"
        "Die Eingaben: **Net GEX** in Mrd. $ und der **Gamma-Flip** – der Indexstand, unter dem Dealer "
        "Bewegungen verstärken statt dämpfen; der **TailDex** als Indexstand; die **VIX-Kurve** aus Spot und "
        "Futures (M1 über M2 = invertiert = akute Angst); der **DIX** in Prozent; der **SMFI** aus "
        "Vortageswert sowie Kursänderung in der ersten halben und der letzten Handelsstunde.\n\n"
        "„Wie nah sind die nächsten Schwellen?“ zeigt je Indikator, ab welchem Wert seine nächste Stufe "
        "beginnt und welcher Wert allein den Komposit eine Warnstufe höher heben würde. Das "
        "Was-wäre-wenn-Diagramm variiert einen Indikator und hält die übrigen fest."
    ),
    "needs": [],
    "params": [
        {"name": "spot", "type": "float", "default": 6888.0, "min": 500.0, "max": 20000.0, "step": 1.0,
         "label": "S&P 500 (Spot)", "unit": "Pkt."},
        {"name": "gex", "type": "float", "default": 4.2, "min": -20.0, "max": 25.0, "step": 0.1,
         "label": "Net Gamma Exposure", "unit": "Mrd. $",
         "help": "Positiv: Dealer dämpfen Bewegungen. Negativ: Dealer verstärken sie."},
        {"name": "flip", "type": "float", "default": 6700.0, "min": 500.0, "max": 20000.0, "step": 5.0,
         "label": "Gamma-Flip-Level", "unit": "Pkt.",
         "help": "Indexstand, an dem das Gamma-Profil das Vorzeichen wechselt. Das README nennt keinen Wert – "
                 "6.700 (2,7 % unter Spot) ist eine Annahme."},
        {"name": "tdex", "type": "float", "default": 7.8, "min": 1.0, "max": 60.0, "step": 0.1,
         "label": "TailDex (TDEX)", "help": "Indexstand des CBOE TailDex; Schwellen des Frameworks 5 / 10 / 15 / 20."},
        {"name": "vix", "type": "float", "default": 14.5, "min": 5.0, "max": 90.0, "step": 0.1, "label": "VIX Spot",
         "help": "Das README nennt keinen VIX-Stand – 14,5 ist eine Annahme passend zur dort genannten Vola von 10,4 %."},
        {"name": "m1", "type": "float", "default": 15.7, "min": 5.0, "max": 90.0, "step": 0.05,
         "label": "VIX-Future 1. Monat (M1)"},
        {"name": "m2", "type": "float", "default": 18.05, "min": 5.0, "max": 90.0, "step": 0.05,
         "label": "VIX-Future 2. Monat (M2)",
         "help": "README: M1/M2 = 0,87 (normales Contango). Die Einzelstände von M1, M2 und M4 sind Annahmen."},
        {"name": "m4", "type": "float", "default": 19.6, "min": 0.0, "max": 90.0, "step": 0.05,
         "label": "VIX-Future 4. Monat (M4)", "help": "0 = nicht verfügbar."},
        {"name": "dix", "type": "float", "default": 46.2, "min": 25.0, "max": 65.0, "step": 0.1,
         "label": "Dark Index (DIX)", "unit": "%", "help": "Short-Anteil im Dark-Pool-Volumen; hoch = Institutionelle kaufen."},
        {"name": "smfi_prev", "type": "float", "default": 3.0, "min": -100.0, "max": 100.0, "step": 0.1,
         "label": "SMFI des Vortags"},
        {"name": "first30", "type": "float", "default": 0.1, "min": -5.0, "max": 5.0, "step": 0.05,
         "label": "Kursänderung erste 30 Min.", "unit": "%", "help": "Eröffnung → 30 Minuten danach (eher Privatanleger)."},
        {"name": "last60", "type": "float", "default": 0.3, "min": -5.0, "max": 5.0, "step": 0.05,
         "label": "Kursänderung letzte Stunde", "unit": "%",
         "help": "SMFI = Vortag − erste 30 Min. + letzte Stunde (README: +3,2)."},
        {"name": "what_if", "type": "select", "default": "gex", "label": "Was-wäre-wenn für",
         "options": [{"value": "gex", "label": "Net GEX"}, {"value": "tdex", "label": "TailDex"},
                     {"value": "ratio", "label": "VIX-Kurve (M1/M2)"}, {"value": "dix", "label": "DIX"},
                     {"value": "last60", "label": "SMFI: letzte Stunde"}]},
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
    "gex": ("gex", -1, -20.0, "Mrd. $", "Net GEX"),
    "tdex": ("tdex", +1, 60.0, "", "TDEX"),
    "vix_term": ("ratio", +1, 1.5, "", "M1/M2"),
    "dix": ("dix", -1, 25.0, "%", "DIX"),
    "smfi": ("last60", -1, -5.0, "%", "letzte Stunde"),
}
WHAT_IF = {"gex": ("gex", -15.0, 15.0, "Net GEX (Mrd. $)"), "tdex": ("tdex", 2.0, 40.0, "TailDex"),
           "ratio": ("vix_term", 0.75, 1.3, "VIX-Kurve M1/M2"), "dix": ("dix", 30.0, 56.0, "DIX (%)"),
           "last60": ("smfi", -3.0, 3.0, "Kursänderung letzte Stunde (%)")}


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


def _state_de(key, code):
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
        raise ToolError("Die VIX-Futures M1 und M2 müssen größer als null sein.")
    res = evaluate(p)
    comp = res["composite"]
    level = comp["level"]
    status, level_de = LEVEL[level]
    action_de = ACTION.get(comp["action"], comp["action"])
    score = float(comp["composite_score"])
    ratio = float(res["vix_term"]["ratio_m1_m2"])
    th = res["ews"].thresholds

    out.set_verdict(status, f"{level_de}: Komposit {de(score, 0)} von 100 – {action_de}",
                    "Lesart des Frameworks für diese Stufe: " + assessment_de(res["ews"].get_risk_assessment(score)))
    out.metric("Komposit-Score", score, "num", digits=0, unit="/ 100", status=status, hint=f"{level_de} · {action_de}")
    out.metric("Net GEX", p["gex"], "num", digits=1, unit="Mrd. $", hint=GEX_REGIME[res["gex"]["regime"]],
               status=status_of(res["gex"]["warning_score"]))
    out.metric("Abstand zum Gamma-Flip", res["gex"]["distance_to_flip"] / 100, "pct", digits=1,
               hint="Spot über dem Flip" if res["gex"]["distance_to_flip"] >= 0 else "Spot unter dem Flip: negatives Gamma")
    out.metric("TailDex", p["tdex"], "num", digits=1, hint=f"{TDEX_LEVEL[res['tdex']['level']]} · "
               f"{de(res['tdex']['percentile'], 0)}. Perzentil", status=status_of(res["tdex"]["warning_score"]))
    out.metric("VIX-Kurve M1/M2", ratio, "num", digits=3, hint=STRUCTURE[res["vix_term"]["structure"]],
               status=status_of(res["vix_term"]["warning_score"]))
    out.metric("DIX", p["dix"] / 100, "pct", digits=1, hint=DIX_LEVEL[res["dix"]["level"]],
               status=status_of(res["dix"]["warning_score"]))
    out.metric("SMFI", res["smfi"]["smfi"], "num", digits=1, hint=SMFI_SIGNAL[res["smfi"]["signal"]],
               status=status_of(res["smfi"]["warning_score"]))

    # ---- gauge + score bars --------------------------------------------------------------------------
    out.chart(C.gauge("Komposit-Score", score, thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Erhöht", "Hoch", "Extrem"), subtitle=f"Stufe {level_de}: {action_de}"))
    cats = ["GEX", "TDEX", "VIX-Kurve", "DIX", "SMFI"]
    out.chart(C.xy(
        "Warnscore je Indikator",
        [C.trace("Score (0–100)", cats, [comp["components"][k]["score"] for k in KEYS], type="bar", color=1),
         C.trace("Beitrag zum Komposit", cats, [comp["components"][k]["contribution"] for k in KEYS], type="bar",
                 color=2)],
        x=C.axis("", type="category"), y=C.axis("Punkte", range=(0, 100)),
        subtitle="Beitrag = Score × Gewicht (35 / 25 / 20 / 10 / 10 %); zusammen ergeben sie den Komposit.",
    ))

    # ---- what-if ---------------------------------------------------------------------------------------
    name = p["what_if"]
    key, lo, hi, xlabel = WHAT_IF[name]
    cur = ratio if name == "ratio" else p[name]
    grid = np.linspace(min(lo, cur), max(hi, cur), 241)
    vary = _Vary(p, res, key, "ratio" if name == "ratio" else name)
    sims = [vary(v) for v in grid]
    out.chart(C.xy(
        f"Was wäre, wenn sich nur {xlabel.split(' (')[0]} ändert?",
        [C.trace("Komposit", grid, [c["composite_score"] for _, c in sims], color=1),
         C.trace(f"Score {NAMES[key].split(' (')[0]}", grid, [r["warning_score"] for r, _ in sims], color=2,
                 dash=True)],
        x=C.axis(xlabel, digits=3 if name == "ratio" else 1), y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Erhöht", "warning"), C.hline(th["high"], "Hoch", "serious"),
                C.hline(th["extreme"], "Extrem", "critical")],
        vlines=[C.vline(cur, "heute", "ink")],
        subtitle="Alle anderen Eingaben bleiben fest. Wählbar unter „Was-wäre-wenn für“.",
    ))

    # ---- VIX curve ---------------------------------------------------------------------------------------
    months = [0, 1, 2] + ([4] if p["m4"] > 0 else [])
    levels = [p["vix"], p["m1"], p["m2"]] + ([p["m4"]] if p["m4"] > 0 else [])
    inverted = bool(res["vix_term"]["inverted"])
    out.chart(C.xy(
        "VIX-Terminkurve",
        [C.trace("VIX-Kurve", months, levels, color="serious" if inverted else 1),
         C.trace("Spot und Futures", months, levels, type="scatter", color="serious" if inverted else 1, size=9)],
        x=C.axis("Laufzeit (Monate, 0 = Spot)", fmt="int"), y=C.axis("VIX-Punkte"),
        subtitle=("Fallende Kurve (Backwardation): akute Angst übersteigt die langfristige." if inverted
                  else "Steigende Kurve (Contango): kurzfristig wenig Angst – der Normalfall."),
    ))

    # ---- table: the five indicators -----------------------------------------------------------------
    g, t, v, d, s = (res[k] for k in KEYS)
    gdesc = res["gex_obj"].get_regime_description(g["regime"])
    inputs = [f"{de(p['gex'], 1, sign=True)} Mrd. $ · Flip {de(p['flip'], 0)}", de(p["tdex"], 1),
              f"M1/M2 {de(ratio, 3)}", f"{de(p['dix'], 1)} %", de(s["smfi"], 1, sign=True)]
    states = [(GEX_REGIME[g["regime"]], GEX_SIGNAL[g["signal"]]), (TDEX_LEVEL[t["level"]], TDEX_SIGNAL[t["signal"]]),
              (STRUCTURE[v["structure"]], STRUCTURE_SIGNAL[v["signal"]]), (DIX_LEVEL[d["level"]], DIX_SIGNAL[d["signal"]]),
              ("Trend " + {"UP": "steigend", "DOWN": "fallend"}.get(s["trend"], "– (keine Historie)"),
               SMFI_SIGNAL[s["signal"]])]
    rows = [[NAMES[k], inputs[i], states[i][0], states[i][1], comp["components"][k]["score"],
             comp["components"][k]["weight"], comp["components"][k]["contribution"]] for i, k in enumerate(KEYS)]
    rows.append(["Komposit", "", level_de, action_de, score, 1.0, score])
    out.table("Die fünf Indikatoren", ["Indikator", "Eingabe", "Stufe", "Signal", "Score", "Gewicht", "Beitrag"], rows,
              formats=["text", "text", "text", "text", "num", "pct:0", "num"],
              note=f"GEX-Regime: {GEX_DESCRIPTION.get(gdesc, gdesc)}")

    # ---- table: how close are the next thresholds? ----------------------------------------------------
    next_level = {"NORMAL": "ELEVATED", "ELEVATED": "HIGH", "HIGH": "EXTREME"}.get(level)
    trows = []
    for k in KEYS:
        pname, direction, limit, unit, qlabel = SCAN[k]
        now = ratio if pname == "ratio" else p[pname]
        vary = _Vary(p, res, k, pname)
        digits = 3 if pname == "ratio" else 2
        fmt_v = (lambda x, sign=pname in ("gex", "last60"): f"{de(x, digits, sign=sign)}{(' ' + unit) if unit else ''}")
        name = NAMES[k] + (" – letzte Stunde" if k == "smfi" else "")
        if (limit - now) * direction <= 0:
            trows.append([name, fmt_v(now), "–", "Rand des Suchbereichs", "–", "–"])
            continue
        hit_state = _first_change(lambda v: _state_code(k, vary(v)[0]), now, limit)
        if next_level:
            hit_comp = _first_change(lambda v: vary(v)[1]["level"], now, limit)
            if hit_comp:
                comp_txt = f"{LEVEL[hit_comp[1]][1]} bei {qlabel} {fmt_v(hit_comp[0])}"
            else:
                comp_txt = f"allein nicht erreichbar (höchstens {de(vary(limit)[1]['composite_score'], 0)} Punkte)"
        else:
            comp_txt = "schon höchste Stufe"
        trows.append([
            name, fmt_v(now),
            _state_de(k, hit_state[1]) if hit_state else "keine weitere Stufe",
            fmt_v(hit_state[0]) if hit_state else "–",
            fmt_v(hit_state[0] - now, True) if hit_state else "–",
            comp_txt,
        ])
    out.table("Wie nah sind die nächsten Schwellen?",
              ["Indikator", "Heute", "Nächste Stufe (Richtung Risiko)", "ab", "Abstand",
               f"Komposit springt auf {LEVEL[next_level][1] if next_level else '–'} (nur dieser Indikator)"],
              trows, formats=["text"] * 6,
              note="Gesucht wird in Richtung höheren Risikos: GEX, DIX und SMFI-Schlussstunde fallend, TDEX und M1/M2 "
                   "steigend (M2 bleibt fest). Beim SMFI wird die Kursänderung der letzten Stunde variiert, alle "
                   "übrigen Eingaben bleiben fest.")

    # ---- table: details --------------------------------------------------------------------------------
    drows = [
        ["Abstand Spot → Gamma-Flip", g["distance_to_flip"] / 100, "pct",
         "Unter 1 % gibt der Flip volle 30 Punkte, ab 3 % keine"],
        ["TDEX-Perzentil", t["percentile"] / 100, "pct",
         "Ohne eigene Historie gegen die Normalverteilung N(8; 3) des Frameworks"],
        ["TDEX z-Wert", t["z_score"], "num", "(TDEX − 8) / 3"],
        ["VIX Spot / M1", v["ratio_spot_m1"], "num", "> 1: Spot über dem Front-Future – akute Panik"],
        ["M1 / M4", v["ratio_m1_m4"], "num", "Langes Ende der Kurve" if v["ratio_m1_m4"] is not None else "kein M4 angegeben"],
        ["Roll-Rendite p. a.", v["roll_yield"], "pct", "(M2 − M1) / M1 × 12: Ertrag eines Short-Vol-Carry"],
        ["DIX-Perzentil", d["percentile"] / 100, "pct", "Ohne Historie setzt das Framework 50 %"],
        ["SMFI: erste 30 Min. / letzte Stunde", f"{de(s['first_30min'], 2, sign=True)} % / {de(s['last_60min'], 2, sign=True)} %",
         "text", "Warnung erst bei SMFI < −5 und schwacher Schlussstunde"],
    ]
    def show(label, value, kind):
        if value is None:
            return "–"
        if kind == "pct":
            return f"{de(value * 100, 1, sign=label.startswith(('Roll', 'Abstand')))} %"
        return de(value, 3) if kind == "num" else value

    out.table("Details der Berechnung", ["Größe", "Wert", "Erklärung"],
              [[a, show(a, b, f), c] for a, b, f, c in drows], formats=["text", "text", "text"])

    out.note("Vorbelegt mit dem README-Stand vom 3. Januar 2026. Das README nennt dafür Teil-Scores von 18 / 22 / 15 / "
             "25 / 20 und einen Komposit von 20 – illustrative Werte. Hier rechnen die Formeln des Frameworks, "
             "deshalb fällt der Score anders aus.", "info")
