"""Frühwarn-Dashboard: the five indicators panel by panel, the status table and the framework's dashboard figure."""

import numpy as np

from src.analysis.early_warning import CompositeEarlyWarningSystem
from src.utils.data_loader import generate_vix_from_returns
from src.visualization.early_warning_dashboard import (
    EarlyWarning3DVisualization,
    EarlyWarningDashboard,
    create_indicator_summary_table,
)

from .. import charts as C
from ..data import index_axis
from ._ews_common import ACTION, KEYS, LEVEL, assessment_de, de

SPEC = {
    "id": "ews-dashboard",
    "title": "Frühwarn-Dashboard: die fünf Indikatoren",
    "group": "warning",
    "order": 15,
    "summary": "GEX, TailDex, VIX-Kurve, Dark Index und Smart Money Flow einzeln mit ihren Warnschwellen – "
               "und wie viel jeder Indikator zum Komposit-Score beiträgt.",
    "description": (
        "Das Dashboard zerlegt den Komposit-Score des Frühwarnsystems in seine fünf Bausteine. **Net Gamma "
        "Exposure (GEX):** Sind Options-Dealer netto short Gamma (GEX < 0), verkaufen sie in fallende Kurse und "
        "verstärken Bewegungen. **TailDex (TDEX):** der Preis für Crash-Absicherung, über 15 erhöht, über 20 "
        "extrem. **VIX-Kurve:** Liegt der kurze über dem langen VIX (Verhältnis > 1), ist die Kurve invertiert – "
        "akute Angst. **Dark Index (DIX):** Anteil verdeckter Käufe, unter 43 % schwach, unter 40 % Distribution. "
        "**Smart Money Flow (SMFI):** Fällt er, steigen Profis aus.\n\n"
        "**Wichtig:** GEX, TDEX, DIX und SMFI beruhen auf Options- und Dark-Pool-Daten, die hier nicht "
        "vorliegen. Das Framework *modelliert* sie aus deinen Renditen und dem VIX – die Verläufe sind "
        "illustrativ, keine echten Messwerte. Fehlt der VIX, wird er aus den Renditen geschätzt. Enthält der "
        "Datensatz VIX3M (S&P 500), ist die VIX-Kurve echt. Gerechnet wird kausal: Jeder Tag nutzt nur Daten "
        "bis zu diesem Tag.\n\n"
        "So liest du es: Tacho und Beitragsbalken zeigen den letzten Tag. Jeder Indikator hat einen Teil-Score von "
        "0 (unauffällig) bis 100 (Maximum), der Komposit ist deren gewichtete Summe. GEX erreicht 50 beim "
        "Vorzeichenwechsel, TDEX bei 15; DIX und SMFI steigen ab ihrer Warnschwelle (43 % bzw. −5 in 20 Tagen) "
        "von 0 an. Die Panels darunter zeigen die Verläufe mit den Schwellen des Frameworks, beim SMFI die "
        "Änderung über 20 Tage, nach der er bewertet wird. Der 3D-Phasenraum zeigt die Bahn aus GEX-Risiko, "
        "TDEX-Risiko und Komposit. Darunter folgen die Statustabelle und die Original-Grafiken. Der **Seed** "
        "steuert das Modellrauschen."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "days", "type": "int", "default": 252, "min": 60, "max": 2520, "step": 21,
         "label": "Angezeigter Zeitraum", "unit": "Tage",
         "help": "Die letzten Handelstage in Panels und Grafiken. Modelliert wird immer die ganze Historie."},
        {"name": "term_source", "type": "select", "default": "auto", "label": "VIX-Kurve",
         "options": [{"value": "auto", "label": "Echt (VIX/VIX3M), falls im Datensatz"},
                     {"value": "model", "label": "Immer Modellkurve (aus dem VIX simuliert)"}]},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Zufalls-Seed des Modells",
         "help": "GEX, TDEX, DIX und SMFI enthalten Modellrauschen. Gleicher Seed = gleiche Reihen."},
        {"name": "show_figures", "type": "bool", "default": True, "label": "Original-Grafiken (Matplotlib) zeigen",
         "help": "Das 11-Panel-Dashboard und den 3D-Phasenraum des Frameworks als Bild rendern."},
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

SHORT = {"gex": "GEX", "tdex": "TDEX", "vix_term": "VIX-Kurve", "dix": "DIX", "smfi": "SMFI"}
# rows of create_indicator_summary_table -> (indicator key, German name)
TABLE_ROWS = {"1. GEX (B$)": ("gex", "Net Gamma (GEX), Mrd. $"), "2. TDEX": ("tdex", "TailDex (TDEX)"),
              "3. VIX M1/M2": ("vix_term", "VIX-Kurve (M1/M2)"), "4. DIX (%)": ("dix", "Dark Index (DIX), %"),
              "5. SMFI": ("smfi", "Smart Money Flow (SMFI)"), "COMPOSITE SCORE": (None, "Komposit-Score")}
SIGNAL = {"NEGATIVE": "Negativ – Dealer verstärken Bewegungen", "POSITIVE": "Positiv – Dealer dämpfen Bewegungen",
          "EXTREME": "Extrem", "ELEVATED": "Erhöht", "NORMAL": "Normal",
          "INVERTED": "Invertiert (Backwardation)", "CONTANGO": "Contango (normal)",
          "DISTRIBUTION": "Distribution", "WEAK": "Schwach",
          "SELLING": "Smart Money verkauft", "BUYING": "Smart Money kauft",
          "EXTREME RISK": "Extrem", "HIGH RISK": "Hoch"}


def _date_de(d):
    d = str(d)
    return f"{d[8:10]}.{d[5:7]}.{d[:4]}" if len(d) >= 10 and d[4] == "-" and d[7] == "-" else d


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
    """Status of one indicator's 0-100 sub-score (quarters: unauffällig / erhöht / hoch / kritisch)."""
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
    label = (lambda i: _date_de(x[i])) if data.dates else (lambda i: f"Tag {n - m + i}")

    vix_real = data.vix is not None
    vix = np.asarray(data.vix, dtype=float) if vix_real else np.asarray(generate_vix_from_returns(r), dtype=float)
    use_real_curve = params["term_source"] == "auto" and vix_real and data.vix3m is not None
    vix3m = np.asarray(data.vix3m, dtype=float) if use_real_curve else None
    real_prices = data.prices is not None
    prices = np.asarray(data.prices, dtype=float) if real_prices else 100.0 * np.exp(np.cumsum(r))

    out.progress("Modelliere die fünf Indikatoren …", 0.1)
    ews = CompositeEarlyWarningSystem()
    np.random.seed(int(params["seed"]))
    full = ews.simulate_all_indicators(r, prices, vix, causal=True, vix3m=vix3m)
    ind = {k: np.asarray(v, dtype=float)[n - m:] for k, v in full.items()}
    p = prices[n - m:]
    comp = ind["composite_score"]
    th = ews.thresholds
    last = ews.compute_composite(*(float(ind[k + "_score"][-1]) for k in KEYS))
    level = last["level"]
    status, level_de = LEVEL[level]
    action_de = ACTION.get(last["action"], last["action"])
    ratio = ind["vix_m1"] / ind["vix_m2"]
    ratio_name = "VIX/VIX3M" if use_real_curve else "M1/M2 (Modell)"
    score = {k: ind[k + "_score"] for k in KEYS}
    g, td, dx = (float(ind[k][-1]) for k in ("gex", "tdex", "dix"))
    # the SMFI level is a running sum since the first day of the data set; its sub-score rates the 20-day change
    smfi_full = np.asarray(full["smfi"], dtype=float)
    smfi_chg = np.full(n, np.nan)
    smfi_chg[21:] = smfi_full[21:] - smfi_full[1:-20]
    smfi_chg = smfi_chg[n - m:]

    # ---- headline: the last day, indicator by indicator ------------------------------------------------
    def sub(k):
        return f"Teil-Score {score[k][-1]:.0f} / 100"

    out.metric("Komposit (letzter Tag)", comp[-1], "num", digits=0, unit="/ 100", status=status,
               hint=f"{level_de} · {action_de}")
    out.metric("Net Gamma (GEX)", g, "num", digits=1, unit="Mrd. $", status=_sub_status(score["gex"][-1]),
               hint=("negativ: Dealer verstärken Bewegungen" if g < 0 else "positiv: Dealer dämpfen Bewegungen")
               + " · " + sub("gex"))
    out.metric("TailDex (TDEX)", td, "num", digits=1, status=_sub_status(score["tdex"][-1]),
               hint="über 15 erhöht, über 20 extrem · " + sub("tdex"))
    out.metric(f"VIX-Kurve {ratio_name}", ratio[-1], "num", digits=3, status=_sub_status(score["vix_term"][-1]),
               hint=("> 1: invertiert, akute Angst" if ratio[-1] > 1 else "≤ 1: Contango, normal") + " · "
               + sub("vix_term"))
    out.metric("Dark Index (DIX)", dx / 100, "pct", digits=1, status=_sub_status(score["dix"][-1]),
               hint="unter 43 % schwach, unter 40 % Distribution · " + sub("dix"))
    out.metric("Smart Money Flow (SMFI), 20 Tage", smfi_chg[-1], "num", digits=1, status=_sub_status(score["smfi"][-1]),
               hint="Warnung ab −5, Teil-Score 100 ab −15 · " + sub("smfi"))

    i_max = int(np.argmax(comp))
    driver = max(KEYS, key=lambda k: last["components"][k]["contribution"])
    d_pts = last["components"][driver]["contribution"]
    out.metric("Höchster Komposit im Zeitraum", comp[i_max], "num", digits=0, unit="/ 100",
               status=LEVEL[ews.compute_composite(*(float(score[k][i_max]) for k in KEYS))["level"]][0],
               hint=f"am {label(i_max)}")
    out.metric("Tage ≥ 50 (erhöht)", float(np.mean(comp >= th["elevated"])), "pct", digits=1,
               hint=f"in den letzten {de(m, 0)} Handelstagen")
    out.metric("Tage ≥ 75 (hoch)", float(np.mean(comp >= th["high"])), "pct", digits=1,
               hint=f"in den letzten {de(m, 0)} Handelstagen")
    out.metric("Stärkster Treiber heute", SHORT[driver] if d_pts > 0 else "–", "text",
               hint=(f"{de(d_pts, 1)} von {de(float(comp[-1]), 1)} Punkten" if d_pts > 0
                     else "kein Indikator schlägt an"))
    out.note("**Modellierte Indikatoren:** GEX, TailDex, DIX und SMFI brauchen proprietäre Options- bzw. "
             "Dark-Pool-Daten. Das Framework modelliert sie hier aus Renditen und VIX – illustrative Verläufe, "
             "keine echten Marktmessungen.", "warning")
    src_notes = []
    if use_real_curve:
        src_notes.append("**VIX-Kurve echt:** VIX/VIX3M aus dem Datensatz.")
    elif vix_real and data.vix3m is not None:
        src_notes.append("**VIX-Kurve modelliert** (so gewählt): M1/M2 aus dem VIX simuliert.")
    else:
        src_notes.append("**VIX-Kurve modelliert:** kein VIX3M im Datensatz, M1/M2 aus dem VIX simuliert.")
    if not vix_real:
        src_notes.append("Der VIX selbst fehlt und wird aus den Renditen geschätzt (`generate_vix_from_returns`).")
    if not real_prices:
        src_notes.append("Keine Kurse im Datensatz: Der Kurs ist ein Index aus den Renditen (Start 100).")
    src_notes.append(f"Modelliert über alle {de(n, 0)} Tage, kausal (`causal=True`); angezeigt die letzten {de(m, 0)}.")
    out.note(" ".join(src_notes), "info")

    # ---- today: gauge + contributions -----------------------------------------------------------------------
    out.chart(C.gauge("Komposit am letzten Tag", float(comp[-1]),
                      thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Erhöht", "Hoch", "Extrem"),
                      subtitle=f"{label(m - 1)} · Stufe {level_de}: {action_de}"))
    cats = [f"{SHORT[k]} · {ews.weights[k]:.0%}".replace("%", " %") for k in KEYS]
    out.chart(C.xy(
        "Beitrag der Indikatoren am letzten Tag",
        [C.trace("Beitrag zum Komposit", cats, [last["components"][k]["contribution"] for k in KEYS],
                 type="bar", color=1),
         C.trace("Höchstmöglich (Gewicht × 100)", cats, [100 * ews.weights[k] for k in KEYS],
                 type="scatter", color="muted", size=9, decimate_to=None)],
        x=C.axis("", type="category"), y=C.axis("Punkte", range=(0, 40)),
        subtitle="Beitrag = Teil-Score × Gewicht. Die Summe aller Beiträge ist der Komposit-Score.",
    ))

    # ---- price with warning phases + composite history --------------------------------------------------
    bands = []
    for key, col in (("elevated", "warning"), ("high", "serious"), ("extreme", "critical")):
        for a, b in _runs(comp >= th[key])[:60]:
            bands.append(C.band(x[a], x[min(b + 1, m - 1)], "", col))
    out.chart(C.xy(
        "Kurs mit Warnphasen",
        [C.trace("Kurs" if real_prices else "Index aus Renditen (Start 100)", x, p, color=1, width=1.5)],
        x=xa, y=C.axis("Kurs", type="log" if p.max() > 8 * p.min() else "linear"), bands=bands,
        subtitle=f"Hinterlegt: Tage mit Komposit ≥ {th['elevated']:g} (erhöht), ≥ {th['high']:g} (hoch) und "
                 f"≥ {th['extreme']:g} (extrem) – je höher, desto kräftiger.",
    ))
    out.chart(C.xy(
        "Komposit-Score",
        [C.trace("Komposit", x, comp, color=1, width=1.5)],
        x=xa, y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Erhöht", "warning"), C.hline(th["high"], "Hoch", "serious"),
                C.hline(th["extreme"], "Extrem", "critical")],
        subtitle="Stufengrenzen des Frameworks: 50 / 75 / 90.",
    ))

    # ---- the five panels ----------------------------------------------------------------------------------
    pos, neg = _split_sign(ind["gex"])
    out.chart(C.xy(
        f"1 · Net Gamma Exposure (GEX) – {ews.weights['gex']:.0%} Gewicht".replace("%", " %"),
        [C.trace("GEX ≥ 0: dämpfend", x, pos, type="bar", color=1),
         C.trace("GEX < 0: verstärkend", x, neg, type="bar", color="serious")],
        x=xa, y=C.axis("Mrd. $"),
        hlines=[C.hline(-5, "Teil-Score 100", "critical")],
        subtitle="Unter 0 verkaufen Dealer in fallende Kurse. Ab −5 Mrd. $ steht der Teil-Score auf 100.",
        height=300,
    ))
    out.chart(C.xy(
        f"2 · TailDex (TDEX) – {ews.weights['tdex']:.0%} Gewicht".replace("%", " %"),
        [C.trace("TDEX", x, ind["tdex"], color=2, width=1.5)],
        x=xa, y=C.axis("TDEX"),
        hlines=[C.hline(15, "Erhöht", "warning"), C.hline(20, "Extrem", "critical")],
        subtitle="Preis der Crash-Absicherung. Über 20 steht der Teil-Score auf 100.", height=300,
    ))
    out.chart(C.xy(
        f"3 · VIX-Kurve – {ews.weights['vix_term']:.0%} Gewicht".replace("%", " %"),
        [C.trace(ratio_name, x, ratio, color=3, width=1.5)],
        x=xa, y=C.axis("Verhältnis kurz / lang", digits=3),
        hlines=[C.hline(1.0, "Inversion", "serious")],
        subtitle=("Echtes Verhältnis VIX zu VIX3M." if use_real_curve else "Aus dem VIX simulierte Futures-Kurve.")
        + " Über 1 ist die Kurve invertiert.", height=300,
    ))
    out.chart(C.xy(
        f"4 · Dark Index (DIX) – {ews.weights['dix']:.0%} Gewicht".replace("%", " %"),
        [C.trace("DIX", x, ind["dix"] / 100, color=4, width=1.5)],
        x=xa, y=C.axis("Anteil", fmt="pct", digits=0),
        hlines=[C.hline(0.43, "Schwach", "warning"), C.hline(0.40, "Distribution", "serious")],
        subtitle="Anteil verdeckter Käufe. Darunter verteilen Institutionelle eher, als dass sie kaufen.",
        height=300,
    ))
    out.chart(C.xy(
        f"5 · Smart Money Flow (SMFI) – {ews.weights['smfi']:.0%} Gewicht".replace("%", " %"),
        [C.trace("SMFI-Änderung über 20 Tage", x, smfi_chg, color=5, width=1.5)],
        x=xa, y=C.axis("Punkte in 20 Tagen"),
        hlines=[C.hline(-5, "Warnung", "warning"), C.hline(-15, "Teil-Score 100", "critical")],
        subtitle="Bewertet wird die Änderung, nicht der Stand: Fällt der SMFI in 20 Tagen um mehr als 5 Punkte, "
                 "steigt der Teil-Score, ab −15 steht er auf 100.", height=300,
    ))
    step = max(1, int(np.ceil(m / 260)))
    starts = np.arange(0, m, step)
    out.chart(C.xy(
        "Woraus sich der Komposit zusammensetzt",
        [C.trace(f"{SHORT[k]} · {ews.weights[k]:.0%}".replace("%", " %"), [x[a] for a in starts],
                 [float(np.mean(score[k][a:a + step])) * ews.weights[k] for a in starts], type="bar", color=i + 1)
         for i, k in enumerate(KEYS)],
        x=xa, y=C.axis("Punkte", range=(0, 100)), stacked=True,
        subtitle="Beitrag je Indikator gestapelt; die Oberkante ist der Komposit-Score"
        + (f" (Mittel je {step} Tage)." if step > 1 else "."), height=300,
    ))

    # ---- 3D phase space (EarlyWarning3DVisualization as interactive trajectory) -------------------------
    gx, ty = score["gex"], score["tdex"]
    lv = np.array([("EXTREME" if v >= th["extreme"] else "HIGH" if v >= th["high"]
                    else "ELEVATED" if v >= th["elevated"] else "NORMAL") for v in comp])
    traces = [C.path3d("Bahn", gx, ty, comp, color="muted", mode="lines")]
    for key in ("NORMAL", "ELEVATED", "HIGH", "EXTREME"):
        sel = lv == key
        if sel.any():
            traces.append(C.path3d(LEVEL[key][1], gx[sel], ty[sel], comp[sel], color=LEVEL[key][0], mode="markers",
                                   size=3, text=[label(i) for i in np.flatnonzero(sel)]))
    traces.append(C.path3d("Gefahrenebene (Komposit 75)", [0, 100, 100, 0, 0], [0, 0, 100, 100, 0], [75] * 5,
                           color="critical", mode="lines"))
    traces.append(C.path3d("Letzter Tag", [gx[-1]], [ty[-1]], [comp[-1]], color="ink", mode="markers", size=8,
                           text=[label(m - 1)]))
    out.chart(C.scatter3d(
        "Frühwarn-Phasenraum", traces,
        x=C.axis("GEX-Risiko (Teil-Score)"), y=C.axis("TDEX-Risiko (Teil-Score)"), z=C.axis("Komposit"),
        subtitle="Jeder Punkt ist ein Tag. Über der roten Ebene (75) liegt die hohe Warnstufe.",
    ))

    # ---- the framework's status table ----------------------------------------------------------------
    text = create_indicator_summary_table(ind)
    rows = []
    for name, value, weight, signal in _parse_summary(text):
        key, name_de = TABLE_ROWS.get(name, (None, name))
        if key == "vix_term" and use_real_curve:
            name_de = "VIX-Kurve (VIX/VIX3M)"
        w_txt = f"{weight:.0%}".replace("%", " %")
        if key is None:
            rows.append([name_de, de(value, 1), w_txt, SIGNAL.get(signal, signal), float(comp[-1]), float(comp[-1])])
        else:
            digits = 3 if key == "vix_term" else 2 if key == "gex" else 1
            rows.append([name_de, de(value, digits), w_txt, SIGNAL.get(signal, signal),
                         last["components"][key]["score"], last["components"][key]["contribution"]])
    out.table(f"Status am {label(m - 1)} (Tabelle des Frameworks)",
              ["Indikator", "Wert", "Gewicht", "Signal", "Teil-Score", "Beitrag"], rows,
              formats=["text", "text", "text", "text", "num", "num"],
              note="Signale nach create_indicator_summary_table, ergänzt um Teil-Score und Beitrag (Punkte). "
                   "Die Originaltabelle gibt es unten als Textdatei.")
    out.download("fruehwarn_status.txt", text.strip("\n") + "\n")

    assessment = assessment_de(ews.get_risk_assessment(float(comp[-1])))
    out.set_verdict(status, f"Letzter Stand: {level_de} – {comp[-1]:.0f} von 100",
                    f"{assessment} Modellwert aus Renditen und VIX, kein Live-Signal aus Options- oder "
                    "Dark-Pool-Daten.")

    # ---- the framework's figures --------------------------------------------------------------------------
    if params["show_figures"]:
        out.progress("Zeichne das Original-Dashboard …", 0.6)
        dates64 = np.array(x, dtype="datetime64[D]") if data.dates else None
        title = f"Early Warning System Dashboard · {data.name or 'Datensatz'} · {label(0)} – {label(m - 1)}"
        fig = EarlyWarningDashboard(figsize=(16, 13)).create_dashboard(ind, p, dates=dates64, title=title)
        out.figure(fig, "Frühwarn-Dashboard des Frameworks (EarlyWarningDashboard)", dpi=80)
        out.progress("3D-Phasenraum …", 0.85)
        fig = EarlyWarning3DVisualization().create_3d_phase_space(ind["gex_score"], ind["tdex_score"], comp,
                                                                   figsize=(10, 8))
        out.figure(fig, "3D-Phasenraum des Frameworks (EarlyWarning3DVisualization)", dpi=90)
