"""Frühwarnsystem: the five-indicator composite score over time (causal, real VIX/VIX3M curve when available)."""

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
from ..data import index_axis
from ._ews_common import (
    ACTION, DIVERGENCE, DIX_LEVEL, GEX_DESCRIPTION, GEX_REGIME, KEYS, LEVEL, NAMES, STRUCTURE, TDEX_LEVEL,
    WARNING_LEVEL, assessment_de, de, gex_snapshot,
)

SPEC = {
    "id": "early-warning",
    "title": "Frühwarnsystem: Komposit-Score",
    "group": "warning",
    "order": 10,
    "summary": "Der Komposit-Score aus GEX, TailDex, VIX-Kurve, DIX und SMFI über die Zeit – mit Warnstufen, "
               "Tachonadel für heute und einem ehrlichen Rückblick, was auf Warnungen folgte.",
    "description": (
        "Das Frühwarnsystem des Frameworks verdichtet fünf Indikatoren zu einem Score von 0 bis 100: "
        "**Net Gamma Exposure** (35 %), **TailDex** (25 %), **VIX-Terminkurve** (20 %), **Dark Index** (10 %) "
        "und **Smart Money Flow** (10 %). Ab 50 gilt die Lage als erhöht, ab 75 als hoch, ab 90 als extrem.\n\n"
        "**Wichtig:** GEX, TailDex, DIX und SMFI beruhen auf proprietären Options- und Dark-Pool-Daten, die "
        "hier nicht vorliegen. Das Framework *modelliert* sie aus deinen Renditen und dem VIX (`simulate_*`) – "
        "die Verläufe sind illustrativ, keine echten Messwerte. Echt ist die VIX-Kurve, wenn der Datensatz VIX "
        "und VIX3M enthält (S&P 500): Dann zählt das tatsächliche Verhältnis VIX/VIX3M (> 1 = invertiert = "
        "Stress), sonst simuliert das Framework eine Futures-Kurve aus dem VIX. Gerechnet wird kausal: Jeder "
        "Tageswert nutzt nur Daten bis zu diesem Tag, ohne Blick in die Zukunft.\n\n"
        "So liest du das Ergebnis: Die Tachonadel zeigt den letzten Tag, die Zeitreihe den Komposit mit den "
        "Stufengrenzen, die Heatmap, welcher Indikator wann anschlug. „Was folgte auf die Warnstufen?“ prüft, "
        "ob hohe Stufen tatsächlich Verluste angekündigt haben. Der **Seed** steuert das Modellrauschen, das "
        "**Divergenz-Fenster** die Tage der Divergenz-Checks, der **Prüfhorizont** den Blick nach vorn."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "term_source", "type": "select", "default": "auto", "label": "VIX-Kurve",
         "options": [{"value": "auto", "label": "Echt (VIX/VIX3M), falls im Datensatz"},
                     {"value": "model", "label": "Immer Modellkurve (aus dem VIX simuliert)"}],
         "help": "Zum Vergleich: Wie ändert sich der Score, wenn die echte Kurve durch das Modell ersetzt wird?"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Zufalls-Seed des Modells",
         "help": "GEX, TDEX, DIX und SMFI enthalten Modellrauschen. Gleicher Seed = gleiche Reihe."},
        {"name": "window", "type": "int", "default": 20, "min": 10, "max": 120, "step": 5,
         "label": "Divergenz-Fenster", "unit": "Tage",
         "help": "Letzte Tage für die Divergenz-Checks (Kurs vs. TDEX, DIX, SMFI) und die Kurveninversion."},
        {"name": "horizon", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Prüfhorizont", "unit": "Tage",
         "help": "Wie viele Handelstage nach einem Warnsignal die Kursentwicklung ausgewertet wird."},
        {"name": "show_3d", "type": "bool", "default": True, "label": "3D-Phasenraum zeigen",
         "help": "GEX-Risiko × TDEX-Risiko × Komposit (rechnet die Modellierung ein zweites Mal)."},
    ],
    "covers": [
        "src.analysis.early_warning.analyze_crash_risk",
        "src.analysis.early_warning.compute_early_warning_coordinates",
        "src.analysis.early_warning.CompositeEarlyWarningSystem",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.simulate_all_indicators",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.compute_composite",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.get_risk_assessment",
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

WEIGHT_LABEL = {"gex": "GEX · 35 %", "tdex": "TDEX · 25 %", "vix_term": "VIX-Kurve · 20 %",
                "dix": "DIX · 10 %", "smfi": "SMFI · 10 %"}


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
    label = (lambda i: x[i]) if data.dates else (lambda i: f"Tag {i}")
    win = int(min(params["window"], n))
    h = int(min(params["horizon"], n // 3))

    vix_real = data.vix is not None
    vix = np.asarray(data.vix, dtype=float) if vix_real else np.asarray(generate_vix_from_returns(r), dtype=float)
    use_real_curve = params["term_source"] == "auto" and vix_real and data.vix3m is not None
    vix3m = np.asarray(data.vix3m, dtype=float) if use_real_curve else None
    prices = np.asarray(data.prices, dtype=float) if data.prices is not None else 100 * np.exp(np.cumsum(r))

    out.progress("Modelliere die fünf Indikatoren …", 0.1)
    np.random.seed(int(params["seed"]))
    res = analyze_crash_risk(r, vix, verbose=False, causal=True, vix3m=vix3m)
    ts = res["time_series"]
    comp = np.asarray(ts["composite_score"], dtype=float)
    scores = {k: np.asarray(ts[k + "_score"], dtype=float) for k in KEYS}

    ews = CompositeEarlyWarningSystem()
    th = ews.thresholds
    last = ews.compute_composite(*(float(scores[k][-1]) for k in KEYS))
    level = last["level"]
    status, level_de = LEVEL[level]
    action_de = ACTION.get(last["action"], last["action"])
    ratio = np.asarray(ts["vix_m1"], dtype=float) / np.asarray(ts["vix_m2"], dtype=float)
    ratio_name = "VIX/VIX3M" if use_real_curve else "M1/M2 (Modell)"

    # ---- headline ------------------------------------------------------------------------------------
    i_max = int(np.argmax(comp))
    out.metric("Komposit (letzter Tag)", comp[-1], "num", digits=0, unit="/ 100", status=status,
               hint=f"{level_de} · {action_de}")
    out.metric(ratio_name, ratio[-1], "num", digits=3,
               hint="> 1 = invertiert (Backwardation, Stress)" if use_real_curve
               else "simulierte Futures-Kurve; > 1 = invertiert",
               status="serious" if ratio[-1] > 1 else None)
    out.metric("Höchster Komposit", comp[i_max], "num", digits=0, unit="/ 100", status=LEVEL[_level_of(comp[i_max], th)][0],
               hint=f"am {label(i_max)}")
    out.metric("Tage ≥ 50 (erhöht)", float(np.mean(comp >= th["elevated"])), "pct", digits=1)
    out.metric("Tage ≥ 75 (hoch)", float(np.mean(comp >= th["high"])), "pct", digits=1)

    out.set_verdict(status, f"Letzter Stand: {level_de} – {comp[-1]:.0f} von 100".replace(".", ","),
                    f"{action_de}. {assessment_de(ews.get_risk_assessment(float(comp[-1])))} "
                    "Modellwert aus Renditen und VIX – kein Live-Signal aus Options- oder Dark-Pool-Daten.")

    out.note("**Modellierte Indikatoren:** GEX, TailDex, DIX und SMFI brauchen proprietäre Options- bzw. "
             "Dark-Pool-Daten. Das Framework modelliert sie hier aus Renditen und VIX – das sind illustrative "
             "Verläufe, keine echten Marktmessungen.", "warning")
    if use_real_curve:
        curve = "**VIX-Kurve echt:** Verhältnis VIX/VIX3M aus dem Datensatz (30-Tage- zu 3-Monats-Volatilität)."
    elif vix_real and data.vix3m is not None:
        curve = "**VIX-Kurve modelliert** (so gewählt): M1/M2 werden aus dem VIX simuliert, VIX3M bleibt unbenutzt."
    else:
        curve = "**VIX-Kurve modelliert:** Der Datensatz hat kein VIX3M, M1/M2 werden aus dem VIX simuliert."
    if not vix_real:
        curve += " Auch der VIX fehlt – er wird aus den Renditen abgeleitet (`generate_vix_from_returns`)."
    out.note(curve + " Alle Werte sind kausal: Tag *t* nutzt nur Daten bis *t* (`causal=True`).", "info")

    # ---- gauge + contributions --------------------------------------------------------------------
    out.chart(C.gauge("Komposit am letzten Tag", float(comp[-1]), thresholds=(th["elevated"], th["high"], th["extreme"]),
                      labels=("Normal", "Erhöht", "Hoch", "Extrem"),
                      subtitle=f"{label(n - 1)} · Stufe {level_de}: {action_de}"))
    cats = [WEIGHT_LABEL[k] for k in KEYS]
    out.chart(C.xy(
        "Woraus sich der Score zusammensetzt",
        [C.trace("Score des Indikators (0–100)", cats, [last["components"][k]["score"] for k in KEYS],
                 type="bar", color=1),
         C.trace("Beitrag zum Komposit (Punkte)", cats, [last["components"][k]["contribution"] for k in KEYS],
                 type="bar", color=2)],
        x=C.axis("", type="category"), y=C.axis("Punkte", range=(0, 100)),
        subtitle="Beitrag = Score × Gewicht; die Beiträge ergeben zusammen den Komposit.",
    ))

    # ---- composite + price with warning phases ----------------------------------------------------
    out.chart(C.xy(
        "Komposit-Score über die Zeit",
        [C.trace("Komposit", x, comp, color=1, width=1.2)],
        x=xa, y=C.axis("Score", range=(0, 100)),
        hlines=[C.hline(th["elevated"], "Erhöht", "warning"), C.hline(th["high"], "Hoch", "serious"),
                C.hline(th["extreme"], "Extrem", "critical")],
        subtitle="Gestrichelte Linien = Stufengrenzen des Frameworks (50 / 75 / 90).",
    ))
    episodes = _episodes(comp >= th["high"])
    shown = sorted(episodes, key=lambda e: comp[e[0]:e[1] + 1].max(), reverse=True)[:40]
    out.chart(C.xy(
        "Kurs mit Warnphasen (Komposit ≥ 75)",
        [C.trace("Kurs" if data.prices is not None else "Index aus Renditen (Start 100)", x, prices, color=1,
                 width=1.2)],
        x=xa, y=C.axis("Kurs", type="log"),
        bands=[C.band(x[a], x[min(b + 1, n - 1)], "", "serious") for a, b in shown],
        subtitle="Rot hinterlegt: Tage mit hoher Warnstufe. Kamen sie vor oder während der Einbrüche?",
    ))

    # ---- which indicator fired when ---------------------------------------------------------------
    step = max(1, int(np.ceil(n / 160)))
    starts = np.arange(0, n, step)
    rows = [WEIGHT_LABEL[k] for k in KEYS] + ["Komposit"]
    z = [[float(np.mean(s[a:a + step])) for a in starts] for s in [scores[k] for k in KEYS] + [comp]]
    out.chart(C.heatmap(
        "Welcher Indikator schlug wann an?", [x[a] for a in starts], rows, z,
        x=xa if data.dates else C.axis("Handelstag", fmt="int"), y=C.axis("", type="category"),
        z=C.axis("Score"), scale="risk",
        subtitle=f"Mittlerer Score je {step} Handelstage (0 = ruhig, 100 = maximale Warnung).", height=320,
    ))

    if params["show_3d"]:
        out.progress("Phasenraum …", 0.7)
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
        traces.append(C.path3d("Letzter Tag", [X[-1] * 100], [Y[-1] * 100], [Z[-1] * 100], color="ink",
                               mode="markers", size=8))
        out.chart(C.scatter3d(
            "Frühwarn-Phasenraum", traces,
            x=C.axis("GEX-Risiko (Score)"), y=C.axis("TDEX-Risiko (Score)"), z=C.axis("Komposit"),
            subtitle="Jeder Punkt ist ein Tag. Ruhige Phasen liegen unten links, Stress oben rechts.",
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
    base = "modelliert aus Renditen"
    table_rows = [
        [NAMES["gex"], f"{de(g, 2, sign=True)} Mrd. $", f"{GEX_REGIME[gres['regime']]}: "
         f"{GEX_DESCRIPTION.get(gdesc, gdesc)}", base],
        [NAMES["tdex"], de(td, 1), TDEX_LEVEL[tres["level"]], "modelliert aus Renditen + VIX"],
        [NAMES["vix_term"], f"{ratio_name} {de(ratio[-1], 3)}", STRUCTURE[cur["structure"]],
         "echt: VIX/VIX3M" if use_real_curve else "modelliert aus VIX"],
        [NAMES["dix"], f"{de(dx, 1)} %", DIX_LEVEL[dres["level"]], base],
        [NAMES["smfi"], de(sm, 1, sign=True), f"20-Tage-Trend {de(sm_trend, 1, sign=True)}", base],
    ]
    for row, k in zip(table_rows, KEYS):
        c = last["components"][k]
        row[3:3] = [c["score"], c["weight"], c["contribution"]]
    table_rows.append(["Komposit", "", f"{level_de} · {action_de}", last["composite_score"], 1.0,
                       last["composite_score"], ""])
    out.table(f"Letzter Tag ({label(n - 1)}): die fünf Indikatoren",
              ["Indikator", "Wert", "Einordnung", "Score", "Gewicht", "Beitrag", "Datenbasis"], table_rows,
              formats=["text", "text", "text", "num", "pct", "num", "text"],
              note="Einordnung nach den Snapshot-Regeln des Frameworks (calculate_*). Score und Beitrag in Punkten (0–100).")

    # ---- divergences (last `win` days) ------------------------------------------------------------
    div = [("Kurs vs. TailDex", TailDex().detect_divergence(prices, ts["tdex"], window=win),
            "Kurs steigt, TDEX steigt mit: Smart Money kauft Absicherung in die Stärke"),
           ("Kurs vs. Dark Index", DarkIndex().detect_divergence(prices, ts["dix"], window=win),
            "Kurs steigt, DIX fällt: Distribution · umgekehrt: Akkumulation"),
           ("Kurs vs. Smart Money Flow", SmartMoneyFlowIndex().detect_divergence(prices, ts["smfi"], window=win),
            "Kurs auf Hoch, SMFI fällt: Profis verkaufen in die Rally")]
    drows = [[name, "ja" if d.get("divergence") else "nein", DIVERGENCE.get(d.get("type", "NONE"), "keine"),
              WARNING_LEVEL.get(d.get("warning_level", "NEUTRAL"), "–"), meaning] for name, d, meaning in div]
    drows.append(["VIX-Kurve kippt in Backwardation", "ja" if inv.get("inversion_event") else "nein",
                  STRUCTURE[cur["structure"]], "hoch" if inv.get("inversion_event") else "niedrig",
                  f"{ratio_name} zuletzt {de(inv.get('current_ratio', ratio[-1]), 3)}, "
                  f"Spanne {de(inv.get('min_ratio', ratio[-1]), 3)}–{de(inv.get('max_ratio', ratio[-1]), 3)}"])
    out.table(f"Divergenz-Checks der letzten {win} Tage", ["Prüfung", "Signal", "Typ", "Warnstufe", "Bedeutung"],
              drows, formats=["text", "text", "text", "text", "text"],
              note="TDEX, DIX und SMFI sind modelliert – eine Divergenz zeigt hier die Logik des Frameworks, "
                   "keine echte Positionierung.")

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
    erows.append(["Alle Tage", int(n - h), 1.0, float(np.expm1(np.mean(fwd))), float(np.mean(worst <= np.log(0.95))),
                  float(np.expm1(worst.min()))])
    out.table(f"Was folgte auf die Warnstufen? (nächste {h} Handelstage)",
              ["Stufe am Tag t", "Tage", "Anteil", f"Ø Rendite {h} T", "Anteil mit −5 % im Tief", "Tiefster Rückgang"],
              erows, formats=["text", "int", "pct", "pct", "pct", "pct"],
              note="Ausgewertet ab dem Folgetag. Liegen die Werte hoher Stufen nicht klar über „Alle Tage“, hatte "
                   "der modellierte Score hier keine Vorlaufkraft – er reagiert dann eher, als dass er warnt.")

    if episodes:
        top = sorted(episodes, key=lambda e: comp[e[0]:e[1] + 1].max(), reverse=True)[:8]
        prow = []
        for a, b in sorted(top):
            peak = a + int(np.argmax(comp[a:b + 1]))
            after = min(b + h, n - 1)
            prow.append([label(a), label(b), b - a + 1, float(comp[peak]), float(prices[b] / prices[max(a - 1, 0)] - 1),
                         float(prices[after] / prices[b] - 1) if after > b else None])
        out.table("Stärkste Warnphasen (Komposit ≥ 75)",
                  ["Beginn", "Ende", "Tage", "Spitze", "Kurs in der Phase", f"Kurs {h} T danach"], prow,
                  formats=["text", "text", "int", "num", "pct", "pct"],
                  note="Phasen mit weniger als 6 Tagen Abstand sind zusammengefasst.")
