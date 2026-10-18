"""Fear & Greed: the framework's composite sentiment score from VIX, put/call ratio, breadth and momentum."""

import numpy as np

from src.analysis.sentiment import CompositeSentimentIndicator, PutCallRatioAnalyzer

from .. import charts as C

SPEC = {
    "id": "fear-greed",
    "title": "Angst-und-Gier-Rechner",
    "group": "warning",
    "order": 40,
    "summary": "Verdichtet VIX, Put/Call-Ratio, Marktbreite und Momentum zu einem Stimmungswert von −100 (extreme "
               "Angst) bis +100 (extreme Gier) – und zeigt, was er für das Crash-Risiko bedeutet.",
    "description": (
        "Der `CompositeSentimentIndicator` des Frameworks übersetzt vier Marktablesungen in Teil-Scores von "
        "−100 bis +100 und mittelt sie gewichtet: **VIX** 30 % (neutral bei 20, 5 Punkte je VIX-Punkt), "
        "**Put/Call-Ratio** 25 % (viele Puts = Angst), **Marktbreite** 25 % (Anteil der Aktien über ihrer "
        "50-Tage-Linie, neutral bei 50 %) und **Momentum** 20 % (Indexrendite der letzten Wochen, 10 Punkte je "
        "Prozent). Über +25 herrscht Gier, unter −25 Angst, jenseits von ±75 das Extrem.\n\n"
        "Daraus leitet das Framework ein **Tail-Risiko-Signal** ab – konträr gedacht: Über +50 mahnt es zur "
        "Wachsamkeit, über +75 (extreme Gier = Sorglosigkeit) meldet es erhöhtes Crash-Risiko. Unter −50 "
        "erwartet es mehr Schwankung, unter −75 läuft der Ausverkauf schon, und eine Wende kann nahe sein.\n\n"
        "Die Put/Call-Ratio wertet der `PutCallRatioAnalyzer` ohne Historie mit festen Schwellen (0,6 / 0,7 / "
        "0,85 / 1,0) aus, mit mindestens 20 Werten Historie über ihr Perzentil – dann gibt es auch ein "
        "Konträr-Signal. Hier kannst du eine simulierte Jahres-Historie zuschalten (Seed) – sie ist "
        "illustrativ und verschiebt den Put/Call-Teil-Score. Das Was-wäre-wenn-Diagramm variiert eine Eingabe und "
        "hält die anderen fest."
    ),
    "needs": [],
    "params": [
        {"name": "vix", "type": "float", "default": 14.5, "min": 5.0, "max": 90.0, "step": 0.1, "label": "VIX"},
        {"name": "put_call", "type": "float", "default": 0.65, "min": 0.3, "max": 2.0, "step": 0.01,
         "label": "Put/Call-Ratio (Aktien)", "help": "Put- ÷ Call-Volumen; über 1 = mehr Absicherung als Wetten auf Anstieg."},
        {"name": "breadth", "type": "float", "default": 58.0, "min": 0.0, "max": 100.0, "step": 1.0,
         "label": "Marktbreite", "unit": "%", "help": "Anteil der Aktien über ihrer 50-Tage-Linie."},
        {"name": "momentum", "type": "float", "default": 1.5, "min": -30.0, "max": 30.0, "step": 0.1,
         "label": "Momentum", "unit": "%", "help": "Indexrendite der letzten Wochen in Prozent."},
        {"name": "history", "type": "select", "default": "none", "label": "Put/Call-Historie",
         "options": [{"value": "none", "label": "Keine Historie (feste Schwellen)"},
                     {"value": "sim", "label": "Simulierte Jahres-Historie (Perzentil-Modus)"}],
         "help": "Mit Historie bewertet das Framework die Put/Call-Ratio relativ (Perzentil) und gibt Konträr-Signale."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Seed der Historie", "show_if": {"history": ["sim"]}},
        {"name": "what_if", "type": "select", "default": "vix", "label": "Was-wäre-wenn für",
         "options": [{"value": "vix", "label": "VIX"}, {"value": "put_call", "label": "Put/Call-Ratio"},
                     {"value": "breadth", "label": "Marktbreite"}, {"value": "momentum", "label": "Momentum"}]},
    ],
    "covers": [
        "src.analysis.sentiment.CompositeSentimentIndicator",
        "src.analysis.sentiment.CompositeSentimentIndicator.compute_composite",
        "src.analysis.sentiment.PutCallRatioAnalyzer",
        "src.analysis.sentiment.PutCallRatioAnalyzer.add_data",
        "src.analysis.sentiment.PutCallRatioAnalyzer.fear_greed_indicator",
        "src.analysis.sentiment.PutCallRatioAnalyzer.contrarian_signal",
    ],
    "cost": "fast",
}

LEVEL = {"EXTREME_GREED": "Extreme Gier", "GREED": "Gier", "NEUTRAL": "Neutral", "FEAR": "Angst",
         "EXTREME_FEAR": "Extreme Angst", "UNKNOWN": "Unbekannt"}
TAIL = {  # tail-risk level -> (status, German label)
    "NORMAL": ("good", "Normal"), "MODERATE": ("warning", "Moderat"), "ELEVATED": ("serious", "Erhöht"),
    "HIGH": ("serious", "Hoch"), "EXTREME": ("critical", "Extrem"),
}
TEXT = {
    "Complacency high - crash risk elevated": "Hohe Sorglosigkeit – das Crash-Risiko ist erhöht",
    "Bullish sentiment - monitor for extremes": "Optimistische Stimmung – auf Übertreibungen achten",
    "Panic selling - may be near capitulation": "Panikverkäufe – womöglich nahe der Kapitulation",
    "Fear rising - volatility expected": "Die Angst steigt – mehr Schwankung zu erwarten",
    "Neutral sentiment": "Die Stimmung ist nicht extrem (zwischen −50 und +50)",
    "Reduce risk exposure": "Risiko reduzieren", "Maintain hedges": "Absicherungen halten",
    "Prepare for reversal": "Auf eine Wende vorbereiten", "Increase hedges": "Absicherungen erhöhen",
    "Normal positioning": "Normale Positionierung",
    "Extreme fear often marks bottoms": "Extreme Angst markiert oft Tiefs",
    "Extreme greed often marks tops": "Extreme Gier markiert oft Hochs",
}
CONTRA = {"CONTRARIAN_BULLISH": "Konträr bullisch", "CONTRARIAN_BEARISH": "Konträr bärisch", "NEUTRAL": "Neutral"}
INPUTS = {"vix": ("VIX", "VIX", (5.0, 60.0)), "put_call": ("Put/Call", "Put/Call-Ratio", (0.4, 1.4)),
          "breadth": ("Marktbreite", "Marktbreite (%)", (0.0, 100.0)),
          "momentum": ("Momentum", "Momentum (%)", (-15.0, 15.0))}


def _de(x, digits=1, sign=False):
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace(".", ",").replace("-", "−")


def _history(seed, n=252):
    """An illustrative year of equity put/call ratios: AR(1) around 0.8 (sd about 0.11), clipped to [0.4, 1.6]."""
    rng = np.random.default_rng(int(seed))
    pc = np.empty(n)
    pc[0] = 0.8
    for t in range(1, n):
        pc[t] = 0.8 + 0.85 * (pc[t - 1] - 0.8) + rng.normal(0, 0.06)
    return np.clip(pc, 0.4, 1.6)


def _indicator(pc_hist):
    ind = CompositeSentimentIndicator()
    if pc_hist is not None:
        ind.pc_analyzer = PutCallRatioAnalyzer()
        for v in pc_hist:
            ind.pc_analyzer.add_data(equity=float(v))
    return ind


def run(params, data, out):
    hist = _history(params["seed"]) if params["history"] == "sim" else None
    hist_today = np.append(hist, params["put_call"]) if hist is not None else None
    ind = _indicator(hist_today)
    inputs = {k: float(params[k]) for k in ("vix", "put_call", "breadth", "momentum")}
    res = ind.compute_composite(**inputs)
    comp = float(res["composite"])
    tail = res["tail_risk_signal"]
    status, tail_de = TAIL[tail["level"]]
    level_de = LEVEL[res["level"]]

    out.set_verdict(status, f"{level_de} ({_de(comp, 0, sign=True)}) · Tail-Risiko {tail_de}",
                    f"{TEXT.get(tail['direction'], tail['direction'])}. Empfehlung des Frameworks: "
                    f"{TEXT.get(tail['action'], tail['action'])}.")
    out.metric("Stimmung", comp, "num", digits=0, hint=f"{level_de} · Skala −100 Angst … +100 Gier")
    out.metric("Tail-Risiko-Signal", tail_de, "text", hint=TEXT.get(tail["direction"], tail["direction"]), status=status)
    contra = ind.pc_analyzer.contrarian_signal()
    out.metric("Put/Call konträr", CONTRA.get(contra["signal"], contra["signal"]), "text",
               hint=(TEXT.get(contra.get("reason", ""), "") + (f" · Konfidenz {_de(contra['confidence'] * 100, 0)} %"
                                                               if contra.get("confidence") else ""))
               or ("braucht mindestens 20 Tage Put/Call-Historie" if hist is None
                   else "Put/Call im normalen Bereich der Historie"))

    # ---- sub-scores ---------------------------------------------------------------------------------
    keys = [k for k in ("vix", "put_call", "breadth", "momentum") if k in res["components"]]
    cats = [{"vix": "VIX", "put_call": "Put/Call", "breadth": "Breite", "momentum": "Momentum"}[k] for k in keys] + ["Komposit"]
    vals = [float(res["components"][k]) for k in keys] + [comp]
    out.chart(C.xy(
        "Teil-Scores und Komposit",
        [C.trace("Teil-Score", cats[:-1], vals[:-1], type="bar", color=1),
         C.trace("Komposit (gewichtet)", cats[-1:], vals[-1:], type="bar", color=2)],
        x=C.axis("", type="category"), y=C.axis("Score (−100 Angst … +100 Gier)", range=(-100, 100)),
        hlines=[C.hline(75, "extreme Gier"), C.hline(25, "Gier"), C.hline(-25, "Angst"), C.hline(-75, "extreme Angst")],
        subtitle="Gewichte: VIX 30 %, Put/Call 25 %, Breite 25 %, Momentum 20 %. Linien = Stimmungsstufen.",
    ))

    # ---- what-if --------------------------------------------------------------------------------------
    name = params["what_if"]
    short, xlabel, (lo, hi) = INPUTS[name]
    cur = inputs[name]
    grid = np.linspace(min(lo, cur), max(hi, cur), 161)
    ys = []
    for v in grid:
        q = dict(inputs, **{name: float(v)})
        # compute_composite does not store anything; only a new put/call value changes the history
        sim = _indicator(np.append(hist, q["put_call"]) if hist is not None else None) if name == "put_call" else ind
        ys.append(sim.compute_composite(**q)["composite"])
    out.chart(C.xy(
        f"Was wäre, wenn sich nur {short} ändert?",
        [C.trace("Stimmung", grid, ys, color=1)],
        x=C.axis(xlabel, digits=2 if name == "put_call" else 1), y=C.axis("Stimmung", range=(-100, 100)),
        hlines=[C.hline(75, "extreme Gier"), C.hline(25, "Gier"), C.hline(-25, "Angst"),
                C.hline(-75, "extreme Angst")],
        vlines=[C.vline(cur, "heute", "ink")],
        subtitle="Alle anderen Eingaben bleiben fest. Ab ±50 meldet das Tail-Risiko-Signal mehr als „Normal“, "
                 "jenseits von ±75 das Extrem.",
    ))

    # ---- how the framework reads the put/call ratio ------------------------------------------------
    pcs = np.linspace(0.4, 1.4, 201)
    fixed = PutCallRatioAnalyzer()
    traces = [C.trace("Feste Schwellen (ohne Historie)", pcs, [fixed.fear_greed_indicator(v)["score"] for v in pcs],
                      type="step", color="muted" if hist is not None else 1, dash=hist is not None)]
    if hist is not None:
        traces.append(C.trace("Perzentil der Historie", pcs,
                              [ind.pc_analyzer.fear_greed_indicator(v)["score"] for v in pcs], color=1))
    out.chart(C.xy(
        "So übersetzt das Framework die Put/Call-Ratio",
        traces, x=C.axis("Put/Call-Ratio", digits=2), y=C.axis("Teil-Score", range=(-100, 100)),
        vlines=[C.vline(inputs["put_call"], "heute", "ink")],
        subtitle="Viele Puts = Angst (negativ), wenige = Gier (positiv). Mit Historie zählt die relative Lage.",
    ))
    if hist is not None:
        days = list(range(-len(hist), 0)) + [0]
        p10, p90 = np.percentile(hist_today, [10, 90])
        out.chart(C.xy(
            "Simulierte Put/Call-Historie",
            [C.trace("Put/Call (simuliert)", days[:-1], hist, color=1, width=1.2),
             C.trace("heute (deine Eingabe)", [0], [inputs["put_call"]], type="scatter", color="ink", size=10)],
            x=C.axis("Handelstage bis heute", fmt="int"), y=C.axis("Put/Call-Ratio"),
            hlines=[C.hline(p90, "90. Perzentil: konträr bullisch", "serious"),
                    C.hline(p10, "10. Perzentil: konträr bärisch", "warning")],
            subtitle="Liegt die heutige Ratio jenseits dieser Linien, gibt der Analyzer ein Konträr-Signal.",
        ))
        out.note("Die Put/Call-Historie ist simuliert (AR(1)-Prozess um 0,8, Seed wählbar) und dient nur dazu, den "
                 "Perzentil-Modus und das Konträr-Signal des Frameworks vorzuführen.", "info")

    tot = sum(ind.weights[k] for k in keys)
    rows = [[INPUTS[k][1], _de(inputs[k], 2 if k == "put_call" else 1), float(res["components"][k]),
             ind.weights[k] / tot, float(res["components"][k]) * ind.weights[k] / tot] for k in keys]
    rows.append(["Komposit", "", comp, 1.0, comp])
    out.table("Zusammensetzung", ["Eingabe", "Wert", "Teil-Score", "Gewicht", "Beitrag"], rows,
              formats=["text", "text", "num", "pct:0", "num"],
              note="VIX: 5 × (20 − VIX); Breite: 2 × (Breite − 50); Momentum: 10 × Rendite in %; alle auf ±100 begrenzt.")
