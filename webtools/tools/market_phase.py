"""Market phase: rule-based classifier (STABLE/TRENDING/CRITICAL/CRASH/RECOVERY) and phase-space trajectory."""

import numpy as np

from src.physics.phase_transitions import MarketPhaseClassifier, compute_phase_space_coordinates

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "market-phase",
    "title": "Marktphase: stabil, Trend, kritisch, Crash",
    "group": "regime",
    "order": 20,
    "summary": "In welcher Phase der Markt gerade steckt – fünf transparente Regeln, ihre Zeitleiste und "
               "die Bahn des Markts durch einen 3D-Phasenraum.",
    "description": (
        "Der Klassifikator übersetzt die Physik der Phasenübergänge in fünf einfache Regeln, die der Reihe nach "
        "geprüft werden: **Crash** (Volatilität in den obersten 5 % der Geschichte und im Schnitt mehr als −2 % pro Tag), "
        "**Kritisch** (die Frühwarnsignale Autokorrelation und Varianz steigen beide deutlich), **Erholung** "
        "(Volatilität im obersten Zehntel, aber steigende Kurse), **Trend** (starke Autokorrelation und mehr als "
        "1 % Bewegung pro Tag) – sonst **Stabil**. Die Tabelle zeigt für jede Regel die aktuellen Werte.\n\n"
        "So liest du die Diagramme: Die Zeitleiste klassifiziert jeden Stichtag nur mit den Daten, die damals "
        "verfügbar waren. Der Phasenraum zeigt drei Koordinaten je Tag: **X** ist ein Varianzverhältnis "
        "(„Suszeptibilität“: Varianz des kumulierten Kurses im Fenster ÷ Varianz der Tagesrenditen – groß, wenn "
        "sich die Tage zu einer Bewegung in eine Richtung aufaddieren), **Y** der Betrag der Autokorrelation der "
        "Tagesrenditen (Ordnungsparameter, Herdentrieb), **Z** = 1/(X + 0,1) ein Maß für den Abstand zur "
        "Kritikalität. Alle drei sind auf ihr Maximum normiert.\n\n"
        "Ehrlich gesagt: Z ist eine feste Funktion von X, die Bahn liegt deshalb auf einer gekrümmten Fläche – die "
        "eigentliche Information steckt in X und Y. Die Crash-Regel ist sehr streng (−2 % pro Tag über das ganze "
        "Fenster) und greift praktisch nur in historischen Extremphasen; „Stabil“ ist die Restkategorie. Die "
        "AC(1) des trendbereinigten Log-Kurses liegt fast immer über 0,2 (schon eine Irrfahrt erreicht rund 0,9) – "
        "die Trendregel hängt deshalb praktisch nur an der Durchschnittsrendite. Die EWS-Warnstufe testet den "
        "Trend über die **ganze** Reihe, nicht nur über die letzten Tage wie das Werkzeug „Kritisches Verlangsamen“."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 20, "max": 200, "step": 5,
         "label": "Fenster", "unit": "Tage",
         "help": "für Volatilität, Durchschnittsrendite und Phasenraum. Die EWS-Warnstufe nutzt intern immer 50 Tage."},
        {"name": "timeline", "type": "int", "default": 12, "min": 0, "max": 60, "step": 1,
         "label": "Stichtage in der Zeitleiste", "help": "0 = aus. Jeder Stichtag kostet eine volle Klassifikation."},
    ],
    "covers": [
        "src.physics.phase_transitions.MarketPhaseClassifier",
        "src.physics.phase_transitions.MarketPhaseClassifier.classify",
        "src.physics.phase_transitions.compute_phase_space_coordinates",
    ],
    "cost": "medium",
}

PHASES = {
    "STABLE": ("Stabil", "good"),
    "TRENDING": ("Trend", 1),
    "RECOVERY": ("Erholung", 2),
    "CRITICAL": ("Kritisch", "serious"),
    "CRASH": ("Crash", "critical"),
}
MIN_EXTRA = 20  # days of phase-space coordinates beyond the 2·window warm-up
TIMELINE_OBS = 40_000  # sum of history lengths classified for the timeline (bounded runtime)
EWS_DE = {"NORMAL": "Normal", "MODERATE": "Moderat", "ELEVATED": "Erhöht", "CRITICAL": "Kritisch"}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _date(d):
    """ISO date -> German date for hover texts; trading-day indices become 'Tag n'."""
    return f"{d[8:10]}.{d[5:7]}.{d[:4]}" if isinstance(d, str) and len(d) >= 10 else f"Tag {d}"


def run(params, data, out):
    r = data.returns
    n, w = data.n, params["window"]
    if n < 2 * w + MIN_EXTRA:
        raise ToolError(f"Für ein Fenster von {w} Tagen braucht der Klassifikator mindestens {2 * w + MIN_EXTRA} "
                        f"Beobachtungen (vorhanden: {n}). Verkleinere das Fenster auf höchstens "
                        f"{(n - MIN_EXTRA) // 2} Tage.")

    clf = MarketPhaseClassifier()
    res = clf.classify(r, window=w)
    phase = res["phase"]
    name, color = PHASES[phase]
    vol_pct = float(res["volatility_percentile"])
    recent = r[-w:]
    mean_ret, vol_ann = float(np.mean(recent)), float(np.std(recent) * np.sqrt(252))
    ac1 = float(res["autocorrelation"])
    ews_level = res["ews"]["warning_level"]

    status = color if isinstance(color, str) else "warning"
    out.metric("Aktuelle Phase", name, "text", status=status)
    out.metric("Konfidenz", res["confidence"], "pct", digits=0,
               hint="feste Regel-Formel, keine statistische Sicherheit – siehe Tabelle")
    out.metric("Volatilitäts-Perzentil", vol_pct / 100, "pct", digits=0,
               hint=f"Volatilität der letzten {w} Tage im Vergleich zur ganzen Geschichte",
               status="serious" if vol_pct > 95 else "warning" if vol_pct > 90 else None)
    out.metric(f"Volatilität p. a. ({w} Tage)", vol_ann, "pct", digits=1)
    out.metric(f"Ø-Rendite je Tag ({w} Tage)", mean_ret, "pct", digits=3)
    out.metric("AC(1) des Log-Kurses", ac1, "num", hint="letzter EWS-Wert (Trendregel: > 0,2)")
    out.metric("Streuung um den Trend", np.sqrt(res["susceptibility"]), "pct",
               hint="√ EWS-Varianz des trendbereinigten Log-Kurses (im Klassifikator „Suszeptibilität“)")
    out.metric("EWS-Warnstufe", EWS_DE.get(ews_level, ews_level), "text",
               hint="Kendall-τ-Trend von AC(1) und Varianz über die ganze Reihe")

    def yes(c):
        return "ja" if c else "nein"

    pct_txt, ret_txt = f"{_de(vol_pct, 0)}. Perzentil", f"{_de(100 * mean_ret, 2)} %/Tag"
    rules = [
        ("Crash", "Vol-Perzentil > 95 UND Ø-Rendite < −2 %/Tag", f"{pct_txt} · {ret_txt}",
         vol_pct > 95 and mean_ret < -0.02, "Vol-Perzentil"),
        ("Kritisch", "EWS-Warnstufe = Kritisch", EWS_DE.get(ews_level, ews_level), ews_level == "CRITICAL", "80 % (fest)"),
        ("Erholung", "Vol-Perzentil > 90 UND Ø-Rendite > 0", f"{pct_txt} · {ret_txt}", vol_pct > 90 and mean_ret > 0,
         "70 % (fest)"),
        ("Trend", "AC(1) > 0,2 UND |Ø-Rendite| > 1 %/Tag", f"AC(1) {_de(ac1)} · {ret_txt}",
         ac1 > 0.2 and abs(mean_ret) > 0.01, "AC(1)"),
        ("Stabil", "keine der Regeln oben", "–", True, "100 % − Vol-Perzentil"),
    ]
    rows, hit = [], False
    for label, rule, value, cond, conf_rule in rules:
        mark = yes(cond)
        if cond and not hit:
            mark, hit = "ja → Ergebnis", True
        rows.append([label, rule, value, mark, conf_rule])
    out.table("Regeln (erste erfüllte Regel gilt)", ["Phase", "Regel", "aktuell", "erfüllt", "Konfidenz"], rows,
              formats=["text", "text", "text", "text", "text"],
              note=f"Fenster {w} Tage. Die EWS-Warnstufe rechnet intern immer mit 50 Tagen und σ = 100 und testet "
                   "den Trend über die ganze Reihe.")

    x, xa = index_axis(data)

    # Timeline: classify each cut-off date with the data available then
    k = params["timeline"]
    if k > 0:
        start = max(2 * w + MIN_EXTRA, 250 if n > 500 else 2 * w + MIN_EXTRA)
        # each cut-off re-classifies its whole history: bound the total work for long series
        k_eff = int(max(3, min(k, TIMELINE_OBS // max(1, (start + n) // 2))))
        ends = np.linspace(start, n, min(k, k_eff)).astype(int)
        # add the day with the highest rolling volatility: what did the rules say at peak stress?
        roll = np.std(np.lib.stride_tricks.sliding_window_view(r, w), axis=1)
        peak = int(np.argmax(roll)) + w
        if peak >= start:
            ends = np.append(ends, peak)
        ends = np.unique(ends)
        points = []
        for i, end in enumerate(ends):
            out.progress(f"Zeitleiste: Stichtag {i + 1} von {len(ends)} …", (i + 1) / (len(ends) + 2))
            c = clf.classify(r[:end], window=w)
            points.append((end - 1, c["phase"], float(c["volatility_percentile"]) / 100, float(c["confidence"])))
        traces = [C.trace("Vol-Perzentil", [x[p[0]] for p in points], [p[2] for p in points], color="muted", width=1)]
        for key, (label, col) in PHASES.items():
            sel = [p for p in points if p[1] == key]
            if sel:
                traces.append(C.trace(label, [x[p[0]] for p in sel], [p[2] for p in sel], type="scatter",
                                      color=col, size=11, decimate_to=None,
                                      text=[f"{label} · Konfidenz {_de(100 * p[3], 0)} %" for p in sel]))
        stressed = [p for p in points if p[1] == "STABLE" and p[2] > 0.9]
        if stressed:
            p0 = max(stressed, key=lambda p: p[2])
            out.note(f"Am {_date(x[p0[0]])} lag die Volatilität im {_de(100 * p0[2], 0)}. Perzentil – trotzdem "
                     f"meldet der Klassifikator „Stabil“ (Konfidenz {_de(100 * p0[3], 0)} %). Die Crash-Regel verlangt "
                     "im Schnitt −2 % pro Tag, die Erholungsregel steigende Kurse; greift keine, bleibt nur die "
                     "Restkategorie.", "warning")
        out.chart(C.xy(
            "Zeitleiste der Phasen",
            traces, x=xa, y=C.axis("Volatilitäts-Perzentil", fmt="pct", digits=0, range=(-0.03, 1.1)),
            hlines=[C.hline(0.9), C.hline(0.95, "90 % / 95 %")], height=390,
            subtitle="Jeder Punkt ist eine Klassifikation nur mit den Daten bis zu diesem Tag – inklusive des "
                     "Tages mit der höchsten Volatilität."
                     + (f" Wegen der langen Reihe auf {k_eff} Stichtage begrenzt." if k_eff < k else ""),
        ))

    # Phase space
    X, Y, Z = compute_phase_space_coordinates(r, window=w)
    valid = np.arange(2 * w, n)
    labels = [_date(x[i]) for i in valid]
    recent_n = min(250, len(valid) // 3)
    traces3d = [
        C.path3d("Verlauf", X[valid], Y[valid], Z[valid], color="muted", text=labels),
        C.path3d(f"letzte {recent_n} Tage", X[valid[-recent_n:]], Y[valid[-recent_n:]], Z[valid[-recent_n:]],
                 color=1, mode="lines+markers", size=2, text=labels[-recent_n:]),
        C.path3d("heute", [X[-1]], [Y[-1]], [Z[-1]], color=2, mode="markers", size=7, text=[labels[-1]]),
    ]
    xs = [x[i] for i in valid]
    out.chart(C.xy(
        "Phasenraum X: Suszeptibilität",
        [C.trace("X", xs, X[valid], color=1, width=1.5)],
        x=xa, y=C.axis("X, normiert auf das Maximum", range=(0, 1.02)), height=260,
        subtitle=f"Rollierend über {w} Tage. Spitzen: Die Tagesbewegungen addieren sich zu einem Trend oder "
                 "Ausverkauf. Z = 1/(X + 0,1) ist eine feste Funktion davon und fehlt deshalb hier.",
    ))
    out.chart(C.xy(
        "Phasenraum Y: Ordnung |AC(1)|",
        [C.trace("Y", xs, Y[valid], color=1, width=1)],
        x=xa, y=C.axis("Y, normiert auf das Maximum", range=(0, 1.02)), height=260,
        subtitle="Betrag der Lag-1-Autokorrelation der Tagesrenditen. Hoch: Tage hängen voneinander ab "
                 "(Herdentrieb oder schnelle Umkehr).",
    ))

    out.chart(C.scatter3d(
        "Phasenraum-Bahn des Markts", traces3d,
        x=C.axis("X: Suszeptibilität"), y=C.axis("Y: Ordnung |AC(1)|"), z=C.axis("Z: Abstand zur Kritikalität"),
        subtitle="Nahe an kritischen Zuständen: X groß, Z klein. Ziehen zum Drehen, Scrollen zum Zoomen.",
    ))
    if phase == "STABLE" and vol_pct > 90:
        out.note(f"„Stabil“ ist hier nur die Restkategorie: Die Volatilität liegt im {_de(vol_pct, 0)}. Perzentil, "
                 "aber weder die Crash- noch die Erholungsregel greift (die Durchschnittsrendite ist ≤ 0 bzw. nicht "
                 "unter −2 % pro Tag).", "warning")
        out.set_verdict("warning", "Stabil laut Restregel – aber ungewöhnlich hohe Volatilität",
                        "Die Regeln erkennen Stress nur bei extremen Durchschnittsrenditen.")
        return
    texts = {
        "STABLE": ("good", "Stabile Phase",
                   f"Volatilität im {_de(vol_pct, 0)}. Perzentil, keine Regel für Trend, Kritikalität oder Crash greift."),
        "TRENDING": ("warning", "Trendphase: Herdenverhalten",
                     "Starke Autokorrelation und große Tagesbewegungen in eine Richtung."),
        "RECOVERY": ("warning", "Erholung nach Stress",
                     "Volatilität im obersten Zehntel, aber die Kurse steigen im Schnitt."),
        "CRITICAL": ("serious", "Kritische Phase: Frühwarnsignale steigen",
                     "Autokorrelation und Varianz steigen beide signifikant – typisch vor Regimewechseln."),
        "CRASH": ("critical", "Crash-Phase", "Extreme Volatilität bei stark fallenden Kursen."),
    }
    level, title, text = texts[phase]
    out.set_verdict(level, title, text)
