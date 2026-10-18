"""Critical slowing down: rolling early-warning signals (AC(1), variance, skewness) with Kendall-tau trends."""

import numpy as np

from src.physics.phase_transitions import CriticalSlowingDownDetector

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "critical-slowing-down",
    "title": "Kritisches Verlangsamen (Frühwarnsignale)",
    "group": "regime",
    "order": 35,
    "summary": "Steigen Autokorrelation und Varianz systematisch an? Das klassische Vorzeichen eines Kipppunkts – "
               "mit Kendall-τ-Trendtest und Warnstufe.",
    "description": (
        "Kurz vor einem Kipppunkt erholt sich ein System immer langsamer von kleinen Störungen. Messbar wird das "
        "als **steigende Autokorrelation AC(1)** (der Markt „erinnert“ sich länger an gestern) und **steigende "
        "Varianz** (die Ausschläge werden größer). Der Detektor kumuliert dazu die Renditen zum Log-Kurs, zieht "
        "in jedem rollierenden Fenster einen Gauß-geglätteten Trend ab und misst AC(1) und Varianz des Rests. "
        "Die Schiefe kommt aus den Renditen selbst.\n\n"
        "So liest du die Diagramme: Entscheidend ist nicht das Niveau, sondern der **Trend im Analysefenster** "
        "(grau hinterlegt). Kendall-τ misst ihn: +1 heißt „steigt ständig“, 0 „kein Trend“. Daraus folgt die "
        "Warnstufe: Kritisch, wenn AC(1) **und** Varianz deutlich steigen (τ > 0,3, p < 0,05). Der "
        "Kompositindikator ist der Mittelwert der z-Werte von AC(1) und Varianz über die ganze Reihe.\n\n"
        "Parameter: Das **Fenster** bestimmt, wie viele Tage in jede Messung eingehen (kürzer = schneller, aber "
        "verrauschter). **σ** ist die Breite des Gauß-Trends; ist σ größer als das Fenster, wird praktisch nur "
        "der Mittelwert abgezogen. Das **Analysefenster** legt fest, über welche letzten Tage der Trend getestet wird."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 20, "max": 250, "step": 5,
         "label": "Rollierendes Fenster", "unit": "Tage"},
        {"name": "detrend_window", "type": "int", "default": 100, "min": 5, "max": 200, "step": 5,
         "label": "Glättungsbreite σ des Trends", "unit": "Tage",
         "help": "Gauß-Filter zur Trendbereinigung in jedem Fenster."},
        {"name": "lookback", "type": "int", "default": 500, "min": 100, "max": 20000, "step": 50,
         "label": "Analysefenster (letzte Handelstage)", "unit": "Tage",
         "help": "Über diesen Zeitraum testet Kendall-τ, ob die Signale steigen. Größer als der Datensatz = alles."},
    ],
    "covers": [
        "src.physics.phase_transitions.CriticalSlowingDownDetector",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.detrend",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.autocorrelation_lag1",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.rolling_variance",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.rolling_skewness",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.kendall_tau",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.compute_ews",
        "src.physics.phase_transitions.CriticalSlowingDownDetector._classify_warning",
    ],
    "cost": "fast",
}

LEVELS = {
    "NORMAL": ("good", "Normal", "Kein kritisches Verlangsamen erkennbar"),
    "MODERATE": ("warning", "Moderat", "Leichter Aufwärtstrend in den Frühwarnsignalen"),
    "ELEVATED": ("serious", "Erhöht", "Erhöhte Frühwarnstufe: ein Signal steigt deutlich"),
    "CRITICAL": ("critical", "Kritisch", "Kritisches Verlangsamen: AC(1) und Varianz steigen gemeinsam"),
}
MAX_TAPS = 3e9  # window x Gaussian kernel length x observations (bounded runtime in the browser)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _p(p):
    """'p < 0,001' or 'p = 0,076'."""
    return "p < 0,001" if p < 0.001 else f"p = {_de(p, 3)}"


def _padded(a, w):
    out = np.asarray(a, dtype=float).copy()
    out[:w] = np.nan  # no full window yet
    return out


def run(params, data, out):
    r = data.returns
    n, w, sigma = data.n, params["window"], params["detrend_window"]
    look = min(params["lookback"], n)
    if look - w < 30:
        raise ToolError(f"Das Analysefenster ({look} Tage) ist zu kurz für ein Fenster von {w} Tagen. "
                        "Verkleinere das Fenster oder vergrößere das Analysefenster.")
    if (n + look) * w * (8 * sigma + 1) > MAX_TAPS:
        raise ToolError("Fenster × Glättungsbreite σ sind für diese Datenlänge zu rechenintensiv. "
                        "Verkleinere σ oder das Fenster.")

    det = CriticalSlowingDownDetector(window=w, detrend_window=sigma)
    full = det.compute_ews(r)                                   # charts: the whole history
    ews = det.compute_ews(r[-look:]) if look < n else full      # trend test: the analysis window
    level = ews["warning_level"]
    status, level_de, title = LEVELS[level]
    ac_tau, ac_p = ews["ac1_trend"]["tau"], ews["ac1_trend"]["p_value"]
    var_tau, var_p = ews["var_trend"]["tau"], ews["var_trend"]["p_value"]

    out.metric("Warnstufe", level_de, "text", status=status, hint="aus den Kendall-τ-Trends im Analysefenster")
    out.metric("Trend AC(1): Kendall-τ", ac_tau, "num", hint=_p(ac_p),
               status="warning" if ac_tau > 0.2 and ac_p < 0.1 else None)
    out.metric("Trend Varianz: Kendall-τ", var_tau, "num", hint=_p(var_p),
               status="warning" if var_tau > 0.2 and var_p < 0.1 else None)
    out.metric("AC(1) aktuell", full["autocorrelation"][-1], "num", hint="des trendbereinigten Log-Kurses")
    out.metric("Streuung aktuell", np.sqrt(full["variance"][-1]), "pct",
               hint="√Varianz des trendbereinigten Log-Kurses im letzten Fenster")
    out.metric("Schiefe aktuell", full["skewness"][-1], "num", hint="der Tagesrenditen im letzten Fenster")
    comp_now = full["composite"][-1]
    out.metric("Kompositindikator", comp_now, "num", hint="z-Wert, Mittel aus AC(1) und Varianz",
               status="serious" if comp_now > 2 else "warning" if comp_now > 1 else None)

    x, xa = index_axis(data)
    bands = [C.band(x[n - look], x[-1], "Analysefenster", color="muted")] if look < n else []
    meta = data.meta or {}
    if "crisis_start" in meta and "crisis_end" in meta and not data.dates:
        bands.append(C.band(int(meta["crisis_start"]), int(meta["crisis_end"]), "Krise (synthetisch)"))

    level_path = np.cumsum(r)
    trend = level_path - det.detrend(level_path)
    out.chart(C.xy(
        "Zustand: kumulierte Log-Rendite und Gauß-Trend",
        [C.trace("Kumulierte Log-Rendite", x, level_path, color=1, width=1.5),
         C.trace(f"Gauß-Trend (σ = {sigma} Tage)", x, trend, color=2, dash=True)],
        x=xa, y=C.axis("Kumulierte Log-Rendite", fmt="pct", digits=0), bands=bands, height=300,
        subtitle="So trennt die Trendbereinigung Trend und Schwankung – der Detektor macht das in jedem Fenster neu.",
    ))
    out.chart(C.xy(
        f"Autokorrelation AC(1) – rollierend über {w} Tage",
        [C.trace("AC(1)", x, _padded(full["autocorrelation"], w), color=1, width=1.5)],
        x=xa, y=C.axis("AC(1)"), bands=bands, height=280,
        subtitle=f"Kendall-τ im Analysefenster: {_de(ac_tau)} ({_p(ac_p)}). "
                 "Steigt AC(1), erholt sich der Markt langsamer.",
    ))
    out.chart(C.xy(
        f"Streuung um den Trend – rollierend über {w} Tage",
        [C.trace("√Varianz", x, np.sqrt(_padded(full["variance"], w)), color=1, width=1.5)],
        x=xa, y=C.axis("Streuung des Log-Kurses", fmt="pct", digits=1), bands=bands, height=280,
        subtitle=f"Kendall-τ der Varianz im Analysefenster: {_de(var_tau)} ({_p(var_p)}). "
                 "Steigende Streuung: die Ausschläge werden größer.",
    ))
    out.chart(C.xy(
        f"Schiefe der Tagesrenditen – rollierend über {w} Tage",
        [C.trace("Schiefe", x, _padded(full["skewness"], w), color=1, width=1.5)],
        x=xa, y=C.axis("Schiefe"), bands=bands, hlines=[C.hline(0.0)], height=280,
        subtitle="Stark negative Werte: einzelne Crash-Tage dominieren das Fenster.",
    ))
    out.chart(C.xy(
        "Kompositindikator (z-Wert)",
        [C.trace("½ z(AC1) + ½ z(Varianz)", x, _padded(full["composite"], w), color=1, width=1.5)],
        x=xa, y=C.axis("z-Wert"), bands=bands, height=280,
        hlines=[C.hline(0.0), C.hline(2.0, "+2", color="warning")],
        subtitle="Über +2: AC(1) und Varianz liegen gemeinsam weit über ihrem historischen Mittel.",
    ))

    def ok(cond):
        return "ja" if cond else "nein"

    rules = [
        ("Kritisch", "τ(AC1) > 0,3 mit p < 0,05 UND τ(Var) > 0,3 mit p < 0,05",
         ac_tau > 0.3 and ac_p < 0.05 and var_tau > 0.3 and var_p < 0.05),
        ("Erhöht", "τ(AC1) > 0,2 mit p < 0,1 ODER τ(Var) > 0,2 mit p < 0,1",
         (ac_tau > 0.2 and ac_p < 0.1) or (var_tau > 0.2 and var_p < 0.1)),
        ("Moderat", "τ(AC1) > 0,1 ODER τ(Var) > 0,1", ac_tau > 0.1 or var_tau > 0.1),
        ("Normal", "keine der Regeln oben", True),
    ]
    rows, hit = [], False
    for name, rule, cond in rules:
        mark = ok(cond)
        if cond and not hit:
            mark, hit = "ja → Ergebnis", True
        rows.append([name, rule, mark])
    out.table("Regeln der Warnstufe (erste erfüllte Regel gilt)", ["Stufe", "Regel", "erfüllt"], rows,
              formats=["text", "text", "text"],
              note=f"Analysefenster: letzte {look} Handelstage, davon {look - w} mit vollem Fenster.")
    out.note("Die p-Werte sind zu optimistisch: Benachbarte rollierende Fenster überlappen fast vollständig, "
             "die Messpunkte sind also nicht unabhängig. Achte vor allem auf die Größe von τ und darauf, ob AC(1) "
             "und Varianz gemeinsam steigen.")

    detail = (f"Kendall-τ im Analysefenster: AC(1) {_de(ac_tau)}, Varianz {_de(var_tau)} "
              f"(letzte {look} Handelstage).")
    out.set_verdict(status, title, detail)
