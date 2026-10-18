"""Rollierende Tsallis-Entropie und q: taugen sie als Frühindikator für Volatilität?"""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisEntropyAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tsallis-entropy",
    "title": "Tsallis-Entropie im Zeitverlauf",
    "group": "regime",
    "order": 40,
    "summary": "Rollierende Tsallis-Entropie und Tail-Index q neben realisierter Volatilität und VIX – mit einem "
               "ehrlichen Test, ob die Entropie künftige Turbulenz ankündigt.",
    "description": (
        "`TsallisEntropyAnalyzer.rolling_entropy` schätzt in jedem Fenster zuerst den Index **q** aus dem "
        "Hill-Tail-Index der größten |Renditen| und berechnet dann die Tsallis-Entropie S_q = (1 − Σpᵢ^q)/(q − 1) "
        "des Histogramms der Fensterrenditen. Die Idee aus der Physik: Steigende Entropie zeigt wachsende "
        "Unordnung und soll Turbulenz vorausgehen.\n\n"
        "Das Werkzeug prüft diese Behauptung an deinen Daten: Es vergleicht die Entropie (und q) mit der "
        "**zukünftigen** realisierten Volatilität der nächsten Tage, und zwar gegen die einfachste Konkurrenz – die "
        "Volatilität der Vergangenheit. Die Rangkorrelation nach Spearman ist robust gegen Ausreißer. Im "
        "Quintil-Diagramm siehst du, ob die Tage mit der höchsten Entropie wirklich die unruhigsten Folgewochen "
        "hatten.\n\n"
        "Gut zu wissen: Das Histogramm hat eine feste Zahl von Klassen über die Spannweite des Fensters. Ein "
        "einzelner Crash-Tag streckt die Spannweite und drückt alle übrigen Renditen in wenige Klassen – die Entropie "
        "**sinkt** dann eher. Die Entropie misst die Form der Verteilung, nicht ihre Breite."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 30, "max": 250, "step": 5,
         "label": "Fenster (Handelstage)"},
        {"name": "n_bins", "type": "int", "default": 25, "min": 10, "max": 100, "step": 5,
         "label": "Klassen im Histogramm"},
        {"name": "horizon", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Vorlauf-Test: Vola der nächsten … Tage"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 15000, "step": 250,
         "label": "Zeitraum (letzte Handelstage)", "help": "Begrenzt die Rechenzeit bei sehr langen Reihen."},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.rolling_entropy",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.estimate_optimal_q",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.compute_tsallis_entropy",
    ],
    "cost": "medium",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(stats.spearmanr(a[ok], b[ok])[0])


def _quintile_means(signal, target):
    ok = np.isfinite(signal) & np.isfinite(target)
    s, t = signal[ok], target[ok]
    if len(s) < 50:
        return [None] * 5
    edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
    group = np.searchsorted(edges, s, side="right")
    return [float(np.mean(t[group == g])) if np.any(group == g) else None for g in range(5)]


def run(params, data, out):
    w, bins, h = params["window"], params["n_bins"], params["horizon"]
    m = min(params["last_n"], data.n)
    r = np.asarray(data.returns[-m:], dtype=float)
    if data.n < 30 + 5 + 60:
        raise ToolError(f"Die Reihe ist mit {data.n} Renditen zu kurz: Fenster, Vorlauf-Test und genug Vergleichstage "
                        "brauchen mindestens 95 Handelstage. Wähle einen längeren Datensatz.")
    if m < w + h + 60:
        raise ToolError(f"Für ein Fenster von {w} Tagen und {h} Tage Vorlauf ist die Reihe zu kurz ({m} Tage). "
                        "Verkleinere Fenster oder Vorlauf" + (" oder vergrößere den Zeitraum." if m < data.n else "."))
    x = list(data.dates[-m:]) if data.dates else list(range(data.n - m, data.n))
    xa = C.axis("", type="date") if data.dates else C.axis("Handelstag", fmt="int")
    vix = np.asarray(data.vix[-m:], dtype=float) / 100.0 if data.vix is not None else None
    vix_name = "VIX" if data.meta.get("source") not in ("synthetic", "crisis") else "VIX (aus Renditen abgeleitet)"

    analyzer = TsallisEntropyAnalyzer()
    out.progress("Rollierende Tsallis-Entropie …", 0.1)
    S = np.asarray(analyzer.rolling_entropy(r, window=w, n_bins=bins), dtype=float)
    S[:w] = np.nan
    q = np.full(m, np.nan)
    for i in range(w, m):
        q[i] = analyzer.estimate_optimal_q(r[i - w:i])

    win = np.lib.stride_tricks.sliding_window_view(r, w)          # win[j] = r[j : j+w]
    past = np.full(m, np.nan)
    past[w:] = np.std(win[:-1], axis=1) * np.sqrt(252)              # uses r[i-w : i], like the entropy
    fwd = np.lib.stride_tricks.sliding_window_view(r, h)
    future = np.full(m, np.nan)
    future[: m - h + 1] = np.std(fwd, axis=1) * np.sqrt(252)        # r[i : i+h], strictly after the window
    future[:w] = np.nan

    # --- headline numbers
    valid = np.isfinite(S)
    s_now, q_now = float(S[-1]), float(q[-1])
    pct_s = float(np.mean(S[valid] <= s_now))
    pct_q = float(np.mean(q[valid] <= q_now))
    rho_s, rho_q, rho_p = _rho(S, future), _rho(q, future), _rho(past, future)
    out.metric("Tsallis-Entropie heute", s_now, "num", hint=f"höher als an {_de(100 * pct_s, 0)} % der Tage")
    out.metric("q heute (Hill)", q_now, "num", hint=f"höher als an {_de(100 * pct_q, 0)} % der Tage")
    out.metric("Realisierte Vola heute", float(past[-1]), "pct", digits=1, hint=f"{w} Tage bis gestern, annualisiert – wie das Entropie-Fenster")
    out.metric(f"Rangkorrelation Entropie → Vola {h} T.", rho_s, "num",
               hint="Spearman ρ mit der zukünftigen Vola")
    out.metric(f"Zum Vergleich: Vola → Vola {h} T.", rho_p, "num", hint="vergangene Vola als einfachster Prognostiker")

    # --- charts over time
    med = float(np.nanmedian(S))
    out.chart(C.xy(
        "Rollierende Tsallis-Entropie S_q",
        [C.trace("S_q", x, S, color=1, width=1.3)], x=xa, y=C.axis("S_q"),
        hlines=[C.hline(med, "Median")],
        subtitle=f"Fenster {w} Tage, {bins} Klassen, q je Fenster aus dem Hill-Schätzer.",
    ))
    out.chart(C.xy(
        "Rollierender Tsallis-Index q",
        [C.trace("q (Hill)", x, q, color=2, width=1.3)], x=xa, y=C.axis("q"),
        hlines=[C.hline(1.0, "Gauß (q = 1)"), C.hline(5 / 3, "Varianz → ∞ (q = 5/3)")],
        subtitle="Hill auf nur einem Fenster ist sehr verrauscht; Sprünge in q zeigen einzelne Extremtage im Fenster.",
    ))
    vol_traces = [C.trace(f"Realisierte Vola ({w} T.)", x, past, color=1, width=1.5)]
    if vix is not None:
        vol_traces.append(C.trace(vix_name, x, vix, color=2, width=1.2))
    out.chart(C.xy(
        "Volatilität: realisiert" + (" und VIX" if vix is not None else ""), vol_traces, x=xa,
        y=C.axis("Annualisierte Volatilität", fmt="pct", digits=0),
        subtitle="Zum Abgleich: Wann war es wirklich unruhig?",
    ))

    # --- does it lead? quintile chart
    labels = ["Q1 niedrig", "Q2", "Q3", "Q4", "Q5 hoch"]
    traces = [C.trace("Entropie S_q", labels, _quintile_means(S, future), color=1, width=2.5),
              C.trace("q (Hill)", labels, _quintile_means(q, future), color=2, width=2.5),
              C.trace("vergangene Vola", labels, _quintile_means(past, future), color=3, width=2.5)]
    out.chart(C.xy(
        f"Vola der nächsten {h} Tage je Quintil des Signals", traces,
        x=C.axis("Quintil des Signals am Tag t", type="category"),
        y=C.axis(f"Ø realisierte Vola t … t+{h}", fmt="pct", digits=0),
        subtitle="Je Linie werden alle Tage nach einem Signal in fünf gleich große Gruppen sortiert. Ein gutes "
                 "Frühwarnsignal steigt von Q1 nach Q5 deutlich an.",
    ))

    rows = [["Entropie S_q", rho_s, _rho(S, past)], ["q (Hill)", rho_q, _rho(q, past)],
            [f"Vergangene Vola ({w} T.)", rho_p, 1.0]]
    cols = ["Signal am Tag t", f"mit Vola t … t+{h}", f"mit Vola t−{w} … t"]
    if vix is not None:
        for row, sig in zip(rows, (S, q, past)):
            row.append(_rho(sig, vix))
        cols.append("mit VIX")
    out.table("Rangkorrelationen (Spearman)", cols,
              rows, formats=["text"] + ["num"] * (len(cols) - 1),
              note=f"{int(np.sum(np.isfinite(S) & np.isfinite(future)))} Tage. Die erste Spalte ist der eigentliche "
                   "Vorlauf-Test: Nur Daten bis t−1 fließen ins Signal, die Vola danach ist echte Zukunft.")

    if rho_s is not None and rho_p is not None:
        if rho_s > 0.2 and rho_s >= rho_p - 0.05:
            out.note(f"Hier kündigt die Entropie Turbulenz an: ρ = {_de(rho_s)}, so gut wie die vergangene Vola "
                     f"(ρ = {_de(rho_p)}).", "success")
        elif rho_p > 0.2:
            out.note(f"Auf diesem Datensatz hat die Entropie kaum Vorlauf: ρ = {_de(rho_s)} mit der künftigen Vola, "
                     f"während die schlichte vergangene Vola ρ = {_de(rho_p)} erreicht.", "warning")
        else:
            out.note(f"Hier ist die künftige Vola kaum vorhersagbar: weder die Entropie (ρ = {_de(rho_s)}) noch die "
                     f"vergangene Vola (ρ = {_de(rho_p)}) haben nennenswerten Vorlauf.", "info")
    if m < data.n:
        out.note(f"Ausgewertet sind die letzten {m} von {data.n} Handelstagen (Parameter „Zeitraum“).", "info")
