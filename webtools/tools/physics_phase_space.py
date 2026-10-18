"""Drei physikalische Phasenräume (Lévy, Tsallis, Fokker-Planck) als interaktive 3D-Bahnen der Marktgeschichte."""

import numpy as np

from src.physics.fokker_planck import compute_fokker_planck_coordinates
from src.physics.levy_flight import levy_flight_3d_coordinates
from src.physics.tsallis_statistics import compute_tsallis_coordinates

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "physics-phase-space",
    "title": "Physikalische Phasenräume in 3D",
    "group": "surface",
    "order": 30,
    "summary": "Verwandelt deine Renditen in drei 3D-Bahnen – Lévy-, Tsallis- und Fokker-Planck-Raum – und zeigt, "
               "wo der Markt heute im Vergleich zu seiner Geschichte steht.",
    "description": (
        "Jeder Handelstag wird zu einem Punkt in drei verschiedenen „Physik-Räumen“, berechnet aus einem "
        "rollierenden Fenster der vorangegangenen Renditen. **Lévy-Raum** (`levy_flight_3d_coordinates`): "
        "X = Volatilität relativ zum Durchschnitt (1 = normal), Y = 2/Hill-Tail-Index (höher = fettere Ränder), "
        "Z = heutige |Rendite| in Standardabweichungen – Crash-Tage ragen als Spitzen heraus. **Tsallis-Raum** "
        "(`compute_tsallis_coordinates`): X = entropischer Index q, Y = Tsallis-Entropie S_q, Z = Temperatur 1/β "
        "(wächst mit der Varianz). **Fokker-Planck-Raum** (`compute_fokker_planck_coordinates`): X = Drift "
        "(mittlere Tagesrendite), Y = Diffusion (Varianz der Tagesrenditen), Z = Ordnung α̂ = 6/Kurtosis mit der "
        "gewöhnlichen Kurtosis (Gauß: 3, also α̂ = 2), begrenzt auf [1, 2] (kleiner = fettere Ränder).\n\n"
        "So liest du die Bilder: Graue Punkte sind die Vergangenheit, die farbige Linie sind die letzten Tage, der "
        "rote Punkt ist heute. Ruhige Phasen bilden eine dichte Wolke, Krisen ziehen lange Ausläufer nach außen. "
        "Ziehen dreht die Ansicht, das Mausrad zoomt, beim Überfahren siehst du das Datum. Die Tabelle ordnet jede "
        "Koordinate von heute als Perzentil ihrer eigenen Geschichte ein."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 30, "min": 20, "max": 120, "step": 5,
         "label": "Fenster (Handelstage)", "help": "Für alle drei Räume gleich."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Hervorgehobene letzte Tage"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 15000, "step": 250,
         "label": "Zeitraum (letzte Handelstage)", "help": "Begrenzt Rechenzeit und Datenmenge bei sehr langen Reihen."},
    ],
    "covers": [
        "src.physics.levy_flight.levy_flight_3d_coordinates",
        "src.physics.tsallis_statistics.compute_tsallis_coordinates",
        "src.physics.fokker_planck.compute_fokker_planck_coordinates",
    ],
    "cost": "medium",
}


def _sig(a, digits=4):
    """Round to significant digits: keeps the JSON payload of three 3D clouds small."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.isfinite(a) & (a != 0), np.floor(np.log10(np.abs(a))), 0)
    scale = 10.0 ** (digits - 1 - mag)
    return np.round(a * scale) / scale


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _date_de(d):
    s = str(d)
    return f"{s[8:10]}.{s[5:7]}.{s[:4]}" if len(s) == 10 and s[4] == "-" else s


def _show(v, fmt):
    """German number text: daily drift in %, everything else with 3 significant digits."""
    if not np.isfinite(v):
        return "–"
    if fmt == "pct":
        return f"{100 * v:.3f} %".replace(".", ",").replace("-", "−")
    txt = f"{v:.3g}"
    if "e" in txt:
        mant, exp = txt.split("e")
        return mant.replace(".", ",").replace("-", "−") + "·10" + str(int(exp)).translate(_SUP)
    return txt.replace(".", ",").replace("-", "−")


def _space(title, coords, labels, fmts, days, k_recent, subtitle):
    X, Y, Z = (_sig(c) for c in coords)
    hist = slice(0, max(0, len(X) - k_recent))
    rec = slice(max(0, len(X) - k_recent), len(X))
    traces = [
        C.path3d("Geschichte", X[hist], Y[hist], Z[hist], color="muted", mode="markers", size=1.5, text=days[hist]),
        C.path3d(f"Letzte {k_recent} Tage", X[rec], Y[rec], Z[rec], color=1, mode="lines+markers", size=3,
                 text=days[rec]),
        C.path3d("Heute", [X[-1]], [Y[-1]], [Z[-1]], color="critical", mode="markers", size=11, text=[days[-1]]),
    ]
    return C.scatter3d(title, traces, x=C.axis(labels[0], fmt=fmts[0]), y=C.axis(labels[1], fmt=fmts[1]),
                       z=C.axis(labels[2], fmt=fmts[2]), subtitle=subtitle, height=520)


def run(params, data, out):
    w, k_recent = params["window"], params["recent"]
    m = min(params["last_n"], data.n)
    if m < w + 30:
        raise ToolError(f"Für ein Fenster von {w} Tagen ist die Reihe zu kurz ({m} Tage). Verkleinere das Fenster.")
    r = np.asarray(data.returns[-m:], dtype=float)
    days_all = [_date_de(d) for d in data.dates[-m:]] if data.dates else [f"Tag {i + 1}" for i in range(data.n - m, data.n)]

    out.progress("Lévy-Koordinaten …", 0.05)
    lev = levy_flight_3d_coordinates(r, window=w)
    out.progress("Tsallis-Koordinaten (q-Fit je Fenster) …", 0.25)
    tsa = compute_tsallis_coordinates(r, window=w)
    out.progress("Fokker-Planck-Koordinaten …", 0.75)
    fpc = compute_fokker_planck_coordinates(r, window=w)

    # the first `window` points are placeholders (zeros) in all three transforms
    keep = slice(w, m)
    lev = [np.asarray(c, dtype=float)[keep] for c in lev]
    tsa = [np.asarray(c, dtype=float)[keep] for c in tsa]
    fpc = [np.asarray(c, dtype=float)[keep] for c in fpc]
    days = np.array(days_all[w:], dtype=object)
    k_recent = int(min(k_recent, len(days)))

    out.chart(_space("Lévy-Phasenraum", lev, ("X: Volatilität (Mittel = 1)", "Y: 2 / Tail-Index", "Z: |Rendite| / σ"),
                     ("num", "num", "num"), days, k_recent,
                     "Spitzen in Z sind einzelne Crash- oder Rallye-Tage; nach rechts wandert die Bahn in Stressphasen."))
    out.chart(_space("Tsallis-Phasenraum", tsa, ("X: q", "Y: Entropie S_q", "Z: Temperatur 1/β"),
                     ("num", "num", "num"), days, k_recent,
                     "Die Temperatur 1/β wächst mit der Varianz; q > 1,3 zeigt fette Ränder im Fenster."))
    out.chart(_space("Fokker-Planck-Phasenraum", fpc, ("X: Drift (Ø Tagesrendite)", "Y: Diffusion (Varianz)",
                                                       "Z: Ordnung α̂ = 6/Kurtosis"),
                     ("pct", "num", "num"), days, k_recent,
                     "α̂ = 2: gaußförmiges Fenster, α̂ → 1: fette Ränder. Krisen: Drift negativ, Diffusion hoch."))

    spaces = [
        ("Lévy", "Volatilität X", lev[0], "num"), ("Lévy", "2 / Tail-Index Y", lev[1], "num"),
        ("Lévy", "|Rendite| / σ Z", lev[2], "num"),
        ("Tsallis", "q", tsa[0], "num"), ("Tsallis", "Entropie S_q", tsa[1], "num"),
        ("Tsallis", "Temperatur 1/β", tsa[2], "num"),
        ("Fokker-Planck", "Drift", fpc[0], "pct"), ("Fokker-Planck", "Diffusion (Varianz)", fpc[1], "num"),
        ("Fokker-Planck", "Ordnung α̂", fpc[2], "num"),
    ]
    rows = []
    for space, name, series, fmt in spaces:
        s = series[np.isfinite(series)]
        now = float(series[-1])
        pct = float(0.5 * (np.mean(s < now) + np.mean(s <= now))) if s.size else None  # mid-rank (ties)
        place = "oberes Dezil" if pct is not None and pct >= 0.9 else "unteres Dezil" if pct is not None and pct <= 0.1 else "üblich"
        rows.append([space, name, _show(now, fmt), _show(float(np.median(s)), fmt), pct, place])
    out.table("Heute im Vergleich zur eigenen Geschichte", ["Raum", "Koordinate", "heute", "Median", "Perzentil heute",
                                                             "Einordnung"],
              rows, formats=["text", "text", "text", "text", "pct", "text"],
              note=f"Perzentil = Anteil der {len(days)} Tage mit kleinerem Wert (Gleichstände zur Hälfte).")

    vol_pct = rows[0][4]
    out.metric("Heute" if data.dates else "Letzter Tag", str(days[-1]), "text",
               hint=f"Fenster {w} Tage, {len(days)} Punkte je Raum")
    out.metric("Lévy-Volatilität X", float(lev[0][-1]), "num", hint=f"Perzentil {_de(100 * vol_pct, 0)} %; 1 = Durchschnitt",
               status="serious" if vol_pct >= 0.95 else "warning" if vol_pct >= 0.8 else None)
    out.metric("Tail-Schwere 2/α (Lévy Y)", float(lev[1][-1]), "num", hint=f"Perzentil {_de(100 * rows[1][4], 0)} %")
    out.metric("Tsallis q", float(tsa[0][-1]), "num", hint=f"Perzentil {_de(100 * rows[3][4], 0)} %")
    out.metric("Fokker-Planck α̂", float(fpc[2][-1]), "num", hint=f"Perzentil {_de(100 * rows[8][4], 0)} % · 2 = Gauß")
    j = int(np.nanargmax(lev[2]))
    out.metric("Größter Sprung (Lévy Z)", float(lev[2][j]), "num", hint=f"{days[j]}: |Rendite| in σ")
    if m < data.n:
        out.note(f"Ausgewertet sind die letzten {m} von {data.n} Handelstagen (Parameter „Zeitraum“).", "info")
