"""Kurspfade mit Lévy-Sprüngen: Fächer, Drawdowns, Pfad-VaR und der Pfad im Lévy-Phasenraum."""

import numpy as np

from src.physics.levy_flight import LevyFlightProcess, LevyStableDistribution, levy_flight_3d_coordinates

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "levy-flight",
    "title": "Lévy-Flug: Kurspfade mit Sprüngen",
    "group": "physics",
    "order": 30,
    "summary": "Simuliert Kurse als Brown'sche Bewegung plus Lévy-Sprünge und zeigt, wie seltene große Sprünge "
               "Drawdowns und Value at Risk verändern.",
    "description": (
        "Der `LevyFlightProcess` erweitert die geometrische Brown'sche Bewegung um Sprünge: Der Log-Kurs bekommt "
        "je Zeitschritt (μ − σ²/2)·dt + σ·dW und zusätzlich eine Poisson-Zahl von Sprüngen (im Mittel **λ pro "
        "Jahr**), deren Größe einer Lévy-stabilen Verteilung folgt. Die **Sprung-Skala γ** ist die typische "
        "Sprunggröße in Prozent; **α** steuert, wie oft aus einem typischen Sprung ein Crash wird.\n\n"
        "Zum Vergleich läuft derselbe Prozess **ohne Sprünge** (λ = 0) mit gleich vielen Pfaden mit. Verglichen "
        "werden die Verteilungen über alle Pfade: Die Sprünge verbrauchen eigene Zufallszahlen, deshalb sind die "
        "Brown'schen Pfade trotz gleichem Seed nicht Pfad für Pfad identisch. Der Kursfächer zeigt Median und "
        "Quantilbänder, die Drawdown- und die Crash-Kurve den Anteil der Pfade, die mindestens x % vom Hoch bzw. an "
        "einem einzigen Tag verlieren. Der Value at Risk bezieht sich auf die einfache Rendite über die gesamte "
        "Laufzeit.\n\n"
        "Der 3D-Plot verwandelt die Tagesrenditen des Pfads mit dem tiefsten Einbruch in Koordinaten des "
        "Lévy-Phasenraums: X = rollierende Volatilität (Mittel = 1), Y = 2/Hill-Tail-Index (höher = fettere "
        "Ränder), Z = |Rendite|/σ. Ruhige Phasen liegen nahe am Ursprung, ein Black-Swan-Sprung ragt weit nach oben."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 0.8, "max": 2.0, "step": 0.05,
         "label": "α der Sprunggrößen", "help": "2 = normalverteilte Sprünge, kleiner = häufiger riesige Sprünge."},
        {"name": "beta", "type": "float", "default": -0.2, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "β der Sprunggrößen", "help": "< 0: große Sprünge eher nach unten."},
        {"name": "jump_scale", "type": "float", "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1, "unit": "%",
         "label": "Sprung-Skala γ", "help": "Typische Sprunggröße als Log-Rendite in %."},
        {"name": "jump_intensity", "type": "float", "default": 12.0, "min": 0.0, "max": 100.0, "step": 1.0,
         "label": "Sprünge pro Jahr λ"},
        {"name": "mu", "type": "float", "default": 7.0, "min": -20.0, "max": 30.0, "step": 0.5, "unit": "%",
         "label": "Drift μ p. a."},
        {"name": "sigma", "type": "float", "default": 15.0, "min": 1.0, "max": 80.0, "step": 1.0, "unit": "%",
         "label": "Volatilität σ p. a.", "help": "Nur der Brown'sche Anteil, ohne Sprünge."},
        {"name": "years", "type": "float", "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25,
         "label": "Laufzeit (Jahre)", "help": "252 Handelstage pro Jahr."},
        {"name": "n_paths", "type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100,
         "label": "Anzahl Pfade"},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "VaR-Niveau",
         "options": [{"value": 0.95, "label": "95 %"}, {"value": 0.99, "label": "99 %"},
                     {"value": 0.995, "label": "99,5 %"}]},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyFlightProcess",
        "src.physics.levy_flight.LevyFlightProcess.simulate_path",
        "src.physics.levy_flight.levy_flight_3d_coordinates",
    ],
    "cost": "fast",
}

S0 = 100.0
MAX_CELLS = 3_000_000  # paths x steps (two simulations)
LOG_CAP = 40.0          # |log price ratio| cap for statistics (e^40: an exploded path)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _stats(paths, c):
    """Simple total return, max drawdown per path, VaR/ES of the total return (all in decimals)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logp = np.clip(np.log(paths / S0), -LOG_CAP, LOG_CAP)
    logp = np.where(np.isnan(logp), LOG_CAP, logp)
    total = np.expm1(logp[:, -1])
    dd = np.expm1((logp - np.maximum.accumulate(logp, axis=1)).min(axis=1))  # <= 0
    var = -float(np.quantile(total, 1 - c))
    tail = total[total <= -var]
    es = -float(np.mean(tail)) if tail.size else var
    return logp, total, dd, var, es


def run(params, data, out):
    years, n_paths, c, seed = params["years"], params["n_paths"], float(params["confidence"]), params["seed"]
    n_steps = max(2, int(round(252 * years)))
    if 2 * n_paths * n_steps > MAX_CELLS:
        raise ToolError(f"{n_paths} Pfade × {n_steps} Schritte sind zu viel für den Browser. "
                        "Verringere die Anzahl Pfade oder die Laufzeit.")

    jumps = LevyStableDistribution(alpha=params["alpha"], beta=params["beta"], gamma=params["jump_scale"], delta=0.0)
    mu, sigma, lam = params["mu"] / 100.0, params["sigma"] / 100.0, params["jump_intensity"]
    proc = LevyFlightProcess(jumps, mu=mu, sigma_gaussian=sigma, jump_intensity=lam)
    plain = LevyFlightProcess(jumps, mu=mu, sigma_gaussian=sigma, jump_intensity=0.0)

    out.progress("Pfade mit Lévy-Sprüngen …", 0.1)
    paths = proc.simulate_path(S0, years, n_steps=n_steps, n_paths=n_paths, seed=seed)
    out.progress("Vergleich ohne Sprünge …", 0.4)
    ref = plain.simulate_path(S0, years, n_steps=n_steps, n_paths=n_paths, seed=seed)

    exploded = int(np.sum(~np.isfinite(paths[:, -1]) | (paths[:, -1] <= 0)))
    logp, total, dd, var, es = _stats(paths, c)
    _, total0, dd0, var0, es0 = _stats(ref, c)
    daily = np.diff(logp, axis=1)
    worst_day = daily.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        worst_day0 = np.diff(np.log(ref), axis=1).min(axis=1)

    lvl = f"{_de(100 * c, 1 if c == 0.995 else 0)} %"
    horizon = f"{_de(years, 2).rstrip('0').rstrip(',')} J."
    out.metric(f"VaR {lvl} ({horizon})", var, "pct", digits=1, hint=f"ohne Sprünge: {_de(100 * var0, 1)} %",
               status="critical" if var > 2 * var0 + 0.05 else "serious" if var > 1.25 * var0 else None)
    out.metric(f"Expected Shortfall {lvl}", es, "pct", digits=1, hint=f"ohne Sprünge: {_de(100 * es0, 1)} %")
    out.metric("Median Max-Drawdown", -float(np.median(dd)), "pct", digits=1,
               hint=f"ohne Sprünge: {_de(-100 * np.median(dd0), 1)} %")
    out.metric(f"Max-Drawdown im schlechtesten {_de(100 * (1 - c), 1 if c == 0.995 else 0)} %",
               -float(np.quantile(dd, 1 - c)), "pct", digits=1,
               hint=f"ohne Sprünge: {_de(-100 * np.quantile(dd0, 1 - c), 1)} %")
    out.metric("Pfade mit Crash-Tag < −10 %", float(np.mean(worst_day < np.log(0.9))), "pct", digits=1,
               hint="mindestens ein Tag mit mehr als 10 % Verlust")
    out.metric("Sprünge je Pfad", lam * years, "num", digits=1, hint="erwartet: λ × Laufzeit")

    # --- chart 1: price fan. Example path: its worst day is a "1-in-20" crash (5 % quantile over all paths)
    t = np.arange(n_steps + 1) / 252.0
    price = S0 * np.exp(logp)
    q = np.quantile(price, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    i_crash = int(np.argmin(np.abs(worst_day - np.quantile(worst_day, 0.05))))
    crash_day = float(np.expm1(worst_day[i_crash]))
    order = np.argsort(total)
    i_mid = int(order[len(order) // 2])
    traces = [
        C.envelope("1–99 %", t, q[0], q[4], color=1),
        C.envelope("5–95 %", t, q[1], q[3], color=1),
        C.trace("Median", t, q[2], color=1, width=2.5),
        C.trace("Median ohne Sprünge", t, np.quantile(ref, 0.5, axis=0), color="muted", dash=True),
        C.trace(f"Crash-Beispiel ({_de(100 * crash_day, 1).replace('-', '−')} %)", t, price[i_crash],
                color="critical", width=1.3),
        C.trace("Typischer Pfad", t, price[i_mid], color=3, width=1.3),
    ]
    out.chart(C.xy(
        "Kursfächer mit Lévy-Sprüngen", traces,
        x=C.axis("Zeit (Jahre)"), y=C.axis("Kurs (Start 100)"),
        subtitle="Bänder: Quantile über alle Pfade. Das Crash-Beispiel ist ein Pfad, dessen schlimmster Tag (Wert "
                 "in Klammern) so schlecht ist wie bei jedem 20. Pfad.", height=380,
    ))

    # --- chart 2: drawdown exceedance curve
    levels = np.linspace(0, 0.9, 91)
    share = np.array([np.mean(-dd >= v) for v in levels])
    share0 = np.array([np.mean(-dd0 >= v) for v in levels])
    out.chart(C.xy(
        "Wie tief fallen die Pfade? Max-Drawdown",
        [C.trace("Mit Lévy-Sprüngen", levels, share, color=2, width=2.5),
         C.trace("Ohne Sprünge", levels, share0, color="muted", dash=True)],
        x=C.axis("Max-Drawdown mindestens", fmt="pct", digits=0), y=C.axis("Anteil der Pfade", fmt="pct", digits=0),
        subtitle="Lesebeispiel: Bei x = 30 % steht der Anteil der Pfade, die zwischenzeitlich 30 % oder mehr vom Hoch verlieren.",
    ))

    # --- chart 3: distribution of the total return
    lo_r, hi_r = float(np.quantile(np.r_[total, total0], 0.002)), float(np.quantile(np.r_[total, total0], 0.998))
    edges = np.linspace(lo_r, hi_r, 61)
    w = edges[1] - edges[0]
    h1 = np.histogram(np.clip(total, lo_r, hi_r), bins=edges)[0] / (len(total) * w)
    h0 = np.histogram(np.clip(total0, lo_r, hi_r), bins=edges)[0] / (len(total0) * w)
    mid = 0.5 * (edges[:-1] + edges[1:])
    floor = 0.5 / (len(total) * w)
    out.chart(C.xy(
        f"Verteilung der Rendite über {horizon}",
        [C.trace("Mit Lévy-Sprüngen", mid, np.where(h1 > 0, h1, np.nan), type="bar", color=2),
         C.trace("Ohne Sprünge", mid, np.where(h0 > 0, h0, np.nan), type="step", color="muted")],
        x=C.axis("Einfache Rendite", fmt="pct", digits=0),
        y=C.axis("Dichte", type="log", range=(floor, 3 * max(h1.max(), h0.max()))),
        vlines=[C.vline(-var, f"VaR {lvl}", color="serious")],
        subtitle="Logarithmische Dichte: Sprünge machen den linken Rand dick. Werte jenseits der 0,2-%-Quantile "
                 "sind am Rand gesammelt.",
    ))

    # --- chart 4: how often a single day crashes by at least x
    crash_levels = np.linspace(0.0, 0.3, 61)
    crash_share = np.array([np.mean(np.expm1(worst_day) <= -v) for v in crash_levels])
    crash_share0 = np.array([np.mean(np.expm1(worst_day0) <= -v) for v in crash_levels])
    out.chart(C.xy(
        "Crash-Tage: schlechtester Tag je Pfad",
        [C.trace("Mit Lévy-Sprüngen", crash_levels, crash_share, color=2, width=2.5),
         C.trace("Ohne Sprünge", crash_levels, crash_share0, color="muted", dash=True)],
        x=C.axis("Tagesverlust mindestens", fmt="pct", digits=0), y=C.axis("Anteil der Pfade", fmt="pct", digits=0),
        vlines=[C.vline(0.10, "−10 %")],
        subtitle="Anteil der Pfade mit mindestens einem Tag, der so viel oder mehr verliert. Ohne Sprünge fällt die "
                 f"Kurve jenseits von etwa 4 Tages-σ (≈ {_de(400 * sigma / np.sqrt(252), 1)} %) praktisch auf null.",
    ))

    # --- chart 5: Lévy phase space of the crash example (the first `win` points are placeholders)
    rets = daily[i_crash]
    win = int(min(20, max(5, len(rets) // 4)))
    X, Y, Z = (np.asarray(v, dtype=float)[win:] for v in levy_flight_3d_coordinates(rets, window=win))
    days = np.arange(win + 1, len(rets) + 1)
    txt = [f"Tag {d}: {100 * np.expm1(v):+.1f} %".replace(".", ",") for d, v in zip(days, rets[win:])]
    j = int(np.argmin(rets[win:]))
    out.chart(C.scatter3d(
        "Das Crash-Beispiel im Lévy-Phasenraum",
        [C.path3d("Tagesweise Bahn", X, Y, Z, color=1, mode="lines+markers", size=2, text=txt),
         C.path3d("Schlimmster Tag", [X[j]], [Y[j]], [Z[j]], color="critical", mode="markers", size=8, text=[txt[j]])],
        x=C.axis("X: Volatilität (Mittel = 1)"), y=C.axis("Y: 2 / Tail-Index"), z=C.axis("Z: |Rendite| / σ"),
        subtitle=f"`levy_flight_3d_coordinates` mit Fenster {win} Tage auf den Tagesrenditen des roten Pfads. "
                 "Ziehen dreht, Mausrad zoomt.",
        height=520,
    ))

    out.table(
        "Mit und ohne Sprünge",
        ["Kennzahl", "Mit Lévy-Sprüngen", "Ohne Sprünge"],
        [[f"VaR {lvl} über {horizon}", var, var0],
         [f"Expected Shortfall {lvl}", es, es0],
         ["Median-Rendite", float(np.median(total)), float(np.median(total0))],
         ["Median Max-Drawdown", -float(np.median(dd)), -float(np.median(dd0))],
         ["Schlechtester Tag (Median der Pfade)", float(np.expm1(np.median(worst_day))),
          float(np.expm1(np.median(worst_day0)))]],
        formats=["text", "pct", "pct"],
        note="Verglichen werden Verteilungen über alle Pfade. Die Sprünge verbrauchen eigene Zufallszahlen, deshalb "
             "sind die Brown'schen Pfade trotz gleichem Seed nicht identisch – nur ihre Statistik ist es.",
    )
    if exploded:
        out.note(f"{exploded} von {n_paths} Pfaden sind numerisch explodiert oder auf null gefallen – bei α ≤ 1 "
                 "hat schon ein einzelner Sprung keinen endlichen Mittelwert. Für die Kennzahlen sind diese Pfade "
                 f"bei e^±{LOG_CAP:.0f} gekappt.", "warning")
    if lam == 0:
        out.note("λ = 0: ohne Sprünge sind beide Simulationen identisch.", "info")
