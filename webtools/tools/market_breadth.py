"""Marktbreite: breadth thrust and price/breadth divergence on a simulated multi-asset market."""

import numpy as np

from src.analysis.sentiment import MarketBreadthAnalyzer
from src.utils.data_loader import SyntheticMarketGenerator
from src.utils.numerical import rolling_window

from .. import charts as C

SPEC = {
    "id": "market-breadth",
    "title": "Marktbreite: Thrust & Divergenz",
    "group": "warning",
    "order": 50,
    "summary": "Wie viele Aktien den Index tragen – Breadth-Thrust- und Divergenz-Signale auf einem "
               "simulierten Markt mit vielen Einzelwerten.",
    "description": (
        "**Marktbreite** misst, wie viele Aktien eine Bewegung mittragen. Zwei Kennzahlen: der Anteil der "
        "Aktien über ihrer 50-Tage-Linie und die **Advance/Decline-Linie** (kumuliert: steigende minus "
        "fallende Titel je Tag). Der `MarketBreadthAnalyzer` des Frameworks sucht damit zwei Signale: einen "
        "**Breadth Thrust** (die Breite springt in 10 Tagen um mehr als 40 Prozentpunkte – oft der Start einer "
        "Erholung) und eine **Divergenz** (der Index markiert ein 20-Tage-Hoch, die A/D-Linie nicht – die Rally "
        "steht auf wenigen Schultern, ein klassisches Warnzeichen).\n\n"
        "Da echte Einzelwertdaten fehlen, erzeugt `SyntheticMarketGenerator` einen Markt aus korrelierten "
        "Aktien mit fetten Rändern; der Index ist ihr gleichgewichteter Mittelwert. Die **gemeinsame "
        "Marktkorrelation** steuert, wie stark die Titel gemeinsam schwanken: Im Original (0) schwanken die "
        "Paarkorrelationen um null, die Breite bewegt sich dann kaum. Reale Aktien liegen bei etwa 0,3–0,5.\n\n"
        "So liest du die Diagramme: Grüne Punkte markieren Thrust-Signale, orange Divergenz-Tage. Die Tabelle "
        "zeigt, wie sich der Index danach entwickelte. Jeder Wert nutzt nur Daten bis zum jeweiligen Tag."
    ),
    "needs": [],
    "params": [
        {"name": "n_assets", "type": "int", "default": 40, "min": 5, "max": 150, "step": 5, "label": "Anzahl Aktien"},
        {"name": "n_days", "type": "int", "default": 1000, "min": 260, "max": 3000, "step": 20,
         "label": "Handelstage", "unit": "Tage"},
        {"name": "base_corr", "type": "float", "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
         "label": "Gemeinsame Marktkorrelation",
         "help": "0 = Original-Generator (Paarkorrelationen um null). Reale Aktien: etwa 0,3–0,5."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.utils.data_loader.SyntheticMarketGenerator",
        "src.utils.data_loader.SyntheticMarketGenerator.generate",
        "src.analysis.sentiment.MarketBreadthAnalyzer",
        "src.analysis.sentiment.MarketBreadthAnalyzer.add_data",
        "src.analysis.sentiment.MarketBreadthAnalyzer.breadth_thrust_signal",
        "src.analysis.sentiment.MarketBreadthAnalyzer.divergence_warning",
        "src.utils.numerical.rolling_window",
    ],
    "cost": "fast",
}
H = 20  # trading days after a signal


def _starts(mask, gap=10):
    """First day of every signal event: runs of True less than ``gap`` days apart count as one event."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return idx
    keep = np.concatenate([[True], np.diff(idx) > gap])
    return idx[keep]


def _pct_above(prices, w):
    """Share (0-100) of assets above their w-day moving average, for days w-1 .. n-1."""
    ma = rolling_window(prices.T, w).mean(axis=2).T
    return (prices[w - 1:] > ma).mean(axis=1) * 100


def run(params, data, out):
    n_days, k = int(params["n_days"]), int(params["n_assets"])
    gen = SyntheticMarketGenerator(n_assets=k, seed=int(params["seed"]), base_correlation=float(params["base_corr"]))
    mkt = gen.generate(n_days)
    R = np.asarray(mkt["returns"], dtype=float)
    prices = np.exp(np.cumsum(R, axis=0))
    index = 100 * np.exp(np.cumsum(mkt["market"]))
    ad = np.cumsum((R > 0).sum(axis=1) - (R < 0).sum(axis=1))
    b50 = np.full(n_days, np.nan)
    b50[49:] = _pct_above(prices, 50)
    b200 = np.full(n_days, np.nan)
    if n_days >= 200:
        b200[199:] = _pct_above(prices, 200)

    an = MarketBreadthAnalyzer()
    thrust = np.zeros(n_days, bool)
    diverg = np.zeros(n_days, bool)
    magnitude = np.full(n_days, np.nan)
    for t in range(49, n_days):  # feed the analyzer day by day; it only ever sees the past
        an.add_data(ad_line=float(ad[t]), pct_50ma=float(b50[t]),
                    pct_200ma=float(b200[t]) if np.isfinite(b200[t]) else None)
        s = an.breadth_thrust_signal()
        thrust[t] = bool(s.get("thrust_detected"))
        magnitude[t] = s.get("thrust_magnitude", np.nan)
        diverg[t] = bool(an.divergence_warning(index[:t + 1]).get("divergence"))
    signals = _starts(thrust)
    x = list(range(n_days))

    corr = np.asarray(mkt["correlations"], dtype=float)
    iu = np.triu_indices(k, 1)
    dd = index / np.maximum.accumulate(index) - 1
    out.metric("Breite heute", b50[-1] / 100, "pct", digits=0, hint="Aktien über ihrer 50-Tage-Linie")
    out.metric("Breadth-Thrust-Signale", len(signals), "int",
               hint=f"+40 Prozentpunkte Breite in 10 Tagen · {int(thrust.sum())} Signaltage")
    out.metric("Divergenz-Tage", int(diverg.sum()), "int", hint="Index auf 20-Tage-Hoch, A/D-Linie nicht",
               status="warning" if diverg[-5:].any() else None)
    realised = np.corrcoef(R.T)[iu]
    out.metric("Ø Paarkorrelation", float(realised.mean()), "num", digits=2,
               hint=f"realisiert, {k} Aktien · Vorgabe {corr[iu].mean():.2f}".replace(".", ","))
    out.metric("Indexrendite gesamt", float(index[-1] / 100 - 1), "pct", digits=1)
    out.metric("Max. Drawdown des Index", float(dd.min()), "pct", digits=1)

    out.chart(C.xy(
        "Index mit Breiten-Signalen",
        [C.trace("Index (gleichgewichtet)", x, index, color=1, width=1.4),
         C.trace("Divergenz (Warnung)", [x[i] for i in np.flatnonzero(diverg)], index[diverg], type="scatter",
                 color="warning", size=6, decimate_to=None),
         C.trace("Breadth Thrust", [x[i] for i in signals], index[signals], type="scatter", color="good", size=10,
                 decimate_to=None)],
        x=C.axis("Handelstag", fmt="int"), y=C.axis("Index (Start 100)"),
        subtitle="Orange: Index auf Hoch, aber die Mehrheit der Aktien zieht nicht mit. Grün: breite Erholung setzt ein.",
    ))
    out.chart(C.xy(
        "Anteil der Aktien über ihrem Durchschnitt",
        [C.trace("über 50-Tage-Linie", x, b50 / 100, color=1, width=1.4)]
        + ([C.trace("über 200-Tage-Linie", x, b200 / 100, color=2, width=1.2)] if n_days >= 200 else [])
        + [C.trace("Thrust-Signal", [x[i] for i in signals], b50[signals] / 100, type="scatter", color="good", size=10,
                   decimate_to=None)],
        x=C.axis("Handelstag", fmt="int"), y=C.axis("Anteil", fmt="pct", digits=0, range=(0, 1)),
        subtitle="Ein Thrust: die Breite springt binnen 10 Tagen von überverkauft auf breit getragen.",
    ))
    out.chart(C.xy(
        "Advance/Decline-Linie",
        [C.trace("A/D-Linie (kumuliert)", x, ad, color=3, width=1.4)],
        x=C.axis("Handelstag", fmt="int"), y=C.axis("steigende minus fallende Aktien, kumuliert"),
        subtitle="Steigt der Index, die A/D-Linie aber nicht mehr, tragen nur noch wenige Titel die Rally.",
    ))
    centers, counts = C.histogram(realised, bins=min(40, max(10, len(realised) // 10)), density=False)
    out.chart(C.xy(
        "Paarkorrelationen der simulierten Aktien",
        [C.trace("Aktienpaare", centers, counts, type="bar", color=1)],
        x=C.axis("Korrelation der Tagesrenditen", digits=2), y=C.axis("Anzahl Paare", fmt="int"),
        vlines=[C.vline(float(corr[iu].mean()), "Ziel des Generators"),
                C.vline(float(realised.mean()), "Mittel realisiert", "ink", dash=False)],
        subtitle=f"{len(realised)} Paare aus {k} Aktien. Je höher die Korrelation, desto stärker schwanken Breite "
                 "und Index gemeinsam – und desto eher springt die Breite um 40 Punkte.",
    ))

    c = np.concatenate([[0.0], np.cumsum(mkt["market"])])
    fwd = np.full(n_days, np.nan)
    fwd[:n_days - H] = np.expm1(c[1 + H:] - c[1:n_days + 1 - H])
    rows = []
    for name, idx in (("Breadth Thrust (erster Signaltag)", signals), ("Divergenz-Tag", np.flatnonzero(diverg)),
                      ("Alle Tage ab Tag 50", np.arange(49, n_days))):
        f = fwd[idx]
        f = f[np.isfinite(f)]
        rows.append([name, int(len(idx)), float(np.mean(f)) if f.size else None, float(np.mean(f > 0)) if f.size else None,
                     float(np.min(f)) if f.size else None])
    out.table(f"Was folgte auf die Signale? (Index, nächste {H} Handelstage)",
              ["Signal", "Anzahl", f"Ø Rendite {H} T", "Anteil positiv", "Schlechteste"], rows,
              formats=["text", "int", "pct", "pct", "pct"],
              note="Auf einem simulierten Markt ohne echte Anleger zeigt das, wie die Regeln mechanisch reagieren – "
                   "keine Prognosegüte für reale Märkte.")
    if len(signals):
        trows = [[int(t), float(b50[t - 10]) / 100, float(b50[t]) / 100, float(magnitude[t]) / 100,
                  float(fwd[t]) if np.isfinite(fwd[t]) else None] for t in signals[:15]]
        out.table("Breadth-Thrust-Signale", ["Handelstag", "Breite vor 10 Tagen", "Breite", "Zuwachs",
                                             f"Index nächste {H} T"], trows,
                  formats=["int", "pct", "pct", "pct", "pct"],
                  note="Zuwachs in Prozentpunkten der Breite. Signaltage mit weniger als 10 Tagen Abstand zählen "
                       "als ein Signal, gezeigt ab dem ersten Tag.")
    else:
        out.note("Kein Breadth Thrust in diesem Markt. Mit höherer Marktkorrelation schwingt die Breite stärker – "
                 "probier 0,4–0,5 oder einen anderen Seed.", "info")
