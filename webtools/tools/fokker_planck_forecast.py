"""Fokker-Planck-Tail-Risk: Prognoseverteilung, Tail-Kennzahlen und eine kleine 3D-Risikofläche."""

import numpy as np
from scipy import stats

from src.physics.fokker_planck import FokkerPlanckTailRisk

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "fokker-planck-forecast",
    "title": "Fokker-Planck: Prognose der Ränder",
    "group": "physics",
    "order": 45,
    "summary": "Schreibt die heutige Verteilung mit der fraktionalen Fokker-Planck-Gleichung über h Tage fort und "
               "misst, wie stark die Ränder gegenüber Gauß wachsen.",
    "description": (
        "`FokkerPlanckTailRisk` startet mit einer Gauß-Glocke um die heutige Lage und lässt sie h Handelstage lang "
        "nach der fraktionalen Fokker-Planck-Gleichung fließen: Mean-Reversion θ zur Mitte, dazu Lévy-Diffusion "
        "der Ordnung **α** mit einer Stärke, die mit dem Quadrat des **Vola-Regimes** wächst (D_α = 0,01·Regime² "
        "pro Jahr). Bei α = 2 läuft stattdessen gewöhnliche Diffusion. x ist die Tagesrendite in Einheiten der "
        "heutigen Standardabweichung σ: Die Prognose zeigt, wie die Verteilung der Tagesrendite in h Tagen aussieht – "
        "nicht die über h Tage aufsummierte Rendite. Weil die Diffusion im Modell klein ist, zieht die Mean-Reversion "
        "die Glocke mit wachsendem Horizont sogar leicht zusammen.\n\n"
        "So liest du das Ergebnis: Der Kern der Verteilung ändert sich über wenige Wochen kaum, aber weit draußen "
        "wächst Masse nach, die eine Normalverteilung nie hätte. Die Faktoren in der Tabelle teilen die "
        "Tail-Masse jenseits von kσ durch den Gauß-Wert – bei 2σ sind sie ≈ 1, bei 5σ oft zweistellig.\n\n"
        "Die **3D-Risikofläche** (`generate_3d_risk_surface`) wiederholt die Prognose für ein kleines Gitter aus "
        "Vola-Regime × Horizont, jeweils ab N(0, 1), und zeigt die Tail-Wahrscheinlichkeit jenseits der gewählten "
        "Schwelle. Jede Zelle ist eine eigene Simulation; halte das Gitter im Browser klein."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 1.0, "max": 2.0, "step": 0.05,
         "label": "Fraktionale Ordnung α", "help": "2 = gewöhnliche Diffusion; kleiner = fettere Ränder."},
        {"name": "theta", "type": "float", "default": 0.1, "min": 0.0, "max": 3.0, "step": 0.05,
         "label": "Mean-Reversion θ (pro Jahr)"},
        {"name": "horizon", "type": "int", "default": 20, "min": 1, "max": 252, "step": 1,
         "label": "Horizont (Handelstage)"},
        {"name": "vol_regime", "type": "float", "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
         "label": "Vola-Regime", "help": "1 = normal, 2 = doppelte Vola (vierfache Lévy-Diffusion)."},
        {"name": "current_return", "type": "float", "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
         "label": "Heutige Lage (in σ)", "help": "Mitte der Start-Glocke."},
        {"name": "current_vol", "type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
         "label": "Heutige Breite (in σ)"},
        {"name": "surface", "type": "bool", "default": True, "label": "3D-Risikofläche berechnen"},
        {"name": "surface_n", "type": "int", "default": 3, "min": 2, "max": 5, "step": 1,
         "label": "Fläche: Punkte je Achse", "show_if": {"surface": [True]}},
        {"name": "surface_h_max", "type": "int", "default": 60, "min": 10, "max": 120, "step": 5,
         "label": "Fläche: Horizont bis (Tage)", "show_if": {"surface": [True]}},
        {"name": "surface_vol_max", "type": "float", "default": 3.0, "min": 1.5, "max": 4.0, "step": 0.5,
         "label": "Fläche: Vola-Regime bis", "show_if": {"surface": [True]}},
        {"name": "threshold", "type": "float", "default": 4.0, "min": 2.0, "max": 5.0, "step": 0.5,
         "label": "Fläche: Schwelle (σ)", "show_if": {"surface": [True]}},
    ],
    "covers": [
        "src.physics.fokker_planck.FokkerPlanckTailRisk",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.initial_distribution",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.forecast_distribution",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.compute_tail_risk_metrics",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.generate_3d_risk_surface",
    ],
    "cost": "medium",
}

MAX_DAYS = 1300   # simulated forecast days in one run (~4 Crank-Nicolson steps per day, dense 500 x 500)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def run(params, data, out):
    alpha, theta, h, regime = params["alpha"], params["theta"], params["horizon"], params["vol_regime"]
    n_s = params["surface_n"]
    vols = np.round(np.linspace(0.5, params["surface_vol_max"], n_s), 2)
    h_max = params["surface_h_max"]
    hs = np.linspace(5, h_max, n_s)
    hs = np.unique(np.clip(np.round(hs / 5) * 5, 5, h_max).astype(int))  # round horizons to 5-day steps
    surface_days = int(len(vols) * hs.sum()) if params["surface"] else 0
    if h + surface_days > MAX_DAYS:
        raise ToolError(f"Prognose plus Risikofläche wären {h + surface_days} simulierte Tage (Grenze {MAX_DAYS}). "
                        "Verkleinere das Flächengitter, den maximalen Horizont oder schalte die Fläche ab.")

    model = FokkerPlanckTailRisk(alpha=alpha, theta=theta)
    solver = model.solver
    x = solver.x
    P0 = model.initial_distribution(params["current_return"], params["current_vol"])
    out.progress("Prognose der Verteilung …", 0.05)
    P = model.forecast_distribution(P0, horizon=h, vol_regime=regime)
    m = model.compute_tail_risk_metrics(P)
    mo, tp = m["moments"], m["tail_probabilities"]
    m0 = solver.compute_moments(P0)

    out.metric(f"Streuung in {h} Tagen", mo["std"], "num",
               hint=f"in heutigen σ; Start {_de(m0['std'])}. θ zieht zusammen, die Lévy-Diffusion weitet")
    out.metric("Exzess-Kurtosis", mo["kurtosis"], "num", hint="Normalverteilung = 0",
               status="critical" if mo["kurtosis"] > 6 else "serious" if mo["kurtosis"] > 1 else None)
    out.metric("Masse jenseits 3σ", tp["3sigma"]["total"], "pct", hint=f"Gauß: {_de(200 * stats.norm.sf(3))} %")
    for key, k in (("4sigma", 4), ("5sigma", 5)):
        ratio = tp[key]["ratio_to_gaussian"]
        out.metric(f"Faktor gegenüber Gauß ({k}σ)", ratio, "x", hint=f"Tail-Masse jenseits {k}σ ÷ Gauß-Wert",
                   status="critical" if ratio > 20 else "serious" if ratio > 3 else None)
    out.metric("Fat-Tail-Faktor 2/α", m["fat_tail_factor"], "num", hint=f"Tail-Index α = {_de(m['tail_index'])}")

    # --- chart 1: start vs forecast density, log scale
    floor = 1e-7
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        f"Verteilung heute und in {h} Tagen",
        [C.trace(f"In {h} Tagen (Fokker-Planck)", x, keep(P), color=1, width=2.5),
         C.trace("Heute (Gauß)", x, keep(P0), color=2, width=1.5),
         C.trace("Gauß gleicher Varianz", x, keep(stats.norm.pdf(x, mo["mean"], mo["std"])), color="muted", dash=True)],
        x=C.axis("Rendite (in heutigen σ)"), y=C.axis("Dichte", type="log", range=(floor, 3 * float(max(P.max(), P0.max())))),
        subtitle="Logarithmische Skala: Der Kern bleibt fast gleich, die Ränder heben sich von der Gauß-Parabel ab.",
    ))

    # --- chart 2: how many times more tail mass than a Gaussian, as a function of the distance in sigma
    ks = np.linspace(0.5, 6.0, 56)
    tails = np.array([sum(solver.tail_probability(P, k)) for k in ks])
    ratio = tails / (2 * stats.norm.sf(ks))
    out.chart(C.xy(
        "Tail-Masse gegenüber Gauß nach Abstand",
        [C.trace("Fokker-Planck-Prognose", ks, np.where(tails > 1e-12, ratio, np.nan), color=1, width=2.5),
         C.trace("Normalverteilung = 1", [ks[0], ks[-1]], [1.0, 1.0], color="muted", dash=True)],
        x=C.axis("Abstand k (in σ der Prognose)"), y=C.axis("P(|X − μ| > kσ) ÷ Gauß-Wert", type="log"),
        subtitle="Bis etwa 3σ liegt die Prognose beim Gauß-Wert – weiter draußen hat sie ein Vielfaches an Masse.",
    ))

    rows = []
    for k in (2, 3, 4, 5):
        t = tp[f"{k}sigma"]
        rows.append([f"{k} σ", t["left"], t["right"], t["total"], float(2 * stats.norm.sf(k)), t["ratio_to_gaussian"]])
    out.table("Tail-Kennzahlen der Prognose", ["Schwelle", "links", "rechts", "gesamt", "Gauß gesamt", "Faktor"],
              rows, formats=["text", "pct", "pct", "pct", "pct", "x"],
              note="Aus compute_tail_risk_metrics: Masse jenseits Mittelwert ± k·σ der prognostizierten Dichte.")

    if not params["surface"]:
        out.note("3D-Risikofläche ausgeschaltet.", "info")
        return

    out.progress(f"3D-Risikofläche: {len(vols) * len(hs)} Prognosen …", 0.3)
    thr = params["threshold"]
    Z = model.generate_3d_risk_surface(vols, hs, threshold_sigma=thr)
    points = []
    if params["current_return"] == 0 and params["current_vol"] == 1:
        own = float(sum(solver.tail_probability(P, thr)))
        points.append(C.path3d("Deine Prognose", [h], [regime], [own], color="critical", mode="markers", size=9))
    out.chart(C.surface(
        f"Risikofläche: P(|X| > {_de(thr, 1)}σ)", hs.tolist(), vols.tolist(), Z,
        x=C.axis("Horizont (Tage)", fmt="int"), y=C.axis("Vola-Regime"), z=C.axis("Tail-Wahrscheinlichkeit", fmt="pct"),
        points=points, height=520,
        subtitle=f"{len(vols)} × {len(hs)} Prognosen ab N(0, 1). Gauß-Niveau: {200 * stats.norm.sf(thr):.3g} %".replace(".", ",")
                 + ". Ziehen dreht, Mausrad zoomt.",
    ))
    g_thr = float(2 * stats.norm.sf(thr))
    out.table(f"Risikofläche als Vielfaches des Gauß-Werts ({200 * stats.norm.sf(thr):.3g} %)".replace(".", ","),
              ["Vola-Regime"] + [f"{d} Tage" for d in hs],
              [[float(v)] + [float(z) / g_thr for z in Z[i]] for i, v in enumerate(vols)],
              formats=["num"] + ["x"] * len(hs),
              note=f"P(|X| > {_de(thr, 1)}σ) jeder Zelle geteilt durch den Wert einer Normalverteilung.")
