"""Fokker-Planck-Gleichung: wie sich eine Renditeverteilung mit Drift, Diffusion und Lévy-Sprüngen entwickelt."""

import numpy as np
from scipy import stats

from src.physics.fokker_planck import FokkerPlanckSolver

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "fokker-planck",
    "title": "Fokker-Planck: Dichte im Zeitverlauf",
    "group": "physics",
    "order": 40,
    "summary": "Löst die (fraktionale) Fokker-Planck-Gleichung und zeigt, wie aus einer schmalen Gauß-Glocke "
               "eine Verteilung mit fetten Rändern wird.",
    "description": (
        "Die **Fokker-Planck-Gleichung** beschreibt nicht einen Kurspfad, sondern die ganze Wahrscheinlichkeitsdichte "
        "P(x, t) und wie sie fließt: ∂P/∂t = −∂(μP)/∂x + Diffusion. Die Drift μ(x) = −θ(x − μ∞) zieht die Masse zum "
        "Gleichgewicht μ∞ zurück (Mean-Reversion, θ = Geschwindigkeit). Im **Standardmodell** diffundiert sie mit "
        "D(x) = D₀(1 + λ|x|^β): Je größer λ, desto unruhiger wird es weit weg vom Zentrum – Vola-Clustering. Im "
        "**fraktionalen Modell** ersetzt der Riesz-Operator D_α ∂^α/∂|x|^α die Diffusion: Wahrscheinlichkeit "
        "springt dann über weite Strecken (Lévy-Flüge), und für α < 2 entstehen Potenzgesetz-Ränder.\n\n"
        "So liest du die Diagramme: In der Heatmap läuft die Zeit nach unten, die Farbe ist die logarithmische "
        "Dichte (dunkel = hoch). Je weiter die kräftige Färbung nach links und rechts reicht, desto fetter sind die "
        "Ränder; eine Gauß-Glocke bliebe außen blass. Die Dichte-Kurven vergleichen "
        "das Ergebnis mit einer Normalverteilung gleicher Varianz, der dritte Plot verfolgt die Masse jenseits von "
        "3σ und 4σ über die Zeit. x ist eine standardisierte Rendite, die Zeit ist dimensionslos.\n\n"
        "Numerik: Crank-Nicolson mit einmal faktorisierter Matrix, Gitter [−10, 10], Ränder absorbierend, die Masse "
        "wird nach jedem Schritt renormiert. Sehr fette Ränder (kleines α, großes D_α) erreichen den Gitterrand – "
        "Varianz und Kurtosis gelten dann für die auf [−10, 10] abgeschnittene Verteilung."
    ),
    "needs": [],
    "params": [
        {"name": "model", "type": "select", "default": "fractional", "label": "Modell",
         "options": [{"value": "fractional", "label": "Fraktional (Lévy-Diffusion, α < 2)"},
                     {"value": "standard", "label": "Standard (Diffusion D(x), Vola-Clustering)"}]},
        {"name": "alpha", "type": "float", "default": 1.5, "min": 0.5, "max": 1.95, "step": 0.05,
         "label": "Fraktionale Ordnung α", "help": "Nur im fraktionalen Modell. Kleiner = weitere Sprünge, fettere Ränder."},
        {"name": "D_alpha", "type": "float", "default": 0.5, "min": 0.05, "max": 2.0, "step": 0.05,
         "label": "Lévy-Diffusion D_α", "help": "Nur im fraktionalen Modell."},
        {"name": "D0", "type": "float", "default": 0.5, "min": 0.05, "max": 2.0, "step": 0.05,
         "label": "Basis-Diffusion D₀", "help": "Nur im Standardmodell."},
        {"name": "lam", "type": "float", "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1,
         "label": "Vola-Clustering λ", "help": "Nur im Standardmodell: D(x) = D₀(1 + λ|x|^β)."},
        {"name": "beta_exp", "type": "float", "default": 1.0, "min": 0.5, "max": 2.5, "step": 0.1,
         "label": "Nichtlinearität β", "help": "Nur im Standardmodell."},
        {"name": "theta", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
         "label": "Mean-Reversion θ", "help": "0 = keine Rückstellkraft."},
        {"name": "mu_inf", "type": "float", "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
         "label": "Gleichgewicht μ∞"},
        {"name": "x0", "type": "float", "default": 2.0, "min": -5.0, "max": 5.0, "step": 0.1,
         "label": "Startpunkt x₀", "help": "Mitte der Gauß-Glocke zu t = 0."},
        {"name": "sigma0", "type": "float", "default": 0.5, "min": 0.2, "max": 3.0, "step": 0.1,
         "label": "Startbreite σ₀"},
        {"name": "T", "type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Zeithorizont T"},
        {"name": "dt", "type": "float", "default": 0.01, "min": 0.002, "max": 0.05, "step": 0.002,
         "label": "Zeitschritt dt"},
        {"name": "n_grid", "type": "int", "default": 301, "min": 101, "max": 501, "step": 50, "label": "Gitterpunkte"},
    ],
    "covers": [
        "src.physics.fokker_planck.FokkerPlanckSolver",
        "src.physics.fokker_planck.FokkerPlanckSolver.drift_coefficient",
        "src.physics.fokker_planck.FokkerPlanckSolver.diffusion_coefficient",
        "src.physics.fokker_planck.FokkerPlanckSolver.build_operator_matrix",
        "src.physics.fokker_planck.FokkerPlanckSolver.build_fractional_operator",
        "src.physics.fokker_planck.FokkerPlanckSolver.solve",
        "src.physics.fokker_planck.FokkerPlanckSolver.compute_moments",
        "src.physics.fokker_planck.FokkerPlanckSolver.tail_probability",
    ],
    "cost": "fast",
}

MAX_WORK = 4.0e8      # fractional: steps x n_grid^2 (dense Crank-Nicolson), ~1.5 s native
MAX_STEPS = 5000
G3, G4 = 2 * stats.norm.sf(3), 2 * stats.norm.sf(4)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sci(x, digits=1):
    """German scientific notation for the numerics table: 2,3·10⁻⁴."""
    if not np.isfinite(x) or x == 0:
        return _de(float(x), digits)
    mant, exp = f"{x:.{digits}e}".split("e")
    return mant.replace(".", ",") + "·10" + str(int(exp)).translate(_SUP)


def run(params, data, out):
    frac = params["model"] == "fractional"
    n, T, dt = params["n_grid"], params["T"], params["dt"]
    n_steps = int(round(T / dt))
    if n_steps > MAX_STEPS or (frac and n_steps * n * n > MAX_WORK):
        raise ToolError(f"{n_steps} Zeitschritte auf {n} Gitterpunkten sind zu viel für den Browser. "
                        "Vergrößere dt, verkleinere T oder nimm weniger Gitterpunkte.")
    alpha = params["alpha"] if frac else 2.0
    fp_params = {"theta": params["theta"], "mu_inf": params["mu_inf"], "D0": params["D0"],
                 "lambda": params["lam"], "beta": params["beta_exp"], "D_alpha": params["D_alpha"]}
    solver = FokkerPlanckSolver(x_min=-10, x_max=10, n_grid=n, alpha=alpha)
    x = solver.x
    x0, s0 = params["x0"], params["sigma0"]
    P0 = np.exp(-0.5 * ((x - x0) / s0) ** 2)
    P0 /= np.sum(P0) * solver.dx

    out.progress("Crank-Nicolson-Zeitschritte …", 0.1)
    times, Pe = solver.solve(P0, T=T, dt=dt, params=fp_params, use_fractional=frac)
    moments = [solver.compute_moments(P) for P in Pe]
    tails3 = np.array([sum(solver.tail_probability(P, 3)) for P in Pe])
    tails4 = np.array([sum(solver.tail_probability(P, 4)) for P in Pe])
    end = moments[-1]

    drift0 = float(solver.drift_coefficient(np.array([x0]), 0, fp_params)[0])
    out.metric("Mittelwert bei T", end["mean"], "num", hint=f"Start {_de(x0, 1)} · Gleichgewicht {_de(params['mu_inf'], 1)}")
    out.metric("Streuung σ bei T", end["std"], "num", hint=f"Start {_de(s0, 1)}")
    out.metric("Exzess-Kurtosis bei T", end["kurtosis"], "num", digits=1, hint="Normalverteilung = 0",
               status="critical" if end["kurtosis"] > 6 else "serious" if end["kurtosis"] > 3 else None)
    out.metric("Masse jenseits 3σ", tails3[-1], "pct", hint=f"Gauß: {_de(100 * G3)} %")
    out.metric("Faktor ggü. Gauß (4σ)", tails4[-1] / G4, "x", hint="Masse jenseits 4σ ÷ Gauß-Wert",
               status="critical" if tails4[-1] / G4 > 20 else "serious" if tails4[-1] / G4 > 3 else None)
    out.metric("Rückstellkraft μ(x₀)", drift0, "num", hint="Drift am Startpunkt: −θ(x₀ − μ∞)")

    # --- chart 1: heatmap of log density over (x, t)
    fstep = max(1, int(np.ceil(len(times) / 34)))
    rows = list(range(0, len(times), fstep))
    if rows[-1] != len(times) - 1:
        rows.append(len(times) - 1)
    xstep = max(1, int(np.ceil(n / 75)))
    logP = np.log10(np.maximum(Pe[rows][:, ::xstep], 1e-12))
    top = float(np.max(logP))
    logP = np.round(np.maximum(logP, top - 6), 3)
    out.chart(C.heatmap(
        "Wahrscheinlichkeitsdichte P(x, t)",
        np.round(x[::xstep], 3), np.round(times[rows], 3), logP,
        x=C.axis("x (standardisierte Rendite)"), y=C.axis("Zeit t", type="linear"),
        z=C.axis("log₁₀ Dichte"), scale="sequential", height=420,
        subtitle="Zeit läuft nach unten, Farbe = log₁₀ Dichte über 6 Dekaden (dunkel = hoch). Je weiter die kräftige "
                 "Färbung nach außen reicht, desto fetter die Ränder.",
    ))

    # --- chart 2: snapshots on a log scale vs a Gaussian with the same variance
    picks = sorted({0, int(np.argmin(np.abs(times - T / 10))), int(np.argmin(np.abs(times - T / 3))), len(times) - 1})
    floor = 1e-6
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    traces = [C.trace(f"t = {_de(times[i], 2)}", x, keep(Pe[i]), color=k + 1, width=2.5 if i == picks[-1] else 1.5)
              for k, i in enumerate(picks)]
    traces.append(C.trace("Gauß gleicher Varianz (t = T)", x, keep(stats.norm.pdf(x, end["mean"], end["std"])),
                          color="muted", dash=True))
    out.chart(C.xy(
        "Dichte zu ausgewählten Zeiten", traces,
        x=C.axis("x (standardisierte Rendite)"), y=C.axis("Dichte", type="log", range=(floor, 3 * float(Pe.max()))),
        subtitle="Logarithmische Skala: Liegt die Kurve an den Rändern über der Gauß-Parabel, sind die Ränder fett.",
    ))

    # --- chart 3: tail mass over time, as a multiple of the Gaussian value
    r3, r4 = tails3 / G3, tails4 / G4
    out.chart(C.xy(
        "Fette Ränder über die Zeit: Faktor ggü. Gauß",
        [C.trace("Masse jenseits 3σ", times, np.where(r3 > 0, r3, np.nan), color=1),
         C.trace("Masse jenseits 4σ", times, np.where(r4 > 1e-6, r4, np.nan), color=2),
         C.trace("Normalverteilung = 1", [times[0], times[-1]], [1.0, 1.0], color="muted", dash=True)],
        x=C.axis("Zeit t"), y=C.axis("Vielfaches des Gauß-Werts", type="log"),
        subtitle="Tail-Masse jenseits k·σ (σ = aktuelle Streuung der Dichte) geteilt durch den Gauß-Wert "
                 f"({_de(100 * G3)} % bzw. {_de(100 * G4, 4)} %). Über 1: fette Ränder.",
    ))

    # --- chart 4: what the operator does
    if frac:
        L = solver.build_fractional_operator(alpha, fp_params)
        i0 = n // 2
        k = np.arange(1, n - i0)
        rate = L[i0, i0 + k]
        dist = k * solver.dx
        ref = rate[4] * (dist / dist[4]) ** (-(1 + alpha))
        out.chart(C.xy(
            "Sprungkern des fraktionalen Operators",
            [C.trace("Übergangsrate in eine Zelle im Abstand d", dist, np.where(rate > 0, rate, np.nan), color=2),
             C.trace(f"Potenzgesetz d^−(1+α) = d^−{_de(1 + alpha)}", dist, ref, color="muted", dash=True)],
            x=C.axis("Abstand d", type="log"), y=C.axis("Rate je Zeiteinheit", type="log"),
            subtitle="Gewöhnliche Diffusion verschiebt Masse nur in die Nachbarzelle. Der fraktionale Operator springt "
                     "mit einer Rate ∝ d^−(1+α) über das ganze Gitter – daher die fetten Ränder.",
        ))
        L_full = L + solver.build_operator_matrix(0, fp_params, include_diffusion=False).toarray()
    else:
        D = solver.diffusion_coefficient(x, 0, fp_params)
        out.chart(C.xy(
            "Diffusion D(x) – Vola-Clustering",
            [C.trace("D(x) = D₀(1 + λ|x|^β)", x, D, color=2)],
            x=C.axis("x (standardisierte Rendite)"), y=C.axis("Diffusion D(x)"),
            hlines=[C.hline(params["D0"], "D₀")],
            subtitle="Steigt D mit |x|, fließt Masse in den Rändern schneller nach außen – fette Ränder ohne Sprünge.",
        ))
        L_full = solver.build_operator_matrix(0, fp_params).toarray()

    xs = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    mu_x = solver.drift_coefficient(xs, 0, fp_params)
    if frac:
        out.table("Drift und Lévy-Diffusion", ["x", "Drift μ(x)", "D_α (konstant)"],
                  [[float(a), float(b) + 0.0, params["D_alpha"]] for a, b in zip(xs, mu_x)], formats=["num", "num", "num"])
    else:
        D_x = solver.diffusion_coefficient(xs, 0, fp_params)
        out.table("Drift und Diffusion", ["x", "Drift μ(x)", "Diffusion D(x)"],
                  [[float(a), float(b) + 0.0, float(c)] for a, b, c in zip(xs, mu_x, D_x)], formats=["num", "num", "num"])

    mid = slice(n // 4, 3 * n // 4)
    balance = float(np.max(np.abs(L_full[:, mid].sum(axis=0)) / np.abs(np.diag(L_full)[mid])))
    edge_mass = float(np.sum(Pe[-1][np.abs(x) > 8]) * solver.dx)
    out.table("Numerik", ["Größe", "Wert"], [
        ["Gitter", f"{n} Punkte auf [−10, 10], dx = {_de(float(solver.dx), 4)}"],
        ["Zeitschritte", f"{n_steps} × dt = {_de(T / max(n_steps, 1), 4)}"],
        ["Massenbilanz: |Spaltensumme| ÷ |Diagonale| (Gittermitte, max.)", _sci(balance)],
    ], formats=["text", "text"],
        note="Eine Spaltensumme nahe 0 heißt: Der diskrete Operator erhält die Wahrscheinlichkeit. Beim fraktionalen "
             "Operator fehlt der Anteil der Sprünge, die über den Gitterrand hinausführen würden.")
    out.metric("Masse am Gitterrand", edge_mass, "pct", hint="Anteil bei |x| > 8 zu t = T",
               status="warning" if edge_mass > 1e-3 else None)
    if edge_mass > 1e-3:
        out.note(f"{_de(100 * edge_mass, 1)} % der Masse liegt bei |x| > 8 – die Ränder stoßen an das Gitter. "
                 "Varianz und Kurtosis beschreiben dann die abgeschnittene Verteilung.", "warning")
