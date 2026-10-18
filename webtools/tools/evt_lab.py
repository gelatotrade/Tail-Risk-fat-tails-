"""EVT laboratory: simulate losses with a known tail, watch GEV and GPD recover the shape parameter."""

import numpy as np
from scipy import stats

from src.models.extreme_value import GeneralizedExtremeValue, GeneralizedParetoDistribution

from .. import charts as C
from ..core import ToolError

DAYS = 252

SPEC = {
    "id": "evt-lab",
    "title": "EVT-Labor: GEV und GPD",
    "group": "risk",
    "order": 35,
    "summary": "Simuliere Verluste mit bekanntem Rand und sieh zu, wie Blockmaxima (GEV) und Peaks over Threshold "
               "(GPD) den Formparameter ξ zurückgewinnen.",
    "description": (
        "Das **Fisher-Tippett-Theorem** sagt: Maxima vieler Verluste folgen – egal woher sie stammen – einer von drei "
        "Formen der **GEV-Verteilung**. ξ > 0 (Fréchet) bei Potenzgesetz-Rändern wie der Student-t (ξ = 1/ν), "
        "ξ = 0 (Gumbel) bei exponentiellen Rändern wie der Normalverteilung, ξ < 0 (Weibull) bei begrenzten Verlusten, "
        "etwa durch eine Absicherung. Für Überschreitungen einer hohen Schwelle gilt dasselbe mit der "
        "**Generalisierten Pareto-Verteilung (GPD)**.\n\n"
        "Wähle eine Ausgangsverteilung, simuliere Tagesverluste und vergleiche die geschätzten ξ mit der Theorie. "
        "Der **Likelihood-Quotienten-Test** prüft, ob ξ signifikant von 0 abweicht; bei der Normalverteilung "
        "konvergieren Maxima nur langsam, kleine negative Schätzungen sind daher normal.\n\n"
        "Im **Wiederkehr-Diagramm** ist die Gumbel-Form eine Gerade über der logarithmischen Zeitachse. Fréchet "
        "biegt nach oben ab (die nächste Katastrophe ist größer, als die Gerade vermuten lässt), Weibull flacht ab. "
        "Verluste sind positive Dezimalzahlen (0,05 = 5 % an einem Tag)."
    ),
    "needs": [],
    "params": [
        {"name": "parent", "type": "select", "default": "student_t", "label": "Ausgangsverteilung der Verluste",
         "options": [{"value": "student_t", "label": "Student-t (Fréchet, ξ = 1/ν)"},
                     {"value": "gaussian", "label": "Normalverteilung (Gumbel, ξ = 0)"},
                     {"value": "bounded", "label": "Begrenzt auf 5 % (Weibull, ξ = −½)"}]},
        {"name": "nu", "type": "float", "default": 3.0, "min": 1.5, "max": 20.0, "step": 0.5,
         "label": "Student-t: Freiheitsgrade ν", "help": "Nur für die Student-t. ξ = 1/ν."},
        {"name": "n_obs", "type": "int", "default": 5040, "min": 1000, "max": 50000, "step": 252,
         "label": "Simulierte Handelstage", "help": "5.040 = 20 Jahre."},
        {"name": "block_size", "type": "int", "default": 20, "min": 5, "max": 100, "step": 1,
         "label": "Blockgröße (Handelstage)"},
        {"name": "threshold_quantile", "type": "float", "default": 0.95, "min": 0.8, "max": 0.995, "step": 0.005,
         "label": "GPD-Schwelle (Quantil)"},
        {"name": "seed", "type": "int", "default": 7, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.models.extreme_value.GeneralizedExtremeValue",
        "src.models.extreme_value.GeneralizedExtremeValue.cdf",
        "src.models.extreme_value.GeneralizedExtremeValue.pdf",
        "src.models.extreme_value.GeneralizedExtremeValue.ppf",
        "src.models.extreme_value.GeneralizedExtremeValue.return_level",
        "src.models.extreme_value.GeneralizedExtremeValue.fit_block_maxima",
        "src.models.extreme_value.GeneralizedParetoDistribution",
        "src.models.extreme_value.GeneralizedParetoDistribution.cdf",
        "src.models.extreme_value.GeneralizedParetoDistribution.pdf",
        "src.models.extreme_value.GeneralizedParetoDistribution.ppf",
        "src.models.extreme_value.GeneralizedParetoDistribution.fit_pot",
    ],
    "cost": "fast",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _domain(xi, tol=0.05):
    return "Fréchet" if xi > tol else "Weibull" if xi < -tol else "Gumbel"


def _simulate(parent, nu, n, rng):
    if parent == "student_t":
        return 0.007 * rng.standard_t(nu, n), 1.0 / nu
    if parent == "gaussian":
        return 0.01 * rng.standard_normal(n), 0.0
    return 0.05 * (2.0 * rng.beta(2.0, 2.0, n) - 1.0), -0.5  # endpoint 5 %, density ~ (0.05 - x)^1


def run(params, data, out):
    parent, nu, n = params["parent"], float(params["nu"]), int(params["n_obs"])
    m, q, seed = int(params["block_size"]), float(params["threshold_quantile"]), int(params["seed"])
    nb = n // m
    if nb < 10:
        raise ToolError(f"Nur {nb} Blöcke à {m} Tage. Erhöhe die Zahl der Tage oder verkleinere die Blöcke.")
    if n * (1 - q) < 15:
        raise ToolError("Über der Schwelle liegen weniger als 15 Verluste. Senke die Schwelle oder simuliere mehr Tage.")

    rng = np.random.default_rng(seed)
    x, xi_true = _simulate(parent, nu, n, rng)

    out.progress("Passe GEV an die Blockmaxima an …", 0.3)
    gev = GeneralizedExtremeValue.fit_block_maxima(x, m)
    maxima = x[:nb * m].reshape(nb, m).max(axis=1)
    g_loc, g_scale = stats.gumbel_r.fit(maxima)
    gumbel = GeneralizedExtremeValue(0.0, g_loc, g_scale)
    ll_gev = float(np.sum(np.log(np.maximum(gev.pdf(maxima), 1e-300))))
    ll_gum = float(np.sum(np.log(np.maximum(gumbel.pdf(maxima), 1e-300))))
    lr_gev = max(0.0, 2 * (ll_gev - ll_gum))
    p_gev = float(stats.chi2.sf(lr_gev, 1))
    ks_gev = stats.kstest(maxima, gev.cdf)

    out.progress("Passe GPD an die Überschreitungen an …", 0.6)
    gpd, st = GeneralizedParetoDistribution.fit_pot(x, quantile=q)
    exc = np.sort(x[x > st["threshold"]] - st["threshold"])
    expo = GeneralizedParetoDistribution(0.0, float(np.mean(exc)), st["threshold"])  # exponential = MLE for xi = 0
    ll_gpd = float(np.sum(np.log(np.maximum(gpd.pdf(exc), 1e-300))))
    ll_exp = float(np.sum(np.log(np.maximum(expo.pdf(exc), 1e-300))))
    lr_gpd = max(0.0, 2 * (ll_gpd - ll_exp))
    p_gpd = float(stats.chi2.sf(lr_gpd, 1))

    xi_gev, xi_gpd = float(gev.xi), float(gpd.xi)
    out.metric("ξ Theorie", xi_true, "num", hint=_domain(xi_true, 1e-9))
    out.metric("ξ aus Blockmaxima (GEV)", xi_gev, "num", hint=f"{_domain(xi_gev)} · {nb} Blöcke à {m} Tage")
    out.metric("ξ aus Peaks over Threshold (GPD)", xi_gpd, "num",
               hint=f"{_domain(xi_gpd)} · {st['n_exceedances']} Überschreitungen")
    p_txt = "p < 0,001" if p_gev < 0.001 else f"p = {_de(p_gev, 3)}"
    out.metric("ξ signifikant ≠ 0? (GEV)", "ja" if p_gev < 0.05 else "nein", "text",
               hint=f"Likelihood-Quotienten-Test gegen Gumbel · {p_txt}")
    out.metric("Größter simulierter Verlust", float(x.max()), "pct")

    # --- chart 1: the three GEV shapes, standardized, plus the fitted shape
    z = np.linspace(-3, 8, 441)
    shapes = [(-0.5, "Weibull ξ = −0,5", 3), (0.0, "Gumbel ξ = 0", "muted"), (0.5, "Fréchet ξ = +0,5", 2)]
    tr = [C.trace(lbl, z, GeneralizedExtremeValue(xi, 0.0, 1.0).pdf(z), color=col, dash=True, width=1.5)
          for xi, lbl, col in shapes]
    tr.append(C.trace(f"Geschätzt ξ = {_de(xi_gev)}", z, GeneralizedExtremeValue(xi_gev, 0.0, 1.0).pdf(z), color=1, width=3))
    out.chart(C.xy(
        "Die drei Formen der GEV-Verteilung (standardisiert)", tr,
        x=C.axis("standardisiertes Maximum (x − μ)/σ"), y=C.axis("Dichte"),
        subtitle="Fréchet hat einen langen rechten Rand, Weibull einen festen Endpunkt.",
    ))

    # --- chart 2: block maxima histogram vs GEV and Gumbel fits
    centers, heights = C.histogram(maxima, bins=int(np.clip(np.sqrt(nb) * 2, 12, 60)), density=True)
    grid = np.linspace(min(maxima.min(), centers[0]), maxima.max() * 1.05 if maxima.max() > 0 else maxima.max() * 0.95, 400)
    out.chart(C.xy(
        f"Blockmaxima ({nb} Blöcke à {m} Tage) und angepasste Verteilungen",
        [C.trace("Blockmaxima", centers, heights, type="bar", color=4),
         C.trace(f"GEV (ξ = {_de(xi_gev)})", grid, gev.pdf(grid), color=1, width=2.5),
         C.trace("Gumbel (ξ = 0)", grid, gumbel.pdf(grid), color="muted", dash=True)],
        x=C.axis("größter Tagesverlust im Block", fmt="pct", digits=1), y=C.axis("Dichte"),
    ))

    # --- chart 3: exceedances, conditional survival on a log scale
    k = len(exc)
    surv_emp = 1.0 - np.arange(k) / k
    ys = np.linspace(0, exc.max() * 1.3, 200)
    out.chart(C.xy(
        f"Überschreitungen der Schwelle u = {_de(100 * st['threshold'])} %: P(Y > y | Verlust > u)",
        [C.trace("Beobachtet", exc, surv_emp, type="step", color=4),
         C.trace(f"GPD (ξ = {_de(xi_gpd)})", ys, np.maximum(1 - gpd.cdf(ys), 1e-12), color=1, width=2.5),
         C.trace("Exponential (ξ = 0)", ys, np.maximum(1 - expo.cdf(ys), 1e-12), color="muted", dash=True)],
        x=C.axis("Überschreitung y = Verlust − u", fmt="pct", digits=1),
        y=C.axis("bedingte Wahrscheinlichkeit", type="log", range=(0.3 / k, 1.2)),
        subtitle="Logarithmisch: Die Exponentialverteilung ist eine Gerade, die GPD mit ξ > 0 biegt nach rechts ab.",
    ))

    # --- chart 4: return level plot in years
    t_blocks = np.geomspace(1.05, 2000, 120)
    years = t_blocks * m / DAYS
    top = min(nb, 400)  # the largest maxima carry the information; keeps the payload small
    rank = np.arange(1, top + 1)
    emp_years = (nb + 1) / rank * m / DAYS
    out.chart(C.xy(
        "Wiederkehrniveau: Blockmaximum, das einmal in T Jahren überschritten wird",
        [C.trace(f"GEV (ξ = {_de(xi_gev)})", years, gev.return_level(t_blocks), color=1, width=2.5),
         C.trace("Gumbel (ξ = 0)", years, gumbel.return_level(t_blocks), color="muted", dash=True),
         C.trace("Beobachtete Blockmaxima", emp_years, np.sort(maxima)[::-1][:top], type="scatter", color=4, size=6,
                 decimate_to=None)],
        x=C.axis("Wiederkehrperiode (Jahre)", type="log"), y=C.axis("Verlust", fmt="pct", digits=0),
        subtitle="Gumbel ist hier eine Gerade. Je größer ξ, desto stärker steigt die Kurve für seltene Ereignisse.",
    ))

    q99_gpd, q99_exp = float(gpd.ppf(0.99)), float(expo.ppf(0.99))
    out.table("Schätzungen im Vergleich", ["Methode", "ξ geschätzt", "ξ Theorie", "Datenpunkte", "LR-Test ξ = 0: p",
                                          "Anpassung"],
              [["GEV (Blockmaxima)", xi_gev, xi_true, nb, p_gev, f"KS p = {_de(float(ks_gev.pvalue), 2)}"],
               ["GPD (Peaks over Threshold)", xi_gpd, xi_true, st["n_exceedances"], p_gpd,
                f"99-%-Überschreitung {_de(100 * q99_gpd)} % (exponentiell {_de(100 * q99_exp)} %)"]],
              formats=["text", "num", "num", "int", "num", "text"],
              note="LR-Test: Likelihood-Quotient gegen das Modell mit ξ = 0 (Gumbel bzw. Exponential), χ² mit 1 Freiheitsgrad.")
    if parent == "gaussian" and xi_gev < -0.03:
        out.note("Normalverteilte Maxima konvergieren nur logarithmisch langsam gegen Gumbel. Bei endlichen Blöcken "
                 "ist ein leicht negatives ξ daher typisch; größere Blöcke bringen die Schätzung näher an 0.", "info")
    if parent == "student_t" and nu <= 2:
        out.note("Mit ν ≤ 2 hat die Student-t unendliche Varianz – ein Fall, den die Normalverteilung gar nicht kennt.",
                 "warning")
