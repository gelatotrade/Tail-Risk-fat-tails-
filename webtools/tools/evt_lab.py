"""EVT laboratory: simulate losses with a known tail, watch GEV and GPD recover the shape parameter."""

import numpy as np
from scipy import stats

from src.models.extreme_value import GeneralizedExtremeValue, GeneralizedParetoDistribution

from .. import charts as C
from ..core import ToolError

DAYS = 252

SPEC = {
    "id": "evt-lab",
    "title": "EVT lab: GEV and GPD",
    "group": "risk",
    "order": 30,
    "summary": "Simulates losses with a known tail and shows how block maxima (GEV) and peaks over threshold "
               "(GPD) recover the shape parameter ξ.",
    "description": (
        "The **Fisher–Tippett theorem** states that the maxima of many losses – whatever their origin – follow one "
        "of three forms of the **GEV distribution**: ξ > 0 (Fréchet) for power-law tails such as the Student-t "
        "(ξ = 1/ν), ξ = 0 (Gumbel) for exponential tails such as the normal distribution, and ξ < 0 (Weibull) for "
        "bounded losses, for example due to a hedge. The same holds for exceedances of a high threshold with the "
        "**generalised Pareto distribution (GPD)**.\n\n"
        "Choose a parent distribution, simulate daily losses and compare the estimated ξ with theory. The "
        "**likelihood ratio test** checks whether ξ differs significantly from 0; for the normal distribution, "
        "maxima converge only very slowly, so estimates of around −0.1 to −0.15 are to be expected.\n\n"
        "In the **return level chart**, the Gumbel form is a straight line on the logarithmic time axis. Fréchet "
        "bends upwards (the next catastrophe is larger than the line suggests); Weibull flattens out. Losses are "
        "positive decimals (0.05 = 5% in one day)."
    ),
    "needs": [],
    "params": [
        {"name": "parent", "type": "select", "default": "student_t", "label": "Parent distribution of losses",
         "options": [{"value": "student_t", "label": "Student-t (Fréchet, ξ = 1/ν)"},
                     {"value": "gaussian", "label": "Normal distribution (Gumbel, ξ = 0)"},
                     {"value": "bounded", "label": "Bounded at 5% (Weibull, ξ = −½)"}]},
        {"name": "nu", "type": "float", "default": 3.0, "min": 1.5, "max": 20.0, "step": 0.5,
         "label": "Student-t: degrees of freedom ν", "help": "ξ = 1/ν.", "show_if": {"parent": ["student_t"]}},
        {"name": "n_obs", "type": "int", "default": 5040, "min": 1000, "max": 50000, "step": 252,
         "label": "Simulated trading days", "help": "5,040 = 20 years."},
        {"name": "block_size", "type": "int", "default": 20, "min": 5, "max": 100, "step": 1,
         "label": "Block size (trading days)"},
        {"name": "threshold_quantile", "type": "float", "default": 0.95, "min": 0.8, "max": 0.995, "step": 0.005,
         "label": "GPD threshold (quantile)"},
        {"name": "seed", "type": "int", "default": 7, "min": 0, "max": 99999, "step": 1, "label": "Random seed"},
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


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _p(p):
    return "< 0.001" if p < 0.001 else _num(p, 3)


def _domain(xi, tol=0.05):
    return "Fréchet" if xi > tol else "Weibull" if xi < -tol else "Gumbel"


def _simulate(parent, nu, n, rng):
    if parent == "student_t":
        return 0.007 * rng.standard_t(nu, n), 1.0 / nu
    if parent == "gaussian":
        return 0.01 * rng.standard_normal(n), 0.0
    return 0.05 * (2.0 * rng.beta(2.0, 2.0, n) - 1.0), -0.5  # endpoint 5%, density ~ (0.05 - x)^1


def run(params, data, out):
    parent, nu, n = params["parent"], float(params["nu"]), int(params["n_obs"])
    m, q, seed = int(params["block_size"]), float(params["threshold_quantile"]), int(params["seed"])
    nb = n // m
    if nb < 10:
        raise ToolError(f"Only {nb:,} blocks of {m} days. Increase the number of days or reduce the block size.")
    if n * (1 - q) < 15:
        raise ToolError("Fewer than 15 losses lie above the threshold. Lower the threshold or simulate more days.")

    rng = np.random.default_rng(seed)
    x, xi_true = _simulate(parent, nu, n, rng)

    out.progress("Fitting GEV to the block maxima …", 0.3)
    gev = GeneralizedExtremeValue.fit_block_maxima(x, m)
    maxima = x[:nb * m].reshape(nb, m).max(axis=1)
    g_loc, g_scale = stats.gumbel_r.fit(maxima)
    gumbel = GeneralizedExtremeValue(0.0, g_loc, g_scale)
    ll_gev = float(np.sum(np.log(np.maximum(gev.pdf(maxima), 1e-300))))
    ll_gum = float(np.sum(np.log(np.maximum(gumbel.pdf(maxima), 1e-300))))
    lr_gev = max(0.0, 2 * (ll_gev - ll_gum))
    p_gev = float(stats.chi2.sf(lr_gev, 1))
    ks_gev = stats.kstest(maxima, gev.cdf)

    out.progress("Fitting GPD to the exceedances …", 0.6)
    gpd, st = GeneralizedParetoDistribution.fit_pot(x, quantile=q)
    exc = np.sort(x[x > st["threshold"]] - st["threshold"])
    expo = GeneralizedParetoDistribution(0.0, float(np.mean(exc)), st["threshold"])  # exponential = MLE for xi = 0
    ll_gpd = float(np.sum(np.log(np.maximum(gpd.pdf(exc), 1e-300))))
    ll_exp = float(np.sum(np.log(np.maximum(expo.pdf(exc), 1e-300))))
    lr_gpd = max(0.0, 2 * (ll_gpd - ll_exp))
    p_gpd = float(stats.chi2.sf(lr_gpd, 1))
    ks_gpd = stats.kstest(exc, gpd.cdf)

    xi_gev, xi_gpd = float(gev.xi), float(gpd.xi)
    out.metric("Theoretical ξ", xi_true, "num", hint=_domain(xi_true, 1e-9))
    out.metric("ξ from block maxima (GEV)", xi_gev, "num", hint=f"{_domain(xi_gev)} · {nb:,} blocks of {m} days")
    out.metric("ξ from peaks over threshold (GPD)", xi_gpd, "num",
               hint=f"{_domain(xi_gpd)} · {st['n_exceedances']:,} exceedances")
    p_txt = "p < 0.001" if p_gev < 0.001 else f"p = {_num(p_gev, 3)}"
    out.metric("ξ significantly ≠ 0 (GEV)", "yes" if p_gev < 0.05 else "no", "text",
               hint=f"Likelihood ratio test vs Gumbel · {p_txt}")
    out.metric("Largest simulated loss", float(x.max()), "pct")

    # --- chart 1: the three GEV shapes, standardized, plus the fitted shape
    z = np.linspace(-3, 8, 441)
    shapes = [(-0.5, "Weibull ξ = −0.5", 3), (0.0, "Gumbel ξ = 0", "muted"), (0.5, "Fréchet ξ = +0.5", 2)]
    tr = [C.trace(lbl, z, GeneralizedExtremeValue(xi, 0.0, 1.0).pdf(z), color=col, dash=True, width=1.5)
          for xi, lbl, col in shapes]
    tr.append(C.trace(f"Estimated ξ = {_num(xi_gev)}", z, GeneralizedExtremeValue(xi_gev, 0.0, 1.0).pdf(z), color=1, width=3))
    out.chart(C.xy(
        "The three forms of the GEV distribution (standardised)", tr,
        x=C.axis("Standardised maximum (x − μ)/σ"), y=C.axis("Density"),
        subtitle="Fréchet has a long right tail; Weibull has a fixed endpoint.",
    ))

    # --- chart 2: block maxima histogram vs GEV and Gumbel fits
    centers, heights = C.histogram(maxima, bins=int(np.clip(np.sqrt(nb) * 2, 12, 60)), density=True)
    grid = np.linspace(min(maxima.min(), centers[0]), maxima.max() * 1.05 if maxima.max() > 0 else maxima.max() * 0.95, 400)
    out.chart(C.xy(
        f"Block maxima ({nb:,} blocks of {m} days) and fitted distributions",
        [C.trace("Block maxima", centers, heights, type="bar", color=4),
         C.trace(f"GEV (ξ = {_num(xi_gev)})", grid, gev.pdf(grid), color=1, width=2.5),
         C.trace("Gumbel (ξ = 0)", grid, gumbel.pdf(grid), color="muted", dash=True)],
        x=C.axis("Largest one-day loss in the block", fmt="pct", digits=1), y=C.axis("Density"),
    ))

    # --- chart 3: exceedances, conditional survival on a log scale
    k = len(exc)
    surv_emp = 1.0 - np.arange(k) / k
    ys = np.linspace(0, exc.max() * 1.3, 200)
    out.chart(C.xy(
        f"Exceedances of the threshold u = {_num(100 * st['threshold'])}%: P(Y > y | loss > u)",
        [C.trace("Observed", exc, surv_emp, type="step", color=4),
         C.trace(f"GPD (ξ = {_num(xi_gpd)})", ys, np.maximum(1 - gpd.cdf(ys), 1e-12), color=1, width=2.5),
         C.trace("Exponential (ξ = 0)", ys, np.maximum(1 - expo.cdf(ys), 1e-12), color="muted", dash=True)],
        x=C.axis("Exceedance y = loss − u", fmt="pct", digits=1),
        y=C.axis("Conditional probability", type="log", range=(0.3 / k, 1.2)),
        subtitle="Log scale: the exponential distribution is a straight line; the GPD with ξ > 0 bends to the right.",
    ))

    # --- chart 4: return level plot in years
    t_blocks = np.geomspace(1.05, 2000, 120)
    years = t_blocks * m / DAYS
    k_max = max(2, min(nb, int((nb + 1) / t_blocks[0])))
    rank = np.unique(np.geomspace(1, k_max, 28).round().astype(int))  # evenly spread on the log axis
    emp_years = (nb + 1) / rank * m / DAYS
    out.chart(C.xy(
        "Return level: block maximum exceeded once in T years",
        [C.trace("Observed block maxima", emp_years, np.sort(maxima)[::-1][rank - 1], type="scatter", color=4,
                 size=5, decimate_to=None),  # first, so the fitted curves stay visible on top
         C.trace("Gumbel (ξ = 0)", years, gumbel.return_level(t_blocks), color="muted", dash=True),
         C.trace(f"GEV (ξ = {_num(xi_gev)})", years, gev.return_level(t_blocks), color=1, width=2.5)],
        x=C.axis("Return period (years)", type="log"), y=C.axis("Loss", fmt="pct", digits=0),
        subtitle="Gumbel is a straight line here. The larger ξ, the more steeply the curve rises for rare events.",
    ))

    out.table("Estimates compared", ["Method", "Estimated ξ", "Theoretical ξ", "Data points",
                                    "LR test ξ = 0: p-value", "KS test: p-value"],
              [["GEV (block maxima)", xi_gev, xi_true, nb, _p(p_gev), _p(float(ks_gev.pvalue))],
               ["GPD (peaks over threshold)", xi_gpd, xi_true, st["n_exceedances"], _p(p_gpd),
                _p(float(ks_gpd.pvalue))]],
              formats=["text", "num", "num", "int", "text", "text"],
              note="LR test: likelihood ratio against the model with ξ = 0 (Gumbel or exponential), χ² with "
                   "1 degree of freedom; small p-value = ξ ≠ 0. KS test: large p-value = the fitted model fits "
                   "(optimistic, because the parameters are estimated from the same data).")
    if parent == "gaussian" and xi_gev < -0.03:
        out.note("Maxima of normally distributed losses converge to Gumbel only logarithmically slowly. With finite "
                 "blocks, a negative ξ is therefore typical – even with infinitely many blocks, about −0.14 for 20 "
                 "and −0.10 for 100 days per block. The LR test rightly detects this deviation from Gumbel; it does "
                 "not, however, mean that the tail is bounded.", "info")
    if parent == "student_t" and xi_true - max(xi_gev, xi_gpd) > 0.08:
        out.note(f"Both estimates are below ξ = 1/ν = {_num(xi_true)}. This is typical of the Student-t: its tail "
                 "becomes a pure power law only far out, so finite blocks and thresholds understate ξ. Larger "
                 "blocks or a higher threshold reduce the bias but increase the variance of the estimate.", "info")
    if parent == "student_t" and nu <= 2:
        out.note("With ν ≤ 2 the Student-t has infinite variance – a case the normal distribution cannot represent "
                 "at all.",
                 "warning")
