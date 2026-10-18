"""Extreme value theory on the data: GPD peaks over threshold, GEV block maxima, return levels."""

import numpy as np
from scipy import stats

from src.models.extreme_value import EVTTailRiskAnalyzer, hill_plot, mean_excess_plot

from .. import charts as C
from ..core import ToolError

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"), (0.995, "99.5%"), (0.999, "99.9%"))]
DAYS = 252

SPEC = {
    "id": "evt",
    "title": "Extreme value theory (EVT)",
    "group": "risk",
    "order": 20,
    "summary": "Models the tail only – GPD above a threshold, GEV for block maxima – with VaR, Expected "
               "Shortfall and the losses expected once in 10 or 100 years.",
    "description": (
        "Extreme value theory does not fit a model to every day, only to the extreme ones. **Peaks over "
        "threshold:** all losses above a threshold u approximately follow a **generalised Pareto distribution "
        "(GPD)**. Its shape parameter ξ is the inverse of the tail index α (ξ = 1/α): ξ > 0 means a fat tail (Fréchet domain; only moments "
        "below order 1/ξ exist), ξ = 0 an exponential tail as in the Gaussian case. **Block maxima:** the largest "
        "loss per block (e.g. 20 trading days) follows a **GEV distribution**.\n\n"
        "Choosing the threshold: in the **mean excess plot**, the mean exceedance above u should be roughly "
        "linear; in the **Hill plot**, the estimate should be stable. Thresholds that are too low introduce bias, "
        "thresholds that are too high leave too few points. The **return level chart** shows which one-day loss is "
        "exceeded on average once in T years (252 trading days per year) – GPD, GEV, Gaussian and the data "
        "compared.\n\n"
        "All losses are daily log returns expressed as positive decimals; the 10-year and 100-year losses at the "
        "top come from the GPD because it models the individual day directly. The table from "
        "`comprehensive_report` also shows by what factor the normal distribution understates 3σ, 4σ and 5σ days."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold_quantile", "type": "float", "default": 0.95, "min": 0.85, "max": 0.99, "step": 0.005,
         "label": "Threshold u (quantile)", "help": "0.95 = the largest 5% of losses enter the GPD fit."},
        {"name": "block_size", "type": "int", "default": 20, "min": 5, "max": 63, "step": 1,
         "label": "GEV block size (trading days)", "help": "20 ≈ one month, 63 ≈ one quarter."},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Confidence level", "options": LEVELS},
    ],
    "covers": [
        "src.models.extreme_value.EVTTailRiskAnalyzer",
        "src.models.extreme_value.EVTTailRiskAnalyzer.tail_index",
        "src.models.extreme_value.EVTTailRiskAnalyzer.var",
        "src.models.extreme_value.EVTTailRiskAnalyzer.expected_shortfall",
        "src.models.extreme_value.EVTTailRiskAnalyzer.return_level",
        "src.models.extreme_value.EVTTailRiskAnalyzer.tail_comparison_to_gaussian",
        "src.models.extreme_value.EVTTailRiskAnalyzer.comprehensive_report",
        "src.models.extreme_value.GeneralizedParetoDistribution.fit_pot",
        "src.models.extreme_value.GeneralizedParetoDistribution.var",
        "src.models.extreme_value.GeneralizedParetoDistribution.expected_shortfall",
        "src.models.extreme_value.GeneralizedParetoDistribution.tail_probability",
        "src.models.extreme_value.GeneralizedExtremeValue.fit_block_maxima",
        "src.models.extreme_value.mean_excess_plot",
        "src.models.extreme_value.hill_plot",
    ],
    "cost": "fast",
}


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _once_in(x):
    """'once in … years' for a return period in years; 'never' if it is infinite."""
    if not np.isfinite(x):
        return "never"
    if x >= 1e6:
        mant, exp = f"{x:.1e}".split("e")
        return f"once in {mant} · 10{str(int(exp)).translate(SUP)} years"
    if x >= 100:
        return f"once in {x:,.0f} years"
    return f"once in {_num(x, 1)} years"


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    q, m, conf = float(params["threshold_quantile"]), int(params["block_size"]), float(params["confidence"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])
    if n * (1 - q) < 10:
        lo = next(p["min"] for p in SPEC["params"] if p["name"] == "threshold_quantile")
        if n * (1 - lo) < 10:
            raise ToolError(f"The series is too short for extreme value theory ({n:,} days): even at the "
                            f"{_num(100 * lo, 0)}% quantile, fewer than 10 losses would remain above the threshold. "
                            f"Use a series of at least {int(np.ceil(10 / (1 - lo))):,} days.")
        raise ToolError(f"With {n:,} days, fewer than 10 losses remain above the {_num(100 * q, 1)}% quantile. "
                        f"Lower the threshold to at most {_num(1 - 10 / n, 3)}.")
    if n // m < 10:
        raise ToolError(f"Blocks of {m} days give only {n // m:,} block maxima (at least 10 required). "
                        "Reduce the block size.")

    out.progress("Fitting GPD and GEV to both tails …", 0.2)
    try:
        evt = EVTTailRiskAnalyzer(r, block_size=m, threshold_quantile=q)
    except ValueError as exc:
        raise ToolError(f"The EVT fit is not possible: {exc}")
    report = evt.comprehensive_report()

    xi_l, xi_r = float(evt.tail_index("left")), float(evt.tail_index("right"))
    st = evt.stats_left
    u, p_u = float(st["threshold"]), float(st["exceedance_probability"])
    var_c, es_c = float(evt.var(conf, "gpd")), float(evt.expected_shortfall(conf))
    var_gev = float(evt.var(conf, "gev"))
    rl10, rl100 = -float(report["return_levels"]["10_year"]), -float(report["return_levels"]["100_year"])  # GEV
    t10, t100 = 10 * DAYS, 100 * DAYS
    gpd10, gpd100 = float(evt.gpd_left.var(1 - 1 / t10, p_u)), float(evt.gpd_left.var(1 - 1 / t100, p_u))
    mu, sd = float(np.mean(r)), float(np.std(r))

    def gauss_var(c):
        return -(mu + sd * stats.norm.ppf(1 - c))

    def gauss_years(loss):
        p = stats.norm.cdf((-loss - mu) / sd)
        return 1 / p / DAYS if p > 0 else np.inf

    out.metric("Shape parameter ξ · losses", xi_l, "num", hint="GPD shape: > 0 fat tail, 0 Gaussian-like",
               status="critical" if xi_l > 0.4 else "serious" if xi_l > 0.25 else "warning" if xi_l > 0.1 else "good")
    out.metric("Shape parameter ξ · gains", xi_r, "num")
    out.metric("Threshold u", u, "pct", hint=f"{st['n_exceedances']:,} losses above it ({_num(100 * p_u, 1)}%)")
    out.metric(f"{lvl} VaR · GPD", var_c, "pct", hint=f"Gaussian {_num(100 * gauss_var(conf))}%")
    out.metric(f"{lvl} Expected Shortfall · GPD", es_c, "pct")
    out.metric(f"{lvl} VaR · GEV", var_gev, "pct", hint=f"from {m}-day block maxima")
    out.metric("10-year one-day loss · GPD", gpd10, "pct",
               hint=f"GEV {_num(100 * rl10, 1)}% · Gaussian: {_once_in(gauss_years(gpd10))}")
    out.metric("100-year one-day loss · GPD", gpd100, "pct",
               hint=f"GEV {_num(100 * rl100, 1)}% · Gaussian: {_once_in(gauss_years(gpd100))}")

    # --- chart 1: tail fit, P(loss > x)
    losses = np.sort(-r)[::-1]
    xs = np.linspace(u, max(losses[0] * 1.4, u * 2), 120)
    gpd_tail = np.array([evt.gpd_left.tail_probability(x, p_u) for x in xs], dtype=float)
    emp = np.array([np.mean(-r > x) for x in xs])
    gauss_tail = stats.norm.cdf((-xs - mu) / sd)
    floor = min(0.2 / n, 1e-5)
    keep = lambda y: np.where(y > floor * 0.5, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Loss tail: P(one-day loss > x) above the threshold",
        [C.trace("Observed", xs, np.where(emp > 0, emp, np.nan), type="step", color=4, width=2),
         C.trace(f"GPD (ξ = {_num(xi_l)})", xs, keep(gpd_tail), color=1, width=2.5),
         C.trace("Normal distribution", xs, keep(gauss_tail), color="muted", dash=True)],
        x=C.axis("One-day loss x", fmt="pct", digits=0), y=C.axis("Probability per day", type="log",
                                                                   range=(floor, p_u * 1.5)),
        subtitle="The GPD follows the observed extremes; the Gaussian curve drops away.",
        hlines=[C.hline(1 / DAYS, "once a year"), C.hline(1 / (10 * DAYS), "once in 10 years")],
    ))

    # --- chart 2: return levels in years
    years = np.geomspace(0.1, 200, 90)
    t_days = years * DAYS
    gev_rl = -np.asarray(evt.return_level(t_days, "left"), dtype=float)
    gpd_rl = np.array([evt.gpd_left.var(1 - 1 / T, p_u) if 1 / T < p_u else np.nan for T in t_days], dtype=float)
    g_rl = -(mu + sd * stats.norm.ppf(1 / t_days))
    k_max = max(2, min(n, int((n + 1) / (years[0] * DAYS))))  # observed points inside the 0.1-200-year axis
    ranks = np.unique(np.geomspace(1, k_max, 28).round().astype(int))  # evenly spread on the log axis
    emp_years = (n + 1) / ranks / DAYS
    out.chart(C.xy(
        "Return levels: the one-day loss expected once in T years",
        [C.trace("Observed", emp_years, losses[ranks - 1], type="scatter", color=4, size=5, decimate_to=None),
         C.trace("Normal distribution", years, g_rl, color="muted", dash=True),
         C.trace(f"GEV ({m}-day block maxima)", years, gev_rl, color=2, width=2.5),
         C.trace("GPD (peaks over threshold)", years, gpd_rl, color=1, width=2.5)],
        x=C.axis("Return period (years)", type="log"), y=C.axis("One-day loss", fmt="pct", digits=0),
        subtitle="Dots: observed losses; the k-th largest loss occurs on average once in (n + 1)/k days.",
        vlines=[C.vline(10, "10 years"), C.vline(100, "100 years")],
    ))

    # --- chart 3: mean excess plot for threshold choice
    me = mean_excess_plot(-r)
    ok = np.isfinite(me["mean_excess"])
    out.chart(C.xy(
        "Mean excess plot: mean exceedance e(u)",
        [C.trace("e(u) = E[loss − u | loss > u]", np.asarray(me["thresholds"])[ok], me["mean_excess"][ok],
                 type="scatter", color=1, size=6, decimate_to=None)],
        x=C.axis("Threshold u (one-day loss)", fmt="pct", digits=1), y=C.axis("Mean exceedance", fmt="pct", digits=2),
        subtitle="If e(u) rises linearly above the threshold, the GPD fits; a positive slope means ξ > 0.",
        vlines=[C.vline(u, "selected threshold", color="critical")],
    ))

    # --- chart 4: Hill plot on the loss tail
    loss_only = -r[r < 0]
    hp = hill_plot(loss_only)
    kv, hv = np.asarray(hp["k_values"]), np.asarray(hp["hill_estimates"], dtype=float)
    inv_xi = 1 / xi_l if xi_l > 0.02 else np.nan
    hlines = [C.hline(inv_xi, f"1/ξ (GPD) = {_num(inv_xi, 1)}", color="critical")] if np.isfinite(inv_xi) else []
    if len(kv) and np.any(np.isfinite(hv)):
        half = 1.96 * hv / np.sqrt(kv)  # asymptotic standard error of Hill: alpha / sqrt(k)
        top = np.nanmax([np.nanmax(hv + half), inv_xi if np.isfinite(inv_xi) and inv_xi < 3 * np.nanmax(hv) else 0])
        out.chart(C.xy(
            "Hill plot: tail index α of losses vs number of extremes k",
            [C.envelope("95% band", kv, np.maximum(hv - half, 0), hv + half, color=1),
             C.trace("Hill estimator α(k)", kv, hv, color=1)],
            x=C.axis("k (largest losses)", fmt="int"), y=C.axis("Tail index α", range=(0, top * 1.1)),
            subtitle="A stable plateau indicates a credible tail index; α = 1/ξ, and α < 4 means infinite kurtosis. "
                     "Hill and the GPD fit often differ markedly – both are uncertain, as the band shows.",
            hlines=hlines, vlines=[C.vline(st["n_exceedances"], f"k = {st['n_exceedances']}")],
        ))

    # --- tables
    hist_level = lambda T: float(np.quantile(-r, 1 - 1 / T)) if T <= n else None  # noqa: E731
    rows = [
        ["99% VaR", report["var_99"], float(evt.var(0.99, "gev")), float(np.quantile(-r, 0.99)), gauss_var(0.99)],
        ["99.5% VaR", report["var_995"], float(evt.var(0.995, "gev")), float(np.quantile(-r, 0.995)), gauss_var(0.995)],
        ["99.9% VaR", report["var_999"], float(evt.var(0.999, "gev")), float(np.quantile(-r, 0.999)), gauss_var(0.999)],
        ["99% Expected Shortfall", report["es_99"], None, float(np.mean(-r[-r >= np.quantile(-r, 0.99)])),
         -mu + sd * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01],
        ["10-year one-day loss", gpd10, rl10, hist_level(t10), gauss_var(1 - 1 / t10)],
        ["100-year one-day loss", gpd100, rl100, hist_level(t100), gauss_var(1 - 1 / t100)],
    ]
    out.table("Risk metrics: EVT vs historical and Gaussian",
              ["Metric", "GPD (POT)", f"GEV ({m} days)", "Historical", "Gaussian"], rows,
              formats=["text", "pct", "pct", "pct", "pct"],
              note="Return levels in trading days: 10 years = 2,520 days. Historical values are only available if "
                   "the series is at least that long (otherwise –); the GEV yields no Expected Shortfall.")

    gc = report["gaussian_comparison"]
    out.table("How far the normal distribution understates extreme days",
              ["Threshold", "Gaussian", "Observed", "GPD", "Factor (observed)", "Factor (GPD)"],
              [[f"{k}σ", v["gaussian_probability"], v["empirical_probability"], v["gpd_probability"],
                v["underestimation_factor_empirical"], v["underestimation_factor_gpd"]]
               for k, v in zip((3, 4, 5), (gc["3sigma"], gc["4sigma"], gc["5sigma"]))],
              formats=["text", "pct", "pct", "pct", "x", "x"],
              note="Probability per day of |return| > kσ (both tails). GPD: fitted to both tails.")

    sr = evt.stats_right
    out.table("Fitted models", ["Tail", "Threshold u", "Exceedances", "ξ (GPD)", "σ (GPD)", "ξ (GEV)"],
              [["Losses", u, st["n_exceedances"], xi_l, st["sigma"], evt.gev_minima.xi],
               ["Gains", sr["threshold"], sr["n_exceedances"], xi_r, sr["sigma"], evt.gev_maxima.xi]],
              formats=["text", "pct", "int", "num", "pct", "num"],
              note=f"GEV fitted to {n // m:,} block maxima of {m} trading days each.")

    if var_gev < 0.9 * var_c:
        out.note("The GEV VaR is below the GPD VaR: converting a block maximum into a single-day figure assumes "
                 "independent days. Losses, however, arrive in clusters, so a block often contains several extreme "
                 "days – the GPD figure is the more reliable one here.", "info")

    yrs = gauss_years(gpd10)
    if xi_l > 0.1:
        out.set_verdict(
            "serious" if xi_l > 0.25 else "warning",
            f"Fat loss tail (Fréchet domain, ξ = {_num(xi_l)})",
            f"The normal distribution treats the 10-year one-day loss of {_num(100 * gpd10, 1)}% (GPD) as "
            + (f"an event that occurs {_once_in(yrs)}" if np.isfinite(yrs) else "practically impossible")
            + ". Gaussian-based stress tests understate the extremes here.",
        )
    elif xi_l > -0.05:
        out.set_verdict("good", f"The loss tail is roughly exponential (ξ = {_num(xi_l)})",
                        "Above the threshold, losses behave almost as under a normal or exponential distribution; "
                        "use the Hill plot to check whether the threshold is appropriate.")
    else:
        out.set_verdict("good", f"Bounded loss tail (ξ = {_num(xi_l)})",
                        "The largest losses grow more slowly than exponentially (Weibull domain).")
