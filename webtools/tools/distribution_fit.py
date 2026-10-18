"""Fit every fat-tail family to the data and rank them (KS, AIC), with QQ and log-density views."""

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid

from src.models.fat_tail_distributions import (
    NormalInverseGaussian,
    StudentTDistribution,
    VarianceGamma,
    compare_distributions,
)
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..core import ToolError

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"), (0.995, "99.5%"), (0.999, "99.9%"))]

SPEC = {
    "id": "distribution-fit",
    "title": "Distribution fit and model selection",
    "group": "data",
    "order": 20,
    "summary": "Fits normal, Student-t, NIG and Variance Gamma distributions to your returns and shows which one "
               "captures the tails best.",
    "description": (
        "`compare_distributions` estimates four models: the **normal distribution** and the **Student-t** by "
        "maximum likelihood, **NIG** and **Variance Gamma** by the method of moments (mean, variance, skewness, "
        "kurtosis). The **Kolmogorov-Smirnov distance** measures the largest gap between the empirical and the "
        "model distribution function; the **AIC** rewards a high likelihood and penalises parameters. For both, "
        "smaller is better. The KS p-values are optimistic because the parameters are estimated from the same "
        "data.\n\n"
        "How to read the charts: in the **Q-Q plot** a well-fitting model lies on the diagonal. If the points bend "
        "away at the ends, the model underestimates the extremes. The **logarithmic density** shows the same for "
        "the tails, and the third chart shows the crash frequency P(return < −x) directly.\n\n"
        "VaR and Expected Shortfall per model are one-day losses (positive decimals): analytical for the Gaussian "
        "and Student-t, by Monte Carlo for NIG and Variance Gamma (their crash frequency in the third chart comes "
        "from the numerically integrated density). Moment fits match the kurtosis but not necessarily the shape; "
        "this often shows up as a poor KS value."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Confidence level", "options": LEVELS},
        {"name": "bins", "type": "int", "default": 100, "min": 30, "max": 200, "step": 10,
         "label": "Histogram bins"},
        {"name": "n_mc", "type": "int", "default": 200000, "min": 20000, "max": 1000000, "step": 20000,
         "label": "Monte Carlo draws", "help": "For quantiles, VaR and ES of NIG and Variance Gamma."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.models.fat_tail_distributions.compare_distributions",
        "src.models.fat_tail_distributions.StudentTDistribution.fit",
        "src.models.fat_tail_distributions.NormalInverseGaussian.fit",
        "src.models.fat_tail_distributions.VarianceGamma.fit",
        "src.models.fat_tail_distributions.StudentTDistribution.ppf",
        "src.models.fat_tail_distributions.StudentTDistribution.cdf",
        "src.models.fat_tail_distributions.StudentTDistribution.var",
        "src.models.fat_tail_distributions.StudentTDistribution.expected_shortfall",
        "src.models.fat_tail_distributions.NormalInverseGaussian.pdf",
        "src.models.fat_tail_distributions.NormalInverseGaussian.sample",
        "src.models.fat_tail_distributions.VarianceGamma.pdf",
        "src.models.fat_tail_distributions.VarianceGamma.sample",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
    ],
    "cost": "medium",
}

NAMES = {"gaussian": "Normal distribution", "student_t": "Student-t", "nig": "NIG", "variance_gamma": "Variance Gamma"}
COLORS = {"gaussian": "muted", "student_t": 1, "nig": 2, "variance_gamma": 3}
N_PARAMS = {"gaussian": 2, "student_t": 3, "nig": 4, "variance_gamma": 4}


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _p(p):
    """p-value as text; tiny values as '< 0.001' instead of 3.0e-26."""
    if p is None or not np.isfinite(p):
        return "–"
    return "< 0.001" if p < 0.001 else _num(p, 3)


def _p_text(p):
    txt = _p(p)
    return f"p {txt}" if txt.startswith("<") else f"p = {txt}"


def _left_tail(pdf, xs, far):
    """P(R < -x) for each loss x in xs by integrating the density from -far (smooth, no Monte Carlo noise)."""
    grid = -np.geomspace(far, float(np.min(xs)), 6000)
    f = np.nan_to_num(np.asarray(pdf(grid), dtype=float), nan=0.0, posinf=0.0)
    cum = cumulative_trapezoid(f, grid, initial=0.0)
    return np.interp(-np.asarray(xs), grid, cum)


def _var_es(x, conf):
    var = -float(np.quantile(x, 1 - conf))
    tail = x[x <= -var]
    return var, (-float(np.mean(tail)) if tail.size else var)


def _loglik(pdf, r):
    f = np.asarray(pdf(r), dtype=float)
    return float(np.sum(np.log(np.clip(np.nan_to_num(f, nan=0.0), 1e-300, None))))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    if np.std(r) <= 0:
        raise ToolError("The returns are constant – there is no distribution to fit.")
    conf, seed, n_mc = float(params["confidence"]), int(params["seed"]), int(params["n_mc"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    out.progress("Fitting four distributions …", 0.1)
    res = compare_distributions(r)
    best = res["best_fit"]

    gp = res["gaussian"]["params"]
    gauss = stats.norm(gp["loc"], gp["scale"])
    tp = res["student_t"]["params"]
    t_dist = StudentTDistribution(nu=tp["nu"], loc=tp["loc"], scale=tp["scale"])
    nig = NormalInverseGaussian.fit(r)
    vg = VarianceGamma.fit(r)

    out.progress("Monte Carlo for NIG and Variance Gamma …", 0.5)
    s_nig = np.asarray(nig.sample(n_mc, seed=seed), dtype=float)
    s_vg = np.asarray(vg.sample(n_mc, seed=seed + 1), dtype=float)

    models = {
        "gaussian": {"pdf": gauss.pdf, "ppf": gauss.ppf, "cdf": gauss.cdf, "sample": None,
                     "par": f"μ = {_num(100 * gp['loc'], 3)}%, σ = {_num(100 * gp['scale'])}%"},
        "student_t": {"pdf": t_dist.pdf, "ppf": t_dist.ppf, "cdf": t_dist.cdf, "sample": None,
                      "par": f"ν = {_num(tp['nu'])}, scale = {_num(100 * tp['scale'])}%"},
        "nig": {"pdf": nig.pdf, "ppf": None, "cdf": None, "sample": s_nig,
                "par": f"α = {_num(nig.alpha, 1)}, β = {_num(nig.beta, 1)}, δ = {_num(100 * nig.delta)}%"},
        "variance_gamma": {"pdf": vg.pdf, "ppf": None, "cdf": None, "sample": s_vg,
                           "par": f"σ = {_num(100 * vg.sigma)}%, θ = {_num(100 * vg.theta, 3)}%, ν = {_num(vg.nu)}"},
    }
    alpha = 1 - conf
    for key, m in models.items():
        m["ll"] = _loglik(m["pdf"], r)
        m["aic"] = 2 * N_PARAMS[key] - 2 * m["ll"]
        m["ks"] = res[key].get("ks_statistic")
        m["p"] = res[key].get("ks_pvalue")
        if key == "gaussian":
            z = stats.norm.ppf(alpha)
            m["var"], m["es"] = -(gp["loc"] + gp["scale"] * z), -gp["loc"] + gp["scale"] * stats.norm.pdf(z) / alpha
        elif key == "student_t":
            m["var"], m["es"] = float(t_dist.var(conf)), float(t_dist.expected_shortfall(conf))
        else:
            m["var"], m["es"] = _var_es(m["sample"], conf)
    aic_min = min(m["aic"] for m in models.values())
    best_aic = min(models, key=lambda k: models[k]["aic"])
    trm = TailRiskMetrics(r)
    var_h, es_h = float(trm.var_historical(conf)), float(trm.expected_shortfall(conf, "historical"))

    out.metric("Best model (KS)", NAMES[best], "text", hint=f"KS distance {_num(models[best]['ks'], 3)}")
    out.metric("Best model (AIC)", NAMES[best_aic], "text")
    out.metric("Student-t: degrees of freedom ν", tp["nu"], "num", hint="< 4: infinite kurtosis · < 2: infinite variance",
               status="critical" if tp["nu"] < 2.5 else "serious" if tp["nu"] < 4 else "warning" if tp["nu"] < 8 else None)
    out.metric(f"{lvl} VaR · historical", var_h, "pct", hint=f"ES {_num(100 * es_h)}%")
    out.metric(f"{lvl} VaR · {NAMES[best]}", models[best]["var"], "pct", hint=f"ES {_num(100 * models[best]['es'])}%")
    out.metric(f"{lvl} VaR · Gaussian", models["gaussian"]["var"], "pct", hint=f"ES {_num(100 * models['gaussian']['es'])}%")

    order = sorted(models, key=lambda k: models[k]["ks"] if models[k]["ks"] is not None else np.inf)
    rows = []
    for rank, key in enumerate(order, 1):
        m = models[key]
        rows.append([rank, NAMES[key], m["par"], m["ks"], _p(m["p"]), m["ll"], m["aic"] - aic_min, m["var"], m["es"]])
    rows.append(["", "Empirical", f"{n:,} days", None, None, None, None, var_h, es_h])
    out.table(
        "Model comparison (sorted by KS distance)",
        ["Rank", "Model", "Parameters", "KS distance", "KS p-value", "Log-likelihood", "ΔAIC", f"{lvl} VaR", f"{lvl} ES"],
        rows, formats=["text", "text", "text", "num", "text", "num", "num", "pct", "pct"],
        note="ΔAIC = AIC − best AIC (0 = best model; > 10 = clearly worse). NIG and Variance Gamma are moment fits, "
             "not maximum likelihood estimates; their log-likelihood is therefore not maximised.",
    )

    # --- chart 1: histogram vs fitted densities on a log scale
    centers, heights = C.histogram(r, bins=int(params["bins"]), density=True)
    floor = float(np.min(heights[heights > 0])) / 4
    lo, hi = float(r.min()), float(r.max())
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 600)
    traces = [C.trace("Observed", centers, np.where(heights > 0, heights, np.nan), type="bar", color=4)]
    for key in ("gaussian", "student_t", "nig", "variance_gamma"):
        y = np.asarray(models[key]["pdf"](grid), dtype=float)
        y = np.where(np.isfinite(y) & (y >= floor), np.minimum(y, 10 * heights.max()), np.nan)
        traces.append(C.trace(NAMES[key], grid, y, color=COLORS[key], dash=key == "gaussian"))
    out.chart(C.xy(
        "Daily returns and fitted densities – log scale", traces,
        x=C.axis("Daily return", fmt="pct", digits=1), y=C.axis("Density", type="log", range=(floor, heights.max() * 3)),
        subtitle="Where the bars lie above a curve, the model underestimates the frequency.",
    ))

    # --- chart 2: QQ plot (tails in full, the centre thinned)
    rs = np.sort(r)
    p = (np.arange(1, n + 1) - 0.5) / n
    edge = min(150, n // 4)
    idx = np.unique(np.concatenate([np.arange(edge), np.linspace(edge, n - edge - 1, 200).astype(int),
                                    np.arange(n - edge, n)]))
    qq = [C.trace("Perfect fit", [rs[0], rs[-1]], [rs[0], rs[-1]], color="ink", dash=True, width=1)]
    for key in reversed(order):  # best model last, i.e. drawn on top
        m = models[key]
        q = np.asarray(m["ppf"](p[idx]), dtype=float) if m["ppf"] is not None else np.quantile(m["sample"], p[idx])
        qq.append(C.trace(NAMES[key], q, rs[idx], type="scatter", color=COLORS[key], size=5, decimate_to=None))
    span = (rs[0] * 1.3, rs[-1] * 1.3)
    out.chart(C.xy(
        "Q-Q plot: model quantiles vs observed quantiles", qq,
        x=C.axis("Model quantile", fmt="pct", digits=0, range=span), y=C.axis("Observed quantile", fmt="pct", digits=0),
        subtitle="Points below the diagonal on the left: actual crashes are deeper than the model expects.",
    ))

    # --- chart 3: left tail, P(r < -x)
    losses = np.sort(-r[r < 0])[::-1]
    xs = np.geomspace(max(np.median(losses), 1e-4), losses[0] * 1.5, 70)
    emp = np.array([np.mean(r < -x) for x in xs])
    tail = [C.trace("Observed", xs, np.where(emp > 0, emp, np.nan), type="scatter", color=4, size=6, decimate_to=None)]
    far = max(1.0, 20 * float(losses[0]))  # NIG and VG tails are exponential: no mass left out there
    y_floor = min(1e-6, 0.3 / n)
    for key in ("gaussian", "student_t", "nig", "variance_gamma"):
        m = models[key]
        y = (np.asarray(m["cdf"](-xs), dtype=float) if m["cdf"] is not None
             else _left_tail(m["pdf"], xs, far))
        tail.append(C.trace(NAMES[key], xs, np.where(y > y_floor, y, np.nan), color=COLORS[key],
                            dash=key == "gaussian"))
    out.chart(C.xy(
        "Crash frequency: P(daily return < −x)", tail,
        x=C.axis("One-day loss x", type="log", fmt="pct", digits=0), y=C.axis("Probability per day", type="log",
                                                                                 range=(y_floor, 1)),
        subtitle="Log-log axes. The normal distribution drops away; the fat-tailed models follow the data for longer.",
        hlines=[C.hline(1 / 252, "once a year")],
    ))

    if models["gaussian"]["p"] is not None and models["gaussian"]["p"] < 0.05 and best != "gaussian":
        bp = models[best]["p"]
        fits = bp is not None and bp >= 0.05
        out.set_verdict(
            "good" if fits else "warning",
            f"{NAMES[best]} describes the returns best",
            f"The normal distribution is rejected (KS test: {_p_text(models['gaussian']['p'])}). "
            + (f"{NAMES[best]} is not rejected by the KS test (p = {_num(bp, 2)})."
               if fits else f"However, {NAMES[best]} does not fit perfectly either ({_p_text(bp)}); "
                            "no fixed distribution can capture volatility clustering."),
        )
    elif best == "gaussian":
        out.set_verdict("good", "The normal distribution fits best here",
                        "This can happen with short or calm series; the fat-tailed models offer no advantage.")
