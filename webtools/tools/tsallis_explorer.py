"""Explore the q-Gaussian: how the Tsallis index q changes density, quantiles, tails and entropy."""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisDistribution, TsallisEntropyAnalyzer

from .. import charts as C

SPEC = {
    "id": "tsallis-explorer",
    "title": "Tsallis distribution (q-Gaussian)",
    "group": "physics",
    "order": 70,
    "summary": "How the entropic index q turns the Gaussian bell into a distribution with power-law tails – "
               "density, quantiles, samples and Tsallis entropy.",
    "description": (
        "The **q-Gaussian** P(x) ∝ exp_q(−βx²) maximises the Tsallis entropy S_q = (1 − Σpᵢ^q)/(q − 1), just as "
        "the normal distribution maximises the Shannon entropy. For q > 1 the q-exponential "
        "exp_q(u) = [1 + (1 − q)u]^(1/(1−q)) decays not exponentially but like a power – hence the fat tails "
        "with density ~ |x|^(−2/(q−1)). q = 1 is exactly the normal distribution; q ≈ 1.4–1.5 typically fits "
        "daily returns. Mathematically it is a scaled Student-t distribution with ν = (3 − q)/(q − 1) degrees of "
        "freedom.\n\n"
        "How to read the charts: the Gaussian reference has the same variance (for q ≥ 5/3 the variance is "
        "infinite, so the same β is used instead). On the log scale the q-Gaussian curve lies higher in the "
        "centre, lower at the shoulders and far above in the tails – the typical picture of leptokurtic returns. "
        "On the log-log plot the tail is a straight line; the dashed asymptote from `tail_probability` meets it "
        "far out.\n\n"
        "x is given in units of 1/√β (β = 1, μ = 0). Quantiles come exactly from `ppf`, the samples from the "
        "Student-t representation."
    ),
    "needs": [],
    "params": [
        {"name": "q", "type": "float", "default": 1.4, "min": 1.0, "max": 2.9, "step": 0.05,
         "label": "Entropic index q", "help": "1 = Gaussian · 5/3: variance becomes infinite · 2: Cauchy-like."},
        {"name": "x_range", "type": "int", "default": 8, "min": 3, "max": 30, "step": 1,
         "label": "Density range ± (in 1/√β)"},
        {"name": "n_samples", "type": "int", "default": 20000, "min": 1000, "max": 200000, "step": 1000,
         "label": "Sample size"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisDistribution",
        "src.physics.tsallis_statistics.TsallisDistribution.q_exponential",
        "src.physics.tsallis_statistics.TsallisDistribution.q_logarithm",
        "src.physics.tsallis_statistics.TsallisDistribution.pdf",
        "src.physics.tsallis_statistics.TsallisDistribution.cdf",
        "src.physics.tsallis_statistics.TsallisDistribution.ppf",
        "src.physics.tsallis_statistics.TsallisDistribution.sample",
        "src.physics.tsallis_statistics.TsallisDistribution.variance",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_exponent",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_probability",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.compute_tsallis_entropy",
    ],
    "cost": "fast",
}

QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _tiny(p):
    """Probability cell: below 10⁻⁶ as scientific text in % (2.0·10⁻⁷% instead of 0.0000002%)."""
    if p is None or not np.isfinite(p):
        return None
    if p < 1e-12:  # includes exact 0 from underflow (Gaussian tail far out)
        return "< 10⁻¹⁰%"
    if 0 < p < 1e-6:
        mant, exp = f"{100 * p:.1e}".split("e")
        return mant + "·10" + str(int(exp)).translate(_SUP) + "%"
    return float(p)


def run(params, data, out):
    q, xr, n, seed = params["q"], float(params["x_range"]), params["n_samples"], params["seed"]
    dist = TsallisDistribution(q=q, beta=1.0, mu=0.0)
    var = float(dist.variance())
    same_var = np.isfinite(var)
    g_sd = np.sqrt(var) if same_var else np.sqrt(1 / (2 * dist.beta))
    ref_label = "Gaussian, equal variance" if same_var else "Gaussian, same β"
    nu = (3 - q) / (q - 1) if q > 1 else np.inf

    s = np.asarray(dist.sample(n, seed=seed), dtype=float)
    s_sorted = np.sort(s)

    # --- metrics
    tail_exp = float(dist.tail_exponent())
    out.metric("Tail exponent of the density", tail_exp if np.isfinite(tail_exp) else "∞ (Gaussian)",
               "num" if np.isfinite(tail_exp) else "text", hint="P(x) ~ |x|^−2/(q−1)",
               status="critical" if tail_exp < 3 else "serious" if tail_exp < 5 else None)
    out.metric("Student-t degrees of freedom ν", nu if np.isfinite(nu) else "∞", "num" if np.isfinite(nu) else "text",
               hint="(3 − q)/(q − 1); moments exist up to order < ν")
    out.metric("Variance", var if same_var else "∞", "num" if same_var else "text",
               hint="1/(β(5 − 3q)), finite only for q < 5/3")
    p99, g99 = float(dist.ppf(0.99)), float(stats.norm.ppf(0.99, 0, g_sd))
    out.metric("99% quantile", p99, "num", hint=f"{ref_label}: {_fmt(g99)}")
    p4 = float(dist.tail_probability(4 * g_sd, "both", exact=True))
    g4 = float(2 * stats.norm.sf(4))
    out.metric("P(|X| > 4σ)", p4, "pct", hint=f"σ of the Gaussian reference; Gaussian: {_fmt(100 * g4, 4)}%",
               status="critical" if p4 > 50 * g4 else "serious" if p4 > 5 * g4 else None)

    # --- chart 1: density on log scale with histogram and Gaussian reference
    x = np.linspace(-xr, xr, 601)
    pdf = dist.pdf(x)
    gpdf = stats.norm.pdf(x, 0, g_sd)
    edges = np.linspace(-xr, xr, 121)
    w = edges[1] - edges[0]
    dens = np.histogram(s, bins=edges)[0] / (len(s) * w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    floor = 0.5 / (len(s) * w)
    lo = max(1e-9, min(floor, float(pdf.min()) / 2))
    keep = lambda y: np.where(y >= lo, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Density on a logarithmic scale",
        [C.trace("Sample", centers, np.where(dens > 0, dens, np.nan), type="bar", color=1),
         C.trace(f"q-Gaussian (q = {_fmt(q)})", x, keep(pdf), color=2, width=2.5),
         C.trace(ref_label, x, keep(gpdf), color="muted", dash=True)],
        x=C.axis("x (in 1/√β)"), y=C.axis("Density", type="log", range=(lo, 2.5 * max(pdf.max(), gpdf.max()))),
        subtitle="Higher peak, narrower shoulders and much heavier tails than the Gaussian parabola.",
    ))

    # --- chart 2: tails on log-log axes
    dmax = float(np.clip(np.max(np.abs(s)), 10.0, 1e6))
    d = np.logspace(np.log10(0.3), np.log10(dmax), 70)
    exact = dist.cdf(-d)                       # P(X < -d), exact Student-t cdf
    asym = np.asarray(dist.tail_probability(d, "left"), dtype=float) if q > 1 else np.full_like(d, np.nan)
    emp = np.searchsorted(s_sorted, -d, side="left") / len(s)
    y_lo = 0.3 / len(s)
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Left tail on a log-log plot: P(X < −x)",
        [C.trace("Sample", d, pos(np.where(emp > 0, emp, np.nan)), color=1, width=2.5),
         C.trace("Exact (cdf)", d, pos(exact), color=2),
         C.trace("Power-law asymptote", d, pos(asym), color=2, dash=True, width=1.5),
         C.trace(ref_label, d, pos(stats.norm.cdf(-d, 0, g_sd)), color="muted", dash=True)],
        x=C.axis("x (in 1/√β)", type="log"), y=C.axis("Probability", type="log", fmt="pct", range=(y_lo, 1)),
        subtitle=(f"Slope of the straight line: 1 − 2/(q − 1) = {_fmt(1 - tail_exp)}.".replace("-", "−") if q > 1 else
                  "q = 1: there is no power law; the tail decays as for the normal distribution."),
    ))

    # --- chart 3: the q-exponential behind it all
    u = np.linspace(0, 20, 401)
    eq = dist.q_exponential(-u)
    out.chart(C.xy(
        "q-exponential exp_q(−u)",
        [C.trace(f"exp_q(−u), q = {_fmt(q)}", u, np.where(eq > 1e-9, eq, np.nan), color=2, width=2.5),
         C.trace("exp(−u)", u, np.where(np.exp(-u) > 1e-9, np.exp(-u), np.nan), color="muted", dash=True)],
        x=C.axis("u = βx²"), y=C.axis("Value", type="log", range=(1e-9, 1.5)),
        subtitle="The density is A·exp_q(−βx²). For q > 1, exp_q decays like u^(−1/(q−1)) instead of exponentially.",
        height=320,
    ))

    # --- table: quantiles
    rows = []
    for p in QUANTILES:
        qq = float(dist.ppf(p))
        gq = float(stats.norm.ppf(p, 0, g_sd))
        rows.append([p, qq, gq, float(np.quantile(s, p)), (qq / gq) if p != 0.5 else None,
                     float(dist.cdf(qq))])
    out.table("Quantiles", ["Probability p", "q-Gaussian (ppf)", ref_label, "Sample", "Ratio", "cdf(ppf(p))"],
              rows, formats=["pct", "num", "num", "num", "x", "pct"],
              note="The last column is the cross-check: cdf and ppf are exact inverses.")

    # --- table: tail probabilities, exact vs asymptote
    trows = []
    for k in (2, 3, 4, 6, 10):
        dd = k * g_sd
        ex = float(dist.tail_probability(dd, "both", exact=True))
        asy = float(dist.tail_probability(dd, "both")) if q > 1 else None
        gg = float(2 * stats.norm.sf(k))
        ratio = ex / gg if gg > 1e-300 else None
        trows.append([f"{k}σ", ex, asy, float(np.mean(np.abs(s) > dd)), _tiny(gg),
                      "> 10⁵" if ratio is not None and ratio > 1e5 else ratio])
    out.table("Tail probability P(|X| > kσ)", ["Threshold", "Exact", "Asymptotic", "Sample", "Gaussian", "Ratio"],
              trows, formats=["text", "pct", "pct", "pct", "pct", "x"],
              note=f"σ = standard deviation of the reference ({_fmt(g_sd, 3)}). The asymptote C·d^(1−α)/(α−1) with "
                   "the density exponent α = 2/(q − 1) applies only far out and is capped at 100%.")

    # --- table: entropy of the discretised densities (50 classes)
    analyzer = TsallisEntropyAnalyzer()
    ebins = np.linspace(-xr, xr, 51)
    mid = 0.5 * (ebins[:-1] + ebins[1:])
    erows, s_vals = [], []
    for name, p in ((f"q-Gaussian (q = {_fmt(q)})", dist.pdf(mid)), (ref_label, stats.norm.pdf(mid, 0, g_sd))):
        p = np.asarray(p, dtype=float)
        p = p[p > 0] / p[p > 0].sum()
        s_q = float(analyzer.compute_tsallis_entropy(p, q))
        via_log = float(np.sum(p * dist.q_logarithm(1.0 / p)))
        erows.append([name, s_q, via_log, float(analyzer.compute_tsallis_entropy(p, 1.0))])
        s_vals.append(s_q)
    why = ""
    if q > 1 and s_vals[1] >= s_vals[0]:
        why = (" That the Gaussian bell has the larger S_q here is not a contradiction: the q-Gaussian maximises S_q "
               "at fixed q-variance (escort mean Σpᵢ^q·xᵢ² / Σpᵢ^q), not at fixed ordinary variance.")
    out.table(f"Tsallis entropy S_q (q = {_fmt(q)}), 50 bins on ±{xr:.0f}", ["Distribution", "S_q",
              "Σ p·ln_q(1/p)", "Shannon S₁"], erows, formats=["text", "num", "num", "num"],
              note="S_q = (1 − Σpᵢ^q)/(q − 1) = Σ pᵢ·ln_q(1/pᵢ) with the q-logarithm ln_q(x) = (x^(1−q) − 1)/(1 − q). "
                   "As q → 1 this becomes the Shannon entropy." + why)

    if q >= 2:
        out.note("For q ≥ 2, ν ≤ 1: not even the mean exists any more (Cauchy-like).", "warning")
    elif not same_var:
        out.note("For q ≥ 5/3 the variance is infinite – the Gaussian reference therefore has the same β instead of "
                 "the same variance.", "info")
