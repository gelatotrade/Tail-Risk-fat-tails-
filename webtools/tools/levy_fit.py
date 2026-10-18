"""Fit a Lévy-stable distribution to real returns: tail index, VaR and Expected Shortfall compared."""

import numpy as np
from scipy import stats

from src.physics.levy_flight import LevyFlightProcess, LevyStableDistribution, estimate_tail_index

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "levy-fit",
    "title": "Lévy fit, tail index and VaR",
    "group": "physics",
    "order": 20,
    "summary": "Fits a Lévy-stable distribution to your returns, estimates the tail index with Hill and Pickands, "
               "and compares Lévy, historical and Gaussian VaR.",
    "description": (
        "**McCulloch's quantile estimator** reads α, β, γ and δ from five quantiles of the returns: how far apart "
        "are the 5% and 95% quantiles relative to the interquartile range? From these, `LevyFlightProcess` "
        "simulates the Value at Risk and the Expected Shortfall. Over several days it aggregates exactly as a "
        "stable law: the scale grows with h^(1/α) instead of √h.\n\n"
        "The **Hill estimator** measures the tail index directly on the 10% largest absolute returns. This is the "
        "acid test: a Lévy-stable law requires α < 2 (infinite variance). For equity indices, Hill typically gives "
        "α ≈ 2.5–4, i.e. fat tails with finite variance. The Lévy model then overstates the most extreme losses, "
        "while the normal distribution understates them. The estimator table puts Hill, Pickands and the Lévy α "
        "side by side; Pickands uses only three order statistics and is therefore very noisy.\n\n"
        "How to read the charts: on a log-log plot every power law is a straight line. The Hill line follows the "
        "data, the Lévy line is flatter (slope −α). The VaR curve shows the confidence level at which the models "
        "diverge. Returns are daily log returns; VaR and ES are losses expressed as positive decimals."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Confidence level",
         "options": [{"value": 0.95, "label": "95%"}, {"value": 0.975, "label": "97.5%"},
                     {"value": 0.99, "label": "99%"}, {"value": 0.995, "label": "99.5%"},
                     {"value": 0.999, "label": "99.9%"}]},
        {"name": "horizon", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1,
         "label": "Horizon (trading days)", "help": "Historical: overlapping h-day sums of log returns."},
        {"name": "n_simulations", "type": "int", "default": 100000, "min": 20000, "max": 500000, "step": 10000,
         "label": "Monte Carlo draws"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyStableDistribution.fit",
        "src.physics.levy_flight.LevyStableDistribution.pdf",
        "src.physics.levy_flight.LevyStableDistribution.tail_probability",
        "src.physics.levy_flight.LevyFlightProcess",
        "src.physics.levy_flight.LevyFlightProcess.compute_var",
        "src.physics.levy_flight.LevyFlightProcess.compute_cvar",
        "src.physics.levy_flight.estimate_tail_index",
    ],
    "cost": "fast",
}

TABLE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _pct(x, digits=1):
    return _fmt(100 * x, digits) + "%"


def _lvl(c):
    """Confidence level as text: 0.99 -> '99%', 0.975 -> '97.5%'."""
    return f"{100 * c:.1f}".rstrip("0").rstrip(".") + "%"


def _hist_var_es(rh, c):
    var = -float(np.quantile(rh, 1 - c))
    tail = rh[rh <= -var]
    return var, (-float(np.mean(tail)) if tail.size else var)


def _gauss_var_es(mu, sd, c):
    z = stats.norm.ppf(1 - c)
    return -(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / (1 - c))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    c, h, nsim, seed = float(params["confidence"]), params["horizon"], params["n_simulations"], params["seed"]
    n = len(r)
    if n < 2 * h + 50:
        raise ToolError(f"The series is too short for a horizon of {h} days. Shorten the horizon.")

    try:
        dist = LevyStableDistribution.fit(r, method="quantile")
    except ValueError as exc:
        raise ToolError(f"The Lévy fit is not possible: the returns are (almost) constant ({exc}).")
    hill = float(estimate_tail_index(r, "hill"))
    n_pos = int(np.count_nonzero(r != 0))
    k_hill = min(max(10, int(0.1 * n)), n - 1, n_pos - 1)  # as in estimate_tail_index
    hill_se = hill / np.sqrt(k_hill)                      # asymptotic standard error of Hill
    try:
        pick = float(estimate_tail_index(r, "pickands"))
    except ValueError:
        pick = float("nan")
    proc = LevyFlightProcess(dist)
    es_finite = dist.alpha > 1  # for alpha <= 1 even E|X| is infinite, so ES does not exist

    # h-day returns: Lévy aggregates exactly, historical = overlapping sums, Gauss = sqrt(h)
    rh = np.convolve(r, np.ones(h), mode="valid") if h > 1 else r
    mu, sd = float(np.mean(r)), float(np.std(r))
    mu_h, sd_h = mu * h, sd * np.sqrt(h)

    out.progress("Monte Carlo VaR of the Lévy model…", 0.2)
    var_l = float(proc.compute_var(c, h, nsim, seed=seed))
    es_l = float(proc.compute_cvar(c, h, nsim, seed=seed))
    var_h, es_h = _hist_var_es(rh, c)
    var_g, es_g = _gauss_var_es(mu_h, sd_h, c)

    lvl = _lvl(c)
    days = f"{h} day{'s' if h > 1 else ''}"
    out.metric("Lévy α (quantile fit)", dist.alpha, "num", hint="2 = Gaussian · < 2: infinite variance")
    out.metric("Lévy β (skewness)", dist.beta, "num", hint="< 0: heavier left tail")
    out.metric("Lévy scale γ", dist.gamma, "pct", hint="daily; a Gaussian would have γ = σ/√2")
    out.metric("Tail index (Hill)", hill, "num",
               hint=f"± {_fmt(hill_se)} (std. error), top 10% of |returns|; < 2 would mean infinite variance",
               status="critical" if hill < 2 else "serious" if hill < 3 else "warning" if hill < 4 else "good")
    out.metric(f"{lvl} VaR · Lévy", var_l, "pct", hint=f"{days} · {nsim:,} draws")
    if es_finite:
        out.metric(f"{lvl} Expected Shortfall · Lévy", es_l, "pct",
                   hint="Monte Carlo; very uncertain for α close to 1" if dist.alpha < 1.3 else "Monte Carlo")
    else:
        out.metric(f"{lvl} Expected Shortfall · Lévy", "∞", "text", status="critical",
                   hint="α ≤ 1: the loss has no finite expected value")
    out.metric(f"{lvl} VaR · historical", var_h, "pct", hint=f"ES {_pct(es_h)}")
    out.metric(f"{lvl} VaR · Gaussian", var_g, "pct", hint=f"ES {_pct(es_g)}")

    # --- chart 1: distribution, log scale
    centers, heights = C.histogram(r, bins=90, density=True)
    span = max(abs(float(r.min()) - dist.delta), abs(float(r.max()) - dist.delta)) * 1.05
    span = min(span, 80 * dist.gamma)  # beyond ~π·n_points/100 γ the Fourier grid would alias
    grid = np.linspace(dist.delta - span, dist.delta + span, 361)
    n_points = int(np.clip(100 * span / dist.gamma, 1000, 8000))
    lpdf = np.asarray(dist.pdf(grid, n_points=n_points), dtype=float)
    gpdf = stats.norm.pdf(grid, mu, sd)
    floor = float(np.min(heights[heights > 0])) / 4
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Daily returns: data vs Lévy fit vs Gaussian",
        [C.trace("Observed", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace(f"Lévy-stable (α = {_fmt(dist.alpha)})", grid, keep(lpdf), color=2, width=2.5),
         C.trace("Normal distribution", grid, keep(gpdf), color="muted", dash=True)],
        x=C.axis("Daily return", fmt="pct", digits=1),
        y=C.axis("Density", type="log", range=(floor, float(np.max(heights)) * 3)),
        subtitle="Logarithmic scale: both models fit the centre, but they diverge in the tails.",
    ))

    # --- chart 2: tails on log-log axes, P(|r| > x)
    a = np.sort(np.abs(r))[::-1]
    a = a[a > 0]
    k = min(max(10, int(0.1 * n)), n - 1, len(a) - 1)  # same k as estimate_tail_index
    x_lo, x_hi = float(np.median(a)), float(a[0]) * 1.6
    xs = np.logspace(np.log10(x_lo), np.log10(x_hi), 60)
    emp = np.array([np.mean(np.abs(r) > v) for v in xs])
    hill_line = (k / n) * (xs / a[k]) ** (-hill)
    sample = np.asarray(dist.sample(nsim, seed=seed), dtype=float)
    lev_mc = np.array([np.mean(np.abs(sample - dist.delta) > v) for v in xs])
    lev_asym = np.asarray(dist.tail_probability(dist.delta + xs, "both"), dtype=float)
    gauss = 2 * stats.norm.sf(xs / sd)
    y_lo = 0.5 / n
    pos = lambda y: np.where(np.asarray(y) >= y_lo, y, np.nan)  # noqa: E731
    tail_traces = [
        C.trace("Observed", xs, pos(np.where(emp > 0, emp, np.nan)), type="step", color=1, width=2.5),
        C.trace(f"Hill (α = {_fmt(hill)})", xs[xs >= a[k]], pos(hill_line[xs >= a[k]]), color=3, dash=True),
        C.trace(f"Lévy (α = {_fmt(dist.alpha)})", xs, pos(np.where(lev_mc > 0, lev_mc, np.nan)), color=2),
        C.trace("Lévy asymptote", xs, pos(np.where(lev_asym < 1, lev_asym, np.nan)), color=2, dash=True, width=1.5),
        C.trace("Gaussian", xs, pos(gauss), color="muted", dash=True)]
    out.chart(C.xy(
        "Tails on a log-log plot: P(|r| > x)",
        [t for t in tail_traces if np.any(np.isfinite(t["y"]))],
        x=C.axis("Absolute daily return x", type="log", fmt="pct", digits=1),
        y=C.axis("Share of days", type="log", fmt="pct", range=(y_lo, 1.0)),
        subtitle="Slope = −tail index. The Lévy line is simulated from the fit. If it is flatter than the data, "
                 "the model overstates crashes.", height=380,
    ))

    # --- chart 3: VaR across confidence levels
    out.progress("VaR curve across confidence levels…", 0.6)
    tail_p = np.logspace(np.log10(max(1.0 / len(rh), 5e-4)), -1, 14)
    v_l = [float(proc.compute_var(1 - p, h, nsim, seed=seed)) for p in tail_p]
    v_h = [_hist_var_es(rh, 1 - p)[0] for p in tail_p]
    v_g = [_gauss_var_es(mu_h, sd_h, 1 - p)[0] for p in tail_p]
    out.chart(C.xy(
        f"VaR by confidence level ({days})",
        [C.trace("Historical", tail_p, v_h, color=1, width=2.5),
         C.trace("Lévy-stable", tail_p, v_l, color=2),
         C.trace("Normal distribution", tail_p, v_g, color="muted", dash=True)],
        x=C.axis("Exceedance probability 1 − c", type="log", fmt="pct", digits=1,
                 range=(0.1, float(tail_p[0]))),
        y=C.axis("Value at Risk (loss)", fmt="pct", digits=1),
        subtitle="Left: frequent losses (90% VaR); right: ever rarer ones – out to the edge of the data.",
    ))

    # --- tables
    rows = []
    for lev in TABLE_LEVELS:
        vh, eh = _hist_var_es(rh, lev)
        vg, eg = _gauss_var_es(mu_h, sd_h, lev)
        rows.append([_lvl(lev), float(proc.compute_var(lev, h, nsim, seed=seed)), vh, vg,
                     float(proc.compute_cvar(lev, h, nsim, seed=seed)) if es_finite else "∞", eh, eg])
    out.table(
        f"VaR and Expected Shortfall ({days})",
        ["Level", "VaR Lévy", "VaR historical", "VaR Gaussian", "ES Lévy", "ES historical", "ES Gaussian"],
        rows, formats=["text", "pct", "pct", "pct", "pct", "pct", "pct"],
        note="Lévy: Monte Carlo from S1(α, β, h^(1/α)·γ, h·δ). Historical: empirical quantile of the "
             + ("overlapping h-day sums." if h > 1 else "daily returns.") + " Gaussian: μ·h and σ·√h.",
    )
    # --- table: three views on the tail index
    def hill_text(a):
        return ("< 2: infinite variance" if a < 2 else "2–4: fat tails, finite variance" if a < 4
                else "> 4: only moderately fat tails")

    k_pick = int(0.1 * n)
    if not np.isfinite(pick):
        pick_cell, pick_text = "n/a", "not computable (too few observations or ties)"
    elif pick <= 0:
        pick_cell, pick_text = pick, "≤ 0: no power-law tail detectable – the estimator is unstable here"
    elif pick > 10:
        pick_cell, pick_text = pick, "implausibly large: X₍₄ₖ₎ already lies in the body of the distribution"
    else:
        pick_cell, pick_text = pick, hill_text(pick) + " (very noisy)"
    out.table(
        "Tail index: three estimators compared",
        ["Estimator", "Tail index", "Std. error", "Tail observations k", "Interpretation"],
        [["Hill (top 10% of |returns|)", hill, hill_se, k_hill, hill_text(hill)],
         ["Pickands (X₍ₖ₎, X₍₂ₖ₎, X₍₄ₖ₎)", pick_cell, None, k_pick, pick_text],
         ["Lévy α (McCulloch quantile fit)", dist.alpha, None, None,
          "stable laws always have α ≤ 2; α = 2 is the normal distribution"]],
        formats=["text", "num", "num", "int", "text"],
        note="Hill: std. error ≈ α/√k. Pickands uses only the k-th, 2k-th and 4k-th largest |return|; with k = 10% "
             "of the series, X₍₄ₖ₎ sits at the 60% quantile, i.e. in the body of the distribution – hence the wild "
             "swings. The Lévy α describes the shape of the whole distribution and can never exceed 2, even when "
             "the data have finite variance.",
    )

    if h == 1:
        exc_rows = []
        for lev in (0.95, 0.99, 0.995):
            vl = float(proc.compute_var(lev, 1, nsim, seed=seed))
            vg = _gauss_var_es(mu, sd, lev)[0]
            exc_rows.append([_lvl(lev), 1 - lev, float(np.mean(r < -vl)), float(np.mean(r < -vg)),
                             int(round((1 - lev) * n)), int(np.sum(r < -vl)), int(np.sum(r < -vg))])
        out.table(
            "Exceedances in the dataset",
            ["Level", "Expected rate", "Lévy rate", "Gaussian rate", "Expected days", "Lévy days", "Gaussian days"],
            exc_rows, formats=["text", "pct", "pct", "pct", "int", "int", "int"],
            note="In-sample: share of days on which the loss exceeded the VaR. Below the expected rate = too "
                 "conservative; above it = risk understated.",
        )

    # --- verdict: tail index of the data, and how the two models miss it at 99.9%
    vl9 = float(proc.compute_var(0.999, h, nsim, seed=seed))
    vh9 = _hist_var_es(rh, 0.999)[0]
    vg9 = _gauss_var_es(mu_h, sd_h, 0.999)[0]

    def verb(model, ref):
        q = model / ref if ref > 0 else 1.0
        return "overstates" if q > 1.25 else "understates" if q < 0.8 else "fits"

    level = "critical" if hill < 2 else "serious" if hill < 3 else "warning" if hill < 4 else "good"
    title = (f"Tail index ≈ {_fmt(hill)}: " + ("infinite variance possible" if hill < 2 else
                                              "fat tails with finite variance" if hill < 4 else
                                              "only moderately fat tails"))
    out.set_verdict(level, title,
                    f"99.9% VaR: historical {_pct(vh9)} · Lévy-stable (α = {_fmt(dist.alpha)}) {_pct(vl9)}, "
                    f"{verb(vl9, vh9)} · normal distribution {_pct(vg9)}, {verb(vg9, vh9)}.")
    if not es_finite:
        out.note(f"The quantile fit gives α = {_fmt(dist.alpha)} ≤ 1: such a Lévy law has no finite mean, so the "
                 "Expected Shortfall is infinite. Monte Carlo values would be pure noise and are therefore shown "
                 "as ∞.", "warning")
    if n < 500:
        out.note(f"Only {n} returns: α and β of the quantile fit depend on the 5% and 95% quantiles, the Hill "
                 f"estimator on k = {k_hill} tail observations. Both are very uncertain for such short series.", "info")
    if len(rh) * 0.001 < 3:
        out.note("At the 99.9% level this series contains only a handful of observations – the historical value "
                 "is correspondingly uncertain.", "info")
