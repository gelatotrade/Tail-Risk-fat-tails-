"""Explore the Lévy-stable distribution: α, β, density, samples and power-law tails."""

import numpy as np
from scipy import stats

from src.physics.levy_flight import LevyStableDistribution

from .. import charts as C

SPEC = {
    "id": "levy-explorer",
    "title": "Lévy-stable distribution",
    "group": "physics",
    "order": 10,
    "summary": "How the stability index α and the skewness β change the density, tails and random walks of a "
               "Lévy distribution compared with the Gaussian curve.",
    "description": (
        "A **Lévy-stable distribution** arises when a very large number of independent shocks with infinite "
        "variance are added up – the fat-tailed counterpart of the central limit theorem. **α** (0 < α ≤ 2) "
        "controls the tails: P(|X| > x) decays like x^−α, and α = 2 gives the normal distribution. **β** shifts "
        "weight into one tail: β < 0 makes large losses more likely than large gains. All axes are in multiples "
        "of the scale γ (γ = 1, δ = 0); for daily returns γ is typically 0.5–1%.\n\n"
        "How to read the charts: on the logarithmic density scale the Gaussian is a downward-opening parabola, "
        "whereas the Lévy density decays only like a power. On the log-log tail plot the power law becomes a "
        "straight line with slope −α; the dashed asymptotes C_α(1 ± β)·x^−α only apply far out. The random walk "
        "shows what this means for a price: the same scale, but individual jumps dominate.\n\n"
        "The density is obtained by Fourier inversion of the characteristic function, the samples by the "
        "Chambers-Mallows-Stuck method (S1 parameterisation, as in `scipy.stats.levy_stable`). The Gaussian "
        "reference is the case α = 2 with the same scale γ, i.e. a normal distribution with σ = √2·γ."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 0.5, "max": 2.0, "step": 0.05,
         "label": "α – stability index",
         "help": "2 = Gaussian. The smaller α, the fatter the tails. Equities: typically 1.4–1.9."},
        {"name": "beta", "type": "float", "default": -0.1, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "β – skewness", "help": "< 0: left tail (crashes) heavier than the right. No effect at α = 2."},
        {"name": "x_range", "type": "int", "default": 12, "min": 4, "max": 50, "step": 1,
         "label": "Density range ± (in γ)", "help": "Wider ranges need more nodes in the Fourier inversion."},
        {"name": "n_samples", "type": "int", "default": 20000, "min": 1000, "max": 200000, "step": 1000,
         "label": "Sample size"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyStableDistribution",
        "src.physics.levy_flight.LevyStableDistribution.characteristic_function",
        "src.physics.levy_flight.LevyStableDistribution.pdf",
        "src.physics.levy_flight.LevyStableDistribution.sample",
        "src.physics.levy_flight.LevyStableDistribution.tail_probability",
    ],
    "cost": "fast",
}

TAIL_KS = (2, 3, 5, 10, 20, 50, 100)
WALK_STEPS = 1000


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _tiny(p, underflow=False):
    """Probability cell in %: values far below anything observable become text instead of 1e-200%.

    ``underflow=True``: an exact 0 is a floating-point underflow (Gaussian tail far out), not a true zero.
    """
    if p is None or not np.isfinite(p):
        return None
    if p == 0 and not underflow:
        return 0.0
    if p < 1e-12:
        return "< 10⁻¹⁰%"
    if 0 < p < 1e-6:
        mant, exp = f"{100 * p:.1e}".split("e")
        return mant + "·10" + str(int(exp)).translate(_SUP) + "%"
    return float(p)


def _ratio(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return None
    if b <= 0:
        return "> 10⁶"
    r = a / b
    return "> 10⁶" if r > 1e6 else float(r)


def run(params, data, out):
    alpha, beta = params["alpha"], params["beta"]
    xr, n, seed = float(params["x_range"]), params["n_samples"], params["seed"]

    dist = LevyStableDistribution(alpha=alpha, beta=beta, gamma=1.0, delta=0.0)
    gauss = LevyStableDistribution(alpha=2.0, beta=0.0, gamma=1.0, delta=0.0)
    g_sd = np.sqrt(2.0)  # S1 with alpha = 2 is N(delta, 2 gamma^2)

    # --- density by Fourier inversion; enough nodes that the grid is not aliased (period 2*pi/dt)
    out.progress("Density via Fourier inversion…", 0.1)
    x = np.linspace(-xr, xr, 481)
    n_points = int(np.clip(100 * xr, 1000, 8000))
    pdf = np.asarray(dist.pdf(x, n_points=n_points), dtype=float)
    gpdf = stats.norm.pdf(x, 0.0, g_sd)

    # --- Chambers-Mallows-Stuck samples
    out.progress("Drawing samples…", 0.4)
    s = np.asarray(dist.sample(n, seed=seed), dtype=float)
    s = s[np.isfinite(s)]
    s_sorted = np.sort(s)
    m = len(s_sorted)

    def left(d):
        return np.searchsorted(s_sorted, -np.asarray(d), side="left") / m

    def right(d):
        return 1.0 - np.searchsorted(s_sorted, np.asarray(d), side="right") / m

    # --- headline numbers
    k5 = 5.0
    p_left5, g5 = float(left(k5)), float(stats.norm.sf(k5 / g_sd))
    if alpha < 2:
        out.metric("Variance", "∞", "text", hint="Infinite for every α < 2",
                   status="serious" if alpha < 1.5 else "warning",
                   badge="Very fat tails" if alpha < 1.5 else "Fat tails")
        out.metric("Finite moments E|X|^p", f"p < {_fmt(alpha)}", "text",
                   hint="The mean exists only for α > 1" if alpha > 1 else "Not even the mean exists")
        out.metric("Density exponent", 1 + alpha, "num", hint="f(x) ~ |x|^−(1+α) far out")
    else:
        out.metric("Variance", 2.0, "num", hint="α = 2: normal distribution with σ² = 2γ²")
        out.metric("Finite moments E|X|^p", "all p", "text", hint="Gaussian case")
    out.metric("P(X < −5γ)", p_left5, "pct", hint=f"Sample; Gaussian: {g5 * 100:.2g}%",
               status="critical" if p_left5 > 1000 * g5 else "serious" if p_left5 > 10 * g5 else None)
    out.metric("1% quantile", float(np.quantile(s, 0.01)), "num",
               hint=f"in γ; Gaussian: {_fmt(stats.norm.ppf(0.01) * g_sd)}".replace("-", "−"))
    out.metric("Largest move", float(np.max(np.abs(s))), "num",
               hint="in γ, across " + f"{m:,}" + " draws")

    # --- chart 1: density on a log scale
    edges = np.linspace(-xr, xr, 121)
    counts, _ = np.histogram(s, bins=edges)
    width = edges[1] - edges[0]
    dens = counts / (m * width)  # share of ALL draws, so mass outside the window is not re-normalised away
    centers = 0.5 * (edges[:-1] + edges[1:])
    floor = 1.0 / (m * width)
    positive = pdf[pdf > 0]
    lo = max(1e-8, 0.5 * min(floor, float(positive.min()) if positive.size else floor))
    hi = 2.5 * max(float(pdf.max()), float(dens.max()), float(gpdf.max()))
    clip = lambda y: np.where(y >= lo, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Density on a logarithmic scale",
        [C.trace("Sample", centers, np.where(dens > 0, dens, np.nan), type="bar", color=1),
         C.trace(f"Lévy density (α = {_fmt(alpha)})", x, clip(pdf), color=2, width=2.5),
         C.trace("Gaussian (α = 2, same scale)", x, clip(gpdf), color="muted", dash=True)],
        x=C.axis("x (in γ)"), y=C.axis("Density", type="log", range=(lo, hi)),
        subtitle="The Gaussian parabola plunges, while the Lévy density decays only like |x|^−(1+α).", height=380,
    ))

    # --- chart 2: tails on log-log axes
    dmax = float(np.clip(np.max(np.abs(s)), 20.0, 1e6))
    d = np.logspace(np.log10(0.5), np.log10(dmax), 70)
    emp_l, emp_r = left(d), right(d)
    asym_l = np.asarray(dist.tail_probability(-d, "left"), dtype=float)
    asym_r = np.asarray(dist.tail_probability(d, "right"), dtype=float)
    g_tail = stats.norm.sf(d / g_sd)
    y_lo = 0.3 / m
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Tails on a log-log plot",
        [C.trace("Left: P(X < −x)", d, pos(emp_l), color=1, width=2.5),
         C.trace("Right: P(X > x)", d, pos(emp_r), color=3, width=2.5),
         C.trace("Left asymptote", d, pos(asym_l), color=1, dash=True, width=1.5),
         C.trace("Right asymptote", d, pos(asym_r), color=3, dash=True, width=1.5),
         C.trace("Gaussian", d, pos(g_tail), color="muted", dash=True)],
        x=C.axis("Distance x from the centre (in γ)", type="log"),
        y=C.axis("Probability", type="log", fmt="pct", range=(y_lo, 1.5)),
        subtitle=f"Solid: sample. Dashed: asymptote C_α(1 ± β)·x^−α – a straight line with slope "
                 f"−α = −{_fmt(alpha)}. The Gaussian curve drops away almost vertically.", height=380,
    ))

    # --- chart 3: random walk with the same scale
    steps = min(WALK_STEPS, len(s))
    walk_levy = np.cumsum(s[:steps])
    walk_gauss = np.cumsum(np.asarray(gauss.sample(steps, seed=seed + 1), dtype=float))
    out.chart(C.xy(
        "Lévy flight vs Brownian motion",
        [C.trace(f"Lévy flight (α = {_fmt(alpha)})", np.arange(1, steps + 1), walk_levy, color=2, width=1.5),
         C.trace("Brownian motion (α = 2)", np.arange(1, steps + 1), walk_gauss, color="muted", width=1.5)],
        x=C.axis("Step", fmt="int"), y=C.axis("Position (in γ)"),
        subtitle="Cumulative sum of the first draws: the same scale, but the Lévy path is shaped by a few jumps.",
    ))

    # --- chart 4: characteristic function (the definition of the law)
    t = np.linspace(-3.0, 3.0, 401)
    phi = dist.characteristic_function(t)
    phi_g = gauss.characteristic_function(t)
    out.chart(C.xy(
        "Characteristic function φ(t)",
        [C.trace("Re φ(t)", t, np.real(phi), color=2),
         C.trace("Im φ(t)", t, np.imag(phi), color=4),
         C.trace("Gaussian: e^(−t²)", t, np.real(phi_g), color="muted", dash=True)],
        x=C.axis("t (in 1/γ)"), y=C.axis("φ(t)", range=(-1.05, 1.05)),
        subtitle="|φ(t)| = e^(−|t|^α): the cusp at t = 0 for α < 2 is the fingerprint of fat tails. "
                 "A non-zero imaginary part Im φ indicates skewness.", height=300,
    ))

    # --- table: tail probabilities
    rows = []
    for k in TAIL_KS:
        pl, pr = float(left(k)), float(right(k))
        al = float(dist.tail_probability(-k, "left"))
        ar = float(dist.tail_probability(k, "right"))
        g = float(stats.norm.sf(k / g_sd))
        rows.append([f"{k} γ", pl, _tiny(al), pr, _tiny(ar), _tiny(g, underflow=True), _ratio(pl, g)])
    out.table(
        "Tail probabilities",
        ["Distance", "P(X < −x) sample", "Left, asymptotic", "P(X > x) sample", "Right, asymptotic",
         "Gaussian per side", "Ratio left / Gaussian"],
        rows, formats=["text", "pct", "pct", "pct", "pct", "pct", "x"],
        note="Asymptotic = power law from tail_probability (Samorodnitsky-Taqqu 1.2.15); it applies only for "
             "large x and is capped at 100%. 0% in the sample means the event was not observed.",
    )

    if alpha <= 1:
        out.note("For α ≤ 1 not even the mean exists: the sample average never settles down, however many draws "
                 "you take.", "warning")
    if alpha < 0.8:
        out.note("For small α the characteristic function decays so slowly that a Fourier inversion with a fixed "
                 "integration range underestimates the outer tails of the density. In that region the histogram "
                 "and the log-log plot of the sample give the more reliable values.", "info")
    if alpha == 2 and beta != 0:
        out.note("At α = 2 the distribution is exactly normal – β then has no effect.", "info")
