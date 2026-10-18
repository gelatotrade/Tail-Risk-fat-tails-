"""OU process with compound-Poisson jumps: jump types and their effect on VaR / Expected Shortfall."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import OUWithJumps

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "ou-jumps",
    "title": "OU with jumps: crash risk in VaR",
    "group": "physics",
    "order": 55,
    "summary": "What rare jumps do to VaR and Expected Shortfall – normal, double-exponential or Lévy-distributed, "
               "compared directly with the same process without jumps.",
    "description": (
        "A mean-reverting process **dx = θ(μ − x)dt + σ dW + dJ**: between jumps it behaves like an "
        "Ornstein-Uhlenbeck process, and on top of that there are on average **λ jumps per year** (Poisson). Think "
        "of a position's deviation from fair value – a spread or a commodity price, for example – that normally "
        "reverts calmly but occasionally jumps abruptly. x is a log deviation, so 0.05 means +5%.\n\n"
        "How to read the charts: the example paths mark every jump. The distribution of the change over the VaR "
        "horizon is plotted on a log scale, which shows how the jumps lift the tail while the centre stays almost "
        "unchanged. Both simulations use **the same random numbers** for the diffusion, so the difference is "
        "purely the effect of the jumps.\n\n"
        "Jump distributions: **normal** N(mean, std), **double-exponential** (Kou model: equally likely up and "
        "down, exponentially distributed) and **Lévy** (Student-t with ν = 1.7 as an approximation – infinite "
        "variance, so individual jumps can become absurdly large). Expected Shortfall reacts more strongly than "
        "VaR because it averages the entire tail beyond the VaR.\n\n"
        "Over several days the restoring force pulls the jumps back as well: a jump at the start of the VaR "
        "horizon counts for less at the end than one shortly before it. The simulation accounts for this exactly. "
        "VaR and ES are Monte Carlo estimates – the band shown indicates how much they fluctuate from random "
        "noise alone."
    ),
    "needs": [],
    "params": [
        {"name": "theta", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1,
         "label": "Speed of mean reversion θ", "unit": "per year"},
        {"name": "sigma", "type": "float", "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
         "label": "Diffusion volatility σ", "unit": "per √year", "help": "0.2 = 20% per year."},
        {"name": "jump_intensity", "type": "float", "default": 10.0, "min": 0.0, "max": 252.0, "step": 1.0,
         "label": "Jumps per year λ"},
        {"name": "jump_distribution", "type": "select", "default": "normal", "label": "Jump distribution",
         "options": [{"value": "normal", "label": "Normal"},
                     {"value": "double_exponential", "label": "Double-exponential (Kou)"},
                     {"value": "levy", "label": "Lévy (Student-t, ν = 1.7)"}]},
        {"name": "jump_mean", "type": "float", "default": -0.02, "min": -0.2, "max": 0.2, "step": 0.005,
         "label": "Mean jump size", "help": "−0.02 = −2% per jump on average."},
        {"name": "jump_std", "type": "float", "default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005,
         "label": "Jump dispersion", "help": "Standard deviation (normal) or scale (Kou, Lévy)."},
        {"name": "horizon_days", "type": "int", "default": 1, "min": 1, "max": 21, "step": 1,
         "label": "VaR horizon", "unit": "trading days"},
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999, "step": 0.001,
         "label": "Confidence level"},
        {"name": "n_simulations", "type": "int", "default": 50000, "min": 5000, "max": 200000, "step": 5000,
         "label": "Simulations for VaR/ES", "help": "More scenarios = narrower Monte Carlo band around VaR and ES."},
        {"name": "path_years", "type": "float", "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25,
         "label": "Length of example paths", "unit": "years"},
        {"name": "n_paths", "type": "int", "default": 200, "min": 10, "max": 500, "step": 10,
         "label": "Number of paths", "help": "For the jump statistics and the path fan."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.OUWithJumps",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.sample_jump_size",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.simulate",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.sample_terminal",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.compute_var",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.compute_expected_shortfall",
    ],
    "cost": "fast",
}

LEVELS = [0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
MAX_PATH_STEPS = 300_000  # paths x steps: OUWithJumps.simulate draws the jumps path by path in Python
N_EXAMPLES = 3


def _fmt(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _level(conf):
    """0.99 -> '99%', 0.975 -> '97.5%'."""
    return f"{100 * conf:.1f}".rstrip("0").rstrip(".") + "%"


def _var_es(change, conf):
    """Same estimator as OUWithJumps.compute_var / compute_expected_shortfall, on a given sample."""
    var = -np.percentile(change, (1 - conf) * 100)
    tail = change[change < -var]
    return var, (-np.mean(tail) if len(tail) else var)


def _var_band(change, conf):
    """95% Monte Carlo band of the VaR from the order statistics (binomial count below the quantile)."""
    n, a = len(change), 1 - conf
    half = 1.96 * np.sqrt(n * a * (1 - a))
    srt = np.sort(change)
    lo_i, hi_i = int(max(0, np.floor(n * a - half))), int(min(n - 1, np.ceil(n * a + half)))
    return -srt[hi_i], -srt[lo_i]  # VaR = -quantile: a higher quantile gives the lower VaR


def _pp(x):
    """Width in percentage points."""
    return f"{_fmt(100 * x)} percentage points"


def run(params, data, out):
    kw = dict(theta=params["theta"], mu=0.0, sigma=params["sigma"], jump_mean=params["jump_mean"],
              jump_std=params["jump_std"], jump_distribution=params["jump_distribution"])
    lam = params["jump_intensity"]
    model = OUWithJumps(jump_intensity=lam, **kw)
    plain = OUWithJumps(jump_intensity=0.0, **kw)
    h = params["horizon_days"] / 252
    conf, n_sim, seed = params["confidence"], params["n_simulations"], params["seed"]
    path_steps = int(round(params["path_years"] * 252))
    if path_steps * params["n_paths"] > MAX_PATH_STEPS:
        raise ToolError(f"{params['n_paths']:,} paths × {path_steps:,} days is too much for the browser (limit "
                        f"{MAX_PATH_STEPS:,} path-days). Reduce the path length or the number of paths.")

    # VaR / ES from the framework (same seed -> the same sample for both, and common random numbers)
    var_j = model.compute_var(0.0, h, conf, n_sim, seed=seed)
    es_j = model.compute_expected_shortfall(0.0, h, conf, n_sim, seed=seed)
    var_0 = plain.compute_var(0.0, h, conf, n_sim, seed=seed)
    es_0 = plain.compute_expected_shortfall(0.0, h, conf, n_sim, seed=seed)
    change_j = model.sample_terminal(0.0, h, n_sim, seed=seed)  # the sample behind var_j / es_j
    change_0 = plain.sample_terminal(0.0, h, n_sim, seed=seed)
    band_lo, band_hi = _var_band(change_j, conf)
    # exact reference without jumps: the OU change over h is normal
    sd_0 = float(np.sqrt(plain.base_ou.variance_at_time(h)))
    z = stats.norm.ppf(1 - conf)
    var_0_exact, es_0_exact = -z * sd_0, sd_0 * stats.norm.pdf(z) / (1 - conf)

    hz = f"{params['horizon_days']} day" + ("" if params["horizon_days"] == 1 else "s")
    lvl = _level(conf)
    out.metric(f"{lvl} VaR with jumps", var_j, "pct",
               hint=f"loss over {hz} · Monte Carlo band (95%): {_fmt(100 * band_lo)}–{_fmt(100 * band_hi)}%",
               status="serious" if var_j > 1.5 * var_0 else None)
    out.metric(f"{lvl} Expected Shortfall with jumps", es_j, "pct", hint="average loss beyond the VaR",
               status="critical" if es_j > 2 * es_0 else "serious" if es_j > 1.3 * es_0 else None)
    out.metric("VaR without jumps", var_0, "pct",
               hint=f"pure OU, same random numbers · exact (normal): {_fmt(100 * var_0_exact)}%")
    out.metric("Expected Shortfall without jumps", es_0, "pct", hint=f"exact: {_fmt(100 * es_0_exact)}%")
    out.metric("VaR jump factor", var_j / var_0 if var_0 > 0 else None, "x", hint="VaR with ÷ without jumps")
    out.metric("ES jump factor", es_j / es_0 if es_0 > 0 else None, "x", hint="ES with ÷ without jumps")
    out.metric("Expected jumps within horizon", lam * h, "num", digits=3,
               hint=f"λ·horizon, λ = {_fmt(lam, 0)} per year")
    out.metric("Excess kurtosis of the change", stats.kurtosis(change_j), "num", digits=1,
               hint=f"without jumps: {_fmt(stats.kurtosis(change_0), 1)}")

    out.table("Risk with and without jumps", ["Metric", "With jumps", "Without jumps", "Add-on", "Factor"],
              [[lvl + " VaR", var_j, var_0, var_j - var_0, var_j / var_0 if var_0 > 0 else None],
               [lvl + " Expected Shortfall", es_j, es_0, es_j - es_0, es_j / es_0 if es_0 > 0 else None]],
              formats=["text", "pct", "pct", "pct", "x"],
              note=f"{n_sim:,}" + f" Monte Carlo scenarios over {hz}, starting at the mean "
                   "(x₀ = μ = 0). "
                   + f"Monte Carlo noise alone shifts the VaR with jumps by up to ±{_pp((band_hi - band_lo) / 2)} "
                     "(95%); more scenarios narrow the band.")

    # Example paths with jump markers
    sim = model.simulate(0.0, params["path_years"], n_steps=path_steps, n_paths=params["n_paths"], seed=seed + 1)
    paths, times = sim["paths"], sim["times"]
    dt = times[1] - times[0]
    traces = []
    for k in range(min(N_EXAMPLES, len(paths))):
        traces.append(C.trace(f"Path {k + 1}", times, paths[k], color=k + 1, width=1.3))
    sel = sim["jump_paths"] < N_EXAMPLES
    if np.any(sel):
        steps = np.rint(sim["jump_times"][sel] / dt).astype(int)
        jp = sim["jump_paths"][sel]
        traces.append(C.trace("Jumps", sim["jump_times"][sel], paths[jp, steps], type="scatter", color="critical",
                              size=8, decimate_to=None,
                              text=[f"Path {p + 1}: jump {_fmt(100 * s, 1)}%" for p, s in zip(jp, sim["jump_sizes"][sel])]))
    out.chart(C.xy(
        "Example paths with jumps",
        traces, x=C.axis("Time (years)"), y=C.axis("Deviation x", fmt="pct", digits=0),
        hlines=[C.hline(0.0, "μ")], height=360,
        subtitle="After each jump the restoring force pulls the path back – half-life "
                 f"{_fmt(model.base_ou.half_life() * 252, 0)} trading days.",
    ))

    # Distribution of the change over the VaR horizon (log density)
    lo, hi = np.percentile(np.concatenate([change_j, change_0]), [0.05, 99.95])
    cj, hj = C.histogram(change_j, bins=80, range=(lo, hi))
    c0, h0 = C.histogram(change_0, bins=80, range=(lo, hi))
    positive = np.concatenate([hj[hj > 0], h0[h0 > 0]])
    floor = float(positive.min()) / 3 if len(positive) else 1e-3
    top = float(max(hj.max(), h0.max())) * 3
    out.chart(C.xy(
        f"Change over {hz}: with vs without jumps",
        [C.trace("With jumps", cj, np.where(hj > 0, hj, np.nan), type="bar", color=1),
         C.trace("Without jumps", c0, np.where(h0 > 0, h0, np.nan), color="muted", width=2),
         C.trace(f"−VaR with jumps ({_fmt(100 * var_j)}%)", [-var_j] * 2, [floor, top], color="critical",
                 dash=True, width=1.5, decimate_to=None),
         C.trace(f"−VaR without jumps ({_fmt(100 * var_0)}%)", [-var_0] * 2, [floor, top], color="muted", dash=True,
                 width=1.5, decimate_to=None)],
        x=C.axis("Change in x", fmt="pct", digits=1), y=C.axis("Density (log)", type="log", range=(floor, top)),
        height=340, subtitle="Log scale: the jumps lift the tail, while the centre stays almost unchanged.",
    ))

    # VaR and ES across confidence levels (same samples as above)
    labels = [_level(c) for c in LEVELS]
    vj, ej = zip(*[_var_es(change_j, c) for c in LEVELS])
    v0, e0 = zip(*[_var_es(change_0, c) for c in LEVELS])
    out.chart(C.xy(
        "VaR and Expected Shortfall by confidence level",
        [C.trace("ES with jumps", labels, ej, color=1, width=2.5),
         C.trace("VaR with jumps", labels, vj, color=2, width=2.5),
         C.trace("ES without jumps", labels, e0, color=1, dash=True),
         C.trace("VaR without jumps", labels, v0, color=2, dash=True)],
        x=C.axis("Confidence level", type="category"), y=C.axis(f"Loss over {hz}", fmt="pct", digits=1),
        height=320, subtitle="The further you look into the tail, the more the jumps dominate.",
    ))

    sizes = sim["jump_sizes"]
    expected = lam * params["path_years"] * params["n_paths"]
    if len(sizes) >= 20:
        a, b = np.percentile(sizes, [0.5, 99.5])
        cs, hs = C.histogram(sizes, bins=50, range=(a, b))
        out.chart(C.xy(
            f"Jump sizes: {len(sizes):,} simulated jumps",
            [C.trace("Jumps", cs, hs, type="bar", color=3)],
            x=C.axis("Jump size", fmt="pct", digits=0), y=C.axis("Density"), height=280,
            subtitle=f"Expected: {expected:,.0f} jumps in {params['n_paths']} paths × {params['path_years']:g} "
                     f"year{'' if params['path_years'] == 1 else 's'}. Shown: 0.5% to 99.5% quantile.",
        ))
    elif lam > 0:
        out.note(f"The example paths contained only {len(sizes)} jumps (expected {_fmt(expected, 1)}) – increase λ, "
                 "the path length or the number of paths to get a histogram of jump sizes.")
    if params["jump_distribution"] == "levy":
        big = float(np.max(np.abs(sizes))) if len(sizes) else 0.0
        out.note(f"Lévy jumps have infinite variance: the largest simulated jump was {100 * big:,.0f}%. "
                 "VaR and ES therefore vary strongly with the seed.", "warning")
