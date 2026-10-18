"""Ornstein-Uhlenbeck process: mean reversion, path fan with the analytic band, and an AR(1) fit (VIX-like)."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

from .. import charts as C

SPEC = {
    "id": "ou-process",
    "title": "Ornstein-Uhlenbeck: mean reversion",
    "group": "physics",
    "order": 50,
    "summary": "How quickly a quantity such as the VIX reverts to its mean – path fan with an exact expectation "
               "band, half-life and parameter estimation from data.",
    "description": (
        "The Ornstein-Uhlenbeck process **dx = θ(μ − x)dt + σ dW** comes from physics (the velocity of a "
        "particle subject to friction) and in finance describes anything that reverts to a mean: volatility, "
        "interest rates, spreads. **θ** is the speed of mean reversion per year, **μ** the long-term mean and "
        "**σ** the strength of the random shocks. The **half-life** ln 2/θ tells you how long it takes for half "
        "of a deviation to decay.\n\n"
        "How to read the charts: the fan shows simulated paths (exact discretisation) starting from x₀. The "
        "solid curve is the exact expected value μ + (x₀ − μ)e^(−θt), the shaded area the exact ±2σ(t) band. "
        "The dispersion does not grow without limit as in a random walk but saturates at the stationary "
        "standard deviation σ/√(2θ). The defaults resemble the actual VIX (mean around 19, half-life of about "
        "four weeks), starting from a stressed level of 35.\n\n"
        "The **fit** estimates θ, μ and σ from the AR(1) regression x_{t+1} = a + b·x_t: either on the actual "
        "VIX in your dataset (default) or on a simulated path with your parameters – which shows how precise the "
        "estimate is for a given data length (θ is notoriously hard to pin down). If the dataset has no VIX, the "
        "fit automatically uses the simulated path."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "theta", "type": "float", "default": 9.0, "min": 0.1, "max": 100.0, "step": 0.1,
         "label": "Speed of mean reversion θ", "unit": "per year"},
        {"name": "mu", "type": "float", "default": 20.0, "min": -50.0, "max": 100.0, "step": 0.5,
         "label": "Long-term mean μ"},
        {"name": "sigma", "type": "float", "default": 30.0, "min": 0.1, "max": 200.0, "step": 0.5,
         "label": "Volatility σ", "unit": "per √year"},
        {"name": "x0", "type": "float", "default": 35.0, "min": -50.0, "max": 150.0, "step": 0.5,
         "label": "Starting value x₀"},
        {"name": "horizon", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
         "label": "Horizon", "unit": "years"},
        {"name": "n_steps", "type": "int", "default": 126, "min": 10, "max": 1260, "step": 1,
         "label": "Time steps", "help": "126 steps over ½ year = one step per trading day."},
        {"name": "n_paths", "type": "int", "default": 500, "min": 10, "max": 2000, "step": 10,
         "label": "Simulated paths"},
        {"name": "fit_source", "type": "select", "default": "vix", "label": "Parameter fit",
         "options": [{"value": "vix", "label": "VIX from the dataset"},
                     {"value": "sim", "label": "Simulated path with your parameters"},
                     {"value": "none", "label": "No fit"}]},
        {"name": "fit_years", "type": "float", "default": 4.0, "min": 0.5, "max": 40.0, "step": 0.5,
         "label": "Length of the simulated fit path", "unit": "years",
         "help": "Only for the fit on the simulated path. One step per trading day (dt = 1/252)."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.stationary_variance",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.stationary_std",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.half_life",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.expected_value",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.variance_at_time",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.simulate",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.fit",
    ],
    "cost": "fast",
}

N_SHOW = 10         # paths drawn individually
FORECAST_DAYS = 63  # forecast cone after the fitted series (one quarter)
DT = 1 / 252


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _gapped(times, paths):
    """Several paths as one trace, separated by gaps (NaN), so the legend stays short."""
    xs, ys = [], []
    for p in paths:
        xs.extend(times.tolist() + [None])
        ys.extend(p.tolist() + [np.nan])
    return xs, np.asarray(ys, dtype=float)


def run(params, data, out):
    ou = OrnsteinUhlenbeckProcess(theta=params["theta"], mu=params["mu"], sigma=params["sigma"])
    x0, T, steps = params["x0"], params["horizon"], params["n_steps"]
    paths = ou.simulate(x0, T, n_steps=steps, n_paths=params["n_paths"], seed=params["seed"])
    t = np.linspace(0, T, steps + 1)
    mean_t, sd_t = ou.expected_value(x0, t), np.sqrt(ou.variance_at_time(t))
    mc_mean, mc_sd = paths.mean(axis=0), paths.std(axis=0)
    half_days = ou.half_life() * 252

    out.metric("Half-life (trading days)", half_days, "num", digits=0 if half_days >= 10 else 1,
               hint=f"ln 2/θ = {_fmt(ou.half_life(), 3)} years")
    out.metric("Stationary std σ/√(2θ)", ou.stationary_std(), "num",
               hint=f"stationary variance {_fmt(ou.stationary_variance(), 1)}")
    out.metric("E[X] at horizon (exact)", mean_t[-1], "num", hint=f"simulation: {_fmt(mc_mean[-1])}")
    out.metric("Std at horizon (exact)", sd_t[-1], "num", hint=f"simulation: {_fmt(mc_sd[-1])}")
    out.metric("Initial deviation remaining at horizon", np.exp(-ou.theta * T), "pct", digits=1,
               hint="e^(−θT): the share of x₀ − μ still left in the expected value")

    gx, gy = _gapped(t, paths[:N_SHOW])
    inside = float(np.mean(np.abs(paths[:, -1] - mean_t[-1]) <= 2 * sd_t[-1]))
    fan = [C.envelope("±2σ(t) (exact)", t, mean_t - 2 * sd_t, mean_t + 2 * sd_t, color=2),
           C.trace(f"{min(N_SHOW, len(paths))} simulated paths", gx, gy, color=1, width=1, decimate_to=None),
           C.trace("Expected value (exact)", t, mean_t, color=2, width=3),
           C.trace(f"Mean of all {params['n_paths']:,} paths", t, mc_mean, color=3, dash=True)]
    vlines = [C.vline(ou.half_life(), "1 half-life")] if ou.half_life() < T else []
    out.chart(C.xy(
        "Path fan with exact expectation band", fan,
        x=C.axis("Time (years)"), y=C.axis("x"), hlines=[C.hline(ou.mu, "μ")], vlines=vlines, height=380,
        subtitle=f"The deviation from the mean shrinks exponentially. At the horizon, {_fmt(100 * inside, 0)}% of "
                 f"the {params['n_paths']:,} paths lie within the ±2σ(t) band (exact: 95.4%).",
    ))
    out.chart(C.xy(
        "Standard deviation: growth and saturation",
        [C.trace("σ(t) exact", t, sd_t, color=2, width=2.5),
         C.trace("σ(t) of the simulation", t, mc_sd, color=1, dash=True)],
        x=C.axis("Time (years)"), y=C.axis("Standard deviation"),
        hlines=[C.hline(ou.stationary_std(), "stationary σ/√(2θ)")], height=300,
        subtitle="Unlike in a random walk (√t), the uncertainty does not grow without bound.",
    ))
    end = paths[:, -1]
    centers, heights = C.histogram(end, bins=40)
    grid = np.linspace(centers[0], centers[-1], 300)
    out.chart(C.xy(
        "Distribution at the horizon",
        [C.trace("Simulation", centers, heights, type="bar", color=1),
         C.trace("Exact N(E[X_T], Var[X_T])", grid, stats.norm.pdf(grid, mean_t[-1], sd_t[-1]), color=2),
         C.trace("Stationary N(μ, σ²/2θ)", grid, stats.norm.pdf(grid, ou.mu, ou.stationary_std()),
                 color="muted", dash=True)],
        x=C.axis(f"x after {_fmt(T, 1)} years"), y=C.axis("Density"), height=300,
        subtitle="The distribution is normal. After a few half-lives it is indistinguishable from the "
                 "stationary distribution.",
    ))
    if ou.mu > 0 and ou.mu - 2 * ou.stationary_std() < 0:
        out.note("Caution with VIX applications: the OU process is normally distributed and can turn negative – "
                 "actual volatility cannot.", "warning")

    if params["fit_source"] != "none":
        _fit(params, data, out, ou)


def _fit(params, data, out, true_ou):
    fit_source = params["fit_source"]
    if fit_source == "vix" and (data is None or data.vix is None):
        out.note("Your dataset contains no VIX, so the fit uses a simulated path with your parameters. To fit "
                 "the actual VIX, select the ‘S&P 500’ dataset.")
        fit_source = "sim"
    if fit_source == "vix":
        series = np.asarray(data.vix, dtype=float)
        derived = (data.meta or {}).get("source") in ("synthetic", "crisis")
        label = "derived VIX" if derived else "VIX"
        source = "the VIX derived from the returns" if derived else "the dataset's VIX"
    else:
        n_fit = int(round(params["fit_years"] * 252))
        series = true_ou.simulate(true_ou.mu, n_fit * DT, n_steps=n_fit, n_paths=1, seed=params["seed"] + 1)[0]
        label, source = "simulated path", "the simulated path"

    try:
        fit = OrnsteinUhlenbeckProcess.fit(series, dt=DT)
    except ValueError as exc:
        msg = str(exc)
        reason = ("The AR(1) slope is ≥ 1: the series shows no mean reversion (random walk or trend)."
                  if ">= 1" in msg else
                  "The AR(1) slope is ≤ 0: the series flips back and forth rather than reverting – "
                  "θ would be infinite." if "<= 0" in msg else msg)
        out.note(f"No OU fit possible on {source}. {reason}", "warning")
        return

    b = np.exp(-fit.theta * DT)
    rows = [["θ (per year)", fit.theta], ["μ", fit.mu], ["σ (per √year)", fit.sigma],
            ["Half-life (trading days)", fit.half_life() * 252], ["Stationary std", fit.stationary_std()],
            ["AR(1) slope b = e^(−θ·dt)", b]]
    truth = [true_ou.theta, true_ou.mu, true_ou.sigma, true_ou.half_life() * 252, true_ou.stationary_std(),
             np.exp(-true_ou.theta * DT)]
    if fit_source == "sim":
        rows = [r + [tv, r[1] / tv - 1 if tv else None] for r, tv in zip(rows, truth)]
        out.table(f"Parameter estimate from {len(series):,} days ({_fmt(len(series) / 252, 1)} years)",
                  ["Parameter", "Estimated", "True", "Error"], rows, formats=["text", "num", "num", "pct"],
                  note="θ needs data covering many half-lives: for short series it is systematically overestimated.")
    else:
        out.table(f"OU fit: {label} · {data.name} · {len(series):,} days", ["Parameter", "Estimated", "Your simulation"],
                  [r + [tv] for r, tv in zip(rows, truth)], formats=["text", "num", "num"],
                  note="On the right, for comparison: the parameters you are simulating with above.")
    out.metric("Fit: half-life (trading days)", fit.half_life() * 252, "num", digits=0,
               hint=f"estimated from {source}")
    out.metric("Fit: μ", fit.mu, "num", hint=f"θ = {_fmt(fit.theta, 1)} per year, σ = {_fmt(fit.sigma, 1)}")

    hist = series[-252:]
    h_days = np.arange(-len(hist) + 1, 1)
    f_days = np.arange(0, FORECAST_DAYS + 1)
    tf = f_days * DT
    f_mean = fit.expected_value(series[-1], tf)
    f_sd = np.sqrt(fit.variance_at_time(tf))
    out.chart(C.xy(
        f"Fit and forecast cone – {label}",
        [C.envelope("±2σ forecast band", f_days, f_mean - 2 * f_sd, f_mean + 2 * f_sd, color=2),
         C.trace(f"Observed (last {len(hist)} days)", h_days, hist, color=1, width=1.5),
         C.trace("Forecast E[X]", f_days, f_mean, color=2, width=2.5)],
        x=C.axis("Trading days relative to the last value (0 = today)", fmt="int"),
        y=C.axis("VIX" if fit_source == "vix" else "x"),
        hlines=[C.hline(fit.mu, "estimated μ")], height=340,
        subtitle=f"With the estimated model, half of today's deviation has decayed after "
                 f"{_fmt(fit.half_life() * 252, 0)} trading days.",
    ))
