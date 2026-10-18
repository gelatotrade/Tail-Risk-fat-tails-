"""Price paths with Lévy jumps: fan chart, drawdowns, path VaR and the path in Lévy phase space."""

import numpy as np

from src.physics.levy_flight import LevyFlightProcess, LevyStableDistribution, levy_flight_3d_coordinates

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "levy-flight",
    "title": "Lévy flight: price paths with jumps",
    "group": "physics",
    "order": 30,
    "summary": "Simulates prices as Brownian motion plus Lévy jumps and shows how rare large jumps change "
               "drawdowns and Value at Risk.",
    "description": (
        "`LevyFlightProcess` extends geometric Brownian motion with jumps: in each time step the log price "
        "receives (μ − σ²/2)·dt + σ·dW plus a Poisson number of jumps (on average **λ per year**) whose size "
        "follows a Lévy-stable distribution. The **jump scale γ** is the typical jump size in per cent; **α** "
        "controls how often a typical jump turns into a crash.\n\n"
        "For comparison, the same process runs **without jumps** (λ = 0) with the same number of paths. The "
        "comparison is between distributions across all paths: the jumps consume their own random numbers, so the "
        "Brownian paths are not identical path by path despite the same seed. The price fan shows the median and "
        "quantile bands; the drawdown and crash curves show the share of paths that lose at least x% from their "
        "peak or on a single day, respectively. Value at Risk refers to the simple return over the entire "
        "horizon.\n\n"
        "The 3D plot turns the daily returns of the crash example (the red path) into coordinates in Lévy phase "
        "space: X = rolling volatility (mean = 1), Y = 2/Hill tail index (higher = fatter tails), "
        "Z = |return|/σ. Calm phases sit close to the origin; a black-swan jump towers far above them."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 0.8, "max": 2.0, "step": 0.05,
         "label": "α of jump sizes", "help": "2 = normally distributed jumps; smaller = huge jumps more often."},
        {"name": "beta", "type": "float", "default": -0.2, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "β of jump sizes", "help": "< 0: large jumps tend to be downward."},
        {"name": "jump_scale", "type": "float", "default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1, "unit": "%",
         "label": "Jump scale γ", "help": "Typical jump size as a log return in %."},
        {"name": "jump_intensity", "type": "float", "default": 12.0, "min": 0.0, "max": 100.0, "step": 1.0,
         "label": "Jumps per year λ"},
        {"name": "mu", "type": "float", "default": 7.0, "min": -20.0, "max": 30.0, "step": 0.5, "unit": "%",
         "label": "Drift μ p.a."},
        {"name": "sigma", "type": "float", "default": 15.0, "min": 1.0, "max": 80.0, "step": 1.0, "unit": "%",
         "label": "Volatility σ p.a.", "help": "Brownian component only, excluding jumps."},
        {"name": "years", "type": "float", "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25,
         "label": "Horizon (years)", "help": "252 trading days per year."},
        {"name": "n_paths", "type": "int", "default": 2000, "min": 100, "max": 5000, "step": 100,
         "label": "Number of paths",
         "help": "For 99% VaR and Expected Shortfall only the worst 1% of paths count."},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Confidence level",
         "options": [{"value": 0.95, "label": "95%"}, {"value": 0.99, "label": "99%"},
                     {"value": 0.995, "label": "99.5%"}]},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyFlightProcess",
        "src.physics.levy_flight.LevyFlightProcess.simulate_path",
        "src.physics.levy_flight.levy_flight_3d_coordinates",
    ],
    "cost": "fast",
}

S0 = 100.0
MAX_CELLS = 3_000_000  # paths x steps (two simulations)
LOG_CAP = 40.0          # |log price ratio| cap for statistics (e^40: an exploded path)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _stats(paths, c):
    """Simple total return, max drawdown per path, VaR/ES of the total return (all in decimals)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logp = np.clip(np.log(paths / S0), -LOG_CAP, LOG_CAP)
    logp = np.where(np.isnan(logp), LOG_CAP, logp)
    total = np.expm1(logp[:, -1])
    dd = np.expm1((logp - np.maximum.accumulate(logp, axis=1)).min(axis=1))  # <= 0
    var = -float(np.quantile(total, 1 - c))
    tail = total[total <= -var]
    es = -float(np.mean(tail)) if tail.size else var
    return logp, total, dd, var, es


def run(params, data, out):
    years, n_paths, c, seed = params["years"], params["n_paths"], float(params["confidence"]), params["seed"]
    n_steps = max(2, int(round(252 * years)))
    if 2 * n_paths * n_steps > MAX_CELLS:
        raise ToolError(f"{n_paths:,} paths × {n_steps:,} steps is too much for the browser. "
                        "Reduce the number of paths or the horizon.")

    jumps = LevyStableDistribution(alpha=params["alpha"], beta=params["beta"], gamma=params["jump_scale"], delta=0.0)
    mu, sigma, lam = params["mu"] / 100.0, params["sigma"] / 100.0, params["jump_intensity"]
    proc = LevyFlightProcess(jumps, mu=mu, sigma_gaussian=sigma, jump_intensity=lam)
    plain = LevyFlightProcess(jumps, mu=mu, sigma_gaussian=sigma, jump_intensity=0.0)

    out.progress("Paths with Lévy jumps…", 0.1)
    paths = proc.simulate_path(S0, years, n_steps=n_steps, n_paths=n_paths, seed=seed)
    out.progress("Comparison without jumps…", 0.4)
    ref = plain.simulate_path(S0, years, n_steps=n_steps, n_paths=n_paths, seed=seed)

    exploded = int(np.sum(~np.isfinite(paths[:, -1]) | (paths[:, -1] <= 0)))
    logp, total, dd, var, es = _stats(paths, c)
    _, total0, dd0, var0, es0 = _stats(ref, c)
    daily = np.diff(logp, axis=1)
    worst_day = daily.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        worst_day0 = np.diff(np.log(ref), axis=1).min(axis=1)

    lvl = f"{_fmt(100 * c, 1 if c == 0.995 else 0)}%"
    horizon = f"{_fmt(years, 2).rstrip('0').rstrip('.')} year" + ("" if years == 1 else "s")
    out.metric(f"{lvl} VaR ({horizon})", var, "pct", digits=1, hint=f"without jumps: {_fmt(100 * var0, 1)}%",
               status="critical" if var > 2 * var0 + 0.05 else "serious" if var > 1.25 * var0 else None)
    out.metric(f"{lvl} Expected Shortfall", es, "pct", digits=1, hint=f"without jumps: {_fmt(100 * es0, 1)}%")
    out.metric("Median max drawdown", -float(np.median(dd)), "pct", digits=1,
               hint=f"without jumps: {_fmt(-100 * np.median(dd0), 1)}%")
    out.metric(f"Max drawdown, worst {_fmt(100 * (1 - c), 1 if c == 0.995 else 0)}% of paths",
               -float(np.quantile(dd, 1 - c)), "pct", digits=1,
               hint=f"without jumps: {_fmt(-100 * np.quantile(dd0, 1 - c), 1)}%")
    out.metric("Paths with a crash day < −10%", float(np.mean(worst_day < np.log(0.9))), "pct", digits=1,
               hint="at least one day with a loss of more than 10%")
    out.metric("Jumps per path", lam * years, "num", digits=1, hint="expected: λ × horizon")

    # --- chart 1: price fan. Example path: its worst day is a "1-in-20" crash (5% quantile over all paths)
    t = np.arange(n_steps + 1) / 252.0
    price = S0 * np.exp(logp)
    q = np.quantile(price, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    i_crash = int(np.argmin(np.abs(worst_day - np.quantile(worst_day, 0.05))))
    crash_day = float(np.expm1(worst_day[i_crash]))
    order = np.argsort(total)
    i_mid = int(order[len(order) // 2])
    traces = [
        C.envelope("1–99%", t, q[0], q[4], color=1),
        C.envelope("5–95%", t, q[1], q[3], color=1),
        C.trace("Median", t, q[2], color=1, width=2.5),
        C.trace("Median without jumps", t, np.quantile(ref, 0.5, axis=0), color="muted", dash=True),
        C.trace(f"Crash example ({_fmt(100 * crash_day, 1).replace('-', '−')}%)", t, price[i_crash],
                color="critical", width=1.3),
        C.trace("Typical path", t, price[i_mid], color=3, width=1.3),
    ]
    out.chart(C.xy(
        "Price fan with Lévy jumps", traces,
        x=C.axis("Time (years)"), y=C.axis("Price (start = 100)"),
        subtitle="Bands: quantiles across all paths. The crash example is a path whose worst day (value in "
                 "brackets) is as bad as that of one path in 20.", height=380,
    ))

    # --- chart 2: drawdown exceedance curve
    dd_top = float(np.quantile(-np.r_[dd, dd0], 0.999))
    levels = np.linspace(0, float(np.clip(np.ceil(dd_top * 10) / 10, 0.3, 0.95)), 91)  # axis ends where paths end
    share = np.array([np.mean(-dd >= v) for v in levels])
    share0 = np.array([np.mean(-dd0 >= v) for v in levels])
    out.chart(C.xy(
        "Max drawdown across paths",
        [C.trace("With Lévy jumps", levels, share, color=2, width=2.5),
         C.trace("Without jumps", levels, share0, color="muted", dash=True)],
        x=C.axis("Max drawdown of at least", fmt="pct", digits=0), y=C.axis("Share of paths", fmt="pct", digits=0),
        subtitle="Example: at x = 30% the curve shows the share of paths that lose 30% or more from their peak "
                 "at some point.",
    ))

    # --- chart 3: distribution of the total return
    lo_r, hi_r = float(np.quantile(np.r_[total, total0], 0.002)), float(np.quantile(np.r_[total, total0], 0.998))
    edges = np.linspace(lo_r, hi_r, 61)
    w = edges[1] - edges[0]
    h1 = np.histogram(np.clip(total, lo_r, hi_r), bins=edges)[0] / (len(total) * w)
    h0 = np.histogram(np.clip(total0, lo_r, hi_r), bins=edges)[0] / (len(total0) * w)
    mid = 0.5 * (edges[:-1] + edges[1:])
    floor = 0.5 / (len(total) * w)
    out.chart(C.xy(
        f"Return distribution over {horizon}",
        [C.trace("With Lévy jumps", mid, np.where(h1 > 0, h1, np.nan), type="bar", color=2),
         C.trace("Without jumps", mid, np.where(h0 > 0, h0, np.nan), type="step", color="muted")],
        x=C.axis("Simple return", fmt="pct", digits=0),
        y=C.axis("Density", type="log", range=(floor, 3 * max(h1.max(), h0.max()))),
        vlines=[C.vline(-var, f"{lvl} VaR", color="serious")],
        subtitle="Log density: jumps fatten the left tail. Values beyond the 0.2% and 99.8% quantiles are "
                 "collected at the edges.",
    ))

    # --- chart 4: how often a single day crashes by at least x
    crash_levels = np.linspace(0.0, 0.3, 61)
    crash_share = np.array([np.mean(np.expm1(worst_day) <= -v) for v in crash_levels])
    crash_share0 = np.array([np.mean(np.expm1(worst_day0) <= -v) for v in crash_levels])
    out.chart(C.xy(
        "Crash days: worst day per path",
        [C.trace("With Lévy jumps", crash_levels, crash_share, color=2, width=2.5),
         C.trace("Without jumps", crash_levels, crash_share0, color="muted", dash=True)],
        x=C.axis("One-day loss of at least", fmt="pct", digits=0), y=C.axis("Share of paths", fmt="pct", digits=0),
        vlines=[C.vline(0.10, "−10%")],
        subtitle="Share of paths with at least one day that loses this much or more. Without jumps the curve "
                 f"drops to practically zero beyond about 4 daily σ (≈ {_fmt(400 * sigma / np.sqrt(252), 1)}%).",
    ))

    # --- chart 5: Lévy phase space of the crash example (the first `win` points are placeholders)
    rets = daily[i_crash]
    win = int(min(20, max(5, len(rets) // 4)))
    X, Y, Z = (np.asarray(v, dtype=float)[win:] for v in levy_flight_3d_coordinates(rets, window=win))
    days = np.arange(win + 1, len(rets) + 1)
    txt = [f"Day {d}: {100 * np.expm1(v):+.1f}%".replace("-", "−") for d, v in zip(days, rets[win:])]
    j = int(np.argmin(rets[win:]))
    out.chart(C.scatter3d(
        "The crash example in Lévy phase space",
        [C.path3d("Daily trajectory", X, Y, Z, color=1, mode="lines+markers", size=2, text=txt),
         C.path3d("Worst day", [X[j]], [Y[j]], [Z[j]], color="critical", mode="markers", size=8, text=[txt[j]])],
        x=C.axis("X: volatility (mean = 1)"), y=C.axis("Y: 2 / tail index"), z=C.axis("Z: |return| / σ"),
        subtitle=f"levy_flight_3d_coordinates with a {win}-day window on the daily returns of the red path. "
                 "Drag to rotate, scroll to zoom.",
        height=520,
    ))

    out.table(
        "With and without jumps",
        ["Metric", "With Lévy jumps", "Without jumps"],
        [[f"{lvl} VaR over {horizon}", var, var0],
         [f"{lvl} Expected Shortfall", es, es0],
         ["Median return", float(np.median(total)), float(np.median(total0))],
         ["Median max drawdown", -float(np.median(dd)), -float(np.median(dd0))],
         ["Worst day (median across paths)", float(np.expm1(np.median(worst_day))),
          float(np.expm1(np.median(worst_day0)))]],
        formats=["text", "pct", "pct"],
        note="The comparison is between distributions across all paths. The jumps consume their own random numbers, "
             "so the Brownian paths are not identical despite the same seed – only their statistics are.",
    )
    if exploded:
        out.note(f"{exploded:,} of {n_paths:,} paths exploded numerically or fell to zero – for α ≤ 1 even a single "
                 "jump has no finite mean. For the metrics these paths are capped "
                 f"at e^±{LOG_CAP:.0f}.", "warning")
    if lam == 0:
        out.note("λ = 0: without jumps, both simulations are identical.", "info")
