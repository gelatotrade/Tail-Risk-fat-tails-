"""Stochastic volatility (CIR-like OU with upward jumps): price paths, volatility clustering, drawdowns."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import VolatilityOUModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "stochastic-vol",
    "title": "Stochastic volatility with volatility jumps",
    "group": "physics",
    "order": 60,
    "summary": "When volatility itself fluctuates randomly and jumps: volatility clusters, fat tails and "
               "drawdowns – even though each individual day is normally distributed.",
    "description": (
        "Volatility follows a mean-reverting process **dσ = κ(θ − σ)dt + η√σ dW + dJ**: it reverts at speed "
        "**κ** to the level **θ**, fluctuates with the vol of vol **η** and jumps upwards on average **λ times "
        "per year** (each jump adds 10 percentage points on average, exponentially distributed). The price is a "
        "geometric Brownian motion with exactly this volatility.\n\n"
        "How to read the charts: each individual day is normally distributed – yet the distribution of all daily "
        "returns has fat tails, because calm and turbulent phases are mixed. The clustering chart shows the "
        "autocorrelation: the returns themselves are uncorrelated, but their **absolute values** are not – large "
        "moves are followed by large moves. This is exactly what you see in real markets too.\n\n"
        "Parameters: a large κ makes volatility shocks decay quickly (short clusters); large η and λ produce more "
        "violent swings. Because of the jumps, the long-term volatility level lies above θ, on average at about "
        "θ + λ·0.1/κ."
    ),
    "needs": [],
    "params": [
        {"name": "kappa", "type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
         "label": "Speed of mean reversion κ", "unit": "per year"},
        {"name": "theta", "type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.01,
         "label": "Long-term volatility θ", "help": "0.2 = 20% p.a."},
        {"name": "eta", "type": "float", "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
         "label": "Vol of vol η"},
        {"name": "jump_intensity", "type": "float", "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
         "label": "Volatility jumps per year λ", "help": "Each jump raises σ by 10 percentage points on average."},
        {"name": "sigma0", "type": "float", "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
         "label": "Initial volatility σ₀"},
        {"name": "drift", "type": "float", "default": 0.0, "min": -0.2, "max": 0.3, "step": 0.01,
         "label": "Drift r", "unit": "per year"},
        {"name": "years", "type": "float", "default": 2.0, "min": 0.25, "max": 10.0, "step": 0.25,
         "label": "Horizon", "unit": "years", "help": "One time step = one trading day."},
        {"name": "n_paths", "type": "int", "default": 100, "min": 1, "max": 500, "step": 10,
         "label": "Number of paths"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel",
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel.simulate_volatility",
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel.simulate_price",
    ],
    "cost": "fast",
}

N_SHOW = 8
MAX_STEPS = 300_000  # paths x days (two pure-Python loops per path)
LAGS = np.arange(1, 41)
QUANTS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _gapped(times, rows):
    xs, ys = [], []
    for row in rows:
        xs.extend(times.tolist() + [None])
        ys.extend(row.tolist() + [np.nan])
    return xs, np.asarray(ys, dtype=float)


def _acf(x, lags):
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return np.zeros(len(lags))
    return np.array([np.sum(x[:-k] * x[k:]) / denom if k < len(x) else np.nan for k in lags])


def _max_drawdown(prices):
    peak = np.maximum.accumulate(prices, axis=1)
    return np.min(prices / peak - 1.0, axis=1)  # <= 0


def run(params, data, out):
    n_paths, years = params["n_paths"], params["years"]
    n_steps = int(round(years * 252))
    if n_paths * n_steps > MAX_STEPS:
        raise ToolError(f"{n_paths:,} paths × {n_steps:,} days is too much for the browser (limit "
                        f"{MAX_STEPS:,} daily steps). Reduce the number of paths or the horizon.")
    model = VolatilityOUModel(kappa=params["kappa"], theta=params["theta"], eta=params["eta"],
                              jump_intensity=params["jump_intensity"])

    prices, vols, rets = [], [], []
    for i in range(n_paths):
        if i % 20 == 0:
            out.progress(f"Path {i + 1} of {n_paths}…", i / n_paths)
        # seed once; later paths continue the same random stream
        res = model.simulate_price(100.0, params["sigma0"], years, n_steps=n_steps, r=params["drift"],
                                   seed=params["seed"] if i == 0 else None)
        prices.append(res["prices"])
        vols.append(res["volatility"])
        rets.append(res["returns"])
    prices, vols, rets = np.array(prices), np.array(vols), np.array(rets)
    t = res["times"]
    pooled = rets.ravel()
    dd = _max_drawdown(prices)
    level_jumps = params["theta"] + params["jump_intensity"] * 0.1 / params["kappa"]

    acf_abs = np.nanmean([_acf(np.abs(row), LAGS) for row in rets], axis=0)
    acf_raw = np.nanmean([_acf(row, LAGS) for row in rets], axis=0)

    out.metric("Median final price", np.median(prices[:, -1]), "num", hint=f"start 100 · 5% quantile "
               f"{_fmt(np.percentile(prices[:, -1], 5), 1)}")
    out.metric("Mean volatility σ_t", vols.mean(), "pct", digits=1,
               hint=f"θ = {_fmt(100 * params['theta'], 0)}%, with jumps ≈ {_fmt(100 * level_jumps, 0)}%")
    out.metric("Realised volatility p.a.", np.std(pooled) * np.sqrt(252), "pct", digits=1)
    out.metric("Highest volatility", vols.max(), "pct", digits=0, hint="across all paths and days")
    out.metric("Excess kurtosis of daily returns", stats.kurtosis(pooled), "num", digits=1,
               hint="0 = normal distribution", status="serious" if stats.kurtosis(pooled) > 3 else None)
    out.metric("Autocorrelation |r|, lag 1", acf_abs[0], "num", digits=3,
               hint=f"returns themselves: {_fmt(acf_raw[0], 3)}".replace("-", "−") + ". |r| clearly > 0 = volatility clustering")
    out.metric("Max drawdown (median)", np.median(dd), "pct", digits=1)
    out.metric("Max drawdown (worst 5%)", np.percentile(dd, 5), "pct", digits=1,
               status="serious" if np.percentile(dd, 5) < -0.3 else None)
    out.metric("99% VaR (1 day)", -np.percentile(pooled, 1), "pct", hint="from all simulated days")

    show = min(N_SHOW, n_paths)
    gx, gy = _gapped(t, prices[:show])
    q05, q50, q95 = np.percentile(prices, [5, 50, 95], axis=0)
    band = [C.envelope("5–95% of all paths", t, q05, q95, color=2)] if n_paths >= 20 else []
    out.chart(C.xy(
        "Price paths",
        band + [C.trace(f"{show} example path{'' if show == 1 else 's'}", gx, gy, color=1, width=1, decimate_to=None),
                C.trace("Median of all paths", t, q50, color=2, width=2.5)],
        x=C.axis("Time (years)"), y=C.axis("Price (start = 100)"), hlines=[C.hline(100.0)], height=360,
        subtitle=f"{n_paths} path{'' if n_paths == 1 else 's'}, one step per trading day."
                 + (" The shaded area covers the middle 90% of all paths on each day." if band else ""),
    ))
    gx, gy = _gapped(t, vols[:show])
    v05, v95 = np.percentile(vols, [5, 95], axis=0)
    vband = [C.envelope("5–95% of all paths", t, v05, v95, color=2)] if n_paths >= 20 else []
    out.chart(C.xy(
        "Volatility σ_t",
        vband + [C.trace(f"{show} example path{'' if show == 1 else 's'}", gx, gy, color=1, width=1, decimate_to=None),
                 C.trace("Mean of all paths", t, vols.mean(axis=0), color=2, width=2.5)],
        x=C.axis("Time (years)"), y=C.axis("Volatility p.a.", fmt="pct", digits=0),
        hlines=[C.hline(params["theta"], "θ"), C.hline(level_jumps, "θ + λ·0.1/κ")], height=340,
        subtitle="Vertical jumps are volatility shocks; afterwards σ decays at speed κ.",
    ))
    out.chart(C.xy(
        "Volatility clustering: autocorrelation",
        [C.trace("|Return|", LAGS, acf_abs, color=1, width=2.5),
         C.trace("Return", LAGS, acf_raw, color="muted", dash=True)],
        x=C.axis("Lag (trading days)", fmt="int"), y=C.axis("Autocorrelation"), hlines=[C.hline(0.0)],
        height=300, subtitle="Returns are uncorrelated, but their absolute values are not: large moves come in clusters.",
    ))
    centers, heights = C.histogram(pooled, bins=100)
    sd = float(np.std(pooled))
    grid = np.linspace(centers[0], centers[-1], 300)
    gauss = stats.norm.pdf(grid, np.mean(pooled), sd)
    floor = float(np.min(heights[heights > 0])) / 3
    out.chart(C.xy(
        "Distribution of all daily returns – log scale",
        [C.trace("Simulation", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace("Normal distribution, same volatility", grid, np.where(gauss >= floor, gauss, np.nan),
                 color="muted", dash=True)],
        x=C.axis("Daily return", fmt="pct", digits=1),
        y=C.axis("Density (log)", type="log", range=(floor, float(heights.max()) * 3)), height=320,
        subtitle="A mix of calm and turbulent phases: more quiet days and more extreme days than under a Gaussian.",
    ))

    tot = prices[:, -1] / 100.0 - 1.0
    out.table("Distribution across paths", ["Quantile", "Final price", "Total return", "Max drawdown", "Highest σ_t"],
              [[f"{int(q * 100)}%", np.quantile(prices[:, -1], q), np.quantile(tot, q), np.quantile(dd, q),
                np.quantile(vols.max(axis=1), q)] for q in QUANTS],
              formats=["text", "num", "pct", "pct", "pct"],
              note="Each column is sorted independently (the 5% quantile of the drawdown does not necessarily "
                   "belong to the same path as the 5% quantile of the final price).")
