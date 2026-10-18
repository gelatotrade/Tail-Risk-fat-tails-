"""Fokker-Planck tail risk: forecast distribution, tail metrics and a small 3D risk surface."""

import numpy as np
from scipy import stats

from src.physics.fokker_planck import FokkerPlanckTailRisk

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "fokker-planck-forecast",
    "title": "Fokker-Planck: tail forecast",
    "group": "physics",
    "order": 45,
    "summary": "Projects today's distribution h days ahead with the fractional Fokker-Planck equation and "
               "measures how much the tails grow relative to a Gaussian.",
    "description": (
        "`FokkerPlanckTailRisk` starts with a Gaussian bell centred on today's return and lets it flow for h trading "
        "days according to the fractional Fokker-Planck equation: mean reversion θ towards the centre, plus Lévy "
        "diffusion of order **α** whose strength grows with the square of the **volatility regime** "
        "(D_α = 0.01·regime² per year). At α = 2 ordinary diffusion is used instead. x is the daily return in units "
        "of today's standard deviation σ: the forecast shows what the distribution of the daily return looks like "
        "in h days – not the return accumulated over h days. Because diffusion in the model is small, mean "
        "reversion even narrows the bell slightly as the horizon grows.\n\n"
        "How to read the result: the body of the distribution barely changes over a few weeks, but far out mass "
        "builds up that a normal distribution would never have. The ratios in the table divide the tail mass "
        "beyond kσ by the Gaussian value – at 2σ they are ≈ 1, at 5σ often in double digits.\n\n"
        "The **3D risk surface** (`generate_3d_risk_surface`) repeats the forecast on a small grid of volatility "
        "regime × horizon, each starting from N(0, 1), and shows the tail probability beyond the selected "
        "threshold. Each cell is a separate simulation, so keep the grid small in the browser."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 1.0, "max": 2.0, "step": 0.05,
         "label": "Fractional order α", "help": "2 = ordinary diffusion; smaller = fatter tails."},
        {"name": "theta", "type": "float", "default": 0.1, "min": 0.0, "max": 3.0, "step": 0.05,
         "label": "Mean reversion θ (per year)"},
        {"name": "horizon", "type": "int", "default": 20, "min": 1, "max": 252, "step": 1,
         "label": "Horizon (trading days)"},
        {"name": "vol_regime", "type": "float", "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
         "label": "Volatility regime", "help": "1 = normal, 2 = double volatility (four times the Lévy diffusion)."},
        {"name": "current_return", "type": "float", "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
         "label": "Today's return (in σ)", "help": "Centre of the initial bell."},
        {"name": "current_vol", "type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
         "label": "Today's width (in σ)"},
        {"name": "surface", "type": "bool", "default": True, "label": "Compute 3D risk surface"},
        {"name": "surface_n", "type": "int", "default": 3, "min": 2, "max": 5, "step": 1,
         "label": "Surface: points per axis", "show_if": {"surface": [True]}},
        {"name": "surface_h_max", "type": "int", "default": 60, "min": 10, "max": 120, "step": 5,
         "label": "Surface: maximum horizon (days)", "show_if": {"surface": [True]}},
        {"name": "surface_vol_max", "type": "float", "default": 3.0, "min": 1.5, "max": 4.0, "step": 0.5,
         "label": "Surface: maximum volatility regime", "show_if": {"surface": [True]}},
        {"name": "threshold", "type": "float", "default": 4.0, "min": 2.0, "max": 5.0, "step": 0.5,
         "label": "Surface: threshold (σ)", "show_if": {"surface": [True]}},
    ],
    "covers": [
        "src.physics.fokker_planck.FokkerPlanckTailRisk",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.initial_distribution",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.forecast_distribution",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.compute_tail_risk_metrics",
        "src.physics.fokker_planck.FokkerPlanckTailRisk.generate_3d_risk_surface",
    ],
    "cost": "medium",
}

MAX_DAYS = 1300   # simulated forecast days in one run (~4 Crank-Nicolson steps per day, dense 500 x 500)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def run(params, data, out):
    alpha, theta, h, regime = params["alpha"], params["theta"], params["horizon"], params["vol_regime"]
    n_s = params["surface_n"]
    vols = np.round(np.linspace(0.5, params["surface_vol_max"], n_s), 2)
    h_max = params["surface_h_max"]
    hs = np.linspace(5, h_max, n_s)
    hs = np.unique(np.clip(np.round(hs / 5) * 5, 5, h_max).astype(int))  # round horizons to 5-day steps
    surface_days = int(len(vols) * hs.sum()) if params["surface"] else 0
    if h + surface_days > MAX_DAYS:
        raise ToolError(f"Forecast plus risk surface would require {h + surface_days:,} simulated days (limit "
                        f"{MAX_DAYS:,}). Reduce the surface grid or the maximum horizon, or switch the surface off.")

    model = FokkerPlanckTailRisk(alpha=alpha, theta=theta)
    solver = model.solver
    x = solver.x
    P0 = model.initial_distribution(params["current_return"], params["current_vol"])
    out.progress("Forecasting the distribution…", 0.05)
    P = model.forecast_distribution(P0, horizon=h, vol_regime=regime)
    m = model.compute_tail_risk_metrics(P)
    mo, tp = m["moments"], m["tail_probabilities"]
    m0 = solver.compute_moments(P0)

    out.metric(f"Standard deviation in {h} day{'' if h == 1 else 's'}", mo["std"], "num",
               hint=f"in today's σ; start {_fmt(m0['std'])}. θ narrows the bell, Lévy diffusion widens it")
    out.metric("Excess kurtosis", mo["kurtosis"], "num", hint="normal distribution = 0",
               status="critical" if mo["kurtosis"] > 6 else "serious" if mo["kurtosis"] > 1 else None)
    out.metric("Mass beyond 3σ", tp["3sigma"]["total"], "pct", hint=f"Gaussian: {_fmt(200 * stats.norm.sf(3))}%")
    for key, k in (("4sigma", 4), ("5sigma", 5)):
        ratio = tp[key]["ratio_to_gaussian"]
        out.metric(f"Ratio to Gaussian ({k}σ)", ratio, "x", hint=f"tail mass beyond {k}σ ÷ Gaussian value",
                   status="critical" if ratio > 20 else "serious" if ratio > 3 else None)
    out.metric("Fat-tail factor 2/α", m["fat_tail_factor"], "num", hint=f"tail index α = {_fmt(m['tail_index'])}")

    # --- chart 1: start vs forecast density, log scale
    floor = 1e-7
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        f"Distribution today and in {h} day{'' if h == 1 else 's'}",
        [C.trace(f"In {h} day{'' if h == 1 else 's'} (Fokker-Planck)", x, keep(P), color=1, width=2.5),
         C.trace("Today (Gaussian)", x, keep(P0), color=2, width=1.5),
         C.trace("Gaussian with equal variance", x, keep(stats.norm.pdf(x, mo["mean"], mo["std"])), color="muted", dash=True)],
        x=C.axis("Return (in today's σ)"), y=C.axis("Density", type="log", range=(floor, 3 * float(max(P.max(), P0.max())))),
        subtitle="Logarithmic scale: the body stays almost unchanged, while the tails rise above the Gaussian parabola.",
    ))

    # --- chart 2: how many times more tail mass than a Gaussian, as a function of the distance in sigma
    ks = np.linspace(0.5, 6.0, 56)
    tails = np.array([sum(solver.tail_probability(P, k)) for k in ks])
    ratio = tails / (2 * stats.norm.sf(ks))
    out.chart(C.xy(
        "Tail mass relative to Gaussian by distance",
        [C.trace("Fokker-Planck forecast", ks, np.where(tails > 1e-12, ratio, np.nan), color=1, width=2.5),
         C.trace("Normal distribution = 1", [ks[0], ks[-1]], [1.0, 1.0], color="muted", dash=True)],
        x=C.axis("Distance k (in σ of the forecast)"), y=C.axis("P(|X − μ| > kσ) ÷ Gaussian value", type="log"),
        subtitle="Up to about 3σ the forecast stays close to the Gaussian value – further out it has many times the mass.",
    ))

    rows = []
    for k in (2, 3, 4, 5):
        t = tp[f"{k}sigma"]
        rows.append([f"{k}σ", t["left"], t["right"], t["total"], float(2 * stats.norm.sf(k)), t["ratio_to_gaussian"]])
    out.table("Tail metrics of the forecast", ["Threshold", "Left", "Right", "Total", "Gaussian total", "Ratio"],
              rows, formats=["text", "pct", "pct", "pct", "pct", "x"],
              note="From compute_tail_risk_metrics: mass beyond mean ± k·σ of the forecast density.")

    if not params["surface"]:
        out.note("3D risk surface switched off.", "info")
        return

    out.progress(f"3D risk surface: {len(vols) * len(hs)} forecasts…", 0.3)
    thr = params["threshold"]
    Z = model.generate_3d_risk_surface(vols, hs, threshold_sigma=thr)
    points = []
    if params["current_return"] == 0 and params["current_vol"] == 1:
        own = float(sum(solver.tail_probability(P, thr)))
        points.append(C.path3d("Your forecast", [h], [regime], [own], color="critical", mode="markers", size=9))
    out.chart(C.surface(
        f"Risk surface: P(|X| > {_fmt(thr, 1)}σ)", hs.tolist(), vols.tolist(), Z,
        x=C.axis("Horizon (days)", fmt="int"), y=C.axis("Volatility regime"), z=C.axis("Tail probability", fmt="pct"),
        points=points, height=520,
        subtitle=f"{len(vols)} × {len(hs)} forecasts from N(0, 1). Gaussian level: {200 * stats.norm.sf(thr):.3g}%"
                 + ". Drag to rotate, scroll to zoom.",
    ))
    g_thr = float(2 * stats.norm.sf(thr))
    out.table(f"Risk surface as a multiple of the Gaussian value ({200 * stats.norm.sf(thr):.3g}%)",
              ["Volatility regime"] + [f"{d} days" for d in hs],
              [[float(v)] + [float(z) / g_thr for z in Z[i]] for i, v in enumerate(vols)],
              formats=["num"] + ["x"] * len(hs),
              note=f"P(|X| > {_fmt(thr, 1)}σ) of each cell divided by the value for a normal distribution.")
