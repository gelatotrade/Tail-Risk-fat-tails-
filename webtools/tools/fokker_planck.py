"""Fokker-Planck equation: how a return distribution evolves under drift, diffusion and Lévy jumps."""

import numpy as np
from scipy import stats

from src.physics.fokker_planck import FokkerPlanckSolver

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "fokker-planck",
    "title": "Fokker-Planck: density over time",
    "group": "physics",
    "order": 40,
    "summary": "Solves the (fractional) Fokker-Planck equation and shows how a narrow Gaussian bell turns into "
               "a fat-tailed distribution.",
    "description": (
        "The **Fokker-Planck equation** describes not a single price path but the entire probability density "
        "P(x, t) and how it flows: ∂P/∂t = −∂(μP)/∂x + diffusion. The drift μ(x) = −θ(x − μ∞) pulls mass back "
        "towards the equilibrium μ∞ (mean reversion, θ = speed). In the **standard model** the mass diffuses with "
        "D(x) = D₀(1 + λ|x|^β): the larger λ, the more turbulent it becomes far from the centre – volatility "
        "clustering. In the **fractional model** the Riesz operator D_α ∂^α/∂|x|^α replaces the diffusion: "
        "probability then jumps over long distances (Lévy flights), and for α < 2 power-law tails emerge.\n\n"
        "How to read the charts: in the heatmap time runs downwards and the colour is the log density "
        "(dark = high). The further the strong colouring extends to the left and right, the fatter the tails; a "
        "Gaussian bell would stay pale at the edges. The density curves compare the result with a normal "
        "distribution of equal variance, and the third chart tracks the mass beyond 3σ and 4σ over time. x is a "
        "standardised return; time is dimensionless.\n\n"
        "Numerics: Crank-Nicolson with a matrix factorised once, grid [−10, 10], absorbing boundaries, and the "
        "mass is renormalised after every step. Very fat tails (small α, large D_α) reach the edge of the grid – "
        "variance and kurtosis then refer to the distribution truncated to [−10, 10]."
    ),
    "needs": [],
    "params": [
        {"name": "model", "type": "select", "default": "fractional", "label": "Model",
         "options": [{"value": "fractional", "label": "Fractional (Lévy diffusion, α < 2)"},
                     {"value": "standard", "label": "Standard (diffusion D(x), volatility clustering)"}]},
        {"name": "alpha", "type": "float", "default": 1.5, "min": 0.5, "max": 1.95, "step": 0.05,
         "label": "Fractional order α", "help": "Smaller = longer jumps, fatter tails.",
         "show_if": {"model": ["fractional"]}},
        {"name": "D_alpha", "type": "float", "default": 0.5, "min": 0.05, "max": 2.0, "step": 0.05,
         "label": "Lévy diffusion D_α", "show_if": {"model": ["fractional"]}},
        {"name": "D0", "type": "float", "default": 0.5, "min": 0.05, "max": 2.0, "step": 0.05,
         "label": "Base diffusion D₀", "show_if": {"model": ["standard"]}},
        {"name": "lam", "type": "float", "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1,
         "label": "Volatility clustering λ", "help": "D(x) = D₀(1 + λ|x|^β).",
         "show_if": {"model": ["standard"]}},
        {"name": "beta_exp", "type": "float", "default": 1.0, "min": 0.5, "max": 2.5, "step": 0.1,
         "label": "Non-linearity β", "show_if": {"model": ["standard"]}},
        {"name": "theta", "type": "float", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
         "label": "Mean reversion θ", "help": "0 = no restoring force."},
        {"name": "mu_inf", "type": "float", "default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
         "label": "Equilibrium μ∞"},
        {"name": "x0", "type": "float", "default": 2.0, "min": -5.0, "max": 5.0, "step": 0.1,
         "label": "Starting point x₀", "help": "Centre of the Gaussian bell at t = 0."},
        {"name": "sigma0", "type": "float", "default": 0.5, "min": 0.2, "max": 3.0, "step": 0.1,
         "label": "Initial width σ₀"},
        {"name": "T", "type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5, "label": "Time horizon T"},
        {"name": "dt", "type": "float", "default": 0.01, "min": 0.002, "max": 0.05, "step": 0.002,
         "label": "Time step dt"},
        {"name": "n_grid", "type": "int", "default": 301, "min": 101, "max": 501, "step": 50, "label": "Grid points"},
    ],
    "covers": [
        "src.physics.fokker_planck.FokkerPlanckSolver",
        "src.physics.fokker_planck.FokkerPlanckSolver.drift_coefficient",
        "src.physics.fokker_planck.FokkerPlanckSolver.diffusion_coefficient",
        "src.physics.fokker_planck.FokkerPlanckSolver.build_operator_matrix",
        "src.physics.fokker_planck.FokkerPlanckSolver.build_fractional_operator",
        "src.physics.fokker_planck.FokkerPlanckSolver.solve",
        "src.physics.fokker_planck.FokkerPlanckSolver.compute_moments",
        "src.physics.fokker_planck.FokkerPlanckSolver.tail_probability",
    ],
    "cost": "fast",
}

MAX_WORK = 4.0e8      # fractional: steps x n_grid^2 (dense Crank-Nicolson), ~1.5 s native
MAX_STEPS = 5000
G3, G4 = 2 * stats.norm.sf(3), 2 * stats.norm.sf(4)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sci(x, digits=1):
    """Scientific notation for the numerics table: 2.3·10⁻⁴."""
    if not np.isfinite(x) or x == 0:
        return _fmt(float(x), digits)
    mant, exp = f"{x:.{digits}e}".split("e")
    return mant + "·10" + str(int(exp)).translate(_SUP)


def run(params, data, out):
    frac = params["model"] == "fractional"
    n, T, dt = params["n_grid"], params["T"], params["dt"]
    n_steps = int(round(T / dt))
    if n_steps > MAX_STEPS or (frac and n_steps * n * n > MAX_WORK):
        raise ToolError(f"{n_steps:,} time steps on {n} grid points is too much for the browser. "
                        "Increase dt, reduce T or use fewer grid points.")
    alpha = params["alpha"] if frac else 2.0
    fp_params = {"theta": params["theta"], "mu_inf": params["mu_inf"], "D0": params["D0"],
                 "lambda": params["lam"], "beta": params["beta_exp"], "D_alpha": params["D_alpha"]}
    solver = FokkerPlanckSolver(x_min=-10, x_max=10, n_grid=n, alpha=alpha)
    x = solver.x
    x0, s0 = params["x0"], params["sigma0"]
    P0 = np.exp(-0.5 * ((x - x0) / s0) ** 2)
    P0 /= np.sum(P0) * solver.dx

    out.progress("Crank-Nicolson time steps…", 0.1)
    times, Pe = solver.solve(P0, T=T, dt=dt, params=fp_params, use_fractional=frac)
    moments = [solver.compute_moments(P) for P in Pe]
    tails3 = np.array([sum(solver.tail_probability(P, 3)) for P in Pe])
    tails4 = np.array([sum(solver.tail_probability(P, 4)) for P in Pe])
    end = moments[-1]

    drift0 = float(solver.drift_coefficient(np.array([x0]), 0, fp_params)[0])
    out.metric("Mean at T", end["mean"], "num", hint=f"start {_fmt(x0, 1)} · equilibrium {_fmt(params['mu_inf'], 1)}".replace("-", "−"))
    out.metric("Standard deviation σ at T", end["std"], "num", hint=f"start {_fmt(s0, 1)}")
    out.metric("Excess kurtosis at T", end["kurtosis"], "num", digits=1, hint="normal distribution = 0",
               status="critical" if end["kurtosis"] > 6 else "serious" if end["kurtosis"] > 3 else None)
    out.metric("Mass beyond 3σ", tails3[-1], "pct", hint=f"Gaussian: {_fmt(100 * G3)}%")
    out.metric("Ratio to Gaussian (4σ)", tails4[-1] / G4, "x", hint="mass beyond 4σ ÷ Gaussian value",
               status="critical" if tails4[-1] / G4 > 20 else "serious" if tails4[-1] / G4 > 3 else None)
    out.metric("Restoring force μ(x₀)", drift0, "num", hint="drift at the starting point: −θ(x₀ − μ∞)")

    # --- chart 1: heatmap of log density over (x, t)
    fstep = max(1, int(np.ceil(len(times) / 34)))
    rows = list(range(0, len(times), fstep))
    if rows[-1] != len(times) - 1:
        rows.append(len(times) - 1)
    xstep = max(1, int(np.ceil(n / 75)))
    logP = np.log10(np.maximum(Pe[rows][:, ::xstep], 1e-12))
    top = float(np.max(logP))
    logP = np.round(np.maximum(logP, top - 6), 3)
    out.chart(C.heatmap(
        "Probability density P(x, t)",
        np.round(x[::xstep], 3), np.round(times[rows], 3), logP,
        x=C.axis("x (standardised return)"), y=C.axis("Time t", type="linear"),
        z=C.axis("log₁₀ density"), scale="sequential", height=420, gap=0,
        subtitle="Time runs downwards; colour = log₁₀ density over 6 decades (dark = high). The further the strong "
                 "colouring reaches outwards, the fatter the tails.",
    ))

    # --- chart 2: snapshots on a log scale vs a Gaussian with the same variance
    picks = sorted({0, int(np.argmin(np.abs(times - T / 10))), int(np.argmin(np.abs(times - T / 3))), len(times) - 1})
    floor = 1e-6
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    traces = [C.trace(f"t = {_fmt(times[i], 2)}", x, keep(Pe[i]), color=k + 1, width=2.5 if i == picks[-1] else 1.5)
              for k, i in enumerate(picks)]
    traces.append(C.trace("Gaussian with equal variance (t = T)", x, keep(stats.norm.pdf(x, end["mean"], end["std"])),
                          color="muted", dash=True))
    out.chart(C.xy(
        "Density at selected times", traces,
        x=C.axis("x (standardised return)"), y=C.axis("Density", type="log", range=(floor, 3 * float(Pe.max()))),
        subtitle="Logarithmic scale: where the curve lies above the Gaussian parabola in the tails, the tails are fat.",
    ))

    # --- chart 3: tail mass over time, as a multiple of the Gaussian value
    r3, r4 = tails3 / G3, tails4 / G4
    out.chart(C.xy(
        "Fat tails over time: ratio to Gaussian",
        [C.trace("Mass beyond 3σ", times, np.where(r3 > 0, r3, np.nan), color=1),
         C.trace("Mass beyond 4σ", times, np.where(r4 > 1e-6, r4, np.nan), color=2),
         C.trace("Normal distribution = 1", [times[0], times[-1]], [1.0, 1.0], color="muted", dash=True)],
        x=C.axis("Time t"), y=C.axis("Multiple of the Gaussian value", type="log"),
        subtitle="Tail mass beyond k·σ (σ = current standard deviation of the density) divided by the Gaussian "
                 f"value ({_fmt(100 * G3)}% and {_fmt(100 * G4, 4)}% respectively). Above 1: fat tails.",
    ))

    # --- chart 4: what the operator does
    if frac:
        L = solver.build_fractional_operator(alpha, fp_params)
        i0 = n // 2
        k = np.arange(1, n - i0)
        rate = L[i0, i0 + k]
        dist = k * solver.dx
        ref = rate[4] * (dist / dist[4]) ** (-(1 + alpha))
        out.chart(C.xy(
            "Jump kernel of the fractional operator",
            [C.trace("Transition rate into a cell at distance d", dist, np.where(rate > 0, rate, np.nan), color=2),
             C.trace(f"Power law d^−(1+α) = d^−{_fmt(1 + alpha)}", dist, ref, color="muted", dash=True)],
            x=C.axis("Distance d", type="log"), y=C.axis("Rate per unit time", type="log"),
            subtitle="Ordinary diffusion only moves mass to the neighbouring cell. The fractional operator jumps "
                     "across the whole grid at a rate ∝ d^−(1+α) – hence the fat tails.",
        ))
        L_full = L + solver.build_operator_matrix(0, fp_params, include_diffusion=False).toarray()
    else:
        D = solver.diffusion_coefficient(x, 0, fp_params)
        out.chart(C.xy(
            "Diffusion D(x) – volatility clustering",
            [C.trace("D(x) = D₀(1 + λ|x|^β)", x, D, color=2)],
            x=C.axis("x (standardised return)"), y=C.axis("Diffusion D(x)"),
            hlines=[C.hline(params["D0"], "D₀")],
            subtitle="If D rises with |x|, mass in the tails flows outwards faster – fat tails without jumps.",
        ))
        L_full = solver.build_operator_matrix(0, fp_params).toarray()

    xs = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    mu_x = solver.drift_coefficient(xs, 0, fp_params)
    if frac:
        out.table("Drift and Lévy diffusion", ["x", "Drift μ(x)", "D_α (constant)"],
                  [[float(a), float(b) + 0.0, params["D_alpha"]] for a, b in zip(xs, mu_x)], formats=["num", "num", "num"])
    else:
        D_x = solver.diffusion_coefficient(xs, 0, fp_params)
        out.table("Drift and diffusion", ["x", "Drift μ(x)", "Diffusion D(x)"],
                  [[float(a), float(b) + 0.0, float(c)] for a, b, c in zip(xs, mu_x, D_x)], formats=["num", "num", "num"])

    mid = slice(n // 4, 3 * n // 4)
    balance = float(np.max(np.abs(L_full[:, mid].sum(axis=0)) / np.abs(np.diag(L_full)[mid])))
    edge_mass = float(np.sum(Pe[-1][np.abs(x) > 8]) * solver.dx)
    out.table("Numerics", ["Quantity", "Value"], [
        ["Grid", f"{n} points on [−10, 10], dx = {_fmt(float(solver.dx), 4)}"],
        ["Time steps", f"{n_steps:,} × dt = {_fmt(T / max(n_steps, 1), 4)}"],
        ["Mass balance: |column sum| ÷ |diagonal| (grid centre, max.)", _sci(balance)],
    ], formats=["text", "text"],
        note="A column sum close to 0 means the discrete operator conserves probability. For the fractional "
             "operator, the share of jumps that would lead beyond the edge of the grid is missing.")
    out.metric("Mass at the grid edge", edge_mass if edge_mass >= 1e-6 else "< 0.0001%",
               "pct" if edge_mass >= 1e-6 else "text", hint="share at |x| > 8 at t = T",
               status="warning" if edge_mass > 1e-3 else None)
    if edge_mass > 1e-3:
        out.note(f"{_fmt(100 * edge_mass, 1)}% of the mass lies at |x| > 8 – the tails are hitting the edge of the "
                 "grid. Variance and kurtosis then describe the truncated distribution.", "warning")
