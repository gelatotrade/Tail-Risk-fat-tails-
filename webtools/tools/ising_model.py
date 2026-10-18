"""Ising market model: herding, model price and the order-disorder transition."""

import numpy as np
from scipy import stats

from src.physics.phase_transitions import IsingMarketModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "ising-model",
    "title": "Ising model: trader herding",
    "group": "regime",
    "order": 60,
    "summary": "Shows how trends, price surges and a transition from order to disorder emerge from many "
               "buy and sell decisions driven by herding.",
    "description": (
        "Each trader is a spin: **+1 buys, −1 sells**. The coupling **J** rewards doing the same as both "
        "neighbours (herding), the field **h** is market-wide sentiment and the temperature **T** is noise – "
        "private information and chance. The Metropolis algorithm lets the traders change their minds one after "
        "another. The magnetisation **M** (share of buyers minus share of sellers) drives the model price: at "
        "each recorded step it changes by M × 1%.\n\n"
        "How to read the charts: in the space-time picture, herds appear as single-coloured stripes. The smaller "
        "T is relative to J, the larger the herds (correlation length ξ) and the longer a trend persists in M "
        "and in the price. The phase diagram shows ⟨|M|⟩ and the susceptibility χ against temperature – both "
        "simulated (`phase_diagram`) and as the exact solution for the chain (transfer matrix).\n\n"
        "Limitations: the traders sit on a **one-dimensional** ring. The critical temperature shown, "
        "T_c = 2J/ln(1+√2) ≈ 2.27·J, is Onsager's value for the **2D** lattice and serves only as a reference "
        "scale. A 1D chain has no true phase transition at T > 0, only a smooth crossover from order to "
        "disorder. At very low T the herds in the simulation also freeze (100 equilibration sweeps are not "
        "enough) – the simulation then lies below the exact curve."
    ),
    "needs": [],
    "params": [
        {"name": "n_agents", "type": "int", "default": 80, "min": 20, "max": 400, "step": 10,
         "label": "Number of traders N"},
        {"name": "J", "type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
         "label": "Herding J", "help": "Coupling to the two neighbours. 0 = everyone decides independently."},
        {"name": "h", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "Market sentiment h", "help": "> 0 buying pressure, < 0 selling pressure on all traders."},
        {"name": "T", "type": "float", "default": 0.8, "min": 0.1, "max": 6.0, "step": 0.05,
         "label": "Temperature T (noise)", "help": "Low = traders follow their neighbours, high = chance dominates."},
        {"name": "n_steps", "type": "int", "default": 400, "min": 50, "max": 4000, "step": 50,
         "label": "Monte Carlo sweeps", "help": "One sweep = N individual decisions."},
        {"name": "record_interval", "type": "int", "default": 4, "min": 1, "max": 50, "step": 1,
         "label": "Record every … sweeps", "help": "Each recorded step is one trading day of the model price."},
        {"name": "phase", "type": "bool", "default": True, "label": "Compute phase diagram",
         "help": "Takes most of the computation time: 100 equilibration sweeps per temperature plus the "
                 "measurements."},
        {"name": "T_min", "type": "float", "default": 0.8, "min": 0.1, "max": 5.0, "step": 0.1,
         "label": "Phase diagram: T from", "show_if": {"phase": [True]}},
        {"name": "T_max", "type": "float", "default": 4.0, "min": 0.5, "max": 8.0, "step": 0.1,
         "label": "Phase diagram: T to", "show_if": {"phase": [True]}},
        {"name": "n_T", "type": "int", "default": 8, "min": 3, "max": 25, "step": 1,
         "label": "Phase diagram: temperatures", "show_if": {"phase": [True]}},
        {"name": "n_samples", "type": "int", "default": 50, "min": 10, "max": 400, "step": 10,
         "label": "Measurements per temperature", "help": "Sweeps over which ⟨|M|⟩ and χ are averaged.",
         "show_if": {"phase": [True]}},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.phase_transitions.IsingMarketModel",
        "src.physics.phase_transitions.IsingMarketModel.magnetization",
        "src.physics.phase_transitions.IsingMarketModel.energy",
        "src.physics.phase_transitions.IsingMarketModel.susceptibility",
        "src.physics.phase_transitions.IsingMarketModel.monte_carlo_step",
        "src.physics.phase_transitions.IsingMarketModel.simulate",
        "src.physics.phase_transitions.IsingMarketModel.phase_diagram",
    ],
    "cost": "medium",
}

N_VIEW = 60               # sweeps shown in the space-time picture
N_ROWS = 50               # agents shown in the space-time picture (bigger cells)
EQUILIBRATION = 100       # sweeps phase_diagram spends per temperature before measuring
MAX_FLIPS = 1_500_000     # spin-flip attempts (~15 s native, pure-Python Metropolis loop)


def _exact_ring(n, J, h, T):
    """Exact Boltzmann distribution of the number of buyers on a periodic 1D ring.

    Tr T(z)^n of the transfer matrix with a counting variable z is a polynomial whose coefficients are the
    weights of 0..n buyers; it is evaluated on the unit circle and inverted with an FFT.
    Returns (<|M|>, chi) with chi = N (<M^2> - <|M|>^2) / T, the same estimator phase_diagram uses.
    """
    b = 1.0 / T
    u = np.arange(n + 1)
    w = np.exp(1j * np.pi * u / (n + 1))  # w^2 = z_k; each spin's z-weight is split over its two bonds
    tm = np.empty((n + 1, 2, 2), dtype=complex)
    tm[:, 0, 0] = np.exp(b * J + b * h) * w * w
    tm[:, 0, 1] = tm[:, 1, 0] = np.exp(-b * J) * w
    tm[:, 1, 1] = np.exp(b * J - b * h)
    tm /= np.max(np.abs(np.linalg.eigvals(tm[0].real)))
    trace = np.sum(np.linalg.eigvals(tm) ** n, axis=1)
    p = np.clip(np.fft.fft(trace).real, 0.0, None)
    p /= p.sum()
    m = np.abs(2 * u - n) / n
    mean_abs = float(np.sum(p * m))
    return mean_abs, n * (float(np.sum(p * m * m)) - mean_abs ** 2) / T


def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma."""
    return f"{x:,.{digits}f}"


def _correlation_length(J, h, T):
    """xi = 1 / ln(lambda+ / lambda-) of the 1D transfer matrix (in agents)."""
    b = 1.0 / T
    root = np.sqrt(np.exp(2 * b * J) * np.sinh(b * h) ** 2 + np.exp(-2 * b * J))
    lp, lm = np.exp(b * J) * np.cosh(b * h) + root, np.exp(b * J) * np.cosh(b * h) - root
    ratio = lp / lm if lm > 0 else np.inf
    return 1.0 / np.log(ratio) if ratio > 1 else np.inf


def run(params, data, out):
    n, J, h, T = params["n_agents"], params["J"], params["h"], params["T"]
    n_steps, every = params["n_steps"], params["record_interval"]
    t_lo, t_hi, n_T = params["T_min"], params["T_max"], params["n_T"]
    if params["phase"] and t_lo >= t_hi:
        raise ToolError("For the phase diagram, ‘T from’ must be smaller than ‘T to’.")
    flips = n * (n_steps + N_VIEW + (n_T * (EQUILIBRATION + params["n_samples"]) if params["phase"] else 0))
    if flips > MAX_FLIPS:
        raise ToolError(f"That would be {_num(flips / 1e6, 1)} million individual decisions – too many for the "
                        f"browser (limit {_num(MAX_FLIPS / 1e6, 1)} million). Reduce the number of traders, sweeps, "
                        "temperatures or measurements, or switch off the phase diagram.")

    np.random.seed(params["seed"])
    model = IsingMarketModel(n_agents=n, J=J, h=h, T=T)
    out.progress("Running Metropolis simulation …", 0.05)
    sim = model.simulate(n_steps=n_steps, record_interval=every)
    M, E, ret, price = sim["magnetization"], sim["energy"], sim["returns"], sim["prices"]
    sweeps = np.arange(0, n_steps, every) + 1  # the record is taken after this many sweeps

    # Space-time picture: continue the same chain and keep every configuration
    history = np.empty((N_VIEW, n))
    for k in range(N_VIEW):
        model.monte_carlo_step()
        history[k] = model.spins
    m_now, e_now, chi_now = model.magnetization(), model.energy(), model.susceptibility()

    xi = _correlation_length(J, h, T)
    exact_abs, exact_chi = _exact_ring(n, J, h, T)
    tc = float(model.T_c)
    out.metric("Magnetisation M (end)", m_now, "num", hint="+1: everyone buys · −1: everyone sells")
    out.metric("Mean |M| over the run", np.mean(np.abs(M)), "num",
               hint=f"exact equilibrium solution for the chain: {_num(exact_abs)}")
    out.metric("Correlation length ξ", xi, "num", digits=1, unit="traders",
               hint="typical herd size (exact for the 1D chain)")
    out.metric("Energy per trader", e_now / n, "num",
               hint=f"−{_num(J + abs(h))}: complete order · ≈ 0: chaos" if J + abs(h) > 0 else
               "without herding or sentiment (J = h = 0) the energy is always 0")
    out.metric("Susceptibility χ", exact_chi, "num",
               hint=f"exact for the chain · snapshot without herding (1−M²)/T: {_num(chi_now)}")
    if tc > 0:
        out.metric("T / T_c (2D reference)", T / tc, "num",
                   hint="scale only: the 1D chain has no true phase transition")
    else:
        out.metric("T / T_c (2D reference)", "–", "text", hint="J = 0: without herding there is no T_c")
    out.metric("Model price volatility", np.std(ret), "pct",
               hint="standard deviation of the log return per recorded step")
    out.metric("Total price change", price[-1] / 100.0 - 1.0, "pct", digits=1)
    if len(ret) > 3 and np.std(ret) > 0:
        out.metric("Excess kurtosis of model returns", stats.kurtosis(ret), "num", digits=1,
                   hint="0 = normally distributed")

    out.chart(C.xy(
        "Magnetisation M – net buying pressure",
        [C.trace("M", sweeps, M, color=1, width=1.5)],
        x=C.axis("Monte Carlo sweep", fmt="int"), y=C.axis("M", range=(-1.05, 1.05)),
        hlines=[C.hline(0.0, "balanced")],
        subtitle="Long excursions away from 0 are herding phases: many traders act alike.",
    ))
    out.chart(C.xy(
        "Model price from the magnetisation",
        [C.trace("Price", sweeps, price, color=2)],
        x=C.axis("Monte Carlo sweep", fmt="int"), y=C.axis("Price (start 100)"),
        hlines=[C.hline(100.0, "Start")],
        subtitle="At each recorded step the price changes by M × 1% – herds create trends.",
    ))
    rows = min(n, N_ROWS)
    out.chart(C.heatmap(
        f"Space-time picture: the next {N_VIEW} sweeps" + (f" (traders 1–{rows} of {n})" if rows < n else ""),
        list(range(n_steps + 1, n_steps + N_VIEW + 1)), list(range(1, rows + 1)), history[:, :rows].T,
        x=C.axis("Monte Carlo sweep", fmt="int"), y=C.axis("Trader", fmt="int"),
        z=C.axis("Position (+1 buys, −1 sells)", range=(-1, 1)), scale="diverging", height=400, gap=0,
        subtitle="Each row is a trader: red buys, blue sells. Blocks of one colour are herds – their height ≈ ξ, "
                 "their lifetime grows with ξ².",
    ))

    if not params["phase"]:
        out.note("Phase diagram switched off – switch it on to see ⟨|M|⟩ and χ against temperature.")
        return

    out.progress("Phase diagram: stepping through the temperatures …", 0.4)
    temps = np.linspace(t_lo, t_hi, n_T)
    pd = model.phase_diagram(temps, n_samples=params["n_samples"])
    grid = np.linspace(t_lo, t_hi, 60)
    exact = np.array([_exact_ring(n, J, h, t) for t in grid])
    exact_at = np.array([_exact_ring(n, J, h, t) for t in pd["temperature"]])

    def markers(top):
        """Vertical reference lines as legend entries (your T, the 2D reference T_c)."""
        out_ = []
        if t_lo <= T <= t_hi:
            out_.append(C.trace("Your T", [T, T], [0, top], color=2, dash=True, width=1.5, decimate_to=None))
        if t_lo <= tc <= t_hi:
            out_.append(C.trace("T_c (2D reference)", [tc, tc], [0, top], color=3, dash=True, width=1.5,
                                decimate_to=None))
        return out_

    out.chart(C.xy(
        "Phase diagram: order ⟨|M|⟩ against temperature",
        [C.trace("Exact (1D ring)", grid, exact[:, 0], color="muted", width=2.5),
         C.trace("Simulation", pd["temperature"], pd["magnetization"], type="scatter", color=1, size=9)]
        + markers(1.0),
        x=C.axis("Temperature T"), y=C.axis("⟨|M|⟩", range=(0, 1.02)),
        subtitle="A smooth crossover rather than a jump at T_c. If the simulation lies below the curve, the herds "
                 "are frozen.",
    ))
    chi_top = float(np.nanmax(np.concatenate([exact[:, 1], pd["susceptibility"]]))) * 1.05
    out.chart(C.xy(
        "Susceptibility χ against temperature",
        [C.trace("Exact (1D ring)", grid, exact[:, 1], color="muted", width=2.5),
         C.trace("Simulation", pd["temperature"], pd["susceptibility"], type="scatter", color=1, size=9)]
        + markers(chi_top),
        x=C.axis("Temperature T"), y=C.axis("χ = N·Var(|M|)/T", range=(0, chi_top)),
        subtitle="Where χ is large, sentiment flips even after small nudges. The peak is a finite-size effect and "
                 "moves towards T → 0 as N grows.",
    ))
    rows = [[float(t), float(t / tc) if tc > 0 else None, float(m), float(ex[0]), float(c), float(ex[1])]
            for t, m, c, ex in zip(pd["temperature"], pd["magnetization"], pd["susceptibility"], exact_at)]
    out.table("Phase diagram", ["T", "T / T_c (2D)", "⟨|M|⟩ simulation", "⟨|M|⟩ exact", "χ simulation", "χ exact"],
              rows, formats=["num", "num", "num", "num", "num", "num"],
              note=f"Per temperature, {EQUILIBRATION} equilibration sweeps from a random start, then "
                   f"{params['n_samples']} measurements. Exact = Boltzmann equilibrium of the ring with {n} traders.")
    worst = int(np.argmax(exact_at[:, 0] - pd["magnetization"]))
    gap = exact_at[worst, 0] - pd["magnetization"][worst]
    if gap > 0.15:
        out.note(f"At T = {_num(pd['temperature'][worst])} the simulation lies {_num(gap)} below the exact "
                 f"solution: the herds grow so slowly there that {EQUILIBRATION} equilibration sweeps are not "
                 "enough.", "warning")
