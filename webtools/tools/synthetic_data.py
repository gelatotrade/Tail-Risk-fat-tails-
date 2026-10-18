"""Synthetic data: the framework's six return models side by side, the crisis scenario and the VIX proxy."""

import numpy as np
from scipy import stats

from src.utils.data_loader import (
    generate_crisis_scenario,
    generate_synthetic_returns,
    generate_vix_from_returns,
    load_sample_data,
)
from src.utils.statistics import compute_moments

from .. import charts as C

SPEC = {
    "id": "synthetic-data",
    "title": "Synthetic markets compared",
    "group": "data",
    "order": 60,
    "summary": "The framework's six return models side by side – Gaussian, Student-t, Lévy, GARCH, regime "
               "switching and Lévy jumps – plus the crisis scenario with a derived VIX.",
    "description": (
        "The framework generates artificial daily returns with `generate_synthetic_returns` in six models: the "
        "**normal distribution** (the benchmark), **Student-t** with ν = 4, **Lévy-stable** with α = 1.7 "
        "(infinite variance), **GARCH(1,1)** (volatility clustering), **regime switching** between calm and stress, "
        "and **Gaussian + Lévy jumps** (the most realistic). All use the same seed and the same length.\n\n"
        "How to read the charts: the price paths show how different paths can look at a similar daily "
        "volatility. The density is **robustly standardised** (median and MAD) so that all models have the same "
        "width in the centre – on the logarithmic axis you then see only the tails: the further a curve lies above "
        "the Gaussian curve, the more frequent crash days are. `generate_crisis_scenario` builds a market from a "
        "calm phase, a crisis (10% shock, 40% main crisis, 50% aftershocks) and a recovery; "
        "`generate_vix_from_returns` derives a VIX from it.\n\n"
        "You can use any model as the dataset for all other tools: select **Dataset → Change** at the top, then "
        "the **Synthetic** or **Crisis scenario** tab."
    ),
    "needs": [],
    "params": [
        {"name": "n", "type": "int", "default": 2520, "min": 250, "max": 10000, "step": 10,
         "label": "Length per model", "unit": "days", "help": "2,520 trading days ≈ 10 years."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 999999, "step": 1, "label": "Random seed"},
        {"name": "n_pre", "type": "int", "default": 500, "min": 50, "max": 2000, "step": 10,
         "label": "Crisis scenario: calm days before", "unit": "days"},
        {"name": "n_crisis", "type": "int", "default": 100, "min": 10, "max": 500, "step": 5,
         "label": "Crisis scenario: crisis days", "unit": "days"},
        {"name": "n_post", "type": "int", "default": 400, "min": 50, "max": 2000, "step": 10,
         "label": "Crisis scenario: days after", "unit": "days"},
    ],
    "covers": [
        "src.utils.data_loader.generate_synthetic_returns",
        "src.utils.data_loader.generate_crisis_scenario",
        "src.utils.data_loader.generate_vix_from_returns",
        "src.utils.data_loader.load_sample_data",
        "src.utils.statistics.compute_moments",
    ],
    "cost": "fast",
}
ORDER = ["gaussian", "student_t", "levy", "garch", "regime_switching", "levy_jump"]
SHORT = {"gaussian": "Normal distribution", "student_t": "Student-t (ν = 4)", "levy": "Lévy-stable (α = 1.7)",
         "garch": "GARCH(1,1)", "regime_switching": "Regime switching", "levy_jump": "Gaussian + Lévy jumps"}


def _max_dd(r):
    w = np.exp(np.cumsum(r))
    w = np.concatenate([[1.0], w])
    return float(np.min(w / np.maximum.accumulate(w) - 1))


def _row(name, r):
    m = compute_moments(r)
    sd = float(m["std"])
    return [name, sd * np.sqrt(252), float(m["skewness"]), float(m["kurtosis"]), float(np.min(r)), float(np.max(r)),
            _max_dd(r), int(np.sum(np.abs(r - m["mean"]) > 4 * sd))]


def run(params, data, out):
    n, seed = int(params["n"]), int(params["seed"])
    series = {m: np.asarray(generate_synthetic_returns(n, m, seed=seed), dtype=float) for m in ORDER}
    days = list(range(n))

    rows = [_row(SHORT[m], series[m]) for m in ORDER]
    kurt = {m: r[3] for m, r in zip(ORDER, rows)}
    fattest = max(kurt, key=kurt.get)
    out.metric("Models", len(ORDER), "int", hint=f"{n:,} trading days each, seed {seed}")
    out.metric("Fattest tails", SHORT[fattest], "text", hint=f"Excess kurtosis {kurt[fattest]:.1f}")
    out.metric("Gaussian: excess kurtosis", kurt["gaussian"], "num", digits=2, hint="Benchmark ≈ 0")
    out.metric("Worst day (all models)", min(r[4] for r in rows), "pct", digits=1)

    # ---- price paths --------------------------------------------------------------------------------
    paths = {m: 100 * np.exp(np.cumsum(series[m])) for m in ORDER}
    lo_p, hi_p = min(float(v.min()) for v in paths.values()), max(float(v.max()) for v in paths.values())
    out.chart(C.xy(
        "Price paths of the six models",
        [C.trace(SHORT[m], days, paths[m], color="muted" if m == "gaussian" else i, width=1.2)
         for i, m in enumerate(ORDER)],
        x=C.axis("Trading day", fmt="int"),
        y=C.axis("Price (start = 100)", type="log" if hi_p > 10 * lo_p else "linear"),
        subtitle="Same length, same seed. The Lévy and jump models show sudden jumps; GARCH and regime switching show "
                 "turbulent phases.",
    ))

    # ---- robustly standardised densities on a log axis ------------------------------------------------
    edges = (-12, 12)
    traces, floor = [], np.inf
    for i, m in enumerate(ORDER):
        r = series[m]
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        z = (r - np.median(r)) / scale
        centers, h = C.histogram(z, bins=97, density=True, range=edges)
        floor = min(floor, float(np.min(h[h > 0])))
        traces.append(C.trace(SHORT[m], centers, np.where(h > 0, h, np.nan), color="muted" if m == "gaussian" else i,
                              width=1.4, dash=m == "gaussian"))
    grid = np.linspace(*edges, 400)
    pdf = stats.norm.pdf(grid)
    traces.append(C.trace("Gaussian curve (theory)", grid, np.where(pdf >= floor / 4, pdf, np.nan), color="ink",
                          width=1, dash=True))
    out.chart(C.xy(
        "Distribution of daily returns – log scale",
        traces, x=C.axis("Return in robust standard deviations (median ± MAD)"),
        y=C.axis("Density", type="log", range=(floor / 4, 1.0)),
        subtitle="Equally wide in the centre, fundamentally different in the tails: fat tails lie far above the "
                 "Gaussian curve.",
    ))
    out.table("Model metrics",
              ["Model", "Volatility p.a.", "Skewness", "Excess kurtosis", "Worst day", "Best day",
               "Max. drawdown", "Days beyond 4σ"], rows,
              formats=["text", "pct", "num", "num", "pct", "pct", "pct", "int"],
              note=f"Over {n:,} days a normal distribution would produce on average "
                   f"{n * 2 * stats.norm.sf(4):.2f} days beyond 4σ.")

    # ---- crisis scenario + VIX proxy ------------------------------------------------------------------
    cr, meta = generate_crisis_scenario(n_pre=int(params["n_pre"]), n_crisis=int(params["n_crisis"]),
                                        n_post=int(params["n_post"]), seed=seed)
    cr = np.asarray(cr, dtype=float)
    a, b, peak = int(meta["crisis_start"]), int(meta["crisis_end"]), int(meta["crisis_peak"])
    cdays = list(range(len(cr)))
    vix = np.asarray(generate_vix_from_returns(cr), dtype=float)
    band = [C.band(a, b, "Crisis", "critical")]
    out.chart(C.xy(
        "Crisis scenario: price",
        [C.trace("Price (start = 100)", cdays, 100 * np.exp(np.cumsum(cr)), color=1, width=1.4)],
        x=C.axis("Trading day", fmt="int"), y=C.axis("Price"), bands=[C.band(a, b, "", "critical")],
        vlines=[C.vline(peak, "Crisis peak (metadata)")],
        subtitle=f"Shaded red: crisis. Calm ({a} days) → crisis ({b - a} days) → recovery ({len(cr) - b} days).",
    ))
    out.chart(C.xy(
        "Crisis scenario: derived VIX",
        [C.trace("VIX from returns", cdays, vix, color=2, width=1.4)],
        x=C.axis("Trading day", fmt="int"), y=C.axis("VIX points"), bands=band,
        subtitle="generate_vix_from_returns: 15 + 100 × rolling annualised volatility, with a jump on one-day "
                 "losses above 2%, limited to the range 10–80.",
    ))
    phases = [("Pre-crisis", cr[:a]), ("Crisis", cr[a:b]), ("Post-crisis", cr[b:])]
    prow = [[name, len(r), float(np.expm1(r.sum())) if r.size else None, float(np.std(r) * np.sqrt(252)) if r.size else None,
             float(np.min(r)) if r.size else None, _max_dd(r) if r.size else None] for name, r in phases]
    out.table("Crisis scenario phases", ["Phase", "Days", "Return", "Volatility p.a.", "Worst day",
                                         "Max. drawdown"], prow, formats=["text", "int", "pct", "pct", "pct", "pct"])
    if prow[1][2] is not None and prow[1][2] > 0:
        out.note(f"With seed {seed} the price rises during the crisis phase ({prow[1][2] * 100:+.0f}%): the main "
                 "crisis draws from a t-distribution with ν = 3, whose tails are fat on both sides. Try a different "
                 "seed for a genuine crash.", "warning")

    # ---- the framework's fixed sample bundle -----------------------------------------------------------
    sample = load_sample_data()
    sm = sample["crisis_metadata"]
    srows = []
    for key, model, seed_s in (("normal", "GARCH(1,1)", 42), ("crisis", "Crisis scenario", 43),
                               ("fat_tails", "Lévy-stable", 44), ("regime_switching", "Regime switching", 45)):
        r = np.asarray(sample[key], dtype=float)
        mom = compute_moments(r)
        hint = ""
        if key == "crisis":
            cret = float(np.expm1(r[sm["crisis_start"]:sm["crisis_end"]].sum()))
            hint = f"Crisis window (days {sm['crisis_start']}–{sm['crisis_end']}): {cret * 100:+.0f}%".replace("-", "−")
        srows.append([key, f"{model} · seed {seed_s}", len(r), float(mom["std"]) * np.sqrt(252), float(mom["kurtosis"]),
                      float(np.expm1(r.sum())), hint])
    for key, src in (("vix_normal", "from ‘normal’"), ("vix_crisis", "from ‘crisis’")):
        v = np.asarray(sample[key], dtype=float)
        srows.append([key, f"VIX {src}", len(v), None, None, None,
                      f"Mean {v.mean():.1f}, peak {v.max():.1f}"])
    sample_crisis = float(np.expm1(np.asarray(sample["crisis"])[sm["crisis_start"]:sm["crisis_end"]].sum()))
    if sample_crisis > 0:
        out.note(f"The sample series ‘crisis’ (seed 43) rises by {sample_crisis * 100:.0f}% within its crisis "
                 "window. The main crisis draws from a symmetric t-distribution (ν = 3) with a very large scale, "
                 "whose fluctuations swamp the negative drift – for roughly one seed in five the crisis turns into "
                 "a rally. Seed 43 is also the default for the **Crisis scenario** dataset; choose a different seed "
                 "there (e.g. 42) for a genuine crash.",
                 "warning")
    out.table("Framework sample data (load_sample_data)",
              ["Series", "Model", "Length", "Volatility p.a.", "Excess kurtosis", "Total return", "Note"], srows,
              formats=["text", "text", "int", "pct", "num", "pct", "text"],
              note="Fixed seeds 42–45: the same series that the framework's demos and tests use.")
    out.note("Use any model as the dataset for all other tools: select **Dataset → Change** at the top, then the "
             "**Synthetic** tab (any model, length and seed) or the **Crisis scenario** tab.", "success")
