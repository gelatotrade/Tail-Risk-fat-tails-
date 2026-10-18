"""Chaos or chance? Lyapunov exponent and recurrence analysis, honestly tested against shuffled returns."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.visualization.phase_space import LyapunovExponentEstimator, RecurrencePlotAnalyzer

from .. import charts as C
from ..data import fmt_date
from ..core import ToolError

SPEC = {
    "id": "chaos",
    "title": "Chaos test: Lyapunov and recurrence",
    "group": "regime",
    "order": 70,
    "summary": "Estimates the Lyapunov exponent and recurrence structure of your returns and tests them against "
               "randomly shuffled returns to tell genuine chaos from mere noise.",
    "description": (
        "A chaotic system is deterministic but sensitive: two almost identical states diverge exponentially, at "
        "the rate **λ** (largest Lyapunov exponent). `LyapunovExponentEstimator` reconstructs a phase space from "
        "the return series for this (**delay embedding**: vectors of m returns spaced τ apart), finds the "
        "nearest neighbour of random starting points and measures how quickly the two separate. "
        "`classify_dynamics` calls λ > 0.1 ‘chaotic’. The **recurrence plot** (`RecurrencePlotAnalyzer`) marks "
        "every pair of days whose embedded states lie closer than ε; recurrence quantification analysis (RQA) "
        "derives the recurrence rate (RR), determinism (DET, share on diagonal lines) and laminarity (LAM, share "
        "on vertical lines) from it.\n\n"
        "**The honest test:** shuffle the returns randomly. This destroys any temporal order while the "
        "distribution stays the same. Genuine dynamics must differ from the surrogate. For daily returns, however, "
        "λ is just as positive as for the surrogate and for Gaussian noise – here the estimator measures the "
        "dispersion of the noise, not chaos. The control with the **logistic map** (genuine chaos, "
        "λ = ln 2 ≈ 0.69) shows the difference: its λ stays the same for every m, whereas for noise it falls "
        "with m.\n\n"
        "What the recurrence plot really shows for returns is **volatility clustering**: calm weeks form dark "
        "blocks, stress phases bright bands. If DET or LAM lies above all surrogates, that is temporal structure "
        "(volatility clusters) – but no evidence that the direction of prices is predictable."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "embedding_dim", "type": "int", "default": 3, "min": 2, "max": 8, "step": 1,
         "label": "Embedding dimension m", "help": "How many consecutive returns form one state."},
        {"name": "tau", "type": "int", "default": 1, "min": 1, "max": 10, "step": 1, "label": "Delay τ",
         "help": "Spacing of the returns in the state vector, in steps of the series (days or weeks)."},
        {"name": "rp_scale", "type": "select", "default": "week", "label": "Recurrence plot time scale",
         "options": [{"value": "week", "label": "Weekly returns (5 trading days each)"},
                     {"value": "day", "label": "Daily returns"}],
         "help": "Weeks cover several years and so show calm and crisis periods in the same picture."},
        {"name": "rp_points", "type": "int", "default": 400, "min": 100, "max": 400, "step": 50,
         "label": "Points in the recurrence plot",
         "help": "The last N weeks or days. Computation time and memory grow with N²."},
        {"name": "recurrence_rate", "type": "int", "default": 10, "min": 2, "max": 30, "step": 1,
         "label": "Recurrence rate (sets ε)", "unit": "%",
         "help": "ε = this quantile of all distances. 10% is the original's default."},
        {"name": "n_surrogates", "type": "int", "default": 19, "min": 5, "max": 49, "step": 1,
         "label": "Surrogates (shuffled returns)",
         "help": "19 surrogates: if the actual value lies above all of them, it is significant at the 5% level."},
        {"name": "max_iter", "type": "int", "default": 200, "min": 50, "max": 1000, "step": 50,
         "label": "Starting points for λ", "help": "More starting points = smoother estimate, longer computation time."},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 10000, "step": 250,
         "label": "Period for λ (most recent trading days)", "unit": "days",
         "help": "Most recent trading days used to estimate the Lyapunov exponent."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed",
         "help": "Controls starting points, surrogates and control series."},
    ],
    "covers": [
        "src.visualization.phase_space.LyapunovExponentEstimator",
        "src.visualization.phase_space.LyapunovExponentEstimator.estimate_largest_lyapunov",
        "src.visualization.phase_space.LyapunovExponentEstimator.classify_dynamics",
        "src.visualization.phase_space.RecurrencePlotAnalyzer",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.compute_recurrence_matrix",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.recurrence_quantification",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.plot_recurrence",
    ],
    "cost": "medium",
}

CLASS_LABELS = {"CHAOTIC": "Chaotic", "WEAKLY_CHAOTIC": "Weakly chaotic", "QUASI_PERIODIC": "Quasi-periodic",
                "STABLE": "Stable", "UNKNOWN": "Unknown"}
M_CURVE = (2, 3, 4, 5, 6, 7)   # embedding dimensions of the sensitivity curve
CURVE_SURR = 3                 # surrogates averaged per point of that curve
HEAT_CELLS = 60                # heatmap cells per axis (blocks of consecutive points)
N_NEIGHBORS = 5                # default of the original

def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma, typographic minus."""
    return f"{x:,.{digits}f}".replace("-", "−")




def _logistic(n, x0):
    """Logistic map x -> 4x(1-x): fully chaotic, largest Lyapunov exponent ln 2."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    return x


def _lyap(series, m, tau, iters, seed):
    return float(LyapunovExponentEstimator(series, embedding_dim=m, tau=tau).estimate_largest_lyapunov(
        n_neighbors=N_NEIGHBORS, max_iter=iters, seed=seed))


def _epsilon(rpa, q):
    """The threshold the original uses by default (a quantile of all pairwise distances), for any quantile q."""
    return float(np.percentile(squareform(pdist(rpa.embedded)), 100.0 * q))


def _blocks(R, cells):
    """Average the 0/1 recurrence matrix over b x b blocks of days (b = ceil(N / cells))."""
    n = R.shape[0]
    b = max(1, int(np.ceil(n / cells)))
    nb = int(np.ceil(n / b))
    pad = nb * b - n
    Rp = np.pad(R.astype(float), ((0, pad), (0, pad)), constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        D = np.nanmean(Rp.reshape(nb, b, nb, b), axis=(1, 3))
    return D, b


def _rank_text(value, sur):
    sur = np.asarray(sur, dtype=float)
    sur = sur[np.isfinite(sur)]
    if not np.isfinite(value) or not sur.size:
        return "–"
    if value > sur.max():
        return f"above all {sur.size} surrogates"
    if value < sur.min():
        return f"below all {sur.size} surrogates"
    above = int(np.sum(sur < value))
    return f"within range (rank {above + 1} of {sur.size + 1})"


def _rp_series(data, scale, points):
    """The series of the recurrence plot: the last `points` daily or weekly (5-day) log returns, with dates."""
    r = np.asarray(data.returns, dtype=float)
    step = 5 if scale == "week" else 1
    k = int(min(points, len(r) // step))
    tail = r[len(r) - k * step:]
    x = tail.reshape(k, step).sum(axis=1)
    pos = np.arange(len(r) - k * step + step - 1, len(r), step)      # last day of every block
    labels = [data.dates[i] for i in pos] if data.dates else [int(i) + 1 for i in pos]
    return x, labels


def run(params, data, out):
    m, tau = int(params["embedding_dim"]), int(params["tau"])
    seed, n_sur, iters = int(params["seed"]), int(params["n_surrogates"]), int(params["max_iter"])
    q = float(params["recurrence_rate"]) / 100.0
    weekly = params["rp_scale"] == "week"
    n = int(min(params["last_n"], data.n))
    span = (max(M_CURVE[-1], m) - 1) * tau
    need = span + 40                   # the estimator follows neighbours 9 steps ahead from > 10 start vectors
    if n < need:
        raise ToolError(f"For the λ curve up to m = {max(M_CURVE[-1], m)} with τ = {tau} you need at least {need:,} "
                        f"daily returns (available: {n:,}). "
                        + ("Reduce τ." if tau > 1 else "Choose a longer dataset."))
    r = np.asarray(data.returns[-n:], dtype=float)
    rng = np.random.default_rng(seed)

    # series of the recurrence plot; too few weeks -> fall back to daily returns
    x, x_lab = _rp_series(data, params["rp_scale"], int(params["rp_points"]))
    fallback = weekly and len(x) - (m - 1) * tau < 60
    if fallback:
        weekly = False
        x, x_lab = _rp_series(data, "day", int(params["rp_points"]))
    unit, units = ("week", "weeks") if weekly else ("day", "days")
    if len(x) - (m - 1) * tau < 30:
        raise ToolError(f"Too few points in the recurrence plot ({len(x)} {units}) for m = {m} and τ = {tau}. "
                        "Increase the number of points or reduce m or τ.")

    # --- Lyapunov: real series, shuffled surrogates (same seed for the start points) ----------
    out.progress("Lyapunov exponent of the returns …", 0.05)
    est = LyapunovExponentEstimator(r, embedding_dim=m, tau=tau)
    cls = est.classify_dynamics(n_neighbors=N_NEIGHBORS, max_iter=iters, seed=seed)
    lam = float(cls["lyapunov"])
    out.progress(f"Lyapunov exponent of {n_sur} shuffled series …", 0.15)
    surrogates = [rng.permutation(r) for _ in range(n_sur)]
    lam_sur = np.array([_lyap(s, m, tau, iters, seed) for s in surrogates])

    # --- sensitivity to m: returns, shuffled, Gaussian noise, chaotic logistic map ------------
    out.progress("λ for m = 2 … 7 and control series …", 0.4)
    gauss = rng.normal(0.0, float(np.std(r)), n)
    logistic = _logistic(n, 0.1 + 0.8 * float(rng.random()))
    curves = {"r": [], "s": [], "g": [], "l": []}
    for mm in M_CURVE:
        curves["r"].append(lam if mm == m else _lyap(r, mm, tau, iters, seed))
        curves["s"].append(float(np.mean([lam_sur[i] if mm == m else _lyap(surrogates[i], mm, tau, iters, seed)
                                          for i in range(min(CURVE_SURR, n_sur))])))
        curves["g"].append(_lyap(gauss, mm, tau, iters, seed))
        curves["l"].append(_lyap(logistic, mm, tau, iters, seed))
    lam_logistic = curves["l"][M_CURVE.index(m)] if m in M_CURVE else _lyap(logistic, m, tau, iters, seed)

    # --- recurrence analysis on the last N weeks / days, real vs shuffled ---------------------
    out.progress("Recurrence plot and RQA …", 0.7)
    rpa = RecurrencePlotAnalyzer(x, embedding_dim=m, tau=tau)
    eps = _epsilon(rpa, q)
    R = rpa.compute_recurrence_matrix(epsilon=eps)
    rqa = rpa.recurrence_quantification(epsilon=eps)
    sur_rqa, R_sur = [], None
    for i in range(n_sur):
        a = RecurrencePlotAnalyzer(rng.permutation(x), embedding_dim=m, tau=tau)
        e = _epsilon(a, q)
        sur_rqa.append(a.recurrence_quantification(epsilon=e))
        if R_sur is None:
            R_sur = a.compute_recurrence_matrix(epsilon=e)
    det_s = np.array([d["determinism"] for d in sur_rqa], dtype=float)
    lam_s = np.array([d["laminarity"] for d in sur_rqa], dtype=float)
    stab_s = np.array([d["stability_index"] for d in sur_rqa], dtype=float)
    det, lamr = float(rqa["determinism"]), float(rqa["laminarity"])

    # --- metrics ------------------------------------------------------------------------
    cls_name = CLASS_LABELS.get(cls["classification"], cls["classification"])
    lam_lo, lam_hi = float(np.nanmin(lam_sur)), float(np.nanmax(lam_sur))
    out.metric("Lyapunov exponent λ", lam, "num", digits=3,
               hint=f"daily returns · m = {m}, τ = {tau} · {iters:,} starting points · {n:,} days")
    out.metric("Original classification", cls_name, "text",
               hint="classify_dynamics: λ > 0.1 ⇒ chaotic – but this also holds for pure noise")
    out.metric("λ of shuffled returns", float(np.nanmean(lam_sur)), "num", digits=3,
               hint=f"avg. of {n_sur} surrogates · range {_num(lam_lo, 3)} to {_num(lam_hi, 3)}")
    out.metric("λ of a known chaotic series", lam_logistic, "num", digits=3,
               hint="logistic map, theoretically ln 2 ≈ 0.693")
    out.metric("Determinism DET", det, "pct", digits=1,
               hint=f"{len(x)} {units} · surrogates avg. {_num(100 * np.mean(det_s), 1)}% · {_rank_text(det, det_s)}")
    out.metric("Laminarity LAM", lamr, "pct", digits=1,
               hint=f"surrogates avg. {_num(100 * np.mean(lam_s), 1)}% · {_rank_text(lamr, lam_s)}")
    out.metric("Threshold ε", eps / float(np.std(x)), "num", digits=2,
               hint=f"in σ of the {'weekly' if weekly else 'daily'} returns · recurrence rate "
                    f"{_num(100 * float(rqa['recurrence_rate']), 1)}%")

    # --- charts 1 + 2: recurrence plots, real and shuffled -----------------------------
    D, b = _blocks(R, HEAT_CELLS)
    D_sur, _ = _blocks(R_sur, HEAT_CELLS)
    centers = [min(len(R) - 1, i * b + b // 2) for i in range(D.shape[0])]
    ticks = [x_lab[c] for c in centers]
    hx = C.axis("", type="date") if data.dates else C.axis("Trading day", fmt="int")
    top = float(max(np.nanmax(D), np.nanmax(D_sur)))
    z_ax = C.axis("Recurrence density", fmt="pct", digits=0, range=(0.0, top))   # same colour range for both plots
    dat = units
    lead = (f"Each cell aggregates {b} × {b} pairs of {dat}: the share of pairs whose states" if b > 1 else
            f"Each cell is a pair of {dat}: 100% if their states")
    period = (f"{fmt_date(x_lab[0])} – {fmt_date(x_lab[-1])}" if data.dates else f"last {len(x)} {units}")
    scale_txt = "weekly returns" if weekly else "daily returns"
    out.chart(C.heatmap(
        f"Recurrence plot of {scale_txt} ({period})", ticks, ticks, np.round(D, 3), x=hx, y=hx, z=z_ax,
        scale="sequential", height=560,
        subtitle=f"{lead} (m = {m}, τ = {tau}) lie closer than ε. Dark blocks = calm phases that resemble one "
                 "another; bright crosses = crises whose states never recur. Time runs to the right and "
                 "downwards.",
    ))
    out.chart(C.heatmap(
        f"For comparison: the same {scale_txt}, randomly shuffled", ticks, ticks, np.round(D_sur, 3), x=hx, y=hx,
        z=z_ax, scale="sequential", height=560,
        subtitle=f"Same distribution but no temporal order (seed {seed}). The densest cell reaches "
                 f"{_num(100 * float(np.nanmax(D_sur)), 0)}% here, compared with "
                 f"{_num(100 * float(np.nanmax(D)), 0)}% for the actual returns. "
                 "Any additional blocks and crosses visible above are temporal structure – for returns, "
                 "volatility clustering.",
    ))
    out.chart(C.xy(
        f"{scale_txt.capitalize()} in the recurrence plot window",
        [C.trace("Weekly return" if weekly else "Daily return", x_lab, x, type="bar", color=1)],
        x=hx, y=C.axis("Return (log)", fmt="pct", digits=1), height=260,
        subtitle="Large swings appear in the recurrence plot as bright crosses: their states never recur.",
    ))

    # --- chart 3: lambda versus embedding dimension -------------------------------------
    ms = list(M_CURVE)
    out.chart(C.xy(
        "λ depends on m – except for genuine chaos",
        [C.trace("Returns", ms, curves["r"], color=1, width=2.5),
         C.trace(f"Shuffled returns (avg. of {min(CURVE_SURR, n_sur)})", ms, curves["s"], color=2, dash=True),
         C.trace("Gaussian noise (same σ)", ms, curves["g"], color="muted", dash=True),
         C.trace("Logistic map (chaos)", ms, curves["l"], color=4, width=2.5)],
        x=C.axis("Embedding dimension m", fmt="int"), y=C.axis("Estimated λ"), height=360,
        hlines=[C.hline(np.log(2.0), "ln 2 = true λ of the logistic map"),
                C.hline(0.1, "original's ‘chaotic’ threshold")],
        subtitle="Genuine chaos has a λ that does not depend on the embedding (plateau at ln 2). For returns, "
                 "the surrogate and noise, λ falls with m – an artefact of the estimator, not chaos.",
    ))

    # --- chart 4: delay embedding of the daily series --------------------------------------
    emb = np.asarray(est.embedded, dtype=float)
    dl = list(data.dates[-n:]) if data.dates else None
    lab = [fmt_date(d) for d in dl[:len(emb)]] if dl else None
    if m >= 3:
        out.chart(C.scatter3d(
            f"Reconstructed phase space of daily returns (delay embedding, {len(emb):,} states)",
            [C.path3d("State (r(t), r(t+τ), r(t+2τ))", emb[:, 0], emb[:, 1], emb[:, 2], color=1, mode="markers",
                      size=2, text=lab)],
            x=C.axis("r(t)", fmt="pct", digits=0), y=C.axis("r(t+τ)", fmt="pct", digits=0),
            z=C.axis("r(t+2τ)", fmt="pct", digits=0), height=500,
            subtitle="A chaotic attractor would lie on a surface or a curve. Returns form a cloud: a dense core "
                     "(calm days) and isolated distant points, mostly close to an axis – a shock day appears in m "
                     "states, each time in a different coordinate. Noise with fat tails.",
        ))
    else:
        out.chart(C.xy(
            f"Reconstructed phase space of daily returns (delay embedding, {len(emb):,} states)",
            [C.trace("State (r(t), r(t+τ))", emb[:, 0], emb[:, 1], type="scatter", color=1, size=3,
                     decimate_to=None, text=lab)],
            x=C.axis("r(t)", fmt="pct", digits=0), y=C.axis("r(t+τ)", fmt="pct", digits=0), height=400,
            subtitle="A chaotic system would lie on a curve (the logistic map on a parabola). Returns form a cloud "
                     "with streaks along the axes.",
        ))

    # --- tables ---------------------------------------------------------------------------
    rows = [
        ["Lyapunov exponent λ (days)", "num", lam, float(np.nanmean(lam_sur)),
         f"{_num(lam_lo, 3)} … {_num(lam_hi, 3)}", _rank_text(lam, lam_sur)],
        [f"Determinism DET ({units})", "pct", det, float(np.mean(det_s)),
         f"{_num(100 * det_s.min(), 1)} … {_num(100 * det_s.max(), 1)}%", _rank_text(det, det_s)],
        [f"Laminarity LAM ({units})", "pct", lamr, float(np.mean(lam_s)),
         f"{_num(100 * lam_s.min(), 1)} … {_num(100 * lam_s.max(), 1)}%", _rank_text(lamr, lam_s)],
        ["Stability index DET/RR", "num", float(rqa["stability_index"]), float(np.mean(stab_s)),
         f"{_num(stab_s.min())} … {_num(stab_s.max())}", _rank_text(float(rqa["stability_index"]), stab_s)],
    ]
    out.table("Surrogate test: actual vs randomly shuffled returns",
              ["Metric", "Returns", "Surrogates avg.", "Surrogates range", "Assessment"],
              [[row[0], _fmt(row[2], row[1]), _fmt(row[3], row[1]), row[4], row[5]] for row in rows],
              formats=["text", "text", "text", "text", "text"],
              note=f"λ from {n:,} daily returns, RQA from the last {len(x)} {scale_txt} "
                   f"(N = {int(rqa['n_points']):,} states, recurrence rate fixed at {_num(100 * q, 0)}%, lines of "
                   f"3 points or more, main diagonal excluded). Surrogates: {n_sur} random permutations "
                   f"(seed {seed}). If a value lies above all surrogates, the series contains temporal structure.")
    out.table("λ by embedding dimension", ["m", "Returns", "Shuffled", "Gaussian noise", "Logistic map"],
              [[mm] + [_num(curves[k][i], 3) for k in ("r", "s", "g", "l")] for i, mm in enumerate(ms)],
              formats=["int", "text", "text", "text", "text"],
              note="The original's Rosenstein-type estimator: average of log(dₜ/d₀)/t for t = 1 … 9 steps, nearest "
                   "neighbour at least 6 steps away.")

    # --- the honest reading --------------------------------------------------------------
    if fallback:
        out.note(f"The series is too short for weekly returns – the recurrence plot uses the last {len(x)} "
                 "daily returns.", "warning")
    if lam <= lam_hi:
        out.note(f"**No evidence of deterministic chaos.** λ = {_num(lam, 3)} lies "
                 + ("within the range of the" if lam >= lam_lo else "even below all")
                 + f" shuffled returns ({_num(lam_lo, 3)} … {_num(lam_hi, 3)}). Any noisy series earns the "
                 "original's ‘chaotic’ label – Gaussian noise included.", "success")
    else:
        out.note(f"λ = {_num(lam, 3)} lies above all surrogates. This points to faster separation than for "
                 "noise – check with a different m, τ and seed whether this is robust before calling it chaos.",
                 "warning")
    if lam < lam_lo:
        out.note("That λ lies even slightly **below** the surrogates reflects volatility clustering: in calm phases, "
                 "neighbours lie close together and stay that way for a few days. This makes the range of "
                 "fluctuation somewhat predictable, not the direction.")
    above = [name for name, v, s in (("DET", det, det_s), ("LAM", lamr, lam_s)) if v > s.max()]
    if above:
        out.note(f"**Temporal structure in the recurrence plot:** {' and '.join(above)} "
                 f"{'lie' if len(above) > 1 else 'lies'} above all {n_sur} surrogates. Calm and turbulent "
                 "phases alternate in blocks (volatility clustering) – exactly what GARCH and regime models "
                 "describe. It says nothing about the direction of prices.")
    else:
        out.note(f"DET and LAM lie within the range of the surrogates: over these {len(x)} {dat} the recurrence "
                 "plot shows no structure that shuffled returns would not also have. "
                 + ("A longer series with calm and turbulent phases shows volatility clusters more clearly. "
                    if fallback else
                    "Weekly returns over several years usually show volatility clusters more clearly. "
                    if not weekly else "A period that includes a crisis usually shows volatility clusters more "
                                       "clearly. ")
                 + "For independently simulated returns, exactly this result is to be expected.")

    out.progress("Drawing the original chart …", 0.9)
    fig = rpa.plot_recurrence(figsize=(7, 7), epsilon=eps)
    out.figure(fig, f"Original: RecurrencePlotAnalyzer.plot_recurrence – full resolution, {len(R)} × {len(R)} "
                    f"states of the {scale_txt}, black = recurrence (time runs to the right and upwards).", dpi=100)


def _fmt(v, kind):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{_num(100 * v, 1)}%" if kind == "pct" else _num(v, 3)
