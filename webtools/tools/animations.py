"""Animations: the framework's four FuncAnimations, rendered frame by frame to a GIF in the browser."""

import os
import tempfile

import numpy as np

from src.analysis.early_warning import CompositeEarlyWarningSystem
from src.models.risk_metrics import RollingRiskMetrics
from src.physics.levy_flight import levy_flight_3d_coordinates
from src.utils.data_loader import generate_vix_from_returns
from src.visualization.animated_risk import (
    DistributionMorphAnimator,
    PhaseSpaceAnimator,
    RiskGaugeAnimator,
    create_comprehensive_animation,
)

from .. import charts as C
from ..core import ToolError
from ..data import index_axis, fmt_date
from ._ews_common import LEVEL

SPEC = {
    "id": "animations",
    "title": "Animations: tail risk in motion",
    "group": "dashboard",
    "order": 30,
    "summary": "The framework's four animations as GIFs – Lévy phase space, distribution morphing, risk gauge "
               "and a four-panel dashboard, rendered directly in the browser.",
    "description": (
        "A still image shows a state; an animation shows how the market got there. The framework comes with four "
        "animations: the **Lévy phase space** traces the market's path through volatility, tail heaviness (2/α) "
        "and jump intensity (|r|/σ) – calm phases circle tightly, crises swing far out. **Distribution morphing** "
        "shows the return distribution in a rolling 60-day window against the Gaussian curve; on the log scale "
        "you can see when fat tails emerge. The needle of the **risk gauge** tracks the composite score of the "
        "early-warning system. The **dashboard** synchronises phase space, distribution, jump risk and the "
        "rolling 99% VaR.\n\n"
        "Rendering uses Matplotlib in the browser, frame by frame. The original animation would have hundreds of "
        "frames (one per day or every few days). The tool shows an evenly spaced selection of them; ‘Medium’ "
        "takes about 10–15 seconds to compute. The time axis in the frame titles (t) counts trading days from "
        "the start of the selected period.\n\n"
        "The charts above the GIF show the same quantities interactively. The dots mark the days that appear as "
        "frames in the GIF. The gauge uses the modelled early-warning score (computed causally); only there does "
        "the **seed** matter."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "kind", "type": "select", "default": "phase", "label": "Animation",
         "options": [{"value": "phase", "label": "Lévy phase space (3D path)"},
                     {"value": "morph", "label": "Distribution morphing (fat tails emerge)"},
                     {"value": "gauge", "label": "Risk gauge (early-warning score)"},
                     {"value": "dashboard", "label": "Four-panel dashboard"}]},
        {"name": "length", "type": "select", "default": "medium", "label": "Length",
         "options": [{"value": "short", "label": "Short – half as many frames, fast"},
                     {"value": "medium", "label": "Medium"},
                     {"value": "long", "label": "Long – more frames, takes about twice as long"}]},
        {"name": "days", "type": "int", "default": 500, "min": 120, "max": 2520, "step": 20,
         "label": "Period", "unit": "days", "help": "The most recent trading days of the dataset to animate."},
        {"name": "dpi", "type": "int", "default": 75, "min": 50, "max": 100, "step": 5, "label": "Resolution",
         "unit": "dpi", "help": "Higher = sharper, but a larger file and longer computing time."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Random seed (risk gauge only)", "show_if": {"kind": ["gauge"]},
         "help": "Model noise of the early-warning indicators. Same seed = same score."},
    ],
    "covers": [
        "src.visualization.animated_risk.PhaseSpaceAnimator",
        "src.visualization.animated_risk.PhaseSpaceAnimator.create_animation",
        "src.visualization.animated_risk.DistributionMorphAnimator",
        "src.visualization.animated_risk.DistributionMorphAnimator.create_animation",
        "src.visualization.animated_risk.RiskGaugeAnimator",
        "src.visualization.animated_risk.RiskGaugeAnimator.create_animation",
        "src.visualization.animated_risk.create_comprehensive_animation",
        "src.physics.levy_flight.levy_flight_3d_coordinates",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_var",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_kurtosis",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.simulate_all_indicators",
    ],
    "cost": "medium",
}

# size (inches; the framework's figures are 10-16 in wide, too small to read as a web GIF), fps and frame counts
# (browser rendering costs about 0.5 s per phase frame, 1.3 s per morph frame and 3 s per dashboard frame)
KINDS = {
    "phase": {"name": "Lévy phase space", "src": "PhaseSpaceAnimator", "size": (9.0, 7.0), "fps": 6,
              "frames": {"short": 11, "medium": 22, "long": 44}},
    "morph": {"name": "Distribution morphing", "src": "DistributionMorphAnimator", "size": (11.0, 4.6), "fps": 2,
              "frames": {"short": 4, "medium": 8, "long": 16}},
    "gauge": {"name": "Risk gauge", "src": "RiskGaugeAnimator", "size": (9.0, 4.5), "fps": 8,
              "frames": {"short": 18, "medium": 36, "long": 72}},
    "dashboard": {"name": "Tail-risk dashboard", "src": "create_comprehensive_animation", "size": (11.0, 8.25),
                  "fps": 2, "frames": {"short": 3, "medium": 5, "long": 10}},
}
WINDOW = 60          # rolling window of DistributionMorphAnimator (default) and create_comprehensive_animation
COORD_WINDOW = 20    # levy_flight_3d_coordinates default
GAUGE_ZONES = ((0, 25, "Low"), (25, 50, "Moderate"), (50, 75, "Elevated"), (75, 100.01, "Extreme"))



def _gauge_layout(fig, size):
    """Frame the polar half-dial: the framework sizes it for a 10 x 8 in figure, where it floats in white space.

    A polar axes draws the half-dial in the middle of a square box, so the box is made almost as wide as the
    figure and reaches below its lower edge; the title gets a fixed height (``axes.titley``) above the dial.
    """
    w, h = size
    d = min(0.9 * w, 2 * (h - 1.0))                    # side of the square axes box (in)
    mid = d / 4 + 0.35                                 # vertical centre of the box (in)
    side = (w - d) / 2 / w
    fig.subplots_adjust(left=side, right=1 - side, bottom=(mid - d / 2) / h, top=(mid + d / 2) / h)
    return {"axes.titley": (h - 0.8 - (mid - d / 2)) / d}


def _render_gif(anim, n_frames, fps, dpi, size, out, first=0, layout=None):
    """Save the original FuncAnimation as GIF, but only ``n_frames`` evenly spaced frames of its own sequence."""
    import matplotlib
    from matplotlib.animation import PillowWriter

    fig = anim._fig
    fig.set_size_inches(*size)
    rc = layout(fig, size) if layout else {}
    frames = list(anim.new_frame_seq())
    total = len(frames)
    frames = [f for f in frames if f >= first] or frames
    if len(frames) > n_frames:
        pick = np.unique(np.linspace(0, len(frames) - 1, n_frames).round().astype(int))
        frames = [frames[i] for i in pick]
    # same frame function, fewer frames: FuncAnimation iterates _iter_gen() and saves _save_count frames
    anim._iter_gen = lambda: iter(frames)
    anim._save_count = len(frames)
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        with matplotlib.rc_context(rc):
            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                      progress_callback=lambda i, k: out.progress(f"Frame {i + 1} of {k} …",
                                                                   0.15 + 0.8 * (i + 1) / k))
        with open(path, "rb") as fh:
            gif = fh.read()
    finally:
        os.remove(path)
    return gif, [int(f) for f in frames], total


def run(params, data, out):
    kind = params["kind"]
    cfg = KINDS[kind]
    r_all = np.asarray(data.returns, dtype=float)
    n_all = data.n
    m = int(min(params["days"], n_all))
    off = n_all - m
    r = r_all[off:]
    x_all, xa = index_axis(data)
    x = x_all[off:]
    label = (lambda i: fmt_date(x[i])) if data.dates else (lambda i: f"Day {off + i}")
    need = WINDOW + 11 if kind == "dashboard" else WINDOW + 1 if kind == "morph" else 2
    if m < need:
        raise ToolError(f"This animation needs at least {need} trading days – the dataset has {m}.")

    out.progress("Preparing the animation …", 0.05)
    if kind in ("phase", "dashboard"):
        X, Y, Z = (np.asarray(a, dtype=float) for a in levy_flight_3d_coordinates(r, window=COORD_WINDOW))
    if kind == "phase":
        anim = PhaseSpaceAnimator(X, Y, Z, figsize=cfg["size"]).create_animation(trail_length=50)
    elif kind == "morph":
        anim = DistributionMorphAnimator(r, window=WINDOW).create_animation()
    elif kind == "gauge":
        vix = (np.asarray(data.vix, dtype=float) if data.vix is not None
               else np.asarray(generate_vix_from_returns(r_all), dtype=float))
        vix3m = np.asarray(data.vix3m, dtype=float) if data.vix is not None and data.vix3m is not None else None
        prices = np.asarray(data.prices, dtype=float) if data.prices is not None else 100.0 * np.exp(np.cumsum(r_all))
        ews = CompositeEarlyWarningSystem()
        np.random.seed(int(params["seed"]))
        ind = ews.simulate_all_indicators(r_all, prices, vix, causal=True, vix3m=vix3m)
        comp = np.asarray(ind["composite_score"], dtype=float)[off:]
        scores = [np.asarray(ind[k + "_score"], dtype=float) for k in ("gex", "tdex", "vix_term", "dix", "smfi")]
        anim = RiskGaugeAnimator(comp).create_animation()
    else:
        anim = create_comprehensive_animation(r, (X, Y, Z))

    n_frames = cfg["frames"][params["length"]]
    gif, frames, total = _render_gif(anim, n_frames, cfg["fps"], int(params["dpi"]), cfg["size"], out,
                                     first=COORD_WINDOW if kind == "phase" else 0,
                                     layout=_gauge_layout if kind == "gauge" else None)
    fx = [x[f] for f in frames]

    # ---- per kind: the numbers behind the picture -------------------------------------------------------------
    if kind == "phase":
        jumps = np.flatnonzero(Z > 3)
        out.metric("Volatility (relative)", X[-1], "num", hint=f"on {label(m - 1)}; 1 = average of the period")
        out.metric("Tail heaviness 2/α", Y[-1], "num", hint="Hill estimator over 20 days; higher = fatter tails")
        big_txt = f"{Z.max():.1f}"
        out.metric("Jump intensity |r|/σ", Z[-1], "num",
                   hint=f"largest jump {big_txt}σ on {label(int(np.argmax(Z)))}; σ of the period")
        out.metric("Days with jump > 3σ", len(jumps), "int",
                   hint=f"{100 * len(jumps) / m:.1f}% of days")
        s = slice(COORD_WINDOW, None)  # the tail index needs a full window first (2/α is 0 before)
        idx = np.arange(m)[s]
        traces = [C.path3d("Path", X[s], Y[s], Z[s], color=1, mode="lines", text=[label(i) for i in idx])]
        big = jumps[jumps >= COORD_WINDOW]
        if big.size:
            traces.append(C.path3d("Jump > 3σ", X[big], Y[big], Z[big], color="critical", mode="markers", size=4,
                                   text=[label(i) for i in big]))
        shown = [f for f in frames if f >= COORD_WINDOW]
        traces.append(C.path3d("Frames in the GIF", X[shown], Y[shown], Z[shown], color=3, mode="markers", size=3,
                               text=[label(i) for i in shown]))
        traces.append(C.path3d("Last day", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers", size=7,
                               text=[label(m - 1)]))
        out.chart(C.scatter3d(
            "Lévy phase space: the full path", traces,
            x=C.axis("Volatility (relative)"), y=C.axis("Tail heaviness 2/α"), z=C.axis("Jump intensity |r|/σ"),
            subtitle="Each point on the line is one day (from day 20, once the tail index is estimated). Rotate "
                     "the view with the mouse.", height=520,
        ))
    elif kind == "morph":
        # RollingRiskMetrics' value at index t+1 uses returns[t-59 .. t], i.e. the window ending on day t (as in
        # the GIF). The appended 0 never enters a window; it only makes the window ending on the last day exist.
        kurt = np.asarray(RollingRiskMetrics(np.append(r, 0.0), window=WINDOW).rolling_kurtosis(), dtype=float)[1:]
        ok = np.isfinite(kurt)
        last = float(kurt[ok][-1]) if ok.any() else float("nan")
        i_max = int(np.nanargmax(kurt)) if ok.any() else 0
        out.metric("Excess kurtosis (last window)", last, "num", digits=1,
                   status="critical" if last > 6 else "serious" if last > 3 else None,
                   badge="Fat tails" if last > 3 else "",
                   hint=f"60 days to {label(m - 1)}; normal distribution = 0")
        out.metric("Highest excess kurtosis", kurt[i_max], "num", digits=1, hint=f"window ending {label(i_max)}")
        out.metric("Windows with excess kurtosis > 3", float(np.mean(kurt[ok] > 3)) if ok.any() else 0.0, "pct", digits=0,
                   hint="share of 60-day windows with clearly fat tails")
        out.chart(C.xy(
            "Excess kurtosis in the rolling 60-day window",
            [C.trace("Excess kurtosis", x, kurt, color=1, width=1.5),
             C.trace("Frames in the GIF", fx, kurt[frames], type="scatter", color=2, size=7, decimate_to=None)],
            x=xa, y=C.axis("Excess kurtosis"),
            hlines=[C.hline(0, "Normal distribution"), C.hline(3, "clearly fat tails", "serious")],
            subtitle="Value on day t: window of the 60 trading days up to t. When kurtosis jumps, individual extreme "
                     "days have entered the window – in the GIF, bars then sit far out above the Gaussian curve.",
        ))
    elif kind == "gauge":
        st, lv = LEVEL[ews.compute_composite(*(float(s_[-1]) for s_ in scores))["level"]]
        zone = next(z for lo, hi, z in GAUGE_ZONES if lo <= min(comp[-1], 100) < hi)
        i_max = int(np.argmax(comp))
        out.metric("Score (last day)", comp[-1], "num", digits=0, unit="/ 100", status=st,
                   hint=f"gauge zone {zone} · early-warning level: {lv}")
        out.metric("Average", float(np.mean(comp)), "num", digits=0, unit="/ 100",
                   hint=f"over {m:,} trading days")
        out.metric("Highest score", comp[i_max], "num", digits=0, unit="/ 100", hint=f"on {label(i_max)}")
        out.chart(C.xy(
            "Composite score shown by the gauge needle",
            [C.trace("Composite score", x, comp, color=1, width=1.5),
             C.trace("Frames in the GIF", fx, comp[frames], type="scatter", color=2, size=6, decimate_to=None)],
            x=xa, y=C.axis("Score", range=(0, 100)),
            hlines=[C.hline(25, "Moderate"), C.hline(50, "Elevated"), C.hline(75, "Extreme")],
            subtitle="Lines = gauge zones (25 / 50 / 75). The alert levels of the early-warning system are at "
                     "50 / 75 / 90.",
        ))
        rows = [[z, int(np.sum((comp >= lo) & (comp < hi))), float(np.mean((comp >= lo) & (comp < hi)))]
                for lo, hi, z in GAUGE_ZONES]
        out.table("Time in the gauge zones", ["Zone", "Days", "Share"], rows, formats=["text", "int", "pct"],
                  note="Zones as in the animated gauge: 0–25 low, 25–50 moderate, 50–75 elevated, 75–100 extreme.")
        out.note("The score is the composite of the early-warning system. GEX, TailDex, DIX and SMFI are *modelled* "
                 "for it from returns and VIX (illustrative, not actual measurements) and computed causally. "
                 "‘Current Level: … %’ in the GIF title means points out of 100, not a probability."
                 + ("" if data.vix is not None else " The VIX is missing and is estimated from the returns."),
                 "warning")
    else:
        var = np.asarray(RollingRiskMetrics(r, window=WINDOW).rolling_var(0.99), dtype=float)
        loss = -r
        ok = np.isfinite(var)
        breach = ok & (loss > np.where(ok, var, np.inf))
        p90 = float(np.percentile(Z, 90))
        i_max = int(np.nanargmax(var))
        rate = float(np.mean(breach[ok]))
        out.metric("99% VaR (60 days)", var[-1], "pct", digits=2,
                   hint=f"historical, for {label(m - 1)} from the preceding 60 days")
        out.metric("VaR breaches", rate, "pct", digits=1,
                   hint=f"{int(breach.sum())} of {int(ok.sum())} days · expected 1%",
                   status="critical" if rate > 0.025 else "serious" if rate > 0.015 else "good")
        out.metric("Highest 99% VaR", var[i_max], "pct", digits=2, hint=f"on {label(i_max)}")
        n_jump = int(np.sum(Z > 3))
        out.metric("Days with jump > 3σ", n_jump, "int",
                   hint=f"{100 * n_jump / m:.1f}% of days; σ of the period")
        out.chart(C.xy(
            "One-day loss and rolling 99% VaR",
            [C.trace("One-day loss", x, np.where(loss > 0, loss, np.nan), type="bar", color="muted"),
             C.trace("99% VaR (60 days)", x, var, color=1, width=1.5),
             C.trace("Loss above VaR", [x[i] for i in np.flatnonzero(breach)], loss[breach], type="scatter",
                     color="critical", size=6, decimate_to=None)],
            x=xa, y=C.axis("Loss", fmt="pct", digits=1),
            subtitle="As in panel 4 of the animation: VaR on day t from the preceding 60 days. With only 60 values "
                     "the 1% quantile can hardly be estimated – even if the distribution does not change, just over 2% of days "
                     "lie above it.",
        ))
        out.chart(C.xy(
            "Jump intensity |r|/σ",
            [C.trace("|r|/σ", x, Z, color=2, width=1),
             C.trace("Frames in the GIF", fx, Z[frames], type="scatter", color=3, size=6, decimate_to=None)],
            x=xa, y=C.axis("Multiples of σ"), hlines=[C.hline(p90, "90th percentile", "serious")],
            subtitle="As in panel 3 of the animation (‘Tail Risk Evolution’).",
        ))

    out.metric("Frames in the GIF", len(frames), "int", hint=f"out of {total:,} frames of the original animation")
    kb = f"{len(gif) / 1024:,.0f}"
    out.note(f"**{cfg['name']}** (`{cfg['src']}`): {len(frames)} of {total:,} frames, {cfg['fps']} frames/s, "
             f"{kb} KB. Period {label(0)} – {label(m - 1)}; t in the frame title = trading day from {label(0)} "
             "(t = 0).", "info")
    out.image(gif, f"{cfg['name']} – framework animation ({cfg['src']}, {len(frames)} frames)",
              mime="image/gif")
