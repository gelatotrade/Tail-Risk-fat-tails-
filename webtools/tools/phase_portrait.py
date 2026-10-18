"""Market phase portrait: momentum x volatility x skewness, attractors, velocity and distance from equilibrium."""

import numpy as np

from src.visualization.phase_space import PhaseSpaceAnalyzer

from .. import charts as C
from ..data import fmt_date
from ..core import ToolError

SPEC = {
    "id": "phase-portrait",
    "title": "Market phase portrait: momentum, volatility, skewness",
    "group": "regime",
    "order": 40,
    "summary": "Tracks the market as a point in a space of momentum, volatility and skewness – with attractors, "
               "speed, acceleration and distance from equilibrium.",
    "description": (
        "In physics, a **phase space** describes the complete state of a system; its trajectory shows where the "
        "system is heading. `PhaseSpaceAnalyzer` applies this to the market: each trading day becomes a point "
        "defined by **momentum** X (average daily return in the window), **volatility** Y (standard deviation in "
        "the window) and **skewness** Z (of the preceding returns). Calm periods form a dense cloud at low "
        "volatility; crises stretch out in long streaks upwards and towards negative momentum.\n\n"
        "**Attractors** are regions to which the market keeps returning. `detect_attractor_regions` finds them by "
        "Ward clustering and orders them by volatility (calm → stress). Standardise the axes for this: skewness "
        "is of the order of ±1, return and volatility of the order of 0.01 – unscaled, the clustering separates "
        "almost solely by skewness. The average stay shows how ‘sticky’ an attractor is.\n\n"
        "**Velocity** and **acceleration** (`compute_velocity`, `compute_acceleration`) measure how fast the "
        "state is changing – rapid movement signals a regime switch. The **distance from equilibrium** "
        "(`phase_space_distance`) measures how far the market is from its resting point (momentum 0, average "
        "volatility, skewness 0), here in standard deviations per axis. The original chart draws the trajectory "
        "with velocity arrows (`Arrow3D`)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 20, "min": 5, "max": 120, "step": 1,
         "label": "Window", "unit": "days",
         "help": "For momentum, volatility and skewness. Short = jittery, long = sluggish."},
        {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 6, "step": 1,
         "label": "Number of attractors"},
        {"name": "standardize", "type": "bool", "default": True, "label": "Standardise axes for clustering",
         "help": "Off: the unscaled original behaviour, which separates almost solely by skewness."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Highlighted recent days", "unit": "days"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Period (most recent trading days)", "unit": "days",
         "help": "Most recent trading days analysed. Clustering needs memory ∝ N²; 2,520 days ≈ 10 years."},
        {"name": "show_velocity", "type": "bool", "default": True,
         "label": "Velocity arrows in the original chart"},
    ],
    "covers": [
        "src.visualization.phase_space.Arrow3D",
        "src.visualization.phase_space.Arrow3D.do_3d_projection",
        "src.visualization.phase_space.PhaseSpaceAnalyzer",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.compute_velocity",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.compute_acceleration",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.phase_space_distance",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.detect_attractor_regions",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.plot_phase_portrait",
    ],
    "cost": "medium",
}

REGIME_LABELS = {"LOW_VOL": "Calm (low vol)", "NORMAL": "Normal", "HIGH_VOL": "Stress (high vol)"}
RANK_COLORS = (1, 4, 2, 7, 5, 8)  # attractors ordered calm -> stress: blue, amber, orange, ... (never green for stress)
MAX_CLOUD = 500   # points per attractor in the 3D cloud: denser clouds hide the recent path (the original figure
                  # and the 2D charts always get every day)

def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma, typographic minus."""
    return f"{x:,.{digits}f}".replace("-", "−")


def _signed(x, digits=2, unit=""):
    """Signed number with at least two significant digits (tiny daily changes must not round to 0)."""
    if x != 0 and np.isfinite(x):
        digits = int(min(7, max(digits, 1 - np.floor(np.log10(abs(x))))))
    s = _num(abs(x), digits)
    return ("+" if x > 0 else "−" if x < 0 else "±") + s + unit




def _regime_label(label):
    if label in REGIME_LABELS:
        return REGIME_LABELS[label]
    if label.startswith("NORMAL_"):
        return f"Normal {label.split('_')[1]}"
    return label


def _rank_pct(series, value):
    """Mid-rank percentile of value within series (ties count half)."""
    s = series[np.isfinite(series)]
    return float(0.5 * (np.mean(s < value) + np.mean(s <= value))) if s.size else np.nan


def _mean_run(labels, cid):
    """Mean length (days) of uninterrupted stays in cluster cid."""
    hit = np.concatenate([[0], (labels == cid).astype(np.int8), [0]])
    edges = np.diff(hit)
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.mean()) if lengths.size else np.nan


def run(params, data, out):
    w = int(params["window"])
    k_clusters = int(params["n_clusters"])
    n = int(min(params["last_n"], data.n))
    need = max(3 * w, 60)              # at least two full windows of clustered points after the warm-up
    if n < need:
        fix = ("Reduce the window" + (" or extend the period." if n < data.n else ".")
               if 3 * w > 60 else "Choose a longer dataset.")
        raise ToolError(f"With a window of {w} days you need at least {need:,} daily returns in the period "
                        f"(available: {n:,}). {fix}")
    r = np.asarray(data.returns[-n:], dtype=float)
    dates = list(data.dates[-n:]) if data.dates else None
    x_time = dates if dates else list(range(data.n - n + 1, data.n + 1))
    x_axis = C.axis("", type="date") if dates else C.axis("Trading day", fmt="int")
    day_txt = [fmt_date(d) for d in dates] if dates else [f"Day {i}" for i in x_time]

    out.progress("Phase-space coordinates …", 0.05)
    psa = PhaseSpaceAnalyzer(r, window=w)
    X, Y, Z = (np.asarray(a, dtype=float) for a in (psa.X, psa.Y, psa.Z))
    vel = np.column_stack(psa.compute_velocity())
    acc = np.column_stack(psa.compute_acceleration())
    dist = np.asarray(psa.phase_space_distance(standardize=True), dtype=float)
    dist_raw = np.asarray(psa.phase_space_distance(), dtype=float)

    out.progress("Attractors (Ward clustering) …", 0.25)
    att = psa.detect_attractor_regions(n_clusters=k_clusters, standardize=bool(params["standardize"]))
    if "error" in att:
        raise ToolError("Too few valid points for clustering – extend the period.")

    # the first `window` days are warm-up (skewness is still 0, the windows are not full yet)
    valid = np.arange(n) >= w
    coords = np.column_stack([X, Y, Z])
    finite = ~np.any(np.isnan(coords), axis=1)
    labels = np.zeros(n, dtype=int)
    labels[finite] = np.asarray(att["cluster_labels"], dtype=int)
    order = [int(c) for c in att["regime_order"]]          # 0-based cluster index, low -> high volatility
    rank = {c + 1: i for i, c in enumerate(order)}         # cluster id -> volatility rank 0..k-1
    names = {cid: _regime_label(lbl) for cid, lbl in att["cluster_regimes"].items()}
    centers = np.asarray(att["centers"], dtype=float)

    # speed and acceleration in standard deviations per axis (raw units would be pure skewness)
    scale = np.nanstd(coords[valid], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    speed = np.sqrt(np.sum((vel / scale) ** 2, axis=1))
    accel = np.sqrt(np.sum((acc / scale) ** 2, axis=1))

    cur = int(labels[-1])
    cur_rank = rank.get(cur, 0)
    status = "good" if cur_rank == 0 else "serious" if cur_rank == k_clusters - 1 else "warning"
    vol_pct = _rank_pct(Y[valid], Y[-1])
    dist_pct = _rank_pct(dist[valid], dist[-1])
    speed_pct = _rank_pct(speed[valid], speed[-1])
    acc_pct = _rank_pct(accel[valid], accel[-1])

    # --- metrics ------------------------------------------------------------------------
    stay = _run_today(labels)
    out.metric("Attractor today", names.get(cur, "–"), "text", status=status,
               badge=f"Vol rank {cur_rank + 1}/{k_clusters}",
               hint="entered this attractor today" if stay == 1 else f"in this attractor for {stay:,} days")
    out.metric(f"Volatility ({w} days)", Y[-1], "pct", digits=2,
               hint=f"daily · ≈ {_num(100 * Y[-1] * np.sqrt(252), 0)}% p.a. · percentile rank "
                    f"{_num(100 * vol_pct, 0)}%")
    out.metric(f"Momentum ({w} days)", X[-1], "pct", digits=3, hint="avg. daily return in the window")
    out.metric("Skewness", Z[-1], "num", digits=2,
               hint=f"of the preceding {w} returns · < 0: large losses more frequent than large gains")
    out.metric("Distance from equilibrium", dist[-1], "num", digits=2,
               hint=f"in σ per axis · percentile rank {_num(100 * dist_pct, 0)}%",
               status="serious" if dist_pct >= 0.97 else "warning" if dist_pct >= 0.9 else None)
    out.metric("Speed in phase space", speed[-1], "num", digits=3,
               hint=f"σ per day · percentile rank {_num(100 * speed_pct, 0)}%",
               status="warning" if speed_pct >= 0.9 else None)
    out.metric("Acceleration", accel[-1], "num", digits=3,
               hint=f"σ per day² · percentile rank {_num(100 * acc_pct, 0)}%")

    # --- chart 1: the phase portrait in 3D ---------------------------------------------
    idx = np.flatnonzero(valid)
    k_recent = int(min(params["recent"], len(idx)))
    recent = idx[-k_recent:]
    traces, thinned = [], False
    for cid in sorted(rank, key=rank.get):
        ii = idx[labels[idx] == cid]
        if ii.size > MAX_CLOUD:
            ii, thinned = ii[::int(np.ceil(ii.size / MAX_CLOUD))], True
        if ii.size:
            traces.append(C.path3d(f"{names[cid]} · {int(np.sum(labels[idx] == cid)):,} days",
                                   X[ii], Y[ii], Z[ii], color=RANK_COLORS[rank[cid]], mode="markers", size=2,
                                   text=[day_txt[i] for i in ii]))
    traces.append(C.path3d("Attractor centres", centers[:, 0], centers[:, 1], centers[:, 2], color="muted",
                           mode="markers", size=9, text=[names[c + 1] for c in range(len(centers))]))
    traces.append(C.path3d(f"Last {k_recent} days", X[recent], Y[recent], Z[recent], color="ink",
                           mode="lines+markers", size=3, text=[day_txt[i] for i in recent]))
    traces.append(C.path3d(f"Today ({day_txt[-1]})", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers",
                           size=8, text=[names.get(cur, "")]))
    out.chart(C.scatter3d(
        "Phase portrait: momentum × volatility × skewness", traces,
        x=C.axis("Momentum (avg. daily return)", fmt="pct", digits=2), y=C.axis("Volatility (daily)", fmt="pct",
                                                                               digits=1),
        z=C.axis("Skewness"), height=580,
        subtitle=f"Each point is a trading day ({w}-day window), coloured by attractor (calm → stress)"
                 + (f"; large attractors show every n-th day (at most {MAX_CLOUD} points)" if thinned else "")
                 + ". The dark line is the path of the most recent days. Drag to rotate, scroll to zoom.",
    ))

    # --- chart 2: volatility over time coloured by attractor ----------------------------
    tr = []
    for cid in sorted(rank, key=rank.get):
        ii = idx[labels[idx] == cid]
        if ii.size:
            tr.append(C.trace(names[cid], [x_time[i] for i in ii], Y[ii], type="scatter", color=RANK_COLORS[rank[cid]],
                              size=3, decimate_to=None, text=[day_txt[i] for i in ii]))
    out.chart(C.xy(
        "Volatility over time – coloured by attractor", tr,
        x=x_axis, y=C.axis(f"Volatility ({w} days, daily)", fmt="pct", digits=1), height=320,
        subtitle="Long stretches of one colour: the market stays in its attractor. Colour changes are regime "
                 "switches.",
    ))

    # --- chart 3: distance from equilibrium ----------------------------------------------
    xs = [x_time[i] for i in idx]
    d95 = float(np.nanpercentile(dist[valid], 95))
    out.chart(C.xy(
        "Distance from equilibrium (standardised)",
        [C.trace("Distance", xs, dist[valid], color=1, width=1.2)],
        x=x_axis, y=C.axis("Distance (σ per axis)"), height=300,
        hlines=[C.hline(d95, "95th percentile")],
        subtitle="Euclidean distance to the resting point (momentum 0, average volatility, skewness 0), each axis "
                 "measured in its own standard deviations. Peaks are exceptional states.",
    ))

    # --- chart 4: speed in phase space ------------------------------------------------
    sp = speed[valid]
    sp_avg = np.convolve(sp, np.ones(w) / w, mode="full")[:len(sp)]
    sp_avg[:w - 1] = np.nan
    out.chart(C.xy(
        "Speed in phase space",
        [C.trace("Daily value", xs, sp, color=2, width=0.8),
         C.trace(f"{w}-day average", xs, sp_avg, color=1, width=2.2)],
        x=x_axis, y=C.axis("σ per day"), height=320,
        subtitle=f"Magnitude of the velocity (np.gradient per axis, standardised). A shock day produces two "
                 f"peaks {w} days apart – when it enters the window and when it leaves it. The average shows "
                 "turbulent phases in which the state shifts constantly.",
    ))

    # --- table 1: attractors ---------------------------------------------------------
    rows = []
    lab_v = labels[idx]
    for cid in sorted(rank, key=rank.get):
        c = centers[cid - 1]
        mask = lab_v == cid
        last = fmt_date(x_time[idx[np.flatnonzero(mask)[-1]]]) if mask.any() and dates else (
            f"Day {x_time[idx[np.flatnonzero(mask)[-1]]]}" if mask.any() else "–")
        rows.append([names[cid] + (" ← today" if cid == cur else ""), float(c[0]), float(c[1]),
                     float(c[1] * np.sqrt(252)), float(c[2]), float(mask.mean()), _mean_run(lab_v, cid), last])
    eta = _eta_squared(coords[idx], lab_v)
    eta_txt = (f"The attractors explain {_num(100 * eta[1], 0)}% of the variation in volatility, "
               f"{_num(100 * eta[2], 0)}% in skewness and {_num(100 * eta[0], 0)}% in momentum (η²). ")
    out.table("Attractor regions (Ward clustering)", [
        "Attractor", "Avg. momentum", "Avg. vol (daily)", "Avg. vol p.a.", "Avg. skewness", "Share of days",
        "Avg. stay (days)",
        "Last seen"], rows, formats=["text", "pct", "pct", "pct", "num", "pct", "num", "text"],
        note=eta_txt + ("Centres in original units, clustering on standardised axes. "
                        if params["standardize"] else "Clustering on unscaled axes (original behaviour). ")
        + "Stay = mean length of uninterrupted spells. Excludes the first "
        + f"{w} days (warm-up).")

    # --- table 2: motion today per axis ----------------------------------------------
    axes = [("Momentum X", X, "pct", 3), ("Volatility Y", Y, "pct", 2), ("Skewness Z", Z, "num", 2)]
    rows = []
    for j, (name, s, kind, dg) in enumerate(axes):
        unit = "%" if kind == "pct" else ""
        f = 100.0 if kind == "pct" else 1.0
        v, a = float(vel[-1, j]), float(acc[-1, j])
        trend = ("rising" if v > 0 else "falling" if v < 0 else "flat")
        trend += (", accelerating" if v * a > 0 else ", decelerating" if v * a < 0 else "")
        rows.append([name, _num(f * s[-1], dg) + unit, _rank_pct(s[valid], s[-1]), _signed(f * v, dg + 1, unit),
                     _signed(f * a, dg + 1, unit), float(abs(v) / scale[j]), trend])
    out.table("Motion today by axis", [
        "Axis", "Today", "Percentile", "Velocity per day", "Acceleration per day²", "Speed (σ/day)",
        "Direction"], rows, formats=["text", "text", "pct", "text", "text", "num", "text"],
        note="Central differences (np.gradient) of the coordinates; one-sided on the last day. Speed = "
             "|velocity| divided by the standard deviation of the axis.")

    # --- notes ---------------------------------------------------------------------
    # the original clusters every row, including the warm-up (partial windows, skewness 0): on short series one of
    # those artefacts can become an "attractor" of its own
    warm_only = [names[cid] for cid in sorted(rank, key=rank.get) if not np.any(lab_v == cid)]
    if warm_only:
        out.note(f"The attractor ‘{warm_only[0]}’ consists only of warm-up days (the first {w} days, window not "
                 "yet full) – an artefact of the short series, not a market state. Reduce the window or use a "
                 "longer series.", "warning")
    raw_corr =float(np.corrcoef(dist_raw[valid], np.abs(Z[valid]))[0, 1]) if np.std(dist_raw[valid]) > 0 else np.nan
    if np.isfinite(raw_corr) and params["standardize"]:
        out.note(f"Why standardise? Unscaled, the distance from equilibrium is practically just |skewness| "
                 f"(correlation {_num(raw_corr)}): return and volatility are of the order of 0.01, skewness of the "
                 "order of 1. The same applies to clustering – switch off ‘Standardise axes for clustering’ to see "
                 "the effect.")
    if not params["standardize"] and eta[2] > 1.5 * eta[1]:
        out.note(f"Unscaled clustering: the attractors explain {_num(100 * eta[2], 0)}% of the variation in "
                 f"skewness but only {_num(100 * eta[1], 0)}% of the variation in volatility – the ‘volatility "
                 "regimes’ here are really skewness groups. Switch standardisation back on.", "warning")
    elif not params["standardize"]:
        out.note("Unscaled clustering (original behaviour): in the distance measure, skewness (order of magnitude "
                 "1) weighs roughly a hundred times more than return and volatility (order of magnitude 0.01).",
                 "info")

    out.progress("Drawing the original chart …", 0.6)
    fig = psa.plot_phase_portrait(figsize=(11, 8), show_velocity=bool(params["show_velocity"]),
                                  show_attractors=True)
    out.figure(fig, "Original: PhaseSpaceAnalyzer.plot_phase_portrait – colour = time, red arrows = velocity "
                    "(Arrow3D, ×10, unscaled), stars = 3 attractors, green diamond = today.", dpi=72)


def _eta_squared(values, labels):
    """Share of each column's variance explained by the cluster means (between / total sum of squares)."""
    values = np.asarray(values, dtype=float)
    total = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    between = np.zeros(values.shape[1])
    for c in np.unique(labels):
        grp = values[labels == c]
        between += len(grp) * (grp.mean(axis=0) - values.mean(axis=0)) ** 2
    return np.where(total > 0, between / np.where(total > 0, total, 1.0), 0.0)


def _run_today(labels):
    """Length of the current uninterrupted stay in today's cluster."""
    same = labels[::-1] == labels[-1]
    return int(np.argmin(same)) if not same.all() else int(len(labels))
