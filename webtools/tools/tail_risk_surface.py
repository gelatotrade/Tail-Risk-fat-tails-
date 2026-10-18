"""The README's flagship: the 3D tail-risk surface with the active dataset's trajectory projected onto it."""

import numpy as np

import generate_tail_risk_surface as surface_script  # repo-root script; main() only runs under __main__
from src.physics.levy_flight import LevyStableDistribution, estimate_tail_index

from .. import charts as C
from ..data import fmt_date
from ..core import ToolError

SPEC = {
    "id": "tail-risk-surface",
    "title": "3D tail-risk surface with market trajectory",
    "group": "surface",
    "order": 10,
    "summary": "The framework's flagship chart, fully rotatable: crash probability across volatility and tail index, "
               "with your dataset's trajectory and today's position on the cliff.",
    "description": (
        "The surface shows how likely **at least one one-day loss beyond the crash threshold** (default 5%) is "
        "over the next 30 trading days, as a function of **annualised volatility** (x) and the **tail index α** "
        "(y; 2 = Gaussian tail, lower = fatter tails). It is computed with `crash_probability` from "
        "`generate_tail_risk_surface.py`: the power-law tail of a Lévy-stable distribution, "
        "p(day) = sin(πα/2)·α·(threshold/σ_day)^(−α)/π, scaled to the horizon with 1 − (1 − p)^30.\n\n"
        "On top sits **your dataset's trajectory**: for each day, volatility and α are estimated from the last N "
        "returns and placed on the surface; the dark line covers the last few weeks, the dark dot is today. The "
        "floor contours (5 / 15 / 30 / 50%) are the README's early-warning thresholds – each one crossed "
        "escalates the recommended response. Drag to rotate, scroll to zoom and hover over the trajectory to see "
        "the date.\n\n"
        "The **crash threshold** moves the cliff (3%: wide and early, 10%: narrow and late). The **window** sets "
        "how quickly the trajectory reacts. The **estimator** selects Hill (the largest daily moves; values above "
        "2 land on the Gaussian edge, where the surface shows no crash risk) or McCulloch's quantile estimator for "
        "Lévy-stable distributions. The calibration table shows what actually happened after days in each zone."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 5.0, "min": 3.0, "max": 10.0, "step": 0.5,
         "label": "Crash threshold (one-day loss)", "unit": "%",
         "help": "The one-day loss above which a day counts as a crash."},
        {"name": "window", "type": "int", "default": 60, "min": 20, "max": 250, "step": 5,
         "label": "Window for volatility and α", "unit": "days"},
        {"name": "estimator", "type": "select", "default": "hill", "label": "Estimator for α",
         "options": [{"value": "hill", "label": "Hill (largest daily moves)"},
                     {"value": "stable", "label": "Lévy-stable (McCulloch)"}],
         "help": "Hill measures the tail of the largest daily moves; McCulloch reads α from quantile ratios."},
        {"name": "horizon", "type": "int", "default": 30, "min": 5, "max": 60, "step": 5,
         "label": "Horizon", "unit": "trading days", "help": "The README uses 30 trading days."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Highlighted recent days", "unit": "days"},
        {"name": "script_paths", "type": "bool", "default": False,
         "label": "Show the script's sketched crisis paths",
         "help": "COVID-19, 2022 bear market, 2025 tariff crash – hand-drawn example paths, not measured data."},
    ],
    "covers": [
        "script:generate_tail_risk_surface.py",
        "src.physics.levy_flight.estimate_tail_index",
        "src.physics.levy_flight.LevyStableDistribution.fit",
    ],
    "cost": "fast",
}

# Zones of the README's reading table; the upper bound is the contour that ends the zone.
ZONES = [
    (0.05, "Calm plateau", "good", "green", "Standard positioning"),
    (0.15, "Stress ramp", "warning", "yellow", "Increase hedges"),
    (0.50, "Crash cliff", "serious", "orange-red", "Reduce exposure, add tail hedges"),
    (np.inf, "Black-swan zone", "critical", "violet", "Defensive positioning"),
]
# The floor contours of the README with the escalation of the script's reading guide.
CONTOURS = [
    (0.05, "Watch: the market is leaving the calm plateau",
     "Increase hedges – especially if the move happens within a few weeks"),
    (0.15, "Elevated: start of the crash cliff", "Reduce exposure, monitor GEX and TDEX"),
    (0.30, "Critical: the red ridge", "Defensive positioning, tail hedges"),
    (0.50, "Black-swan territory", "Fully defensive – any further volatility spike can trigger the crash"),
]
GRID = 46          # surface resolution (46 x 46 keeps the payload small and the mesh smooth)
MAX_EVAL = {"hill": 3000, "stable": 1500}  # rolling estimates per run (long series: every k-th day)
MAX_PATH = 420     # points of the drawn history path
Z_LIFT = 0.012     # draw paths slightly above the surface so they are not hidden by it

def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def _zone(p):
    for i, (hi, name, status, region, action) in enumerate(ZONES):
        if p < hi:
            return i, name, status, region, action
    return len(ZONES) - 1, *ZONES[-1][1:]


def _alpha(window_returns, estimator):
    try:
        if estimator == "stable":
            return float(LevyStableDistribution.fit(window_returns).alpha)
        return float(estimate_tail_index(window_returns))
    except ValueError:  # (nearly) constant window
        return np.nan


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    w = int(params["window"])
    thr = float(params["threshold"]) / 100.0
    horizon = int(params["horizon"])
    estimator = params["estimator"]
    if n < w + 20:
        raise ToolError(f"The dataset has only {n} days. A window of {w} days needs at least {w + 20} – "
                        "reduce the window.")

    # --- the surface: the script's own axes and risk model ------------------------------
    vol_axis, alpha_axis, vol_grid, alpha_grid, _ = surface_script._surface_grid(GRID)
    risk_grid = surface_script.crash_probability(vol_grid, alpha_grid, threshold=thr, horizon_days=horizon)
    v_lo, v_hi = float(vol_axis[0]), float(vol_axis[-1])
    a_lo, a_hi = float(alpha_axis[0]), float(alpha_axis[-1])

    # --- the dataset's trajectory: rolling volatility and tail index -------------------
    stride = max(1, int(np.ceil((n - w + 1) / MAX_EVAL[estimator])))
    ends = np.arange(w, n + 1, stride)
    if ends[-1] != n:
        ends = np.append(ends, n)
    win = np.lib.stride_tricks.sliding_window_view(r, w)[ends - w]
    vol = win.std(axis=1) * np.sqrt(252.0)
    alpha = np.array([_alpha(block, estimator) for block in win])
    ok = np.isfinite(alpha) & np.isfinite(vol)
    if ok.sum() < 2:
        raise ToolError("No tail index could be estimated for this dataset (too many constant returns).")
    ends, vol, alpha = ends[ok], vol[ok], alpha[ok]
    day = ends - 1                                    # index of the last return in each window
    vol_s = np.clip(vol, v_lo, v_hi)                  # position on the drawn surface
    alpha_s = np.clip(alpha, a_lo, a_hi)
    prob = surface_script.crash_probability(vol_s, alpha_s, threshold=thr, horizon_days=horizon)
    dates = [data.dates[i] for i in day] if data.dates else [f"Day {i + 1}" for i in day]
    x_time = [data.dates[i] for i in day] if data.dates else list(day + 1)
    x_axis = C.axis("", type="date") if data.dates else C.axis("Trading day", fmt="int")

    p_now, v_now, a_now, a_now_s = float(prob[-1]), float(vol[-1]), float(alpha[-1]), float(alpha_s[-1])
    zi, zname, zstatus, zregion, zaction = _zone(p_now)
    i_max = int(np.argmax(prob))
    at_edge = float(np.mean(alpha >= a_hi))

    # --- next contour: how far is the cliff? ------------------------------------------
    next_level = next((lvl for lvl, *_ in CONTOURS if lvl > p_now), None)
    vol_needed = alpha_needed = None
    if next_level is not None:
        vv = np.linspace(min(max(v_now, v_lo), v_hi), v_hi, 600)
        pv = surface_script.crash_probability(vv, np.full_like(vv, a_now_s), threshold=thr, horizon_days=horizon)
        hit = np.flatnonzero(pv >= next_level)
        vol_needed = float(vv[hit[0]]) if hit.size else None
        aa = np.linspace(a_now_s, a_lo, 600)
        pa = surface_script.crash_probability(np.full_like(aa, min(max(v_now, v_lo), v_hi)), aa,
                                              threshold=thr, horizon_days=horizon)
        hit = np.flatnonzero(pa >= next_level)
        alpha_needed = float(aa[hit[0]]) if hit.size else None

    # --- metrics ------------------------------------------------------------------------
    thr_txt = _pct(thr, 1 if thr * 1000 % 10 else 0)
    est_name = "Hill" if estimator == "hill" else "McCulloch"
    out.metric(f"Crash probability ({horizon} days)", p_now, "pct", digits=1, status=zstatus, badge=zname,
               hint=f"P(at least one one-day loss > {thr_txt} within {horizon} trading days) from the surface")
    out.metric("Volatility p.a.", v_now, "pct", digits=1, hint=f"rolling over {w} days")
    edge_hint = (f"{est_name} estimator · {_num(a_now_s)} on the surface (Gaussian edge)" if a_now > a_hi else
                 f"{est_name} estimator · {_num(a_now_s)} on the surface (end of axis)" if a_now < a_lo else
                 f"{est_name} estimator over {w} days · lower = fatter tails")
    out.metric("Tail index α", a_now, "num", digits=2, hint=edge_hint)
    if next_level is not None:
        lvl_txt = f"{round(next_level * 100)}% contour"
        if alpha_needed is not None and alpha_needed < a_now_s - 1e-9:
            out.metric(f"α threshold for the {lvl_txt}", alpha_needed, "num", digits=2,
                       hint=f"if α falls below this value at unchanged volatility, the market crosses the {lvl_txt}")
        if vol_needed is not None:
            out.metric(f"Volatility threshold for the {lvl_txt}", vol_needed, "pct", digits=0,
                       hint=f"at unchanged α ({_num(a_now_s)})")
    out.metric("Historical peak", float(prob[i_max]), "pct", digits=1,
               hint=(f"on {_date(dates[i_max])} · volatility {_pct(vol[i_max])}, α {_num(alpha[i_max])}"
                     if prob[i_max] > 0.001 else "the trajectory never leaves the plateau"))
    above15 = float(np.mean(prob >= 0.15))
    out.metric("Days beyond the 15% contour", above15, "pct", digits=1,
               hint="share of all evaluated days on the cliff or above")

    # --- verdict ------------------------------------------------------------------------
    a_txt = _num(a_now) if abs(a_now - a_now_s) < 0.005 else f"{_num(a_now)} ({_num(a_now_s)} on the surface)"
    where = (f"At {_pct(v_now)} volatility and α = {a_txt}, the surface shows {_pct(p_now)} for "
             f"a one-day loss beyond {thr_txt} within {horizon} trading days.")
    if next_level is None:
        reach = " This is the highest zone of the surface."
    else:
        parts = []
        if vol_needed is not None:
            parts.append(f"at unchanged α once volatility reaches {_pct(vol_needed, 0)}")
        if alpha_needed is not None and alpha_needed < a_now_s - 1e-9:
            parts.append(f"at unchanged volatility if α falls to {_num(alpha_needed)}")
        contour = f"{round(next_level * 100)}% contour"
        reach = (f" The market reaches the {contour} " + " or ".join(parts) + "."
                 if parts else f" From here, the {contour} can only be reached through volatility and α together.")
    out.set_verdict(zstatus, f"{zname}: {zaction[0].lower()}{zaction[1:]}", where + reach)

    # --- chart 1: the surface with the trajectory -----------------------------------------
    keep = _thin(prob, MAX_PATH)
    k_recent = int(min(max(2, np.ceil(params["recent"] / stride)), len(prob)))
    paths = [C.path3d(f"Trajectory of {data.name or 'the dataset'} (since {_date(dates[0])})",
                      vol_s[keep], alpha_s[keep], prob[keep] + Z_LIFT, color=1,
                      text=[f"{_date(dates[i])} · {_pct(prob[i])}" for i in keep])]
    rec = slice(len(prob) - k_recent, len(prob))
    paths.append(C.path3d(f"Last {params['recent']} days", vol_s[rec], alpha_s[rec], prob[rec] + Z_LIFT,
                          color="ink", text=[f"{_date(d)} · {_pct(p)}" for d, p in zip(dates[rec], prob[rec])]))
    if params["script_paths"]:
        for k, (name, crisis) in enumerate(surface_script.CRISES.items()):
            cz = surface_script.crash_probability(crisis["vol"], crisis["alpha"], threshold=thr, horizon_days=horizon)
            paths.append(C.path3d(f"Sketch: {name}", crisis["vol"], crisis["alpha"], cz + Z_LIFT, color=3 + k,
                                  mode="lines+markers", size=3))
    points = [
        C.path3d("Today: drop line to the floor", [vol_s[-1], vol_s[-1]], [alpha_s[-1], alpha_s[-1]],
                 [0.0, p_now + Z_LIFT], color="muted"),
        C.path3d(f"Today ({_date(dates[-1])})", [vol_s[-1]], [alpha_s[-1]], [p_now + Z_LIFT], color="ink",
                 mode="markers", size=8, text=[f"{zname} · {_pct(p_now)}"]),
    ]
    out.chart(C.surface(
        f"Crash probability across volatility and tail index (threshold {thr_txt}, {horizon} days)",
        vol_axis, alpha_axis, risk_grid,
        # both axes reversed: the calm corner (low vol, α = 2) faces the viewer, the cliff rises behind it
        x=C.axis("Volatility p.a.", fmt="pct", digits=0, range=(v_hi, v_lo)),
        y=C.axis("Tail index α", range=(a_hi, a_lo)),
        z=C.axis(f"P(loss > {thr_txt})", fmt="pct", digits=0, range=(0, 1.05)),
        paths=paths, points=points, scale="risk", height=600,
        # chosen by comparing views in the browser: the slope reads plateau -> ramp -> cliff -> peak
        camera={"eye": {"x": 1.2, "y": 1.7, "z": 1.0}},
        subtitle=f"Green plateau → yellow ramp → red cliff → violet black-swan peak. Blue: the trajectory "
                 f"({est_name} α and volatility over {w} days); dark: the most recent days and today.",
    ))

    # --- chart 2: crash probability over time, with the real crash days ----------------
    loss = -np.expm1(r)
    events = np.flatnonzero(loss > thr)
    prev_prob = np.interp(events - 1, day, prob, left=np.nan)       # what the surface showed the day before
    ev_ok = np.isfinite(prev_prob) & (events - 1 >= day[0])
    ev_x = [data.dates[i] for i in events[ev_ok]] if data.dates else list(events[ev_ok] + 1)
    traces = [C.trace("Crash probability", x_time, prob, color=1, width=1.5)]
    if ev_ok.any():
        traces.append(C.trace(f"Day with loss > {thr_txt} (value on the previous day)", ev_x, prev_prob[ev_ok],
                              type="scatter", color="critical", size=7, decimate_to=None))
    out.chart(C.xy(
        f"Crash probability over time ({horizon} days)",
        traces, x=x_axis, y=C.axis("Probability", fmt="pct", digits=0, range=(0, 1.02)),
        hlines=[C.hline(lvl, _pct(lvl, 0), color="muted") for lvl, *_ in CONTOURS], height=390,
        subtitle="Red dots: actual crash days, plotted at the value the surface showed on the previous day.",
    ))

    # --- charts 3 + 4: the two coordinates --------------------------------------------
    out.chart(C.xy(
        f"Volatility p.a. (rolling, {w} days)",
        [C.trace("Volatility", x_time, vol, color=1, width=1.5)],
        x=x_axis, y=C.axis("Volatility p.a.", fmt="pct", digits=0), height=280,
        hlines=[C.hline(v_hi, "End of surface axis")] if np.nanmax(vol) > v_hi else [],
    ))
    a_top = float(min(8.0, max(2.6, np.nanpercentile(alpha, 99) * 1.08)))
    out.chart(C.xy(
        f"Tail index α ({est_name}, {w} days)",
        [C.trace("α", x_time, alpha, color=1, width=1.5)],
        x=x_axis, y=C.axis("α", range=(min(0.8, float(np.nanmin(alpha)) * 0.95), a_top)), height=280,
        hlines=[C.hline(2.0, "Gaussian edge α = 2"), C.hline(a_lo, "End of axis")],
        subtitle=(f"On {_pct(at_edge, 0)} of days α is above 2 – the surface shows 0% there."
                  if at_edge > 0.05 else "Lower = fatter tails."),
    ))

    # --- tables ---------------------------------------------------------------------
    rows = []
    for lvl, signal, action in CONTOURS:
        above = prob >= lvl
        last = _date(dates[int(np.flatnonzero(above)[-1])]) if above.any() else "never"
        rows.append([_pct(lvl, 0), signal, action, "yes" if p_now >= lvl else "no", float(above.mean()), last])
    out.table("How to read the floor contours", ["Contour", "Signal", "Response", "Crossed today",
                                                 "Share of days above", "Last above"],
              rows, formats=["text", "text", "text", "text", "pct", "text"],
              note="Reading guide from the README and the script: each contour crossed escalates the response.")

    # calibration: what really happened in the `horizon` days after each evaluated day
    csum = np.concatenate([[0], np.cumsum(loss > thr)])
    has_future = day + horizon < n
    realized = (csum[np.minimum(day + 1 + horizon, n)] - csum[day + 1]) > 0
    zone_idx = np.array([_zone(p)[0] for p in prob])
    rows, zone_stats = [], []
    for i, (hi, name, status, region, action) in enumerate(ZONES):
        lo = ZONES[i - 1][0] if i else 0.0
        band = f"{100 * lo:.0f}⁠–⁠{_pct(hi, 0)}" if np.isfinite(hi) else f"≥ {_pct(lo, 0)}"
        band = band.replace(" ", " ")  # keep "15–50%" on one line: no break at the space or the dash
        mk = zone_idx == i
        mf = mk & has_future
        rows.append([f"{name} ({region})", band, action, float(mk.mean()),
                     float(prob[mf].mean()) if mf.any() else None,
                     float(realized[mf].mean()) if mf.any() else None, int(mf.sum())])
        if mf.sum() >= 20:                                # enough days to compare model and outcome
            zone_stats.append((name, float(prob[mf].mean()), float(realized[mf].mean())))
    out.table("Surface zones – and what actually happened next", [
        "Zone", "Range", "Action (README)", "Share of days",
        "Avg. surface", f"Actual: loss > {thr_txt} within {horizon} days", "Days"],
        rows, formats=["text", "text", "text", "pct", "pct", "pct", "int"],
        note=f"‘Actual’: share of days that were followed by a one-day loss beyond {thr_txt} within {horizon} "
             "trading days. Well below ‘Avg. surface’ means the model overestimates; above it, the model "
             "underestimates. The periods overlap, so the days are not independent.")

    # --- notes ------------------------------------------------------------------------
    pred, real = float(prob[has_future].mean()) if has_future.any() else np.nan, \
        float(realized[has_future].mean()) if has_future.any() else np.nan
    if estimator == "hill" and at_edge > 0.05:
        out.note(f"On **{_pct(at_edge, 0)}** of days the Hill estimate is above 2 – finite variance, typical of "
                 "equity indices. These days sit on the Gaussian edge α = 2, where the surface's power-law term "
                 "vanishes (0%), even at high volatility. The trajectory only climbs onto the surface when individual "
                 "days in the window stand far out from the rest – but then it climbs steeply.")
    if estimator == "stable":
        out.note("The McCulloch estimator reads α from quantile ratios. For daily returns it almost always lands "
                 "below 2, so the trajectory clings to the cliff. Compare ‘Avg. surface’ in the table with the "
                 "actual frequency.", "warning")
    if np.isfinite(pred) and np.isfinite(real):
        if max(pred, real) < 0.005:
            verdict = "both are practically zero"
        else:
            ratio = pred / real if real > 0 else np.inf
            verdict = ("the model clearly overestimates the frequency" if ratio > 2 else
                       "the model clearly underestimates the frequency" if ratio < 0.5 else
                       "on average the order of magnitude is right")
        # the overall mean can hide opposite errors per zone (too calm on the plateau, too alarmed on the cliff)
        parts = []
        for name, p, q in zone_stats:
            if q > 2 * p and q - p > 0.01:
                parts.append(f"‘{name}’ avg. {_pct(p)} vs actual {_pct(q)} (underestimated)")
            elif p > 2 * q and p - q > 0.05:
                parts.append(f"‘{name}’ avg. {_pct(p)} vs actual {_pct(q)} (overestimated)")
        zone_txt = (" By zone, the errors diverge widely: " + "; ".join(parts) + "." if parts else
                    " The order of magnitude also holds zone by zone." if len(zone_stats) > 1 else "")
        seq = [q for _, _, q in zone_stats]
        if len(seq) > 1:
            zone_txt += (" The ranking holds: the higher the zone, the more often a crash day followed."
                         if all(b >= a for a, b in zip(seq, seq[1:])) else
                         " The ranking is not consistent: days in a higher zone were not always followed by a "
                         "crash day more often.")
        out.note(f"Calibration over the full history: on average the surface shows **{_pct(pred)}**; in practice a "
                 f"one-day loss beyond {thr_txt} followed within {horizon} days in **{_pct(real)}** of cases – "
                 f"{verdict}.{zone_txt} Read the surface as a picture of the mechanism (volatility and fat tails "
                 "drive crash risk), not as a calibrated forecast. Values outside the axes "
                 f"(volatility {_pct(v_lo, 0)}–{_pct(v_hi, 0)}, α {_num(a_lo)}–{_num(a_hi)}) sit on the edge."
                 + (f" Long series: one day in {stride} is evaluated." if stride > 1 else ""))


def _thin(prob, max_points):
    """Indices of a thinned path that keeps every bucket's peak (crises must not be thinned away)."""
    n = len(prob)
    if n <= max_points:
        return np.arange(n)
    idx, _ = C.decimate(list(range(n)), prob, max_points)
    idx = np.asarray(idx, dtype=int)
    return np.unique(np.concatenate([idx, [0, n - 1]]))


def _date(d):
    """'2026-10-15' -> '15 Oct 2026' (non-breaking spaces: one line in table cells); anything else unchanged."""
    return fmt_date(d).replace(" ", " ")
