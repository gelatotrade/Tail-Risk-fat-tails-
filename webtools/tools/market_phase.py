"""Market phase: rule-based classifier (STABLE/TRENDING/CRITICAL/CRASH/RECOVERY) and phase-space trajectory."""

import numpy as np

from src.physics.phase_transitions import MarketPhaseClassifier, compute_phase_space_coordinates

from .. import charts as C
from ..core import ToolError
from ..data import index_axis, fmt_date

SPEC = {
    "id": "market-phase",
    "title": "Market phase: stable, trending, critical, crash",
    "group": "regime",
    "order": 20,
    "summary": "Identifies the market's current phase with five transparent rules and shows the phase timeline and "
               "the market's path through a 3D phase space.",
    "description": (
        "The classifier translates the physics of phase transitions into five simple rules, checked in order: "
        "**Crash** (volatility in the top 5% of its history and an average return below −2% per day), "
        "**Critical** (the early-warning signals autocorrelation and variance are both rising markedly), "
        "**Recovery** (volatility in the top decile, but rising prices), **Trending** (strong autocorrelation and "
        "more than 1% movement per day) – otherwise **Stable**. The table shows the current values for each rule.\n\n"
        "How to read the charts: the timeline classifies each cut-off date using only the data available at the "
        "time. The phase space shows three coordinates per day: **X** is a variance ratio (‘susceptibility’: "
        "variance of the cumulative price within the window ÷ variance of daily returns – large when the daily "
        "moves add up to a move in one direction), **Y** is the absolute autocorrelation of daily returns (order "
        "parameter, herding) and **Z** = 1/(X + 0.1) is a measure of the distance to criticality. All three are "
        "normalised to their maximum.\n\n"
        "Limitations: Z is a fixed function of X, so the path lies on a curved surface – the actual information is "
        "in X and Y. The crash rule is very strict (−2% per day across the whole window) and in practice only "
        "triggers in historical extremes; ‘Stable’ is the residual category. The AC(1) of the detrended log price "
        "is almost always above 0.2 (even a random walk reaches about 0.9), so the trend rule depends almost "
        "entirely on the average return. The EWS warning level tests the trend over the **entire** series, not "
        "just over the most recent days as the ‘Critical slowing down’ tool does."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 20, "max": 200, "step": 5,
         "label": "Window", "unit": "days",
         "help": "For volatility, average return and phase space. The EWS warning level always uses 50 days "
                 "internally."},
        {"name": "timeline", "type": "int", "default": 12, "min": 0, "max": 60, "step": 1,
         "label": "Cut-off dates in the timeline",
         "help": "0 = off. Each cut-off date requires a full classification."},
    ],
    "covers": [
        "src.physics.phase_transitions.MarketPhaseClassifier",
        "src.physics.phase_transitions.MarketPhaseClassifier.classify",
        "src.physics.phase_transitions.compute_phase_space_coordinates",
    ],
    "cost": "medium",
}

PHASES = {
    "STABLE": ("Stable", "good"),
    "TRENDING": ("Trending", 1),
    "RECOVERY": ("Recovery", 2),
    "CRITICAL": ("Critical", "serious"),
    "CRASH": ("Crash", "critical"),
}
MIN_EXTRA = 20  # days of phase-space coordinates beyond the 2·window warm-up
TIMELINE_OBS = 40_000  # sum of history lengths classified for the timeline (bounded runtime)
EWS_LABELS = {"NORMAL": "Normal", "MODERATE": "Moderate", "ELEVATED": "Elevated", "CRITICAL": "Critical"}

def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma."""
    return f"{x:,.{digits}f}".replace("-", "−")


def _ordinal(k):
    k = int(k)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _pctl(p):
    """Percentile (0..100) as an ordinal, rounded like the displayed numbers: '96th percentile'."""
    return f"{_ordinal(f'{p:.0f}')} percentile"


def _date(d):
    """ISO date -> '15 Oct 2026' for hover texts; trading-day indices become 'Day n'."""
    return fmt_date(d, "Day {}")


def run(params, data, out):
    r = data.returns
    n, w = data.n, params["window"]
    if n < 2 * w + MIN_EXTRA:
        raise ToolError(f"With a window of {w} days the classifier needs at least {2 * w + MIN_EXTRA:,} "
                        f"observations (available: {n:,}). Reduce the window to at most "
                        f"{(n - MIN_EXTRA) // 2:,} days.")

    clf = MarketPhaseClassifier()
    res = clf.classify(r, window=w)
    phase = res["phase"]
    name, color = PHASES[phase]
    vol_pct = float(res["volatility_percentile"])
    recent = r[-w:]
    mean_ret, vol_ann = float(np.mean(recent)), float(np.std(recent) * np.sqrt(252))
    ac1 = float(res["autocorrelation"])
    ews_level = res["ews"]["warning_level"]

    status = color if isinstance(color, str) else "warning"
    out.metric("Current phase", name, "text", status=status)
    out.metric("Confidence", res["confidence"], "pct", digits=0,
               hint="fixed rule formula, not statistical certainty – see table")
    out.metric("Volatility percentile", vol_pct / 100, "pct", digits=0,
               hint=f"volatility of the last {w} days compared with the full history",
               status="serious" if vol_pct > 95 else "warning" if vol_pct > 90 else None)
    out.metric(f"Volatility p.a. ({w} days)", vol_ann, "pct", digits=1)
    out.metric(f"Avg. daily return ({w} days)", mean_ret, "pct", digits=3)
    out.metric("AC(1) of the log price", ac1, "num", hint="latest EWS value (trend rule: > 0.2)")
    out.metric("Dispersion around the trend", np.sqrt(res["susceptibility"]), "pct",
               hint="√ EWS variance of the detrended log price (‘susceptibility’ in the classifier)")
    out.metric("EWS warning level", EWS_LABELS.get(ews_level, ews_level), "text",
               hint="Kendall τ trend of AC(1) and variance over the full series")

    def yes(c):
        return "yes" if c else "no"

    pct_txt, ret_txt = _pctl(vol_pct), f"{_num(100 * mean_ret, 2)}%/day"
    rules = [
        ("Crash", "Vol percentile > 95 AND avg. return < −2%/day", f"{pct_txt} · {ret_txt}",
         vol_pct > 95 and mean_ret < -0.02, "Vol percentile"),
        ("Critical", "EWS warning level = Critical", EWS_LABELS.get(ews_level, ews_level), ews_level == "CRITICAL",
         "80% (fixed)"),
        ("Recovery", "Vol percentile > 90 AND avg. return > 0", f"{pct_txt} · {ret_txt}",
         vol_pct > 90 and mean_ret > 0, "70% (fixed)"),
        ("Trending", "AC(1) > 0.2 AND |avg. return| > 1%/day", f"AC(1) {_num(ac1)} · {ret_txt}",
         ac1 > 0.2 and abs(mean_ret) > 0.01, "AC(1)"),
        ("Stable", "none of the rules above", "–", True, "100% − vol percentile"),
    ]
    rows, hit = [], False
    for label, rule, value, cond, conf_rule in rules:
        mark = yes(cond)
        if cond and not hit:
            mark, hit = "yes → result", True
        rows.append([label, rule, value, mark, conf_rule])
    out.table("Rules (the first rule met applies)", ["Phase", "Rule", "Current", "Met", "Confidence"], rows,
              formats=["text", "text", "text", "text", "text"],
              note=f"Window {w} days. The EWS warning level always uses 50 days and σ = 100 internally and tests "
                   "the trend over the full series.")

    x, xa = index_axis(data)

    # Timeline: classify each cut-off date with the data available then
    k = params["timeline"]
    if k > 0:
        start = max(2 * w + MIN_EXTRA, 250 if n > 500 else 2 * w + MIN_EXTRA)
        # each cut-off re-classifies its whole history: bound the total work for long series
        k_eff = int(max(3, min(k, TIMELINE_OBS // max(1, (start + n) // 2))))
        ends = np.linspace(start, n, min(k, k_eff)).astype(int)
        # add the day with the highest rolling volatility: what did the rules say at peak stress?
        roll = np.std(np.lib.stride_tricks.sliding_window_view(r, w), axis=1)
        peak = int(np.argmax(roll)) + w
        if peak >= start:
            ends = np.append(ends, peak)
        ends = np.unique(ends)
        points = []
        for i, end in enumerate(ends):
            out.progress(f"Timeline: cut-off date {i + 1} of {len(ends)} …", (i + 1) / (len(ends) + 2))
            c = clf.classify(r[:end], window=w)
            points.append((end - 1, c["phase"], float(c["volatility_percentile"]) / 100, float(c["confidence"])))
        traces = [C.trace("Vol percentile", [x[p[0]] for p in points], [p[2] for p in points], color="muted", width=1)]
        for key, (label, col) in PHASES.items():
            sel = [p for p in points if p[1] == key]
            if sel:
                traces.append(C.trace(label, [x[p[0]] for p in sel], [p[2] for p in sel], type="scatter",
                                      color=col, size=11, decimate_to=None,
                                      text=[f"{label} · confidence {_num(100 * p[3], 0)}%" for p in sel]))
        stressed = [p for p in points if p[1] == "STABLE" and p[2] > 0.9]
        if stressed:
            p0 = max(stressed, key=lambda p: p[2])
            out.note(f"On {_date(x[p0[0]])} volatility was in the {_pctl(100 * p0[2])} – yet the classifier "
                     f"reports ‘Stable’ (confidence {_num(100 * p0[3], 0)}%). The crash rule requires an average "
                     "of −2% per day and the recovery rule rising prices; if neither applies, only the residual "
                     "category remains.", "warning")
        out.chart(C.xy(
            "Phase timeline",
            traces, x=xa, y=C.axis("Volatility percentile", fmt="pct", digits=0, range=(-0.03, 1.1)),
            hlines=[C.hline(0.9), C.hline(0.95, "90% / 95%")], height=390,
            subtitle="Each point is a classification using only the data up to that day – including the day with "
                     "the highest volatility."
                     + (f" Limited to {k_eff} cut-off dates because of the long series." if k_eff < k else ""),
        ))

    # Phase space
    X, Y, Z = compute_phase_space_coordinates(r, window=w)
    valid = np.arange(2 * w, n)
    labels = [_date(x[i]) for i in valid]
    recent_n = min(250, len(valid) // 3)
    traces3d = [
        C.path3d("History", X[valid], Y[valid], Z[valid], color="muted", text=labels),
        C.path3d(f"Last {recent_n} days", X[valid[-recent_n:]], Y[valid[-recent_n:]], Z[valid[-recent_n:]],
                 color=1, mode="lines+markers", size=2, text=labels[-recent_n:]),
        C.path3d("Today", [X[-1]], [Y[-1]], [Z[-1]], color=2, mode="markers", size=7, text=[labels[-1]]),
    ]
    xs = [x[i] for i in valid]
    out.chart(C.xy(
        "Phase space X: susceptibility",
        [C.trace("X", xs, X[valid], color=1, width=1.5)],
        x=xa, y=C.axis("X, normalised to its maximum", range=(0, 1.02)), height=260,
        subtitle=f"Rolling over {w} days. Peaks: the daily moves add up to a trend or a sell-off. "
                 "Z = 1/(X + 0.1) is a fixed function of X and is therefore not shown here.",
    ))
    out.chart(C.xy(
        "Phase space Y: order |AC(1)|",
        [C.trace("Y", xs, Y[valid], color=1, width=1)],
        x=xa, y=C.axis("Y, normalised to its maximum", range=(0, 1.02)), height=260,
        subtitle="Absolute lag-1 autocorrelation of daily returns. High: days depend on one another "
                 "(herding or rapid reversal).",
    ))

    out.chart(C.scatter3d(
        "Phase-space trajectory of the market", traces3d,
        x=C.axis("X: susceptibility"), y=C.axis("Y: order |AC(1)|"), z=C.axis("Z: distance to criticality"),
        subtitle="Close to critical states: X large, Z small. Drag to rotate, scroll to zoom.",
    ))
    if phase == "STABLE" and vol_pct > 90:
        out.note(f"‘Stable’ is only the residual category here: volatility is in the {_pctl(vol_pct)}, but "
                 "neither the crash rule nor the recovery rule applies (the average return is ≤ 0 or not below "
                 "−2% per day).", "warning")
        out.set_verdict("warning", "Stable under the residual rule – but unusually high volatility",
                        "The rules only detect stress when average returns are extreme.")
        return
    texts = {
        "STABLE": ("good", "Stable phase",
                   f"Volatility in the {_pctl(vol_pct)}; no rule for trend, criticality or crash applies."),
        "TRENDING": ("warning", "Trending phase: herding",
                     "Strong autocorrelation and large daily moves in one direction."),
        "RECOVERY": ("warning", "Recovery after stress",
                     "Volatility in the top decile, but prices are rising on average."),
        "CRITICAL": ("serious", "Critical phase: early-warning signals rising",
                     "Autocorrelation and variance are both rising significantly – typical ahead of regime switches."),
        "CRASH": ("critical", "Crash phase", "Extreme volatility with sharply falling prices."),
    }
    level, title, text = texts[phase]
    out.set_verdict(level, title, text)
