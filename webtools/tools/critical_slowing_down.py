"""Critical slowing down: rolling early-warning signals (AC(1), variance, skewness) with Kendall-tau trends."""

import numpy as np

from src.physics.phase_transitions import CriticalSlowingDownDetector

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "critical-slowing-down",
    "title": "Critical slowing down: AC(1) and variance",
    "group": "regime",
    "order": 30,
    "summary": "Tests whether autocorrelation and variance are rising systematically – the classic precursor of a "
               "tipping point – with a Kendall τ trend test and a warning level.",
    "description": (
        "Shortly before a tipping point, a system recovers from small disturbances ever more slowly. This shows up "
        "as **rising autocorrelation AC(1)** (the market ‘remembers’ yesterday for longer) and **rising "
        "variance** (the swings become larger). To measure this, the detector sums the returns into a cumulative log "
        "price, subtracts a Gaussian-smoothed trend in each rolling window and measures AC(1) and variance of the "
        "residual. Skewness is taken from the returns themselves.\n\n"
        "How to read the charts: what matters is not the level but the **trend within the analysis window** "
        "(shaded grey). Even a pure random walk has a high AC(1) here – the dashed line shows this reference "
        "level. Kendall's τ measures the trend: +1 means ‘rises consistently’, 0 means ‘no trend’. The warning "
        "level follows from it: Critical when AC(1) **and** variance are both rising markedly (τ > 0.3, "
        "p < 0.05). The composite indicator is the mean of the z-scores of AC(1) and variance over the full "
        "series.\n\n"
        "Parameters: the **window** sets how many days enter each measurement (shorter = faster but noisier). "
        "**σ** is the width of the Gaussian trend; if σ is larger than the window, effectively only the mean is "
        "subtracted (the default σ = 100 matches the classifier in the ‘Market phase’ tool). The **analysis "
        "window** sets the most recent days over which the trend is tested."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 50, "min": 20, "max": 250, "step": 5,
         "label": "Rolling window", "unit": "days"},
        {"name": "detrend_window", "type": "int", "default": 100, "min": 5, "max": 200, "step": 5,
         "label": "Trend smoothing width σ", "unit": "days",
         "help": "Gaussian filter used to detrend each window."},
        {"name": "lookback", "type": "int", "default": 500, "min": 100, "max": 20000, "step": 50,
         "label": "Analysis window", "unit": "days",
         "help": "The most recent trading days over which Kendall's τ tests whether the signals are rising. "
                 "Larger than the dataset = all data."},
    ],
    "covers": [
        "src.physics.phase_transitions.CriticalSlowingDownDetector",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.detrend",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.autocorrelation_lag1",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.rolling_variance",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.rolling_skewness",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.kendall_tau",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.compute_ews",
        "src.physics.phase_transitions.CriticalSlowingDownDetector._classify_warning",
    ],
    "cost": "fast",
}

LEVELS = {
    "NORMAL": ("good", "Normal", "No critical slowing down detected"),
    "MODERATE": ("warning", "Moderate", "Slight upward trend in the early-warning signals"),
    "ELEVATED": ("serious", "Elevated", "Elevated warning level: one signal is rising markedly"),
    "CRITICAL": ("critical", "Critical", "Critical slowing down: AC(1) and variance are rising together"),
}
# The detector detrends each window with a (window x window) kernel matrix: cost ~ observations x window².
MAX_WORK = 6e9  # ~6 s native at the limit (two passes plus the reference walk), bounded runtime in the browser
N_REFERENCE = 1000  # windows of the random-walk reference for AC(1)


def _num(x, digits=2):
    """Number inside text: decimal point, thousands comma."""
    return f"{x:,.{digits}f}".replace("-", "−")


def _p(p):
    """'p < 0.001' or 'p = 0.076'."""
    return "p < 0.001" if p < 0.001 else f"p = {_num(p, 3)}"


def _padded(a, w):
    out = np.asarray(a, dtype=float).copy()
    out[:w] = np.nan  # no full window yet
    return out


def run(params, data, out):
    r = data.returns
    n, w, sigma = data.n, params["window"], params["detrend_window"]
    look = min(params["lookback"], n)
    if look - w < 30:
        if look == n:  # the dataset itself is the limit
            raise ToolError(f"The dataset has only {n:,} days – with a window of {w} days too few data points remain "
                            f"for the trend test. Reduce the window to at most {n - 30:,} days.")
        raise ToolError(f"The analysis window ({look:,} days) is too short for a window of {w} days. "
                        "Reduce the window or lengthen the analysis window.")
    if (n + look + N_REFERENCE + w) * w * w > MAX_WORK:
        raise ToolError(f"A window of {w} days is too computationally expensive for {n:,} observations. "
                        "Reduce the window or the analysis window.")

    det = CriticalSlowingDownDetector(window=w, detrend_window=sigma)
    full = det.compute_ews(r)                                   # charts: the whole history
    ews = det.compute_ews(r[-look:]) if look < n else full      # trend test: the analysis window
    level = ews["warning_level"]
    status, level_label, title = LEVELS[level]
    ac_tau, ac_p = ews["ac1_trend"]["tau"], ews["ac1_trend"]["p_value"]
    var_tau, var_p = ews["var_trend"]["tau"], ews["var_trend"]["p_value"]
    # Reference: AC(1) of a pure random walk (i.i.d. returns) with the same window and σ
    walk = np.cumsum(np.random.default_rng(0).standard_normal(N_REFERENCE + w))
    ac_walk = float(np.median(det.autocorrelation_lag1(walk)[w:]))

    out.metric("Warning level", level_label, "text", status=status,
               hint="from the Kendall τ trends in the analysis window")
    out.metric("Kendall τ of AC(1)", ac_tau, "num", hint=_p(ac_p),
               status="warning" if ac_tau > 0.2 and ac_p < 0.1 else None)
    out.metric("Kendall τ of variance", var_tau, "num", hint=_p(var_p),
               status="warning" if var_tau > 0.2 and var_p < 0.1 else None)
    out.metric("Current AC(1)", full["autocorrelation"][-1], "num",
               hint=f"of the detrended log price · pure random walk: ≈ {_num(ac_walk)}")
    out.metric("Current dispersion", np.sqrt(full["variance"][-1]), "pct",
               hint="√variance of the detrended log price in the latest window")
    out.metric("Current skewness", full["skewness"][-1], "num", hint="of daily returns in the latest window")
    comp_now = full["composite"][-1]
    out.metric("Composite indicator", comp_now, "num", hint="z-score, mean of AC(1) and variance",
               status="serious" if comp_now > 2 else "warning" if comp_now > 1 else None)

    x, xa = index_axis(data)
    bands = [C.band(x[n - look], x[-1], "Analysis window", color="muted")] if look < n else []
    meta = data.meta or {}
    if "crisis_start" in meta and "crisis_end" in meta and not data.dates:
        bands.append(C.band(int(meta["crisis_start"]), int(meta["crisis_end"]), "Crisis (synthetic)"))

    level_path = np.cumsum(r)
    out.chart(C.xy(
        "State: cumulative log return",
        [C.trace("Cumulative log return", x, level_path, color=1, width=1.5)],
        x=xa, y=C.axis("Cumulative log return", fmt="pct", digits=0), bands=bands, height=280,
        subtitle="The detector works on this log price (skewness: on daily returns).",
    ))
    # What the detector subtracts in its most recent window (entry i uses the days i-w .. i-1)
    last = level_path[n - 1 - w:n - 1]
    trend = last - det.detrend(last)
    flat = sigma >= w
    out.chart(C.xy(
        f"Detrending in the latest window ({w} days)",
        [C.trace("Cumulative log return", x[n - 1 - w:n - 1], last, color=1, width=2),
         C.trace(f"Subtracted Gaussian trend (σ = {sigma} days)", x[n - 1 - w:n - 1], trend, color=2, dash=True)],
        x=xa, y=C.axis("Cumulative log return", fmt="pct", digits=1), height=280,
        subtitle=("σ is not smaller than the window: the trend is practically flat and only the mean is "
                  "subtracted. AC(1) and variance then measure how the price wanders within the window."
                  if flat else
                  "AC(1) and variance are measured on the gap between price and trend – recomputed in every window."),
    ))
    out.chart(C.xy(
        f"Autocorrelation AC(1) – rolling {w}-day window",
        [C.trace("AC(1)", x, _padded(full["autocorrelation"], w), color=1, width=1.5)],
        x=xa, y=C.axis("AC(1)"), bands=bands, height=280,
        hlines=[C.hline(ac_walk, "pure random walk")],
        subtitle=f"Kendall τ in the analysis window: {_num(ac_tau)} ({_p(ac_p)}). "
                 "Rising AC(1) means the market recovers more slowly.",
    ))
    out.chart(C.xy(
        f"Dispersion around the trend – rolling {w}-day window",
        [C.trace("√Variance", x, np.sqrt(_padded(full["variance"], w)), color=1, width=1.5)],
        x=xa, y=C.axis("Dispersion of the log price", fmt="pct", digits=1), bands=bands, height=280,
        subtitle=f"Kendall τ of the variance in the analysis window: {_num(var_tau)} ({_p(var_p)}). "
                 "Rising dispersion: the swings are becoming larger.",
    ))
    out.chart(C.xy(
        f"Skewness of daily returns – rolling {w}-day window",
        [C.trace("Skewness", x, _padded(full["skewness"], w), color=1, width=1.5)],
        x=xa, y=C.axis("Skewness"), bands=bands, hlines=[C.hline(0.0)], height=280,
        subtitle="Strongly negative values: individual crash days dominate the window.",
    ))
    out.chart(C.xy(
        "Composite indicator (z-score)",
        [C.trace("½ z(AC1) + ½ z(variance)", x, _padded(full["composite"], w), color=1, width=1.5)],
        x=xa, y=C.axis("z-score"), bands=bands, height=280,
        hlines=[C.hline(0.0), C.hline(2.0, "+2", color="warning")],
        subtitle="Above +2: AC(1) and variance are both well above their mean. Mean and standard deviation come "
                 "from the full series – earlier values are therefore normalised with later data.",
    ))

    def ok(cond):
        return "yes" if cond else "no"

    rules = [
        ("Critical", "τ(AC1) > 0.3 with p < 0.05 AND τ(Var) > 0.3 with p < 0.05",
         ac_tau > 0.3 and ac_p < 0.05 and var_tau > 0.3 and var_p < 0.05),
        ("Elevated", "τ(AC1) > 0.2 with p < 0.1 OR τ(Var) > 0.2 with p < 0.1",
         (ac_tau > 0.2 and ac_p < 0.1) or (var_tau > 0.2 and var_p < 0.1)),
        ("Moderate", "τ(AC1) > 0.1 OR τ(Var) > 0.1", ac_tau > 0.1 or var_tau > 0.1),
        ("Normal", "none of the rules above", True),
    ]
    rows, hit = [], False
    for name, rule, cond in rules:
        mark = ok(cond)
        if cond and not hit:
            mark, hit = "yes → result", True
        rows.append([name, rule, mark])
    out.table("Warning-level rules (the first rule met applies)", ["Level", "Rule", "Met"], rows,
              formats=["text", "text", "text"],
              note=f"Analysis window: last {look:,} trading days, {look - w:,} of them with a full window.")
    out.note("The p-values are too optimistic: adjacent rolling windows overlap almost completely, so the data "
             "points are not independent. Focus mainly on the size of τ and on whether AC(1) and variance rise "
             "together.")

    detail = (f"Kendall τ in the analysis window: AC(1) {_num(ac_tau)}, variance {_num(var_tau)} "
              f"(last {look:,} trading days).")
    out.set_verdict(status, title, detail)
