"""Power-law tail: Clauset-style x_min / exponent fit, log-log CCDF and which moments exist."""

import numpy as np
from scipy import stats

from src.physics.phase_transitions import PowerLawAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "power-law-tail",
    "title": "Power-law tail exponent",
    "group": "data",
    "order": 40,
    "summary": "How quickly the probability of large moves decays – the tail exponent, where the tail begins and "
               "which moments (variance, kurtosis) exist at all.",
    "description": (
        "Above a threshold x_min, large daily moves often follow a **power law**: the density falls like x^−α "
        "and the probability of a day beyond x like x^−(α−1). The **tail index α − 1** is the decisive figure: a "
        "moment k (variance k = 2, skewness 3, kurtosis 4) exists only if α − 1 > k. For equities α − 1 is "
        "typically around 3 (the ‘cubic law’) – the variance exists, the kurtosis does not.\n\n"
        "How to read the main chart: on log-log axes a power law becomes a **straight line**. The points show how "
        "often a day was at least that large; the line is the fit from x_min upwards, and the dashed curve is the "
        "normal distribution with the same volatility. Where the points lie above the Gaussian curve, the normal "
        "distribution underestimates the risk – often by orders of magnitude.\n\n"
        "Method: **Clauset** chooses x_min to minimise the distance (KS distance) between data and fit, and "
        "estimates α by maximum likelihood. **Fixed threshold** simply takes the largest 10%. The bootstrap p-value "
        "tests whether a genuine power law would fit this well (simplified: x_min is not re-optimised for each "
        "sample, so the p-value tends to be too high)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "tail", "type": "select", "default": "both", "label": "Tail",
         "options": [{"value": "both", "label": "|Return| – both tails"},
                     {"value": "left", "label": "Losses (left tail)"},
                     {"value": "right", "label": "Gains (right tail)"}]},
        {"name": "method", "type": "select", "default": "clauset", "label": "Threshold x_min",
         "options": [{"value": "clauset", "label": "Clauset (KS-optimal)"},
                     {"value": "fixed", "label": "Fixed: largest 10%"}]},
        {"name": "n_bootstrap", "type": "int", "default": 100, "min": 20, "max": 1000, "step": 20,
         "label": "Bootstrap samples", "help": "For the p-value of the power-law test."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed",
         "help": "Used for the bootstrap only."},
    ],
    "covers": [
        "src.physics.phase_transitions.PowerLawAnalyzer",
        "src.physics.phase_transitions.PowerLawAnalyzer.fit_power_law",
        "src.physics.phase_transitions.PowerLawAnalyzer._bootstrap_pvalue",
    ],
    "cost": "fast",
}

SIDES = {"left": "Losses", "right": "Gains", "both": "|Return|"}
MOMENTS = [(1, "Mean"), (2, "Variance"), (3, "Skewness"), (4, "Kurtosis")]
LEVELS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _side_values(r, side):
    if side == "left":
        return -r[r < 0]
    if side == "right":
        return r[r > 0]
    return np.abs(r)


def _gauss_tail(x, side, mu, sd):
    if side == "left":
        return stats.norm.cdf(-x, mu, sd)
    if side == "right":
        return stats.norm.sf(x, mu, sd)
    return stats.norm.cdf(-x, mu, sd) + stats.norm.sf(x, mu, sd)


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _years(y):
    """Waiting time in years, readable even when astronomically large."""
    if y < 1:
        return f"{_num(y * 252, 0)} trading days"
    if y < 1e5:
        return f"{y:,.0f} years" if y >= 10 else f"{_num(y, 1)} years"
    exp = int(np.floor(np.log10(y)))
    return f"{_num(y / 10 ** exp, 1)}·10{str(exp).translate(_SUP)} years"


def _se(tail_index, n_tail):
    """Standard error of the MLE tail index (Clauset et al. 2009, eq. 3.2)."""
    return tail_index / np.sqrt(n_tail)


def _existing(tail_index):
    names = [name for k, name in MOMENTS if tail_index > k]
    return ", ".join(names).capitalize() if names else "None"


def run(params, data, out):
    r = data.returns
    n = data.n
    mu, sd = float(np.mean(r)), float(np.std(r))
    side, method = params["tail"], params["method"]

    fits = {}
    for key in ("left", "right", "both"):
        values = _side_values(r, key)
        try:
            fits[key] = (values, PowerLawAnalyzer().fit_power_law(
                values, x_min_method=method, n_bootstrap=params["n_bootstrap"], seed=params["seed"]))
        except ValueError:
            fits[key] = (values, None)
    values, fit = fits[side]
    if fit is None:
        raise ToolError(f"The ‘{SIDES[side]}’ tail has too few non-zero values (at least 10 are required). "
                        "Use a longer series or choose a different tail.")

    alpha, x_min, n_tail = float(fit["alpha"]), float(fit["x_min"]), int(fit["n_tail"])
    tail_index = alpha - 1.0
    se = _se(tail_index, n_tail)
    p_val = float(fit["p_value"])
    out.metric("Tail index α − 1", tail_index, "num",
               hint=f"± {_num(se)} (standard error) · P(X > x) ∝ x^−(α−1) · cubic law ≈ 3",
               status="critical" if tail_index < 2 else "serious" if tail_index < 3 else
               "warning" if tail_index < 4 else None)
    out.metric("Density exponent α", alpha, "num", hint="Density p(x) ∝ x^−α")
    out.metric("Tail start x_min", x_min, "pct", hint="The power law applies from here")
    out.metric("Days in the tail", n_tail, "int", hint=f"{_num(100 * n_tail / n, 1)}% of all days")
    if fit["ks_statistic"] is not None:
        out.metric("KS distance", fit["ks_statistic"], "num", digits=3, hint="Distance data ↔ fit (smaller = better)")
    out.metric("Bootstrap p-value", p_val, "num", hint="> 0.1: power law plausible",
               status="warning" if p_val < 0.1 else None)
    out.metric("Existing moments", _existing(tail_index), "text", hint="Moment k exists if α − 1 > k")

    # Log-log CCDF: unconditional P(X >= x) for the chosen side, from a tenth of x_min upwards
    xs = np.sort(values[values > 0])[::-1]
    x_lo = max(float(xs[-1]), x_min / 10)
    ccdf = np.arange(1, len(xs) + 1) / n
    m = int(np.sum(xs >= x_lo))
    keep = np.unique(np.round(np.logspace(0, np.log10(m), 700)).astype(int)) - 1
    grid = np.logspace(np.log10(x_lo), np.log10(xs[0] * 1.5), 200)
    floor = 0.3 / n
    fit_line = np.where(grid >= x_min, (n_tail / n) * (grid / x_min) ** (-tail_index), np.nan)
    gauss = _gauss_tail(grid, side, mu, sd)
    gauss = np.where(gauss >= floor, gauss, np.nan)
    out.chart(C.xy(
        f"Probability of a day beyond x – {SIDES[side]}",
        [C.trace("Observed", xs[keep], ccdf[keep], type="scatter", color=1, size=5, decimate_to=None),
         C.trace(f"Power law (α − 1 = {_num(tail_index)})", grid, fit_line, color=2, width=2.5),
         C.trace("Normal distribution", grid, gauss, color="muted", dash=True),
         C.trace(f"x_min = {_num(100 * x_min)}%", [x_min, x_min], [floor, 1.0], color=3, dash=True, width=1.5,
                 decimate_to=None)],
        x=C.axis("Daily move x", type="log", fmt="pct", digits=1, range=(x_lo, xs[0] * 1.6)),
        y=C.axis("P(X ≥ x)", type="log", range=(floor, 1.2)), height=420,
        subtitle="On log-log axes a power law becomes a straight line. The Gaussian curve, by contrast, plunges.",
    ))

    # Comparison of the two tails (same method)
    order = [k for k in ("left", "right", "both") if fits[k][1] is not None]
    out.chart(C.xy(
        "Tail index by tail",
        [C.trace("α − 1", [SIDES[k] for k in order], [fits[k][1]["alpha"] - 1 for k in order], type="bar", color=1)],
        x=C.axis("Tail", type="category"), y=C.axis("Tail index α − 1"),
        hlines=[C.hline(2.0, "above: finite variance"), C.hline(4.0, "above: finite kurtosis")],
        height=300, subtitle="A smaller index means a fatter tail. For equities this is usually the loss side.",
    ))
    out.table("Comparison of tails",
              ["Tail", "α", "α − 1", "± std. error", "x_min", "Days in the tail", "p-value", "Existing moments"],
              [[SIDES[k], fits[k][1]["alpha"], fits[k][1]["alpha"] - 1,
                _se(fits[k][1]["alpha"] - 1, fits[k][1]["n_tail"]), fits[k][1]["x_min"], fits[k][1]["n_tail"],
                fits[k][1]["p_value"], _existing(fits[k][1]["alpha"] - 1)] for k in order],
              formats=["text", "num", "num", "num", "pct", "int", "num", "text"],
              note="Method: " + ("Clauset (KS-optimal x_min)" if method == "clauset" else "fixed threshold (largest 10%)")
                   + ". If two tails differ by less than about two standard errors, the difference is not "
                     "significant.")

    sample = [_num(100 * mu, 3) + "%", _num(100 * sd, 2) + "% (volatility)",
              _num(stats.skew(r)), _num(stats.kurtosis(r), 1) + " (excess)"]
    out.table("Existence of moments", ["Moment", "Condition", "Exists", "Sample (daily returns)"],
              [[name, f"α − 1 > {k}", "Yes" if tail_index > k else "No – diverges", s]
               for (k, name), s in zip(MOMENTS, sample)],
              formats=["text", "text", "text", "text"],
              note="Even a divergent moment yields a finite number in every sample – but that number is unstable "
                   "and grows with every new extreme day.")

    rows = []
    for lvl in LEVELS:
        if lvl > xs[0] * 2:
            break
        pl = (n_tail / n) * (lvl / x_min) ** (-tail_index) if lvl >= x_min else None
        g = float(_gauss_tail(lvl, side, mu, sd))
        rows.append([lvl, int(np.sum(values >= lvl)), pl, _years(1 / (pl * 252)) if pl else "–",
                     _years(1 / (g * 252)) if g > 0 else "never (numerically 0)"])
    out.table("Frequency of days beyond x",
              ["Threshold x", "Days observed", "Power law: frequency per day", "Power law: once every …",
               "Normal distribution: once every …"], rows,
              formats=["pct", "int", "pct", "text", "text"],
              note=f"Observed over {n:,} trading days ({_num(n / 252, 1)} years). The power law applies only from "
                   "x_min upwards; below that its columns stay empty.")

    if n_tail < 50:
        out.note(f"Only {n_tail} days in the tail: the tail index is highly uncertain (± {_num(se)}, roughly 95%: "
                 f"{_num(tail_index - 2 * se)} to {_num(tail_index + 2 * se)}). Robust conclusions require a "
                 "longer series.", "warning")
    if p_val < 0.1:
        out.note(f"The bootstrap rejects a pure power law (p = {_num(p_val)}). The exponent is then only an "
                 "approximation of the slope in the tail.", "warning")
    if tail_index < 2:
        out.set_verdict("critical", "Infinite variance in the tail",
                        f"α − 1 = {_num(tail_index)} < 2: volatility and variance-based risk measures cannot be "
                        "estimated reliably here.")
    elif tail_index < 3:
        out.set_verdict("serious", "Fat tail: finite variance, infinite skewness and kurtosis",
                        f"α − 1 = {_num(tail_index)}. Existing moments: {_existing(tail_index).lower()}. Sample skewness "
                        "and kurtosis are therefore not reliable measures.")
    elif tail_index < 4:
        out.set_verdict("warning", "Fat tail: finite variance and skewness, infinite kurtosis",
                        f"α − 1 = {_num(tail_index)}. Existing moments: {_existing(tail_index).lower()}. Sample kurtosis is "
                        "therefore not a reliable measure.")
    else:
        out.set_verdict("good", "Thin tail: all four moments finite",
                        f"α − 1 = {_num(tail_index)} > 4 – the tail decays quickly enough.")
