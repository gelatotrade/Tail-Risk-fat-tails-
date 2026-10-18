"""Fit a q-Gaussian to real returns: q, β, Tsallis entropy and the risk assessment of the Tsallis model."""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisDistribution, TsallisEntropyAnalyzer, TsallisTailRiskModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tsallis-fit",
    "title": "Tsallis fit and risk level",
    "group": "physics",
    "order": 80,
    "summary": "Estimates the Tsallis index q of your returns in three ways, compares model and data in the "
               "tails, and shows the risk level of the Tsallis model.",
    "description": (
        "Three estimators for the entropic index **q**: the **method of moments** inverts the kurtosis "
        "(k = 6(q − 1)/(7 − 5q)) and can therefore only detect q < 7/5 – above that the kurtosis is infinite. "
        "**Maximum likelihood** fits q and β directly to the density and also captures heavier tails. The "
        "**Hill route** (`estimate_optimal_q`) converts the tail index of the 5% largest |returns| into q.\n\n"
        "`TsallisTailRiskModel.update` uses the moment fit: it computes the probability of 3- to 6-sigma days, "
        "the ratio to the normal distribution, the Tsallis entropy of the return distribution (50 bins) and, "
        "from these, a risk level. ‘Elevated’ applies from q > 1.3 and S_q > 1, ‘High’ from q > 1.5 and "
        "S_q > 1.5, ‘Extreme’ from q > 1.8 and S_q > 2. Because the moment fit stays below 1.4, the model "
        "practically never reaches the two top levels – so compare the MLE q as well.\n\n"
        "How to read the charts: the log-log plot shows whether the model matches the frequency of large days; "
        "on the Q-Q plot the points lie on the diagonal if the q-Gaussian also describes the extreme quantiles."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "fit", "type": "select", "default": "mle", "label": "Fit for tails and Q-Q plot",
         "options": [{"value": "mle", "label": "Maximum likelihood"}, {"value": "moments", "label": "Moments (kurtosis)"}]},
        {"name": "bins", "type": "int", "default": 90, "min": 30, "max": 200, "step": 10,
         "label": "Histogram bins"},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisDistribution.fit",
        "src.physics.tsallis_statistics.TsallisDistribution.log_pdf",
        "src.physics.tsallis_statistics.TsallisDistribution.ppf",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_probability",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.estimate_optimal_q",
        "src.physics.tsallis_statistics.TsallisTailRiskModel",
        "src.physics.tsallis_statistics.TsallisTailRiskModel.update",
    ],
    "cost": "fast",
}

LEVELS = {"NORMAL": ("good", "Normal"), "ELEVATED": ("warning", "Elevated"),
          "HIGH": ("serious", "High"), "EXTREME": ("critical", "Extreme")}
STEP = {"NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "EXTREME": 4}


def _days(x):
    """Expected number of days: tiny Gaussian expectations as text instead of 4.96e-6."""
    return "< 0.001" if x < 1e-3 else float(x)


def _fmt(x, digits=2):
    return f"{x:.{digits}f}"


def _sd(dist):
    v = float(dist.variance())
    return np.sqrt(v) if np.isfinite(v) else "∞"


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    try:
        mom = TsallisDistribution.fit(r, method="moments")
        mle = TsallisDistribution.fit(r, method="mle")
    except ValueError as exc:
        raise ToolError(f"The Tsallis fit is not possible: the returns are (almost) constant ({exc}).")
    q_hill = float(TsallisEntropyAnalyzer().estimate_optimal_q(r))
    model = TsallisTailRiskModel()
    u = model.update(r)
    mu, sd = float(np.mean(r)), float(np.std(r))
    gauss = TsallisDistribution(q=1.0, beta=1.0 / (2 * sd ** 2), mu=mu)  # q = 1 is the normal distribution
    chosen, cname = (mle, "MLE") if params["fit"] == "mle" else (mom, "moments")

    level, label = LEVELS.get(u["risk_level"], ("warning", u["risk_level"]))
    out.metric("q (maximum likelihood)", mle.q, "num", hint="1 = Gaussian; equities typically 1.4–1.6")
    out.metric("q (moments)", mom.q, "num", hint="from the kurtosis; identifiable only up to 7/5")
    out.metric("q (Hill)", q_hill, "num", hint="from the tail index of the largest 5%")
    out.metric("Tail exponent of the density", float(chosen.tail_exponent()), "num",
               hint=f"2/(q − 1), {cname} fit; < 3 would mean infinite variance")
    out.metric("Tsallis entropy S_q", u["tsallis_entropy"], "num", hint="histogram with 50 bins, q from moments")
    out.metric("Risk level (Tsallis model)", label, "text", status=level,
               badge=f"Level {STEP.get(u['risk_level'], '?')} of 4",
               hint=f"from q = {_fmt(u['q'])} and S_q = {_fmt(u['tsallis_entropy'])}")
    ratio4 = float(u["fat_tail_ratio"]["4sigma"])
    out.metric("4σ days relative to Gaussian", ratio4, "x", hint="probability from the moment fit ÷ Gaussian value",
               status="serious" if ratio4 > 5 else None,
               badge="Gaussian understates risk" if ratio4 > 5 else "")

    # --- table: the estimators side by side
    def ll(d):
        return float(np.sum(d.log_pdf(r)))

    fits = [("Moments (kurtosis)", mom, 3), ("Maximum likelihood", mle, 3), ("Normal distribution (q = 1)", gauss, 2)]
    rows = []
    for name, d, k in fits:
        lv = ll(d)
        rows.append([name, float(d.q), float(d.beta), _sd(d), float(d.tail_exponent()) if d.q > 1 else "∞", lv, 2 * k - 2 * lv])
    rows.insert(2, ["Hill (estimate_optimal_q)", q_hill, None, None, 2 / (q_hill - 1), None, None])
    out.table("Three routes to q", ["Method", "q", "β", "Model σ (daily)", "Tail exponent", "Log-likelihood", "AIC"],
              rows, formats=["text", "num", "num", "pct", "num", "num", "num"],
              note="Lower AIC = better description of the whole distribution. β is large for daily returns because "
                   "1/√β is the width in return units.")

    # --- table: model vs data in the tails, as numbers of days
    med = float(np.median(r))
    trows = []
    for k in (3, 4, 5, 6):
        key = f"{k}sigma"
        obs = int(np.sum(np.abs(r - med) > k * sd))
        p_mle = float(mle.tail_probability(mle.mu + k * sd, "both", exact=True))
        ratio = float(u["fat_tail_ratio"][key])
        trows.append([f"{k}σ", obs, _days(p_mle * n), _days(float(u["extreme_probabilities"][key]) * n),
                      _days(float(2 * stats.norm.sf(k)) * n), "> 10⁵" if ratio > 1e5 else ratio])
    out.table("Extreme days: model vs data",
              ["Threshold", "Observed", "q-Gaussian (MLE)", "q-Gaussian (moments)", "Gaussian",
               "Moments ÷ Gaussian"],
              trows, formats=["text", "int", "num", "num", "num", "x"],
              note=f"Observed: days with |r − median| > k·σ out of {n:,} days. Model columns: expected number of "
                   f"days = model probability × {n:,}. The moments column and the ratio come directly from "
                   "TsallisTailRiskModel.update.")

    # --- chart 1: histogram vs both fits, log scale
    centers, heights = C.histogram(r, bins=params["bins"], density=True)
    grid = np.linspace(centers[0], centers[-1], 500)
    floor = float(np.min(heights[heights > 0])) / 4
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Daily returns and q-Gaussian fits",
        [C.trace("Observed", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace(f"q-Gaussian, MLE (q = {_fmt(mle.q)})", grid, keep(mle.pdf(grid)), color=2, width=2.5),
         C.trace(f"q-Gaussian, moments (q = {_fmt(mom.q)})", grid, keep(mom.pdf(grid)), color=3, width=1.5),
         C.trace("Normal distribution", grid, keep(gauss.pdf(grid)), color="muted", dash=True)],
        x=C.axis("Daily return", fmt="pct", digits=1), y=C.axis("Density", type="log", range=(floor, float(heights.max()) * 3)),
        subtitle="Logarithmic scale. The Gaussian parabola drops off far too early in the tails.",
    ))

    # --- chart 2: tails P(|r - mu| > x), log-log
    a = np.abs(r - chosen.mu)
    xs = np.logspace(np.log10(max(np.median(a), 1e-6)), np.log10(a.max() * 1.6), 60)
    emp = np.array([np.mean(a > v) for v in xs])
    exact = np.asarray(chosen.tail_probability(chosen.mu + xs, "both", exact=True), dtype=float)
    asym = np.asarray(chosen.tail_probability(chosen.mu + xs, "both"), dtype=float)
    y_lo = 0.5 / n
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    tail_traces = [
        C.trace("Observed", xs, pos(np.where(emp > 0, emp, np.nan)), type="step", color=1, width=2.5),
        C.trace(f"q-Gaussian ({cname})", xs, pos(exact), color=2),
        C.trace("Asymptote", xs, pos(asym), color=2, dash=True, width=1.5),
        C.trace("Gaussian", xs, pos(2 * stats.norm.sf(xs / sd)), color="muted", dash=True)]
    out.chart(C.xy(
        "Tails on a log-log plot: P(|r − μ| > x)",
        [t for t in tail_traces if np.any(np.isfinite(t["y"]))],
        x=C.axis("Distance x from the centre", type="log", fmt="pct", digits=1),
        y=C.axis("Share of days", type="log", fmt="pct", range=(y_lo, 1.0)),
        subtitle="Power laws are straight lines. The asymptote from tail_probability applies only far out.", height=370,
    ))

    # --- chart 3: QQ plot against the chosen fit (tails kept, centre thinned)
    rs = np.sort(r)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = chosen.ppf(probs)
    tail = max(20, n // 50)
    idx = np.unique(np.r_[np.arange(0, n, max(1, n // 800)), np.arange(tail), np.arange(n - tail, n)])
    lim = [float(min(rs[0], theo[0])), float(max(rs[-1], theo[-1]))]
    out.chart(C.xy(
        f"Q-Q plot against the q-Gaussian ({cname})",
        [C.trace("Perfect fit", lim, lim, color="muted", dash=True),
         C.trace("Returns", theo[idx], rs[idx], type="scatter", color=1, size=5, decimate_to=None)],
        x=C.axis("q-Gaussian quantile", fmt="pct", digits=1), y=C.axis("Observed quantile", fmt="pct", digits=1),
        subtitle="If the ends bend away from the diagonal, the real tails are heavier (or lighter) than in the model.",
    ))

    thresholds = "Elevated if q > 1.3 and S_q > 1 · High if q > 1.5 and S_q > 1.5 · Extreme if q > 1.8 and S_q > 2"
    out.set_verdict(level, f"Tsallis risk level: {label}",
                    f"q = {_fmt(u['q'])} (moments) and S_q = {_fmt(u['tsallis_entropy'])}. According to the model, "
                    f"4σ days are {ratio4:,.0f} times as frequent as under a Gaussian. Thresholds: {thresholds}.")
    if mle.q > 1.42 and mom.q < 1.4:
        out.note(f"The MLE q ({_fmt(mle.q)}) lies above 7/5, where the kurtosis is infinite, so the moment fit "
                 f"necessarily stays below it ({_fmt(mom.q)}). The risk level is based on the moment fit.", "info")
