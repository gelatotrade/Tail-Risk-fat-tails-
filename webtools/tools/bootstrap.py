"""How reliable are the numbers? Bootstrap confidence intervals for VaR, Expected Shortfall, kurtosis and tail index."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import TailRiskMetrics
from src.physics.levy_flight import estimate_tail_index
from src.utils.numerical import numerical_gradient, numerical_hessian
from src.utils.statistics import bootstrap_confidence_interval, compute_moments

from .. import charts as C

SPEC = {
    "id": "bootstrap",
    "title": "Bootstrap confidence intervals",
    "group": "data",
    "order": 50,
    "summary": "How much VaR, Expected Shortfall, kurtosis and the tail index would vary under a different but "
               "equally plausible history – expressed as confidence intervals.",
    "description": (
        "Every risk metric is an estimate from a single history. The **bootstrap** "
        "(`bootstrap_confidence_interval`) draws many new samples of the same length from your daily returns "
        "with replacement, recomputes the metric on each one and takes the middle 95% (adjustable) of the "
        "results as the confidence interval. A wide interval means the figure is uncertain, however many decimal "
        "places it shows.\n\n"
        "The tool examines historical **VaR**, **Expected Shortfall** (the average loss beyond VaR), **excess "
        "kurtosis** and the **Hill tail index α** of absolute returns (smaller = fatter tails; below 2, infinite "
        "variance). The histograms show the bootstrap distribution of each metric with its point estimate and "
        "interval bounds. For α the table also shows the asymptotic Wald interval derived from the curvature of "
        "the log-likelihood (`numerical_hessian`).\n\n"
        "Important: the simple bootstrap draws days independently of each other, so volatility clusters are lost "
        "– in real markets the true uncertainty tends to be larger. The **seed** makes the result reproducible; "
        "more samples smooth the histograms but take longer to compute."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "n_boot", "type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100,
         "label": "Bootstrap samples", "help": "For long series the tool caps this number so that it stays "
                                                "responsive in the browser."},
        {"name": "confidence", "type": "float", "default": 0.95, "min": 0.80, "max": 0.99, "step": 0.01,
         "label": "Interval confidence level"},
        {"name": "level", "type": "float", "default": 0.99, "min": 0.90, "max": 0.995, "step": 0.005,
         "label": "VaR/ES confidence level"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.utils.statistics.bootstrap_confidence_interval",
        "src.utils.statistics.compute_moments",
        "src.utils.numerical.numerical_gradient",
        "src.utils.numerical.numerical_hessian",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
        "src.physics.levy_flight.estimate_tail_index",
    ],
    "cost": "medium",
}
MAX_WORK = 4e6  # observations x resamples, keeps the browser responsive


def _pct(x, digits=2):
    return f"{x * 100:.{digits}f}%"


def _lvl(p):
    """Confidence level as a short percentage: 0.99 -> '99%', 0.975 -> '97.5%'."""
    return f"{p * 100:.1f}".rstrip("0").rstrip(".") + "%"


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _hill_wald(r, alpha, z):
    """Wald interval for the Hill estimate: the Pareto log-likelihood of the k largest |r| (same k as
    ``estimate_tail_index``) has curvature -k/alpha^2 at its maximum; numerical_hessian measures it."""
    x = np.sort(np.abs(r[np.isfinite(r)]))[::-1]
    n, n_pos = len(x), int(np.count_nonzero(x > 0))
    k = min(max(10, int(0.1 * n)), n - 1, n_pos - 1)
    s = float(np.sum(np.log(x[:k]) - np.log(x[k])))

    def loglik(a):
        return k * np.log(a) - a * s

    if not np.isclose(k / s, alpha, rtol=1e-9):  # not the same estimator -> no comparison
        return None
    grad = numerical_gradient(loglik, alpha)
    hess = numerical_hessian(loglik, alpha)
    if not hess < 0:
        return None
    se = 1.0 / np.sqrt(-hess)
    return alpha - z * se, alpha + z * se, se, grad, k


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    level, conf, seed = float(params["level"]), float(params["confidence"]), int(params["seed"])
    n_boot = int(params["n_boot"])
    if n * n_boot > MAX_WORK:
        n_boot = max(100, int(MAX_WORK // n // 50 * 50))
        out.note(f"For {n:,} observations the tool uses {n_boot:,} instead of {int(params['n_boot']):,} samples "
                 "so that it stays responsive in the browser.", "info")

    # TailRiskMetrics precomputes skew/kurtosis in __init__ (milliseconds per call). Its historical VaR and
    # ES only read .returns, so one instance is reused and handed each resample.
    trm = TailRiskMetrics(r)

    def var_hist(s):
        trm.returns = s
        return trm.var_historical(level)

    def es_hist(s):
        trm.returns = s
        return trm.expected_shortfall(level)

    stats_def = [
        ("var", f"{_lvl(level)} VaR (1 day)", var_hist, True),
        ("es", f"{_lvl(level)} Expected Shortfall", es_hist, True),
        ("kurt", "Excess kurtosis", lambda s: compute_moments(s)["kurtosis"], False),
        ("hill", "Hill tail index α", lambda s: estimate_tail_index(s, "hill"), False),
    ]
    results = {}
    for i, (key, name, fn, _) in enumerate(stats_def):
        out.progress(f"Bootstrap: {name} …", 0.05 + 0.22 * i)
        point = float(fn(r))
        reps = []

        def recorded(s, fn=fn, reps=reps):
            v = float(fn(s))
            reps.append(v)
            return v

        np.random.seed(seed)  # same seed -> the same resamples for every statistic
        lo, hi = bootstrap_confidence_interval(r, recorded, n_bootstrap=n_boot, confidence=conf)
        results[key] = (point, float(lo), float(hi), np.asarray(reps))
    trm.returns = r

    cl = _lvl(conf)
    tail_obs = int(np.sum(r < -results["var"][0]))
    out.metric(f"{_lvl(level)} VaR (1 day)", results["var"][0], "pct", digits=2,
               hint=f"{cl} interval: {_pct(results['var'][1])} to {_pct(results['var'][2])}")
    out.metric(f"{_lvl(level)} Expected Shortfall", results["es"][0], "pct", digits=2,
               hint=f"{cl} interval: {_pct(results['es'][1])} to {_pct(results['es'][2])}")
    out.metric("Excess kurtosis", results["kurt"][0], "num", digits=1,
               hint=f"{cl} interval: {_num(results['kurt'][1], 1)} to {_num(results['kurt'][2], 1)}")
    out.metric("Hill tail index α", results["hill"][0], "num", digits=2,
               hint=f"{cl} interval: {_num(results['hill'][1])} to {_num(results['hill'][2])}",
               status="critical" if results["hill"][2] < 2 else "serious" if results["hill"][0] < 3 else None)
    out.metric("Days beyond VaR", tail_obs, "int", hint=f"Out of {n:,} – Expected Shortfall rests on these days")

    # ---- bootstrap distributions --------------------------------------------------------------------
    sub = {"var": "Largest one-day loss that is not exceeded on {lv} of days.",
           "es": "Average loss on the worst days beyond VaR.",
           "kurt": "A handful of extreme days drive the kurtosis – hence its wide interval.",
           "hill": "Below 2: infinite variance (Lévy-stable range). Typical for equities: 3 to 4."}
    for key, name, _, is_pct in stats_def:
        point, lo, hi, reps = results[key]
        centers, counts = C.histogram(reps, bins=40, density=False)
        out.chart(C.xy(
            name, [C.trace("Bootstrap samples", centers, counts, type="bar", color=1)],
            x=C.axis(name, fmt="pct" if is_pct else "num", digits=2), y=C.axis("Number of samples", fmt="int"),
            vlines=[C.vline(lo, f"{cl}: {_pct(lo) if is_pct else _num(lo)}", 2),
                    C.vline(hi, _pct(hi) if is_pct else _num(hi), 2),
                    C.vline(point, "Estimate", "ink", dash=False)],
            subtitle=sub[key].format(lv=_lvl(level)), height=300,
        ))

    # ---- summary table --------------------------------------------------------------------------------
    rows = []
    for key, name, _, is_pct in stats_def:
        point, lo, hi, reps = results[key]
        f = _pct if is_pct else _num
        rel = (hi - lo) / 2 / abs(point) if point else None
        rows.append([name, "Bootstrap (percentile)", f(point), f(lo), f(hi), rel])
    z = stats.norm.ppf(0.5 + conf / 2)
    wald = _hill_wald(r, results["hill"][0], z)
    note = (f"± relative = half the interval width ÷ estimate. {n_boot:,} samples with seed {seed}; "
            "all metrics use the same samples.")
    if wald:
        lo_w, hi_w, se, grad, k = wald
        rows.append(["Hill tail index α", f"Wald (log-likelihood curvature, k = {k})", _num(results["hill"][0]),
                     _num(lo_w), _num(hi_w), (hi_w - lo_w) / 2 / results["hill"][0]])
        slope = f"{grad:.1e}".replace("-", "−")
        note += (f" Wald: standard error α/√k = {_num(se, 3)} from numerical_hessian; the slope of the "
                 f"log-likelihood at the estimate is {slope} (≈ 0, i.e. a maximum).")
    out.table(f"Confidence intervals ({cl})", ["Metric", "Method", "Estimate", "Lower bound", "Upper bound",
                                               "± relative"], rows,
              formats=["text", "text", "text", "text", "text", "pct"], note=note)
    if tail_obs < 10:
        out.note(f"Only {tail_obs} day{'s' if tail_obs != 1 else ''} beyond VaR: Expected Shortfall rests on so few "
                 "losses that even the bootstrap interval understates the uncertainty – the bootstrap cannot invent "
                 "losses worse than the worst observed day. Use a longer series or a lower VaR/ES confidence level.", "warning")
    out.note("The bootstrap draws days independently. Volatility clusters (calm and turbulent phases) are lost in the "
             "process – in real markets the intervals tend to be wider than shown here.", "warning")
