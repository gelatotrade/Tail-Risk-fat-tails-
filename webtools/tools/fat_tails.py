"""Why the normal distribution fails - the README's opening argument, on live data."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import TailRiskMetrics
from src.utils.statistics import compute_moments, jarque_bera_test, kolmogorov_smirnov_test

from .. import charts as C
from ..data import index_axis

SPEC = {
    "id": "fat-tails",
    "title": "Fat tails vs the normal distribution",
    "group": "data",
    "order": 10,
    "summary": "How often 3, 4 and 5-sigma days actually occur, compared with the Gaussian assumption.",
    "description": (
        "Many risk models assume normally distributed returns. This tool measures how much more often extreme "
        "days occur in your dataset than that assumption predicts, and plots the distribution on a logarithmic "
        "scale – where the fat tails become visible.\n\n"
        "The Jarque-Bera and Kolmogorov-Smirnov tests check formally whether the normal distribution can be "
        "retained."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "bins", "type": "int", "default": 90, "min": 30, "max": 200, "step": 10,
         "label": "Histogram bins"},
        {"name": "max_sigma", "type": "int", "default": 6, "min": 3, "max": 10, "step": 1,
         "label": "Sigma thresholds up to", "help": "The table counts days beyond 2σ up to this value."},
    ],
    "covers": [
        "src.utils.statistics.compute_moments",
        "src.utils.statistics.jarque_bera_test",
        "src.utils.statistics.kolmogorov_smirnov_test",
        "src.models.risk_metrics.TailRiskMetrics.tail_ratio",
    ],
    "cost": "fast",
}


def run(params, data, out):
    r = data.returns
    mu, sd = float(np.mean(r)), float(np.std(r))
    moments = compute_moments(r)
    jb = jarque_bera_test(r)
    ks = kolmogorov_smirnov_test(r)
    metrics = TailRiskMetrics(r)

    out.metric("Observations", data.n, "int")
    out.metric("Annualised volatility", sd * np.sqrt(252), "pct", digits=1)
    out.metric("Skewness", moments["skewness"], "num", hint="< 0: crashes more frequent than rallies")
    out.metric("Excess kurtosis", moments["kurtosis"], "num", digits=1,
               hint="Normal distribution = 0; > 3 indicates fat tails",
               status="critical" if moments["kurtosis"] > 6 else "serious" if moments["kurtosis"] > 3 else None)

    # Sigma table: empirical vs Gaussian frequency
    rows = []
    worst_ratio = 0.0
    for k in range(2, int(params["max_sigma"]) + 1):
        tr = metrics.tail_ratio(k)
        emp = tr["left_tail_prob"] + tr["right_tail_prob"]
        gauss = 2 * tr["gaussian_tail_prob"]
        days = int(np.sum(np.abs(r - 0) > k * metrics.std))
        ratio = emp / gauss if gauss > 0 else None
        if ratio:
            worst_ratio = max(worst_ratio, ratio if k <= 5 else worst_ratio)
        years_gauss = (1 / gauss) / 252 if gauss > 0 else None
        rows.append([f"{k}σ", days, emp, gauss, ratio, years_gauss])
    out.table(
        "Extreme days: observed vs normal distribution",
        ["Threshold", "Days observed", "Observed frequency", "Gaussian frequency", "Factor",
         "Gaussian: once every … years"],
        rows, formats=["text", "int", "pct", "pct", "x", "num"],
        note="Factor = observed frequency ÷ frequency predicted by the normal distribution (both tails).",
    )

    # Distribution on linear and log scale
    centers, heights = C.histogram(r, bins=int(params["bins"]), density=True)
    grid = np.linspace(centers[0], centers[-1], 400)
    gauss_pdf = stats.norm.pdf(grid, mu, sd)
    t_params = stats.t.fit(r)
    t_pdf = stats.t.pdf(grid, *t_params)
    hist_trace = C.trace("Observed", centers, heights, type="bar", color=1)
    floor = float(np.min(heights[heights > 0])) / 4  # one bin with one observation, with headroom
    log_range = (floor, float(np.max(heights)) * 3)
    for log in (False, True):
        g = np.where(gauss_pdf >= floor, gauss_pdf, np.nan) if log else gauss_pdf
        tp = np.where(t_pdf >= floor, t_pdf, np.nan) if log else t_pdf
        out.chart(C.xy(
            "Distribution of daily returns" + (" – log scale" if log else ""),
            [hist_trace,
             C.trace("Normal distribution", grid, g, color="muted", dash=True),
             C.trace(f"Student-t (ν = {t_params[0]:.1f})", grid, tp, color=2)],
            x=C.axis("Daily return", fmt="pct", digits=1),
            y=C.axis("Density", type="log", range=log_range) if log else C.axis("Density"),
            subtitle=("On the log scale the observed tails lie far above the Gaussian curve."
                      if log else "In the centre both curves look similar …"),
        ))

    # QQ plot against the normal
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    q = np.asarray(osm)
    out.chart(C.xy(
        "Q-Q plot against the normal distribution",
        [C.trace("Perfect normal distribution", [q.min(), q.max()], [q.min(), q.max()], color="muted", dash=True),
         C.trace("Returns (standardised)", q, (np.asarray(osr) - mu) / sd, type="scatter", color=1, size=5,
                 decimate_to=None)],
        x=C.axis("Theoretical quantiles (σ)"), y=C.axis("Observed quantiles (σ)"),
        subtitle="Points that bend away from the diagonal at both ends are the fat tails.",
    ))

    # Time series with extreme days flagged
    x, xa = index_axis(data)
    extreme = np.abs(r - mu) > 3 * sd
    out.chart(C.xy(
        "Daily returns with 3-sigma outliers",
        [C.trace("Daily return", x, r, color=1, width=1),
         C.trace("Beyond 3σ", [xi for xi, e in zip(x, extreme) if e], r[extreme], type="scatter",
                 color="critical", size=7, decimate_to=None)],
        x=xa, y=C.axis("Return", fmt="pct", digits=1),
        hlines=[C.hline(mu + 3 * sd, "+3σ"), C.hline(mu - 3 * sd, "−3σ")],
    ))

    out.table("Formal normality tests", ["Test", "Statistic", "p-value", "Result"], [
        ["Jarque-Bera", jb["statistic"], jb["p_value"],
         "Normality rejected" if not jb["is_normal"] else "Not rejected"],
        ["Kolmogorov-Smirnov", ks["statistic"], ks["p_value"],
         "Normality rejected" if not ks["matches_distribution"] else "Not rejected"],
    ], formats=["text", "num", "num", "text"])

    if not jb["is_normal"]:
        out.set_verdict(
            "serious" if moments["kurtosis"] > 3 else "warning",
            "The normal distribution understates crash risk here",
            f"Days beyond 4σ occur {rows[2][4]:.0f} times more often than the Gaussian assumption allows."
            if len(rows) > 2 and rows[2][4] else "The tails are considerably fatter than normal.",
        )
    else:
        out.set_verdict("good", "No significant deviation from the normal distribution",
                        "This can happen with short or calm series – the tails may still be fat.")
