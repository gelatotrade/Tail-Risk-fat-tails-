"""Jump-diffusion risk: split returns into jumps and diffusion, decompose the VaR, OU coordinates of |r|."""

import numpy as np
from scipy import optimize, stats

from src.physics.ornstein_uhlenbeck import JumpDiffusionRiskAnalyzer, compute_ou_coordinates

from .. import charts as C
from ..data import index_axis, fmt_date

SPEC = {
    "id": "jump-diffusion",
    "title": "Jumps vs diffusion: VaR decomposition",
    "group": "risk",
    "order": 60,
    "summary": "How much of the tail risk comes from rare jumps and how much from ordinary noise – with jump "
               "detection, a Monte Carlo VaR decomposition and OU coordinates.",
    "description": (
        "The model splits your returns into two parts: **jumps** are all days that lie further from the mean "
        "than the threshold (in standard deviations); the rest is **diffusion**. An Ornstein-Uhlenbeck process is "
        "fitted to the calm days, while the jumps provide frequency, mean and standard deviation. A Monte Carlo "
        "simulation of the model with and without jumps – using **the same random numbers** for the diffusion – "
        "then shows how much of the one-day VaR is attributable to jumps.\n\n"
        "How to read the charts: red dots are detected jumps. The bar chart by year shows whether jumps arrive "
        "evenly (as the Poisson model assumes) or cluster in crises – usually the latter. The decomposition by "
        "threshold shows how strongly the result depends on this definition. The OU coordinates fit an OU process "
        "to |return| in each rolling window: X is the volatility level, Z the speed of mean reversion (Y, the "
        "latest move |r|, already appears in the first chart).\n\n"
        "As a check, the tool also computes the model VaR analytically (Poisson mixture of normal distributions) – "
        "any difference from the Monte Carlo figure is pure simulation noise.\n\n"
        "Note: daily returns have very little memory. The OU fit therefore usually ends up in the limiting case "
        "of ‘white noise’ (θ is capped, half-life ≈ 0.1 days) – the diffusion then corresponds to a normal "
        "distribution with the standard deviation of the calm days."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 3.0, "min": 2.0, "max": 6.0, "step": 0.1,
         "label": "Jump threshold", "unit": "σ", "help": "Days beyond mean ± threshold·σ count as jumps."},
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999, "step": 0.001,
         "label": "Confidence level"},
        {"name": "n_simulations", "type": "int", "default": 50000, "min": 5000, "max": 200000, "step": 5000,
         "label": "Monte Carlo scenarios", "help": "The jump share of the VaR depends on a few scenarios – more "
                                                   "scenarios make it more stable."},
        {"name": "window", "type": "int", "default": 30, "min": 10, "max": 120, "step": 5,
         "label": "OU coordinate window", "unit": "days"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer",
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer.fit",
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer.tail_risk_decomposition",
        "src.physics.ornstein_uhlenbeck.compute_ou_coordinates",
    ],
    "cost": "fast",
}

FIT_STATUS = {
    "ok": "OK",
    "theta_capped": "white noise (θ capped)",
    "no_mean_reversion": "no mean reversion",
    "too_few_points": "too few calm days",
}
SWEEP = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]



def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")




def _level(conf):
    return f"{100 * conf:.1f}".rstrip("0").rstrip(".") + "%"


def _model_var(model, dt, p, jump_mean, jump_std, conf):
    """Exact one-day VaR of the simulated model: OU step from 0 plus Poisson(p) jumps N(jump_mean, jump_std²).

    The k-jump scenario is normal with mean m0 + k·jump_mean and variance v0 + k·jump_std², so the
    distribution is a Poisson mixture of normals; its (1 − conf)-quantile is found numerically.
    """
    m0, v0 = float(model.expected_value(0.0, dt)), float(model.variance_at_time(dt))
    ks = np.arange(0, 12)
    w = stats.poisson.pmf(ks, p)
    mk, sk = m0 + ks * jump_mean, np.sqrt(v0 + ks * jump_std ** 2)

    def cdf(x):
        return float(np.sum(w * stats.norm.cdf((x - mk) / sk)))

    lo, hi = m0 - 50 * np.sqrt(v0) - 50 * jump_std - 1.0, m0 + 1.0
    return -optimize.brentq(lambda x: cdf(x) - (1 - conf), lo, hi, xtol=1e-10)


def run(params, data, out):
    r = data.returns
    n = data.n
    thr, conf, n_sim, seed = params["threshold"], params["confidence"], params["n_simulations"], params["seed"]
    ja = JumpDiffusionRiskAnalyzer()
    dec = ja.tail_risk_decomposition(r, confidence=conf, threshold=thr, n_simulations=n_sim, seed=seed)
    has_model = ja.ou_model is not None
    mean, std = float(np.mean(r)), float(np.std(r))
    is_jump = np.abs(r - mean) > thr * std  # the analyzer's own definition, for display
    var_hist = -float(np.percentile(r, (1 - conf) * 100))
    var_gauss = -(mean + stats.norm.ppf(1 - conf) * std)
    lvl = _level(conf)
    share = dec["var_jump"] / dec["var_total"] if dec["var_total"] > 0 else None
    exact = (_model_var(ja.ou_model, ja.DT, dec["jump_intensity"], dec["jump_mean"], dec["jump_std"], conf)
             if has_model else None)

    out.metric(f"{lvl} VaR · model total", dec["var_total"], "pct",
               hint="OU + jumps, 1 day" + (f" · analytical: {_num(100 * exact)}%" if exact is not None else ""))
    out.metric("Diffusion VaR", dec["var_continuous"], "pct", hint="the same model without jumps")
    out.metric("Jump add-on", dec["var_jump"], "pct",
               hint=f"{_num(100 * share, 0)}% of the VaR" if share is not None else "",
               status="serious" if share is not None and share > 0.25 else None)
    out.metric(f"{lvl} historical VaR", var_hist, "pct", hint=f"Normal distribution: {_num(100 * var_gauss)}%")
    out.metric("Detected jumps", int(dec["n_jumps"]), "int",
               hint=f"{_num(dec['jump_intensity'] * 252, 1)} per year ({_num(100 * dec['jump_intensity'], 2)}% of days)")
    n_jumps = int(dec["n_jumps"])
    if n_jumps:
        out.metric("Average jump", dec["jump_mean"], "pct",
                   hint=f"Std. dev. {_num(100 * dec['jump_std'])}%" if n_jumps > 1 else "too few jumps for a std. dev.")
    else:
        out.metric("Average jump", "–", "text", hint=f"no day beyond {_num(thr, 1)}σ")
    out.metric("Jump share of Σ|r|", dec["jump_contribution"], "pct", digits=1,
               hint="the share of total movement that falls on jump days")
    if has_model:
        out.metric("Std. dev. of calm days", ja.ou_model.stationary_std(), "pct",
                   hint=f"stationary std. dev. of the OU process · fit: {FIT_STATUS[dec['ou_fit_status']]}")
    else:
        out.metric("OU fit", FIT_STATUS[dec["ou_fit_status"]], "text", status="warning",
                   hint="without a model, the VaR is the historical one")

    x, xa = index_axis(data)
    out.chart(C.xy(
        f"Daily returns with detected jumps (beyond {_num(thr, 1)}σ)",
        [C.trace("Daily return", x, r, color=1, width=1),
         C.trace("Jump", [xi for xi, j in zip(x, is_jump) if j], r[is_jump], type="scatter", color="critical",
                 size=7, decimate_to=None)],
        x=xa, y=C.axis("Return", fmt="pct", digits=1),
        hlines=[C.hline(mean + thr * std, f"+{_num(thr, 1)}σ"), C.hline(mean - thr * std, f"−{_num(thr, 1)}σ")],
        height=390,
    ))

    # Jumps per (calendar) year: Poisson would spread them evenly
    if data.dates:
        years, inv = np.unique([d[:4] for d in data.dates], return_inverse=True)
        years = [str(y) for y in years]
    else:
        inv = np.arange(n) // 252
        years = [f"Year {i + 1}" for i in range(int(inv[-1]) + 1)]
    counts = np.bincount(inv, weights=is_jump.astype(float), minlength=len(years))
    days = np.bincount(inv, minlength=len(years))
    expected = dec["jump_intensity"] * days
    # Pearson dispersion index with each year's exposure (partial years count less); Poisson: about 1
    ok = expected > 0
    disp = (float(np.sum((counts[ok] - expected[ok]) ** 2 / expected[ok]) / max(1, ok.sum() - 1))
            if counts.sum() > 0 and ok.sum() > 1 else None)
    if counts.sum() > 0:
        out.chart(C.xy(
            "Jumps per year",
            [C.trace("Jumps", years, counts, type="bar", color=1),
             C.trace("Poisson expectation", years, expected, color="muted", dash=True)],
            x=C.axis("Year", type="category"), y=C.axis("Count", fmt="int"), height=300,
            subtitle=(f"Dispersion index = {_num(disp, 1)} (Poisson: ≈ 1, partial years weighted). Higher means "
                      "that jumps cluster." if disp is not None else
                      "The series covers only one year, so it cannot show whether jumps cluster."),
        ))

    # Decomposition across thresholds
    out.progress("VaR decomposition for several thresholds …", 0.5)
    sweep = sorted(set(SWEEP + [round(thr, 2)]))
    sw = [JumpDiffusionRiskAnalyzer().tail_risk_decomposition(r, confidence=conf, threshold=t,
                                                              n_simulations=min(n_sim, 20000), seed=seed)
          for t in sweep]
    labels = [f"{_num(t, 1)}σ" +(" ◆" if abs(t - thr) < 1e-9 else "") for t in sweep]
    out.chart(C.xy(
        "VaR decomposition by jump threshold",
        [C.trace("Diffusion", labels, [d["var_continuous"] for d in sw], type="bar", color=1),
         C.trace("Jump add-on", labels, [d["var_jump"] for d in sw], type="bar", color=2)],
        x=C.axis("Jump threshold (◆ = your choice)", type="category"), y=C.axis(f"{lvl} VaR", fmt="pct", digits=1),
        hlines=[C.hline(var_hist, "historical VaR")], stacked=True, height=320,
        subtitle="Low threshold: more days count as jumps and the diffusion becomes calmer – and vice versa.",
    ))

    # OU coordinates of |r|
    w = params["window"]
    if n >= 2 * w + 20:
        X, _, Z = compute_ou_coordinates(r, window=w)
        valid = np.arange(2 * w, n)
        xs = [x[i] for i in valid]
        xv = X[valid]
        q99 = float(np.percentile(xv, 99))
        clip = q99 > 0 and float(xv.max()) > 2.5 * q99  # one estimation spike would flatten everything else
        top = min(1.02, 1.6 * q99) if clip else 1.02
        n_cut = int(np.sum(xv > top))
        peak = fmt_date(xs[int(np.argmax(xv))])
        out.chart(C.xy(
            f"OU coordinate X: volatility level ({w}-day window)",
            [C.trace("X: level μ of |return|", xs, xv, color=1, width=1.5)],
            x=xa, y=C.axis("Scaled (0 = minimum, 1 = maximum)", range=(0, top)), height=260,
            subtitle="Long-run level μ of the OU fit to |return| in the window. If the AR(1) slope b is close to 1 "
                     "(for instance because a shock sits at the end of the window), μ = a/(1 − b) becomes unstable – "
                     "such spikes are estimation errors, not a genuine volatility level."
                     + (f" {n_cut:,} {'day extends' if n_cut == 1 else 'days extend'} beyond the top of the chart "
                        f"(maximum 1 on {peak if data.dates else f'trading day {peak}'}); double-click to show "
                        "everything." if clip else ""),
        ))
        out.chart(C.xy(
            f"OU coordinate Z: mean-reversion speed θ ({w}-day window)",
            [C.trace("Z: θ of |return|", xs, Z[valid], color=1, width=1)],
            x=xa, y=C.axis("Scaled (0 = minimum, 1 = maximum)", range=(0, 1.02)), height=260,
            subtitle="1 = |return| is white noise in the window (θ capped); dips = volatility shocks persist "
                     "(volatility clustering).",
        ))
    else:
        out.note(f"The series is too short for OU coordinates with a {w}-day window (at least "
                 f"{2 * w + 20:,} days required). Reduce the window to at most {(n - 20) // 2:,} days.")

    rows = [["Model: OU + jumps (Monte Carlo)", dec["var_total"]]]
    if exact is not None:
        rows.append(["Model: OU + jumps (analytical check)", exact])
    rows += [["Diffusion only (OU)", dec["var_continuous"]],
             ["Jump add-on", dec["var_jump"]], ["Historical (empirical quantile)", var_hist],
             ["Normal distribution (μ − z·σ)", var_gauss]]
    out.table(f"{lvl} one-day VaR compared", ["Method", "VaR"], rows, formats=["text", "pct"],
              note=f"{n_sim:,} scenarios, seed {seed}. OU fit: "
                   f"{FIT_STATUS[dec['ou_fit_status']]}. Analytical = exact Poisson mixture of normal distributions "
                   "for the same model; the gap to the Monte Carlo figure is simulation noise.")
    order = np.argsort(-np.abs(r - mean))
    top = [i for i in order if is_jump[i]][:15]
    if top:
        out.table("Largest jumps", ["Date" if data.dates else "Day", "Return", "σ multiple"],
                  [[fmt_date(x[i]), r[i], abs(r[i] - mean) / std] for i in top], formats=["text", "pct", "num"])

    if not has_model:
        out.note("There were too few calm days or no mean reversion – without an OU model, the model VaR is "
                 "simply the historical VaR.", "warning")
        return
    if var_hist > 0 and dec["var_total"] < 0.85 * var_hist:
        out.note(f"The model is {_num(100 * (1 - dec['var_total'] / var_hist), 0)}% below the historical VaR. "
                 "It has no volatility clustering: calm and turbulent phases are averaged into a single "
                 "diffusion, and jumps arrive evenly rather than in clusters.", "warning")
    if int(dec["n_jumps"]) == 0:
        out.set_verdict("good", f"No jumps beyond {_num(thr, 1)}σ",
                        "Tail risk here comes entirely from diffusion.")
        return
    if share is None:
        return
    if share > 0.25:
        out.set_verdict("serious", "A large part of the tail risk lies in jumps",
                        f"{_num(100 * share, 0)}% of the one-day VaR is attributable to jumps – a pure diffusion "
                        "model materially understates risk.")
    elif share > 0.1:
        out.set_verdict("warning", "Jumps raise the VaR noticeably",
                        f"{_num(100 * share, 0)}% of the one-day VaR is attributable to jumps.")
    else:
        out.set_verdict("good", "The VaR is driven mainly by diffusion",
                        f"Only {_num(100 * max(share, 0), 0)}% of the one-day VaR is attributable to jumps. "
                        "Expected Shortfall and longer horizons may look different.")
