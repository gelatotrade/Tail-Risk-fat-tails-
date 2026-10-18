"""The 3D risk visualisers of src/visualization/risk_surface_3d.py: VaR surface, model surface, joint tails,
approach to a critical point and the comprehensive 3D dashboard."""

import numpy as np

from src.models.fat_tail_distributions import StudentTDistribution
from src.models.risk_metrics import TailRiskMetrics
from src.physics.levy_flight import estimate_tail_index, levy_flight_3d_coordinates
from src.physics.phase_transitions import CriticalSlowingDownDetector, compute_phase_space_coordinates
from src.physics.tsallis_statistics import compute_tsallis_coordinates
from src.visualization.risk_surface_3d import (CriticalPointVisualizer, TailRisk3DSurface, TailRiskDensityPlot,
                                               create_comprehensive_3d_dashboard)

from .. import charts as C
from ..data import fmt_date
from ..core import ToolError

SPEC = {
    "id": "risk-surfaces",
    "title": "3D risk surfaces of the framework",
    "group": "surface",
    "order": 20,
    "summary": "Five perspectives in 3D: VaR across horizon and confidence, a Student-t crash surface, joint tails, "
               "the approach to the critical point and the 3D phase-space dashboard.",
    "description": (
        "The `risk_surface_3d` module bundles the framework's 3D charts. Choose the **view** above: "
        "**VaR surface** shows historical VaR across horizon × confidence – the original scales it with the "
        "√t rule; alongside it you see the multi-day VaR actually measured. **Model surface** runs "
        "`create_risk_surface` with Student-t tails: P(at least one one-day loss beyond the threshold within "
        "30 days) across volatility and tail index ν – unlike the README surface, the risk does not fall to zero "
        "for thin tails (large ν). **Joint tails** checks whether extreme days occur together (today and "
        "yesterday, or return and VIX spike). **Critical point** tracks AC(1), variance and skewness on the way "
        "to a tipping point. **Physics dashboard** shows the Lévy, Tsallis and phase-transition spaces.\n\n"
        "In every view, the interactive chart at the top is the main display (drag to rotate, scroll to zoom); "
        "the original figure from `src/visualization` follows below it. Only the parameters that apply to the "
        "selected view are shown."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "view", "type": "select", "default": "var", "label": "View",
         "options": [{"value": "var", "label": "VaR surface: horizon × confidence"},
                     {"value": "model", "label": "Model surface: Student-t crash risk"},
                     {"value": "joint", "label": "Joint tails"},
                     {"value": "critical", "label": "Approach to the critical point"},
                     {"value": "physics", "label": "Physics dashboard (Lévy, Tsallis, phase)"}]},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Period", "unit": "days", "help": "The most recent trading days of the dataset."},
        {"name": "max_horizon", "type": "int", "default": 20, "min": 2, "max": 250, "step": 1,
         "label": "Longest horizon", "unit": "days", "show_if": {"view": ["var"]}},
        {"name": "threshold", "type": "float", "default": 5.0, "min": 2.0, "max": 15.0, "step": 0.5,
         "label": "Crash threshold (one-day loss)", "unit": "%", "show_if": {"view": ["model"]}},
        {"name": "partner", "type": "select", "default": "lag", "label": "Second series",
         "show_if": {"view": ["joint"]},
         "options": [{"value": "lag", "label": "Previous day's return"}, {"value": "vix", "label": "VIX change"}]},
        {"name": "ews_window", "type": "int", "default": 50, "min": 20, "max": 250, "step": 5,
         "label": "Window for AC(1), variance, skewness", "unit": "days", "show_if": {"view": ["critical"]}},
        {"name": "ac_threshold", "type": "float", "default": 0.7, "min": 0.3, "max": 0.99, "step": 0.01,
         "label": "AC(1) threshold for ‘near critical’", "help": "Original: 0.7.",
         "show_if": {"view": ["critical"]}},
    ],
    "covers": [
        "src.visualization.risk_surface_3d.TailRisk3DSurface",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_risk_surface",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_phase_space_trajectory",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_multi_coordinate_view",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot.create_tail_probability_surface",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot.create_joint_tail_density",
        "src.visualization.risk_surface_3d.CriticalPointVisualizer",
        "src.visualization.risk_surface_3d.CriticalPointVisualizer.create_critical_surface",
        "src.visualization.risk_surface_3d.create_comprehensive_3d_dashboard",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.tail_dependence",
        "src.models.fat_tail_distributions.StudentTDistribution.cdf",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.compute_ews",
    ],
    "cost": "medium",
}

FIG_PX = 900       # target width of the original figures (they are shown at about 860 px)

def _num(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}%".replace("-", "−")




def _x(data, k):
    if data.dates:
        return list(data.dates[-k:]), C.axis("", type="date")
    return list(range(data.n - k + 1, data.n + 1)), C.axis("Trading day", fmt="int")


def _fig(out, fig, caption):
    dpi = int(np.clip(FIG_PX / fig.get_figwidth(), 48, 88))   # sharp 10-inch figures, light 20-inch dashboards
    fig.set_dpi(dpi)
    out.figure(fig, caption, dpi=dpi)


def run(params, data, out):
    k = int(min(params["last_n"], data.n))
    if k < 120:
        raise ToolError(f"The 3D surfaces need at least 120 daily returns; the dataset has {data.n}.")
    r = np.asarray(data.returns[-k:], dtype=float)
    {"var": _var_view, "model": _model_view, "joint": _joint_view, "critical": _critical_view,
     "physics": _physics_view}[params["view"]](params, data, out, r, k)


# --------------------------------------------------------------------------- #
# VaR surface: horizon x confidence (TailRiskDensityPlot.create_tail_probability_surface)
# --------------------------------------------------------------------------- #
def _var_view(params, data, out, r, k):
    hmax = int(min(params["max_horizon"], k // 4))
    horizons = np.arange(1, hmax + 1, dtype=float) if hmax <= 60 else np.unique(
        np.round(np.geomspace(1, hmax, 60))).astype(float)
    confs = np.linspace(0.90, 0.995, 20)
    tm = TailRiskMetrics(r)
    # identical to create_tail_probability_surface: VaR(conf) * sqrt(horizon)
    var1 = np.array([tm.var_historical(c) for c in confs])
    surface = var1[:, None] * np.sqrt(horizons)[None, :]

    # what the horizon really does: VaR of the overlapping h-day sums of log returns
    csum = np.concatenate([[0.0], np.cumsum(r)])

    def emp_var(h, conf):
        h = int(h)
        sums = csum[h:] - csum[:-h]
        return float(-np.percentile(sums, (1 - conf) * 100)) if len(sums) >= 50 else np.nan

    v99_1, v99_10 = float(tm.var_historical(0.99)), float(tm.var_historical(0.99) * np.sqrt(min(10, hmax)))
    e99_10 = emp_var(min(10, hmax), 0.99)
    out.metric("99% VaR, 1 day", v99_1, "pct", digits=2, hint="historical, from the days shown")
    out.metric(f"99% VaR, {min(10, hmax)} days (√t)", v99_10, "pct", digits=1, hint="as the surface computes it")
    out.metric(f"99% VaR, {min(10, hmax)} days measured", e99_10, "pct", digits=1,
               hint="from overlapping multi-day returns",
               status="warning" if np.isfinite(e99_10) and e99_10 > 1.15 * v99_10 else None)
    if np.isfinite(e99_10) and v99_10 > 0:
        out.metric("Measured ÷ √t", e99_10 / v99_10, "x",
                   hint="> 1: the √t rule understates multi-day risk (volatility clustering)")

    out.chart(C.surface(
        "VaR surface: horizon × confidence (√t scaling)", horizons, confs, surface,
        x=C.axis("Horizon (trading days)", fmt="int"), y=C.axis("Confidence level", fmt="pct", digits=1),
        z=C.axis("VaR", fmt="pct", digits=0), scale="risk", height=540,
        subtitle="z = historical one-day VaR × √horizon – exactly the values of create_tail_probability_surface.",
    ))
    hs = horizons[horizons <= hmax]
    traces = []
    for i, conf in enumerate((0.99, 0.95)):
        emp = [emp_var(h, conf) for h in hs]
        traces.append(C.trace(f"{int(conf * 100)}% measured", hs, emp, color=1 + i, width=2))
        traces.append(C.trace(f"{int(conf * 100)}% √t rule", hs, tm.var_historical(conf) * np.sqrt(hs),
                              color=1 + i, dash=True, width=1.5))
    out.chart(C.xy(
        "√t rule check: measured multi-day VaR vs scaling",
        traces, x=C.axis("Horizon (trading days)", fmt="int"), y=C.axis("VaR", fmt="pct", digits=0),
        subtitle="Solid: VaR of the actual h-day returns. Dashed: one-day VaR × √h.",
    ))
    rows = []
    for h in [h for h in (1, 5, 10, 20, 60, 120, 250) if h <= hmax]:
        row = [h]
        for conf in (0.95, 0.99, 0.995):
            row += [tm.var_historical(conf) * np.sqrt(h), emp_var(h, conf)]
        rows.append(row)
    out.table("VaR by horizon: √t rule and measured", [
        "Horizon (days)", "95% √t", "95% measured", "99% √t", "99% measured", "99.5% √t", "99.5% measured"],
        rows, formats=["int", "pct", "pct", "pct", "pct", "pct", "pct"],
        note="Log returns, losses positive. Measured: percentile of the overlapping h-day sums – few independent "
             "observations at long horizons.")
    fig = TailRiskDensityPlot(figsize=(10, 8)).create_tail_probability_surface(r, horizons, confs)
    _fig(out, fig, "Original: TailRiskDensityPlot.create_tail_probability_surface")


# --------------------------------------------------------------------------- #
# Model surface: TailRisk3DSurface.create_risk_surface with Student-t tails
# --------------------------------------------------------------------------- #
def _model_view(params, data, out, r, k):
    thr = float(params["threshold"]) / 100.0
    horizon = 30
    vols = np.linspace(0.05, 0.80, 28)          # annualised
    nus = np.linspace(2.2, 8.0, 28)              # tail index = degrees of freedom
    cache = {}

    def crash_prob(vol, nu):
        """P(at least one daily loss > thr in 30 days), Student-t with std = vol/sqrt(252)."""
        key = (round(float(vol), 10), round(float(nu), 10))
        if key not in cache:
            sd = float(vol) / np.sqrt(252.0)
            dist = StudentTDistribution(nu=float(nu), loc=0.0, scale=sd * np.sqrt((nu - 2.0) / nu))
            p_day = float(dist.cdf(np.log1p(-thr)))
            cache[key] = 1.0 - (1.0 - p_day) ** horizon
        return cache[key]

    Z = np.array([[crash_prob(v, a) for v in vols] for a in nus])
    w = min(60, k)
    vol_now = float(np.std(r[-w:]) * np.sqrt(252))
    try:
        nu_raw = float(estimate_tail_index(r[-w:]))
    except ValueError:
        nu_raw = np.nan
    nu_now = float(np.clip(nu_raw, nus[0], nus[-1])) if np.isfinite(nu_raw) else nus[-1]
    p_now = crash_prob(np.clip(vol_now, vols[0], vols[-1]), nu_now)
    gauss = 1 - (1 - float(StudentTDistribution(nu=1e6, scale=vol_now / np.sqrt(252)).cdf(np.log1p(-thr)))) ** horizon

    thr_txt = _pct(thr, 1 if thr * 1000 % 10 else 0)
    out.metric(f"Crash probability ({horizon} days)", p_now, "pct", digits=1,
               hint=f"P(at least one one-day loss > {thr_txt}) at today's volatility and ν = Hill α")
    out.metric("Volatility p.a. (60 days)", vol_now, "pct", digits=1)
    out.metric("Tail index ν (Hill, 60 days)", nu_raw, "num", digits=2,
               hint=f"{_num(nu_now)} on the surface" if abs(nu_now - nu_raw) > 1e-9 else "lower = fatter tails")
    if gauss >= 1e-5:
        out.metric("Same volatility, normal distribution", gauss, "pct", digits=1,
                   hint="ν → ∞: what a Gaussian model would say")
    else:
        exp = str(int(np.floor(np.log10(max(gauss, 1e-300))))).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        out.metric("Same volatility, normal distribution", "< 0.001%", "text",
                   hint=f"ν → ∞: what a Gaussian model would say (≈ 10{exp})")

    out.chart(C.surface(
        f"Student-t crash surface: P(one-day loss > {thr_txt} within {horizon} days)", vols, nus, Z,
        x=C.axis("Volatility p.a.", fmt="pct", digits=0, range=(vols[-1], vols[0])),
        y=C.axis("Tail index ν", range=(nus[-1], nus[0])),
        z=C.axis("Probability", fmt="pct", digits=0, range=(0, 1.05)),
        points=[C.path3d("Today", [np.clip(vol_now, vols[0], vols[-1])], [nu_now], [p_now + 0.012], color="ink",
                         mode="markers", size=8, text=[f"{_pct(p_now)}"])],
        scale="risk", height=560,
        camera={"eye": {"x": 1.2, "y": 1.7, "z": 1.0}},   # same view as the README surface (same axes)
        subtitle="The tail index ν does not change the volatility (standard deviation), only how it is distributed: "
                 "small ν = rarer but more violent moves.",
    ))
    rows = [[_pct(v, 0)] + [crash_prob(v, nu) for nu in (2.5, 3.0, 4.0, 6.0, 8.0)] for v in (0.10, 0.20, 0.30, 0.50)]
    out.table(f"Crash probability ({horizon} days) by volatility and ν",
              ["Volatility p.a.", "ν = 2.5", "ν = 3", "ν = 4", "ν = 6", "ν = 8"], rows,
              formats=["text", "pct", "pct", "pct", "pct", "pct"],
              note="At low volatility, fat tails (small ν) raise crash risk; at very high volatility, a large ν can "
                   "even mean more medium-sized losses – the variance is fixed, after all.")
    out.note("Difference from the README surface (‘3D tail-risk surface’): there, only the power-law tail of a "
             "Lévy distribution (α ≤ 2) counts, and it vanishes at α = 2. Here the variance is fixed and ν > 2 – "
             "even near-Gaussian tails produce crash days at high volatility.")
    fig = TailRisk3DSurface(figsize=(10, 8)).create_risk_surface(
        vols, nus, crash_prob, title=f"P(daily loss > {thr * 100:g}% within 30 days), Student-t")
    _fig(out, fig, "Original: TailRisk3DSurface.create_risk_surface (Student-t risk function)")


# --------------------------------------------------------------------------- #
# Joint tails: TailRiskDensityPlot.create_joint_tail_density + TailRiskMetrics.tail_dependence
# --------------------------------------------------------------------------- #
def _joint_view(params, data, out, r, k):
    if params["partner"] == "vix":
        if data.vix is None:
            raise ToolError("The active dataset contains no VIX. Choose ‘Previous day's return’ or a dataset "
                            "with VIX (e.g. S&P 500).")
        vix = np.asarray(data.vix[-k:], dtype=float)
        a, b = r[1:], np.diff(np.log(vix))
        names = ("Return", "VIX change")
        low = ("return", "VIX change")              # the same names inside a sentence
        lab_a, lab_b = "Daily return (today)", "VIX change (log, today)"
    else:
        a, b = r[1:], r[:-1]
        names = ("Today", "Previous day")
        low = ("today", "previous day")
        lab_a, lab_b = "Daily return today", "Daily return on the previous day"
    vix_mode = params["partner"] == "vix"
    tm = TailRiskMetrics(a)
    # the VIX rises when stocks fall: its "crash tail" is the upper one, so the dependence is measured on -ΔVIX
    dep = tm.tail_dependence(-b if vix_mode else b, quantile=0.95)
    corr = -dep["standard_correlation"] if vix_mode else dep["standard_correlation"]
    qa, qb = np.percentile(a, 5), np.percentile(b, 5)
    qb_hi = np.percentile(b, 95)
    joint_low = (a < qa) & (b < qb)
    joint_cross = (a < qa) & (b > qb_hi)       # crash today with a VIX spike / a rally the day before
    out.metric("Correlation", corr, "num", digits=2)
    if vix_mode:
        out.metric("Crash and VIX spike (5%)", dep["lower_tail_dependence"], "pct", digits=1,
                   hint="P(return below its 5% quantile and VIX change above its 95% quantile) ÷ 5% · "
                        "independent: 5%")
        out.metric("Rally and VIX drop (5%)", dep["upper_tail_dependence"], "pct", digits=1,
                   hint="P(return above its 95% quantile and VIX change below its 5% quantile) ÷ 5%")
        out.metric("Asymmetry (crash − rally)", dep["tail_asymmetry"], "pct", digits=1,
                   hint="> 0: crashes come with a VIX spike more often than rallies with a VIX drop")
    else:
        out.metric("Joint left tail (5%)", dep["lower_tail_dependence"], "pct", digits=1,
                   hint="P(both below their 5% quantile) ÷ 5% · independent: 5%")
        out.metric("Joint right tail (5%)", dep["upper_tail_dependence"], "pct", digits=1,
                   hint="P(both above their 95% quantile) ÷ 5%")
        out.metric("Asymmetry (left − right)", dep["tail_asymmetry"], "pct", digits=1,
                   hint="> 0: sell-offs occur together more often than rallies")

    dates = data.dates[-k:][1:] if data.dates else None
    normal = ~(joint_low | joint_cross)
    text = (lambda m: [dates[i] for i in np.flatnonzero(m)]) if dates else (lambda m: None)
    traces = [C.trace("All days",a[normal], b[normal], type="scatter", color=1, size=4, decimate_to=None,
                      text=text(normal))]
    if joint_low.any():
        traces.append(C.trace("Both in the left 5% tail",a[joint_low], b[joint_low], type="scatter",
                              color="critical", size=7, decimate_to=None, text=text(joint_low)))
    if joint_cross.any():
        traces.append(C.trace(f"{names[0]} in the left, {low[1]} in the right tail", a[joint_cross],
                              b[joint_cross], type="scatter", color=2, size=6, decimate_to=None,
                              text=text(joint_cross)))
    out.chart(C.xy(
        f"Joint tails: {low[0]} vs {low[1]}", traces,
        x=C.axis(lab_a, fmt="pct", digits=1), y=C.axis(lab_b, fmt="pct", digits=1),
        vlines=[C.vline(float(qa), "5% quantile")], hlines=[C.hline(float(qb), "5% quantile"),
                                                          C.hline(float(qb_hi), "95% quantile")],
        height=440,
        subtitle=(f"Independent series would put only 0.25% of days in the upper-left corner (crash and VIX spike) – "
                  f"here it is {_pct(float(joint_cross.mean()), 2)}." if vix_mode else
                  f"Independent series would put only 0.25% of days in the lower-left corner (two loss days in a "
                  f"row) – here it is {_pct(float(joint_low.mean()), 2)}."),
    ))
    rows = []
    for q in (0.10, 0.05, 0.01):
        both = float(np.mean((a < np.percentile(a, 100 * q)) & (b < np.percentile(b, 100 * q))))
        cross = float(np.mean((a < np.percentile(a, 100 * q)) & (b > np.percentile(b, 100 * (1 - q)))))
        key = cross if vix_mode else both
        rows.append([_pct(q, 0), both, cross, q * q, key / (q * q) if q > 0 else None])
    out.table("Joint extreme days vs independence", [
        "Tail", "Both left", f"{names[0]} left, {low[1]} right", "Expected if independent",
        "Factor (crash and VIX spike)" if vix_mode else "Factor (both left)"],
        rows, formats=["text", "pct", "pct", "pct", "x"],
        note=("Factor = observed ÷ expected if independent. A crash and a VIX spike on the same day are the rule, "
              "not the exception – which is why the VIX works as a hedging signal." if vix_mode else
              "Factor > 1: extreme days cluster (volatility clustering); the left/right column counts crash days "
              "directly after a strong up day."))
    fig = TailRiskDensityPlot(figsize=(10, 8)).create_joint_tail_density(a, b, asset_names=names,
                                                                         title="Joint Tail Density")
    _fig(out, fig, "Original: TailRiskDensityPlot.create_joint_tail_density")


# --------------------------------------------------------------------------- #
# Critical point: CriticalPointVisualizer.create_critical_surface on the framework's EWS
# --------------------------------------------------------------------------- #
def _critical_view(params, data, out, r, k):
    w = int(params["ews_window"])
    if k < w + 60:
        raise ToolError(f"A window of {w} days needs at least {w + 60} days in the period.")
    ews = CriticalSlowingDownDetector(window=w).compute_ews(r)
    ac = np.asarray(ews["autocorrelation"], dtype=float).copy()
    var = np.asarray(ews["variance"], dtype=float).copy()
    skew = np.asarray(ews["skewness"], dtype=float).copy()
    ac[:w] = var[:w] = skew[:w] = np.nan                 # no full window yet
    thr = float(params["ac_threshold"])
    var_cut = float(np.nanpercentile(var, 80))
    crit = (ac > thr) & (var > var_cut)
    x, xa = _x(data, k)
    idx = np.flatnonzero(np.isfinite(ac))
    var_pct_now = float(np.mean(var[idx] <= var[-1]))
    ac_pct_now = float(np.mean(ac[idx] <= ac[-1]))
    above_thr = float(np.mean(ac[idx] > thr))
    crit_share = float(np.mean(crit[idx]))
    # AC(1) of a detrended price is above 0.7 almost always - only a high value *relative to its history* is a signal
    out.metric("AC(1) today", ac[-1], "num", digits=2,
               hint=f"of the detrended log price · percentile rank {_num(100 * ac_pct_now, 0)}%",
               status="warning" if ac[-1] > thr and ac_pct_now >= 0.8 else None)
    out.metric("Variance percentile today", var_pct_now, "pct", digits=0, hint="relative to the period shown",
               status="warning" if var_pct_now > 0.8 else None)
    out.metric("Skewness today", skew[-1], "num", digits=2, hint=f"of daily returns over {w} days")
    out.metric("Days ‘near critical’", int(crit.sum()), "int",
               hint=f"AC(1) > {_num(thr)} and variance > 80th percentile · {_pct(crit_share, 1)} of days")
    if crit.any():
        last = x[int(np.flatnonzero(crit)[-1])]
        out.metric("Last ‘near critical’ day",fmt_date(last) if isinstance(last, str) else f"Day {last}", "text")

    step = max(1, len(idx) // 900)
    keep = idx[::step]
    recent = idx[-min(60, len(idx)):]
    label = (lambda ii: [x[i] for i in ii]) if data.dates else (lambda ii: [f"Day {x[i]}" for i in ii])
    traces = [C.path3d("History", ac[keep], var[keep], skew[keep], color=1, mode="markers", size=2, text=label(keep)),
              C.path3d("Last 60 days",ac[recent], var[recent], skew[recent], color="ink", mode="lines+markers",
                       size=2, text=label(recent))]
    if crit.any():
        ci = np.flatnonzero(crit)
        traces.append(C.path3d("Near critical",ac[ci], var[ci], skew[ci], color="critical", mode="markers", size=4,
                               text=label(ci)))
    out.chart(C.scatter3d(
        "Approach to the critical point: AC(1) × variance × skewness", traces,
        x=C.axis("AC(1)"), y=C.axis("Variance (detrended)"), z=C.axis("Skewness"), height=560,
        subtitle="Critical slowing down: AC(1) and variance rise together while skewness turns negative. "
                 "Red: days with high AC(1) and variance in the top fifth.",
    ))
    tr = [C.trace("AC(1)", x, ac, color=1, width=1.2)]
    if crit.any():
        ci = np.flatnonzero(crit)
        tr.append(C.trace("Near critical",[x[i] for i in ci], ac[ci], type="scatter", color="critical", size=5,
                          decimate_to=None))
    out.chart(C.xy(f"AC(1) over time ({w}-day window)", tr, x=xa, y=C.axis("AC(1)"),
                   hlines=[C.hline(thr, f"Threshold {_num(thr)}", color="warning")],
                   subtitle="Red: days on which the variance is also above its 80th percentile."))
    mech = (f" Here it is above {_num(thr)} on {_pct(above_thr, 0)} of days – so ‘near critical’ effectively just "
            "means ‘variance in the top fifth’, which by definition applies to 20% of days."
            if above_thr > 0.9 else "")
    out.note("AC(1) is measured on the detrended log price (as in the *Critical slowing down* tool); even a pure "
             "random walk has high values there." + mech + " What matters is the joint rise of AC(1) and variance, "
             "not the level alone.")
    fig = CriticalPointVisualizer(figsize=(10, 8)).create_critical_surface(ac, var, skew, np.arange(k, dtype=float),
                                                                           ac_threshold=thr)
    _fig(out, fig, "Original: CriticalPointVisualizer.create_critical_surface")


# --------------------------------------------------------------------------- #
# Physics dashboard: create_comprehensive_3d_dashboard and TailRisk3DSurface views
# --------------------------------------------------------------------------- #
def _physics_view(params, data, out, r, k):
    k = min(k, 2520)
    r = r[-k:]
    out.progress("Lévy, Tsallis and phase coordinates …", 0.1)
    levy = levy_flight_3d_coordinates(r)
    tsallis = compute_tsallis_coordinates(r)
    phase = compute_phase_space_coordinates(r)
    warm = min(100, k // 3)                       # the coordinates are 0 until their windows are full
    risk = (levy[2] + 0.5 * tsallis[0] + phase[0]) / 2.5   # composite of create_comprehensive_3d_dashboard
    rv = risk[warm:]
    lo, mid = np.percentile(rv, [50, 90])
    x, xa = _x(data, k)
    xs = x[warm:]
    out.metric("Composite today", rv[-1], "num", digits=2, hint="(λ + 0.5·q + χ) / 2.5")
    out.metric("Percentile today", float(np.mean(rv <= rv[-1])), "pct", digits=0, hint="relative to the period")
    out.metric("Lévy volatility today", levy[0][-1], "num", digits=2, hint="σ ÷ avg. σ · 1 = average volatility")
    out.metric("Tsallis q today", tsallis[0][-1], "num", digits=2, hint="1 = Gaussian, higher = fatter tails")

    traces = []
    for name, mask, col in (("Composite below median", rv <= lo, "good"),
                            ("Median to 90th percentile", (rv > lo) & (rv <= mid), "warning"),
                            ("Top 10%", rv > mid, "critical")):
        ii = np.flatnonzero(mask)
        ii = ii[::max(1, len(ii) // 700)]
        traces.append(C.path3d(name, levy[0][warm:][ii], tsallis[1][warm:][ii], rv[ii], color=col,
                               mode="markers", size=2.5, text=[str(xs[i]) for i in ii]))
    traces.append(C.path3d("Today", [levy[0][-1]], [tsallis[1][-1]], [rv[-1]], color="ink", mode="markers", size=7))
    out.chart(C.scatter3d(
        "Composite risk in volatility–entropy space", traces,
        x=C.axis("Volatility σ ÷ avg. σ (Lévy)"), y=C.axis("Tsallis entropy S_q"), z=C.axis("Composite risk"),
        height=540,
        subtitle="Panel 4 of the 3D dashboard, rotatable. Colour shows the composite percentile (median, 90%).",
    ))
    out.chart(C.xy("Composite risk over time", [C.trace("Composite", xs, rv, color=1, width=1)],
                   x=xa, y=C.axis("Composite"), hlines=[C.hline(float(mid), "90th percentile", color="warning")],
                   subtitle="The composite mainly tracks today's daily move |r|/σ (λ)."))
    out.note("The interactive individual spaces (Lévy, Tsallis, Fokker-Planck) with date hover are in the "
             "*Physics phase spaces in 3D* tool. This view shows the original figures of the `risk_surface_3d` module.")
    out.progress("Drawing the original figures …", 0.5)
    fig = create_comprehensive_3d_dashboard(r)    # no save_path: nothing is written
    _fig(out, fig, "Original: create_comprehensive_3d_dashboard")
    vis = TailRisk3DSurface(figsize=(10, 8))
    trim = slice(warm, None)
    fig = vis.create_multi_coordinate_view(tuple(c[trim] for c in levy), tuple(c[trim] for c in tsallis),
                                           tuple(c[trim] for c in phase))
    _fig(out, fig, "Original: TailRisk3DSurface.create_multi_coordinate_view (without the warm-up period)")
    m = min(250, k - warm)
    fig = vis.create_phase_space_trajectory(levy[0][-m:], levy[1][-m:], levy[2][-m:],
                                            time=np.arange(m, dtype=float),
                                            title=f"Lévy phase-space trajectory (last {m} days)",
                                            axis_labels=("σ / mean σ", "2 / α (tail heaviness)", "|r| / σ (jump)"))
    _fig(out, fig, f"Original: TailRisk3DSurface.create_phase_space_trajectory (last {m} days in Lévy space)")
