"""Fit every fat-tail family to the data and rank them (KS, AIC), with QQ and log-density views."""

import numpy as np
from scipy import stats

from src.models.fat_tail_distributions import (
    NormalInverseGaussian,
    StudentTDistribution,
    VarianceGamma,
    compare_distributions,
)
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..core import ToolError

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"), (0.995, "99,5 %"), (0.999, "99,9 %"))]

SPEC = {
    "id": "distribution-fit",
    "title": "Verteilungs-Fit & Modellwahl",
    "group": "data",
    "order": 35,
    "summary": "Passt Normal-, Student-t-, NIG- und Variance-Gamma-Verteilung an deine Renditen an und zeigt, "
               "welche die Ränder am besten trifft.",
    "description": (
        "`compare_distributions` schätzt vier Modelle: die **Normalverteilung** und die **Student-t** per "
        "Maximum Likelihood, **NIG** und **Variance Gamma** über die Momentenmethode (Mittelwert, Varianz, Schiefe, "
        "Kurtosis). Der **Kolmogorov-Smirnov-Abstand** misst die größte Lücke zwischen empirischer und Modell-"
        "Verteilungsfunktion; das **AIC** belohnt hohe Likelihood und bestraft Parameter. Kleiner ist bei beiden "
        "besser. Die KS-p-Werte sind optimistisch, weil die Parameter aus denselben Daten stammen.\n\n"
        "So liest du die Diagramme: Im **Q-Q-Plot** liegt ein passendes Modell auf der Diagonale. Knicken die "
        "Punkte an den Enden weg, unterschätzt das Modell die Extreme. Die **logarithmische Dichte** zeigt dasselbe "
        "für die Ränder, das dritte Diagramm direkt die Crash-Häufigkeit P(Rendite < −x).\n\n"
        "VaR und Expected Shortfall je Modell sind Tagesverluste (positive Dezimalzahl): Gauß und Student-t "
        "analytisch, NIG und Variance Gamma per Monte Carlo. Momenten-Fits treffen die Kurtosis, aber nicht "
        "unbedingt die Form; das zeigt sich oft in einem schlechten KS-Wert."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Konfidenzniveau", "options": LEVELS},
        {"name": "bins", "type": "int", "default": 100, "min": 30, "max": 200, "step": 10,
         "label": "Klassen im Histogramm"},
        {"name": "n_mc", "type": "int", "default": 200000, "min": 20000, "max": 1000000, "step": 20000,
         "label": "Monte-Carlo-Ziehungen", "help": "Für Quantile, VaR und ES von NIG und Variance Gamma."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.models.fat_tail_distributions.compare_distributions",
        "src.models.fat_tail_distributions.StudentTDistribution.fit",
        "src.models.fat_tail_distributions.NormalInverseGaussian.fit",
        "src.models.fat_tail_distributions.VarianceGamma.fit",
        "src.models.fat_tail_distributions.StudentTDistribution.ppf",
        "src.models.fat_tail_distributions.StudentTDistribution.cdf",
        "src.models.fat_tail_distributions.StudentTDistribution.var",
        "src.models.fat_tail_distributions.StudentTDistribution.expected_shortfall",
        "src.models.fat_tail_distributions.NormalInverseGaussian.pdf",
        "src.models.fat_tail_distributions.NormalInverseGaussian.sample",
        "src.models.fat_tail_distributions.VarianceGamma.pdf",
        "src.models.fat_tail_distributions.VarianceGamma.sample",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
    ],
    "cost": "medium",
}

NAMES = {"gaussian": "Normalverteilung", "student_t": "Student-t", "nig": "NIG", "variance_gamma": "Variance Gamma"}
COLORS = {"gaussian": "muted", "student_t": 1, "nig": 2, "variance_gamma": 3}
N_PARAMS = {"gaussian": 2, "student_t": 3, "nig": 4, "variance_gamma": 4}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _sci(x):
    return f"{x:.3g}".replace(".", ",")


def _var_es(x, conf):
    var = -float(np.quantile(x, 1 - conf))
    tail = x[x <= -var]
    return var, (-float(np.mean(tail)) if tail.size else var)


def _loglik(pdf, r):
    f = np.asarray(pdf(r), dtype=float)
    return float(np.sum(np.log(np.clip(np.nan_to_num(f, nan=0.0), 1e-300, None))))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    if np.std(r) <= 0:
        raise ToolError("Die Renditen sind konstant – es gibt keine Verteilung anzupassen.")
    conf, seed, n_mc = float(params["confidence"]), int(params["seed"]), int(params["n_mc"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    out.progress("Passe vier Verteilungen an …", 0.1)
    res = compare_distributions(r)
    best = res["best_fit"]

    gp = res["gaussian"]["params"]
    gauss = stats.norm(gp["loc"], gp["scale"])
    tp = res["student_t"]["params"]
    t_dist = StudentTDistribution(nu=tp["nu"], loc=tp["loc"], scale=tp["scale"])
    nig = NormalInverseGaussian.fit(r)
    vg = VarianceGamma.fit(r)

    out.progress("Monte Carlo für NIG und Variance Gamma …", 0.5)
    s_nig = np.asarray(nig.sample(n_mc, seed=seed), dtype=float)
    s_vg = np.asarray(vg.sample(n_mc, seed=seed + 1), dtype=float)

    models = {
        "gaussian": {"pdf": gauss.pdf, "ppf": gauss.ppf, "cdf": gauss.cdf, "sample": None,
                     "par": f"μ = {_de(100 * gp['loc'], 3)} %, σ = {_de(100 * gp['scale'])} %"},
        "student_t": {"pdf": t_dist.pdf, "ppf": t_dist.ppf, "cdf": t_dist.cdf, "sample": None,
                      "par": f"ν = {_de(tp['nu'])}, Skala = {_de(100 * tp['scale'])} %"},
        "nig": {"pdf": nig.pdf, "ppf": None, "cdf": None, "sample": s_nig,
                "par": f"α = {_de(nig.alpha, 1)}, β = {_de(nig.beta, 1)}, δ = {_sci(nig.delta)}"},
        "variance_gamma": {"pdf": vg.pdf, "ppf": None, "cdf": None, "sample": s_vg,
                           "par": f"σ = {_de(100 * vg.sigma)} %, θ = {_sci(vg.theta)}, ν = {_de(vg.nu)}"},
    }
    alpha = 1 - conf
    for key, m in models.items():
        m["ll"] = _loglik(m["pdf"], r)
        m["aic"] = 2 * N_PARAMS[key] - 2 * m["ll"]
        m["ks"] = res[key].get("ks_statistic")
        m["p"] = res[key].get("ks_pvalue")
        if key == "gaussian":
            z = stats.norm.ppf(alpha)
            m["var"], m["es"] = -(gp["loc"] + gp["scale"] * z), -gp["loc"] + gp["scale"] * stats.norm.pdf(z) / alpha
        elif key == "student_t":
            m["var"], m["es"] = float(t_dist.var(conf)), float(t_dist.expected_shortfall(conf))
        else:
            m["var"], m["es"] = _var_es(m["sample"], conf)
    aic_min = min(m["aic"] for m in models.values())
    best_aic = min(models, key=lambda k: models[k]["aic"])
    trm = TailRiskMetrics(r)
    var_h, es_h = float(trm.var_historical(conf)), float(trm.expected_shortfall(conf, "historical"))

    out.metric("Bestes Modell (KS)", NAMES[best], "text", hint=f"KS-Abstand {_de(models[best]['ks'], 3)}",
               status="good" if best != "gaussian" else None)
    out.metric("Bestes Modell (AIC)", NAMES[best_aic], "text")
    out.metric("Student-t: Freiheitsgrade ν", tp["nu"], "num", hint="< 4: unendliche Kurtosis · < 2: unendliche Varianz",
               status="critical" if tp["nu"] < 2.5 else "serious" if tp["nu"] < 4 else "warning" if tp["nu"] < 8 else None)
    out.metric(f"VaR {lvl} · historisch", var_h, "pct", hint=f"ES {_de(100 * es_h)} %")
    out.metric(f"VaR {lvl} · {NAMES[best]}", models[best]["var"], "pct", hint=f"ES {_de(100 * models[best]['es'])} %")
    out.metric(f"VaR {lvl} · Gauß", models["gaussian"]["var"], "pct", hint=f"ES {_de(100 * models['gaussian']['es'])} %")

    order = sorted(models, key=lambda k: models[k]["ks"] if models[k]["ks"] is not None else np.inf)
    rows = []
    for rank, key in enumerate(order, 1):
        m = models[key]
        rows.append([rank, NAMES[key], m["par"], m["ks"], m["p"], m["ll"], m["aic"] - aic_min, m["var"], m["es"]])
    rows.append(["", "Empirisch", f"{n} Tage", None, None, None, None, var_h, es_h])
    out.table(
        "Modellvergleich (sortiert nach KS-Abstand)",
        ["Rang", "Modell", "Parameter", "KS-Abstand", "KS p-Wert", "Log-Likelihood", "ΔAIC", f"VaR {lvl}", f"ES {lvl}"],
        rows, formats=["text", "text", "text", "num", "num", "num", "num", "pct", "pct"],
        note="ΔAIC = AIC − bestes AIC (0 = bestes Modell; > 10 = deutlich schlechter). NIG und Variance Gamma sind "
             "Momenten-Fits, keine Maximum-Likelihood-Schätzungen; ihre Log-Likelihood ist daher nicht maximal.",
    )

    # --- chart 1: histogram vs fitted densities on a log scale
    centers, heights = C.histogram(r, bins=int(params["bins"]), density=True)
    floor = float(np.min(heights[heights > 0])) / 4
    lo, hi = float(r.min()), float(r.max())
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 600)
    traces = [C.trace("Beobachtet", centers, np.where(heights > 0, heights, np.nan), type="bar", color=4)]
    for key in ("gaussian", "student_t", "nig", "variance_gamma"):
        y = np.asarray(models[key]["pdf"](grid), dtype=float)
        y = np.where(np.isfinite(y) & (y >= floor), np.minimum(y, 10 * heights.max()), np.nan)
        traces.append(C.trace(NAMES[key], grid, y, color=COLORS[key], dash=key == "gaussian"))
    out.chart(C.xy(
        "Tagesrenditen und angepasste Dichten – logarithmisch", traces,
        x=C.axis("Tagesrendite", fmt="pct", digits=1), y=C.axis("Dichte", type="log", range=(floor, heights.max() * 3)),
        subtitle="Wo die Balken über einer Kurve liegen, unterschätzt das Modell die Häufigkeit.",
    ))

    # --- chart 2: QQ plot (tails in full, the centre thinned)
    rs = np.sort(r)
    p = (np.arange(1, n + 1) - 0.5) / n
    edge = min(150, n // 4)
    idx = np.unique(np.concatenate([np.arange(edge), np.linspace(edge, n - edge - 1, 200).astype(int),
                                    np.arange(n - edge, n)]))
    qq = [C.trace("Perfekter Fit", [rs[0], rs[-1]], [rs[0], rs[-1]], color="ink", dash=True, width=1)]
    for key in ("gaussian", "student_t", "nig", "variance_gamma"):
        m = models[key]
        q = np.asarray(m["ppf"](p[idx]), dtype=float) if m["ppf"] is not None else np.quantile(m["sample"], p[idx])
        qq.append(C.trace(NAMES[key], q, rs[idx], type="scatter", color=COLORS[key], size=5, decimate_to=None))
    span = (rs[0] * 1.3, rs[-1] * 1.3)
    out.chart(C.xy(
        "Q-Q-Plot: Modell-Quantile gegen beobachtete Quantile", qq,
        x=C.axis("Modell-Quantil", fmt="pct", digits=0, range=span), y=C.axis("Beobachtetes Quantil", fmt="pct", digits=0),
        subtitle="Punkte unter der Diagonale links: Die echten Crashs sind tiefer, als das Modell erwartet.",
    ))

    # --- chart 3: left tail, P(r < -x)
    losses = np.sort(-r[r < 0])[::-1]
    xs = np.geomspace(max(np.median(losses), 1e-4), losses[0] * 1.5, 70)
    emp = np.array([np.mean(r < -x) for x in xs])
    tail = [C.trace("Beobachtet", xs, np.where(emp > 0, emp, np.nan), type="scatter", color=4, size=6, decimate_to=None)]
    for key in ("gaussian", "student_t", "nig", "variance_gamma"):
        m = models[key]
        if m["cdf"] is not None:
            y = np.asarray(m["cdf"](-xs), dtype=float)
        else:
            s = np.sort(m["sample"])
            y = np.searchsorted(s, -xs, side="left") / len(s)
        tail.append(C.trace(NAMES[key], xs, np.where(y > 1e-9, y, np.nan), color=COLORS[key], dash=key == "gaussian"))
    out.chart(C.xy(
        "Crash-Häufigkeit: P(Tagesrendite < −x)", tail,
        x=C.axis("Tagesverlust x", type="log", fmt="pct", digits=0), y=C.axis("Wahrscheinlichkeit pro Tag", type="log",
                                                                                  range=(min(1e-6, 0.3 / n), 1)),
        subtitle="Log-Log-Achsen. Die Normalverteilung bricht ab; die Fat-Tail-Modelle folgen den Daten länger.",
        hlines=[C.hline(1 / 252, "einmal pro Jahr")],
    ))

    if models["gaussian"]["p"] is not None and models["gaussian"]["p"] < 0.05 and best != "gaussian":
        bp = models[best]["p"]
        fits = bp is not None and bp >= 0.05
        out.set_verdict(
            "good" if fits else "warning",
            f"{NAMES[best]} beschreibt die Renditen am besten",
            f"Die Normalverteilung wird abgelehnt (KS p = {_sci(models['gaussian']['p'])}). "
            + (f"{NAMES[best]} wird vom KS-Test nicht abgelehnt (p = {_de(bp, 2)})."
               if fits else f"Aber auch {NAMES[best]} passt nicht perfekt (p = {_sci(bp)}); "
                            "Volatilitäts-Cluster lassen sich mit keiner festen Verteilung erfassen."),
        )
    elif best == "gaussian":
        out.set_verdict("good", "Die Normalverteilung passt hier am besten",
                        "Das ist bei kurzen oder ruhigen Reihen möglich; die Fat-Tail-Modelle bringen keinen Vorteil.")
