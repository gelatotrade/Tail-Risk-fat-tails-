"""Tail risk parity: allocate so that every asset carries the same share of Expected Shortfall."""

import numpy as np

from src.models.risk_metrics import TailRiskMetrics, TailRiskParity
from src.utils.data_loader import SyntheticMarketGenerator

from .. import charts as C

LEVELS = [{"value": v, "label": l} for v, l in ((0.9, "90%"), (0.95, "95%"), (0.975, "97.5%"), (0.99, "99%"))]

SPEC = {
    "id": "tail-risk-parity",
    "title": "Tail risk parity",
    "group": "risk",
    "order": 70,
    "summary": "Allocates a portfolio so that every asset contributes equally to Expected Shortfall – compared "
               "with equal weights and classic volatility parity.",
    "description": (
        "Classic risk parity allocates by **volatility**. With fat tails, however, what matters is who loses on the "
        "worst days. `TailRiskParity` measures this through the portfolio's **Expected Shortfall** (ES): an asset's "
        "**marginal ES** is its average loss on the days when the portfolio is beyond its VaR; **risk "
        "contribution** = weight × marginal ES. The contributions add up exactly to the portfolio ES. The "
        "optimisation searches for weights at which each contribution makes up 1/n of the total.\n\n"
        "The markets come from `SyntheticMarketGenerator`: correlated assets with different volatilities and "
        "Student-t-like tails. The **base correlation** mixes in a common market factor (equities: about "
        "0.3–0.5). Change the seed for a new market, or vary the number of assets or the confidence level.\n\n"
        "How to read the charts: with equal weights, volatile and highly correlated assets carry a much larger "
        "share of the crash risk. Tail risk parity shifts weight towards the assets that lose little on stress "
        "days. If an asset can even gain on stress days (a hedge), an exact equal split is not attainable – the "
        "deviation is then shown."
    ),
    "needs": [],
    "params": [
        {"name": "n_assets", "type": "int", "default": 5, "min": 2, "max": 10, "step": 1, "label": "Number of assets"},
        {"name": "n_days", "type": "int", "default": 2000, "min": 500, "max": 5000, "step": 100,
         "label": "Simulated trading days"},
        {"name": "base_correlation", "type": "float", "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
         "label": "Base correlation", "help": "Common market factor: 0 = pairwise correlations around 0; "
                                              "0.3–0.5 is typical for equities."},
        {"name": "confidence", "type": "select", "default": 0.95, "label": "ES confidence level", "options": LEVELS,
         "help": "Higher = only the most extreme days count, but with fewer data points."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.models.risk_metrics.TailRiskParity",
        "src.models.risk_metrics.TailRiskParity.marginal_es",
        "src.models.risk_metrics.TailRiskParity.risk_contribution",
        "src.models.risk_metrics.TailRiskParity.tail_risk_parity_weights",
        "src.utils.data_loader.SyntheticMarketGenerator",
        "src.utils.data_loader.SyntheticMarketGenerator.generate",
    ],
    "cost": "fast",
}


def _num(x, digits=2):
    return f"{x:.{digits}f}"


def run(params, data, out):
    k, n_days, conf, seed = int(params["n_assets"]), int(params["n_days"]), float(params["confidence"]), int(params["seed"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    market = SyntheticMarketGenerator(n_assets=k, seed=seed,
                                      base_correlation=float(params["base_correlation"])).generate(n_days)
    R = np.asarray(market["returns"], dtype=float)
    names = [f"Asset {i + 1}" for i in range(k)]
    trp = TailRiskParity(R, confidence=conf)

    out.progress("Optimising the weights …", 0.3)
    w_trp = np.asarray(trp.tail_risk_parity_weights(), dtype=float)
    vols = R.std(axis=0)
    portfolios = {
        "ew": ("Equal weights", np.ones(k) / k, "muted"),
        "iv": ("Volatility parity", (1 / vols) / np.sum(1 / vols), 2),
        "trp": ("Tail risk parity", w_trp, 1),
    }
    stats_p = {}
    for key, (label, w, _) in portfolios.items():
        rc = np.asarray(trp.risk_contribution(w), dtype=float)
        mes = np.asarray(trp.marginal_es(w), dtype=float)
        port = R @ w
        tm = TailRiskMetrics(port)
        total = float(np.sum(rc))
        shares = rc / total if total > 0 else np.full(k, np.nan)
        stats_p[key] = {"w": w, "rc": rc, "mes": mes, "es": total, "shares": shares,
                        "dev": float(np.nanmax(np.abs(shares - 1 / k))), "vol": float(np.std(port) * np.sqrt(252)),
                        "var": float(tm.var_historical(conf)), "es_hist": float(tm.expected_shortfall(conf)),
                        "mdd": float(tm.maximum_drawdown()["max_drawdown"])}

    ew, tp = stats_p["ew"], stats_p["trp"]
    top_ew = int(np.nanargmax(ew["shares"]))
    out.metric(f"{lvl} portfolio ES · equal weights", ew["es"], "pct", hint="Sum of risk contributions")
    out.metric(f"{lvl} portfolio ES · tail risk parity", tp["es"], "pct")
    out.metric("Largest ES share · equal weights", ew["shares"][top_ew], "pct", digits=0,
               hint=f"{names[top_ew]} · target {_num(100 / k, 0)}%",
               status="serious" if ew["shares"][top_ew] > 2 / k else "warning" if ew["shares"][top_ew] > 1.4 / k else None)
    out.metric("Max. deviation from target share · TRP", tp["dev"], "pct", digits=1,
               status="good" if tp["dev"] < 0.02 else "warning" if tp["dev"] < 0.06 else "serious")
    out.metric("Max. drawdown · equal weights", ew["mdd"], "pct", digits=1,
               hint=f"Tail risk parity: {_num(100 * tp['mdd'], 1)}%")

    traces = lambda field: [C.trace(lbl, names, stats_p[key][field], type="bar", color=col)  # noqa: E731
                            for key, (lbl, _, col) in portfolios.items()]
    out.chart(C.xy("Weights by asset", traces("w"), x=C.axis("", type="category"),
                   y=C.axis("Weight", fmt="pct", digits=0), barmode="group",
                   subtitle="Tail risk parity shifts weight towards assets that lose little on stress days."))
    out.chart(C.xy(f"Share of the {lvl} portfolio ES", traces("shares"), x=C.axis("", type="category"),
                   y=C.axis("Share of Expected Shortfall", fmt="pct", digits=0), barmode="group",
                   hlines=[C.hline(1 / k, f"Target 1/{k}")],
                   subtitle="With tail risk parity, all bars sit on the target line."))
    out.chart(C.xy(f"{lvl} marginal ES: average asset loss on the portfolio's stress days", traces("mes"),
                   x=C.axis("", type="category"), y=C.axis("One-day loss", fmt="pct", digits=1), barmode="group",
                   subtitle="Negative means that the asset gains on average when the portfolio crashes."))
    corr = np.corrcoef(R.T)  # realised: what the optimiser actually sees
    out.chart(C.heatmap("Correlations of the simulated returns", names, names, corr,
                        z=C.axis("Correlation", digits=2, range=(min(0.0, float(corr.min())), 1)), scale="sequential",
                        annotate=k <= 8,
                        subtitle="Highly correlated assets crash together and share the tail.",
                        height=320 + 12 * k))

    rows = []
    for i in range(k):
        r_i = R[:, i]
        tm_i = TailRiskMetrics(r_i)
        rows.append([names[i], vols[i] * np.sqrt(252), float(tm_i.expected_shortfall(conf)),
                     ew["w"][i], portfolios["iv"][1][i], tp["w"][i], ew["shares"][i], tp["shares"][i]])
    out.table("Assets in detail", ["Asset", "Vol. p.a.", f"Stand-alone {lvl} ES", "Weight (equal)",
                                   "Weight (vol. parity)", "Weight (TRP)", "ES share (equal)", "ES share (TRP)"], rows,
              formats=["text", "pct", "pct", "pct", "pct", "pct", "pct", "pct"])
    out.table("Portfolios compared", ["Portfolio", "Vol. p.a.", f"{lvl} VaR", f"{lvl} ES",
                                      "Max. deviation from target share", "Max. drawdown"],
              [[lbl, stats_p[key]["vol"], stats_p[key]["var"], stats_p[key]["es_hist"], stats_p[key]["dev"],
                stats_p[key]["mdd"]] for key, (lbl, _, _) in portfolios.items()],
              formats=["text", "pct", "pct", "pct", "pct", "pct"],
              note=f"{n_days:,} simulated trading days; historical ES. Risk contributions from the "
                   f"{int(round(n_days * (1 - conf))):,} worst portfolio days.")
    if tp["dev"] > 0.05:
        hedge = [names[i] for i in range(k) if tp["mes"][i] <= 0]
        out.note("No exact equal split found"
                 + (f": {', '.join(hedge)} {'loses' if len(hedge) == 1 else 'lose'} nothing on average on stress days "
                    "and therefore cannot carry a positive share." if hedge else
                    ": the ES from a few stress days is a step function of the weights. More trading days or a "
                    "lower confidence level help."), "warning")
