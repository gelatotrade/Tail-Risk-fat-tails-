"""Tail risk parity: allocate so that every asset carries the same share of Expected Shortfall."""

import numpy as np

from src.models.risk_metrics import TailRiskMetrics, TailRiskParity
from src.utils.data_loader import SyntheticMarketGenerator

from .. import charts as C

LEVELS = [{"value": v, "label": l} for v, l in ((0.9, "90 %"), (0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"))]

SPEC = {
    "id": "tail-risk-parity",
    "title": "Tail-Risk-Parity",
    "group": "risk",
    "order": 50,
    "summary": "Verteilt ein Portfolio so, dass jedes Asset gleich viel zum Expected Shortfall beiträgt – im "
               "Vergleich mit gleichen Gewichten und klassischer Volatilitäts-Parität.",
    "description": (
        "Klassische Risk Parity verteilt nach **Volatilität**. Bei fetten Rändern zählt aber, wer in den schlimmsten "
        "Tagen verliert. `TailRiskParity` misst das über den **Expected Shortfall** (ES) des Portfolios: Der "
        "**marginale ES** eines Assets ist sein mittlerer Verlust an den Tagen, an denen das Portfolio jenseits seines "
        "VaR liegt; **Risikobeitrag** = Gewicht × marginaler ES. Die Beiträge summieren sich exakt zum Portfolio-ES. "
        "Die Optimierung sucht Gewichte, bei denen jeder Beitrag 1/n des Ganzen ausmacht.\n\n"
        "Die Märkte stammen aus `SyntheticMarketGenerator`: korrelierte Assets mit unterschiedlicher Volatilität und "
        "Student-t-artigen Rändern. Ändere den Seed für eine neue Welt, die Zahl der Assets oder das Konfidenzniveau.\n\n"
        "So liest du die Diagramme: Bei gleichen Gewichten tragen volatile, hoch korrelierte Assets einen viel "
        "größeren Teil des Crash-Risikos. Tail-Risk-Parity verschiebt Gewicht zu den Assets, die in Stresstagen "
        "wenig verlieren. Kann ein Asset in Stresstagen sogar gewinnen (Absicherung), ist eine exakte Gleichverteilung "
        "nicht erreichbar – die Abweichung wird dann angezeigt."
    ),
    "needs": [],
    "params": [
        {"name": "n_assets", "type": "int", "default": 5, "min": 2, "max": 10, "step": 1, "label": "Anzahl Assets"},
        {"name": "n_days", "type": "int", "default": 2000, "min": 500, "max": 5000, "step": 100,
         "label": "Simulierte Handelstage"},
        {"name": "confidence", "type": "select", "default": 0.95, "label": "ES-Konfidenzniveau", "options": LEVELS,
         "help": "Höher = nur die extremsten Tage zählen, aber weniger Datenpunkte."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
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


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def run(params, data, out):
    k, n_days, conf, seed = int(params["n_assets"]), int(params["n_days"]), float(params["confidence"]), int(params["seed"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    market = SyntheticMarketGenerator(n_assets=k, seed=seed).generate(n_days)
    R = np.asarray(market["returns"], dtype=float)
    names = [f"Asset {i + 1}" for i in range(k)]
    trp = TailRiskParity(R, confidence=conf)

    out.progress("Optimiere die Gewichte …", 0.3)
    w_trp = np.asarray(trp.tail_risk_parity_weights(), dtype=float)
    vols = R.std(axis=0)
    portfolios = {
        "ew": ("Gleiche Gewichte", np.ones(k) / k, "muted"),
        "iv": ("Volatilitäts-Parität", (1 / vols) / np.sum(1 / vols), 2),
        "trp": ("Tail-Risk-Parity", w_trp, 1),
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
    out.metric(f"Portfolio-ES {lvl} · gleich gewichtet", ew["es"], "pct", hint="Summe der Risikobeiträge")
    out.metric(f"Portfolio-ES {lvl} · Tail-Risk-Parity", tp["es"], "pct")
    out.metric("Größter ES-Anteil · gleich gewichtet", ew["shares"][top_ew], "pct", digits=0,
               hint=f"{names[top_ew]} · Ziel {_de(100 / k, 0)} %",
               status="serious" if ew["shares"][top_ew] > 2 / k else "warning" if ew["shares"][top_ew] > 1.4 / k else None)
    out.metric("Max. Abweichung vom Zielanteil · TRP", tp["dev"], "pct", digits=1,
               status="good" if tp["dev"] < 0.02 else "warning" if tp["dev"] < 0.06 else "serious")
    out.metric("Max. Drawdown · gleich / TRP", ew["mdd"], "pct", digits=1, hint=f"TRP: {_de(100 * tp['mdd'], 1)} %")

    traces = lambda field: [C.trace(lbl, names, stats_p[key][field], type="bar", color=col)  # noqa: E731
                            for key, (lbl, _, col) in portfolios.items()]
    out.chart(C.xy("Gewichte je Asset", traces("w"), x=C.axis("", type="category"),
                   y=C.axis("Gewicht", fmt="pct", digits=0), barmode="group",
                   subtitle="Tail-Risk-Parity verschiebt Gewicht zu Assets, die in Stresstagen wenig verlieren."))
    out.chart(C.xy(f"Anteil am Portfolio-ES {lvl}", traces("shares"), x=C.axis("", type="category"),
                   y=C.axis("Anteil am Expected Shortfall", fmt="pct", digits=0), barmode="group",
                   hlines=[C.hline(1 / k, f"Ziel 1/{k}")],
                   subtitle="Bei Tail-Risk-Parity liegen alle Balken auf der Ziellinie."))
    out.chart(C.xy(f"Marginaler ES {lvl}: Ø Verlust des Assets an den Stresstagen des Portfolios", traces("mes"),
                   x=C.axis("", type="category"), y=C.axis("Tagesverlust", fmt="pct", digits=1), barmode="group",
                   subtitle="Negativ heißt: Das Asset gewinnt im Mittel, wenn das Portfolio crasht."))
    corr = np.asarray(market["correlations"], dtype=float)
    out.chart(C.heatmap("Korrelationen der Assets", names, names, corr, z=C.axis("Korrelation", digits=2),
                        scale="sequential", subtitle="Hoch korrelierte Assets crashen gemeinsam und teilen sich den Rand.",
                        height=320 + 12 * k))

    rows = []
    for i in range(k):
        r_i = R[:, i]
        tm_i = TailRiskMetrics(r_i)
        rows.append([names[i], vols[i] * np.sqrt(252), float(tm_i.expected_shortfall(conf)),
                     ew["w"][i], portfolios["iv"][1][i], tp["w"][i], ew["shares"][i], tp["shares"][i]])
    out.table("Assets im Detail", ["Asset", "Vola p. a.", f"ES {lvl} einzeln", "Gewicht gleich", "Gewicht Vola-Parität",
                                   "Gewicht TRP", "ES-Anteil gleich", "ES-Anteil TRP"], rows,
              formats=["text", "pct", "pct", "pct", "pct", "pct", "pct", "pct"])
    out.table("Portfolios im Vergleich", ["Portfolio", "Vola p. a.", f"VaR {lvl}", f"ES {lvl}",
                                          "max. Abweichung vom Zielanteil", "max. Drawdown"],
              [[lbl, stats_p[key]["vol"], stats_p[key]["var"], stats_p[key]["es_hist"], stats_p[key]["dev"],
                stats_p[key]["mdd"]] for key, (lbl, _, _) in portfolios.items()],
              formats=["text", "pct", "pct", "pct", "pct", "pct"],
              note=f"{n_days} simulierte Handelstage; ES historisch. Risikobeiträge aus den "
                   f"{int(round(n_days * (1 - conf)))} schlechtesten Portfolio-Tagen.")
    if tp["dev"] > 0.05:
        hedge = [names[i] for i in range(k) if tp["mes"][i] <= 0]
        out.note("Keine exakte Gleichverteilung gefunden"
                 + (f": {', '.join(hedge)} verliert an den Stresstagen im Mittel nichts und kann daher keinen positiven "
                    "Anteil tragen." if hedge else ": Der ES aus wenigen Stresstagen ist eine Stufenfunktion der Gewichte. "
                    "Mehr Handelstage oder ein niedrigeres Konfidenzniveau helfen."), "warning")
