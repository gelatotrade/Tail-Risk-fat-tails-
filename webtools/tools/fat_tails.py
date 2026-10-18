"""Warum die Normalverteilung versagt - the README's opening argument, on live data."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import TailRiskMetrics
from src.utils.statistics import compute_moments, jarque_bera_test, kolmogorov_smirnov_test

from .. import charts as C
from ..data import index_axis

SPEC = {
    "id": "fat-tails",
    "title": "Warum die Normalverteilung versagt",
    "group": "data",
    "order": 10,
    "summary": "Wie oft 3-, 4- und 5-Sigma-Tage wirklich vorkommen – im Vergleich zur Gauß-Annahme.",
    "description": (
        "Viele Risikomodelle unterstellen normalverteilte Renditen. Dieses Werkzeug misst an deinem "
        "Datensatz, wie viel häufiger extreme Tage tatsächlich auftreten, und zeigt die Verteilung auf "
        "logarithmischer Skala – dort werden die fetten Ränder sichtbar.\n\n"
        "Jarque-Bera- und Kolmogorov-Smirnov-Test prüfen formal, ob die Normalverteilung haltbar ist."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "bins", "type": "int", "default": 90, "min": 30, "max": 200, "step": 10,
         "label": "Klassen im Histogramm"},
        {"name": "max_sigma", "type": "int", "default": 6, "min": 3, "max": 10, "step": 1,
         "label": "Sigma-Schwellen bis", "help": "Tabelle zählt Tage jenseits von 2σ bis zu diesem Wert."},
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

    out.metric("Beobachtungen", data.n, "int")
    out.metric("Volatilität p. a.", sd * np.sqrt(252), "pct", digits=1)
    out.metric("Schiefe", moments["skewness"], "num", hint="< 0: Crashs häufiger als Rallyes")
    out.metric("Exzess-Kurtosis", moments["kurtosis"], "num", digits=1,
               hint="Normalverteilung = 0; > 3 deutet auf fette Ränder",
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
        rows.append([f"{k} σ", days, emp, gauss, ratio, years_gauss])
    out.table(
        "Extreme Tage: beobachtet vs. Normalverteilung",
        ["Schwelle", "Tage beobachtet", "Häufigkeit beobachtet", "Häufigkeit Gauß", "Faktor", "Gauß: einmal alle … Jahre"],
        rows, formats=["text", "int", "pct", "pct", "x", "num"],
        note="Faktor = beobachtete ÷ von der Normalverteilung vorhergesagte Häufigkeit (beide Ränder).",
    )

    # Distribution on linear and log scale
    centers, heights = C.histogram(r, bins=int(params["bins"]), density=True)
    grid = np.linspace(centers[0], centers[-1], 400)
    gauss_pdf = stats.norm.pdf(grid, mu, sd)
    t_params = stats.t.fit(r)
    t_pdf = stats.t.pdf(grid, *t_params)
    hist_trace = C.trace("Beobachtet", centers, heights, type="bar", color=1)
    floor = float(np.min(heights[heights > 0])) / 4  # one bin with one observation, with headroom
    log_range = (floor, float(np.max(heights)) * 3)
    for log in (False, True):
        g = np.where(gauss_pdf >= floor, gauss_pdf, np.nan) if log else gauss_pdf
        tp = np.where(t_pdf >= floor, t_pdf, np.nan) if log else t_pdf
        out.chart(C.xy(
            "Verteilung der Tagesrenditen" + (" – logarithmisch" if log else ""),
            [hist_trace,
             C.trace("Normalverteilung", grid, g, color="muted", dash=True),
             C.trace(f"Student-t (ν = {t_params[0]:.1f})", grid, tp, color=2)],
            x=C.axis("Tagesrendite", fmt="pct", digits=1),
            y=C.axis("Dichte", type="log", range=log_range) if log else C.axis("Dichte"),
            subtitle=("Auf der Log-Skala liegen die beobachteten Ränder weit über der Gauß-Kurve."
                      if log else "In der Mitte sehen beide Kurven ähnlich aus …"),
        ))

    # QQ plot against the normal
    osm, osr = stats.probplot(r, dist="norm", fit=False)
    q = np.asarray(osm)
    out.chart(C.xy(
        "Q-Q-Plot gegen die Normalverteilung",
        [C.trace("Perfekte Normalverteilung", [q.min(), q.max()], [q.min(), q.max()], color="muted", dash=True),
         C.trace("Renditen (standardisiert)", q, (np.asarray(osr) - mu) / sd, type="scatter", color=1, size=5,
                 decimate_to=None)],
        x=C.axis("Theoretische Quantile (σ)"), y=C.axis("Beobachtete Quantile (σ)"),
        subtitle="Punkte, die an den Enden von der Diagonale abknicken, sind die fetten Ränder.",
    ))

    # Time series with extreme days flagged
    x, xa = index_axis(data)
    extreme = np.abs(r - mu) > 3 * sd
    out.chart(C.xy(
        "Tagesrenditen mit 3-Sigma-Ausreißern",
        [C.trace("Tagesrendite", x, r, color=1, width=1),
         C.trace("Jenseits von 3 σ", [xi for xi, e in zip(x, extreme) if e], r[extreme], type="scatter",
                 color="critical", size=7, decimate_to=None)],
        x=xa, y=C.axis("Rendite", fmt="pct", digits=1),
        hlines=[C.hline(mu + 3 * sd, "+3 σ"), C.hline(mu - 3 * sd, "−3 σ")],
    ))

    out.table("Formale Normalitätstests", ["Test", "Statistik", "p-Wert", "Ergebnis"], [
        ["Jarque-Bera", jb["statistic"], jb["p_value"],
         "Normalverteilung abgelehnt" if not jb["is_normal"] else "nicht abgelehnt"],
        ["Kolmogorov-Smirnov", ks["statistic"], ks["p_value"],
         "Normalverteilung abgelehnt" if not ks["matches_distribution"] else "nicht abgelehnt"],
    ], formats=["text", "num", "num", "text"])

    if not jb["is_normal"]:
        out.set_verdict(
            "serious" if moments["kurtosis"] > 3 else "warning",
            "Die Normalverteilung unterschätzt hier das Crash-Risiko",
            f"Tage jenseits von 4 σ treten {rows[2][4]:.0f}-mal häufiger auf, als die Gauß-Annahme erlaubt."
            if len(rows) > 2 and rows[2][4] else "Die Ränder sind deutlich fetter als normal.",
        )
    else:
        out.set_verdict("good", "Keine signifikante Abweichung von der Normalverteilung",
                        "Bei kurzen oder ruhigen Reihen ist das möglich – die Ränder können trotzdem fett sein.")
