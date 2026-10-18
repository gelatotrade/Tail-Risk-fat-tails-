"""Extreme value theory on the data: GPD peaks over threshold, GEV block maxima, return levels."""

import numpy as np
from scipy import stats

from src.models.extreme_value import EVTTailRiskAnalyzer, hill_plot, mean_excess_plot

from .. import charts as C
from ..core import ToolError

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"), (0.995, "99,5 %"), (0.999, "99,9 %"))]
DAYS = 252

SPEC = {
    "id": "evt",
    "title": "Extremwerttheorie (EVT)",
    "group": "risk",
    "order": 30,
    "summary": "Modelliert nur den Rand: GPD oberhalb einer Schwelle, GEV für Blockmaxima – mit VaR, Expected "
               "Shortfall und Verlusten, die einmal in 10 oder 100 Jahren auftreten.",
    "description": (
        "Die Extremwerttheorie passt kein Modell an alle Tage an, sondern nur an die extremen. **Peaks over "
        "Threshold:** Alle Verluste oberhalb einer Schwelle u folgen näherungsweise einer **Generalisierten "
        "Pareto-Verteilung (GPD)**. Ihr Formparameter ξ ist der Tail-Index: ξ > 0 heißt fetter Rand (Fréchet-Bereich, "
        "nur Momente unterhalb der Ordnung 1/ξ existieren), ξ = 0 exponentieller Rand wie bei Gauß. **Blockmaxima:** "
        "Der größte Verlust je Block (z. B. 20 Handelstage) folgt einer **GEV-Verteilung**.\n\n"
        "So wählst du die Schwelle: Im **Mean-Excess-Plot** sollte die mittlere Überschreitung oberhalb von u etwa "
        "linear verlaufen; im **Hill-Plot** sollte die Schätzung stabil sein. Zu niedrige Schwellen verzerren, zu hohe "
        "lassen zu wenige Punkte übrig. Das **Wiederkehr-Diagramm** zeigt, welcher Tagesverlust im Mittel einmal in "
        "T Jahren (252 Handelstage pro Jahr) überschritten wird – GPD, GEV, Gauß und die Daten im Vergleich.\n\n"
        "Alle Verluste sind tägliche Log-Renditen als positive Dezimalzahl. Die Tabelle aus `comprehensive_report` "
        "zeigt zusätzlich, um welchen Faktor die Normalverteilung 3-, 4- und 5-σ-Tage unterschätzt."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold_quantile", "type": "float", "default": 0.95, "min": 0.85, "max": 0.99, "step": 0.005,
         "label": "Schwelle u (Quantil)", "help": "0,95 = die 5 % größten Verluste fließen in den GPD-Fit."},
        {"name": "block_size", "type": "int", "default": 20, "min": 5, "max": 63, "step": 1,
         "label": "Blockgröße GEV (Handelstage)", "help": "20 ≈ ein Monat, 63 ≈ ein Quartal."},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Konfidenzniveau", "options": LEVELS},
    ],
    "covers": [
        "src.models.extreme_value.EVTTailRiskAnalyzer",
        "src.models.extreme_value.EVTTailRiskAnalyzer.tail_index",
        "src.models.extreme_value.EVTTailRiskAnalyzer.var",
        "src.models.extreme_value.EVTTailRiskAnalyzer.expected_shortfall",
        "src.models.extreme_value.EVTTailRiskAnalyzer.return_level",
        "src.models.extreme_value.EVTTailRiskAnalyzer.tail_comparison_to_gaussian",
        "src.models.extreme_value.EVTTailRiskAnalyzer.comprehensive_report",
        "src.models.extreme_value.GeneralizedParetoDistribution.fit_pot",
        "src.models.extreme_value.GeneralizedParetoDistribution.var",
        "src.models.extreme_value.GeneralizedParetoDistribution.expected_shortfall",
        "src.models.extreme_value.GeneralizedParetoDistribution.tail_probability",
        "src.models.extreme_value.GeneralizedExtremeValue.fit_block_maxima",
        "src.models.extreme_value.mean_excess_plot",
        "src.models.extreme_value.hill_plot",
    ],
    "cost": "fast",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _years(x):
    if not np.isfinite(x):
        return "nie"
    if x >= 1e6:
        mant, exp = f"{x:.1e}".split("e")
        return f"{mant.replace('.', ',')} · 10{str(int(exp)).translate(SUP)} Jahren"
    if x >= 100:
        return f"{x:,.0f} Jahren".replace(",", ".")
    return f"{_de(x, 1)} Jahren"


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    q, m, conf = float(params["threshold_quantile"]), int(params["block_size"]), float(params["confidence"])
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])
    if n * (1 - q) < 10:
        raise ToolError(f"Bei {n} Tagen bleiben über dem {_de(100 * q, 1)}-%-Quantil weniger als 10 Verluste. "
                        "Senke die Schwelle oder nimm eine längere Reihe.")
    if n // m < 10:
        raise ToolError(f"Mit Blöcken à {m} Tagen gibt es nur {n // m} Blockmaxima (mindestens 10 nötig). "
                        "Verkleinere die Blockgröße.")

    out.progress("Passe GPD und GEV an beide Ränder an …", 0.2)
    try:
        evt = EVTTailRiskAnalyzer(r, block_size=m, threshold_quantile=q)
    except ValueError as exc:
        raise ToolError(f"Der EVT-Fit ist nicht möglich: {exc}")
    report = evt.comprehensive_report()

    xi_l, xi_r = float(evt.tail_index("left")), float(evt.tail_index("right"))
    st = evt.stats_left
    u, p_u = float(st["threshold"]), float(st["exceedance_probability"])
    var_c, es_c = float(evt.var(conf, "gpd")), float(evt.expected_shortfall(conf))
    var_gev = float(evt.var(conf, "gev"))
    rl10, rl100 = -float(report["return_levels"]["10_year"]), -float(report["return_levels"]["100_year"])
    mu, sd = float(np.mean(r)), float(np.std(r))

    def gauss_var(c):
        return -(mu + sd * stats.norm.ppf(1 - c))

    def gauss_years(loss):
        p = stats.norm.cdf((-loss - mu) / sd)
        return 1 / p / DAYS if p > 0 else np.inf

    out.metric("Tail-Index ξ · Verluste", xi_l, "num", hint="GPD-Form: > 0 fetter Rand, 0 wie Gauß",
               status="critical" if xi_l > 0.4 else "serious" if xi_l > 0.25 else "warning" if xi_l > 0.1 else "good")
    out.metric("Tail-Index ξ · Gewinne", xi_r, "num")
    out.metric("Schwelle u", u, "pct", hint=f"{st['n_exceedances']} Verluste darüber ({_de(100 * p_u, 1)} %)")
    out.metric(f"VaR {lvl} · GPD", var_c, "pct", hint=f"Gauß {_de(100 * gauss_var(conf))} %")
    out.metric(f"Expected Shortfall {lvl} · GPD", es_c, "pct")
    out.metric(f"VaR {lvl} · GEV", var_gev, "pct", hint=f"aus {m}-Tage-Blockmaxima")
    out.metric("10-Jahres-Tagesverlust", rl10, "pct", hint=f"GEV · Gauß: einmal in {_years(gauss_years(rl10))}")
    out.metric("100-Jahres-Tagesverlust", rl100, "pct", hint=f"GEV · Gauß: einmal in {_years(gauss_years(rl100))}")

    # --- chart 1: tail fit, P(loss > x)
    losses = np.sort(-r)[::-1]
    xs = np.linspace(u, max(losses[0] * 1.4, u * 2), 120)
    gpd_tail = np.array([evt.gpd_left.tail_probability(x, p_u) for x in xs], dtype=float)
    emp = np.array([np.mean(-r > x) for x in xs])
    gauss_tail = stats.norm.cdf((-xs - mu) / sd)
    floor = min(0.2 / n, 1e-5)
    keep = lambda y: np.where(y > floor * 0.5, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Verlust-Rand: P(Tagesverlust > x) oberhalb der Schwelle",
        [C.trace("Beobachtet", xs, np.where(emp > 0, emp, np.nan), type="step", color=4, width=2),
         C.trace(f"GPD (ξ = {_de(xi_l)})", xs, keep(gpd_tail), color=1, width=2.5),
         C.trace("Normalverteilung", xs, keep(gauss_tail), color="muted", dash=True)],
        x=C.axis("Tagesverlust x", fmt="pct", digits=0), y=C.axis("Wahrscheinlichkeit pro Tag", type="log",
                                                                   range=(floor, p_u * 1.5)),
        subtitle="Die GPD folgt den beobachteten Extremen; die Gauß-Kurve stürzt ab.",
        hlines=[C.hline(1 / DAYS, "einmal pro Jahr"), C.hline(1 / (10 * DAYS), "einmal in 10 Jahren")],
    ))

    # --- chart 2: return levels in years
    years = np.geomspace(0.1, 200, 90)
    t_days = years * DAYS
    gev_rl = -np.asarray(evt.return_level(t_days, "left"), dtype=float)
    gpd_rl = np.array([evt.gpd_left.var(1 - 1 / T, p_u) if 1 / T < p_u else np.nan for T in t_days], dtype=float)
    g_rl = -(mu + sd * stats.norm.ppf(1 / t_days))
    k = min(200, n)
    emp_years = (n + 1) / np.arange(1, k + 1) / DAYS
    out.chart(C.xy(
        "Wiederkehrniveaus: Welcher Tagesverlust kommt einmal in T Jahren?",
        [C.trace("Beobachtet", emp_years, losses[:k], type="scatter", color=4, size=6, decimate_to=None),
         C.trace(f"GEV ({m}-Tage-Blockmaxima)", years, gev_rl, color=2, width=2.5),
         C.trace("GPD (Peaks over Threshold)", years, gpd_rl, color=1, width=2.5),
         C.trace("Normalverteilung", years, g_rl, color="muted", dash=True)],
        x=C.axis("Wiederkehrperiode (Jahre)", type="log"), y=C.axis("Tagesverlust", fmt="pct", digits=0),
        subtitle="Punkte: tatsächliche Verluste, sortiert nach ihrer Seltenheit in den Daten.",
        vlines=[C.vline(10, "10 Jahre"), C.vline(100, "100 Jahre")],
    ))

    # --- chart 3: mean excess plot for threshold choice
    me = mean_excess_plot(-r)
    ok = np.isfinite(me["mean_excess"])
    out.chart(C.xy(
        "Mean-Excess-Plot: mittlere Überschreitung e(u)",
        [C.trace("e(u) = E[Verlust − u | Verlust > u]", np.asarray(me["thresholds"])[ok], me["mean_excess"][ok],
                 type="scatter", color=1, size=6, decimate_to=None)],
        x=C.axis("Schwelle u (Tagesverlust)", fmt="pct", digits=1), y=C.axis("Mittlere Überschreitung", fmt="pct", digits=2),
        subtitle="Steigt e(u) oberhalb der Schwelle linear, passt die GPD; Steigung > 0 heißt ξ > 0.",
        vlines=[C.vline(u, "gewählte Schwelle", color="critical")],
    ))

    # --- chart 4: Hill plot on the loss tail
    loss_only = -r[r < 0]
    hp = hill_plot(loss_only)
    kv, hv = np.asarray(hp["k_values"]), np.asarray(hp["hill_estimates"], dtype=float)
    hlines = [C.hline(1 / xi_l, f"1/ξ (GPD) = {_de(1 / xi_l, 1)}", color="critical")] if xi_l > 0.02 else []
    if len(kv):
        out.chart(C.xy(
            "Hill-Plot: Tail-Index α der Verluste je Zahl k der Extremwerte",
            [C.trace("Hill-Schätzer α(k)", kv, hv, color=1)],
            x=C.axis("k (größte Verluste)", fmt="int"), y=C.axis("Tail-Index α"),
            subtitle="Ein stabiles Plateau ist der glaubwürdige Tail-Index. α = 1/ξ; α < 4 heißt unendliche Kurtosis.",
            hlines=hlines, vlines=[C.vline(st["n_exceedances"], "k = Überschreitungen")],
        ))

    # --- tables
    t10, t100 = 10 * DAYS, 100 * DAYS
    hist_level = lambda T: float(np.quantile(-r, 1 - 1 / T)) if T <= n else None  # noqa: E731
    rows = [
        ["VaR 99 %", report["var_99"], float(evt.var(0.99, "gev")), float(np.quantile(-r, 0.99)), gauss_var(0.99)],
        ["VaR 99,5 %", report["var_995"], float(evt.var(0.995, "gev")), float(np.quantile(-r, 0.995)), gauss_var(0.995)],
        ["VaR 99,9 %", report["var_999"], float(evt.var(0.999, "gev")), float(np.quantile(-r, 0.999)), gauss_var(0.999)],
        ["Expected Shortfall 99 %", report["es_99"], None, float(np.mean(-r[-r >= np.quantile(-r, 0.99)])),
         -mu + sd * stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01],
        ["10-Jahres-Tagesverlust", float(evt.gpd_left.var(1 - 1 / t10, p_u)), rl10, hist_level(t10), gauss_var(1 - 1 / t10)],
        ["100-Jahres-Tagesverlust", float(evt.gpd_left.var(1 - 1 / t100, p_u)), rl100, hist_level(t100),
         gauss_var(1 - 1 / t100)],
    ]
    out.table("Risikokennzahlen: EVT gegen historisch und Gauß",
              ["Kennzahl", "GPD (POT)", f"GEV ({m} Tage)", "Historisch", "Gauß"], rows,
              formats=["text", "pct", "pct", "pct", "pct"],
              note="Wiederkehrniveaus in Handelstagen: 10 Jahre = 2.520 Tage. Historische Werte gibt es nur, "
                   "wenn die Reihe mindestens so lang ist.")

    gc = report["gaussian_comparison"]
    out.table("Wie stark die Normalverteilung extreme Tage unterschätzt",
              ["Schwelle", "Gauß", "Beobachtet", "GPD", "Faktor beobachtet", "Faktor GPD"],
              [[f"{k} σ", v["gaussian_probability"], v["empirical_probability"], v["gpd_probability"],
                v["underestimation_factor_empirical"], v["underestimation_factor_gpd"]]
               for k, v in zip((3, 4, 5), (gc["3sigma"], gc["4sigma"], gc["5sigma"]))],
              formats=["text", "pct", "pct", "pct", "x", "x"],
              note="Wahrscheinlichkeit pro Tag für |Rendite| > k σ (beide Ränder). GPD: Fits beider Ränder.")

    sr = evt.stats_right
    out.table("Angepasste Modelle", ["Rand", "Schwelle u", "Überschreitungen", "ξ (GPD)", "σ (GPD)", "ξ (GEV)"],
              [["Verluste", u, st["n_exceedances"], xi_l, st["sigma"], evt.gev_minima.xi],
               ["Gewinne", sr["threshold"], sr["n_exceedances"], xi_r, sr["sigma"], evt.gev_maxima.xi]],
              formats=["text", "pct", "int", "num", "pct", "num"],
              note=f"GEV an {n // m} Blockmaxima à {m} Handelstage angepasst.")

    if var_gev < 0.9 * var_c:
        out.note("Der GEV-VaR liegt unter dem GPD-VaR: Die Umrechnung vom Blockmaximum auf den einzelnen Tag "
                 "unterstellt unabhängige Tage. Verluste kommen aber in Clustern, ein Block enthält dann mehrere "
                 "extreme Tage – der GPD-Wert ist hier der verlässlichere.", "info")

    yrs = gauss_years(rl10)
    if xi_l > 0.1:
        out.set_verdict(
            "serious" if xi_l > 0.25 else "warning",
            f"Fetter Verlust-Rand (Fréchet-Bereich, ξ = {_de(xi_l)})",
            f"Den 10-Jahres-Tagesverlust von {_de(100 * rl10, 1)} % hält die Normalverteilung für ein Ereignis "
            f"einmal in {_years(yrs)}. Gauß-basierte Stresstests unterschätzen hier die Extreme.",
        )
    elif xi_l > -0.05:
        out.set_verdict("good", f"Der Verlust-Rand ist etwa exponentiell (ξ = {_de(xi_l)})",
                        "Oberhalb der Schwelle verhalten sich die Verluste nahezu wie bei einer Normal- oder "
                        "Exponentialverteilung; prüfe mit dem Hill-Plot, ob die Schwelle passt.")
    else:
        out.set_verdict("good", f"Beschränkter Verlust-Rand (ξ = {_de(xi_l)})",
                        "Die größten Verluste wachsen langsamer als exponentiell (Weibull-Bereich).")
