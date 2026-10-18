"""TailRiskMetrics: VaR four ways, Expected Shortfall, drawdown, Omega/Sortino/Calmar, tail dependence."""

import numpy as np

from src.models.extreme_value import GeneralizedParetoDistribution
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"), (0.995, "99,5 %"), (0.999, "99,9 %"))]

SPEC = {
    "id": "risk-metrics",
    "title": "VaR & Expected Shortfall",
    "group": "risk",
    "order": 20,
    "summary": "Value at Risk auf vier Arten, Expected Shortfall, maximaler Drawdown und Omega-, Sortino- und "
               "Calmar-Ratio – und wie stark die Gauß-Annahme das Risiko unterschätzt.",
    "description": (
        "Der **Value at Risk** (VaR) ist der Tagesverlust, der mit der gewählten Konfidenz nicht überschritten wird. "
        "Der **Expected Shortfall** (ES) ist der mittlere Verlust an den Tagen, an denen der VaR doch gerissen wird – "
        "bei fetten Rändern das ehrlichere Maß. Vier VaR-Methoden: **historisch** (Quantil der Daten), **Gauß** "
        "(nur Mittelwert und Volatilität), **Cornish-Fisher** (Gauß-Quantil korrigiert um Schiefe und Kurtosis) "
        "und **EVT** (Generalisierte Pareto-Verteilung oberhalb einer Schwelle).\n\n"
        "So liest du die Diagramme: Die **VaR-Kurve** zeigt, ab welchem Konfidenzniveau sich die Methoden trennen – "
        "je weiter rechts, desto seltener das Ereignis. Das **Unterwasser-Diagramm** zeigt den Abstand zum letzten "
        "Höchststand des Vermögens (Log-Renditen werden mit exp(Σr) verzinst). Die **Tail-Abhängigkeit** misst, wie "
        "oft zwei Extreme am selben Tag zusammenfallen – mit VIX-Daten: Crash-Tag und VIX-Sprung; ohne VIX: "
        "Verlusttag heute und gestern (Volatilitäts-Cluster).\n\n"
        "Alle Verluste sind Tagesverluste als positive Dezimalzahl. Omega und Sortino beziehen sich auf die "
        "Zielrendite; Omega > 1 heißt: Die Gewinne darüber überwiegen die Verluste darunter."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Konfidenzniveau", "options": LEVELS},
        {"name": "evt_threshold", "type": "float", "default": 0.95, "min": 0.85, "max": 0.99, "step": 0.005,
         "label": "EVT-Schwelle (Quantil)", "help": "Der GPD-Fit nutzt die Verluste oberhalb dieses Quantils."},
        {"name": "target_annual", "type": "float", "default": 0.0, "min": -0.2, "max": 0.3, "step": 0.01,
         "label": "Zielrendite p. a.", "help": "Für Omega und Sortino; 0,05 = 5 % pro Jahr."},
        {"name": "dependence_quantile", "type": "float", "default": 0.95, "min": 0.8, "max": 0.99, "step": 0.01,
         "label": "Quantil für Tail-Abhängigkeit"},
    ],
    "covers": [
        "src.models.risk_metrics.TailRiskMetrics",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.var_parametric_gaussian",
        "src.models.risk_metrics.TailRiskMetrics.var_cornish_fisher",
        "src.models.risk_metrics.TailRiskMetrics.var_evt",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
        "src.models.risk_metrics.TailRiskMetrics.tail_ratio",
        "src.models.risk_metrics.TailRiskMetrics.maximum_drawdown",
        "src.models.risk_metrics.TailRiskMetrics.omega_ratio",
        "src.models.risk_metrics.TailRiskMetrics.sortino_ratio",
        "src.models.risk_metrics.TailRiskMetrics.calmar_ratio",
        "src.models.risk_metrics.TailRiskMetrics.tail_dependence",
    ],
    "cost": "fast",
}

TABLE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _label(c):
    return next(o["label"] for o in LEVELS if o["value"] == c)


def _date(data, i):
    return data.dates[i] if data.dates else f"Tag {i}"


def _fmt_date(s):
    if isinstance(s, str) and len(s) == 10 and s[4] == "-":
        return f"{s[8:10]}.{s[5:7]}.{s[:4]}"
    return str(s)


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    conf, q_evt = float(params["confidence"]), float(params["evt_threshold"])
    target = float(params["target_annual"]) / 252.0
    if n * (1 - q_evt) < 10:
        raise ToolError("Über der EVT-Schwelle liegen weniger als 10 Verluste. Senke die Schwelle.")
    lvl = _label(params["confidence"])

    trm = TailRiskMetrics(r)
    table = {}
    for c in TABLE_LEVELS:
        table[c] = {
            "hist": float(trm.var_historical(c)), "gauss": float(trm.var_parametric_gaussian(c)),
            "cf": float(trm.var_cornish_fisher(c)), "evt": float(trm.var_evt(c, threshold_quantile=q_evt)),
            "es_hist": float(trm.expected_shortfall(c, "historical")),
            "es_gauss": float(trm.expected_shortfall(c, "gaussian")),
            "es_cf": float(trm.expected_shortfall(c, "cornish_fisher")),
        }
    v = table[conf]
    dd = trm.maximum_drawdown()
    calmar, sortino, omega = float(trm.calmar_ratio()), float(trm.sortino_ratio(target)), float(trm.omega_ratio(target))

    i0, i1 = int(dd["max_dd_start"]), int(dd["max_dd_end"])
    rec = dd["recovery_time"]
    rec_txt = "noch nicht erholt" if not np.isfinite(rec) else f"erholt nach {int(rec)} Tagen"
    out.metric(f"VaR {lvl} · historisch", v["hist"], "pct", hint=f"ES {_de(100 * v['es_hist'])} %")
    out.metric(f"VaR {lvl} · Gauß", v["gauss"], "pct", hint=f"ES {_de(100 * v['es_gauss'])} %")
    out.metric(f"VaR {lvl} · Cornish-Fisher", v["cf"], "pct", hint=f"ES {_de(100 * v['es_cf'])} %")
    out.metric(f"VaR {lvl} · EVT (GPD)", v["evt"], "pct", hint=f"Schwelle: {_de(100 * q_evt, 1)}-%-Quantil")
    out.metric(f"Expected Shortfall {lvl}", v["es_hist"], "pct", hint="historisch: Ø Verlust jenseits des VaR",
               status="serious" if v["es_hist"] > 1.5 * v["es_gauss"] else "warning" if v["es_hist"] > 1.15 * v["es_gauss"] else None)
    out.metric("Maximaler Drawdown", dd["max_drawdown"], "pct", digits=1,
               hint=f"{_fmt_date(_date(data, i0))} → {_fmt_date(_date(data, i1))} · {rec_txt}",
               status="critical" if dd["max_drawdown"] > 0.4 else "serious" if dd["max_drawdown"] > 0.2 else None)
    out.metric("Calmar-Ratio", calmar, "num", hint="Rendite p. a. ÷ max. Drawdown")
    out.metric("Sortino-Ratio", sortino, "num", hint="annualisiert, nur Abwärts-Volatilität")
    out.metric("Omega-Ratio", omega, "num", hint="Gewinne ÷ Verluste relativ zur Zielrendite")

    # --- chart 1: VaR across confidence levels (rarer events to the right)
    tails = np.geomspace(0.1, 0.001, 60)
    gpd, st = GeneralizedParetoDistribution.fit_pot(-r, quantile=q_evt)  # the fit var_evt uses
    p_u = st["exceedance_probability"]
    curves = {"hist": [], "gauss": [], "cf": [], "evt": []}
    for a in tails:
        c = 1 - a
        curves["hist"].append(trm.var_historical(c))
        curves["gauss"].append(trm.var_parametric_gaussian(c))
        curves["cf"].append(trm.var_cornish_fisher(c))
        curves["evt"].append(gpd.var(c, p_u) if a < p_u else np.nan)
    out.chart(C.xy(
        "VaR je Konfidenzniveau: Wo sich die Methoden trennen",
        [C.trace("Historisch", tails, curves["hist"], color=4, type="step"),
         C.trace("Gauß", tails, curves["gauss"], color="muted", dash=True),
         C.trace("Cornish-Fisher", tails, curves["cf"], color=3),
         C.trace("EVT (GPD)", tails, curves["evt"], color=1, width=2.5)],
        x=C.axis("Überschreitungswahrscheinlichkeit 1 − c (pro Tag)", type="log", fmt="pct", digits=1, range=(0.1, 0.001)),
        y=C.axis("VaR (Tagesverlust)", fmt="pct", digits=1),
        subtitle="Links: häufige Verluste (90 %-VaR), rechts: seltene (99,9 %). Gauß bleibt bei seltenen Ereignissen zurück.",
        vlines=[C.vline(1 - conf, f"gewählt: {lvl}")],
    ))

    # --- chart 2: underwater curve
    x, xa = index_axis(data)
    ddser = np.asarray(dd["drawdown_series"], dtype=float)
    end_rec = i1 + int(rec) if np.isfinite(rec) else n - 1
    out.chart(C.xy(
        "Unterwasser-Diagramm: Abstand zum letzten Höchststand",
        [C.trace("Drawdown", x, ddser, type="area", color=1, width=1)],
        x=xa, y=C.axis("Drawdown", fmt="pct", digits=0),
        bands=[C.band(x[i0], x[i1], f"Max. Drawdown −{_de(100 * dd['max_drawdown'], 1)} %", color="critical"),
               C.band(x[i1], x[min(end_rec, n - 1)], "Erholung" if np.isfinite(rec) else "", color="warning")],
    ))

    # --- tail dependence: VIX changes if available, else yesterday's return
    qd = float(params["dependence_quantile"])
    if data.vix is not None and np.all(np.asarray(data.vix) > 0):
        dv = np.diff(np.log(np.asarray(data.vix, dtype=float)))
        a, b = r[1:], -dv  # a VIX spike is the "loss" of the second series
        partner, ylab, yser = "VIX-Veränderung", "VIX-Veränderung (log, täglich)", dv
        explain = "Crash-Tag und VIX-Sprung am selben Tag"
    else:
        a, b = r[1:], r[:-1]
        partner, ylab, yser = "Vortagesrendite", "Rendite am Vortag", r[:-1]
        explain = "Verlusttag heute und gestern (Cluster)"
    tdep = trm.__class__(a).tail_dependence(b, quantile=qd)
    lo_a, lo_b = np.quantile(a, 1 - qd), np.quantile(b, 1 - qd)
    joint = (a < lo_a) & (b < lo_b)
    out.chart(C.xy(
        f"Tail-Abhängigkeit: Tagesrendite gegen {partner}",
        [C.trace("alle Tage", a[~joint], yser[~joint], type="scatter", color=1, size=4, decimate_to=None),
         C.trace(f"beide im {_de(100 * (1 - qd), 0)}-%-Rand", a[joint], yser[joint], type="scatter", color="critical",
                 size=7, decimate_to=None)],
        x=C.axis("Tagesrendite", fmt="pct", digits=0), y=C.axis(ylab, fmt="pct", digits=0),
        subtitle=f"Rot: {explain}. Unter Unabhängigkeit wären es {_de(100 * (1 - qd), 0)} % der Randtage.",
    ))

    # --- tables
    rows = [[_label(c), t["hist"], t["gauss"], t["cf"], t["evt"], t["es_hist"], t["es_gauss"], t["es_cf"]]
            for c, t in table.items()]
    out.table("VaR und Expected Shortfall nach Methode",
              ["Konfidenz", "VaR historisch", "VaR Gauß", "VaR Cornish-Fisher", "VaR EVT", "ES historisch",
               "ES Gauß", "ES Cornish-Fisher"], rows,
              formats=["text"] + ["pct"] * 7,
              note="Tagesverluste. ES Cornish-Fisher = Ø der beobachteten Verluste jenseits des Cornish-Fisher-VaR.")

    tr_rows = []
    for k in (2, 3, 4, 5):
        t = trm.tail_ratio(k)
        tr_rows.append([f"{k} σ", t["left_tail_prob"], t["right_tail_prob"], t["gaussian_tail_prob"],
                        t["left_ratio"], t["right_ratio"], t["asymmetry"] if np.isfinite(t["asymmetry"]) else None])
    out.table("Randhäufigkeiten gegen die Normalverteilung", ["Schwelle", "P(Verlust > kσ)", "P(Gewinn > kσ)",
                                                             "Gauß (je Rand)", "Faktor links", "Faktor rechts",
                                                             "Asymmetrie links/rechts"],
              tr_rows, formats=["text", "pct", "pct", "pct", "x", "x", "num"])

    out.table(f"Tail-Abhängigkeit (Quantil {_de(qd)})", ["Kennzahl", "Wert"], [
        ["Untere Tail-Abhängigkeit (beide extrem schlecht)", tdep["lower_tail_dependence"]],
        ["Obere Tail-Abhängigkeit (beide extrem gut)", tdep["upper_tail_dependence"]],
        ["Korrelation (Pearson)", tdep["standard_correlation"]],
        ["Asymmetrie (unten − oben)", tdep["tail_asymmetry"]],
    ], formats=["text", "num"],
        note=f"Partner-Reihe: {partner}" + (" (mit −1 multipliziert, damit ein VIX-Sprung der „schlechte“ Rand ist)."
                                            if partner == "VIX-Veränderung" else ".")
        + f" Unabhängigkeit ergäbe ≈ {_de(1 - qd, 2)}, vollständige Abhängigkeit 1.")

    cf_vals = [table[c]["cf"] for c in TABLE_LEVELS]
    if any(b2 < a2 for a2, b2 in zip(cf_vals, cf_vals[1:])):
        out.note("Cornish-Fisher ist hier nicht monoton: Bei sehr hoher Kurtosis bricht die Reihenentwicklung zusammen "
                 "und ein höheres Konfidenzniveau liefert einen kleineren VaR. Verlass dich dann auf EVT oder historisch.",
                 "warning")

    best_alt = max(v["hist"], v["evt"])
    if v["cf"] > 1.5 * best_alt:
        out.note(f"Cornish-Fisher übertreibt hier (Exzess-Kurtosis {_de(float(trm.kurtosis), 1)}): Die Korrektur um "
                 "Schiefe und Kurtosis ist nur für mäßig fette Ränder gedacht. Historischer und EVT-VaR sind "
                 "verlässlicher.", "warning")

    fat = best_alt
    r_var = fat / v["gauss"] if v["gauss"] > 0 else np.nan
    r_es = v["es_hist"] / v["es_gauss"] if v["es_gauss"] > 0 else np.nan
    worst = np.nanmax([r_var, r_es])
    if np.isfinite(worst):
        level = "serious" if worst > 1.25 else "warning" if worst > 1.05 else "good"
        pct = _de(100 * (worst - 1), 0)
        if level == "good":
            out.set_verdict("good", "Gauß- und Fat-Tail-Risiko liegen nah beieinander",
                            f"Bei {lvl} ist die Normalverteilung hier eine brauchbare Näherung – prüfe höhere "
                            "Konfidenzniveaus in der VaR-Kurve.")
        elif r_var >= r_es:
            out.set_verdict(level, f"Der Gauß-VaR unterschätzt das Risiko um {pct} %",
                            f"Bei {lvl} liegt der Fat-Tail-VaR bei {_de(100 * fat)} %, die Normalverteilung sagt "
                            f"{_de(100 * v['gauss'])} %. Der Expected Shortfall ({_de(100 * v['es_hist'])} % gegen "
                            f"{_de(100 * v['es_gauss'])} %) zeigt, wie tief es an solchen Tagen im Mittel geht.")
        else:
            out.set_verdict(level, f"Gauß unterschätzt den Expected Shortfall um {pct} %",
                            f"Der VaR {lvl} ({_de(100 * fat)} %) liegt nah am Gauß-Wert ({_de(100 * v['gauss'])} %), "
                            f"doch jenseits davon geht es viel tiefer: im Mittel {_de(100 * v['es_hist'])} % statt "
                            f"{_de(100 * v['es_gauss'])} %. Genau das verbirgt ein reiner VaR.")
