"""Rolling VaR/ES backtest, rolling skewness/kurtosis/tail index and the 3D risk phase space."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import RollingRiskMetrics, compute_risk_3d_coordinates

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

LEVELS = [{"value": v, "label": l} for v, l in ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"))]

SPEC = {
    "id": "rolling-risk",
    "title": "Rollierendes Risiko & VaR-Backtest",
    "group": "risk",
    "order": 40,
    "summary": "VaR, Expected Shortfall, Schiefe, Kurtosis und Tail-Index im gleitenden Fenster – mit ehrlichem "
               "Backtest und dem Weg des Marktes durch den 3D-Risikoraum.",
    "description": (
        "`RollingRiskMetrics` schätzt jede Kennzahl nur aus den letzten N Handelstagen **vor** dem jeweiligen Tag. "
        "Der rollierende VaR ist damit eine echte Prognose: Fällt die Rendite eines Tages unter −VaR, ist das eine "
        "**VaR-Verletzung**. Bei 99 % sollten das etwa 1 % der Tage sein; der **Kupiec-Test** prüft, ob die Quote "
        "dazu passt. Häufen sich Verletzungen nach ruhigen Phasen, reagiert das Fenster zu langsam auf "
        "Volatilitäts-Cluster.\n\n"
        "Die **Exzess-Kurtosis** (0 = Gauß) und die **Schiefe** zeigen, wie sich die Form der Verteilung ändert; der "
        "**Tail-Index α** (Hill oder Pickands) misst den Rand direkt: α < 2 hieße unendliche Varianz, α < 4 "
        "unendliche Kurtosis. In kurzen Fenstern sind alle drei verrauscht – längere Fenster glätten, reagieren aber "
        "später.\n\n"
        "Der **3D-Risikoraum** (`compute_risk_3d_coordinates`) normiert drei Achsen auf 0–1: VaR-Niveau, ES/VaR-"
        "Verhältnis (wie tief es jenseits des VaR geht) und Kurtosis. Oben rechts hinten liegen die gefährlichsten "
        "Phasen. Verluste sind Tagesverluste als positive Dezimalzahl."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 60, "min": 20, "max": 500, "step": 5,
         "label": "Fenster (Handelstage)", "help": "60 ≈ drei Monate, 250 ≈ ein Jahr."},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "VaR-Konfidenz", "options": LEVELS},
        {"name": "tail_method", "type": "select", "default": "hill", "label": "Tail-Index-Schätzer",
         "options": [{"value": "hill", "label": "Hill"}, {"value": "pickands", "label": "Pickands"}]},
    ],
    "covers": [
        "src.models.risk_metrics.RollingRiskMetrics",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_var",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_expected_shortfall",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_kurtosis",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_skewness",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_tail_index",
        "src.models.risk_metrics.RollingRiskMetrics.compute_all",
        "src.models.risk_metrics.compute_risk_3d_coordinates",
    ],
    "cost": "fast",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _fmt_date(s):
    if isinstance(s, str) and len(s) == 10 and s[4] == "-":
        return f"{s[8:10]}.{s[5:7]}.{s[:4]}"
    return f"Tag {s}"


def _kupiec(breaches, n, p):
    """Kupiec proportion-of-failures likelihood-ratio test."""
    if n == 0:
        return np.nan
    x = int(breaches)
    phat = x / n

    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)

    lr = -2 * (ll(p) - ll(phat))
    return float(stats.chi2.sf(max(lr, 0.0), 1))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    w, conf, method = int(params["window"]), float(params["confidence"]), params["tail_method"]
    if n < w + 60:
        raise ToolError(f"Die Reihe ({n} Tage) ist für ein Fenster von {w} Tagen zu kurz. Verkleinere das Fenster.")
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    rr = RollingRiskMetrics(r, window=w)
    allm = rr.compute_all()
    var = allm["var_99"] if conf == 0.99 else rr.rolling_var(conf)
    es = allm["es_99"] if conf == 0.99 else rr.rolling_expected_shortfall(conf)
    kurt, skew = allm["kurtosis"], allm["skewness"]
    alpha = allm["tail_index"] if method == "hill" else rr.rolling_tail_index(method)
    X, Y, Z = compute_risk_3d_coordinates(r, window=w)

    valid = np.isfinite(var)
    breach = valid & (r < -np.where(valid, var, 0.0))
    n_valid, n_breach = int(valid.sum()), int(breach.sum())
    rate = n_breach / n_valid if n_valid else np.nan
    p_kup = _kupiec(n_breach, n_valid, 1 - conf)
    x, xa = index_axis(data)
    i_max = int(np.nanargmax(var))
    last_alpha = alpha[np.isfinite(alpha)][-1] if np.any(np.isfinite(alpha)) else np.nan

    out.metric(f"VaR {lvl} heute", var[-1], "pct", hint=f"aus den letzten {w} Tagen",
               status="serious" if var[-1] > 1.5 * np.nanmedian(var) else None)
    out.metric(f"Expected Shortfall {lvl} heute", es[-1], "pct")
    out.metric(f"VaR {lvl} · Median", np.nanmedian(var), "pct", hint=f"Höchststand {_de(100 * var[i_max])} % am "
               f"{_fmt_date(x[i_max]) if data.dates else _fmt_date(i_max)}")
    out.metric("VaR-Verletzungen", rate, "pct", hint=f"{n_breach} von {n_valid} Tagen · erwartet {_de(100 * (1 - conf), 1)} %",
               status="critical" if rate > 2.5 * (1 - conf) else "serious" if rate > 1.5 * (1 - conf) else "good")
    out.metric("Kupiec-Test (p-Wert)", p_kup, "num", digits=3, hint="< 0,05: Verletzungsquote passt nicht zum Niveau",
               status="warning" if p_kup < 0.05 else None)
    out.metric("Exzess-Kurtosis heute", kurt[-1], "num", digits=1)
    out.metric("Schiefe heute", skew[-1], "num")
    out.metric(f"Tail-Index α heute ({'Hill' if method == 'hill' else 'Pickands'})", last_alpha, "num",
               hint="< 2: unendliche Varianz · < 4: unendliche Kurtosis",
               status="critical" if last_alpha < 2 else "serious" if last_alpha < 3 else None)

    # --- chart 1: VaR / ES band against realised returns
    bx = [xi for xi, b in zip(x, breach) if b]
    out.chart(C.xy(
        f"Rollierender VaR und Expected Shortfall {lvl} gegen die tatsächlichen Renditen",
        [C.trace("Tagesrendite", x, r, color="muted", width=1),
         C.trace(f"−VaR {lvl}", x, -var, color=1, width=2),
         C.trace(f"−ES {lvl}", x, -es, color=2, width=1.5),
         C.trace("VaR-Verletzung", bx, r[breach], type="scatter", color="critical", size=6, decimate_to=None)],
        x=xa, y=C.axis("Tagesrendite", fmt="pct", digits=0),
        subtitle=f"Rote Punkte: Tage unter −VaR ({n_breach}, erwartet ≈ {round(n_valid * (1 - conf))}).",
    ))

    # --- chart 2: shape of the distribution
    out.chart(C.xy(
        f"Form der Verteilung im {w}-Tage-Fenster",
        [C.trace("Exzess-Kurtosis", x, kurt, color=1), C.trace("Schiefe", x, skew, color=2)],
        x=xa, y=C.axis("dimensionslos"),
        hlines=[C.hline(0, "Gauß")],
        subtitle="Kurtosis-Spitzen markieren Crash-Fenster; dann wird die Schiefe meist deutlich negativ.",
    ))

    # --- chart 3: tail index
    out.chart(C.xy(
        f"Rollierender Tail-Index α ({'Hill' if method == 'hill' else 'Pickands'})",
        [C.trace("α", x, alpha, color=1)],
        x=xa, y=C.axis("Tail-Index α"),
        hlines=[C.hline(2, "α = 2: Grenze zur unendlichen Varianz", color="critical"),
                C.hline(4, "α = 4: Kurtosis endlich")],
        subtitle="Kleiner = fetterer Rand. In kurzen Fenstern stark verrauscht.",
    ))

    # --- chart 4: 3D risk phase space
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    idx = np.flatnonzero(ok)
    if idx.size > 10:
        step = max(1, idx.size // 3000)
        hist = idx[::step]
        labels = [_fmt_date(x[i]) if data.dates else f"Tag {i}" for i in range(n)]
        score = X + Y + Z
        hot = idx[score[idx] >= np.quantile(score[idx], 0.97)]
        recent = idx[-min(w, idx.size):]
        out.chart(C.scatter3d(
            "Weg durch den 3D-Risikoraum",
            [C.path3d("Verlauf", X[hist], Y[hist], Z[hist], color="muted", mode="lines",
                      text=[labels[i] for i in hist]),
             C.path3d("Riskanteste 3 % der Tage", X[hot], Y[hot], Z[hot], color="critical", mode="markers", size=3,
                      text=[labels[i] for i in hot]),
             C.path3d(f"Letzte {len(recent)} Tage", X[recent], Y[recent], Z[recent], color=1, mode="lines+markers",
                      size=2.5, text=[labels[i] for i in recent]),
             C.path3d("Heute", [X[idx[-1]]], [Y[idx[-1]]], [Z[idx[-1]]], color=2, mode="markers", size=7,
                      text=[labels[idx[-1]]])],
            x=C.axis("VaR-Niveau (0–1)"), y=C.axis("ES ÷ VaR (0–1)"), z=C.axis("Kurtosis (0–1)"),
            subtitle="Alle Achsen min-max-normiert über die ganze Reihe. Ziehen zum Drehen.",
        ))

    beyond = w * (1 - conf)
    if beyond < 2:
        out.note(f"Im {w}-Tage-Fenster liegen im Mittel nur {_de(beyond, 1)} Tage jenseits des {lvl}-VaR. Der historische "
                 "VaR ist dann kaum mehr als der schlechteste Tag im Fenster und unterschätzt das Risiko systematisch – "
                 "nimm ein längeres Fenster oder ein niedrigeres Konfidenzniveau.", "warning")

    # --- compact table: calmest vs most stressed windows
    rows = []
    for name, sel in (("ruhigstes Viertel", var <= np.nanquantile(var, 0.25)),
                      ("stressigstes Viertel", var >= np.nanquantile(var, 0.75))):
        sel = sel & valid
        nxt = np.flatnonzero(sel)
        rows.append([name, np.nanmean(var[sel]), np.nanmean(es[sel]), np.nanmean(kurt[sel]), np.nanmean(skew[sel]),
                     np.nanmean(alpha[sel]), float(np.mean(breach[nxt])) if nxt.size else None])
    out.table("Ruhige gegen gestresste Phasen", ["Phase (nach VaR)", f"Ø VaR {lvl}", f"Ø ES {lvl}", "Ø Kurtosis",
                                                 "Ø Schiefe", "Ø Tail-Index", "Verletzungsquote"],
              rows, formats=["text", "pct", "pct", "num", "num", "num", "pct"],
              note="Verletzungsquote: Anteil der Tage in dieser Phase, an denen die Rendite unter −VaR fiel.")
