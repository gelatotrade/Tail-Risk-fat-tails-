"""Volatilität schätzen: rolling window vs. EWMA (RiskMetrics) vs. realised, with a VaR backtest."""

import numpy as np
from scipy import stats

from src.utils.numerical import exponential_weighted_mean, rolling_window, safe_divide
from src.utils.statistics import compute_moments

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "volatility",
    "title": "Volatilität: rollierend, EWMA, realisiert",
    "group": "risk",
    "order": 60,
    "summary": "Wie schnell reagiert eine Volatilitätsschätzung auf Stress? Rollierendes Fenster gegen "
               "RiskMetrics-EWMA – gemessen an der danach realisierten Schwankung und am VaR-Backtest.",
    "description": (
        "Volatilität lässt sich nicht beobachten, nur schätzen. Das **rollierende Fenster** "
        "(`rolling_window`) gibt den letzten *n* Tagen gleiches Gewicht und vergisst alles davor schlagartig. "
        "Der **EWMA** (`exponential_weighted_mean` auf den quadrierten Renditen, RiskMetrics-Standard λ = 0,94) "
        "gewichtet jeden älteren Tag um den Faktor λ weniger: Er reagiert sofort auf einen Schock und lässt ihn "
        "langsam ausklingen. Beides wird mit der **realisierten** Volatilität der folgenden Tage verglichen – "
        "der Wert, den eine Prognose hätte treffen sollen.\n\n"
        "So liest du die Ergebnisse: Im Zeitverlauf siehst du, wer schneller auf Krisen reagiert. Die Tabelle "
        "misst den Prognosefehler (RMSE) und zählt **VaR-Verletzungen**: Tage, an denen der Verlust den "
        "Normal-VaR aus der Vola von gestern übertraf. Bei 99 % sollten es 1 % sein; deutlich mehr heißt, das "
        "Modell unterschätzt das Risiko. Teilt man die Renditen durch die Vola von gestern (`safe_divide`), "
        "schrumpft die Kurtosis: Ein Teil der fetten Ränder ist Volatilitäts-Clustering – der Rest bleibt.\n\n"
        "Parameter: Das **Fenster** bestimmt das Gedächtnis des rollierenden Schätzers, **λ** das des EWMA "
        "(näher an 1 = langsamer), der **Horizont** die Tage, über die die realisierte Vola gemessen wird."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 21, "min": 5, "max": 250, "step": 1,
         "label": "Rollierendes Fenster", "unit": "Tage", "help": "21 Handelstage ≈ ein Monat."},
        {"name": "lam", "type": "float", "default": 0.94, "min": 0.80, "max": 0.995, "step": 0.005,
         "label": "EWMA-Abklingfaktor λ", "help": "RiskMetrics: 0,94 für Tagesdaten. Halbwertszeit ≈ ln 0,5 / ln λ Tage."},
        {"name": "horizon", "type": "int", "default": 21, "min": 5, "max": 63, "step": 1,
         "label": "Horizont der realisierten Vola", "unit": "Tage"},
        {"name": "level", "type": "select", "default": "0.99", "label": "VaR-Niveau im Backtest",
         "options": [{"value": "0.99", "label": "99 %"}, {"value": "0.975", "label": "97,5 %"},
                     {"value": "0.95", "label": "95 %"}]},
    ],
    "covers": [
        "src.utils.numerical.rolling_window",
        "src.utils.numerical.exponential_weighted_mean",
        "src.utils.numerical.safe_divide",
        "src.utils.statistics.compute_moments",
    ],
    "cost": "fast",
}
ANN = np.sqrt(252)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    w, h, lam = int(params["window"]), int(params["horizon"]), float(params["lam"])
    level = float(params["level"])
    if n < w + h + 20:
        raise ToolError(f"Für Fenster {w} und Horizont {h} braucht es mindestens {w + h + 20} Beobachtungen.")
    x, xa = index_axis(data)

    # estimates known at the close of day t
    roll = np.full(n, np.nan)
    roll[w - 1:] = rolling_window(r, w).std(axis=1)
    ewma = np.sqrt(exponential_weighted_mean(r ** 2, alpha=1.0 - lam))
    ewma[0] = np.nan  # a single squared return is no estimate yet
    # what actually happened over the next h days (ex post)
    realised = np.full(n, np.nan)
    realised[:n - h] = np.sqrt(rolling_window(r ** 2, h).mean(axis=1))[1:]
    long_run = float(np.std(r))

    # forecasts of day t are judged against days t+1..t+h
    valid = np.isfinite(roll) & np.isfinite(realised) & np.isfinite(ewma)
    q = stats.norm.ppf(level)
    prev = lambda s: np.concatenate([[np.nan], s[:-1]])  # noqa: E731 - estimate from the day before
    rows, z_kurt, breaches_ewma = [], {}, None
    for key, name, est in (("roll", f"Rollierend ({w} Tage)", roll), ("ewma", f"EWMA (λ = {_de(lam, 3)})", ewma),
                           ("const", "Konstant (Gesamt-σ, Referenz)", np.full(n, long_run))):
        e, rv = est[valid] * ANN, realised[valid] * ANN
        sig_prev = prev(est)
        z = safe_divide(r, sig_prev, fill=np.nan)
        ok = np.isfinite(z)
        z_kurt[key] = float(compute_moments(z[ok])["kurtosis"])
        breach = ok & (r < -q * sig_prev)
        rate = float(breach.sum() / ok.sum())
        pval = float(stats.binomtest(int(breach.sum()), int(ok.sum()), 1 - level).pvalue)
        if key == "ewma":
            breaches_ewma = breach
        rmse = float(np.sqrt(np.mean((e - rv) ** 2))) if key != "const" else None
        rows.append([name, float(np.mean(e)), rmse, float(np.mean(e - rv)) if key != "const" else None,
                     float(np.corrcoef(e, rv)[0, 1]) if key != "const" else None, rate, pval, z_kurt[key]])
    raw_kurt = float(compute_moments(r)["kurtosis"])

    ratio = float(safe_divide(ewma[-1], long_run, fill=np.nan))
    out.metric("Vola heute (EWMA)", ewma[-1] * ANN, "pct", digits=1, hint="annualisiert",
               status="critical" if ratio > 2.5 else "serious" if ratio > 1.6 else "warning" if ratio > 1.2 else None)
    out.metric(f"Vola heute ({w} Tage)", roll[-1] * ANN, "pct", digits=1, hint="annualisiert")
    out.metric("Langfristige Vola", long_run * ANN, "pct", digits=1, hint=f"über alle {n} Tage")
    out.metric("EWMA ÷ langfristig", ratio, "x", hint="über 1: unruhiger als im Durchschnitt")
    out.metric(f"VaR-Verletzungen EWMA ({_de(level * 100, 1)} %)", rows[1][5], "pct", digits=2,
               hint=f"Soll {_de((1 - level) * 100, 1)} %", status="serious" if rows[1][6] < 0.05 and rows[1][5] > 1 - level else None)
    out.metric("Kurtosis nach Vola-Skalierung", z_kurt["ewma"], "num", digits=1,
               hint=f"vorher {_de(raw_kurt, 1)} – Rest = echte Sprünge")

    out.chart(C.xy(
        "Volatilität im Zeitverlauf (annualisiert)",
        [C.trace(f"realisiert in den nächsten {h} Tagen", x, realised * ANN, color="muted", width=1, dash=True),
         C.trace(f"rollierend ({w} Tage)", x, roll * ANN, color=1, width=1.3),
         C.trace(f"EWMA (λ = {_de(lam, 3)})", x, ewma * ANN, color=2, width=1.3)],
        x=xa, y=C.axis("Volatilität p. a.", fmt="pct", digits=0),
        hlines=[C.hline(long_run * ANN, "langfristig")],
        subtitle="Gestrichelt: was danach tatsächlich passierte. Der EWMA springt bei Schocks sofort, das Fenster "
                 "erst, wenn genug Krisentage darin liegen – und fällt abrupt, wenn sie herausfallen.",
    ))

    lags = np.arange(0, min(251, n))
    wt_e = (1 - lam) * lam ** lags
    wt_r = np.where(lags < w, 1.0 / w, 0.0)
    half = np.log(0.5) / np.log(lam)
    out.chart(C.xy(
        "Wie viel Gewicht hat ein Tag?",
        [C.trace(f"rollierend ({w} Tage)", lags, wt_r, type="step", color=1),
         C.trace(f"EWMA (λ = {_de(lam, 3)})", lags, wt_e, color=2)],
        x=C.axis("Tage zurück", fmt="int"), y=C.axis("Gewicht", fmt="pct", digits=1),
        vlines=[C.vline(half, f"EWMA-Halbwertszeit {_de(half, 1)} T")],
        subtitle="Das Fenster vergisst nach genau n Tagen alles, der EWMA vergisst allmählich.",
    ))

    edges = (-10, 10)
    traces, floor = [], np.inf
    for key, name, col, scale in (("const", "Rendite ÷ Gesamt-σ", "muted", np.full(n, long_run)),
                                  ("roll", f"÷ rollierende σ von gestern", 1, roll),
                                  ("ewma", "÷ EWMA-σ von gestern", 2, ewma)):
        z = safe_divide(r, prev(scale), fill=np.nan)
        z = z[np.isfinite(z)]
        centers, dens = C.histogram(z, bins=81, density=True, range=edges)
        floor = min(floor, float(np.min(dens[dens > 0])))
        traces.append(C.trace(f"{name} (Kurtosis {_de(z_kurt[key], 1)})", centers, np.where(dens > 0, dens, np.nan),
                              color=col, width=1.4))
    grid = np.linspace(*edges, 300)
    pdf = stats.norm.pdf(grid)
    traces.append(C.trace("Normalverteilung", grid, np.where(pdf >= floor / 4, pdf, np.nan), color="ink", width=1,
                          dash=True))
    out.chart(C.xy(
        "Standardisierte Renditen – logarithmisch",
        traces, x=C.axis("Rendite in Standardabweichungen"), y=C.axis("Dichte", type="log", range=(floor / 4, 1.0)),
        subtitle="Durch die Vola von gestern geteilt, rücken die Ränder näher an die Gauß-Kurve – aber nicht ganz.",
    ))

    var_line = -q * prev(ewma)
    idx = np.flatnonzero(breaches_ewma)
    out.chart(C.xy(
        f"VaR-Backtest mit EWMA ({_de(level * 100, 1)} %)",
        [C.trace("Tagesrendite", x, r, color=1, width=1),
         C.trace(f"−VaR aus der EWMA-Vola von gestern", x, var_line, color=2, width=1.2),
         C.trace("Verletzung", [x[i] for i in idx], r[idx], type="scatter", color="critical", size=6, decimate_to=None)],
        x=xa, y=C.axis("Rendite", fmt="pct", digits=1),
        subtitle=f"{len(idx)} Verletzungen bei {int(np.isfinite(var_line).sum())} Tagen – erwartet wären "
                 f"{_de((1 - level) * np.isfinite(var_line).sum(), 0)}.",
    ))

    out.table(f"Prognosegüte gegen die realisierte Vola der nächsten {h} Tage",
              ["Schätzer", "Ø Vola p. a.", "RMSE", "Verzerrung", "Korrelation", "VaR-Verletzungen", "p-Wert",
               "Kurtosis standardisiert"], rows,
              formats=["text", "pct", "pct", "pct", "num", "pct", "num", "num"],
              note=f"RMSE und Verzerrung in Vola-Punkten p. a. VaR mit Normalquantil {_de(q, 3)} × σ von gestern; "
                   "p-Wert: Binomialtest gegen die Soll-Quote (unter 0,05 = Modell passt nicht). Der konstante Schätzer "
                   "kennt die ganze Reihe und dient nur als Referenz.")
