"""q-Gauß an echte Renditen anpassen: q, β, Tsallis-Entropie und das Risikourteil des Tsallis-Modells."""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisDistribution, TsallisEntropyAnalyzer, TsallisTailRiskModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tsallis-fit",
    "title": "Tsallis-Fit & Risikostufe",
    "group": "physics",
    "order": 70,
    "summary": "Schätzt den Tsallis-Index q deiner Renditen auf drei Wegen, vergleicht Modell und Daten an den "
               "Rändern und zeigt die Risikostufe des Tsallis-Modells.",
    "description": (
        "Drei Schätzer für den entropischen Index **q**: Die **Momenten-Methode** invertiert die Kurtosis "
        "(k = 6(q − 1)/(7 − 5q)) und kann deshalb nur q < 7/5 erkennen – darüber ist die Kurtosis unendlich. Die "
        "**Maximum-Likelihood-Methode** passt q und β direkt an die Dichte an und sieht auch schwerere Ränder. Der "
        "**Hill-Weg** (`estimate_optimal_q`) rechnet den Tail-Index der 5 % größten |Renditen| in q um.\n\n"
        "`TsallisTailRiskModel.update` nutzt den Momenten-Fit: Es berechnet die Wahrscheinlichkeit für 3- bis "
        "6-Sigma-Tage, das Verhältnis zur Normalverteilung, die Tsallis-Entropie der Renditeverteilung (50 Klassen) "
        "und daraus eine Risikostufe. „Erhöht“ gilt ab q > 1,3 und S_q > 1, „Hoch“ ab q > 1,5 und S_q > 1,5, "
        "„Extrem“ ab q > 1,8 und S_q > 2. Weil der Momenten-Fit unter 1,4 bleibt, erreicht das Modell die beiden "
        "obersten Stufen praktisch nicht – vergleiche deshalb auch das MLE-q.\n\n"
        "So liest du die Diagramme: Im Log-Log-Plot zeigt sich, ob das Modell die Häufigkeit großer Tage trifft; im "
        "Q-Q-Plot liegen die Punkte auf der Diagonale, wenn der q-Gauß auch die extremen Quantile beschreibt."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "fit", "type": "select", "default": "mle", "label": "Fit für Ränder und Q-Q-Plot",
         "options": [{"value": "mle", "label": "Maximum Likelihood"}, {"value": "moments", "label": "Momente (Kurtosis)"}]},
        {"name": "bins", "type": "int", "default": 90, "min": 30, "max": 200, "step": 10,
         "label": "Klassen im Histogramm"},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisDistribution.fit",
        "src.physics.tsallis_statistics.TsallisDistribution.log_pdf",
        "src.physics.tsallis_statistics.TsallisDistribution.ppf",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_probability",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.estimate_optimal_q",
        "src.physics.tsallis_statistics.TsallisTailRiskModel",
        "src.physics.tsallis_statistics.TsallisTailRiskModel.update",
    ],
    "cost": "fast",
}

LEVELS = {"NORMAL": ("good", "Normal"), "ELEVATED": ("warning", "Erhöht"),
          "HIGH": ("serious", "Hoch"), "EXTREME": ("critical", "Extrem")}
STEP = {"NORMAL": 1, "ELEVATED": 2, "HIGH": 3, "EXTREME": 4}


def _days(x):
    """Expected number of days: tiny Gaussian expectations as text instead of 4,96e-6."""
    return "< 0,001" if x < 1e-3 else float(x)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _sd(dist):
    v = float(dist.variance())
    return np.sqrt(v) if np.isfinite(v) else "∞"


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = len(r)
    try:
        mom = TsallisDistribution.fit(r, method="moments")
        mle = TsallisDistribution.fit(r, method="mle")
    except ValueError as exc:
        raise ToolError(f"Der Tsallis-Fit ist nicht möglich: Die Renditen sind (fast) konstant ({exc}).")
    q_hill = float(TsallisEntropyAnalyzer().estimate_optimal_q(r))
    model = TsallisTailRiskModel()
    u = model.update(r)
    mu, sd = float(np.mean(r)), float(np.std(r))
    gauss = TsallisDistribution(q=1.0, beta=1.0 / (2 * sd ** 2), mu=mu)  # q = 1 is the normal distribution
    chosen, cname = (mle, "MLE") if params["fit"] == "mle" else (mom, "Momente")

    level, label = LEVELS.get(u["risk_level"], ("warning", u["risk_level"]))
    out.metric("q (Maximum Likelihood)", mle.q, "num", hint="1 = Gauß; Aktien meist 1,4–1,6")
    out.metric("q (Momente)", mom.q, "num", hint="aus der Kurtosis, nur bis 7/5 identifizierbar")
    out.metric("q (Hill)", q_hill, "num", hint="aus dem Tail-Index der größten 5 %")
    out.metric("Tail-Exponent der Dichte", float(chosen.tail_exponent()), "num",
               hint=f"2/(q − 1), {cname}-Fit; < 3 hieße unendliche Varianz")
    out.metric("Tsallis-Entropie S_q", u["tsallis_entropy"], "num", hint="Histogramm mit 50 Klassen, q aus Momenten")
    out.metric("Risikostufe (Tsallis-Modell)", label, "text", status=level,
               badge=f"Stufe {STEP.get(u['risk_level'], '?')} von 4",
               hint=f"aus q = {_de(u['q'])} und S_q = {_de(u['tsallis_entropy'])}")
    ratio4 = float(u["fat_tail_ratio"]["4sigma"])
    out.metric("4σ-Tage gegenüber Gauß", ratio4, "x", hint="Wahrscheinlichkeit laut Momenten-Fit ÷ Gauß-Wert",
               status="serious" if ratio4 > 5 else None,
               badge="Gauß unterschätzt" if ratio4 > 5 else "")

    # --- table: the estimators side by side
    def ll(d):
        return float(np.sum(d.log_pdf(r)))

    fits = [("Momente (Kurtosis)", mom, 3), ("Maximum Likelihood", mle, 3), ("Normalverteilung (q = 1)", gauss, 2)]
    rows = []
    for name, d, k in fits:
        lv = ll(d)
        rows.append([name, float(d.q), float(d.beta), _sd(d), float(d.tail_exponent()) if d.q > 1 else "∞", lv, 2 * k - 2 * lv])
    rows.insert(2, ["Hill (estimate_optimal_q)", q_hill, None, None, 2 / (q_hill - 1), None, None])
    out.table("Drei Wege zu q", ["Methode", "q", "β", "Modell-σ täglich", "Tail-Exponent", "Log-Likelihood", "AIC"],
              rows, formats=["text", "num", "num", "pct", "num", "num", "num"],
              note="Kleineres AIC = bessere Beschreibung der gesamten Verteilung. β ist für Tagesrenditen groß, weil "
                   "1/√β die Breite in Renditeeinheiten ist.")

    # --- table: model vs data in the tails, as numbers of days
    med = float(np.median(r))
    trows = []
    for k in (3, 4, 5, 6):
        key = f"{k}sigma"
        obs = int(np.sum(np.abs(r - med) > k * sd))
        p_mle = float(mle.tail_probability(mle.mu + k * sd, "both", exact=True))
        ratio = float(u["fat_tail_ratio"][key])
        trows.append([f"{k} σ", obs, _days(p_mle * n), _days(float(u["extreme_probabilities"][key]) * n),
                      _days(float(2 * stats.norm.sf(k)) * n), "> 10⁵" if ratio > 1e5 else ratio])
    out.table("Extreme Tage: Modell vs. Daten",
              ["Schwelle", "Tage beobachtet", "erwartet q-Gauß (MLE)", "erwartet q-Gauß (Momente)", "erwartet Gauß",
               "Faktor Momente / Gauß"],
              trows, formats=["text", "int", "num", "num", "num", "x"],
              note=f"Tage mit |r − Median| > k·σ unter {n} Tagen; erwartet = Modellwahrscheinlichkeit × {n}. Die "
                   "Momente-Spalte und der Faktor stammen direkt aus `TsallisTailRiskModel.update`.")

    # --- chart 1: histogram vs both fits, log scale
    centers, heights = C.histogram(r, bins=params["bins"], density=True)
    grid = np.linspace(centers[0], centers[-1], 500)
    floor = float(np.min(heights[heights > 0])) / 4
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Tagesrenditen und q-Gauß-Fits",
        [C.trace("Beobachtet", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace(f"q-Gauß MLE (q = {_de(mle.q)})", grid, keep(mle.pdf(grid)), color=2, width=2.5),
         C.trace(f"q-Gauß Momente (q = {_de(mom.q)})", grid, keep(mom.pdf(grid)), color=3, width=1.5),
         C.trace("Normalverteilung", grid, keep(gauss.pdf(grid)), color="muted", dash=True)],
        x=C.axis("Tagesrendite", fmt="pct", digits=1), y=C.axis("Dichte", type="log", range=(floor, float(heights.max()) * 3)),
        subtitle="Logarithmische Skala. Die Gauß-Parabel fällt an den Rändern viel zu früh ab.",
    ))

    # --- chart 2: tails P(|r - mu| > x), log-log
    a = np.abs(r - chosen.mu)
    xs = np.logspace(np.log10(max(np.median(a), 1e-6)), np.log10(a.max() * 1.6), 60)
    emp = np.array([np.mean(a > v) for v in xs])
    exact = np.asarray(chosen.tail_probability(chosen.mu + xs, "both", exact=True), dtype=float)
    asym = np.asarray(chosen.tail_probability(chosen.mu + xs, "both"), dtype=float)
    y_lo = 0.5 / n
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    tail_traces = [
        C.trace("Beobachtet", xs, pos(np.where(emp > 0, emp, np.nan)), type="step", color=1, width=2.5),
        C.trace(f"q-Gauß {cname}", xs, pos(exact), color=2),
        C.trace("Asymptote", xs, pos(asym), color=2, dash=True, width=1.5),
        C.trace("Gauß", xs, pos(2 * stats.norm.sf(xs / sd)), color="muted", dash=True)]
    out.chart(C.xy(
        "Ränder im Log-Log-Plot: P(|r − μ| > x)",
        [t for t in tail_traces if np.any(np.isfinite(t["y"]))],
        x=C.axis("Abstand x vom Zentrum", type="log", fmt="pct", digits=1),
        y=C.axis("Anteil der Tage", type="log", fmt="pct", range=(y_lo, 1.0)),
        subtitle="Potenzgesetze sind Geraden. Die Asymptote aus `tail_probability` gilt erst weit draußen.", height=370,
    ))

    # --- chart 3: QQ plot against the chosen fit (tails kept, centre thinned)
    rs = np.sort(r)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = chosen.ppf(probs)
    tail = max(20, n // 50)
    idx = np.unique(np.r_[np.arange(0, n, max(1, n // 800)), np.arange(tail), np.arange(n - tail, n)])
    lim = [float(min(rs[0], theo[0])), float(max(rs[-1], theo[-1]))]
    out.chart(C.xy(
        f"Q-Q-Plot gegen den q-Gauß ({cname})",
        [C.trace("Perfekte Übereinstimmung", lim, lim, color="muted", dash=True),
         C.trace("Renditen", theo[idx], rs[idx], type="scatter", color=1, size=5, decimate_to=None)],
        x=C.axis("Quantil des q-Gauß", fmt="pct", digits=1), y=C.axis("Beobachtetes Quantil", fmt="pct", digits=1),
        subtitle="Knicken die Enden von der Diagonale ab, sind die echten Ränder schwerer (oder leichter) als im Modell.",
    ))

    thresholds = "Erhöht ab q > 1,3 und S_q > 1 · Hoch ab q > 1,5 und S_q > 1,5 · Extrem ab q > 1,8 und S_q > 2"
    out.set_verdict(level, f"Tsallis-Risikostufe: {label}",
                    f"q = {_de(u['q'])} (Momente) und S_q = {_de(u['tsallis_entropy'])}. 4σ-Tage sind laut Modell "
                    f"{ratio4:.0f}-mal so häufig wie bei Gauß. Schwellen: {thresholds}.")
    if mle.q > 1.42 and mom.q < 1.4:
        out.note(f"Das MLE-q ({_de(mle.q)}) liegt über 7/5 – dort ist die Kurtosis unendlich und der Momenten-Fit "
                 f"bleibt zwangsläufig darunter ({_de(mom.q)}). Die Risikostufe beruht auf dem Momenten-Fit.", "info")
