"""q-Gauß-Verteilung erkunden: wie der Tsallis-Index q Dichte, Quantile, Ränder und Entropie verändert."""

import numpy as np
from scipy import stats

from src.physics.tsallis_statistics import TsallisDistribution, TsallisEntropyAnalyzer

from .. import charts as C

SPEC = {
    "id": "tsallis-explorer",
    "title": "Tsallis-Verteilung (q-Gauß)",
    "group": "physics",
    "order": 70,
    "summary": "Wie der entropische Index q aus der Gauß-Glocke eine Verteilung mit Potenzgesetz-Rändern macht – "
               "Dichte, Quantile, Stichproben und Tsallis-Entropie.",
    "description": (
        "Die **q-Gauß-Verteilung** P(x) ∝ exp_q(−βx²) maximiert die Tsallis-Entropie S_q = (1 − Σpᵢ^q)/(q − 1), "
        "so wie die Normalverteilung die Shannon-Entropie maximiert. Die q-Exponentialfunktion "
        "exp_q(u) = [1 + (1 − q)u]^(1/(1−q)) fällt für q > 1 nicht exponentiell, sondern wie eine Potenz – daher die "
        "fetten Ränder mit Dichte ~ |x|^(−2/(q−1)). q = 1 ist exakt die Normalverteilung, q ≈ 1,4–1,5 passt "
        "typischerweise zu Tagesrenditen. Mathematisch ist sie eine skalierte Student-t-Verteilung mit "
        "ν = (3 − q)/(q − 1) Freiheitsgraden.\n\n"
        "So liest du die Diagramme: Die Gauß-Referenz hat dieselbe Varianz (für q ≥ 5/3 ist die Varianz unendlich, "
        "dann dasselbe β). Auf der Log-Skala liegt die q-Gauß-Kurve in der Mitte höher, an den Schultern tiefer und "
        "an den Rändern weit darüber – das typische Bild leptokurtischer Renditen. Im Log-Log-Plot ist der "
        "Rand eine Gerade; die gestrichelte Asymptote aus `tail_probability` trifft sie weit draußen.\n\n"
        "x ist in Einheiten von 1/√β angegeben (β = 1, μ = 0). Quantile kommen exakt aus `ppf`, die Stichproben "
        "aus der Student-t-Darstellung."
    ),
    "needs": [],
    "params": [
        {"name": "q", "type": "float", "default": 1.4, "min": 1.0, "max": 2.9, "step": 0.05,
         "label": "Entropischer Index q", "help": "1 = Gauß · 5/3: Varianz wird unendlich · 2: Cauchy-artig."},
        {"name": "x_range", "type": "int", "default": 8, "min": 3, "max": 30, "step": 1,
         "label": "Dichte-Bereich ± (in 1/√β)"},
        {"name": "n_samples", "type": "int", "default": 20000, "min": 1000, "max": 200000, "step": 1000,
         "label": "Stichproben"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.tsallis_statistics.TsallisDistribution",
        "src.physics.tsallis_statistics.TsallisDistribution.q_exponential",
        "src.physics.tsallis_statistics.TsallisDistribution.q_logarithm",
        "src.physics.tsallis_statistics.TsallisDistribution.pdf",
        "src.physics.tsallis_statistics.TsallisDistribution.cdf",
        "src.physics.tsallis_statistics.TsallisDistribution.ppf",
        "src.physics.tsallis_statistics.TsallisDistribution.sample",
        "src.physics.tsallis_statistics.TsallisDistribution.variance",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_exponent",
        "src.physics.tsallis_statistics.TsallisDistribution.tail_probability",
        "src.physics.tsallis_statistics.TsallisEntropyAnalyzer.compute_tsallis_entropy",
    ],
    "cost": "fast",
}

QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _tiny(p):
    """Probability cell: below 10⁻⁶ as German scientific text in % (2,0·10⁻⁷ % instead of 0,0000002 %)."""
    if p is None or not np.isfinite(p):
        return None
    if p < 1e-12:  # includes exact 0 from underflow (Gaussian tail far out)
        return "< 10⁻¹⁰ %"
    if 0 < p < 1e-6:
        mant, exp = f"{100 * p:.1e}".split("e")
        return mant.replace(".", ",") + "·10" + str(int(exp)).translate(_SUP) + " %"
    return float(p)


def run(params, data, out):
    q, xr, n, seed = params["q"], float(params["x_range"]), params["n_samples"], params["seed"]
    dist = TsallisDistribution(q=q, beta=1.0, mu=0.0)
    var = float(dist.variance())
    same_var = np.isfinite(var)
    g_sd = np.sqrt(var) if same_var else np.sqrt(1 / (2 * dist.beta))
    ref_label = "Gauß gleicher Varianz" if same_var else "Gauß mit gleichem β"
    nu = (3 - q) / (q - 1) if q > 1 else np.inf

    s = np.asarray(dist.sample(n, seed=seed), dtype=float)
    s_sorted = np.sort(s)

    # --- metrics
    tail_exp = float(dist.tail_exponent())
    out.metric("Tail-Exponent der Dichte", tail_exp if np.isfinite(tail_exp) else "∞ (Gauß)",
               "num" if np.isfinite(tail_exp) else "text", hint="P(x) ~ |x|^−2/(q−1)",
               status="critical" if tail_exp < 3 else "serious" if tail_exp < 5 else None)
    out.metric("Student-t-Freiheitsgrade ν", nu if np.isfinite(nu) else "∞", "num" if np.isfinite(nu) else "text",
               hint="(3 − q)/(q − 1); Momente existieren bis Ordnung < ν")
    out.metric("Varianz", var if same_var else "∞", "num" if same_var else "text",
               hint="1/(β(5 − 3q)), endlich nur für q < 5/3")
    p99, g99 = float(dist.ppf(0.99)), float(stats.norm.ppf(0.99, 0, g_sd))
    out.metric("99-%-Quantil", p99, "num", hint=f"{ref_label}: {_de(g99)}")
    p4 = float(dist.tail_probability(4 * g_sd, "both", exact=True))
    g4 = float(2 * stats.norm.sf(4))
    out.metric("P(|X| > 4σ)", p4, "pct", hint=f"σ der Gauß-Referenz; Gauß: {_de(100 * g4, 4)} %",
               status="critical" if p4 > 50 * g4 else "serious" if p4 > 5 * g4 else None)

    # --- chart 1: density on log scale with histogram and Gaussian reference
    x = np.linspace(-xr, xr, 601)
    pdf = dist.pdf(x)
    gpdf = stats.norm.pdf(x, 0, g_sd)
    edges = np.linspace(-xr, xr, 121)
    w = edges[1] - edges[0]
    dens = np.histogram(s, bins=edges)[0] / (len(s) * w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    floor = 0.5 / (len(s) * w)
    lo = max(1e-9, min(floor, float(pdf.min()) / 2))
    keep = lambda y: np.where(y >= lo, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Dichte auf logarithmischer Skala",
        [C.trace("Stichprobe", centers, np.where(dens > 0, dens, np.nan), type="bar", color=1),
         C.trace(f"q-Gauß (q = {_de(q)})", x, keep(pdf), color=2, width=2.5),
         C.trace(ref_label, x, keep(gpdf), color="muted", dash=True)],
        x=C.axis("x (in 1/√β)"), y=C.axis("Dichte", type="log", range=(lo, 2.5 * max(pdf.max(), gpdf.max()))),
        subtitle="Höhere Spitze, schmalere Schultern, viel dickere Ränder als die Gauß-Parabel.",
    ))

    # --- chart 2: tails on log-log axes
    dmax = float(np.clip(np.max(np.abs(s)), 10.0, 1e6))
    d = np.logspace(np.log10(0.3), np.log10(dmax), 70)
    exact = dist.cdf(-d)                       # P(X < -d), exact Student-t cdf
    asym = np.asarray(dist.tail_probability(d, "left"), dtype=float) if q > 1 else np.full_like(d, np.nan)
    emp = np.searchsorted(s_sorted, -d, side="left") / len(s)
    y_lo = 0.3 / len(s)
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Linker Rand im Log-Log-Plot: P(X < −x)",
        [C.trace("Stichprobe", d, pos(np.where(emp > 0, emp, np.nan)), color=1, width=2.5),
         C.trace("Exakt (cdf)", d, pos(exact), color=2),
         C.trace("Potenzgesetz-Asymptote", d, pos(asym), color=2, dash=True, width=1.5),
         C.trace(ref_label, d, pos(stats.norm.cdf(-d, 0, g_sd)), color="muted", dash=True)],
        x=C.axis("x (in 1/√β)", type="log"), y=C.axis("Wahrscheinlichkeit", type="log", fmt="pct", range=(y_lo, 1)),
        subtitle=(f"Steigung der Geraden: 1 − 2/(q − 1) = {_de(1 - tail_exp)}." if q > 1 else
                  "q = 1: Es gibt kein Potenzgesetz, der Rand fällt wie bei der Normalverteilung."),
    ))

    # --- chart 3: the q-exponential behind it all
    u = np.linspace(0, 20, 401)
    eq = dist.q_exponential(-u)
    out.chart(C.xy(
        "q-Exponentialfunktion exp_q(−u)",
        [C.trace(f"exp_q(−u), q = {_de(q)}", u, np.where(eq > 1e-9, eq, np.nan), color=2, width=2.5),
         C.trace("exp(−u)", u, np.where(np.exp(-u) > 1e-9, np.exp(-u), np.nan), color="muted", dash=True)],
        x=C.axis("u = βx²"), y=C.axis("Wert", type="log", range=(1e-9, 1.5)),
        subtitle="Die Dichte ist A·exp_q(−βx²). Für q > 1 fällt exp_q wie u^(−1/(q−1)) statt exponentiell.",
        height=320,
    ))

    # --- table: quantiles
    rows = []
    for p in QUANTILES:
        qq = float(dist.ppf(p))
        gq = float(stats.norm.ppf(p, 0, g_sd))
        rows.append([p, qq, gq, float(np.quantile(s, p)), (qq / gq) if p != 0.5 else None,
                     float(dist.cdf(qq))])
    out.table("Quantile", ["Wahrscheinlichkeit p", "q-Gauß (ppf)", ref_label, "Stichprobe", "Verhältnis", "cdf(ppf(p))"],
              rows, formats=["pct", "num", "num", "num", "x", "pct"],
              note="Die letzte Spalte ist die Probe aufs Exempel: cdf und ppf sind exakt invers.")

    # --- table: tail probabilities, exact vs asymptote
    trows = []
    for k in (2, 3, 4, 6, 10):
        dd = k * g_sd
        ex = float(dist.tail_probability(dd, "both", exact=True))
        asy = float(dist.tail_probability(dd, "both")) if q > 1 else None
        gg = float(2 * stats.norm.sf(k))
        ratio = ex / gg if gg > 1e-300 else None
        trows.append([f"{k} σ", ex, asy, float(np.mean(np.abs(s) > dd)), _tiny(gg),
                      "> 10⁵" if ratio is not None and ratio > 1e5 else ratio])
    out.table("Tail-Wahrscheinlichkeit P(|X| > kσ)", ["Schwelle", "exakt", "asymptotisch", "Stichprobe", "Gauß", "Faktor"],
              trows, formats=["text", "pct", "pct", "pct", "pct", "x"],
              note=f"σ = Standardabweichung der Referenz ({_de(g_sd, 3)}). Die Asymptote C·d^(1−α)/(α−1) mit dem "
                   "Dichte-Exponenten α = 2/(q − 1) gilt nur weit draußen und ist bei 100 % gekappt.")

    # --- table: entropy of the discretised densities (50 classes)
    analyzer = TsallisEntropyAnalyzer()
    ebins = np.linspace(-xr, xr, 51)
    mid = 0.5 * (ebins[:-1] + ebins[1:])
    erows, s_vals = [], []
    for name, p in ((f"q-Gauß (q = {_de(q)})", dist.pdf(mid)), (ref_label, stats.norm.pdf(mid, 0, g_sd))):
        p = np.asarray(p, dtype=float)
        p = p[p > 0] / p[p > 0].sum()
        s_q = float(analyzer.compute_tsallis_entropy(p, q))
        via_log = float(np.sum(p * dist.q_logarithm(1.0 / p)))
        erows.append([name, s_q, via_log, float(analyzer.compute_tsallis_entropy(p, 1.0))])
        s_vals.append(s_q)
    why = ""
    if q > 1 and s_vals[1] >= s_vals[0]:
        why = (" Dass die Gauß-Glocke hier das größere S_q hat, ist kein Widerspruch: Der q-Gauß maximiert S_q bei "
               "fester q-Varianz (Escort-Mittel Σpᵢ^q·xᵢ² / Σpᵢ^q), nicht bei fester gewöhnlicher Varianz.")
    out.table(f"Tsallis-Entropie S_q (q = {_de(q)}), 50 Klassen auf ±{xr:.0f}", ["Verteilung", "S_q",
              "Σ p·ln_q(1/p)", "Shannon S₁"], erows, formats=["text", "num", "num", "num"],
              note="S_q = (1 − Σpᵢ^q)/(q − 1) = Σ pᵢ·ln_q(1/pᵢ) mit dem q-Logarithmus ln_q(x) = (x^(1−q) − 1)/(1 − q). "
                   "Für q → 1 wird daraus die Shannon-Entropie." + why)

    if q >= 2:
        out.note("Für q ≥ 2 ist ν ≤ 1: Auch der Mittelwert existiert nicht mehr (Cauchy-artig).", "warning")
    elif not same_var:
        out.note("Für q ≥ 5/3 ist die Varianz unendlich – die Gauß-Referenz hat deshalb dasselbe β statt derselben Varianz.",
                 "info")
