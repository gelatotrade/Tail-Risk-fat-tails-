"""Stochastic volatility (CIR-like OU with upward jumps): price paths, volatility clustering, drawdowns."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import VolatilityOUModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "stochastic-vol",
    "title": "Stochastische Volatilität mit Vola-Sprüngen",
    "group": "physics",
    "order": 50,
    "summary": "Wenn die Volatilität selbst zufällig schwankt und springt: Vola-Cluster, fette Ränder und "
               "Drawdowns – obwohl jeder einzelne Tag normalverteilt ist.",
    "description": (
        "Die Volatilität folgt einem zum Mittel zurückkehrenden Prozess **dσ = κ(θ − σ)dt + η√σ dW + dJ**: Sie kehrt mit "
        "Geschwindigkeit **κ** zum Niveau **θ** zurück, schwankt mit der Vola-of-Vol **η** und springt im Schnitt "
        "**λ-mal pro Jahr** nach oben (je Sprung im Mittel +10 Prozentpunkte, exponentialverteilt). Der Kurs ist "
        "eine geometrische Brownsche Bewegung mit genau dieser Volatilität.\n\n"
        "So liest du die Diagramme: Jeder einzelne Tag ist normalverteilt – trotzdem hat die Verteilung aller "
        "Tagesrenditen fette Ränder, weil sich ruhige und stürmische Phasen mischen. Das Cluster-Diagramm zeigt "
        "die Autokorrelation: Die Renditen selbst sind unkorreliert, ihre **Beträge** aber nicht – auf große "
        "Tage folgen große Tage. Genau das sieht man auch an echten Märkten.\n\n"
        "Parameter: Großes κ lässt Vola-Schocks schnell abklingen (kurze Cluster), großes η und λ erzeugen "
        "heftigere Ausschläge. Durch die Sprünge liegt das langfristige Vola-Niveau über θ, im Mittel etwa bei "
        "θ + λ·0,1/κ."
    ),
    "needs": [],
    "params": [
        {"name": "kappa", "type": "float", "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
         "label": "Rückstellgeschwindigkeit κ", "unit": "pro Jahr"},
        {"name": "theta", "type": "float", "default": 0.2, "min": 0.05, "max": 1.0, "step": 0.01,
         "label": "Langfristige Volatilität θ", "help": "0,2 = 20 % p. a."},
        {"name": "eta", "type": "float", "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
         "label": "Vola-of-Vol η"},
        {"name": "jump_intensity", "type": "float", "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
         "label": "Vola-Sprünge pro Jahr λ", "help": "Jeder Sprung hebt σ im Mittel um 10 Prozentpunkte."},
        {"name": "sigma0", "type": "float", "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
         "label": "Startvolatilität σ₀"},
        {"name": "drift", "type": "float", "default": 0.0, "min": -0.2, "max": 0.3, "step": 0.01,
         "label": "Drift r", "unit": "pro Jahr"},
        {"name": "years", "type": "float", "default": 2.0, "min": 0.25, "max": 10.0, "step": 0.25,
         "label": "Laufzeit", "unit": "Jahre", "help": "Ein Zeitschritt = ein Handelstag."},
        {"name": "n_paths", "type": "int", "default": 100, "min": 1, "max": 500, "step": 10,
         "label": "Anzahl Pfade"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel",
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel.simulate_volatility",
        "src.physics.ornstein_uhlenbeck.VolatilityOUModel.simulate_price",
    ],
    "cost": "fast",
}

N_SHOW = 8
MAX_STEPS = 300_000  # paths x days (two pure-Python loops per path)
LAGS = np.arange(1, 41)
QUANTS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _gapped(times, rows):
    xs, ys = [], []
    for row in rows:
        xs.extend(times.tolist() + [None])
        ys.extend(row.tolist() + [np.nan])
    return xs, np.asarray(ys, dtype=float)


def _acf(x, lags):
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom <= 0:
        return np.zeros(len(lags))
    return np.array([np.sum(x[:-k] * x[k:]) / denom if k < len(x) else np.nan for k in lags])


def _max_drawdown(prices):
    peak = np.maximum.accumulate(prices, axis=1)
    return np.min(prices / peak - 1.0, axis=1)  # <= 0


def run(params, data, out):
    n_paths, years = params["n_paths"], params["years"]
    n_steps = int(round(years * 252))
    if n_paths * n_steps > MAX_STEPS:
        raise ToolError(f"{n_paths} Pfade × {n_steps} Tage sind zu viel für den Browser (Grenze "
                        f"{MAX_STEPS // 1000}.000 Tagesschritte). Verringere Pfade oder Laufzeit.")
    model = VolatilityOUModel(kappa=params["kappa"], theta=params["theta"], eta=params["eta"],
                              jump_intensity=params["jump_intensity"])

    prices, vols, rets = [], [], []
    for i in range(n_paths):
        if i % 20 == 0:
            out.progress(f"Pfad {i + 1} von {n_paths} …", i / n_paths)
        # seed once; later paths continue the same random stream
        res = model.simulate_price(100.0, params["sigma0"], years, n_steps=n_steps, r=params["drift"],
                                   seed=params["seed"] if i == 0 else None)
        prices.append(res["prices"])
        vols.append(res["volatility"])
        rets.append(res["returns"])
    prices, vols, rets = np.array(prices), np.array(vols), np.array(rets)
    t = res["times"]
    pooled = rets.ravel()
    dd = _max_drawdown(prices)
    level_jumps = params["theta"] + params["jump_intensity"] * 0.1 / params["kappa"]

    acf_abs = np.nanmean([_acf(np.abs(row), LAGS) for row in rets], axis=0)
    acf_raw = np.nanmean([_acf(row, LAGS) for row in rets], axis=0)

    out.metric("Median-Endkurs", np.median(prices[:, -1]), "num", hint=f"Start 100 · 5-%-Quantil "
               f"{_de(np.percentile(prices[:, -1], 5), 1)}")
    out.metric("Mittlere Volatilität σ_t", vols.mean(), "pct", digits=1,
               hint=f"θ = {_de(100 * params['theta'], 0)} %, mit Sprüngen ≈ {_de(100 * level_jumps, 0)} %")
    out.metric("Realisierte Vola p. a.", np.std(pooled) * np.sqrt(252), "pct", digits=1)
    out.metric("Höchste Volatilität", vols.max(), "pct", digits=0, hint="über alle Pfade und Tage")
    out.metric("Exzess-Kurtosis Tagesrenditen", stats.kurtosis(pooled), "num", digits=1,
               hint="0 = normalverteilt", status="serious" if stats.kurtosis(pooled) > 3 else None)
    out.metric("Autokorrelation |r|, Lag 1", acf_abs[0], "num", digits=3,
               hint=f"Renditen selbst: {_de(acf_raw[0], 3)}. |r| deutlich > 0 = Vola-Clustering")
    out.metric("Max. Drawdown (Median)", np.median(dd), "pct", digits=1)
    out.metric("Max. Drawdown (schlechteste 5 %)", np.percentile(dd, 5), "pct", digits=1,
               status="serious" if np.percentile(dd, 5) < -0.3 else None)
    out.metric("VaR 99 % (1 Tag)", -np.percentile(pooled, 1), "pct", hint="aus allen simulierten Tagen")

    show = min(N_SHOW, n_paths)
    gx, gy = _gapped(t, prices[:show])
    q05, q50, q95 = np.percentile(prices, [5, 50, 95], axis=0)
    band = [C.envelope("5–95 % aller Pfade", t, q05, q95, color=2)] if n_paths >= 20 else []
    out.chart(C.xy(
        "Kurspfade",
        band + [C.trace(f"{show} Beispielpfade", gx, gy, color=1, width=1, decimate_to=None),
                C.trace("Median aller Pfade", t, q50, color=2, width=2.5)],
        x=C.axis("Zeit (Jahre)"), y=C.axis("Kurs (Start 100)"), hlines=[C.hline(100.0)], height=360,
        subtitle=f"{n_paths} Pfade, ein Schritt je Handelstag."
                 + (" Die Fläche umfasst an jedem Tag die mittleren 90 % aller Pfade." if band else ""),
    ))
    gx, gy = _gapped(t, vols[:show])
    v05, v95 = np.percentile(vols, [5, 95], axis=0)
    vband = [C.envelope("5–95 % aller Pfade", t, v05, v95, color=2)] if n_paths >= 20 else []
    out.chart(C.xy(
        "Volatilität σ_t",
        vband + [C.trace(f"{show} Beispielpfade", gx, gy, color=1, width=1, decimate_to=None),
                 C.trace("Mittel aller Pfade", t, vols.mean(axis=0), color=2, width=2.5)],
        x=C.axis("Zeit (Jahre)"), y=C.axis("Volatilität p. a.", fmt="pct", digits=0),
        hlines=[C.hline(params["theta"], "θ"), C.hline(level_jumps, "θ + λ·0,1/κ")], height=340,
        subtitle="Senkrechte Sprünge sind Vola-Schocks; danach klingt σ mit Geschwindigkeit κ ab.",
    ))
    out.chart(C.xy(
        "Volatilitäts-Cluster: Autokorrelation",
        [C.trace("|Rendite|", LAGS, acf_abs, color=1, width=2.5),
         C.trace("Rendite", LAGS, acf_raw, color="muted", dash=True)],
        x=C.axis("Lag (Handelstage)", fmt="int"), y=C.axis("Autokorrelation"), hlines=[C.hline(0.0)],
        height=300, subtitle="Die Renditen sind unkorreliert, ihre Beträge nicht: Große Tage kommen in Clustern.",
    ))
    centers, heights = C.histogram(pooled, bins=100)
    sd = float(np.std(pooled))
    grid = np.linspace(centers[0], centers[-1], 300)
    gauss = stats.norm.pdf(grid, np.mean(pooled), sd)
    floor = float(np.min(heights[heights > 0])) / 3
    out.chart(C.xy(
        "Verteilung aller Tagesrenditen – logarithmisch",
        [C.trace("Simulation", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace("Normalverteilung, gleiche Vola", grid, np.where(gauss >= floor, gauss, np.nan),
                 color="muted", dash=True)],
        x=C.axis("Tagesrendite", fmt="pct", digits=1),
        y=C.axis("Dichte (log)", type="log", range=(floor, float(heights.max()) * 3)), height=320,
        subtitle="Mischung aus ruhigen und stürmischen Phasen: mehr ruhige Tage und mehr Extremtage als bei Gauß.",
    ))

    tot = prices[:, -1] / 100.0 - 1.0
    out.table("Verteilung über die Pfade", ["Quantil", "Endkurs", "Rendite gesamt", "Max. Drawdown", "Höchste σ_t"],
              [[f"{int(q * 100)} %", np.quantile(prices[:, -1], q), np.quantile(tot, q), np.quantile(dd, q),
                np.quantile(vols.max(axis=1), q)] for q in QUANTS],
              formats=["text", "num", "pct", "pct", "pct"],
              note="Jede Spalte ist für sich sortiert (das 5-%-Quantil des Drawdowns gehört nicht zwingend zum "
                   "selben Pfad wie das 5-%-Quantil des Endkurses).")
