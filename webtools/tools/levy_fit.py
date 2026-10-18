"""Lévy-stabile Verteilung an echte Renditen anpassen: Tail-Index, VaR und Expected Shortfall im Vergleich."""

import numpy as np
from scipy import stats

from src.physics.levy_flight import LevyFlightProcess, LevyStableDistribution, estimate_tail_index

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "levy-fit",
    "title": "Lévy-Fit, Tail-Index & VaR",
    "group": "physics",
    "order": 20,
    "summary": "Passt eine Lévy-stabile Verteilung an deine Renditen an, schätzt den Tail-Index nach Hill und "
               "Pickands und vergleicht Lévy-, historischen und Gauß-VaR.",
    "description": (
        "Der **Quantil-Schätzer von McCulloch** liest α, β, γ und δ aus fünf Quantilen der Renditen ab: Wie weit "
        "liegen 5 %- und 95 %-Quantil im Verhältnis zum Interquartilsabstand auseinander? Daraus simuliert "
        "`LevyFlightProcess` den Value at Risk und den Expected Shortfall. Über mehrere Tage aggregiert er exakt "
        "stabil: Die Skala wächst mit h^(1/α) statt mit √h.\n\n"
        "Der **Hill-Schätzer** misst den Tail-Index direkt an den 10 % größten absoluten Renditen. Das ist der "
        "Prüfstein: Lévy-stabil verlangt α < 2 (unendliche Varianz). Aktienindizes zeigen nach Hill meist α ≈ 2,5–4, "
        "also fette Ränder bei endlicher Varianz. Dann überzeichnet das Lévy-Modell die extremsten Verluste, "
        "während die Normalverteilung sie unterschätzt. Pickands ist ein zweiter, sehr verrauschter Schätzer.\n\n"
        "So liest du die Diagramme: Im Log-Log-Plot ist jedes Potenzgesetz eine Gerade. Die Hill-Gerade folgt den "
        "Daten, die Lévy-Gerade ist flacher (Steigung −α). Die VaR-Kurve zeigt, ab welchem Konfidenzniveau sich "
        "die Modelle trennen. Renditen sind tägliche Log-Renditen; VaR und ES sind Verluste als positive Dezimalzahl."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Konfidenzniveau",
         "options": [{"value": 0.95, "label": "95 %"}, {"value": 0.975, "label": "97,5 %"},
                     {"value": 0.99, "label": "99 %"}, {"value": 0.995, "label": "99,5 %"},
                     {"value": 0.999, "label": "99,9 %"}]},
        {"name": "horizon", "type": "int", "default": 1, "min": 1, "max": 20, "step": 1,
         "label": "Horizont (Handelstage)", "help": "Historisch: überlappende h-Tages-Summen der Log-Renditen."},
        {"name": "n_simulations", "type": "int", "default": 100000, "min": 20000, "max": 500000, "step": 10000,
         "label": "Monte-Carlo-Ziehungen"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyStableDistribution.fit",
        "src.physics.levy_flight.LevyStableDistribution.pdf",
        "src.physics.levy_flight.LevyStableDistribution.tail_probability",
        "src.physics.levy_flight.LevyFlightProcess",
        "src.physics.levy_flight.LevyFlightProcess.compute_var",
        "src.physics.levy_flight.LevyFlightProcess.compute_cvar",
        "src.physics.levy_flight.estimate_tail_index",
    ],
    "cost": "fast",
}

TABLE_LEVELS = (0.95, 0.975, 0.99, 0.995, 0.999)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _pct(x, digits=1):
    return _de(100 * x, digits) + " %"


def _hist_var_es(rh, c):
    var = -float(np.quantile(rh, 1 - c))
    tail = rh[rh <= -var]
    return var, (-float(np.mean(tail)) if tail.size else var)


def _gauss_var_es(mu, sd, c):
    z = stats.norm.ppf(1 - c)
    return -(mu + sd * z), -(mu - sd * stats.norm.pdf(z) / (1 - c))


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    c, h, nsim, seed = float(params["confidence"]), params["horizon"], params["n_simulations"], params["seed"]
    n = len(r)
    if n < 2 * h + 50:
        raise ToolError(f"Für einen Horizont von {h} Tagen ist die Reihe zu kurz. Verkürze den Horizont.")

    try:
        dist = LevyStableDistribution.fit(r, method="quantile")
    except ValueError as exc:
        raise ToolError(f"Der Lévy-Fit ist nicht möglich: Die Renditen sind (fast) konstant ({exc}).")
    hill = float(estimate_tail_index(r, "hill"))
    try:
        pick = float(estimate_tail_index(r, "pickands"))
    except ValueError:
        pick = float("nan")
    proc = LevyFlightProcess(dist)

    # h-day returns: Lévy aggregates exactly, historical = overlapping sums, Gauss = sqrt(h)
    rh = np.convolve(r, np.ones(h), mode="valid") if h > 1 else r
    mu, sd = float(np.mean(r)), float(np.std(r))
    mu_h, sd_h = mu * h, sd * np.sqrt(h)

    out.progress("Monte-Carlo-VaR des Lévy-Modells …", 0.2)
    var_l = float(proc.compute_var(c, h, nsim, seed=seed))
    es_l = float(proc.compute_cvar(c, h, nsim, seed=seed))
    var_h, es_h = _hist_var_es(rh, c)
    var_g, es_g = _gauss_var_es(mu_h, sd_h, c)

    lvl = f"{_de(100 * c, 1 if c in (0.975, 0.995, 0.999) else 0)} %"
    out.metric("Lévy α (Quantil-Fit)", dist.alpha, "num", hint="2 = Gauß · < 2: unendliche Varianz")
    out.metric("Lévy β (Schiefe)", dist.beta, "num", hint="< 0: linker Rand schwerer")
    out.metric("Lévy-Skala γ", dist.gamma, "pct", hint="täglich; Gauß hätte γ = σ/√2")
    out.metric("Tail-Index (Hill)", hill, "num", hint="Top 10 % der |Renditen|; < 2 hieße unendliche Varianz",
               status="critical" if hill < 2 else "serious" if hill < 3 else "warning" if hill < 4 else "good")
    out.metric("Tail-Index (Pickands)", pick if np.isfinite(pick) else "n. v.", "num" if np.isfinite(pick) else "text",
               hint="sehr verrauscht, oft unplausibel")
    out.metric(f"VaR {lvl} · Lévy", var_l, "pct", hint=f"{h} Tag{'e' if h > 1 else ''} · {nsim:,} Ziehungen".replace(",", "."))
    out.metric(f"Expected Shortfall {lvl} · Lévy", es_l, "pct")
    out.metric(f"VaR {lvl} · historisch", var_h, "pct", hint=f"ES {_pct(es_h)}")
    out.metric(f"VaR {lvl} · Gauß", var_g, "pct", hint=f"ES {_pct(es_g)}")

    # --- chart 1: distribution, log scale
    centers, heights = C.histogram(r, bins=90, density=True)
    span = max(abs(float(r.min()) - dist.delta), abs(float(r.max()) - dist.delta)) * 1.05
    span = min(span, 80 * dist.gamma)  # beyond ~π·n_points/100 γ the Fourier grid would alias
    grid = np.linspace(dist.delta - span, dist.delta + span, 361)
    n_points = int(np.clip(100 * span / dist.gamma, 1000, 8000))
    lpdf = np.asarray(dist.pdf(grid, n_points=n_points), dtype=float)
    gpdf = stats.norm.pdf(grid, mu, sd)
    floor = float(np.min(heights[heights > 0])) / 4
    keep = lambda y: np.where(y >= floor, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Tagesrenditen: Daten vs. Lévy-Fit vs. Gauß",
        [C.trace("Beobachtet", centers, np.where(heights > 0, heights, np.nan), type="bar", color=1),
         C.trace(f"Lévy-stabil (α = {_de(dist.alpha)})", grid, keep(lpdf), color=2, width=2.5),
         C.trace("Normalverteilung", grid, keep(gpdf), color="muted", dash=True)],
        x=C.axis("Tagesrendite", fmt="pct", digits=1),
        y=C.axis("Dichte", type="log", range=(floor, float(np.max(heights)) * 3)),
        subtitle="Logarithmische Skala: In der Mitte passen beide Modelle, an den Rändern trennen sie sich.",
    ))

    # --- chart 2: tails on log-log axes, P(|r| > x)
    a = np.sort(np.abs(r))[::-1]
    a = a[a > 0]
    k = min(max(10, int(0.1 * n)), n - 1, len(a) - 1)  # same k as estimate_tail_index
    x_lo, x_hi = float(np.median(a)), float(a[0]) * 1.6
    xs = np.logspace(np.log10(x_lo), np.log10(x_hi), 60)
    emp = np.array([np.mean(np.abs(r) > v) for v in xs])
    hill_line = (k / n) * (xs / a[k]) ** (-hill)
    sample = np.asarray(dist.sample(nsim, seed=seed), dtype=float)
    lev_mc = np.array([np.mean(np.abs(sample - dist.delta) > v) for v in xs])
    lev_asym = np.asarray(dist.tail_probability(dist.delta + xs, "both"), dtype=float)
    gauss = 2 * stats.norm.sf(xs / sd)
    y_lo = 0.5 / n
    pos = lambda y: np.where(np.asarray(y) >= y_lo, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Ränder im Log-Log-Plot: P(|r| > x)",
        [C.trace("Beobachtet", xs, pos(np.where(emp > 0, emp, np.nan)), type="step", color=1, width=2.5),
         C.trace(f"Hill (α = {_de(hill)})", xs[xs >= a[k]], pos(hill_line[xs >= a[k]]), color=3, dash=True),
         C.trace(f"Lévy (α = {_de(dist.alpha)})", xs, pos(np.where(lev_mc > 0, lev_mc, np.nan)), color=2),
         C.trace("Lévy-Asymptote", xs, pos(np.where(lev_asym < 1, lev_asym, np.nan)), color=2, dash=True, width=1.5),
         C.trace("Gauß", xs, pos(gauss), color="muted", dash=True)],
        x=C.axis("Absolute Tagesrendite x", type="log", fmt="pct", digits=1),
        y=C.axis("Anteil der Tage", type="log", fmt="pct", range=(y_lo, 1.0)),
        subtitle="Steigung = −Tail-Index. Die Lévy-Linie ist eine Simulation aus dem Fit. Ist sie flacher als die "
                 "Daten, überschätzt das Modell Crashs.", height=380,
    ))

    # --- chart 3: VaR across confidence levels
    out.progress("VaR-Kurve über die Konfidenzniveaus …", 0.6)
    tail_p = np.logspace(np.log10(max(1.0 / len(rh), 5e-4)), -1, 14)
    v_l = [float(proc.compute_var(1 - p, h, nsim, seed=seed)) for p in tail_p]
    v_h = [_hist_var_es(rh, 1 - p)[0] for p in tail_p]
    v_g = [_gauss_var_es(mu_h, sd_h, 1 - p)[0] for p in tail_p]
    out.chart(C.xy(
        f"VaR je Konfidenzniveau ({h} Tag{'e' if h > 1 else ''})",
        [C.trace("Historisch", tail_p, v_h, color=1, width=2.5),
         C.trace("Lévy-stabil", tail_p, v_l, color=2),
         C.trace("Normalverteilung", tail_p, v_g, color="muted", dash=True)],
        x=C.axis("Überschreitungswahrscheinlichkeit 1 − c", type="log", fmt="pct", digits=1),
        y=C.axis("Value at Risk (Verlust)", fmt="pct", digits=1),
        subtitle="Links liegen die extremen Niveaus (99,9 %), rechts das 90 %-Niveau.",
    ))

    # --- tables
    rows = []
    for lev in TABLE_LEVELS:
        vh, eh = _hist_var_es(rh, lev)
        vg, eg = _gauss_var_es(mu_h, sd_h, lev)
        rows.append([lev, float(proc.compute_var(lev, h, nsim, seed=seed)), vh, vg,
                     float(proc.compute_cvar(lev, h, nsim, seed=seed)), eh, eg])
    out.table(
        f"VaR und Expected Shortfall ({h} Tag{'e' if h > 1 else ''})",
        ["Niveau", "VaR Lévy", "VaR historisch", "VaR Gauß", "ES Lévy", "ES historisch", "ES Gauß"],
        rows, formats=["pct", "pct", "pct", "pct", "pct", "pct", "pct"],
        note="Lévy: Monte-Carlo aus S1(α, β, h^(1/α)·γ, h·δ). Historisch: empirisches Quantil der "
             + ("überlappenden h-Tages-Summen." if h > 1 else "Tagesrenditen.") + " Gauß: μ·h und σ·√h.",
    )
    if h == 1:
        exc_rows = []
        for lev in (0.95, 0.99, 0.995):
            vl = float(proc.compute_var(lev, 1, nsim, seed=seed))
            vg = _gauss_var_es(mu, sd, lev)[0]
            exc_rows.append([lev, 1 - lev, float(np.mean(r < -vl)), float(np.mean(r < -vg)),
                             int(round((1 - lev) * n)), int(np.sum(r < -vl)), int(np.sum(r < -vg))])
        out.table(
            "Überschreitungen im Datensatz",
            ["Niveau", "Soll-Quote", "Quote Lévy", "Quote Gauß", "Soll-Tage", "Tage Lévy", "Tage Gauß"],
            exc_rows, formats=["pct", "pct", "pct", "pct", "int", "int", "int"],
            note="In-sample: Anteil der Tage, an denen der Verlust den VaR überstieg. Weniger als die Soll-Quote "
                 "= zu vorsichtig, mehr = Risiko unterschätzt.",
        )

    # --- verdict: tail index of the data, and how the two models miss it at 99.9 %
    vl9 = float(proc.compute_var(0.999, h, nsim, seed=seed))
    vh9 = _hist_var_es(rh, 0.999)[0]
    vg9 = _gauss_var_es(mu_h, sd_h, 0.999)[0]

    def verb(model, ref):
        q = model / ref if ref > 0 else 1.0
        return "überzeichnet" if q > 1.25 else "unterschätzt" if q < 0.8 else "passt"

    level = "critical" if hill < 2 else "serious" if hill < 3 else "warning" if hill < 4 else "good"
    title = (f"Tail-Index ≈ {_de(hill)}: " + ("unendliche Varianz möglich" if hill < 2 else
                                               "fette Ränder bei endlicher Varianz" if hill < 4 else
                                               "nur mäßig fette Ränder"))
    out.set_verdict(level, title,
                    f"99,9-%-VaR historisch {_pct(vh9)} · Lévy-stabil (α = {_de(dist.alpha)}) {_pct(vl9)}, "
                    f"{verb(vl9, vh9)} · Normalverteilung {_pct(vg9)}, {verb(vg9, vh9)}.")
    if len(rh) * 0.001 < 3:
        out.note("Für das 99,9-%-Niveau gibt es in dieser Reihe nur eine Handvoll Beobachtungen – der historische "
                 "Wert ist entsprechend unsicher.", "info")
