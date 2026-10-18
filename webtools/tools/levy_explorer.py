"""Lévy-stabile Verteilung erkunden: α, β, Dichte, Stichproben und Potenzgesetz-Ränder."""

import numpy as np
from scipy import stats

from src.physics.levy_flight import LevyStableDistribution

from .. import charts as C

SPEC = {
    "id": "levy-explorer",
    "title": "Lévy-stabile Verteilung erkunden",
    "group": "physics",
    "order": 10,
    "summary": "Wie der Stabilitätsindex α und die Schiefe β Dichte, Ränder und Zufallspfade einer "
               "Lévy-Verteilung gegenüber der Gauß-Kurve verändern.",
    "description": (
        "Eine **Lévy-stabile Verteilung** entsteht, wenn sich sehr viele unabhängige Schocks mit unendlicher "
        "Varianz addieren – das Gegenstück zum zentralen Grenzwertsatz für fette Ränder. **α** (0 < α ≤ 2) "
        "steuert die Ränder: P(|X| > x) fällt wie x^−α, bei α = 2 erhältst du die Normalverteilung. **β** "
        "verlagert Gewicht in einen Rand: β < 0 macht große Verluste wahrscheinlicher als große Gewinne. Alle "
        "Achsen sind in Vielfachen der Skala γ angegeben (γ = 1, δ = 0); für Tagesrenditen liegt γ meist bei "
        "0,5–1 %.\n\n"
        "So liest du die Diagramme: Auf der logarithmischen Dichte-Skala ist die Gauß-Kurve eine nach unten "
        "offene Parabel, die Lévy-Dichte fällt dagegen nur wie eine Potenz. Im Log-Log-Plot der Ränder wird das "
        "Potenzgesetz zur Geraden mit Steigung −α; die gestrichelten Asymptoten C_α(1 ± β)·x^−α gelten erst weit "
        "draußen. Der Zufallspfad zeigt, was das für einen Kurs bedeutet: gleiche Skala, aber einzelne Sprünge "
        "dominieren.\n\n"
        "Die Dichte kommt aus der Fourier-Inversion der charakteristischen Funktion, die Stichproben aus dem "
        "Chambers-Mallows-Stuck-Verfahren (S1-Parametrisierung wie `scipy.stats.levy_stable`). Die Gauß-Referenz "
        "ist der Fall α = 2 mit derselben Skala γ, also eine Normalverteilung mit σ = √2·γ."
    ),
    "needs": [],
    "params": [
        {"name": "alpha", "type": "float", "default": 1.7, "min": 0.5, "max": 2.0, "step": 0.05,
         "label": "α – Stabilitätsindex", "help": "2 = Gauß. Je kleiner, desto fetter die Ränder. Aktien: meist 1,4–1,9."},
        {"name": "beta", "type": "float", "default": -0.1, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "β – Schiefe", "help": "< 0: linker Rand (Crashs) schwerer als rechter. Bei α = 2 ohne Wirkung."},
        {"name": "x_range", "type": "int", "default": 12, "min": 4, "max": 50, "step": 1,
         "label": "Dichte-Bereich ± (in γ)", "help": "Größere Bereiche brauchen mehr Stützstellen der Fourier-Inversion."},
        {"name": "n_samples", "type": "int", "default": 20000, "min": 1000, "max": 200000, "step": 1000,
         "label": "Stichproben"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.levy_flight.LevyStableDistribution",
        "src.physics.levy_flight.LevyStableDistribution.characteristic_function",
        "src.physics.levy_flight.LevyStableDistribution.pdf",
        "src.physics.levy_flight.LevyStableDistribution.sample",
        "src.physics.levy_flight.LevyStableDistribution.tail_probability",
    ],
    "cost": "fast",
}

TAIL_KS = (2, 3, 5, 10, 20, 50, 100)
WALK_STEPS = 1000


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _tiny(p, underflow=False):
    """Probability cell in %: values far below anything observable become text instead of 1e-200 %.

    ``underflow=True``: an exact 0 is a floating-point underflow (Gaussian tail far out), not a true zero.
    """
    if p is None or not np.isfinite(p):
        return None
    if p == 0 and not underflow:
        return 0.0
    if p < 1e-12:
        return "< 10⁻¹⁰ %"
    if 0 < p < 1e-6:
        mant, exp = f"{100 * p:.1e}".split("e")
        return mant.replace(".", ",") + "·10" + str(int(exp)).translate(_SUP) + " %"
    return float(p)


def _ratio(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return None
    if b <= 0:
        return "> 10⁶"
    r = a / b
    return "> 10⁶" if r > 1e6 else float(r)


def run(params, data, out):
    alpha, beta = params["alpha"], params["beta"]
    xr, n, seed = float(params["x_range"]), params["n_samples"], params["seed"]

    dist = LevyStableDistribution(alpha=alpha, beta=beta, gamma=1.0, delta=0.0)
    gauss = LevyStableDistribution(alpha=2.0, beta=0.0, gamma=1.0, delta=0.0)
    g_sd = np.sqrt(2.0)  # S1 with alpha = 2 is N(delta, 2 gamma^2)

    # --- density by Fourier inversion; enough nodes that the grid is not aliased (period 2*pi/dt)
    out.progress("Dichte per Fourier-Inversion …", 0.1)
    x = np.linspace(-xr, xr, 481)
    n_points = int(np.clip(100 * xr, 1000, 8000))
    pdf = np.asarray(dist.pdf(x, n_points=n_points), dtype=float)
    gpdf = stats.norm.pdf(x, 0.0, g_sd)

    # --- Chambers-Mallows-Stuck samples
    out.progress("Stichproben ziehen …", 0.4)
    s = np.asarray(dist.sample(n, seed=seed), dtype=float)
    s = s[np.isfinite(s)]
    s_sorted = np.sort(s)
    m = len(s_sorted)

    def left(d):
        return np.searchsorted(s_sorted, -np.asarray(d), side="left") / m

    def right(d):
        return 1.0 - np.searchsorted(s_sorted, np.asarray(d), side="right") / m

    # --- headline numbers
    k5 = 5.0
    p_left5, g5 = float(left(k5)), float(stats.norm.sf(k5 / g_sd))
    if alpha < 2:
        out.metric("Varianz", "∞", "text", hint="Für jedes α < 2 ist die Varianz unendlich",
                   status="serious" if alpha < 1.5 else "warning",
                   badge="sehr fette Ränder" if alpha < 1.5 else "fette Ränder")
        out.metric("Endliche Momente E|X|^p", f"p < {_de(alpha)}", "text",
                   hint="Mittelwert existiert nur für α > 1" if alpha > 1 else "Auch der Mittelwert existiert nicht")
        out.metric("Dichte-Exponent", 1 + alpha, "num", hint="f(x) ~ |x|^−(1+α) weit draußen")
    else:
        out.metric("Varianz", 2.0, "num", hint="α = 2: Normalverteilung mit σ² = 2γ²")
        out.metric("Endliche Momente E|X|^p", "alle p", "text", hint="Gauß-Fall")
    out.metric("P(X < −5γ)", p_left5, "pct", hint=f"Stichprobe; Gauß: {g5 * 100:.2g} %".replace(".", ","),
               status="critical" if p_left5 > 1000 * g5 else "serious" if p_left5 > 10 * g5 else None)
    out.metric("1 %-Quantil", float(np.quantile(s, 0.01)), "num",
               hint=f"in γ; Gauß: {_de(stats.norm.ppf(0.01) * g_sd)}".replace("-", "−"))
    out.metric("Größter Ausschlag", float(np.max(np.abs(s))), "num",
               hint="in γ, unter " + f"{m:,}".replace(",", ".") + " Ziehungen")

    # --- chart 1: density on a log scale
    edges = np.linspace(-xr, xr, 121)
    counts, _ = np.histogram(s, bins=edges)
    width = edges[1] - edges[0]
    dens = counts / (m * width)  # share of ALL draws, so mass outside the window is not re-normalised away
    centers = 0.5 * (edges[:-1] + edges[1:])
    floor = 1.0 / (m * width)
    positive = pdf[pdf > 0]
    lo = max(1e-8, 0.5 * min(floor, float(positive.min()) if positive.size else floor))
    hi = 2.5 * max(float(pdf.max()), float(dens.max()), float(gpdf.max()))
    clip = lambda y: np.where(y >= lo, y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Dichte auf logarithmischer Skala",
        [C.trace("Stichprobe", centers, np.where(dens > 0, dens, np.nan), type="bar", color=1),
         C.trace(f"Lévy-Dichte (α = {_de(alpha)})", x, clip(pdf), color=2, width=2.5),
         C.trace("Gauß (α = 2, gleiche Skala)", x, clip(gpdf), color="muted", dash=True)],
        x=C.axis("x (in γ)"), y=C.axis("Dichte", type="log", range=(lo, hi)),
        subtitle="Die Gauß-Parabel stürzt ab, die Lévy-Dichte fällt nur wie |x|^−(1+α).", height=380,
    ))

    # --- chart 2: tails on log-log axes
    dmax = float(np.clip(np.max(np.abs(s)), 20.0, 1e6))
    d = np.logspace(np.log10(0.5), np.log10(dmax), 70)
    emp_l, emp_r = left(d), right(d)
    asym_l = np.asarray(dist.tail_probability(-d, "left"), dtype=float)
    asym_r = np.asarray(dist.tail_probability(d, "right"), dtype=float)
    g_tail = stats.norm.sf(d / g_sd)
    y_lo = 0.3 / m
    pos = lambda y: np.where((np.asarray(y) >= y_lo) & (np.asarray(y) < 1), y, np.nan)  # noqa: E731
    out.chart(C.xy(
        "Ränder im Log-Log-Plot",
        [C.trace("links: P(X < −x)", d, pos(emp_l), color=1, width=2.5),
         C.trace("rechts: P(X > x)", d, pos(emp_r), color=3, width=2.5),
         C.trace("Asymptote links", d, pos(asym_l), color=1, dash=True, width=1.5),
         C.trace("Asymptote rechts", d, pos(asym_r), color=3, dash=True, width=1.5),
         C.trace("Gauß", d, pos(g_tail), color="muted", dash=True)],
        x=C.axis("Abstand x vom Zentrum (in γ)", type="log"),
        y=C.axis("Wahrscheinlichkeit", type="log", fmt="pct", range=(y_lo, 1.5)),
        subtitle=f"Durchgezogen: Stichprobe. Gestrichelt: Asymptote C_α(1 ± β)·x^−α – eine Gerade mit Steigung "
                 f"−α = −{_de(alpha)}. Die Gauß-Kurve knickt senkrecht ab.", height=380,
    ))

    # --- chart 3: random walk with the same scale
    steps = min(WALK_STEPS, len(s))
    walk_levy = np.cumsum(s[:steps])
    walk_gauss = np.cumsum(np.asarray(gauss.sample(steps, seed=seed + 1), dtype=float))
    out.chart(C.xy(
        "Lévy-Flug vs. Brown'sche Bewegung",
        [C.trace(f"Lévy-Flug (α = {_de(alpha)})", np.arange(1, steps + 1), walk_levy, color=2, width=1.5),
         C.trace("Brown'sche Bewegung (α = 2)", np.arange(1, steps + 1), walk_gauss, color="muted", width=1.5)],
        x=C.axis("Schritt", fmt="int"), y=C.axis("Position (in γ)"),
        subtitle="Summe der ersten Ziehungen: gleiche Skala, aber der Lévy-Pfad wird von wenigen Sprüngen geprägt.",
    ))

    # --- chart 4: characteristic function (the definition of the law)
    t = np.linspace(-3.0, 3.0, 401)
    phi = dist.characteristic_function(t)
    phi_g = gauss.characteristic_function(t)
    out.chart(C.xy(
        "Charakteristische Funktion φ(t)",
        [C.trace("Re φ(t)", t, np.real(phi), color=2),
         C.trace("Im φ(t)", t, np.imag(phi), color=4),
         C.trace("Gauß: e^(−t²)", t, np.real(phi_g), color="muted", dash=True)],
        x=C.axis("t (in 1/γ)"), y=C.axis("φ(t)", range=(-1.05, 1.05)),
        subtitle="|φ(t)| = e^(−|t|^α): Die Spitze bei t = 0 für α < 2 ist der Fingerabdruck der fetten Ränder. "
                 "Im φ ≠ 0 bedeutet Schiefe.", height=300,
    ))

    # --- table: tail probabilities
    rows = []
    for k in TAIL_KS:
        pl, pr = float(left(k)), float(right(k))
        al = float(dist.tail_probability(-k, "left"))
        ar = float(dist.tail_probability(k, "right"))
        g = float(stats.norm.sf(k / g_sd))
        rows.append([f"{k} γ", pl, _tiny(al), pr, _tiny(ar), _tiny(g, underflow=True), _ratio(pl, g)])
    out.table(
        "Tail-Wahrscheinlichkeiten",
        ["Abstand", "P(X < −x) Stichprobe", "links asymptotisch", "P(X > x) Stichprobe", "rechts asymptotisch",
         "Gauß je Seite", "Faktor links / Gauß"],
        rows, formats=["text", "pct", "pct", "pct", "pct", "pct", "x"],
        note="Asymptotisch = Potenzgesetz aus `tail_probability` (Samorodnitsky-Taqqu 1.2.15), gilt erst für "
             "große x und ist bei 100 % gekappt. 0 % in der Stichprobe heißt: nicht beobachtet.",
    )

    if alpha <= 1:
        out.note("Für α ≤ 1 existiert nicht einmal der Mittelwert: Der Durchschnitt der Stichprobe stabilisiert sich "
                 "nie, egal wie viele Ziehungen du nimmst.", "warning")
    if alpha < 0.8:
        out.note("Für kleine α fällt die charakteristische Funktion so langsam ab, dass die Fourier-Inversion mit "
                 "festem Integrationsbereich die äußeren Ränder der Dichte unterschätzt. Das Histogramm und der "
                 "Log-Log-Plot der Stichprobe zeigen dort die zuverlässigeren Werte.", "info")
    if alpha == 2 and beta != 0:
        out.note("Bei α = 2 ist die Verteilung exakt normal – β hat dann keine Wirkung.", "info")
