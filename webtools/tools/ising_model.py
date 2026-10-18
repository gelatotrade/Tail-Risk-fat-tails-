"""Ising-Marktmodell: Herdentrieb, Modellkurs und Ordnungs-Unordnungs-Übergang."""

import numpy as np
from scipy import stats

from src.physics.phase_transitions import IsingMarketModel

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "ising-model",
    "title": "Ising-Modell: Herdentrieb der Händler",
    "group": "regime",
    "order": 60,
    "summary": "Wie aus vielen Kauf- und Verkaufsentscheidungen mit Herdentrieb Trends, Kursschübe und ein "
               "Übergang von Ordnung zu Unordnung entstehen.",
    "description": (
        "Jeder Händler ist ein Spin: **+1 kauft, −1 verkauft**. Die Kopplung **J** belohnt es, dasselbe zu tun "
        "wie die beiden Nachbarn (Herdentrieb), das Feld **h** ist eine marktweite Stimmung, die Temperatur **T** "
        "ist Rauschen – private Information und Zufall. Der Metropolis-Algorithmus lässt die Händler nacheinander "
        "ihre Meinung ändern. Die Magnetisierung **M** (Anteil Käufer minus Anteil Verkäufer) treibt den "
        "Modellkurs: Je Aufzeichnung ändert er sich um M × 1 %.\n\n"
        "So liest du die Diagramme: Im Raum-Zeit-Bild sind Herden einfarbige Streifen. Je kleiner T im Verhältnis "
        "zu J, desto größer die Herden (Korrelationslänge ξ) und desto länger hält ein Trend in M und im Kurs. "
        "Das Phasendiagramm zeigt ⟨|M|⟩ und die Suszeptibilität χ über der Temperatur – als Simulation "
        "(`phase_diagram`) und als exakte Lösung der Kette (Transfermatrix).\n\n"
        "Ehrlich gesagt: Die Händler sitzen auf einem **eindimensionalen** Ring. Die eingezeichnete kritische "
        "Temperatur T_c = 2J/ln(1+√2) ≈ 2,27·J ist der Onsager-Wert des **2D**-Gitters und dient nur als "
        "Referenzskala. Eine 1D-Kette hat bei T > 0 keinen echten Phasenübergang, sondern einen fließenden "
        "Übergang von Ordnung zu Unordnung. Bei sehr kleinem T frieren die Herden in der Simulation außerdem ein "
        "(100 Sweeps zum Einschwingen reichen nicht) – dann liegt sie unter der exakten Kurve."
    ),
    "needs": [],
    "params": [
        {"name": "n_agents", "type": "int", "default": 80, "min": 20, "max": 400, "step": 10,
         "label": "Anzahl Händler N"},
        {"name": "J", "type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
         "label": "Herdentrieb J", "help": "Kopplung an die beiden Nachbarn. 0 = jeder entscheidet unabhängig."},
        {"name": "h", "type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
         "label": "Marktstimmung h", "help": "> 0 Kaufdruck, < 0 Verkaufsdruck für alle Händler."},
        {"name": "T", "type": "float", "default": 0.8, "min": 0.1, "max": 6.0, "step": 0.05,
         "label": "Temperatur T (Rauschen)", "help": "Klein = Händler folgen den Nachbarn, groß = Zufall dominiert."},
        {"name": "n_steps", "type": "int", "default": 400, "min": 50, "max": 4000, "step": 50,
         "label": "Monte-Carlo-Sweeps", "help": "Ein Sweep = N einzelne Entscheidungen."},
        {"name": "record_interval", "type": "int", "default": 4, "min": 1, "max": 50, "step": 1,
         "label": "Aufzeichnung alle … Sweeps", "help": "Jede Aufzeichnung ist ein Handelstag des Modellkurses."},
        {"name": "phase", "type": "bool", "default": True, "label": "Phasendiagramm berechnen",
         "help": "Kostet die meiste Rechenzeit: je Temperatur 100 Sweeps Einschwingen plus die Messungen."},
        {"name": "T_min", "type": "float", "default": 0.8, "min": 0.1, "max": 5.0, "step": 0.1,
         "label": "Phasendiagramm: T von"},
        {"name": "T_max", "type": "float", "default": 4.0, "min": 0.5, "max": 8.0, "step": 0.1,
         "label": "Phasendiagramm: T bis"},
        {"name": "n_T", "type": "int", "default": 8, "min": 3, "max": 25, "step": 1,
         "label": "Phasendiagramm: Temperaturen"},
        {"name": "n_samples", "type": "int", "default": 50, "min": 10, "max": 400, "step": 10,
         "label": "Messungen je Temperatur", "help": "Sweeps, über die ⟨|M|⟩ und χ gemittelt werden."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.phase_transitions.IsingMarketModel",
        "src.physics.phase_transitions.IsingMarketModel.magnetization",
        "src.physics.phase_transitions.IsingMarketModel.energy",
        "src.physics.phase_transitions.IsingMarketModel.susceptibility",
        "src.physics.phase_transitions.IsingMarketModel.monte_carlo_step",
        "src.physics.phase_transitions.IsingMarketModel.simulate",
        "src.physics.phase_transitions.IsingMarketModel.phase_diagram",
    ],
    "cost": "medium",
}

N_VIEW = 60               # sweeps shown in the space-time picture
N_ROWS = 50               # agents shown in the space-time picture (bigger cells)
EQUILIBRATION = 100       # sweeps phase_diagram spends per temperature before measuring
MAX_FLIPS = 1_500_000     # spin-flip attempts (~15 s native, pure-Python Metropolis loop)


def _exact_ring(n, J, h, T):
    """Exact Boltzmann distribution of the number of buyers on a periodic 1D ring.

    Tr T(z)^n of the transfer matrix with a counting variable z is a polynomial whose coefficients are the
    weights of 0..n buyers; it is evaluated on the unit circle and inverted with an FFT.
    Returns (<|M|>, chi) with chi = N (<M^2> - <|M|>^2) / T, the same estimator phase_diagram uses.
    """
    b = 1.0 / T
    u = np.arange(n + 1)
    w = np.exp(1j * np.pi * u / (n + 1))  # w^2 = z_k; each spin's z-weight is split over its two bonds
    tm = np.empty((n + 1, 2, 2), dtype=complex)
    tm[:, 0, 0] = np.exp(b * J + b * h) * w * w
    tm[:, 0, 1] = tm[:, 1, 0] = np.exp(-b * J) * w
    tm[:, 1, 1] = np.exp(b * J - b * h)
    tm /= np.max(np.abs(np.linalg.eigvals(tm[0].real)))
    trace = np.sum(np.linalg.eigvals(tm) ** n, axis=1)
    p = np.clip(np.fft.fft(trace).real, 0.0, None)
    p /= p.sum()
    m = np.abs(2 * u - n) / n
    mean_abs = float(np.sum(p * m))
    return mean_abs, n * (float(np.sum(p * m * m)) - mean_abs ** 2) / T


def _de(x, digits=2):
    """German decimal comma for numbers inside text."""
    return f"{x:.{digits}f}".replace(".", ",")


def _correlation_length(J, h, T):
    """xi = 1 / ln(lambda+ / lambda-) of the 1D transfer matrix (in agents)."""
    b = 1.0 / T
    root = np.sqrt(np.exp(2 * b * J) * np.sinh(b * h) ** 2 + np.exp(-2 * b * J))
    lp, lm = np.exp(b * J) * np.cosh(b * h) + root, np.exp(b * J) * np.cosh(b * h) - root
    ratio = lp / lm if lm > 0 else np.inf
    return 1.0 / np.log(ratio) if ratio > 1 else np.inf


def run(params, data, out):
    n, J, h, T = params["n_agents"], params["J"], params["h"], params["T"]
    n_steps, every = params["n_steps"], params["record_interval"]
    t_lo, t_hi, n_T = params["T_min"], params["T_max"], params["n_T"]
    if params["phase"] and t_lo >= t_hi:
        raise ToolError("Für das Phasendiagramm muss „T von“ kleiner sein als „T bis“.")
    flips = n * (n_steps + N_VIEW + (n_T * (EQUILIBRATION + params["n_samples"]) if params["phase"] else 0))
    if flips > MAX_FLIPS:
        raise ToolError(f"Das wären {flips / 1e6:.1f} Mio. Einzelentscheidungen – zu viel für den Browser "
                        f"(Grenze {MAX_FLIPS / 1e6:.1f} Mio.). Verringere Händler, Sweeps, Temperaturen oder "
                        "Messungen, oder schalte das Phasendiagramm ab.")

    np.random.seed(params["seed"])
    model = IsingMarketModel(n_agents=n, J=J, h=h, T=T)
    out.progress("Metropolis-Simulation läuft …", 0.05)
    sim = model.simulate(n_steps=n_steps, record_interval=every)
    M, E, ret, price = sim["magnetization"], sim["energy"], sim["returns"], sim["prices"]
    sweeps = np.arange(0, n_steps, every) + 1  # the record is taken after this many sweeps

    # Space-time picture: continue the same chain and keep every configuration
    history = np.empty((N_VIEW, n))
    for k in range(N_VIEW):
        model.monte_carlo_step()
        history[k] = model.spins
    m_now, e_now, chi_now = model.magnetization(), model.energy(), model.susceptibility()

    xi = _correlation_length(J, h, T)
    exact_abs, exact_chi = _exact_ring(n, J, h, T)
    tc = float(model.T_c)
    out.metric("Magnetisierung M (Ende)", m_now, "num", hint="+1: alle kaufen · −1: alle verkaufen")
    out.metric("Mittlere |M| im Lauf", np.mean(np.abs(M)), "num",
               hint=f"exakte Lösung der Kette im Gleichgewicht: {_de(exact_abs)}")
    out.metric("Korrelationslänge ξ", xi, "num", digits=1, unit="Händler",
               hint="typische Herdengröße (exakt für die 1D-Kette)")
    out.metric("Energie je Händler", e_now / n, "num",
               hint=f"{_de(-(J + abs(h)))}: völlige Ordnung · ≈ 0: Chaos")
    out.metric("Suszeptibilität χ ohne Herdentrieb", chi_now, "num",
               hint=f"(1−M²)/T für unabhängige Händler · mit Herdentrieb (exakt): {_de(exact_chi)}")
    out.metric("T / T_c (2D-Referenz)", T / tc if tc > 0 else None, "num",
               hint="nur Skala: die 1D-Kette hat keinen echten Phasenübergang")
    out.metric("Schwankung des Modellkurses", np.std(ret), "pct",
               hint="Standardabweichung der Log-Rendite je Aufzeichnung")
    out.metric("Kursänderung gesamt", price[-1] / 100.0 - 1.0, "pct", digits=1)
    if len(ret) > 3 and np.std(ret) > 0:
        out.metric("Exzess-Kurtosis der Modellrenditen", stats.kurtosis(ret), "num", digits=1,
                   hint="0 = normalverteilt")

    out.chart(C.xy(
        "Magnetisierung M – Nettokaufdruck",
        [C.trace("M", sweeps, M, color=1, width=1.5)],
        x=C.axis("Monte-Carlo-Sweep", fmt="int"), y=C.axis("M", range=(-1.05, 1.05)),
        hlines=[C.hline(0.0, "ausgeglichen")],
        subtitle="Lange Ausflüge weg von 0 sind Herdenphasen: viele Händler handeln gleich.",
    ))
    out.chart(C.xy(
        "Modellkurs aus der Magnetisierung",
        [C.trace("Kurs", sweeps, price, color=2)],
        x=C.axis("Monte-Carlo-Sweep", fmt="int"), y=C.axis("Kurs (Start 100)"),
        hlines=[C.hline(100.0, "Start")],
        subtitle="Je Aufzeichnung ändert sich der Kurs um M × 1 % – Herden erzeugen Trends.",
    ))
    rows = min(n, N_ROWS)
    out.chart(C.heatmap(
        f"Raum-Zeit-Bild: die nächsten {N_VIEW} Sweeps" + (f" (Händler 1–{rows} von {n})" if rows < n else ""),
        list(range(n_steps + 1, n_steps + N_VIEW + 1)), list(range(1, rows + 1)), history[:, :rows].T,
        x=C.axis("Monte-Carlo-Sweep", fmt="int"), y=C.axis("Händler", fmt="int"),
        z=C.axis("Position (+1 kauft, −1 verkauft)"), scale="diverging", height=400,
        subtitle="Jede Zeile ist ein Händler: rot kauft, blau verkauft. Gleichfarbige Blöcke sind Herden – "
                 "ihre Höhe ≈ ξ, ihre Lebensdauer wächst mit ξ².",
    ))

    if not params["phase"]:
        out.note("Phasendiagramm ausgeschaltet – schalte es ein, um ⟨|M|⟩ und χ über der Temperatur zu sehen.")
        return

    out.progress("Phasendiagramm: Temperaturen werden durchlaufen …", 0.4)
    temps = np.linspace(t_lo, t_hi, n_T)
    pd = model.phase_diagram(temps, n_samples=params["n_samples"])
    grid = np.linspace(t_lo, t_hi, 60)
    exact = np.array([_exact_ring(n, J, h, t) for t in grid])
    exact_at = np.array([_exact_ring(n, J, h, t) for t in pd["temperature"]])

    def markers(top):
        """Vertical reference lines as legend entries (your T, the 2D reference T_c)."""
        out_ = []
        if t_lo <= T <= t_hi:
            out_.append(C.trace("dein T", [T, T], [0, top], color=2, dash=True, width=1.5, decimate_to=None))
        if t_lo <= tc <= t_hi:
            out_.append(C.trace("T_c (2D-Referenz)", [tc, tc], [0, top], color=3, dash=True, width=1.5,
                                decimate_to=None))
        return out_

    out.chart(C.xy(
        "Phasendiagramm: Ordnung ⟨|M|⟩ über der Temperatur",
        [C.trace("Exakt (1D-Ring)", grid, exact[:, 0], color="muted", width=2.5),
         C.trace("Simulation", pd["temperature"], pd["magnetization"], type="scatter", color=1, size=9)]
        + markers(1.0),
        x=C.axis("Temperatur T"), y=C.axis("⟨|M|⟩", range=(0, 1.02)),
        subtitle="Fließender Übergang statt Sprung bei T_c. Liegt die Simulation darunter, sind die Herden "
                 "eingefroren.",
    ))
    chi_top = float(np.nanmax(np.concatenate([exact[:, 1], pd["susceptibility"]]))) * 1.05
    out.chart(C.xy(
        "Suszeptibilität χ über der Temperatur",
        [C.trace("Exakt (1D-Ring)", grid, exact[:, 1], color="muted", width=2.5),
         C.trace("Simulation", pd["temperature"], pd["susceptibility"], type="scatter", color=1, size=9)]
        + markers(chi_top),
        x=C.axis("Temperatur T"), y=C.axis("χ = N·Var(|M|)/T", range=(0, chi_top)),
        subtitle="Wo χ groß ist, kippt die Stimmung schon bei kleinen Anstößen. Das Maximum ist ein "
                 "Finite-Size-Effekt und wandert mit wachsendem N zu T → 0.",
    ))
    rows = [[float(t), float(t / tc) if tc > 0 else None, float(m), float(ex[0]), float(c), float(ex[1])]
            for t, m, c, ex in zip(pd["temperature"], pd["magnetization"], pd["susceptibility"], exact_at)]
    out.table("Phasendiagramm", ["T", "T / T_c (2D)", "⟨|M|⟩ Simulation", "⟨|M|⟩ exakt", "χ Simulation", "χ exakt"],
              rows, formats=["num", "num", "num", "num", "num", "num"],
              note=f"Je Temperatur {EQUILIBRATION} Sweeps Einschwingen ab Zufallsstart, dann "
                   f"{params['n_samples']} Messungen. Exakt = Boltzmann-Gleichgewicht des Rings mit {n} Händlern.")
    worst = int(np.argmax(exact_at[:, 0] - pd["magnetization"]))
    gap = exact_at[worst, 0] - pd["magnetization"][worst]
    if gap > 0.15:
        out.note(f"Bei T = {_de(pd['temperature'][worst])} liegt die Simulation um {_de(gap)} unter der exakten "
                 f"Lösung: Die Herden wachsen dort so langsam, dass {EQUILIBRATION} Sweeps zum Einschwingen nicht "
                 "reichen.", "warning")
