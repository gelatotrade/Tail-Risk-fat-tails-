"""Power-law tail: Clauset-style x_min / exponent fit, log-log CCDF and which moments exist."""

import numpy as np
from scipy import stats

from src.physics.phase_transitions import PowerLawAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "power-law-tail",
    "title": "Potenzgesetz im Rand: Tail-Exponent",
    "group": "data",
    "order": 40,
    "summary": "Wie schnell fällt die Wahrscheinlichkeit großer Tage ab? Exponent, Beginn des Rands und welche "
               "Momente (Varianz, Kurtosis) überhaupt existieren.",
    "description": (
        "Oberhalb einer Schwelle x_min folgen große Tagesbewegungen oft einem **Potenzgesetz**: Die Dichte fällt "
        "wie x^−α, die Wahrscheinlichkeit eines Tages jenseits von x wie x^−(α−1). Der **Tail-Index α − 1** ist "
        "die entscheidende Zahl: Ein Moment k (Varianz k = 2, Schiefe 3, Kurtosis 4) existiert nur, wenn α − 1 > k. "
        "Für Aktien liegt α − 1 typischerweise um 3 („kubisches Gesetz“) – die Varianz existiert, die Kurtosis nicht.\n\n"
        "So liest du das Hauptdiagramm: Doppelt-logarithmisch wird ein Potenzgesetz zur **Geraden**. Die Punkte "
        "zeigen, wie oft ein Tag mindestens so groß war; die Gerade ist der Fit ab x_min, die gestrichelte Kurve "
        "die Normalverteilung mit derselben Volatilität. Wo die Punkte über der Gauß-Kurve liegen, unterschätzt "
        "die Normalverteilung das Risiko – oft um Größenordnungen.\n\n"
        "Methode: **Clauset** wählt x_min so, dass der Abstand (KS-Distanz) zwischen Daten und Fit am kleinsten "
        "ist, und schätzt α per Maximum-Likelihood. **Feste Schwelle** nimmt einfach die größten 10 %. Der "
        "Bootstrap-p-Wert prüft, ob ein echtes Potenzgesetz so gut passen würde (vereinfacht: x_min wird je "
        "Stichprobe nicht neu optimiert, der p-Wert fällt deshalb eher zu hoch aus)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "tail", "type": "select", "default": "both", "label": "Rand",
         "options": [{"value": "both", "label": "|Rendite| – beide Ränder"},
                     {"value": "left", "label": "Verluste (linker Rand)"},
                     {"value": "right", "label": "Gewinne (rechter Rand)"}]},
        {"name": "method", "type": "select", "default": "clauset", "label": "Schwelle x_min",
         "options": [{"value": "clauset", "label": "Clauset (KS-optimal)"},
                     {"value": "fixed", "label": "Fest: größte 10 %"}]},
        {"name": "n_bootstrap", "type": "int", "default": 100, "min": 20, "max": 1000, "step": 20,
         "label": "Bootstrap-Stichproben", "help": "für den p-Wert des Potenzgesetz-Tests"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed",
         "help": "nur für den Bootstrap"},
    ],
    "covers": [
        "src.physics.phase_transitions.PowerLawAnalyzer",
        "src.physics.phase_transitions.PowerLawAnalyzer.fit_power_law",
        "src.physics.phase_transitions.PowerLawAnalyzer._bootstrap_pvalue",
    ],
    "cost": "fast",
}

SIDES = {"left": "Verluste", "right": "Gewinne", "both": "|Rendite|"}
MOMENTS = [(1, "Mittelwert"), (2, "Varianz"), (3, "Schiefe"), (4, "Kurtosis")]
LEVELS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _side_values(r, side):
    if side == "left":
        return -r[r < 0]
    if side == "right":
        return r[r > 0]
    return np.abs(r)


def _gauss_tail(x, side, mu, sd):
    if side == "left":
        return stats.norm.cdf(-x, mu, sd)
    if side == "right":
        return stats.norm.sf(x, mu, sd)
    return stats.norm.cdf(-x, mu, sd) + stats.norm.sf(x, mu, sd)


def _existing(tail_index):
    names = [name for k, name in MOMENTS if tail_index > k]
    return ", ".join(names) if names else "keines"


def run(params, data, out):
    r = data.returns
    n = data.n
    mu, sd = float(np.mean(r)), float(np.std(r))
    side, method = params["tail"], params["method"]

    fits = {}
    for key in ("left", "right", "both"):
        values = _side_values(r, key)
        try:
            fits[key] = (values, PowerLawAnalyzer().fit_power_law(
                values, x_min_method=method, n_bootstrap=params["n_bootstrap"], seed=params["seed"]))
        except ValueError:
            fits[key] = (values, None)
    values, fit = fits[side]
    if fit is None:
        raise ToolError(f"Für den Rand „{SIDES[side]}“ gibt es zu wenige von null verschiedene Werte "
                        "(mindestens 10 nötig).")

    alpha, x_min, n_tail = float(fit["alpha"]), float(fit["x_min"]), int(fit["n_tail"])
    tail_index = alpha - 1.0
    p_val = float(fit["p_value"])
    out.metric("Tail-Index α − 1", tail_index, "num",
               hint="P(X > x) ∝ x^−(α−1) · kubisches Gesetz ≈ 3",
               status="critical" if tail_index < 2 else "serious" if tail_index < 3 else
               "warning" if tail_index < 4 else None)
    out.metric("Dichte-Exponent α", alpha, "num", hint="Dichte p(x) ∝ x^−α")
    out.metric("Beginn des Rands x_min", x_min, "pct", hint="ab hier gilt das Potenzgesetz")
    out.metric("Tage im Rand", n_tail, "int", hint=f"{_de(100 * n_tail / n, 1)} % aller Tage")
    if fit["ks_statistic"] is not None:
        out.metric("KS-Distanz", fit["ks_statistic"], "num", digits=3, hint="Abstand Daten ↔ Fit (kleiner = besser)")
    out.metric("Bootstrap-p-Wert", p_val, "num", hint="> 0,1: Potenzgesetz plausibel",
               status="warning" if p_val < 0.1 else None)
    out.metric("Existierende Momente", _existing(tail_index), "text", hint="Moment k existiert, wenn α − 1 > k")

    # Log-log CCDF: unconditional P(X >= x) for the chosen side
    xs = np.sort(values[values > 0])[::-1]
    ranks = np.arange(1, len(xs) + 1)
    keep = np.unique(np.round(np.logspace(0, np.log10(len(xs)), 700)).astype(int)) - 1
    ccdf = ranks / n
    grid = np.logspace(np.log10(xs[-1]), np.log10(xs[0] * 1.5), 200)
    floor = 0.3 / n
    fit_line = np.where(grid >= x_min, (n_tail / n) * (grid / x_min) ** (-tail_index), np.nan)
    gauss = _gauss_tail(grid, side, mu, sd)
    gauss = np.where(gauss >= floor, gauss, np.nan)
    out.chart(C.xy(
        f"Wahrscheinlichkeit eines Tages jenseits von x – {SIDES[side]}",
        [C.trace("Beobachtet", xs[keep], ccdf[keep], type="scatter", color=1, size=5, decimate_to=None),
         C.trace(f"Potenzgesetz (α − 1 = {_de(tail_index)})", grid, fit_line, color=2, width=2.5),
         C.trace("Normalverteilung", grid, gauss, color="muted", dash=True)],
        x=C.axis("Tagesbewegung x", type="log", fmt="pct", digits=1),
        y=C.axis("P(X ≥ x)", type="log", range=(floor, 1.2)),
        vlines=[C.vline(x_min, "x_min")], height=420,
        subtitle="Doppelt-logarithmisch wird ein Potenzgesetz zur Geraden. Die Gauß-Kurve stürzt dagegen ab.",
    ))

    # Comparison of the two tails (same method)
    order = [k for k in ("left", "right", "both") if fits[k][1] is not None]
    out.chart(C.xy(
        "Tail-Index je Rand",
        [C.trace("α − 1", [SIDES[k] for k in order], [fits[k][1]["alpha"] - 1 for k in order], type="bar", color=1)],
        x=C.axis("Rand", type="category"), y=C.axis("Tail-Index α − 1"),
        hlines=[C.hline(2.0, "darüber: Varianz endlich"), C.hline(4.0, "darüber: Kurtosis endlich")],
        height=300, subtitle="Ein kleinerer Index heißt: dieser Rand ist fetter. Bei Aktien meist die Verlustseite.",
    ))
    out.table("Vergleich der Ränder", ["Rand", "α", "α − 1", "x_min", "Tage im Rand", "p-Wert", "Existierende Momente"],
              [[SIDES[k], fits[k][1]["alpha"], fits[k][1]["alpha"] - 1, fits[k][1]["x_min"], fits[k][1]["n_tail"],
                fits[k][1]["p_value"], _existing(fits[k][1]["alpha"] - 1)] for k in order],
              formats=["text", "num", "num", "pct", "int", "num", "text"],
              note="Methode: " + ("Clauset (KS-optimales x_min)" if method == "clauset" else "feste Schwelle (größte 10 %)"))

    sample = [_de(100 * mu, 3) + " %", _de(100 * sd, 2) + " % (Volatilität)",
              _de(stats.skew(r)), _de(stats.kurtosis(r), 1) + " (Exzess)"]
    out.table("Welche Momente existieren?", ["Moment", "Bedingung", "existiert?", "Stichprobenwert"],
              [[name, f"α − 1 > {k}", "ja" if tail_index > k else "nein – unendlich", s]
               for (k, name), s in zip(MOMENTS, sample)],
              formats=["text", "text", "text", "text"],
              note="Auch ein unendliches Moment liefert in jeder Stichprobe eine endliche Zahl – sie ist aber "
                   "instabil und wächst mit jedem neuen Extremtag.")

    rows = []
    for lvl in LEVELS:
        if lvl > xs[0] * 2:
            break
        emp = float(np.sum(values >= lvl)) / n
        pl = (n_tail / n) * (lvl / x_min) ** (-tail_index) if lvl >= x_min else None
        g = float(_gauss_tail(lvl, side, mu, sd))
        rows.append([lvl, int(np.sum(values >= lvl)), emp, pl, g,
                     1 / (pl * 252) if pl else None, 1 / (g * 252) if g > 0 else None])
    out.table("Wie oft kommt ein Tag jenseits von x vor?",
              ["Schwelle x", "Tage beobachtet", "Häufigkeit beobachtet", "Potenzgesetz", "Normalverteilung",
               "Potenzgesetz: einmal alle … Jahre", "Gauß: einmal alle … Jahre"], rows,
              formats=["pct", "int", "pct", "pct", "pct", "num", "num"],
              note="Häufigkeiten je Handelstag. Das Potenzgesetz gilt erst ab x_min, darunter bleibt es leer.")

    if p_val < 0.1:
        out.note(f"Der Bootstrap lehnt ein reines Potenzgesetz ab (p = {_de(p_val)}). Der Exponent ist dann nur "
                 "eine Näherung für die Steigung im Rand.", "warning")
    if tail_index < 2:
        out.set_verdict("critical", "Unendliche Varianz im Rand",
                        f"α − 1 = {_de(tail_index)} < 2: Volatilität und varianzbasierte Risikomaße sind hier "
                        "nicht stabil schätzbar.")
    elif tail_index < 4:
        out.set_verdict("serious", "Fetter Rand: Varianz endlich, Kurtosis nicht",
                        f"α − 1 = {_de(tail_index)}. Existierende Momente: {_existing(tail_index)}. Die "
                        "Stichproben-Kurtosis ist deshalb kein verlässliches Maß.")
    else:
        out.set_verdict("good", "Dünner Rand: alle vier Momente endlich",
                        f"α − 1 = {_de(tail_index)} > 4 – die Ränder fallen schnell genug ab.")
