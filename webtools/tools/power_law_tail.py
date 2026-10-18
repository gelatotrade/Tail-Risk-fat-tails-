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


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _years(y):
    """Waiting time in years, readable in German even when astronomically large."""
    if y < 1:
        return f"{_de(y * 252, 0)} Handelstage"
    if y < 1e5:
        return f"{y:,.0f} Jahre".replace(",", ".") if y >= 10 else f"{_de(y, 1)} Jahre"
    exp = int(np.floor(np.log10(y)))
    return f"{_de(y / 10 ** exp, 1)}·10{str(exp).translate(_SUP)} Jahre"


def _se(tail_index, n_tail):
    """Standard error of the MLE tail index (Clauset et al. 2009, eq. 3.2)."""
    return tail_index / np.sqrt(n_tail)


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
    se = _se(tail_index, n_tail)
    p_val = float(fit["p_value"])
    out.metric("Tail-Index α − 1", tail_index, "num",
               hint=f"± {_de(se)} (Standardfehler) · P(X > x) ∝ x^−(α−1) · kubisches Gesetz ≈ 3",
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

    # Log-log CCDF: unconditional P(X >= x) for the chosen side, from a tenth of x_min upwards
    xs = np.sort(values[values > 0])[::-1]
    x_lo = max(float(xs[-1]), x_min / 10)
    ccdf = np.arange(1, len(xs) + 1) / n
    m = int(np.sum(xs >= x_lo))
    keep = np.unique(np.round(np.logspace(0, np.log10(m), 700)).astype(int)) - 1
    grid = np.logspace(np.log10(x_lo), np.log10(xs[0] * 1.5), 200)
    floor = 0.3 / n
    fit_line = np.where(grid >= x_min, (n_tail / n) * (grid / x_min) ** (-tail_index), np.nan)
    gauss = _gauss_tail(grid, side, mu, sd)
    gauss = np.where(gauss >= floor, gauss, np.nan)
    out.chart(C.xy(
        f"Wahrscheinlichkeit eines Tages jenseits von x – {SIDES[side]}",
        [C.trace("Beobachtet", xs[keep], ccdf[keep], type="scatter", color=1, size=5, decimate_to=None),
         C.trace(f"Potenzgesetz (α − 1 = {_de(tail_index)})", grid, fit_line, color=2, width=2.5),
         C.trace("Normalverteilung", grid, gauss, color="muted", dash=True),
         C.trace(f"x_min = {_de(100 * x_min)} %", [x_min, x_min], [floor, 1.0], color=3, dash=True, width=1.5,
                 decimate_to=None)],
        x=C.axis("Tagesbewegung x", type="log", fmt="pct", digits=1, range=(x_lo, xs[0] * 1.6)),
        y=C.axis("P(X ≥ x)", type="log", range=(floor, 1.2)), height=420,
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
    out.table("Vergleich der Ränder",
              ["Rand", "α", "α − 1", "± Std.-Fehler", "x_min", "Tage im Rand", "p-Wert", "Existierende Momente"],
              [[SIDES[k], fits[k][1]["alpha"], fits[k][1]["alpha"] - 1,
                _se(fits[k][1]["alpha"] - 1, fits[k][1]["n_tail"]), fits[k][1]["x_min"], fits[k][1]["n_tail"],
                fits[k][1]["p_value"], _existing(fits[k][1]["alpha"] - 1)] for k in order],
              formats=["text", "num", "num", "num", "pct", "int", "num", "text"],
              note="Methode: " + ("Clauset (KS-optimales x_min)" if method == "clauset" else "feste Schwelle (größte 10 %)")
                   + ". Unterscheiden sich zwei Ränder um weniger als etwa zwei Standardfehler, ist der Unterschied "
                     "nicht belastbar.")

    sample = [_de(100 * mu, 3) + " %", _de(100 * sd, 2) + " % (Volatilität)",
              _de(stats.skew(r)), _de(stats.kurtosis(r), 1) + " (Exzess)"]
    out.table("Welche Momente existieren?", ["Moment", "Bedingung", "existiert?", "Stichprobe (Tagesrenditen)"],
              [[name, f"α − 1 > {k}", "ja" if tail_index > k else "nein – unendlich", s]
               for (k, name), s in zip(MOMENTS, sample)],
              formats=["text", "text", "text", "text"],
              note="Auch ein unendliches Moment liefert in jeder Stichprobe eine endliche Zahl – sie ist aber "
                   "instabil und wächst mit jedem neuen Extremtag.")

    rows = []
    for lvl in LEVELS:
        if lvl > xs[0] * 2:
            break
        pl = (n_tail / n) * (lvl / x_min) ** (-tail_index) if lvl >= x_min else None
        g = float(_gauss_tail(lvl, side, mu, sd))
        rows.append([lvl, int(np.sum(values >= lvl)), pl, _years(1 / (pl * 252)) if pl else "–",
                     _years(1 / (g * 252)) if g > 0 else "nie (numerisch 0)"])
    out.table("Wie oft kommt ein Tag jenseits von x vor?",
              ["Schwelle x", "Tage beobachtet", "Potenzgesetz: Häufigkeit je Tag", "Potenzgesetz: einmal alle …",
               "Normalverteilung: einmal alle …"], rows,
              formats=["pct", "int", "pct", "text", "text"],
              note=f"Beobachtet in {n} Handelstagen ({_de(n / 252, 1)} Jahre). Das Potenzgesetz gilt erst ab x_min, "
                   "darunter bleibt es leer.")

    if n_tail < 50:
        out.note(f"Nur {n_tail} Tage im Rand: Der Tail-Index ist sehr unsicher (± {_de(se)}, grob 95 %: "
                 f"{_de(tail_index - 2 * se)} bis {_de(tail_index + 2 * se)}). Für belastbare Aussagen brauchst du "
                 "eine längere Reihe.", "warning")
    if p_val < 0.1:
        out.note(f"Der Bootstrap lehnt ein reines Potenzgesetz ab (p = {_de(p_val)}). Der Exponent ist dann nur "
                 "eine Näherung für die Steigung im Rand.", "warning")
    if tail_index < 2:
        out.set_verdict("critical", "Unendliche Varianz im Rand",
                        f"α − 1 = {_de(tail_index)} < 2: Volatilität und varianzbasierte Risikomaße sind hier "
                        "nicht stabil schätzbar.")
    elif tail_index < 4:
        out.set_verdict("serious" if tail_index < 3 else "warning", "Fetter Rand: Varianz endlich, Kurtosis nicht",
                        f"α − 1 = {_de(tail_index)}. Existierende Momente: {_existing(tail_index)}. Die "
                        "Stichproben-Kurtosis ist deshalb kein verlässliches Maß.")
    else:
        out.set_verdict("good", "Dünner Rand: alle vier Momente endlich",
                        f"α − 1 = {_de(tail_index)} > 4 – die Ränder fallen schnell genug ab.")
