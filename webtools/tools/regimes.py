"""Volatility regimes: Markov switching, realised-vol regimes and regime-conditional tail risk."""

import numpy as np

from src.models.regime_detection import (
    MarkovRegimeSwitching,
    RegimeAwareTailRisk,
    VolatilityClusteringDetector,
    compute_regime_coordinates,
)
from src.models.risk_metrics import TailRiskMetrics

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

LEVELS = [{"value": v, "label": l} for v, l in ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"))]
STATE_NAMES = {2: ["ruhig", "Stress"], 3: ["ruhig", "erhöht", "Stress"]}
STATE_COLORS = {2: ["good", "critical"], 3: ["good", "warning", "critical"]}
VOL_NAMES = {"LOW_VOL": "niedrig", "MEDIUM_VOL": "mittel", "HIGH_VOL": "hoch"}
DAYS = 252

SPEC = {
    "id": "regimes",
    "title": "Volatilitäts-Regime",
    "group": "regime",
    "order": 10,
    "summary": "Erkennt ruhige und gestresste Marktphasen mit einem Markov-Regimewechselmodell und zeigt, wie "
               "stark VaR und Expected Shortfall vom Regime abhängen.",
    "description": (
        "Das **Markov-Regimewechselmodell** (`MarkovRegimeSwitching`) nimmt an, dass der Markt zwischen wenigen "
        "Zuständen springt, jeder mit eigener Volatilität. Welcher Zustand gerade herrscht, ist unbeobachtet; der "
        "**Hamilton-Filter** schätzt die Wahrscheinlichkeit jeden Tag nur aus den Daten bis zu diesem Tag, der "
        "EM-Algorithmus lernt Mittelwerte, Volatilitäten und die **Übergangsmatrix**. Ihre Diagonale ist die "
        "Beharrung: p = 0,98 heißt, ein Regime hält im Mittel 1/(1 − p) = 50 Tage.\n\n"
        "Daneben steht eine einfache Klassifikation über die realisierte 20-Tage-Volatilität "
        "(`detect_vol_regime`): oberhalb des gewählten Perzentils hoch, unterhalb des Gegenperzentils niedrig. "
        "`RegimeAwareTailRisk` rechnet damit den VaR je Regime – im Stress ist er oft doppelt so hoch wie im "
        "Durchschnitt. Der **Ljung-Box-Test** auf quadrierte Renditen prüft, ob Volatilität überhaupt clustert.\n\n"
        "So liest du die Diagramme: Die Kursfarbe zeigt das wahrscheinlichste Markov-Regime, darunter die "
        "Stress-Wahrscheinlichkeit. Der **3D-Regimeraum** (`compute_regime_coordinates`) zeigt Volatilitätsniveau, "
        "Beharrung und Wechselneigung; hohe Volatilität mit geringer Beharrung ist die instabilste Lage."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "n_states", "type": "select", "default": 3, "label": "Markov-Zustände",
         "options": [{"value": 2, "label": "2 (ruhig / Stress)"}, {"value": 3, "label": "3 (ruhig / erhöht / Stress)"}]},
        {"name": "confidence", "type": "select", "default": 0.99, "label": "VaR-Konfidenz", "options": LEVELS},
        {"name": "vol_percentile", "type": "int", "default": 75, "min": 55, "max": 95, "step": 5,
         "label": "Perzentil für hohe Volatilität", "help": "75: die volatilsten 25 % der Tage gelten als „hoch“."},
        {"name": "forecast_days", "type": "int", "default": 20, "min": 1, "max": 120, "step": 1,
         "label": "Prognosehorizont (Tage)"},
        {"name": "coord_window", "type": "int", "default": 30, "min": 8, "max": 120, "step": 2,
         "label": "Fenster für den 3D-Regimeraum"},
        {"name": "max_years", "type": "int", "default": 15, "min": 2, "max": 60, "step": 1,
         "label": "Zeitraum (letzte Jahre)", "help": "Lange Reihen werden auf die letzten Jahre gekürzt (Rechenzeit)."},
    ],
    "covers": [
        "src.models.regime_detection.MarkovRegimeSwitching",
        "src.models.regime_detection.MarkovRegimeSwitching.fit",
        "src.models.regime_detection.MarkovRegimeSwitching.current_regime",
        "src.models.regime_detection.MarkovRegimeSwitching.regime_probability",
        "src.models.regime_detection.MarkovRegimeSwitching.forecast_regime",
        "src.models.regime_detection.VolatilityClusteringDetector",
        "src.models.regime_detection.VolatilityClusteringDetector.ljung_box_test",
        "src.models.regime_detection.VolatilityClusteringDetector.rolling_realized_volatility",
        "src.models.regime_detection.VolatilityClusteringDetector.detect_vol_regime",
        "src.models.regime_detection.RegimeAwareTailRisk",
        "src.models.regime_detection.RegimeAwareTailRisk.current_regime",
        "src.models.regime_detection.RegimeAwareTailRisk.regime_conditional_var",
        "src.models.regime_detection.RegimeAwareTailRisk.regime_transition_risk",
        "src.models.regime_detection.compute_regime_coordinates",
    ],
    "cost": "medium",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _fmt_date(s):
    if isinstance(s, str) and len(s) == 10 and s[4] == "-":
        return f"{s[8:10]}.{s[5:7]}.{s[:4]}"
    return f"Tag {s}"


def _pct0(p):
    return "unter 1 %" if p < 0.01 else f"{_de(100 * p, 0)} %"


def _days(d):
    return "1 Tag" if round(d) == 1 else f"{d:.0f} Tage"


def _var_es(x, conf):
    if len(x) < 20:
        return None, None
    tm = TailRiskMetrics(x)
    return float(tm.var_historical(conf)), float(tm.expected_shortfall(conf))


def _segments(x, y, state, k):
    """One trace per state; each segment repeats its first point of the next state so lines join."""
    out = []
    for s in range(k):
        mask = state == s
        joined = mask.copy()
        joined[1:] |= mask[:-1]  # extend each run by one point to meet the next segment
        out.append(np.where(joined, y, np.nan))
    return out


def run(params, data, out):
    r_all = np.asarray(data.returns, dtype=float)
    k, conf = int(params["n_states"]), float(params["confidence"])
    keep = min(len(r_all), int(params["max_years"]) * DAYS)
    off = len(r_all) - keep
    r = r_all[off:]
    n = len(r)
    if n < 250:
        raise ToolError(f"Für Regime braucht es mindestens 250 Tage (vorhanden: {n}).")
    if np.std(r) <= 0:
        raise ToolError("Die Renditen sind konstant – es gibt keine Volatilitäts-Regime.")
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])
    names, colors = STATE_NAMES[k], STATE_COLORS[k]

    out.progress("Erkenne Volatilitäts-Regime (EM-Algorithmus) …", 0.1)
    ra = RegimeAwareTailRisk(r, vol_percentile=float(params["vol_percentile"]), markov_states=k)
    msm: MarkovRegimeSwitching = ra.markov
    if msm.filtered_probs is None:
        raise ToolError("Das Markov-Modell ist nicht konvergiert. Wähle einen anderen Zeitraum.")

    probs = np.asarray(msm.filtered_probs, dtype=float)
    state = np.argmax(probs, axis=1)
    P = np.asarray(msm.transition_matrix, dtype=float)
    stds, means = np.asarray(msm.state_stds, dtype=float), np.asarray(msm.state_means, dtype=float)
    cur = int(msm.current_regime())
    cur_p = np.asarray(msm.regime_probability(), dtype=float)
    fc = np.asarray(msm.forecast_regime(int(params["forecast_days"])), dtype=float)

    det = VolatilityClusteringDetector(r)
    lb = det.ljung_box_test(20)
    rv = det.rolling_realized_volatility(20)
    cond = ra.regime_conditional_var(conf)
    trans = ra.regime_transition_risk()
    simple = ra.current_regime()
    X, Y, Z = compute_regime_coordinates(r, window=int(params["coord_window"]))

    dates = data.dates[off:] if data.dates else None
    x = dates if dates else list(range(n))
    xa = index_axis(data)[1]
    stay = 1 / np.maximum(1 - np.diag(P), 1e-9)
    var_by_state = [_var_es(r[state == s], conf) for s in range(k)]
    var_all = _var_es(r, conf)

    status = {"good": "good", "warning": "warning", "critical": "serious"}
    out.metric("Aktuelles Regime (Markov)", names[cur], "text", hint=f"Wahrscheinlichkeit {_de(100 * cur_p[cur], 0)} %",
               status=status[colors[cur]])
    out.metric("P(Stress-Regime) heute", cur_p[-1], "pct", digits=0)
    out.metric("Vola im Stress-Regime", stds[-1] * np.sqrt(DAYS), "pct", digits=1,
               hint=f"ruhig: {_de(100 * stds[0] * np.sqrt(DAYS), 1)} % p. a.")
    out.metric("Ø Dauer des Stress-Regimes", stay[-1], "num", digits=0, unit="Tage",
               hint=f"ruhig: {_days(stay[0])}")
    if var_by_state[-1][0] is not None and var_all[0]:
        out.metric(f"VaR {lvl} im Stress ÷ gesamt", var_by_state[-1][0] / var_all[0], "x",
                   hint=f"{_de(100 * var_by_state[-1][0])} % statt {_de(100 * var_all[0])} %")
    out.metric("Realisierte Vola heute (20 Tage)", rv[-1], "pct", digits=1,
               hint=f"Vola-Regime: {VOL_NAMES.get(simple['simple_regime'], simple['simple_regime'])}")
    out.metric("Ljung-Box (quadrierte Renditen)", lb["p_value"], "num", digits=3,
               hint="p < 0,05: Volatilität clustert" if lb["significant"] else "kein signifikantes Clustering")

    # --- chart 1: price coloured by regime
    if data.prices is not None:
        level, ylab = np.asarray(data.prices, dtype=float)[off:], "Kurs"
    else:
        level, ylab = np.exp(np.cumsum(r)), "Vermögen (Start = 1)"
    segs = _segments(x, level, state, k)
    out.chart(C.xy(
        "Kursverlauf nach wahrscheinlichstem Markov-Regime",
        [C.trace(f"Regime {names[s]}", x, segs[s], color=colors[s], width=1.8) for s in range(k)],
        x=xa, y=C.axis(ylab, type="log" if level.max() / max(level.min(), 1e-12) > 8 else "linear"),
        subtitle="Gefilterte Zustände: Jeder Tag nutzt nur Daten bis zu diesem Tag.",
    ))

    # --- chart 2: probability of the stress regime
    out.chart(C.xy(
        "Wahrscheinlichkeit des Stress-Regimes",
        [C.trace("P(Stress)", x, probs[:, -1], type="area", color="critical", width=1)],
        x=xa, y=C.axis("Wahrscheinlichkeit", fmt="pct", digits=0, range=(0, 1)),
        hlines=[C.hline(0.5, "50 %")], height=260,
    ))

    # --- chart 3: transition matrix
    out.chart(C.heatmap(
        "Übergangsmatrix: P(morgen | heute)", [f"nach {s}" for s in names], [f"von {s}" for s in names], P,
        z=C.axis("Wahrscheinlichkeit", fmt="pct", digits=1), scale="sequential", height=260 + 30 * k,
        subtitle="Diagonale = Beharrung. Ø Verweildauer: " + ", ".join(f"{s} {_days(d)}" for s, d in zip(names, stay)),
    ))

    # --- chart 4: 3D regime space coloured by Markov state
    start = 2 * int(params["coord_window"])
    if n - start > 20:
        idx = np.arange(start, n)
        step = max(1, len(idx) // 3000)
        lab = [_fmt_date(v) if dates else f"Tag {v}" for v in x]
        traces = []
        for s in range(k):
            sel = idx[state[idx] == s][::step]
            if sel.size:
                traces.append(C.path3d(f"Regime {names[s]}", X[sel], Y[sel], Z[sel], color=colors[s], mode="markers",
                                       size=2.5, text=[lab[i] for i in sel]))
        traces.append(C.path3d("Heute", [X[-1]], [Y[-1]], [Z[-1]], color=1, mode="markers", size=8, text=[lab[-1]]))
        out.chart(C.scatter3d(
            "3D-Regimeraum", traces,
            x=C.axis("Volatilitätsniveau (0–1)"), y=C.axis("Beharrung (0–1)"), z=C.axis("Wechselneigung (0–1)"),
            subtitle="Hohe Volatilität, geringe Beharrung und hohe Wechselneigung = instabiler Stress.",
        ))

    # --- tables
    rows = []
    for s in range(k):
        share = float(np.mean(state == s))
        v_s, e_s = var_by_state[s]
        rows.append([names[s], stds[s] * np.sqrt(DAYS), means[s] * DAYS, stay[s], share, v_s, e_s, cur_p[s]])
    out.table("Markov-Zustände", ["Regime", "Vola p. a.", "Ø Rendite p. a.", "Ø Dauer (Tage)", "Anteil Tage",
                                  f"VaR {lvl}", f"ES {lvl}", "Wahrscheinlichkeit heute"], rows,
              formats=["text", "pct", "pct", "num", "pct", "pct", "pct", "pct"],
              note="VaR und ES historisch aus den Tagen, an denen das Regime am wahrscheinlichsten war.")

    vrows = []
    for key in ("LOW_VOL", "MEDIUM_VOL", "HIGH_VOL", "UNCONDITIONAL"):
        c = cond[key]
        sub = r if key == "UNCONDITIONAL" else r[ra.regimes == ["LOW_VOL", "MEDIUM_VOL", "HIGH_VOL"].index(key)]
        vrows.append([VOL_NAMES.get(key, "alle Tage"), c["n_observations"], c["var"], _var_es(sub, conf)[1]])
    out.table(f"VaR {lvl} je Volatilitäts-Regime (realisierte 20-Tage-Vola)",
              ["Vola-Regime", "Tage", f"VaR {lvl}", f"ES {lvl}"], vrows, formats=["text", "int", "pct", "pct"],
              note=f"Hoch = über dem {int(params['vol_percentile'])}. Perzentil, niedrig = unter dem "
                   f"{100 - int(params['vol_percentile'])}. Perzentil der realisierten Volatilität.")

    hs = [h for h in (1, 5, 10, 20, 60, 120) if h <= len(fc)]
    if len(fc) not in hs:
        hs.append(len(fc))
    out.table("Regime-Prognose", ["in … Tagen"] + [f"P({s})" for s in names],
              [[f"{h}"] + list(fc[h - 1]) for h in hs], formats=["text"] + ["pct"] * k,
              note="Aus der Übergangsmatrix fortgeschrieben; für lange Horizonte nähert sie sich der Langfrist-Verteilung.")

    out.table("Nach einem Wechsel des Vola-Regimes (folgende 5 Tage)", ["Wechsel", "Anzahl", "Ø Tagesrendite", "Volatilität (täglich)"],
              [["zu höherer Vola", trans["n_up_transitions"], trans["post_up_mean_return"], trans["post_up_volatility"]],
               ["zu niedrigerer Vola", trans["n_down_transitions"], trans["post_down_mean_return"],
                trans["post_down_volatility"]]],
              formats=["text", "int", "pct", "pct"])
    if off > 0:
        out.note(f"Analysiert werden die letzten {params['max_years']} Jahre ({n} von {len(r_all)} Tagen).", "info")

    persistent = bool(lb["significant"]) and float(np.min(np.diag(P))) > 0.5
    if not persistent:
        out.set_verdict("good", "Keine ausgeprägten Volatilitäts-Regime",
                        f"Der Ljung-Box-Test findet {'kein' if not lb['significant'] else 'nur schwaches'} "
                        f"Clustering, und mindestens ein Zustand hält im Mittel nur {_days(float(np.min(stay)))}. "
                        "Das Modell trennt hier eher normale Tage von Sprüngen als echte Marktphasen.")
        return

    p_stress = float(cur_p[-1])
    v_s = var_by_state[-1][0]
    stress_txt = (f" Im Stress-Regime liegt der VaR {lvl} bei {_de(100 * v_s)} % statt {_de(100 * var_all[0])} % im "
                  "Gesamtschnitt." if v_s is not None and var_all[0] else "")
    if p_stress > 0.5:
        out.set_verdict("serious", f"Der Markt ist im Stress-Regime (P = {_de(100 * p_stress, 0)} %)",
                        f"Die Volatilität liegt bei rund {_de(100 * stds[-1] * np.sqrt(DAYS), 0)} % p. a.; Stressphasen "
                        f"halten im Mittel {_days(stay[-1])}." + stress_txt)
    elif cur != 0:
        out.set_verdict("warning", f"Erhöhte Volatilität (Regime „{names[cur]}“)",
                        f"Stress-Wahrscheinlichkeit {_pct0(p_stress)}." + stress_txt)
    else:
        out.set_verdict("good", f"Ruhiges Regime (P = {_de(100 * cur_p[0], 0)} %)",
                        f"Ruhige Phasen halten im Mittel {_days(stay[0])}. Die Stress-Wahrscheinlichkeit liegt bei "
                        f"{_pct0(p_stress)}." + stress_txt)
