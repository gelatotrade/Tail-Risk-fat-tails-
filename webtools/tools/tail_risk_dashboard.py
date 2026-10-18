"""The framework's all-in-one tail-risk dashboard (TailRiskDashboard) plus its text report, made interactive."""

import re

import numpy as np
from scipy import stats

from src.visualization.dashboard import TailRiskDashboard, create_summary_report

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tail-risk-dashboard",
    "title": "Tail-Risk-Gesamtdashboard",
    "group": "dashboard",
    "order": 10,
    "summary": "Alles auf einen Blick: VaR, Expected Shortfall, Tail-Index, Volatilitätsregime und der "
               "Composite-Risikoindex des Frameworks – mit Original-Dashboard und Textbericht.",
    "description": (
        "`TailRiskDashboard` rechnet in einem Durchgang alles, was das Framework über die Ränder deiner Renditen "
        "weiß: rollierenden VaR und Expected Shortfall (99 %, 60 Tage), Schiefe, Kurtosis und Hill-Tail-Index, "
        "die drei Physik-Koordinatensysteme (Lévy, Tsallis, Phasenübergang) und das aktuelle Volatilitätsregime "
        "(einfache Vola-Perzentile und ein Markov-Modell mit drei Zuständen). Die Original-Grafik "
        "(`create_full_dashboard`) zeigt alles in 15 Feldern; oben findest du die wichtigsten Reihen interaktiv.\n\n"
        "Der **Composite-Risikoindex** ist die Formel des Dashboards: (λ + 0,5·q + χ) / 2,5 aus Sprungintensität "
        "λ = |r|/σ (Lévy), Tsallis-q und Suszeptibilität χ, danach auf 0–100 über die gezeigte Historie skaliert. "
        "Über 75 heißt EXTREM, über 50 ERHÖHT, über 25 MODERAT. Weil λ die heutige Tagesbewegung misst, springt "
        "der Index von Tag zu Tag – lies deshalb auch das 20-Tage-Mittel.\n\n"
        "Der **Textbericht** (`create_summary_report`) fasst Momente, Tail-Index, VaR/ES und den 3σ-Vergleich "
        "mit der Normalverteilung zusammen; du kannst ihn im Original herunterladen. **Zeitraum** begrenzt die "
        "Rechnung auf die letzten Handelstage (das Markov-Modell ist der teuerste Teil)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Zeitraum (letzte Handelstage)", "unit": "Tage",
         "help": "2520 Tage ≈ 10 Jahre. Längere Zeiträume brauchen deutlich mehr Rechenzeit."},
        {"name": "figure", "type": "bool", "default": True, "label": "Original-Dashboard (Matplotlib) zeigen",
         "help": "Die 15-Felder-Grafik aus create_full_dashboard – kostet im Browser einige Sekunden."},
    ],
    "covers": [
        "src.visualization.dashboard.TailRiskDashboard",
        "src.visualization.dashboard.TailRiskDashboard._compute_metrics",
        "src.visualization.dashboard.TailRiskDashboard._generate_synthetic_vix",
        "src.visualization.dashboard.TailRiskDashboard.create_full_dashboard",
        "src.visualization.dashboard.create_summary_report",
    ],
    "cost": "medium",
}

CLASSES = [(75, "EXTREM", "critical"), (50, "ERHÖHT", "serious"), (25, "MODERAT", "warning"), (-1, "NIEDRIG", "good")]
REGIMES = {"LOW_VOL": "Niedrige Vola", "MEDIUM_VOL": "Mittlere Vola", "HIGH_VOL": "Hohe Vola", "UNKNOWN": "unbekannt",
           "UNCONDITIONAL": "Gesamt (unbedingt)"}
REGIME_STATUS = {"LOW_VOL": "good", "MEDIUM_VOL": "warning", "HIGH_VOL": "serious"}

# create_summary_report: English labels -> German
REPORT_SECTIONS = {"BASIC STATISTICS": "Grundstatistik", "TAIL RISK METRICS": "Tail-Risiko",
                   "TAIL PROBABILITY COMPARISON (3σ)": "3σ-Vergleich (beide Ränder)",
                   "INTERPRETATION": "Einordnung des Berichts"}
REPORT_LABELS = {"Mean Return": "Mittlere Tagesrendite", "Standard Deviation": "Standardabweichung (täglich)",
                 "Skewness": "Schiefe", "Excess Kurtosis": "Exzess-Kurtosis", "Tail Index (α)": "Tail-Index α (Hill)",
                 "VaR (99%)": "VaR 99 % (historisch)", "ES (99%)": "Expected Shortfall 99 %",
                 "ES/VaR Ratio": "Verhältnis ES / VaR", "Gaussian Probability": "Normalverteilung: P(|r| > 3σ)",
                 "Empirical Probability": "Beobachtet: P(|r| > 3σ)", "Fat Tail Ratio": "Faktor beobachtet / Gauß"}
REPORT_REMARKS = {"(Fat tails!)": "fette\u00a0Ränder", "(Normal)": "im\u00a0Normalbereich",
                  "(High tail risk)": "hohes\u00a0Tail-Risiko"}
REPORT_LINES = {
    "Distribution has significant fat tails": "Die Verteilung hat deutlich fette Ränder (Exzess-Kurtosis > 3)",
    "Distribution close to normal": "Verteilung nahe der Normalverteilung (Exzess-Kurtosis ≤ 3)",
    "Negative skewness indicates crash risk": "Negative Schiefe (< −0,5): Crashs häufiger als Rallyes",
    "Symmetric distribution": "Weitgehend symmetrisch (Schiefe ≥ −0,5)",
    "Tail events are significantly more likely than Gaussian predicts":
        "3σ-Tage mehr als doppelt so häufig wie von der Normalverteilung vorhergesagt",
    "Tail probabilities close to Gaussian": "3σ-Tage etwa so häufig wie bei einer Normalverteilung",
}


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _report_rows(report):
    """Turn the box-drawn text of create_summary_report into German table rows."""
    rows, section = [], ""
    for line in report.splitlines():
        if not line.startswith("║"):
            continue
        body = line.strip("║").strip()
        if not body or set(body) <= set("─ "):
            continue
        if body in REPORT_SECTIONS:
            section = REPORT_SECTIONS[body]
            continue
        if body.startswith(("⚠", "✓")):
            text = body.lstrip("⚠️✓ ").strip()
            rows.append([section, REPORT_LINES.get(text, text), "", "Warnung" if body.startswith("⚠") else "ok"])
            continue
        if ":" not in body:
            continue
        label, value = (s.strip() for s in body.split(":", 1))
        m = re.match(r"(-?[\d.]+)\s*(%|x)?\s*(\(.*\))?", value)
        if not m:
            continue
        num, unit, remark = m.groups()
        digits = len(num.split(".")[1]) if "." in num else 0
        shown = _de(float(num), digits) + ("\u00a0%" if unit == "%" else "×" if unit == "x" else "")
        rows.append([section, REPORT_LABELS.get(label, label), shown, REPORT_REMARKS.get(remark or "", "")])
    return rows


def _last(a, default=np.nan):
    a = np.asarray(a, dtype=float)
    return float(a[-1]) if len(a) and np.isfinite(a[-1]) else default


def run(params, data, out):
    n_all = data.n
    k = int(min(params["last_n"], n_all))
    if k < 150:
        raise ToolError(f"Das Dashboard braucht mindestens 150 Tagesrenditen, der Datensatz hat {n_all}.")
    r = np.asarray(data.returns[-k:], dtype=float)
    vix = np.asarray(data.vix[-k:], dtype=float) if data.vix is not None else None
    dates = data.dates[-k:] if data.dates else None
    x = list(dates) if dates else list(range(n_all - k + 1, n_all + 1))
    xa = C.axis("", type="date") if dates else C.axis("Handelstag", fmt="int")

    out.progress("Physik-Koordinaten, rollierende Kennzahlen und Regime …", 0.1)
    dash = TailRiskDashboard(r, vix=vix)      # __init__ runs _compute_metrics()
    report = create_summary_report(r)
    rm = dash.rolling_metrics

    # composite exactly as in create_full_dashboard (panels 1.4 and 4.2)
    risk_level = (dash.levy_coords[2] + 0.5 * dash.tsallis_coords[0] + dash.phase_coords[0]) / 2.5
    composite = (risk_level - np.nanmin(risk_level)) / (np.nanmax(risk_level) - np.nanmin(risk_level) + 1e-10) * 100
    comp_now = float(composite[-1])
    comp_avg = np.convolve(composite, np.ones(20) / 20, mode="full")[:k]
    comp_avg[:19] = np.nan
    cls, cls_status = next((name, st) for lim, name, st in CLASSES if comp_now > lim)

    var99, es99 = _last(rm["var_99"]), _last(rm["es_99"])
    alpha_now, kurt_now = _last(rm["tail_index"]), _last(rm["kurtosis"])
    reg = dash.current_regime
    simple = reg.get("simple_regime", "UNKNOWN")
    markov = reg.get("markov_regime")
    probs = reg.get("regime_probabilities", {})

    # --- metrics ------------------------------------------------------------------------
    out.metric("Risikostufe (Composite)", cls, "text", status=cls_status,
               hint=f"Index heute {_de(comp_now, 0)} von 100 · 20-Tage-Mittel {_de(float(comp_avg[-1]), 0)}")
    out.metric("VaR 99 % (60 Tage)", var99, "pct", digits=2, hint="historisch, Tagesverlust",
               status="serious" if var99 > 0.03 else "warning" if var99 > 0.02 else None)
    out.metric("Expected Shortfall 99 %", es99, "pct", digits=2,
               hint=f"Ø Verlust jenseits des VaR · ES/VaR = {_de(es99 / var99)}" if var99 > 0 else "")
    out.metric("Tail-Index α (60 Tage)", alpha_now, "num", digits=2, hint="Hill · < 3: fette Ränder",
               status="warning" if alpha_now < 3 else None, badge="Fette Ränder" if alpha_now < 3 else "")
    out.metric("Exzess-Kurtosis (60 Tage)", kurt_now, "num", digits=1, hint="Normalverteilung = 0")
    out.metric("Volatilitätsregime", REGIMES.get(simple, simple), "text", status=REGIME_STATUS.get(simple),
               hint="einfach: 20-Tage-Vola gegen ihre Perzentile")
    if markov:
        key = {"LOW_VOL": "low_vol", "MEDIUM_VOL": "medium_vol", "HIGH_VOL": "high_vol"}.get(markov)
        p = probs.get(key)
        out.metric("Markov-Regime", REGIMES.get(markov, markov), "text", status=REGIME_STATUS.get(markov),
                   hint=f"Wahrscheinlichkeit {_de(100 * float(p), 0)} % (3-Zustands-Markov-Modell)" if p is not None else "")
    synth = dash._generate_synthetic_vix() if vix is not None else np.asarray(dash.vix, dtype=float)
    out.metric("Synthetischer VIX", _last(synth), "num", digits=1,
               hint="20-Tage-Vola p. a. × 100" + (f" · echter VIX heute {_de(float(vix[-1]), 1)}" if vix is not None else ""))

    # --- interactive key charts ---------------------------------------------------------
    out.chart(C.xy(
        "VaR und Expected Shortfall (99 %, rollierend 60 Tage)",
        [C.envelope("Abstand ES – VaR", x, rm["var_99"], rm["es_99"], color=2),
         C.trace("VaR 99 %", x, rm["var_99"], color=1, width=1.5),
         C.trace("Expected Shortfall 99 %", x, rm["es_99"], color=2, width=1.5)],
        x=xa, y=C.axis("Tagesverlust", fmt="pct", digits=1),
        subtitle="Der ES ist der Durchschnitt der Verluste jenseits des VaR – je größer der Abstand, desto fetter der Rand.",
    ))
    tail = np.asarray(rm["tail_index"], dtype=float)
    out.chart(C.xy(
        "Tail-Index α (Hill, rollierend 60 Tage)",
        [C.trace("α", x, tail, color=1, width=1.5)],
        x=xa, y=C.axis("α", range=(0, 6)),
        hlines=[C.hline(2.0, "Lévy-Grenze α = 2"), C.hline(3.0, "Kubisches Gesetz α = 3", color="warning")],
        subtitle="Unter 3 liegen die Ränder im Bereich typischer Aktien-Crashs; unter 2 wäre die Varianz unendlich.",
    ))
    out.chart(C.xy(
        "Composite-Risikoindex des Dashboards (0–100)",
        [C.trace("Composite (Tageswert)", x, composite, color=1, width=1),
         C.trace("20-Tage-Mittel", x, comp_avg, color=2, width=2)],
        x=xa, y=C.axis("Index", range=(0, 100)),
        hlines=[C.hline(25, "Moderat"), C.hline(50, "Erhöht", color="serious"), C.hline(75, "Extrem", color="critical")],
        subtitle="(λ + 0,5·q + χ) / 2,5, auf die gezeigte Historie skaliert. Der Tageswert folgt vor allem |r|/σ von heute.",
    ))
    sd = float(np.std(r))
    sig = [2, 3, 4, 5]
    gauss = [2 * stats.norm.sf(s) for s in sig]                      # as in panel 4.4: both tails
    emp = [float(np.mean(np.abs(r) > s * sd)) for s in sig]
    floor = min(g for g in gauss) / 3
    out.chart(C.xy(
        "Extreme Tage: Normalverteilung vs. beobachtet (beide Ränder)",
        [C.trace("Normalverteilung", [f"{s} σ" for s in sig], gauss, type="bar", color="muted"),
         C.trace("Beobachtet", [f"{s} σ" for s in sig], [e if e > 0 else None for e in emp], type="bar", color=1)],
        x=C.axis("Schwelle", type="category"), y=C.axis("Häufigkeit", type="log", fmt="pct", range=(floor, 0.2)),
        subtitle="Log-Skala: jeder Teilstrich ist Faktor 10. Fehlt ein Balken, kam der Fall nie vor.",
    ))

    # --- tables ---------------------------------------------------------------------
    rows = _report_rows(report)
    if rows:
        out.table("Tail-Risk-Bericht (create_summary_report)", ["Abschnitt", "Kennzahl", "Wert", "Einordnung"], rows,
                  formats=["text", "text", "text", "text"],
                  note=f"Über alle {k} gezeigten Tage. Schwellen des Originalberichts: Kurtosis > 3, Schiefe < −0,5, "
                       "Faktor > 2, α < 3, ES/VaR > 1,5.")
    rv = dash.regime_var or {}
    if rv:
        out.table("VaR 99 % je Volatilitätsregime", ["Regime", "VaR 99 %", "Beobachtungen"],
                  [[REGIMES.get(key, key), v.get("var"), v.get("n_observations")] for key, v in rv.items()],
                  formats=["text", "pct", "int"],
                  note="Historischer VaR, nur aus den Tagen des jeweiligen Regimes (RegimeAwareTailRisk).")
    if probs:
        out.table("Markov-Modell: Wahrscheinlichkeit der Regime heute", ["Regime", "Wahrscheinlichkeit"],
                  [[REGIMES.get(key.upper(), key), float(v)] for key, v in probs.items()], formats=["text", "pct"])
    elif simple == "UNKNOWN":
        out.note("Die Regime-Analyse ist für diesen Datensatz fehlgeschlagen; die Regimefelder bleiben leer.", "warning")
    out.download("tail_risk_report.txt", report.strip("\n") + "\n", "text/plain;charset=utf-8")

    # --- verdict: the dashboard's own risk class ----------------------------------------
    texts = {
        "EXTREM": "Der Composite-Index liegt im obersten Viertel seiner Historie – die heutige Bewegung, die Tsallis-q "
                  "und die Suszeptibilität sind gemeinsam hoch.",
        "ERHÖHT": "Der Composite-Index liegt über 50: mehrere Physik-Koordinaten zeigen erhöhte Spannung.",
        "MODERAT": "Der Composite-Index liegt zwischen 25 und 50 – leicht erhöht, aber ohne Extremsignal.",
        "NIEDRIG": "Der Composite-Index liegt im unteren Viertel seiner Historie.",
    }
    tail_txt = (f" Der Tail-Index von {_de(alpha_now)} zeigt fette Ränder." if alpha_now < 3
                else f" Der Tail-Index von {_de(alpha_now)} liegt im Normalbereich.")
    out.set_verdict(cls_status, f"Risikostufe {cls} (Composite {_de(comp_now, 0)})", texts[cls] + tail_txt)

    if params["figure"]:
        out.progress("Original-Dashboard wird gezeichnet …", 0.7)
        fig = dash.create_full_dashboard()      # no save_path: nothing is written
        out.figure(fig, "Original-Dashboard: create_full_dashboard()", dpi=56)
