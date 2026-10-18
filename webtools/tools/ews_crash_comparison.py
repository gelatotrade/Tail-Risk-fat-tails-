"""Frühwarnindikatoren bei historischen Crashs: the framework's stylised GEX / TDEX / VIX patterns."""

import numpy as np

from src.visualization.early_warning_dashboard import CrashComparisonChart

from .. import charts as C
from ._ews_common import de

SPEC = {
    "id": "ews-crash-comparison",
    "title": "Frühwarnindikatoren bei historischen Crashs",
    "group": "warning",
    "order": 25,
    "summary": "Wie sich GEX, TailDex und VIX-Kurve vor und nach Volmageddon 2018, COVID 2020, dem Bärenmarkt "
               "2022 und dem Zoll-Crash 2025 typischerweise verhalten – und welcher Indikator vorher anschlug.",
    "description": (
        "Das Framework hinterlegt für vier Crashs ein typisches Muster von drei Indikatoren: **Net Gamma "
        "Exposure** (GEX < 0: Dealer verstärken Bewegungen), **TailDex** (über 15: Crash-Absicherung wird teuer) "
        "und das Verhältnis der **VIX-Futures M1/M2** (über 1: invertierte Kurve, akute Angst). Gezeigt werden "
        "60 Handelstage vor bis 30 Tage nach dem Crash-Tag.\n\n"
        "**Wichtig:** Die Verläufe sind *stilisierte Muster*, die das Framework aus Formeln plus festem Rauschen "
        "erzeugt – keine historischen Messwerte. Sie zeigen, welches Signal bei welchem Crash-Typ als "
        "charakteristisch gilt: der Volatilitäts-Schock 2018, der Gamma-Flip 2020, der langsame Abverkauf 2022 "
        "und die TailDex-Divergenz vor dem Zoll-Crash 2025.\n\n"
        "So liest du es: Je Indikator liegen die vier Crashs übereinander, Tag 0 ist der Crash-Tag. Die Heatmap "
        "zählt, an wie vielen Tagen des **Vorlauf-Fensters** ein Indikator im Warnbereich lag – das ist die "
        "eigentliche Frühwarnfrage: Kam das Signal vor dem Crash oder erst mit ihm?"
    ),
    "needs": [],
    "params": [
        {"name": "lookback", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Vorlauf-Fenster", "unit": "Tage",
         "help": "Die letzten Handelstage vor dem Crash, in denen die Warnsignale gezählt werden."},
        {"name": "show_figure", "type": "bool", "default": True, "label": "Original-Grafik (Matplotlib) zeigen"},
    ],
    "covers": [
        "src.visualization.early_warning_dashboard.CrashComparisonChart",
        "src.visualization.early_warning_dashboard.CrashComparisonChart.create_comparison_chart",
    ],
    "cost": "fast",
}

# German name, date and what happened (facts as of the event; the framework's own numbers are in the table)
CRASH_DE = {
    "2018 Volmageddon": (
        "Volmageddon 2018", "05.02.2018", "Implosion inverser VIX-Produkte",
        ("Der VIX hat sich an einem Tag mehr als verdoppelt. Inverse VIX-Produkte wie XIV brachen zusammen, ihr "
         "Rebalancing kaufte VIX-Futures in den Anstieg – die Kurve kippte schlagartig in Backwardation.")),
    "2020 COVID Crash": (
        "COVID-Crash 2020", "16.03.2020", "Pandemie und Liquiditätskrise",
        ("Vom Rekordhoch am 19.02. bis zum Tief am 23.03. verlor der S&P 500 rund ein Drittel. Der VIX schloss am "
         "16.03. auf dem Rekordwert 82,7; negatives Dealer-Gamma beschleunigte die Tagesbewegungen.")),
    "2022 Bear Market": (
        "Bärenmarkt 2022", "12.10.2022", "Inflation und Zinsschock",
        ("Hohe Inflation und die schnellsten Zinserhöhungen der Fed seit Jahrzehnten drückten den S&P 500 von "
         "Januar bis Oktober um ein Viertel. Ein langsamer Abverkauf: Der VIX blieb unter 40, die Signale schwach.")),
    "2025 Tariff Crash": (
        "Zoll-Crash 2025", "07.04.2025", "„Liberation Day“-Zölle",
        ("Nach den Zollankündigungen vom 02.04. fiel der S&P 500 in zwei Handelstagen um gut 10 %, der VIX "
         "sprang am 07.04. kurz auf 60. Die Zollpause vom 09.04. brachte die Wende.")),
}
# the framework's own reading of each crash (comments in CrashComparisonChart._plot_crash_indicators)
PATTERN_DE = {"2018 Volmageddon": "VIX-Kurve war der Schlüssel", "2020 COVID Crash": "Gamma-Flip dominierte",
              "2022 Bear Market": "langsamer Abverkauf, schwache Signale",
              "2025 Tariff Crash": "scharfer Gamma-Flip + TDEX-Divergenz"}
# label of the line in the framework's figure -> (key, chart title, axis label, warning test, threshold text)
INDICATORS = {
    "GEX": ("gex", "Net Gamma Exposure (GEX)", "Mrd. $", lambda v: v < 0, "GEX < 0"),
    "TDEX": ("tdex", "TailDex (TDEX)", "TDEX", lambda v: v > 15, "TDEX > 15"),
    "VIX Ratio": ("vix", "VIX-Kurve M1/M2", "Verhältnis M1 / M2", lambda v: v > 1, "M1/M2 > 1"),
}


def _extract(fig):
    """The patterns exactly as the framework plotted them: {crash: {label: (days, values)}}."""
    series = {}
    for ax in fig.axes:
        title = ax.get_title()
        if not title:
            continue  # the twin axes carry no title; their lines are reached through the shared x axis
        crash = title.split("\n")[0]
        lines = {}
        for sib in ax.get_shared_x_axes().get_siblings(ax):
            for line in sib.get_lines():
                if line.get_label() in INDICATORS:
                    lines[line.get_label()] = (np.asarray(line.get_xdata(), dtype=float),
                                               np.asarray(line.get_ydata(), dtype=float))
        series[crash] = lines
    return series


def run(params, data, out):
    chart = CrashComparisonChart()
    events = chart.crash_events
    fig = chart.create_comparison_chart(figsize=(16, 12))
    series = _extract(fig)
    crashes = [c for c in events if c in series]
    lookback = int(params["lookback"])

    # ---- warning days before the crash (the early-warning question) ----------------------------------------
    share = np.zeros((len(crashes), len(INDICATORS)))
    for i, crash in enumerate(crashes):
        for j, (lab, (_, _, _, warn, _)) in enumerate(INDICATORS.items()):
            days, vals = series[crash][lab]
            pre = (days >= -lookback) & (days < 0)
            share[i, j] = float(np.mean(warn(vals[pre])))

    drops = {c: events[c]["drop"] / 100 for c in crashes}
    worst = min(drops, key=drops.get)
    early = int(np.sum(share >= 0.5))
    out.metric("Crashs im Vergleich", len(crashes), "int", hint="je 60 Tage vor bis 30 Tage nach dem Crash-Tag")
    out.metric("Größter Rückgang", drops[worst], "pct", digits=1, hint=CRASH_DE.get(worst, (worst,))[0])
    out.metric(f"Signale im Vorlauf ({lookback} Tage)", early, "int", unit=f"von {share.size}",
               hint="Crash × Indikator mit mindestens der Hälfte der Tage im Warnbereich")
    i, j = np.unravel_index(int(np.argmax(share)), share.shape)
    lab = list(INDICATORS)[j]
    out.metric("Deutlichstes Vorlauf-Signal", share[i, j], "pct", digits=0,
               hint=f"{INDICATORS[lab][1]} vor dem {CRASH_DE.get(crashes[i], (crashes[i],))[0]}")

    out.note("**Stilisierte Muster, keine Messwerte:** Das Framework erzeugt die Verläufe aus Formeln je Crash-Typ "
             "plus festem Rauschen (`CrashComparisonChart`). Sie veranschaulichen typische Signale – historische "
             "GEX-, TailDex- oder VIX-Futures-Daten stecken nicht darin.", "warning")

    # ---- one chart per indicator, the four crashes on top of each other ----------------------------------
    for lab, (key, title, unit, _, thr) in INDICATORS.items():
        traces = [C.trace(CRASH_DE.get(c, (c,))[0], series[c][lab][0], series[c][lab][1], color=k + 1, width=1.8)
                  for k, c in enumerate(crashes)]
        hl = {"gex": [C.hline(0, "Gamma-Flip (0)", "serious")],
              "tdex": [C.hline(15, "Erhöht", "warning"), C.hline(20, "Extrem", "critical")],
              "vix": [C.hline(1.0, "Inversion", "serious")]}[key]
        out.chart(C.xy(
            title, traces,
            x=C.axis("Handelstage relativ zum Crash", fmt="int"), y=C.axis(unit, digits=3 if key == "vix" else 1),
            hlines=hl, vlines=[C.vline(0, "Crash-Tag", "ink")],
            bands=[C.band(-lookback, 0, "Vorlauf", "muted")],
            subtitle=f"Warnbereich: {thr}. Grau hinterlegt: das Vorlauf-Fenster.",
        ))
    out.chart(C.heatmap(
        "Wie oft lag der Indikator im Vorlauf im Warnbereich?", [v[4] for v in INDICATORS.values()],
        [CRASH_DE.get(c, (c,))[0] for c in crashes], share,
        z=C.axis("Anteil der Tage", fmt="pct", digits=0), scale="risk", annotate=True,
        subtitle=f"Anteil der letzten {lookback} Handelstage vor dem Crash. Hoch = das Signal kam vorher.",
    ))

    # ---- what happened -------------------------------------------------------------------------------
    rows = []
    for c in crashes:
        name, date, trigger, story = CRASH_DE.get(c, (c, events[c]["date"], events[c]["trigger"], ""))
        spike = events[c]["vix_spike"]
        spike_txt = f"+{de(spike, 1)} % an einem Tag" if "2018" in c else f"Hoch bei {de(spike, 1)}"
        rows.append([name, date, trigger, PATTERN_DE.get(c, ""), drops[c], spike_txt, story])
    out.table("Die vier Crashs", ["Crash", "Crash-Tag", "Auslöser", "Signal laut Framework", "Rückgang", "VIX",
                                  "Was passierte"], rows,
              formats=["text", "text", "text", "text", "pct", "text", "text"],
              note="Rückgang und VIX wie im Framework hinterlegt: 2018 ein einzelner Handelstag, sonst vom Hoch zum "
                   "Tief; VIX 2018 als Tagesanstieg, sonst als Höchststand.")

    rows = []
    for c in crashes:
        row = [CRASH_DE.get(c, (c,))[0]]
        for lab, (key, _, _, _, _) in INDICATORS.items():
            days, vals = series[c][lab]
            post = vals[(days >= 0) & (days <= 10)]
            row += [vals[days == -1][0], post.min() if key == "gex" else post.max()]
        rows.append(row)
    out.table("Am Vortag und in der Crash-Phase", ["Crash", "GEX T−1", "GEX Tief danach", "TDEX T−1",
                                                   "TDEX Hoch danach", "M1/M2 T−1", "M1/M2 Hoch danach"],
              rows, formats=["text", "num", "num", "num", "num", "num", "num"],
              note="T−1 = Handelstag vor dem Crash; „danach“ = Crash-Tag und die zehn Handelstage darauf. "
                   "GEX in Mrd. $.")

    if params["show_figure"]:
        out.figure(fig, "Crash-Vergleich des Frameworks (CrashComparisonChart)", dpi=80)
