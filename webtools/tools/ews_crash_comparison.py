"""Early-warning indicators in historical crashes: the framework's stylised GEX / TDEX / VIX patterns."""

import numpy as np

from src.visualization.early_warning_dashboard import CrashComparisonChart

from .. import charts as C
from ._ews_common import num

SPEC = {
    "id": "ews-crash-comparison",
    "title": "Early-warning indicators in historical crashes",
    "group": "warning",
    "order": 25,
    "summary": "Stylised paths of GEX, TailDex and the VIX curve around four crashes (2018, 2020, 2022, 2025), "
               "and which indicator fired beforehand.",
    "description": (
        "For four crashes, the framework stores a typical pattern of three indicators: **Net Gamma Exposure** "
        "(GEX < 0: dealers amplify moves), **TailDex** (above 15: crash protection becomes expensive) and the "
        "ratio of the **VIX futures M1/M2** (above 1: inverted curve, acute fear). The charts cover 60 trading "
        "days before to 30 days after the crash day.\n\n"
        "**Important:** the paths are *stylised patterns* that the framework generates from formulas plus fixed "
        "noise – not historical measurements. They show which signal is considered characteristic of which "
        "type of crash: the volatility shock of 2018, the gamma flip of 2020, the slow sell-off of 2022 and the "
        "TailDex divergence before the 2025 tariff crash.\n\n"
        "How to read it: for each indicator the four crashes are overlaid, with day 0 as the crash day. The "
        "heatmap counts on how many days of the **lead-up window** an indicator was in its warning zone. That is "
        "the real early-warning question: did the signal come before the crash, or only with it?"
    ),
    "needs": [],
    "params": [
        {"name": "lookback", "type": "int", "default": 20, "min": 5, "max": 60, "step": 5,
         "label": "Lead-up window", "unit": "days",
         "help": "Trading days before the crash in which warning signals are counted."},
        {"name": "show_figure", "type": "bool", "default": True, "label": "Show original figure (Matplotlib)"},
    ],
    "covers": [
        "src.visualization.early_warning_dashboard.CrashComparisonChart",
        "src.visualization.early_warning_dashboard.CrashComparisonChart.create_comparison_chart",
    ],
    "cost": "fast",
}

# display name, date and what happened (facts as of the event; the framework's own numbers are in the table)
CRASH_INFO = {
    "2018 Volmageddon": (
        "2018 Volmageddon", "5 Feb 2018", "Implosion of inverse VIX products",
        ("The VIX more than doubled in a single day. Inverse VIX products such as XIV collapsed, and their "
         "rebalancing bought VIX futures into the spike – the curve flipped abruptly into backwardation.")),
    "2020 COVID Crash": (
        "2020 COVID crash", "16 Mar 2020", "Pandemic and liquidity crisis",
        ("From its record high on 19 Feb to the low on 23 Mar, the S&P 500 lost around a third. The VIX closed "
         "at a record 82.7 on 16 Mar; negative dealer gamma accelerated the daily moves.")),
    "2022 Bear Market": (
        "2022 bear market", "12 Oct 2022", "Inflation and interest-rate shock",
        ("High inflation and the Fed's fastest rate hikes in decades pushed the S&P 500 down by a quarter from "
         "January to October. A slow sell-off: the VIX stayed below 40 and the signals were weak.")),
    "2025 Tariff Crash": (
        "2025 tariff crash", "7 Apr 2025", "‘Liberation Day’ tariffs",
        ("After the tariff announcements on 2 Apr, the S&P 500 fell by just over 10% in two trading days, and by "
         "almost 19% from its record high on 19 Feb to its closing low on 8 Apr. The VIX briefly jumped to 60 "
         "intraday on 7 Apr. The tariff pause on 9 Apr marked the turnaround.")),
}
# the framework's own reading of each crash (comments in CrashComparisonChart._plot_crash_indicators)
PATTERN = {"2018 Volmageddon": "VIX curve was the key", "2020 COVID Crash": "Gamma flip dominated",
           "2022 Bear Market": "Slow sell-off, weak signals",
           "2025 Tariff Crash": "Sharp gamma flip plus TDEX divergence"}
# label of the line in the framework's figure -> (key, chart title, axis label, warning test, threshold text)
INDICATORS = {
    "GEX": ("gex", "Net Gamma Exposure (GEX)", "$bn", lambda v: v < 0, "GEX < 0"),
    "TDEX": ("tdex", "TailDex (TDEX)", "TDEX", lambda v: v > 15, "TDEX > 15"),
    "VIX Ratio": ("vix", "VIX curve M1/M2", "Ratio M1 / M2", lambda v: v > 1, "M1/M2 > 1"),
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
    out.metric("Crashes compared", len(crashes), "int", hint="60 days before to 30 days after each crash day")
    out.metric("Largest decline", drops[worst], "pct", digits=1, hint=CRASH_INFO.get(worst, (worst,))[0])
    out.metric(f"Signals in the lead-up ({lookback} days)", early, "int", unit=f"of {share.size}",
               hint="Crash × indicator pairs with at least half of the days in the warning zone")
    i, j = np.unravel_index(int(np.argmax(share)), share.shape)
    lab = list(INDICATORS)[j]
    out.metric("Clearest lead-up signal", share[i, j], "pct", digits=0,
               hint=f"{INDICATORS[lab][1]} before the {CRASH_INFO.get(crashes[i], (crashes[i],))[0]}")

    out.note("**Stylised patterns, not measurements:** the framework generates the paths from formulas per crash "
             "type plus fixed noise (`CrashComparisonChart`). They illustrate typical signals but contain no "
             "historical GEX, TailDex or VIX futures data.", "warning")

    # ---- one chart per indicator, the four crashes on top of each other ----------------------------------
    for lab, (key, title, unit, _, thr) in INDICATORS.items():
        traces = [C.trace(CRASH_INFO.get(c, (c,))[0], series[c][lab][0], series[c][lab][1], color=k + 1, width=1.8)
                  for k, c in enumerate(crashes)]
        hl = {"gex": [C.hline(0, "Gamma flip (0)", "serious")],
              "tdex": [C.hline(15, "Elevated", "warning"), C.hline(20, "Extreme", "critical")],
              "vix": [C.hline(1.0, "Inversion", "serious")]}[key]
        out.chart(C.xy(
            title, traces,
            x=C.axis("Trading days relative to the crash", fmt="int"), y=C.axis(unit, digits=3 if key == "vix" else 1),
            hlines=hl, vlines=[C.vline(0, "Crash day", "ink")],
            bands=[C.band(-lookback, 0, "Lead-up", "muted")],
            subtitle=f"Warning zone: {thr}. Grey shading: the lead-up window.",
        ))
    out.chart(C.heatmap(
        "Share of lead-up days in the warning zone", [v[4] for v in INDICATORS.values()],
        [CRASH_INFO.get(c, (c,))[0] for c in crashes], share,
        z=C.axis("Share of days", fmt="pct", digits=0, range=(0, 1)), scale="sequential", annotate=True,
        subtitle=f"Share of the last {lookback} trading days before the crash. High = the signal came beforehand.",
    ))

    # ---- what happened -------------------------------------------------------------------------------
    rows = []
    for c in crashes:
        name, date, trigger, story = CRASH_INFO.get(c, (c, events[c]["date"], events[c]["trigger"], ""))
        spike = events[c]["vix_spike"]
        spike_txt = f"+{num(spike, 1)}% in one day" if "2018" in c else f"Closing high {num(spike, 1)}"
        rows.append([name, date, trigger, PATTERN.get(c, ""), drops[c], spike_txt, story])
    out.table("The four crashes", ["Crash", "Crash day", "Trigger", "Framework signal", "Decline",
                                   "VIX", "What happened"], rows,
              formats=["text", "text", "text", "text", "pct", "text", "text"],
              note="Decline and VIX as stored in the framework, based on closing prices: for 2018 a single trading "
                   "day, otherwise peak to trough; VIX for 2018 as a one-day rise, otherwise as the highest close.")

    rows = []
    for c in crashes:
        row = [CRASH_INFO.get(c, (c,))[0]]
        for lab, (key, _, _, _, _) in INDICATORS.items():
            days, vals = series[c][lab]
            post = vals[(days >= 0) & (days <= 10)]
            row += [vals[days == -1][0], post.min() if key == "gex" else post.max()]
        rows.append(row)
    out.table("The day before and the crash phase", ["Crash", "GEX T−1", "GEX low after", "TDEX T−1",
                                                     "TDEX high after", "M1/M2 T−1", "M1/M2 high after"],
              rows, formats=["text", "num", "num", "num", "num", "num", "num"],
              note="T−1 = trading day before the crash; ‘after’ = the crash day and the ten trading days that "
                   "follow. GEX in $bn.")

    if params["show_figure"]:
        out.figure(fig, "The framework's crash comparison (CrashComparisonChart)", dpi=80)
