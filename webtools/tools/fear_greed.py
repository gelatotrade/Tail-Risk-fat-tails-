"""Fear and greed: the framework's composite sentiment score from VIX, put/call ratio, breadth and momentum."""

import numpy as np

from src.analysis.sentiment import CompositeSentimentIndicator, PutCallRatioAnalyzer

from .. import charts as C
from ._ews_common import num

SPEC = {
    "id": "fear-greed",
    "title": "Fear and greed calculator",
    "group": "warning",
    "order": 40,
    "summary": "Combines the VIX, put/call ratio, market breadth and momentum into a sentiment score from −100 "
               "(extreme fear) to +100 (extreme greed), and shows what it implies for crash risk.",
    "description": (
        "The framework's `CompositeSentimentIndicator` translates four market readings into sub-scores from "
        "−100 to +100 and takes a weighted average: **VIX** 30% (neutral at 20, 5 points per VIX point), "
        "**put/call ratio** 25% (many puts = fear), **market breadth** 25% (share of stocks above their 50-day "
        "moving average, neutral at 50%) and **momentum** 20% (index return over recent weeks, 10 points per "
        "percentage point). Above +25 the market is greedy, below −25 fearful, and beyond ±75 extreme.\n\n"
        "From this the framework derives a **tail-risk signal**, read in a contrarian way: above +50 it calls "
        "for vigilance, above +75 (extreme greed = complacency) it reports elevated crash risk. Below −50 it "
        "expects higher volatility; below −75 the sell-off is already under way and a reversal may be near.\n\n"
        "Without history, the `PutCallRatioAnalyzer` rates the put/call ratio against fixed thresholds "
        "(0.6 / 0.7 / 0.85 / 1.0); with at least 20 values of history it uses the percentile, which also yields "
        "a contrarian signal. You can switch on a simulated one-year history here (seed); it is illustrative and "
        "shifts the put/call sub-score. The what-if chart varies one input and holds the others fixed."
    ),
    "needs": [],
    "params": [
        {"name": "vix", "type": "float", "default": 14.5, "min": 5.0, "max": 90.0, "step": 0.1, "label": "VIX"},
        {"name": "put_call", "type": "float", "default": 0.65, "min": 0.3, "max": 2.0, "step": 0.01,
         "label": "Put/call ratio (equities)",
         "help": "Put volume ÷ call volume; above 1 = more hedging than bets on a rise."},
        {"name": "breadth", "type": "float", "default": 58.0, "min": 0.0, "max": 100.0, "step": 1.0,
         "label": "Market breadth", "unit": "%", "help": "Share of stocks above their 50-day moving average."},
        {"name": "momentum", "type": "float", "default": 1.5, "min": -30.0, "max": 30.0, "step": 0.1,
         "label": "Momentum", "unit": "%", "help": "Index return over recent weeks, in per cent."},
        {"name": "history", "type": "select", "default": "none", "label": "Put/call history",
         "options": [{"value": "none", "label": "No history (fixed thresholds)"},
                     {"value": "sim", "label": "Simulated one-year history (percentile mode)"}],
         "help": "With history, the framework rates the put/call ratio in relative terms (percentile) and "
                 "issues contrarian signals."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "History seed", "show_if": {"history": ["sim"]}},
        {"name": "what_if", "type": "select", "default": "vix", "label": "What-if variable",
         "options": [{"value": "vix", "label": "VIX"}, {"value": "put_call", "label": "Put/call ratio"},
                     {"value": "breadth", "label": "Market breadth"}, {"value": "momentum", "label": "Momentum"}]},
    ],
    "covers": [
        "src.analysis.sentiment.CompositeSentimentIndicator",
        "src.analysis.sentiment.CompositeSentimentIndicator.compute_composite",
        "src.analysis.sentiment.PutCallRatioAnalyzer",
        "src.analysis.sentiment.PutCallRatioAnalyzer.add_data",
        "src.analysis.sentiment.PutCallRatioAnalyzer.fear_greed_indicator",
        "src.analysis.sentiment.PutCallRatioAnalyzer.contrarian_signal",
    ],
    "cost": "fast",
}

LEVEL = {"EXTREME_GREED": "Extreme greed", "GREED": "Greed", "NEUTRAL": "Neutral", "FEAR": "Fear",
         "EXTREME_FEAR": "Extreme fear", "UNKNOWN": "Unknown"}
TAIL = {  # tail-risk level -> (status, label)
    "NORMAL": ("good", "Normal"), "MODERATE": ("warning", "Moderate"), "ELEVATED": ("serious", "Elevated"),
    "HIGH": ("serious", "High"), "EXTREME": ("critical", "Extreme"),
}
TEXT = {
    "Complacency high - crash risk elevated": "High complacency – crash risk is elevated",
    "Bullish sentiment - monitor for extremes": "Bullish sentiment – watch for excesses",
    "Panic selling - may be near capitulation": "Panic selling – possibly close to capitulation",
    "Fear rising - volatility expected": "Fear is rising – expect higher volatility",
    "Neutral sentiment": "Sentiment is not extreme (between −50 and +50)",
    "Reduce risk exposure": "Reduce risk exposure", "Maintain hedges": "Maintain hedges",
    "Prepare for reversal": "Prepare for a reversal", "Increase hedges": "Increase hedges",
    "Normal positioning": "Normal positioning",
    "Extreme fear often marks bottoms": "Extreme fear often marks bottoms",
    "Extreme greed often marks tops": "Extreme greed often marks tops",
}
CONTRA = {"CONTRARIAN_BULLISH": "Contrarian bullish", "CONTRARIAN_BEARISH": "Contrarian bearish", "NEUTRAL": "Neutral"}
INPUTS = {"vix": ("VIX", "VIX", (5.0, 60.0)), "put_call": ("Put/call ratio", "Put/call ratio", (0.4, 1.4)),
          "breadth": ("Market breadth", "Market breadth (%)", (0.0, 100.0)),
          "momentum": ("Momentum", "Momentum (%)", (-15.0, 15.0))}


def _history(seed, n=252):
    """An illustrative year of equity put/call ratios: AR(1) around 0.8 (sd about 0.11), clipped to [0.4, 1.6]."""
    rng = np.random.default_rng(int(seed))
    pc = np.empty(n)
    pc[0] = 0.8
    for t in range(1, n):
        pc[t] = 0.8 + 0.85 * (pc[t - 1] - 0.8) + rng.normal(0, 0.06)
    return np.clip(pc, 0.4, 1.6)


def _indicator(pc_hist):
    ind = CompositeSentimentIndicator()
    if pc_hist is not None:
        ind.pc_analyzer = PutCallRatioAnalyzer()
        for v in pc_hist:
            ind.pc_analyzer.add_data(equity=float(v))
    return ind


def run(params, data, out):
    hist = _history(params["seed"]) if params["history"] == "sim" else None
    hist_today = np.append(hist, params["put_call"]) if hist is not None else None
    ind = _indicator(hist_today)
    inputs = {k: float(params[k]) for k in ("vix", "put_call", "breadth", "momentum")}
    res = ind.compute_composite(**inputs)
    comp = float(res["composite"])
    tail = res["tail_risk_signal"]
    status, tail_txt = TAIL[tail["level"]]
    level_txt = LEVEL[res["level"]]

    out.set_verdict(status, f"{level_txt} ({num(comp, 0, sign=True)}) · tail risk: {tail_txt}",
                    f"{TEXT.get(tail['direction'], tail['direction'])}. Framework recommendation: "
                    f"{TEXT.get(tail['action'], tail['action'])}.")
    out.metric("Sentiment", comp, "num", digits=0, hint=f"{level_txt} · scale −100 fear … +100 greed")
    out.metric("Tail-risk signal", tail_txt, "text", hint=TEXT.get(tail["direction"], tail["direction"]), status=status)
    contra = ind.pc_analyzer.contrarian_signal()
    out.metric("Put/call contrarian signal", CONTRA.get(contra["signal"], contra["signal"]), "text",
               hint=(TEXT.get(contra.get("reason", ""), "") + (f" · confidence {num(contra['confidence'] * 100, 0)}%"
                                                               if contra.get("confidence") else ""))
               or ("Requires at least 20 days of put/call history" if hist is None
                   else "Put/call within the normal range of its history"))

    # ---- sub-scores ---------------------------------------------------------------------------------
    keys = [k for k in ("vix", "put_call", "breadth", "momentum") if k in res["components"]]
    cats = [{"vix": "VIX", "put_call": "Put/call", "breadth": "Breadth", "momentum": "Momentum"}[k] for k in keys] + ["Composite"]
    vals = [float(res["components"][k]) for k in keys] + [comp]
    out.chart(C.xy(
        "Sub-scores and composite",
        [C.trace("Sub-score", cats[:-1], vals[:-1], type="bar", color=1),
         C.trace("Composite (weighted)", cats[-1:], vals[-1:], type="bar", color=2)],
        x=C.axis("", type="category"), y=C.axis("Score (−100 fear … +100 greed)", range=(-100, 100)),
        hlines=[C.hline(75, "Extreme greed"), C.hline(25, "Greed"), C.hline(-25, "Fear"), C.hline(-75, "Extreme fear")],
        subtitle="Weights: VIX 30%, put/call 25%, breadth 25%, momentum 20%. Lines = sentiment levels.",
    ))

    # ---- what-if --------------------------------------------------------------------------------------
    name = params["what_if"]
    short, xlabel, (lo, hi) = INPUTS[name]
    cur = inputs[name]
    grid = np.linspace(min(lo, cur), max(hi, cur), 161)
    ys = []
    for v in grid:
        q = dict(inputs, **{name: float(v)})
        # compute_composite does not store anything; only a new put/call value changes the history
        sim = _indicator(np.append(hist, q["put_call"]) if hist is not None else None) if name == "put_call" else ind
        ys.append(sim.compute_composite(**q)["composite"])
    out.chart(C.xy(
        f"What-if: {short}",
        [C.trace("Sentiment", grid, ys, color=1)],
        x=C.axis(xlabel, digits=2 if name == "put_call" else 1), y=C.axis("Sentiment", range=(-100, 100)),
        hlines=[C.hline(75, "Extreme greed"), C.hline(25, "Greed"), C.hline(-25, "Fear"),
                C.hline(-75, "Extreme fear")],
        vlines=[C.vline(cur, "today", "ink")],
        subtitle="All other inputs stay fixed. Beyond ±50 the tail-risk signal reports more than ‘Normal’, "
                 "beyond ±75 it rises by one more level.",
    ))

    # ---- how the framework reads the put/call ratio ------------------------------------------------
    pcs = np.linspace(0.4, 1.4, 201)
    fixed = PutCallRatioAnalyzer()
    traces = [C.trace("Fixed thresholds (no history)", pcs, [fixed.fear_greed_indicator(v)["score"] for v in pcs],
                      type="step", color="muted" if hist is not None else 1, dash=hist is not None)]
    if hist is not None:
        traces.append(C.trace("Percentile of history", pcs,
                              [ind.pc_analyzer.fear_greed_indicator(v)["score"] for v in pcs], color=1))
    out.chart(C.xy(
        "How the framework scores the put/call ratio",
        traces, x=C.axis("Put/call ratio", digits=2), y=C.axis("Sub-score", range=(-100, 100)),
        vlines=[C.vline(inputs["put_call"], "today", "ink")],
        subtitle="Many puts = fear (negative), few puts = greed (positive). With history, the relative position counts.",
    ))
    if hist is not None:
        days = list(range(-len(hist), 0)) + [0]
        p10, p90 = np.percentile(hist_today, [10, 90])
        out.chart(C.xy(
            "Simulated put/call history",
            [C.trace("Put/call (simulated)", days[:-1], hist, color=1, width=1.2),
             C.trace("Today (your input)", [0], [inputs["put_call"]], type="scatter", color="ink", size=10)],
            x=C.axis("Trading days relative to today", fmt="int"), y=C.axis("Put/call ratio"),
            hlines=[C.hline(p90, "90th percentile: contrarian bullish", "serious"),
                    C.hline(p10, "10th percentile: contrarian bearish", "warning")],
            subtitle="If today's ratio lies beyond these lines, the analyser issues a contrarian signal.",
        ))
        out.note("The put/call history is simulated (AR(1) process around 0.8, selectable seed) and serves only to "
                 "demonstrate the framework's percentile mode and contrarian signal.", "info")

    tot = sum(ind.weights[k] for k in keys)
    rows = [[INPUTS[k][1], num(inputs[k], 2 if k == "put_call" else 1), float(res["components"][k]),
             ind.weights[k] / tot, float(res["components"][k]) * ind.weights[k] / tot] for k in keys]
    rows.append(["Composite", "", comp, 1.0, comp])
    out.table("Composition", ["Input", "Value", "Sub-score", "Weight", "Contribution"], rows,
              formats=["text", "text", "num", "pct:0", "num"],
              note="VIX: 5 × (20 − VIX); breadth: 2 × (breadth − 50); momentum: 10 × return in %; all capped at ±100.")
