"""Market breadth: breadth thrust and price/breadth divergence on a simulated multi-asset market."""

import numpy as np

from src.analysis.sentiment import MarketBreadthAnalyzer
from src.utils.data_loader import SyntheticMarketGenerator
from src.utils.numerical import rolling_window

from .. import charts as C

SPEC = {
    "id": "market-breadth",
    "title": "Market breadth: thrust and divergence",
    "group": "warning",
    "order": 50,
    "summary": "How many stocks are carrying the index: breadth-thrust and divergence signals on a simulated "
               "market with many individual stocks.",
    "description": (
        "**Market breadth** measures how many stocks participate in a move. Two measures are used: the share "
        "of stocks above their 50-day moving average and the **advance/decline line** (cumulative advancing "
        "minus declining stocks per day). The framework's `MarketBreadthAnalyzer` uses them to look for two "
        "signals: a **breadth thrust** (breadth jumps by more than 40 percentage points within 10 days, often "
        "the start of a recovery) and a **divergence** (the index makes a 20-day high but the A/D line does "
        "not, so the rally is carried by only a few stocks – a classic warning sign).\n\n"
        "As real single-stock data are not available, `SyntheticMarketGenerator` creates a market of "
        "correlated stocks with fat tails; the index is their equally weighted average. The **common market "
        "correlation** controls how strongly the stocks move together: in the original generator (0) the "
        "pairwise correlations scatter around zero and breadth barely moves. Real stocks sit at roughly "
        "0.3–0.5.\n\n"
        "How to read the charts: green dots mark thrust signals, orange dots divergence days. The table shows "
        "how the index performed afterwards. Each value uses only data up to the day in question."
    ),
    "needs": [],
    "params": [
        {"name": "n_assets", "type": "int", "default": 40, "min": 5, "max": 150, "step": 5, "label": "Number of stocks"},
        {"name": "n_days", "type": "int", "default": 1000, "min": 260, "max": 3000, "step": 20,
         "label": "Trading days", "unit": "days"},
        {"name": "base_corr", "type": "float", "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
         "label": "Common market correlation",
         "help": "0 = original generator (pairwise correlations around zero). Real stocks: about 0.3–0.5."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Random seed"},
    ],
    "covers": [
        "src.utils.data_loader.SyntheticMarketGenerator",
        "src.utils.data_loader.SyntheticMarketGenerator.generate",
        "src.analysis.sentiment.MarketBreadthAnalyzer",
        "src.analysis.sentiment.MarketBreadthAnalyzer.add_data",
        "src.analysis.sentiment.MarketBreadthAnalyzer.breadth_thrust_signal",
        "src.analysis.sentiment.MarketBreadthAnalyzer.divergence_warning",
        "src.utils.numerical.rolling_window",
    ],
    "cost": "fast",
}
H = 20  # trading days after a signal


def _starts(mask, gap=10):
    """First day of every signal event: runs of True less than ``gap`` days apart count as one event."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return idx
    keep = np.concatenate([[True], np.diff(idx) > gap])
    return idx[keep]


def _pct_above(prices, w):
    """Share (0-100) of assets above their w-day moving average, for days w-1 .. n-1."""
    ma = rolling_window(prices.T, w).mean(axis=2).T
    return (prices[w - 1:] > ma).mean(axis=1) * 100


def run(params, data, out):
    n_days, k = int(params["n_days"]), int(params["n_assets"])
    gen = SyntheticMarketGenerator(n_assets=k, seed=int(params["seed"]), base_correlation=float(params["base_corr"]))
    mkt = gen.generate(n_days)
    R = np.asarray(mkt["returns"], dtype=float)
    prices = np.exp(np.cumsum(R, axis=0))
    index = 100 * np.exp(np.cumsum(mkt["market"]))
    ad = np.cumsum((R > 0).sum(axis=1) - (R < 0).sum(axis=1))
    b50 = np.full(n_days, np.nan)
    b50[49:] = _pct_above(prices, 50)
    b200 = np.full(n_days, np.nan)
    if n_days >= 200:
        b200[199:] = _pct_above(prices, 200)

    an = MarketBreadthAnalyzer()
    thrust = np.zeros(n_days, bool)
    diverg = np.zeros(n_days, bool)
    magnitude = np.full(n_days, np.nan)
    for t in range(49, n_days):  # feed the analyzer day by day; it only ever sees the past
        an.add_data(ad_line=float(ad[t]), pct_50ma=float(b50[t]),
                    pct_200ma=float(b200[t]) if np.isfinite(b200[t]) else None)
        s = an.breadth_thrust_signal()
        thrust[t] = bool(s.get("thrust_detected"))
        magnitude[t] = s.get("thrust_magnitude", np.nan)
        diverg[t] = bool(an.divergence_warning(index[:t + 1]).get("divergence"))
    signals = _starts(thrust)
    x = list(range(n_days))

    corr = np.asarray(mkt["correlations"], dtype=float)
    iu = np.triu_indices(k, 1)
    dd = index / np.maximum.accumulate(index) - 1
    out.metric("Breadth today", b50[-1] / 100, "pct", digits=0, hint="Stocks above their 50-day moving average")
    out.metric("Breadth-thrust signals", len(signals), "int",
               hint=f"+40 percentage points of breadth in 10 days · {int(thrust.sum()):,} signal days")
    out.metric("Divergence days", int(diverg.sum()), "int", hint="Index at a 20-day high, A/D line not",
               status="warning" if diverg[-5:].any() else None)
    realised = np.corrcoef(R.T)[iu]
    out.metric("Mean pairwise correlation", float(realised.mean()), "num", digits=2,
               hint=f"realised, {k} stocks · target {corr[iu].mean():.2f}")
    out.metric("Total index return", float(index[-1] / 100 - 1), "pct", digits=1)
    out.metric("Maximum index drawdown", float(dd.min()), "pct", digits=1)

    out.chart(C.xy(
        "Index with breadth signals",
        [C.trace("Index (equally weighted)", x, index, color=1, width=1.4),
         C.trace("Divergence (warning)", [x[i] for i in np.flatnonzero(diverg)], index[diverg], type="scatter",
                 color="warning", size=6, decimate_to=None),
         C.trace("Breadth thrust", [x[i] for i in signals], index[signals], type="scatter", color="good", size=10,
                 decimate_to=None)],
        x=C.axis("Trading day", fmt="int"), y=C.axis("Index (base 100)"),
        subtitle="Orange: index at a high, but most stocks are not following. Green: a broad recovery is starting.",
    ))
    out.chart(C.xy(
        "Share of stocks above their moving average",
        [C.trace("Above 50-day average", x, b50 / 100, color=1, width=1.4)]
        + ([C.trace("Above 200-day average", x, b200 / 100, color=2, width=1.2)] if n_days >= 200 else [])
        + [C.trace("Thrust signal", [x[i] for i in signals], b50[signals] / 100, type="scatter", color="good", size=10,
                   decimate_to=None)],
        x=C.axis("Trading day", fmt="int"), y=C.axis("Share", fmt="pct", digits=0, range=(0, 1)),
        subtitle="A thrust: within 10 days, breadth jumps from oversold to broad participation.",
    ))
    out.chart(C.xy(
        "Advance/decline line",
        [C.trace("A/D line (cumulative)", x, ad, color=3, width=1.4)],
        x=C.axis("Trading day", fmt="int"), y=C.axis("Advancing minus declining stocks, cumulative"),
        subtitle="If the index rises but the A/D line no longer does, only a few stocks are carrying the rally.",
    ))
    centers, counts = C.histogram(realised, bins=min(40, max(10, len(realised) // 10)), density=False)
    out.chart(C.xy(
        "Pairwise correlations of the simulated stocks",
        [C.trace("Stock pairs", centers, counts, type="bar", color=1)],
        x=C.axis("Correlation of daily returns", digits=2), y=C.axis("Number of pairs", fmt="int"),
        vlines=[C.vline(float(corr[iu].mean()), "Generator target"),
                C.vline(float(realised.mean()), "Realised mean", "ink", dash=False)],
        subtitle=f"{len(realised):,} pairs from {k} stocks. The higher the correlation, the more breadth and the "
                 "index move together – and the more likely breadth is to jump by 40 points.",
    ))

    c = np.concatenate([[0.0], np.cumsum(mkt["market"])])
    fwd = np.full(n_days, np.nan)
    fwd[:n_days - H] = np.expm1(c[1 + H:] - c[1:n_days + 1 - H])
    rows = []
    for name, idx in (("Breadth thrust (first signal day)", signals), ("Divergence day", np.flatnonzero(diverg)),
                      ("All days from day 50", np.arange(49, n_days))):
        f = fwd[idx]
        f = f[np.isfinite(f)]
        rows.append([name, int(len(idx)), float(np.mean(f)) if f.size else None, float(np.mean(f > 0)) if f.size else None,
                     float(np.min(f)) if f.size else None])
    out.table(f"Outcomes after the signals (index, next {H} trading days)",
              ["Signal", "Count", f"Mean return, {H}d", "Share positive", "Worst"], rows,
              formats=["text", "int", "pct", "pct", "pct"],
              note="On a simulated market without real investors, this shows how the rules react mechanically – "
                   "not their forecasting power in real markets.")
    if len(signals):
        trows = [[int(t), float(b50[t - 10]) / 100, float(b50[t]) / 100, float(magnitude[t]) / 100,
                  float(fwd[t]) if np.isfinite(fwd[t]) else None] for t in signals[:15]]
        out.table("Breadth-thrust signals", ["Trading day", "Breadth 10 days earlier", "Breadth", "Increase",
                                             f"Index, next {H}d"], trows,
                  formats=["int", "pct", "pct", "pct", "pct"],
                  note="Increase in percentage points of breadth. Signal days less than 10 days apart count as one "
                       "signal, shown from its first day.")
    else:
        out.note("No breadth thrust in this market. With a higher market correlation, breadth swings more "
                 "strongly; try 0.4–0.5 or a different seed.", "info")
