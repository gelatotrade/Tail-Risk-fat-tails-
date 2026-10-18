"""VIX sentiment: regime, percentile, spikes and mean reversion over time, plus the 3D sentiment phase space."""

import numpy as np

from src.analysis.sentiment import VIXAnalyzer, compute_sentiment_coordinates
from src.utils.data_loader import generate_vix_from_returns

from .. import charts as C
from ..data import index_axis, fmt_date
from ._ews_common import ordinal

SPEC = {
    "id": "sentiment",
    "title": "VIX sentiment: fear, spikes and reversals",
    "group": "warning",
    "order": 30,
    "summary": "How much fear the VIX is pricing: regime, percentile, spikes and the mean-reversion signal over "
               "time, and what prices did next.",
    "description": (
        "The VIX measures the 30-day expected volatility of the S&P 500 implied by option prices – the "
        "best-known fear gauge. The framework's `VIXAnalyzer` assigns it to a regime (below 15 complacent, up to "
        "25 normal, up to 35 elevated, up to 50 high fear, above that panic), measures its **percentile** over "
        "the past year, detects **spikes** (a daily rise larger than the threshold × the typical daily change "
        "over the previous 19 days) and gives a **mean-reversion signal**: a z-score above 2 points to an "
        "imminent reversal of fear, one below −1.5 to complacency.\n\n"
        "On each day, the regime, spikes and mean-reversion signal use only the VIX up to that day. The tables "
        "check what prices did over the 20 trading days after a regime or spike. The 3D phase space "
        "(`compute_sentiment_coordinates`) shows VIX level, VIX momentum and the correlation between returns "
        "and VIX changes, each normalised to 0–1 using the minimum and maximum of the full series.\n\n"
        "If the dataset contains no VIX, the framework derives one from the returns "
        "(`generate_vix_from_returns`); the values are then illustrative and no assessment is given."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1,
         "label": "Spike threshold (z)", "help": "A spike is a daily rise of more than z typical daily changes."},
        {"name": "pct_window", "type": "int", "default": 252, "min": 20, "max": 1260, "step": 21,
         "label": "Percentile window", "unit": "days", "help": "252 trading days ≈ one year."},
        {"name": "coord_window", "type": "int", "default": 20, "min": 5, "max": 120, "step": 5,
         "label": "Phase-space window", "unit": "days", "help": "For VIX momentum and the return–VIX correlation."},
        {"name": "lookback_3d", "type": "int", "default": 1000, "min": 100, "max": 20000, "step": 100,
         "label": "3D view: latest trading days", "unit": "days"},
    ],
    "covers": [
        "src.analysis.sentiment.VIXAnalyzer",
        "src.analysis.sentiment.VIXAnalyzer.add_data",
        "src.analysis.sentiment.VIXAnalyzer.current_regime",
        "src.analysis.sentiment.VIXAnalyzer.vix_percentile",
        "src.analysis.sentiment.VIXAnalyzer.vix_spike_detection",
        "src.analysis.sentiment.VIXAnalyzer.mean_reversion_signal",
        "src.analysis.sentiment.compute_sentiment_coordinates",
        "src.utils.data_loader.generate_vix_from_returns",
    ],
    "cost": "medium",
}

REGIME = {  # code -> (label, status colour)
    "COMPLACENT": ("Complacent", "good"),
    "NORMAL": ("Normal", "good"),
    "ELEVATED": ("Elevated", "warning"),
    "HIGH_FEAR": ("High fear", "serious"),
    "PANIC": ("Panic", "critical"),
}
SIGNAL = {"HIGH_FEAR_REVERSAL_LIKELY": "Fear reversal likely", "COMPLACENCY_WARNING": "Complacency warning",
          "NEUTRAL": "Neutral"}
H = 20  # forward window of the outcome tables (trading days)


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    x, xa = index_axis(data)
    label = (lambda i: fmt_date(x[i])) if data.dates else (lambda i: f"Day {i}")
    # synthetic and crisis datasets carry a VIX that data.py derived from the returns (generate_vix_from_returns)
    derived = data.meta.get("source") in ("synthetic", "crisis")
    real = data.vix is not None and not derived
    vix = (np.asarray(data.vix, dtype=float) if data.vix is not None
           else np.asarray(generate_vix_from_returns(r), dtype=float))

    an = VIXAnalyzer()
    an.add_data(vix)
    th = an.thresholds
    regime = an.current_regime()
    pct = an.vix_percentile(int(params["pct_window"]))
    spike = an.vix_spike_detection(float(params["threshold"]))
    mr = an.mean_reversion_signal()

    # the same methods day by day, each on the VIX known at that day
    out.progress("Evaluating each day…", 0.2)
    regimes, z_spike, z_mr = [], np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        a = VIXAnalyzer(vix[:t + 1])
        regimes.append(a.current_regime())
        s = a.vix_spike_detection(float(params["threshold"]))
        if "z_score" in s:
            z_spike[t] = s["z_score"]
        m = a.mean_reversion_signal()
        if "z_score" in m:
            z_mr[t] = m["z_score"]
    regimes = np.array(regimes)
    spikes = np.flatnonzero(np.nan_to_num(z_spike) > params["threshold"])

    reg_txt, reg_status = REGIME[regime]
    out.metric("Latest VIX", vix[-1], "num", digits=1, hint=f"{reg_txt} regime" + ("" if real else " · derived"),
               status=reg_status if real else None)
    out.metric("VIX percentile", pct / 100, "pct", digits=0, hint=f"over a {int(params['pct_window']):,}-day window")
    out.metric("Spike z today", spike.get("z_score"), "num", digits=1,
               hint="Spike detected" if spike.get("spike_detected") else f"Spike threshold: z > {params['threshold']:.1f}",
               status="serious" if spike.get("spike_detected") else None)
    out.metric("Mean-reversion z", mr.get("z_score"), "num", digits=2, hint=SIGNAL.get(mr["signal"], mr["signal"]),
               status="warning" if mr["signal"] != "NEUTRAL" else None)
    last_year = spikes[spikes >= n - 252]
    out.metric("Spikes in the past year", len(last_year), "int", hint=f"{len(spikes):,} in total over {n:,} days")

    if real:
        if regime == "PANIC":
            level, title = "critical", "Panic in the VIX"
        elif regime == "HIGH_FEAR":
            level, title = "serious", "High fear in the VIX"
        elif regime == "ELEVATED" or spike.get("spike_detected"):
            level, title = "warning", "Elevated fear" if regime == "ELEVATED" else "VIX spike today"
        elif mr["signal"] == "COMPLACENCY_WARNING":
            level, title = "warning", "Unusually low fear – complacency"
        else:
            level, title = "good", "Calm VIX regime" if regime == "NORMAL" else "Low fear (complacent)"
        out.set_verdict(level, title,
                        f"VIX {vix[-1]:.1f} in the ‘{reg_txt}’ regime, {ordinal(pct)} percentile of the latest "
                        f"window; mean reversion: {SIGNAL.get(mr['signal'], mr['signal']).lower()}.")
    else:
        out.note(("The VIX of this synthetic dataset is" if derived else
                  "The dataset contains no VIX, so one is")
                 + " derived here from realised volatility (`generate_vix_from_returns`, level 15 + 100 × "
                 "annualised volatility). The values are illustrative and usually higher than a real VIX, so no "
                 "assessment is given.", "warning")

    # ---- charts ---------------------------------------------------------------------------------------
    out.chart(C.xy(
        "VIX with regime boundaries and spikes",
        [C.trace("VIX" if real else "VIX (derived from returns)", x, vix, color=1, width=1.2),
         C.trace("Spike", [x[i] for i in spikes], vix[spikes], type="scatter", color="critical", size=5,
                 decimate_to=None)],
        x=xa, y=C.axis("VIX points"),
        hlines=[C.hline(th["complacent"], "Complacent < 15"), C.hline(th["normal"], "Elevated from 25", "warning"),
                C.hline(th["elevated"], "High fear from 35", "serious"), C.hline(th["panic"], "Panic from 50", "critical")],
        subtitle="Red dots: days on which the VIX rose by more than the spike threshold allows.",
    ))
    out.chart(C.xy(
        "Mean-reversion signal (VIX z-score)",
        [C.trace("z-score", x, z_mr, color=2, width=1.2)],
        x=xa, y=C.axis("z (vs 252-day mean)"),
        hlines=[C.hline(2.0, "Reversal likely (z > 2)", "serious"),
                C.hline(-1.5, "Complacency (z < −1.5)", "warning")],
        subtitle="Above 2: fear is extremely high, historically often close to the market low. Below −1.5: "
                 "unusually calm.",
    ))

    out.progress("Phase space…", 0.7)
    X, Y, Z = compute_sentiment_coordinates(vix, r, window=int(params["coord_window"]))
    w = int(params["coord_window"])
    start = max(w, n - int(params["lookback_3d"]))
    idx = np.arange(start, n)
    traces = []
    for code in ("COMPLACENT", "NORMAL", "ELEVATED", "HIGH_FEAR", "PANIC"):
        m = idx[regimes[idx] == code]
        if m.size:
            traces.append(C.path3d(REGIME[code][0], X[m], Y[m], Z[m], color=REGIME[code][1] if code != "COMPLACENT"
                                   else 1, mode="markers", size=2.5,
                                   text=[str(label(i)) for i in m]))
    traces.append(C.path3d("Latest day", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers", size=8))
    out.chart(C.scatter3d(
        "Sentiment phase space", traces,
        x=C.axis("VIX level (0–1)"), y=C.axis("VIX momentum (0–1)"), z=C.axis("Return–VIX correlation (0–1)"),
        subtitle=f"Last {len(idx):,} days, colour = VIX regime. Panic: high level and rising VIX. Low "
                 "correlation = returns and the VIX move in opposite directions (the normal case).",
    ))

    # ---- what happened next? ----------------------------------------------------------------------------
    c = np.concatenate([[0.0], np.cumsum(r)])
    fwd = np.full(n, np.nan)
    if n > H:
        fwd[:n - H] = np.expm1(c[1 + H:] - c[1:n + 1 - H])
    rows = []
    bounds = {"COMPLACENT": f"< {th['complacent']}", "NORMAL": f"{th['complacent']}–{th['normal']}",
              "ELEVATED": f"{th['normal']}–{th['elevated']}", "HIGH_FEAR": f"{th['elevated']}–{th['panic']}",
              "PANIC": f"≥ {th['panic']}"}
    for code in ("COMPLACENT", "NORMAL", "ELEVATED", "HIGH_FEAR", "PANIC"):
        m = regimes == code
        if not m.any():
            continue
        f = fwd[m][np.isfinite(fwd[m])]
        rows.append([REGIME[code][0], bounds[code], float(m.mean()), float(np.std(r[m]) * np.sqrt(252)),
                     float(np.mean(f)) if f.size else None, float(np.mean(f < 0)) if f.size else None])
    fa = fwd[np.isfinite(fwd)]
    rows.append(["All days", "", 1.0, float(np.std(r) * np.sqrt(252)), float(np.mean(fa)) if fa.size else None,
                 float(np.mean(fa < 0)) if fa.size else None])
    out.table("VIX regimes and what followed",
              ["Regime", "VIX", "Share of days", "Volatility p.a. in regime", f"Mean return, next {H}d",
               f"Share with a loss after {H}d"], rows, formats=["text", "text", "pct", "pct", "pct", "pct"],
              note="The forward return starts on the day after the classification. Historically, high fear has "
                   "often preceded above-average returns – the idea behind the mean-reversion signal.")

    if spikes.size:
        top = spikes[np.argsort(z_spike[spikes])[::-1][:10]]
        srows = [[label(i), float(vix[i]), float(vix[i] - vix[i - 1]), float(z_spike[i]), float(r[i]),
                  float(fwd[i]) if np.isfinite(fwd[i]) else None] for i in sorted(top)]
        out.table(f"The {len(top)} largest VIX spikes (chronological)",
                  ["Date", "VIX", "Rise (points)", "z", "Return on the day", f"Return, next {H}d"], srows,
                  formats=["text", "num", "num", "num", "pct", "pct"],
                  note="z = rise ÷ standard deviation of daily VIX changes over the previous 19 days.")
