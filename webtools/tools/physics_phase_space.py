"""Three physics phase spaces (Lévy, Tsallis, Fokker-Planck) as interactive 3D trajectories of market history."""

import numpy as np

from src.physics.fokker_planck import compute_fokker_planck_coordinates
from src.physics.levy_flight import levy_flight_3d_coordinates
from src.physics.tsallis_statistics import compute_tsallis_coordinates

from .. import charts as C
from ..data import fmt_date
from ..core import ToolError

SPEC = {
    "id": "physics-phase-space",
    "title": "Physics phase spaces in 3D",
    "group": "surface",
    "order": 30,
    "summary": "Turns your returns into three 3D trajectories – Lévy, Tsallis and Fokker-Planck space – and shows "
               "where the market stands today relative to its own history.",
    "description": (
        "Each trading day becomes a point in three different ‘physics spaces’, computed from a rolling window of "
        "the preceding returns. **Lévy space** (`levy_flight_3d_coordinates`): X = volatility relative to its "
        "average (1 = normal), Y = 2/Hill tail index (higher = fatter tails), Z = today's |return| in standard "
        "deviations – crash days stand out as spikes. **Tsallis space** (`compute_tsallis_coordinates`): "
        "X = entropic index q, Y = Tsallis entropy S_q, Z = temperature 1/β (rises with the variance). "
        "**Fokker-Planck space** (`compute_fokker_planck_coordinates`): X = drift (mean daily return), "
        "Y = diffusion (variance of daily returns), Z = order α̂ = 6/kurtosis using ordinary kurtosis "
        "(Gaussian: 3, so α̂ = 2), clipped to [1, 2] (lower = fatter tails).\n\n"
        "How to read the charts: grey dots are the past, the coloured line covers the most recent days and the red "
        "dot is today. Calm phases form a dense cloud; crises stretch out in long excursions. Drag to rotate, "
        "scroll to zoom and hover to see the date. The table ranks each of today's coordinates as a percentile of "
        "its own history."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 30, "min": 20, "max": 120, "step": 5,
         "label": "Window (trading days)", "help": "The same for all three spaces."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Highlighted recent days"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 15000, "step": 250,
         "label": "Period (most recent trading days)",
         "help": "Limits computing time and data volume for very long series."},
    ],
    "covers": [
        "src.physics.levy_flight.levy_flight_3d_coordinates",
        "src.physics.tsallis_statistics.compute_tsallis_coordinates",
        "src.physics.fokker_planck.compute_fokker_planck_coordinates",
    ],
    "cost": "medium",
}


def _sig(a, digits=4):
    """Round to significant digits: keeps the JSON payload of three 3D clouds small."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.isfinite(a) & (a != 0), np.floor(np.log10(np.abs(a))), 0)
    scale = 10.0 ** (digits - 1 - mag)
    return np.round(a * scale) / scale


def _num(x, digits=2):
    return f"{x:.{digits}f}"


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")



def _show(v, fmt):
    """Number text: daily drift in %, everything else with 3 significant digits."""
    if not np.isfinite(v):
        return "–"
    if fmt == "pct":
        return f"{100 * v:.3f}%".replace("-", "−")
    txt = f"{v:.3g}"
    if "e" in txt:
        mant, exp = txt.split("e")
        return mant.replace("-", "−") + "·10" + str(int(exp)).translate(_SUP)
    return txt.replace("-", "−")


def _space(title, coords, labels, fmts, days, k_recent, subtitle):
    X, Y, Z = (_sig(c) for c in coords)
    hist = slice(0, max(0, len(X) - k_recent))
    rec = slice(max(0, len(X) - k_recent), len(X))
    traces = [
        C.path3d("History", X[hist], Y[hist], Z[hist], color="muted", mode="markers", size=1.5, text=days[hist]),
        C.path3d(f"Last {k_recent} days", X[rec], Y[rec], Z[rec], color=1, mode="lines+markers", size=3,
                 text=days[rec]),
        C.path3d("Today", [X[-1]], [Y[-1]], [Z[-1]], color="critical", mode="markers", size=11, text=[days[-1]]),
    ]
    return C.scatter3d(title, traces, x=C.axis(labels[0], fmt=fmts[0]), y=C.axis(labels[1], fmt=fmts[1]),
                       z=C.axis(labels[2], fmt=fmts[2]), subtitle=subtitle, height=520)


def run(params, data, out):
    w, k_recent = params["window"], params["recent"]
    m = min(params["last_n"], data.n)
    if m < w + 30:
        raise ToolError(f"The series is too short for a window of {w} days ({m} days). Reduce the window.")
    r = np.asarray(data.returns[-m:], dtype=float)
    days_all = [fmt_date(d) for d in data.dates[-m:]] if data.dates else [f"Day {i + 1}" for i in range(data.n - m, data.n)]

    out.progress("Lévy coordinates …", 0.05)
    lev = levy_flight_3d_coordinates(r, window=w)
    out.progress("Tsallis coordinates (q fit per window) …", 0.25)
    tsa = compute_tsallis_coordinates(r, window=w)
    out.progress("Fokker-Planck coordinates …", 0.75)
    fpc = compute_fokker_planck_coordinates(r, window=w)

    # the first `window` points are placeholders (zeros) in all three transforms
    keep = slice(w, m)
    lev = [np.asarray(c, dtype=float)[keep] for c in lev]
    tsa = [np.asarray(c, dtype=float)[keep] for c in tsa]
    fpc = [np.asarray(c, dtype=float)[keep] for c in fpc]
    days = np.array(days_all[w:], dtype=object)
    k_recent = int(min(k_recent, len(days)))

    out.chart(_space("Lévy phase space", lev, ("X: volatility (mean = 1)", "Y: 2 / tail index", "Z: |return| / σ"),
                     ("num", "num", "num"), days, k_recent,
                     "Spikes in Z are single crash or rally days; in stress phases the trajectory moves to the right."))
    out.chart(_space("Tsallis phase space", tsa, ("X: q", "Y: entropy S_q", "Z: temperature 1/β"),
                     ("num", "num", "num"), days, k_recent,
                     "The temperature 1/β rises with the variance; q > 1.3 indicates fat tails in the window."))
    out.chart(_space("Fokker-Planck phase space", fpc, ("X: drift (avg. daily return)", "Y: diffusion (variance)",
                                                        "Z: order α̂ = 6/kurtosis"),
                     ("pct", "num", "num"), days, k_recent,
                     "α̂ = 2: Gaussian window, α̂ → 1: fat tails. Crises: negative drift, high diffusion."))

    spaces = [
        ("Lévy", "Volatility X", lev[0], "num"), ("Lévy", "2 / tail index Y", lev[1], "num"),
        ("Lévy", "|Return| / σ Z", lev[2], "num"),
        ("Tsallis", "q", tsa[0], "num"), ("Tsallis", "Entropy S_q", tsa[1], "num"),
        ("Tsallis", "Temperature 1/β", tsa[2], "num"),
        ("Fokker-Planck", "Drift", fpc[0], "pct"), ("Fokker-Planck", "Diffusion (variance)", fpc[1], "num"),
        ("Fokker-Planck", "Order α̂", fpc[2], "num"),
    ]
    rows = []
    for space, name, series, fmt in spaces:
        s = series[np.isfinite(series)]
        now = float(series[-1])
        pct = float(0.5 * (np.mean(s < now) + np.mean(s <= now))) if s.size else None  # mid-rank (ties)
        place = "upper decile" if pct is not None and pct >= 0.9 else "lower decile" if pct is not None and pct <= 0.1 else "typical"
        rows.append([space, name, _show(now, fmt), _show(float(np.median(s)), fmt), pct, place])
    out.table("Today vs its own history", ["Space", "Coordinate", "Today", "Median", "Percentile today",
                                            "Position"],
              rows, formats=["text", "text", "text", "text", "pct", "text"],
              note=f"Percentile = share of the {len(days):,} days with a lower value (ties count as half).")

    vol_pct = rows[0][4]
    out.metric("Today" if data.dates else "Last day", str(days[-1]), "text",
               hint=f"window {w} days, {len(days):,} points per space")
    out.metric("Lévy volatility X", float(lev[0][-1]), "num", hint=f"percentile rank {_num(100 * vol_pct, 0)}%; 1 = average",
               status="serious" if vol_pct >= 0.95 else "warning" if vol_pct >= 0.8 else None)
    out.metric("Tail heaviness 2/α (Lévy Y)", float(lev[1][-1]), "num", hint=f"percentile rank {_num(100 * rows[1][4], 0)}%")
    out.metric("Tsallis q", float(tsa[0][-1]), "num", hint=f"percentile rank {_num(100 * rows[3][4], 0)}%")
    out.metric("Fokker-Planck α̂", float(fpc[2][-1]), "num",
               hint=f"percentile rank {_num(100 * rows[8][4], 0)}% · 2 = Gaussian")
    j = int(np.nanargmax(lev[2]))
    out.metric("Largest jump (Lévy Z)", float(lev[2][j]), "num", hint=f"{days[j]}: |return| in σ")
    if m < data.n:
        out.note(f"Only the last {m:,} of {data.n:,} trading days are evaluated (‘Period’ parameter).", "info")
