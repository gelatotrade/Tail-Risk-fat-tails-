"""The README's flagship: the 3D tail-risk surface with the active dataset's trajectory projected onto it."""

import numpy as np

import generate_tail_risk_surface as surface_script  # repo-root script; main() only runs under __main__
from src.physics.levy_flight import LevyStableDistribution, estimate_tail_index

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "tail-risk-surface",
    "title": "3D-Tail-Risk-Fläche: Wo steht der Markt?",
    "group": "surface",
    "order": 10,
    "summary": "Die Hauptgrafik des Frameworks, drehbar: Crash-Wahrscheinlichkeit über Volatilität und Tail-Index – "
               "mit der Bahn deines Datensatzes und der heutigen Position auf der Klippe.",
    "description": (
        "Die Fläche zeigt, wie wahrscheinlich **mindestens ein Tagesverlust über der Crash-Schwelle** (Standard 5 %) "
        "in den nächsten 30 Handelstagen ist – abhängig von der **annualisierten Volatilität** (x) und dem "
        "**Tail-Index α** (y; 2 = Gauß-Rand, kleiner = fettere Ränder). Gerechnet wird mit `crash_probability` aus "
        "`generate_tail_risk_surface.py`: dem Potenzgesetz-Rand einer Lévy-stabilen Verteilung, "
        "p(Tag) = sin(πα/2)·α·(Schwelle/σ_Tag)^(−α)/π, hochgerechnet auf den Horizont mit 1 − (1 − p)^30.\n\n"
        "Darauf liegt die **Bahn deines Datensatzes**: Für jeden Tag werden Volatilität und α aus den letzten N "
        "Renditen geschätzt und auf die Fläche gesetzt; die dunkle Linie sind die letzten Wochen, der dunkle Punkt "
        "ist heute. Die Konturen am Boden (5 / 15 / 30 / 50 %) sind die Frühwarnschwellen des README – jede "
        "Überschreitung verschärft die empfohlene Reaktion. Ziehen dreht die Fläche, das Mausrad zoomt, beim "
        "Überfahren der Bahn siehst du das Datum.\n\n"
        "Die **Crash-Schwelle** verschiebt die Klippe (3 %: breit und früh, 10 %: schmal und spät). Das **Fenster** "
        "bestimmt, wie schnell die Bahn reagiert. Der **Schätzer** wählt Hill (die größten Tagesbewegungen; Werte "
        "über 2 landen auf dem Gauß-Rand, wo die Fläche kein Crash-Risiko kennt) oder den Lévy-Quantilschätzer nach "
        "McCulloch. Die Kalibrierungstabelle zeigt, was nach Tagen in jeder Zone tatsächlich passiert ist."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 5.0, "min": 3.0, "max": 10.0, "step": 0.5,
         "label": "Crash-Schwelle (Tagesverlust)", "unit": "%",
         "help": "Ab welchem Tagesverlust ein Tag als Crash zählt."},
        {"name": "window", "type": "int", "default": 60, "min": 20, "max": 250, "step": 5,
         "label": "Fenster für Volatilität und α", "unit": "Tage"},
        {"name": "estimator", "type": "select", "default": "hill", "label": "Schätzer für α",
         "options": [{"value": "hill", "label": "Hill (Rand der größten Tagesbewegungen)"},
                     {"value": "stable", "label": "Lévy-stabil (McCulloch-Quantile)"}]},
        {"name": "horizon", "type": "int", "default": 30, "min": 5, "max": 60, "step": 5,
         "label": "Horizont", "unit": "Handelstage", "help": "Das README rechnet mit 30 Handelstagen."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Hervorgehobene letzte Tage", "unit": "Tage"},
        {"name": "script_paths", "type": "bool", "default": False,
         "label": "Skizzierte Krisenpfade des Skripts einblenden",
         "help": "COVID-19, Bärenmarkt 2022, Zoll-Crash 2025 – von Hand gesetzte Beispielpfade, keine Messwerte."},
    ],
    "covers": [
        "script:generate_tail_risk_surface.py",
        "src.physics.levy_flight.estimate_tail_index",
        "src.physics.levy_flight.LevyStableDistribution.fit",
    ],
    "cost": "fast",
}

# Zones of the README's reading table; the upper bound is the contour that ends the zone.
ZONES = [
    (0.05, "Ruhiges Plateau", "good", "grün", "Standardpositionierung"),
    (0.15, "Stress-Rampe", "warning", "gelb", "Absicherungen erhöhen"),
    (0.50, "Crash-Klippe", "serious", "orange-rot", "Exposure reduzieren, Tail-Hedges"),
    (np.inf, "Black-Swan-Zone", "critical", "violett", "Defensive Positionierung"),
]
# The floor contours of the README with the escalation of the script's reading guide.
CONTOURS = [
    (0.05, "Beobachten: der Markt verlässt das ruhige Plateau",
     "Absicherungen erhöhen – besonders, wenn der Schritt binnen weniger Wochen passiert"),
    (0.15, "Erhöht: Beginn der Crash-Klippe", "Exposure reduzieren, GEX und TDEX beobachten"),
    (0.30, "Kritisch: der rote Grat", "Defensive Positionierung, Tail-Hedges"),
    (0.50, "Black-Swan-Territorium", "Voll defensiv – jede weitere Vola-Spitze kann den Crash bringen"),
]
GRID = 46          # surface resolution (46 x 46 keeps the payload small and the mesh smooth)
MAX_EVAL = {"hill": 3000, "stable": 1500}  # rolling estimates per run (long series: every k-th day)
MAX_PATH = 420     # points of the drawn history path
Z_LIFT = 0.012     # draw paths slightly above the surface so they are not hidden by it


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def _zone(p):
    for i, (hi, name, status, region, action) in enumerate(ZONES):
        if p < hi:
            return i, name, status, region, action
    return len(ZONES) - 1, *ZONES[-1][1:]


def _alpha(window_returns, estimator):
    try:
        if estimator == "stable":
            return float(LevyStableDistribution.fit(window_returns).alpha)
        return float(estimate_tail_index(window_returns))
    except ValueError:  # (nearly) constant window
        return np.nan


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    w = int(params["window"])
    thr = float(params["threshold"]) / 100.0
    horizon = int(params["horizon"])
    estimator = params["estimator"]
    if n < w + 20:
        raise ToolError(f"Der Datensatz hat nur {n} Tage. Für ein Fenster von {w} Tagen brauchst du mindestens "
                        f"{w + 20} – verkleinere das Fenster.")

    # --- the surface: the script's own axes and risk model ------------------------------
    vol_axis, alpha_axis, vol_grid, alpha_grid, _ = surface_script._surface_grid(GRID)
    risk_grid = surface_script.crash_probability(vol_grid, alpha_grid, threshold=thr, horizon_days=horizon)
    v_lo, v_hi = float(vol_axis[0]), float(vol_axis[-1])
    a_lo, a_hi = float(alpha_axis[0]), float(alpha_axis[-1])

    # --- the dataset's trajectory: rolling volatility and tail index -------------------
    stride = max(1, int(np.ceil((n - w + 1) / MAX_EVAL[estimator])))
    ends = np.arange(w, n + 1, stride)
    if ends[-1] != n:
        ends = np.append(ends, n)
    win = np.lib.stride_tricks.sliding_window_view(r, w)[ends - w]
    vol = win.std(axis=1) * np.sqrt(252.0)
    alpha = np.array([_alpha(block, estimator) for block in win])
    ok = np.isfinite(alpha) & np.isfinite(vol)
    if ok.sum() < 2:
        raise ToolError("Für diesen Datensatz ließ sich kein Tail-Index schätzen (zu viele konstante Renditen).")
    ends, vol, alpha = ends[ok], vol[ok], alpha[ok]
    day = ends - 1                                    # index of the last return in each window
    vol_s = np.clip(vol, v_lo, v_hi)                  # position on the drawn surface
    alpha_s = np.clip(alpha, a_lo, a_hi)
    prob = surface_script.crash_probability(vol_s, alpha_s, threshold=thr, horizon_days=horizon)
    dates = [data.dates[i] for i in day] if data.dates else [f"Tag {i + 1}" for i in day]
    x_time = [data.dates[i] for i in day] if data.dates else list(day + 1)
    x_axis = C.axis("", type="date") if data.dates else C.axis("Handelstag", fmt="int")

    p_now, v_now, a_now, a_now_s = float(prob[-1]), float(vol[-1]), float(alpha[-1]), float(alpha_s[-1])
    zi, zname, zstatus, zregion, zaction = _zone(p_now)
    i_max = int(np.argmax(prob))
    at_edge = float(np.mean(alpha >= a_hi))

    # --- next contour: how far is the cliff? ------------------------------------------
    next_level = next((lvl for lvl, *_ in CONTOURS if lvl > p_now), None)
    vol_needed = alpha_needed = None
    if next_level is not None:
        vv = np.linspace(min(max(v_now, v_lo), v_hi), v_hi, 600)
        pv = surface_script.crash_probability(vv, np.full_like(vv, a_now_s), threshold=thr, horizon_days=horizon)
        hit = np.flatnonzero(pv >= next_level)
        vol_needed = float(vv[hit[0]]) if hit.size else None
        aa = np.linspace(a_now_s, a_lo, 600)
        pa = surface_script.crash_probability(np.full_like(aa, min(max(v_now, v_lo), v_hi)), aa,
                                              threshold=thr, horizon_days=horizon)
        hit = np.flatnonzero(pa >= next_level)
        alpha_needed = float(aa[hit[0]]) if hit.size else None

    # --- metrics ------------------------------------------------------------------------
    thr_txt = _pct(thr, 1 if thr * 1000 % 10 else 0)
    est_name = "Hill" if estimator == "hill" else "McCulloch"
    out.metric(f"Crash-Wahrscheinlichkeit ({horizon} T)", p_now, "pct", digits=1, status=zstatus, badge=zname,
               hint=f"P(mind. ein Tagesverlust > {thr_txt} in {horizon} Handelstagen) laut Fläche")
    out.metric("Volatilität p. a.", v_now, "pct", digits=1, hint=f"rollierend über {w} Tage")
    edge_hint = (f"{est_name}-Schätzer · auf der Fläche {_de(a_now_s)} (Gauß-Rand)" if a_now > a_hi else
                 f"{est_name}-Schätzer · auf der Fläche {_de(a_now_s)} (Achsenende)" if a_now < a_lo else
                 f"{est_name}-Schätzer über {w} Tage · kleiner = fettere Ränder")
    out.metric("Tail-Index α", a_now, "num", digits=2, hint=edge_hint)
    if next_level is not None:
        lvl_txt = f"{round(next_level * 100)}-%-Kontur"
        if alpha_needed is not None and alpha_needed < a_now_s - 1e-9:
            out.metric(f"α-Schwelle zur {lvl_txt}", alpha_needed, "num", digits=2,
                       hint=f"fällt α bei gleicher Vola unter diesen Wert, kreuzt der Markt die {lvl_txt}")
        if vol_needed is not None:
            out.metric(f"Vola-Schwelle zur {lvl_txt}", vol_needed, "pct", digits=0,
                       hint=f"bei gleichem α ({_de(a_now_s)})")
    out.metric("Höchster Wert der Historie", float(prob[i_max]), "pct", digits=1,
               hint=(f"am {_date_de(dates[i_max])} · Vola {_pct(vol[i_max])}, α {_de(alpha[i_max])}"
                     if prob[i_max] > 0.001 else "die Bahn verlässt das Plateau nie"))
    above15 = float(np.mean(prob >= 0.15))
    out.metric("Tage jenseits der 15-%-Kontur", above15, "pct", digits=1,
               hint="Anteil aller ausgewerteten Tage auf der Klippe oder darüber")

    # --- verdict ------------------------------------------------------------------------
    where = (f"Bei {_pct(v_now)} Volatilität und α = {_de(a_now_s)} zeigt die Fläche {_pct(p_now)} für "
             f"einen Tagesverlust über {thr_txt} in {horizon} Handelstagen.")
    if next_level is None:
        reach = " Das ist die höchste Zone der Fläche."
    else:
        parts = []
        if vol_needed is not None:
            parts.append(f"bei gleichem α ab {_pct(vol_needed, 0)} Volatilität")
        if alpha_needed is not None and alpha_needed < a_now_s - 1e-9:
            parts.append(f"bei gleicher Volatilität, wenn α auf {_de(alpha_needed)} fällt")
        contour = f"{round(next_level * 100)}-%-Kontur"
        reach = (f" Die {contour} erreicht der Markt " + " oder ".join(parts) + "."
                 if parts else f" Die {contour} ist von hier aus nur über Volatilität und α zugleich erreichbar.")
    out.set_verdict(zstatus, f"{zname}: {zaction}", where + reach)

    # --- chart 1: the surface with the trajectory -----------------------------------------
    keep = _thin(prob, MAX_PATH)
    k_recent = int(min(max(2, np.ceil(params["recent"] / stride)), len(prob)))
    paths = [C.path3d(f"Bahn {data.name or 'Datensatz'} (seit {_date_de(dates[0])})",
                      vol_s[keep], alpha_s[keep], prob[keep] + Z_LIFT, color=1,
                      text=[f"{_date_de(dates[i])} · {_pct(prob[i])}" for i in keep])]
    rec = slice(len(prob) - k_recent, len(prob))
    paths.append(C.path3d(f"Letzte {params['recent']} Tage", vol_s[rec], alpha_s[rec], prob[rec] + Z_LIFT,
                          color="ink", text=[f"{_date_de(d)} · {_pct(p)}" for d, p in zip(dates[rec], prob[rec])]))
    if params["script_paths"]:
        for k, (name, crisis) in enumerate(surface_script.CRISES.items()):
            cz = surface_script.crash_probability(crisis["vol"], crisis["alpha"], threshold=thr, horizon_days=horizon)
            paths.append(C.path3d(f"Skizze: {name}", crisis["vol"], crisis["alpha"], cz + Z_LIFT, color=3 + k,
                                  mode="lines+markers", size=3))
    points = [
        C.path3d("Heute: Lot zum Boden", [vol_s[-1], vol_s[-1]], [alpha_s[-1], alpha_s[-1]], [0.0, p_now + Z_LIFT],
                 color="muted"),
        C.path3d(f"Heute ({_date_de(dates[-1])})", [vol_s[-1]], [alpha_s[-1]], [p_now + Z_LIFT], color="ink",
                 mode="markers", size=8, text=[f"{zname} · {_pct(p_now)}"]),
    ]
    out.chart(C.surface(
        f"Crash-Wahrscheinlichkeit über Volatilität und Tail-Index (Schwelle {thr_txt}, {horizon} Tage)",
        vol_axis, alpha_axis, risk_grid,
        # both axes reversed: the calm corner (low vol, α = 2) faces the viewer, the cliff rises behind it
        x=C.axis("Volatilität p. a.", fmt="pct", digits=0, range=(v_hi, v_lo)),
        y=C.axis("Tail-Index α", range=(a_hi, a_lo)),
        z=C.axis(f"P(Verlust > {thr_txt})", fmt="pct", digits=0, range=(0, 1.05)),
        paths=paths, points=points, scale="risk", height=600,
        subtitle=f"Grünes Plateau → gelbe Rampe → rote Klippe → violette Black-Swan-Spitze. Blau: die Bahn "
                 f"({est_name}-α und Vola über {w} Tage), dunkel: die letzten Tage und heute.",
    ))

    # --- chart 2: crash probability over time, with the real crash days ----------------
    loss = -np.expm1(r)
    events = np.flatnonzero(loss > thr)
    prev_prob = np.interp(events - 1, day, prob, left=np.nan)       # what the surface showed the day before
    ev_ok = np.isfinite(prev_prob) & (events - 1 >= day[0])
    ev_x = [data.dates[i] for i in events[ev_ok]] if data.dates else list(events[ev_ok] + 1)
    traces = [C.trace("Crash-Wahrscheinlichkeit", x_time, prob, color=1, width=1.5)]
    if ev_ok.any():
        traces.append(C.trace(f"Tag mit Verlust > {thr_txt} (Wert am Vortag)", ev_x, prev_prob[ev_ok],
                              type="scatter", color="critical", size=7, decimate_to=None))
    out.chart(C.xy(
        f"Crash-Wahrscheinlichkeit im Zeitverlauf ({horizon} Tage)",
        traces, x=x_axis, y=C.axis("Wahrscheinlichkeit", fmt="pct", digits=0, range=(0, 1.02)),
        hlines=[C.hline(lvl, _pct(lvl, 0), color="muted") for lvl, *_ in CONTOURS], height=390,
        subtitle="Rote Punkte: tatsächliche Crash-Tage, eingezeichnet mit dem Wert, den die Fläche am Vortag zeigte.",
    ))

    # --- charts 3 + 4: the two coordinates --------------------------------------------
    out.chart(C.xy(
        f"Volatilität p. a. (rollierend, {w} Tage)",
        [C.trace("Volatilität", x_time, vol, color=1, width=1.5)],
        x=x_axis, y=C.axis("Volatilität p. a.", fmt="pct", digits=0), height=280,
        hlines=[C.hline(v_hi, "Ende der Flächenachse")] if np.nanmax(vol) > v_hi else [],
    ))
    a_top = float(min(8.0, max(2.6, np.nanpercentile(alpha, 99) * 1.08)))
    out.chart(C.xy(
        f"Tail-Index α ({est_name}, {w} Tage)",
        [C.trace("α", x_time, alpha, color=1, width=1.5)],
        x=x_axis, y=C.axis("α", range=(min(0.8, float(np.nanmin(alpha)) * 0.95), a_top)), height=280,
        hlines=[C.hline(2.0, "Gauß-Rand α = 2"), C.hline(a_lo, "Achsenende")],
        subtitle=(f"An {_pct(at_edge, 0)} der Tage liegt α über 2 – dort zeigt die Fläche 0 %."
                  if at_edge > 0.05 else "Kleiner = fettere Ränder."),
    ))

    # --- tables ---------------------------------------------------------------------
    rows = []
    for lvl, signal, action in CONTOURS:
        above = prob >= lvl
        last = _date_de(dates[int(np.flatnonzero(above)[-1])]) if above.any() else "nie"
        rows.append([_pct(lvl, 0), signal, action, "ja" if p_now >= lvl else "nein", float(above.mean()), last])
    out.table("So liest du die Konturen am Boden", ["Kontur", "Signal", "Reaktion", "heute überschritten",
                                                   "Anteil Tage darüber", "zuletzt darüber"],
              rows, formats=["text", "text", "text", "text", "pct", "text"],
              note="Leseanleitung aus README und Skript: Jede Überschreitung einer Kontur verschärft die Reaktion.")

    # calibration: what really happened in the `horizon` days after each evaluated day
    csum = np.concatenate([[0], np.cumsum(loss > thr)])
    has_future = day + horizon < n
    realized = (csum[np.minimum(day + 1 + horizon, n)] - csum[day + 1]) > 0
    zone_idx = np.array([_zone(p)[0] for p in prob])
    rows = []
    for i, (hi, name, status, region, action) in enumerate(ZONES):
        lo = ZONES[i - 1][0] if i else 0.0
        band = f"{_pct(lo, 0)} – {_pct(hi, 0)}" if np.isfinite(hi) else f"≥ {_pct(lo, 0)}"
        mk = zone_idx == i
        mf = mk & has_future
        rows.append([f"{name} ({region})", band, action, float(mk.mean()),
                     float(prob[mf].mean()) if mf.any() else None,
                     float(realized[mf].mean()) if mf.any() else None, int(mf.sum())])
    out.table("Zonen der Fläche – und was danach wirklich passierte", [
        "Zone", "Bereich", "Handlung (README)", "Anteil Tage",
        "Ø Fläche", f"tatsächlich Verlust > {thr_txt} in {horizon} T", "Tage"],
        rows, formats=["text", "text", "text", "pct", "pct", "pct", "int"],
        note=f"„tatsächlich“: Anteil der Tage, nach denen innerhalb von {horizon} Handelstagen wirklich ein Tagesverlust "
             f"über {thr_txt} folgte. Liegt er weit unter „Ø Fläche“, überschätzt das Modell, liegt er darüber, "
             "unterschätzt es. Überlappende Zeiträume: die Tage sind nicht unabhängig.")

    # --- notes ------------------------------------------------------------------------
    pred, real = float(prob[has_future].mean()) if has_future.any() else np.nan, \
        float(realized[has_future].mean()) if has_future.any() else np.nan
    if estimator == "hill" and at_edge > 0.05:
        out.note(f"An **{_pct(at_edge, 0)}** der Tage liegt der Hill-Schätzer über 2 – endliche Varianz, typisch für "
                 "Aktienindizes. Diese Tage sitzen auf dem Gauß-Rand α = 2, wo der Potenzgesetz-Term der Fläche "
                 "verschwindet (0 %), selbst bei hoher Volatilität. Auf die Fläche steigt die Bahn erst, wenn einzelne "
                 "Tage im Fenster weit aus der Reihe fallen – dann aber steil.")
    if estimator == "stable":
        out.note("Der McCulloch-Schätzer liest α aus Quantilverhältnissen. Bei Tagesrenditen landet er fast immer "
                 "unter 2, die Bahn klebt dann an der Klippe. Vergleiche in der Tabelle „Ø Fläche“ mit der "
                 "tatsächlichen Häufigkeit.", "warning")
    if np.isfinite(pred) and np.isfinite(real):
        ratio = pred / real if real > 0 else np.inf
        verdict = ("überschätzt die Häufigkeit deutlich" if ratio > 2 else
                   "unterschätzt die Häufigkeit deutlich" if ratio < 0.5 else "liegt in der richtigen Größenordnung")
        out.note(f"Kalibrierung über die ganze Historie: Im Mittel zeigt die Fläche **{_pct(pred)}**, tatsächlich "
                 f"folgte in **{_pct(real)}** der Fälle ein Tagesverlust über {thr_txt} innerhalb von {horizon} "
                 f"Tagen – das Modell {verdict}. Die Fläche ist eine Lesehilfe für die Richtung "
                 "(Plateau → Klippe), keine kalibrierte Prognose. Werte außerhalb der Achsen "
                 f"(Vola {_pct(v_lo, 0)}–{_pct(v_hi, 0)}, α {_de(a_lo)}–{_de(a_hi)}) sitzen auf dem Rand."
                 + (f" Lange Reihe: ausgewertet wird jeder {stride}. Tag." if stride > 1 else ""))


def _thin(prob, max_points):
    """Indices of a thinned path that keeps every bucket's peak (crises must not be thinned away)."""
    n = len(prob)
    if n <= max_points:
        return np.arange(n)
    idx, _ = C.decimate(list(range(n)), prob, max_points)
    idx = np.asarray(idx, dtype=int)
    return np.unique(np.concatenate([idx, [0, n - 1]]))


def _date_de(d):
    s = str(d)
    if len(s) == 10 and s[4] == "-":
        return f"{s[8:10]}.{s[5:7]}.{s[:4]}"
    return s
