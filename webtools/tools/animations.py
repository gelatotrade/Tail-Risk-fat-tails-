"""Animationen: the framework's four FuncAnimations, rendered frame by frame to a GIF in the browser."""

import os
import tempfile

import numpy as np

from src.analysis.early_warning import CompositeEarlyWarningSystem
from src.models.risk_metrics import RollingRiskMetrics
from src.physics.levy_flight import levy_flight_3d_coordinates
from src.utils.data_loader import generate_vix_from_returns
from src.visualization.animated_risk import (
    DistributionMorphAnimator,
    PhaseSpaceAnimator,
    RiskGaugeAnimator,
    create_comprehensive_animation,
)

from .. import charts as C
from ..core import ToolError
from ..data import index_axis
from ._ews_common import LEVEL, de

SPEC = {
    "id": "animations",
    "title": "Animationen: Tail-Risiko in Bewegung",
    "group": "dashboard",
    "order": 30,
    "summary": "Die vier Animationen des Frameworks als GIF – Lévy-Phasenraum, Verteilungs-Morphing, "
               "Risiko-Tacho und ein Dashboard mit vier Ansichten, direkt im Browser gerendert.",
    "description": (
        "Ein Standbild zeigt einen Zustand, eine Animation zeigt den Weg dorthin. Das Framework bringt vier "
        "Animationen mit: Der **Lévy-Phasenraum** zeichnet die Bahn des Marktes aus Volatilität, Tail-Schwere "
        "(2/α) und Sprungintensität (|r|/σ) – ruhige Phasen kreisen eng, Krisen schlagen weit aus. Das "
        "**Verteilungs-Morphing** zeigt die Renditeverteilung im rollierenden 60-Tage-Fenster gegen die "
        "Gauß-Kurve; auf der Log-Skala siehst du, wann fette Ränder entstehen. Der **Risiko-Tacho** lässt die "
        "Nadel über den Komposit-Score des Frühwarnsystems wandern. Das **Dashboard** synchronisiert Phasenraum, "
        "Verteilung, Sprungrisiko und den rollierenden 99-%-VaR.\n\n"
        "Gerendert wird mit Matplotlib im Browser, Bild für Bild. Die Original-Animation hätte Hunderte Bilder "
        "(eines pro Tag oder alle paar Tage). Das Werkzeug zeigt davon gleichmäßig verteilte Bilder, bei „Mittel“ "
        "etwa 10–15 Sekunden Rechenzeit. Die Zeitachse in den Bildtiteln (t) zählt Handelstage ab Beginn des "
        "gewählten Zeitraums.\n\n"
        "Die Diagramme oberhalb des GIFs zeigen dieselben Größen interaktiv. Die Punkte markieren die Tage, die "
        "im GIF als Bild erscheinen. Der Tacho nutzt den modellierten Frühwarn-Score (kausal gerechnet). Nur "
        "dort spielt der **Seed** eine Rolle."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "kind", "type": "select", "default": "phase", "label": "Animation",
         "options": [{"value": "phase", "label": "Lévy-Phasenraum (3D-Bahn)"},
                     {"value": "morph", "label": "Verteilungs-Morphing (fette Ränder entstehen)"},
                     {"value": "gauge", "label": "Risiko-Tacho (Frühwarn-Score)"},
                     {"value": "dashboard", "label": "Dashboard mit vier Ansichten"}]},
        {"name": "length", "type": "select", "default": "medium", "label": "Länge",
         "options": [{"value": "short", "label": "Kurz – halb so viele Bilder, schnell"},
                     {"value": "medium", "label": "Mittel"},
                     {"value": "long", "label": "Lang – mehr Bilder, dauert etwa doppelt so lange"}]},
        {"name": "days", "type": "int", "default": 500, "min": 120, "max": 2520, "step": 20,
         "label": "Zeitraum", "unit": "Tage", "help": "Die letzten Handelstage des Datensatzes, die animiert werden."},
        {"name": "dpi", "type": "int", "default": 75, "min": 50, "max": 100, "step": 5, "label": "Auflösung",
         "unit": "dpi", "help": "Höher = schärfer, aber größere Datei und längere Rechenzeit."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1,
         "label": "Zufalls-Seed (nur Risiko-Tacho)", "show_if": {"kind": ["gauge"]},
         "help": "Modellrauschen der Frühwarn-Indikatoren. Gleicher Seed = gleicher Score."},
    ],
    "covers": [
        "src.visualization.animated_risk.PhaseSpaceAnimator",
        "src.visualization.animated_risk.PhaseSpaceAnimator.create_animation",
        "src.visualization.animated_risk.DistributionMorphAnimator",
        "src.visualization.animated_risk.DistributionMorphAnimator.create_animation",
        "src.visualization.animated_risk.RiskGaugeAnimator",
        "src.visualization.animated_risk.RiskGaugeAnimator.create_animation",
        "src.visualization.animated_risk.create_comprehensive_animation",
        "src.physics.levy_flight.levy_flight_3d_coordinates",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_var",
        "src.models.risk_metrics.RollingRiskMetrics.rolling_kurtosis",
        "src.analysis.early_warning.CompositeEarlyWarningSystem.simulate_all_indicators",
    ],
    "cost": "medium",
}

# size (inches; the framework's figures are 10-16 in wide, too small to read as a web GIF), fps and frame counts
# (browser rendering costs about 0.5 s per phase frame, 1.3 s per morph frame and 3 s per dashboard frame)
KINDS = {
    "phase": {"name": "Lévy-Phasenraum", "src": "PhaseSpaceAnimator", "size": (9.0, 7.0), "fps": 6,
              "frames": {"short": 11, "medium": 22, "long": 44}},
    "morph": {"name": "Verteilungs-Morphing", "src": "DistributionMorphAnimator", "size": (11.0, 4.6), "fps": 2,
              "frames": {"short": 4, "medium": 8, "long": 16}},
    "gauge": {"name": "Risiko-Tacho", "src": "RiskGaugeAnimator", "size": (9.0, 4.5), "fps": 8,
              "frames": {"short": 18, "medium": 36, "long": 72}},
    "dashboard": {"name": "Tail-Risk-Dashboard", "src": "create_comprehensive_animation", "size": (11.0, 8.25),
                  "fps": 2, "frames": {"short": 3, "medium": 5, "long": 10}},
}
WINDOW = 60          # rolling window of DistributionMorphAnimator (default) and create_comprehensive_animation
COORD_WINDOW = 20    # levy_flight_3d_coordinates default
GAUGE_ZONES = ((0, 25, "Niedrig"), (25, 50, "Moderat"), (50, 75, "Erhöht"), (75, 100.01, "Extrem"))


def _date_de(d):
    d = str(d)
    return f"{d[8:10]}.{d[5:7]}.{d[:4]}" if len(d) >= 10 and d[4] == "-" and d[7] == "-" else d


def _gauge_layout(fig, size):
    """Frame the polar half-dial: the framework sizes it for a 10 x 8 in figure, where it floats in white space.

    A polar axes draws the half-dial in the middle of a square box, so the box is made almost as wide as the
    figure and reaches below its lower edge; the title gets a fixed height (``axes.titley``) above the dial.
    """
    w, h = size
    d = min(0.9 * w, 2 * (h - 1.0))                    # side of the square axes box (in)
    mid = d / 4 + 0.35                                 # vertical centre of the box (in)
    side = (w - d) / 2 / w
    fig.subplots_adjust(left=side, right=1 - side, bottom=(mid - d / 2) / h, top=(mid + d / 2) / h)
    return {"axes.titley": (h - 0.8 - (mid - d / 2)) / d}


def _render_gif(anim, n_frames, fps, dpi, size, out, first=0, layout=None):
    """Save the original FuncAnimation as GIF, but only ``n_frames`` evenly spaced frames of its own sequence."""
    import matplotlib
    from matplotlib.animation import PillowWriter

    fig = anim._fig
    fig.set_size_inches(*size)
    rc = layout(fig, size) if layout else {}
    frames = list(anim.new_frame_seq())
    total = len(frames)
    frames = [f for f in frames if f >= first] or frames
    if len(frames) > n_frames:
        pick = np.unique(np.linspace(0, len(frames) - 1, n_frames).round().astype(int))
        frames = [frames[i] for i in pick]
    # same frame function, fewer frames: FuncAnimation iterates _iter_gen() and saves _save_count frames
    anim._iter_gen = lambda: iter(frames)
    anim._save_count = len(frames)
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        with matplotlib.rc_context(rc):
            anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                      progress_callback=lambda i, k: out.progress(f"Bild {i + 1} von {k} …",
                                                                   0.15 + 0.8 * (i + 1) / k))
        with open(path, "rb") as fh:
            gif = fh.read()
    finally:
        os.remove(path)
    return gif, [int(f) for f in frames], total


def run(params, data, out):
    kind = params["kind"]
    cfg = KINDS[kind]
    r_all = np.asarray(data.returns, dtype=float)
    n_all = data.n
    m = int(min(params["days"], n_all))
    off = n_all - m
    r = r_all[off:]
    x_all, xa = index_axis(data)
    x = x_all[off:]
    label = (lambda i: _date_de(x[i])) if data.dates else (lambda i: f"Tag {off + i}")
    need = WINDOW + 11 if kind == "dashboard" else WINDOW + 1 if kind == "morph" else 2
    if m < need:
        raise ToolError(f"Für diese Animation braucht es mindestens {need} Handelstage – der Datensatz hat {m}.")

    out.progress("Bereite die Animation vor …", 0.05)
    if kind in ("phase", "dashboard"):
        X, Y, Z = (np.asarray(a, dtype=float) for a in levy_flight_3d_coordinates(r, window=COORD_WINDOW))
    if kind == "phase":
        anim = PhaseSpaceAnimator(X, Y, Z, figsize=cfg["size"]).create_animation(trail_length=50)
    elif kind == "morph":
        anim = DistributionMorphAnimator(r, window=WINDOW).create_animation()
    elif kind == "gauge":
        vix = (np.asarray(data.vix, dtype=float) if data.vix is not None
               else np.asarray(generate_vix_from_returns(r_all), dtype=float))
        vix3m = np.asarray(data.vix3m, dtype=float) if data.vix is not None and data.vix3m is not None else None
        prices = np.asarray(data.prices, dtype=float) if data.prices is not None else 100.0 * np.exp(np.cumsum(r_all))
        ews = CompositeEarlyWarningSystem()
        np.random.seed(int(params["seed"]))
        ind = ews.simulate_all_indicators(r_all, prices, vix, causal=True, vix3m=vix3m)
        comp = np.asarray(ind["composite_score"], dtype=float)[off:]
        scores = [np.asarray(ind[k + "_score"], dtype=float) for k in ("gex", "tdex", "vix_term", "dix", "smfi")]
        anim = RiskGaugeAnimator(comp).create_animation()
    else:
        anim = create_comprehensive_animation(r, (X, Y, Z))

    n_frames = cfg["frames"][params["length"]]
    gif, frames, total = _render_gif(anim, n_frames, cfg["fps"], int(params["dpi"]), cfg["size"], out,
                                     first=COORD_WINDOW if kind == "phase" else 0,
                                     layout=_gauge_layout if kind == "gauge" else None)
    fx = [x[f] for f in frames]

    # ---- per kind: the numbers behind the picture -------------------------------------------------------------
    if kind == "phase":
        jumps = np.flatnonzero(Z > 3)
        out.metric("Volatilität (relativ)", X[-1], "num", hint=f"am {label(m - 1)}; 1 = Durchschnitt des Zeitraums")
        out.metric("Tail-Schwere 2/α", Y[-1], "num", hint="Hill-Schätzer über 20 Tage; höher = fettere Ränder")
        big_txt = f"{Z.max():.1f}".replace(".", ",")
        out.metric("Sprungintensität |r|/σ", Z[-1], "num",
                   hint=f"größter Sprung {big_txt} σ am {label(int(np.argmax(Z)))}; σ des Zeitraums")
        out.metric("Tage mit Sprung > 3 σ", len(jumps), "int",
                   hint=f"{100 * len(jumps) / m:.1f} % der Tage".replace(".", ","))
        s = slice(COORD_WINDOW, None)  # the tail index needs a full window first (2/α is 0 before)
        idx = np.arange(m)[s]
        traces = [C.path3d("Bahn", X[s], Y[s], Z[s], color=1, mode="lines", text=[label(i) for i in idx])]
        big = jumps[jumps >= COORD_WINDOW]
        if big.size:
            traces.append(C.path3d("Sprung > 3 σ", X[big], Y[big], Z[big], color="critical", mode="markers", size=4,
                                   text=[label(i) for i in big]))
        shown = [f for f in frames if f >= COORD_WINDOW]
        traces.append(C.path3d("Bilder im GIF", X[shown], Y[shown], Z[shown], color=3, mode="markers", size=3,
                               text=[label(i) for i in shown]))
        traces.append(C.path3d("Letzter Tag", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers", size=7,
                               text=[label(m - 1)]))
        out.chart(C.scatter3d(
            "Lévy-Phasenraum: die ganze Bahn", traces,
            x=C.axis("Volatilität (relativ)"), y=C.axis("Tail-Schwere 2/α"), z=C.axis("Sprungintensität |r|/σ"),
            subtitle="Jeder Punkt der Linie ist ein Tag (ab Tag 20, wenn der Tail-Index geschätzt ist). Dreh die "
                     "Ansicht mit der Maus.", height=520,
        ))
    elif kind == "morph":
        # RollingRiskMetrics' value at index t+1 uses returns[t-59 .. t], i.e. the window ending on day t (as in
        # the GIF). The appended 0 never enters a window; it only makes the window ending on the last day exist.
        kurt = np.asarray(RollingRiskMetrics(np.append(r, 0.0), window=WINDOW).rolling_kurtosis(), dtype=float)[1:]
        ok = np.isfinite(kurt)
        last = float(kurt[ok][-1]) if ok.any() else float("nan")
        i_max = int(np.nanargmax(kurt)) if ok.any() else 0
        out.metric("Exzess-Kurtosis (letztes Fenster)", last, "num", digits=1,
                   status="critical" if last > 6 else "serious" if last > 3 else None,
                   badge="Fette Ränder" if last > 3 else "",
                   hint=f"60 Tage bis {label(m - 1)}; Normalverteilung = 0")
        out.metric("Höchste Kurtosis", kurt[i_max], "num", digits=1, hint=f"Fenster bis {label(i_max)}")
        out.metric("Fenster mit Kurtosis > 3", float(np.mean(kurt[ok] > 3)) if ok.any() else 0.0, "pct", digits=0,
                   hint="Anteil der 60-Tage-Fenster mit deutlich fetten Rändern")
        out.chart(C.xy(
            "Exzess-Kurtosis im rollierenden 60-Tage-Fenster",
            [C.trace("Exzess-Kurtosis", x, kurt, color=1, width=1.5),
             C.trace("Bilder im GIF", fx, kurt[frames], type="scatter", color=2, size=7, decimate_to=None)],
            x=xa, y=C.axis("Exzess-Kurtosis"),
            hlines=[C.hline(0, "Normalverteilung"), C.hline(3, "deutlich fette Ränder", "serious")],
            subtitle="Wert am Tag t: Fenster der 60 Handelstage bis t. Springt die Kurtosis hoch, haben einzelne "
                     "extreme Tage das Fenster erreicht – im GIF liegen dann Balken weit draußen über der Gauß-Kurve.",
        ))
    elif kind == "gauge":
        st, lv = LEVEL[ews.compute_composite(*(float(s_[-1]) for s_ in scores))["level"]]
        zone = next(z for lo, hi, z in GAUGE_ZONES if lo <= min(comp[-1], 100) < hi)
        i_max = int(np.argmax(comp))
        out.metric("Score (letzter Tag)", comp[-1], "num", digits=0, unit="/ 100", status=st,
                   hint=f"Tacho-Zone {zone} · Warnstufe des Frühwarnsystems: {lv}")
        out.metric("Durchschnitt", float(np.mean(comp)), "num", digits=0, unit="/ 100",
                   hint=f"über {de(m, 0)} Handelstage")
        out.metric("Höchster Score", comp[i_max], "num", digits=0, unit="/ 100", hint=f"am {label(i_max)}")
        out.chart(C.xy(
            "Komposit-Score, den die Tachonadel zeigt",
            [C.trace("Komposit-Score", x, comp, color=1, width=1.5),
             C.trace("Bilder im GIF", fx, comp[frames], type="scatter", color=2, size=6, decimate_to=None)],
            x=xa, y=C.axis("Score", range=(0, 100)),
            hlines=[C.hline(25, "Moderat"), C.hline(50, "Erhöht"), C.hline(75, "Extrem")],
            subtitle="Linien = Zonen des Tachos (25 / 50 / 75). Die Warnstufen des Frühwarnsystems liegen bei "
                     "50 / 75 / 90.",
        ))
        rows = [[z, int(np.sum((comp >= lo) & (comp < hi))), float(np.mean((comp >= lo) & (comp < hi)))]
                for lo, hi, z in GAUGE_ZONES]
        out.table("Zeit in den Zonen des Tachos", ["Zone", "Tage", "Anteil"], rows, formats=["text", "int", "pct"],
                  note="Zonen wie im animierten Tacho: 0–25 niedrig, 25–50 moderat, 50–75 erhöht, 75–100 extrem.")
        out.note("Der Score ist der Komposit des Frühwarnsystems. GEX, TailDex, DIX und SMFI sind dafür aus "
                 "Renditen und VIX *modelliert* (illustrativ, keine echten Messwerte), kausal gerechnet. "
                 "„Current Level: … %“ im GIF-Titel meint Punkte von 100, keine Wahrscheinlichkeit."
                 + ("" if data.vix is not None else " Der VIX fehlt und wird aus den Renditen geschätzt."), "warning")
    else:
        var = np.asarray(RollingRiskMetrics(r, window=WINDOW).rolling_var(0.99), dtype=float)
        loss = -r
        ok = np.isfinite(var)
        breach = ok & (loss > np.where(ok, var, np.inf))
        p90 = float(np.percentile(Z, 90))
        i_max = int(np.nanargmax(var))
        rate = float(np.mean(breach[ok]))
        out.metric("VaR 99 % (60 Tage)", var[-1], "pct", digits=2,
                   hint=f"historisch, für den {label(m - 1)} aus den 60 Tagen davor")
        out.metric("VaR-Verletzungen", rate, "pct", digits=1,
                   hint=f"{int(breach.sum())} von {int(ok.sum())} Tagen · Soll 1\u00a0%",
                   status="critical" if rate > 0.025 else "serious" if rate > 0.015 else "good")
        out.metric("Höchster VaR 99 %", var[i_max], "pct", digits=2, hint=f"am {label(i_max)}")
        n_jump = int(np.sum(Z > 3))
        out.metric("Tage mit Sprung > 3 σ", n_jump, "int",
                   hint=f"{100 * n_jump / m:.1f} % der Tage; σ des Zeitraums".replace(".", ","))
        out.chart(C.xy(
            "Tagesverlust und rollierender VaR 99 %",
            [C.trace("Tagesverlust", x, np.where(loss > 0, loss, np.nan), type="bar", color="muted"),
             C.trace("VaR 99 % (60 Tage)", x, var, color=1, width=1.5),
             C.trace("Verlust über dem VaR", [x[i] for i in np.flatnonzero(breach)], loss[breach], type="scatter",
                     color="critical", size=6, decimate_to=None)],
            x=xa, y=C.axis("Verlust", fmt="pct", digits=1),
            subtitle="Wie Panel 4 der Animation: VaR am Tag t aus den 60 Tagen davor. Aus nur 60 Werten ist das "
                     "1-%-Quantil kaum schätzbar – selbst bei stabiler Verteilung liegen gut 2 % der Tage darüber.",
        ))
        out.chart(C.xy(
            "Sprungintensität |r|/σ",
            [C.trace("|r|/σ", x, Z, color=2, width=1),
             C.trace("Bilder im GIF", fx, Z[frames], type="scatter", color=3, size=6, decimate_to=None)],
            x=xa, y=C.axis("Vielfache von σ"), hlines=[C.hline(p90, "90. Perzentil", "serious")],
            subtitle="Wie Panel 3 der Animation („Tail Risk Evolution“).",
        ))

    out.metric("Bilder im GIF", len(frames), "int", hint=f"aus {de(total, 0)} Bildern der Original-Animation")
    kb = f"{len(gif) / 1024:,.0f}".replace(",", ".")
    out.note(f"**{cfg['name']}** (`{cfg['src']}`): {len(frames)} von {de(total, 0)} Bildern, {cfg['fps']} Bilder/s, "
             f"{kb} KB. Zeitraum {label(0)} – {label(m - 1)}; t im Bildtitel = Handelstag ab {label(0)} (t = 0).",
             "info")
    out.image(gif, f"{cfg['name']} – Animation des Frameworks ({cfg['src']}, {len(frames)} Bilder)",
              mime="image/gif")
