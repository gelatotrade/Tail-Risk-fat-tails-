"""Market phase portrait: momentum x volatility x skewness, attractors, velocity and distance from equilibrium."""

import numpy as np

from src.visualization.phase_space import PhaseSpaceAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "phase-portrait",
    "title": "Markt-Phasenporträt: Momentum, Vola, Schiefe",
    "group": "regime",
    "order": 40,
    "summary": "Verfolgt den Markt als Punkt im Raum aus Momentum, Volatilität und Schiefe – mit Attraktoren, "
               "Tempo, Beschleunigung und Abstand vom Gleichgewicht.",
    "description": (
        "In der Physik beschreibt ein **Phasenraum** den vollständigen Zustand eines Systems; seine Bahn zeigt, wohin "
        "es sich bewegt. `PhaseSpaceAnalyzer` überträgt das auf den Markt: Jeder Handelstag wird zum Punkt aus "
        "**Momentum** X (Ø Tagesrendite im Fenster), **Volatilität** Y (Standardabweichung im Fenster) und "
        "**Schiefe** Z (der Renditen davor). Ruhige Phasen bilden eine dichte Wolke bei niedriger Vola, Krisen "
        "ziehen lange Ausläufer nach oben und zu negativem Momentum.\n\n"
        "**Attraktoren** sind Regionen, in denen sich der Markt immer wieder aufhält. `detect_attractor_regions` "
        "findet sie per Ward-Clustering und ordnet sie nach Volatilität (ruhig → Stress). Standardisiere die Achsen "
        "dafür: Schiefe liegt bei ±1, Rendite und Vola bei 0,01 – unskaliert trennt das Clustering fast nur nach der "
        "Schiefe. Die Verweildauer zeigt, wie „klebrig“ ein Attraktor ist.\n\n"
        "**Geschwindigkeit** und **Beschleunigung** (`compute_velocity`, `compute_acceleration`) messen, wie schnell "
        "sich der Zustand ändert – schnelle Bewegung heißt Regimewechsel. Der **Abstand vom Gleichgewicht** "
        "(`phase_space_distance`) misst die Entfernung zum Ruhepunkt (Momentum 0, Ø-Vola, Schiefe 0), hier in "
        "Standardabweichungen je Achse. Die Original-Grafik zeichnet die Bahn mit Geschwindigkeitspfeilen (`Arrow3D`)."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "window", "type": "int", "default": 20, "min": 5, "max": 120, "step": 1,
         "label": "Fenster", "unit": "Tage",
         "help": "Für Momentum, Volatilität und Schiefe. Kurz = nervös, lang = träge."},
        {"name": "n_clusters", "type": "int", "default": 3, "min": 2, "max": 6, "step": 1,
         "label": "Anzahl Attraktoren"},
        {"name": "standardize", "type": "bool", "default": True, "label": "Achsen fürs Clustering standardisieren",
         "help": "Aus: das unskalierte Original-Verhalten, das fast nur nach der Schiefe trennt."},
        {"name": "recent", "type": "int", "default": 60, "min": 10, "max": 250, "step": 10,
         "label": "Hervorgehobene letzte Tage", "unit": "Tage"},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Zeitraum (letzte Handelstage)", "unit": "Tage",
         "help": "Das Clustering braucht Speicher ∝ N²; 2520 Tage ≈ 10 Jahre."},
        {"name": "show_velocity", "type": "bool", "default": True,
         "label": "Geschwindigkeitspfeile in der Original-Grafik"},
    ],
    "covers": [
        "src.visualization.phase_space.Arrow3D",
        "src.visualization.phase_space.Arrow3D.do_3d_projection",
        "src.visualization.phase_space.PhaseSpaceAnalyzer",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.compute_velocity",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.compute_acceleration",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.phase_space_distance",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.detect_attractor_regions",
        "src.visualization.phase_space.PhaseSpaceAnalyzer.plot_phase_portrait",
    ],
    "cost": "medium",
}

REGIME_DE = {"LOW_VOL": "Ruhig (niedrige Vola)", "NORMAL": "Normal", "HIGH_VOL": "Stress (hohe Vola)"}
RANK_COLORS = (1, 4, 2, 7, 5, 8)  # attractors ordered calm -> stress: blue, amber, orange, ... (never green for stress)
MAX_CLOUD = 500   # points per attractor in the 3D cloud: denser clouds hide the recent path (the original figure
                  # and the 2D charts always get every day)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _signed(x, digits=2, unit=""):
    """Signed German number with at least two significant digits (tiny daily changes must not round to 0)."""
    if x != 0 and np.isfinite(x):
        digits = int(min(7, max(digits, 1 - np.floor(np.log10(abs(x))))))
    s = _de(abs(x), digits)
    return ("+" if x > 0 else "−" if x < 0 else "±") + s + unit


def _date_de(d):
    s = str(d)
    return f"{s[8:10]}.{s[5:7]}.{s[:4]}" if len(s) == 10 and s[4] == "-" else s


def _regime_de(label):
    if label in REGIME_DE:
        return REGIME_DE[label]
    if label.startswith("NORMAL_"):
        return f"Normal {label.split('_')[1]}"
    return label


def _rank_pct(series, value):
    """Mid-rank percentile of value within series (ties count half)."""
    s = series[np.isfinite(series)]
    return float(0.5 * (np.mean(s < value) + np.mean(s <= value))) if s.size else np.nan


def _mean_run(labels, cid):
    """Mean length (days) of uninterrupted stays in cluster cid."""
    hit = np.concatenate([[0], (labels == cid).astype(np.int8), [0]])
    edges = np.diff(hit)
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.mean()) if lengths.size else np.nan


def run(params, data, out):
    w = int(params["window"])
    k_clusters = int(params["n_clusters"])
    n = int(min(params["last_n"], data.n))
    need = max(3 * w, 60)              # at least two full windows of clustered points after the warm-up
    if n < need:
        fix = ("Verkleinere das Fenster" + (" oder verlängere den Zeitraum." if n < data.n else ".")
               if 3 * w > 60 else "Wähle einen längeren Datensatz.")
        raise ToolError(f"Für ein Fenster von {w} Tagen brauchst du mindestens {need} Tagesrenditen im "
                        f"Zeitraum (vorhanden: {n}). {fix}")
    r = np.asarray(data.returns[-n:], dtype=float)
    dates = list(data.dates[-n:]) if data.dates else None
    x_time = dates if dates else list(range(data.n - n + 1, data.n + 1))
    x_axis = C.axis("", type="date") if dates else C.axis("Handelstag", fmt="int")
    day_txt = [_date_de(d) for d in dates] if dates else [f"Tag {i}" for i in x_time]

    out.progress("Phasenraum-Koordinaten …", 0.05)
    psa = PhaseSpaceAnalyzer(r, window=w)
    X, Y, Z = (np.asarray(a, dtype=float) for a in (psa.X, psa.Y, psa.Z))
    vel = np.column_stack(psa.compute_velocity())
    acc = np.column_stack(psa.compute_acceleration())
    dist = np.asarray(psa.phase_space_distance(standardize=True), dtype=float)
    dist_raw = np.asarray(psa.phase_space_distance(), dtype=float)

    out.progress("Attraktoren (Ward-Clustering) …", 0.25)
    att = psa.detect_attractor_regions(n_clusters=k_clusters, standardize=bool(params["standardize"]))
    if "error" in att:
        raise ToolError("Zu wenige gültige Punkte für das Clustering – verlängere den Zeitraum.")

    # the first `window` days are warm-up (skewness is still 0, the windows are not full yet)
    valid = np.arange(n) >= w
    coords = np.column_stack([X, Y, Z])
    finite = ~np.any(np.isnan(coords), axis=1)
    labels = np.zeros(n, dtype=int)
    labels[finite] = np.asarray(att["cluster_labels"], dtype=int)
    order = [int(c) for c in att["regime_order"]]          # 0-based cluster index, low -> high volatility
    rank = {c + 1: i for i, c in enumerate(order)}         # cluster id -> volatility rank 0..k-1
    names = {cid: _regime_de(lbl) for cid, lbl in att["cluster_regimes"].items()}
    centers = np.asarray(att["centers"], dtype=float)

    # speed and acceleration in standard deviations per axis (raw units would be pure skewness)
    scale = np.nanstd(coords[valid], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    speed = np.sqrt(np.sum((vel / scale) ** 2, axis=1))
    accel = np.sqrt(np.sum((acc / scale) ** 2, axis=1))

    cur = int(labels[-1])
    cur_rank = rank.get(cur, 0)
    status = "good" if cur_rank == 0 else "serious" if cur_rank == k_clusters - 1 else "warning"
    vol_pct = _rank_pct(Y[valid], Y[-1])
    dist_pct = _rank_pct(dist[valid], dist[-1])
    speed_pct = _rank_pct(speed[valid], speed[-1])
    acc_pct = _rank_pct(accel[valid], accel[-1])

    # --- metrics ------------------------------------------------------------------------
    stay = _run_today(labels)
    out.metric("Attraktor heute", names.get(cur, "–"), "text", status=status,
               badge=f"Vola-Rang {cur_rank + 1}/{k_clusters}",
               hint="erst seit heute in diesem Attraktor" if stay == 1 else f"seit {stay} Tagen in diesem Attraktor")
    out.metric(f"Volatilität ({w} Tage)", Y[-1], "pct", digits=2,
               hint=f"täglich · ≈ {_de(100 * Y[-1] * np.sqrt(252), 0)} % p. a. · Perzentil {_de(100 * vol_pct, 0)} %")
    out.metric(f"Momentum ({w} Tage)", X[-1], "pct", digits=3, hint="Ø Tagesrendite im Fenster")
    out.metric("Schiefe", Z[-1], "num", digits=2,
               hint=f"der {w} Renditen davor · < 0: große Verluste häufiger als große Gewinne")
    out.metric("Abstand vom Gleichgewicht", dist[-1], "num", digits=2,
               hint=f"in σ je Achse · Perzentil {_de(100 * dist_pct, 0)} %",
               status="serious" if dist_pct >= 0.97 else "warning" if dist_pct >= 0.9 else None)
    out.metric("Tempo im Phasenraum", speed[-1], "num", digits=3,
               hint=f"σ pro Tag · Perzentil {_de(100 * speed_pct, 0)} %",
               status="warning" if speed_pct >= 0.9 else None)
    out.metric("Beschleunigung", accel[-1], "num", digits=3, hint=f"σ pro Tag² · Perzentil {_de(100 * acc_pct, 0)} %")

    # --- chart 1: the phase portrait in 3D ---------------------------------------------
    idx = np.flatnonzero(valid)
    k_recent = int(min(params["recent"], len(idx)))
    recent = idx[-k_recent:]
    traces, thinned = [], False
    for cid in sorted(rank, key=rank.get):
        ii = idx[labels[idx] == cid]
        if ii.size > MAX_CLOUD:
            ii, thinned = ii[::int(np.ceil(ii.size / MAX_CLOUD))], True
        if ii.size:
            traces.append(C.path3d(f"{names[cid]} · {int(np.sum(labels[idx] == cid))} Tage",
                                   X[ii], Y[ii], Z[ii], color=RANK_COLORS[rank[cid]], mode="markers", size=2,
                                   text=[day_txt[i] for i in ii]))
    traces.append(C.path3d("Attraktor-Zentren", centers[:, 0], centers[:, 1], centers[:, 2], color="muted",
                           mode="markers", size=9, text=[names[c + 1] for c in range(len(centers))]))
    traces.append(C.path3d(f"Letzte {k_recent} Tage", X[recent], Y[recent], Z[recent], color="ink",
                           mode="lines+markers", size=3, text=[day_txt[i] for i in recent]))
    traces.append(C.path3d(f"Heute ({day_txt[-1]})", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers",
                           size=8, text=[names.get(cur, "")]))
    out.chart(C.scatter3d(
        "Phasenporträt: Momentum × Volatilität × Schiefe", traces,
        x=C.axis("Momentum (Ø Tagesrendite)", fmt="pct", digits=2), y=C.axis("Volatilität (täglich)", fmt="pct",
                                                                            digits=1),
        z=C.axis("Schiefe"), height=580,
        subtitle=f"Jeder Punkt ist ein Handelstag ({w}-Tage-Fenster), gefärbt nach Attraktor (ruhig → Stress)"
                 + (f"; große Attraktoren zeigen jeden n-ten Tag (höchstens {MAX_CLOUD} Punkte)" if thinned else "")
                 + ". Die dunkle Linie ist der Weg der letzten Tage. Ziehen dreht, Mausrad zoomt.",
    ))

    # --- chart 2: volatility over time coloured by attractor ----------------------------
    tr = []
    for cid in sorted(rank, key=rank.get):
        ii = idx[labels[idx] == cid]
        if ii.size:
            tr.append(C.trace(names[cid], [x_time[i] for i in ii], Y[ii], type="scatter", color=RANK_COLORS[rank[cid]],
                              size=3, decimate_to=None, text=[day_txt[i] for i in ii]))
    out.chart(C.xy(
        "Volatilität im Zeitverlauf – gefärbt nach Attraktor", tr,
        x=x_axis, y=C.axis(f"Volatilität ({w} Tage, täglich)", fmt="pct", digits=1), height=320,
        subtitle="Lange gleichfarbige Strecken: Der Markt bleibt in seinem Attraktor. Farbwechsel sind Regimewechsel.",
    ))

    # --- chart 3: distance from equilibrium ----------------------------------------------
    xs = [x_time[i] for i in idx]
    d95 = float(np.nanpercentile(dist[valid], 95))
    out.chart(C.xy(
        "Abstand vom Gleichgewicht (standardisiert)",
        [C.trace("Abstand", xs, dist[valid], color=1, width=1.2)],
        x=x_axis, y=C.axis("Abstand (σ je Achse)"), height=300,
        hlines=[C.hline(d95, "95. Perzentil")],
        subtitle="Euklidischer Abstand zum Ruhepunkt (Momentum 0, Ø-Volatilität, Schiefe 0), jede Achse in ihren "
                 "Standardabweichungen. Spitzen sind Ausnahmezustände.",
    ))

    # --- chart 4: speed in phase space ------------------------------------------------
    sp = speed[valid]
    sp_avg = np.convolve(sp, np.ones(w) / w, mode="full")[:len(sp)]
    sp_avg[:w - 1] = np.nan
    out.chart(C.xy(
        "Tempo im Phasenraum",
        [C.trace("Tageswert", xs, sp, color=2, width=0.8),
         C.trace(f"Mittel über {w} Tage", xs, sp_avg, color=1, width=2.2)],
        x=x_axis, y=C.axis("σ pro Tag"), height=320,
        subtitle=f"Betrag der Geschwindigkeit (np.gradient je Achse, standardisiert). Ein Schocktag erzeugt zwei "
                 f"Spitzen im Abstand von {w} Tagen – beim Eintritt ins Fenster und beim Austritt. Das Mittel zeigt "
                 "unruhige Phasen, in denen sich der Zustand dauernd verschiebt.",
    ))

    # --- table 1: attractors ---------------------------------------------------------
    rows = []
    lab_v = labels[idx]
    for cid in sorted(rank, key=rank.get):
        c = centers[cid - 1]
        mask = lab_v == cid
        last = _date_de(x_time[idx[np.flatnonzero(mask)[-1]]]) if mask.any() and dates else (
            f"Tag {x_time[idx[np.flatnonzero(mask)[-1]]]}" if mask.any() else "–")
        rows.append([names[cid] + (" ← heute" if cid == cur else ""), float(c[0]), float(c[1]),
                     float(c[1] * np.sqrt(252)), float(c[2]), float(mask.mean()), _mean_run(lab_v, cid), last])
    eta = _eta_squared(coords[idx], lab_v)
    eta_txt = (f"Die Attraktoren erklären {_de(100 * eta[1], 0)} % der Streuung der Volatilität, "
               f"{_de(100 * eta[2], 0)} % der Schiefe und {_de(100 * eta[0], 0)} % des Momentums (η²). ")
    out.table("Attraktor-Regionen (Ward-Clustering)", [
        "Attraktor", "Ø Momentum", "Ø Vola täglich", "Ø Vola p. a.", "Ø Schiefe", "Anteil Tage",
        "Ø Verweildauer (Tage)",
        "zuletzt"], rows, formats=["text", "pct", "pct", "pct", "num", "pct", "num", "text"],
        note=eta_txt + ("Zentren in Originaleinheiten, Clustering auf standardisierten Achsen. "
                        if params["standardize"] else "Clustering auf unskalierten Achsen (Original-Verhalten). ")
        + "Verweildauer = mittlere Länge der ununterbrochenen Aufenthalte. Ohne die ersten "
        + f"{w} Tage (Aufwärmphase).")

    # --- table 2: motion today per axis ----------------------------------------------
    axes = [("Momentum X", X, "pct", 3), ("Volatilität Y", Y, "pct", 2), ("Schiefe Z", Z, "num", 2)]
    rows = []
    for j, (name, s, kind, dg) in enumerate(axes):
        unit = " %" if kind == "pct" else ""
        f = 100.0 if kind == "pct" else 1.0
        v, a = float(vel[-1, j]), float(acc[-1, j])
        trend = ("steigt" if v > 0 else "fällt" if v < 0 else "konstant")
        trend += (", beschleunigt" if v * a > 0 else ", bremst ab" if v * a < 0 else "")
        rows.append([name, _de(f * s[-1], dg) + unit, _rank_pct(s[valid], s[-1]), _signed(f * v, dg + 1, unit),
                     _signed(f * a, dg + 1, unit), float(abs(v) / scale[j]), trend])
    out.table("Bewegung heute je Achse", [
        "Achse", "heute", "Perzentil", "Geschwindigkeit pro Tag", "Beschleunigung pro Tag²", "Tempo (σ/Tag)",
        "Richtung"], rows, formats=["text", "text", "pct", "text", "text", "num", "text"],
        note="Zentrale Differenzen (np.gradient) der Koordinaten; am letzten Tag einseitig. Tempo = |Geschwindigkeit| "
             "geteilt durch die Standardabweichung der Achse.")

    # --- notes ---------------------------------------------------------------------
    # the original clusters every row, including the warm-up (partial windows, skewness 0): on short series one of
    # those artefacts can become an "attractor" of its own
    warm_only = [names[cid] for cid in sorted(rank, key=rank.get) if not np.any(lab_v == cid)]
    if warm_only:
        out.note(f"Der Attraktor „{warm_only[0]}“ besteht nur aus Tagen der Aufwärmphase (die ersten {w} Tage, "
                 "Fenster noch nicht voll) – ein Artefakt der kurzen Reihe, kein Marktzustand. Verkleinere das "
                 "Fenster oder nimm eine längere Reihe.", "warning")
    raw_corr =float(np.corrcoef(dist_raw[valid], np.abs(Z[valid]))[0, 1]) if np.std(dist_raw[valid]) > 0 else np.nan
    if np.isfinite(raw_corr) and params["standardize"]:
        out.note(f"Warum standardisiert? Unskaliert ist der Abstand vom Gleichgewicht praktisch nur |Schiefe| "
                 f"(Korrelation {_de(raw_corr)}): Rendite und Vola liegen bei 0,01, die Schiefe bei 1. Dasselbe gilt "
                 "fürs Clustering – schalte „Achsen standardisieren“ aus, um den Effekt zu sehen.")
    if not params["standardize"] and eta[2] > 1.5 * eta[1]:
        out.note(f"Unskaliertes Clustering: Die Attraktoren erklären {_de(100 * eta[2], 0)} % der Schiefe-Streuung, "
                 f"aber nur {_de(100 * eta[1], 0)} % der Vola-Streuung – die „Vola-Regime“ sind hier eigentlich "
                 "Schiefe-Gruppen. Schalte die Standardisierung wieder ein.", "warning")
    elif not params["standardize"]:
        out.note("Unskaliertes Clustering (Original-Verhalten): Die Schiefe (Größenordnung 1) wiegt im Abstandsmaß "
                 "rund hundertmal mehr als Rendite und Volatilität (Größenordnung 0,01).", "info")

    out.progress("Original-Grafik wird gezeichnet …", 0.6)
    fig = psa.plot_phase_portrait(figsize=(11, 8), show_velocity=bool(params["show_velocity"]),
                                  show_attractors=True)
    out.figure(fig, "Original: PhaseSpaceAnalyzer.plot_phase_portrait – Farbe = Zeit, rote Pfeile = Geschwindigkeit "
                    "(Arrow3D, ×10, unskaliert), Sterne = 3 Attraktoren, grüne Raute = heute.", dpi=72)


def _eta_squared(values, labels):
    """Share of each column's variance explained by the cluster means (between / total sum of squares)."""
    values = np.asarray(values, dtype=float)
    total = np.sum((values - values.mean(axis=0)) ** 2, axis=0)
    between = np.zeros(values.shape[1])
    for c in np.unique(labels):
        grp = values[labels == c]
        between += len(grp) * (grp.mean(axis=0) - values.mean(axis=0)) ** 2
    return np.where(total > 0, between / np.where(total > 0, total, 1.0), 0.0)


def _run_today(labels):
    """Length of the current uninterrupted stay in today's cluster."""
    same = labels[::-1] == labels[-1]
    return int(np.argmin(same)) if not same.all() else int(len(labels))
