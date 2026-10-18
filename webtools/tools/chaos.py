"""Chaos or chance? Lyapunov exponent and recurrence analysis, honestly tested against shuffled returns."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.visualization.phase_space import LyapunovExponentEstimator, RecurrencePlotAnalyzer

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "chaos",
    "title": "Chaos-Test: Lyapunov & Rekurrenz",
    "group": "regime",
    "order": 70,
    "summary": "Schätzt den Lyapunov-Exponenten und die Rekurrenzstruktur deiner Renditen – und prüft ehrlich gegen "
               "zufällig gemischte Renditen, ob dahinter Chaos steckt oder nur Rauschen.",
    "description": (
        "Ein chaotisches System ist deterministisch, aber empfindlich: Zwei fast gleiche Zustände laufen exponentiell "
        "auseinander, mit der Rate **λ** (größter Lyapunov-Exponent). `LyapunovExponentEstimator` rekonstruiert dafür "
        "aus der Renditereihe einen Phasenraum (**Delay-Einbettung**: Vektoren aus m Renditen im Abstand τ), sucht "
        "zu zufälligen Startpunkten den nächsten Nachbarn und misst, wie schnell sich beide trennen. "
        "`classify_dynamics` nennt λ > 0,1 „chaotisch“. Der **Recurrence-Plot** (`RecurrencePlotAnalyzer`) markiert "
        "jedes Paar von Tagen, deren eingebettete Zustände näher als ε beieinander liegen; die RQA zählt daraus "
        "Rekurrenzrate (RR), Determinismus (DET, Anteil auf Diagonalen) und Laminarität (LAM, auf Senkrechten).\n\n"
        "**Der ehrliche Test:** Mische die Renditen zufällig. Das zerstört jede zeitliche Ordnung, die Verteilung "
        "bleibt gleich. Echte Dynamik muss sich vom Surrogat unterscheiden. Bei Tagesrenditen ist λ aber genauso "
        "positiv wie beim Surrogat und bei Gauß-Rauschen – der Schätzer misst hier die Streuung des Rauschens, kein "
        "Chaos. Die Kontrolle mit der **logistischen Abbildung** (echtes Chaos, λ = ln 2 ≈ 0,69) zeigt den "
        "Unterschied: Ihr λ bleibt über alle m gleich, beim Rauschen fällt es mit m.\n\n"
        "Was der Recurrence-Plot bei Renditen wirklich zeigt, ist **Volatilitäts-Clustering**: Ruhige Wochen bilden "
        "dunkle Blöcke, Stressphasen helle Bänder. Liegen DET oder LAM über allen Surrogaten, ist das zeitliche "
        "Struktur (Vola-Cluster) – aber kein Beleg für Vorhersagbarkeit der Kursrichtung."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "embedding_dim", "type": "int", "default": 3, "min": 2, "max": 8, "step": 1,
         "label": "Einbettungsdimension m", "help": "Wie viele aufeinanderfolgende Renditen einen Zustand bilden."},
        {"name": "tau", "type": "int", "default": 1, "min": 1, "max": 10, "step": 1, "label": "Verzögerung τ",
         "help": "Abstand der Renditen im Zustandsvektor, in Schritten der Reihe (Tage bzw. Wochen)."},
        {"name": "rp_scale", "type": "select", "default": "week", "label": "Zeitskala des Recurrence-Plots",
         "options": [{"value": "week", "label": "Wochenrenditen (je 5 Handelstage)"},
                     {"value": "day", "label": "Tagesrenditen"}],
         "help": "Wochen zeigen mehrere Jahre und damit Ruhe- und Krisenphasen im selben Bild."},
        {"name": "rp_points", "type": "int", "default": 400, "min": 100, "max": 400, "step": 50,
         "label": "Punkte im Recurrence-Plot",
         "help": "Die letzten N Wochen bzw. Tage. Rechenzeit und Speicher wachsen mit N²."},
        {"name": "recurrence_rate", "type": "int", "default": 10, "min": 2, "max": 30, "step": 1,
         "label": "Rekurrenzrate (legt ε fest)", "unit": "%",
         "help": "ε = dieses Quantil aller Abstände. 10 % ist der Standard des Originals."},
        {"name": "n_surrogates", "type": "int", "default": 19, "min": 5, "max": 49, "step": 1,
         "label": "Surrogate (gemischte Renditen)",
         "help": "19 Surrogate: Liegt der echte Wert über allen, ist das signifikant auf dem 5-%-Niveau."},
        {"name": "max_iter", "type": "int", "default": 200, "min": 50, "max": 1000, "step": 50,
         "label": "Startpunkte für λ", "help": "Mehr Startpunkte = ruhigerer Schätzer, längere Rechenzeit."},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 10000, "step": 250,
         "label": "Zeitraum für λ (letzte Handelstage)", "unit": "Tage"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed",
         "help": "Steuert Startpunkte, Surrogate und Kontrollreihen."},
    ],
    "covers": [
        "src.visualization.phase_space.LyapunovExponentEstimator",
        "src.visualization.phase_space.LyapunovExponentEstimator.estimate_largest_lyapunov",
        "src.visualization.phase_space.LyapunovExponentEstimator.classify_dynamics",
        "src.visualization.phase_space.RecurrencePlotAnalyzer",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.compute_recurrence_matrix",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.recurrence_quantification",
        "src.visualization.phase_space.RecurrencePlotAnalyzer.plot_recurrence",
    ],
    "cost": "medium",
}

CLASS_DE = {"CHAOTIC": "chaotisch", "WEAKLY_CHAOTIC": "schwach chaotisch", "QUASI_PERIODIC": "quasi-periodisch",
            "STABLE": "stabil", "UNKNOWN": "unbekannt"}
M_CURVE = (2, 3, 4, 5, 6, 7)   # embedding dimensions of the sensitivity curve
CURVE_SURR = 3                 # surrogates averaged per point of that curve
HEAT_CELLS = 60                # heatmap cells per axis (blocks of consecutive points)
N_NEIGHBORS = 5                # default of the original


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _date_de(d):
    s = str(d)
    return f"{s[8:10]}.{s[5:7]}.{s[:4]}" if len(s) == 10 and s[4] == "-" else s


def _logistic(n, x0):
    """Logistic map x -> 4x(1-x): fully chaotic, largest Lyapunov exponent ln 2."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    return x


def _lyap(series, m, tau, iters, seed):
    return float(LyapunovExponentEstimator(series, embedding_dim=m, tau=tau).estimate_largest_lyapunov(
        n_neighbors=N_NEIGHBORS, max_iter=iters, seed=seed))


def _epsilon(rpa, q):
    """The threshold the original uses by default (a quantile of all pairwise distances), for any quantile q."""
    return float(np.percentile(squareform(pdist(rpa.embedded)), 100.0 * q))


def _blocks(R, cells):
    """Average the 0/1 recurrence matrix over b x b blocks of days (b = ceil(N / cells))."""
    n = R.shape[0]
    b = max(1, int(np.ceil(n / cells)))
    nb = int(np.ceil(n / b))
    pad = nb * b - n
    Rp = np.pad(R.astype(float), ((0, pad), (0, pad)), constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        D = np.nanmean(Rp.reshape(nb, b, nb, b), axis=(1, 3))
    return D, b


def _rank_text(value, sur):
    sur = np.asarray(sur, dtype=float)
    sur = sur[np.isfinite(sur)]
    if not np.isfinite(value) or not sur.size:
        return "–"
    if value > sur.max():
        return f"über allen {sur.size} Surrogaten"
    if value < sur.min():
        return f"unter allen {sur.size} Surrogaten"
    above = int(np.sum(sur < value))
    return f"im Bereich (Rang {above + 1} von {sur.size + 1})"


def _rp_series(data, scale, points):
    """The series of the recurrence plot: the last `points` daily or weekly (5-day) log returns, with dates."""
    r = np.asarray(data.returns, dtype=float)
    step = 5 if scale == "week" else 1
    k = int(min(points, len(r) // step))
    tail = r[len(r) - k * step:]
    x = tail.reshape(k, step).sum(axis=1)
    pos = np.arange(len(r) - k * step + step - 1, len(r), step)      # last day of every block
    labels = [data.dates[i] for i in pos] if data.dates else [int(i) + 1 for i in pos]
    return x, labels


def run(params, data, out):
    m, tau = int(params["embedding_dim"]), int(params["tau"])
    seed, n_sur, iters = int(params["seed"]), int(params["n_surrogates"]), int(params["max_iter"])
    q = float(params["recurrence_rate"]) / 100.0
    weekly = params["rp_scale"] == "week"
    n = int(min(params["last_n"], data.n))
    span = (max(M_CURVE[-1], m) - 1) * tau
    need = span + 40                   # the estimator follows neighbours 9 steps ahead from > 10 start vectors
    if n < need:
        raise ToolError(f"Für die λ-Kurve bis m = {max(M_CURVE[-1], m)} mit τ = {tau} brauchst du mindestens {need} "
                        f"Tagesrenditen (vorhanden: {n}). "
                        + ("Verkleinere τ." if tau > 1 else "Wähle einen längeren Datensatz."))
    r = np.asarray(data.returns[-n:], dtype=float)
    rng = np.random.default_rng(seed)

    # series of the recurrence plot; too few weeks -> fall back to daily returns
    x, x_lab = _rp_series(data, params["rp_scale"], int(params["rp_points"]))
    fallback = weekly and len(x) - (m - 1) * tau < 60
    if fallback:
        weekly = False
        x, x_lab = _rp_series(data, "day", int(params["rp_points"]))
    unit, units = ("Woche", "Wochen") if weekly else ("Tag", "Tage")
    if len(x) - (m - 1) * tau < 30:
        raise ToolError(f"Zu wenige Punkte im Recurrence-Plot ({len(x)} {units}) für m = {m} und τ = {tau}. "
                        "Erhöhe die Punktzahl oder verkleinere m bzw. τ.")

    # --- Lyapunov: real series, shuffled surrogates (same seed for the start points) ----------
    out.progress("Lyapunov-Exponent der Renditen …", 0.05)
    est = LyapunovExponentEstimator(r, embedding_dim=m, tau=tau)
    cls = est.classify_dynamics(n_neighbors=N_NEIGHBORS, max_iter=iters, seed=seed)
    lam = float(cls["lyapunov"])
    out.progress(f"Lyapunov-Exponent von {n_sur} gemischten Reihen …", 0.15)
    surrogates = [rng.permutation(r) for _ in range(n_sur)]
    lam_sur = np.array([_lyap(s, m, tau, iters, seed) for s in surrogates])

    # --- sensitivity to m: returns, shuffled, Gaussian noise, chaotic logistic map ------------
    out.progress("λ für m = 2 … 7 und Kontrollreihen …", 0.4)
    gauss = rng.normal(0.0, float(np.std(r)), n)
    logistic = _logistic(n, 0.1 + 0.8 * float(rng.random()))
    curves = {"r": [], "s": [], "g": [], "l": []}
    for mm in M_CURVE:
        curves["r"].append(lam if mm == m else _lyap(r, mm, tau, iters, seed))
        curves["s"].append(float(np.mean([lam_sur[i] if mm == m else _lyap(surrogates[i], mm, tau, iters, seed)
                                          for i in range(min(CURVE_SURR, n_sur))])))
        curves["g"].append(_lyap(gauss, mm, tau, iters, seed))
        curves["l"].append(_lyap(logistic, mm, tau, iters, seed))
    lam_logistic = curves["l"][M_CURVE.index(m)] if m in M_CURVE else _lyap(logistic, m, tau, iters, seed)

    # --- recurrence analysis on the last N weeks / days, real vs shuffled ---------------------
    out.progress("Recurrence-Plot und RQA …", 0.7)
    rpa = RecurrencePlotAnalyzer(x, embedding_dim=m, tau=tau)
    eps = _epsilon(rpa, q)
    R = rpa.compute_recurrence_matrix(epsilon=eps)
    rqa = rpa.recurrence_quantification(epsilon=eps)
    sur_rqa, R_sur = [], None
    for i in range(n_sur):
        a = RecurrencePlotAnalyzer(rng.permutation(x), embedding_dim=m, tau=tau)
        e = _epsilon(a, q)
        sur_rqa.append(a.recurrence_quantification(epsilon=e))
        if R_sur is None:
            R_sur = a.compute_recurrence_matrix(epsilon=e)
    det_s = np.array([d["determinism"] for d in sur_rqa], dtype=float)
    lam_s = np.array([d["laminarity"] for d in sur_rqa], dtype=float)
    stab_s = np.array([d["stability_index"] for d in sur_rqa], dtype=float)
    det, lamr = float(rqa["determinism"]), float(rqa["laminarity"])

    # --- metrics ------------------------------------------------------------------------
    cls_name = CLASS_DE.get(cls["classification"], cls["classification"])
    lam_lo, lam_hi = float(np.nanmin(lam_sur)), float(np.nanmax(lam_sur))
    out.metric("Lyapunov-Exponent λ", lam, "num", digits=3,
               hint=f"Tagesrenditen · m = {m}, τ = {tau}, {iters} Startpunkte, {n} Tage")
    out.metric("Klassifikation des Originals", cls_name, "text",
               hint="classify_dynamics: λ > 0,1 ⇒ chaotisch – gilt aber auch für reines Rauschen")
    out.metric("λ gemischter Renditen", float(np.nanmean(lam_sur)), "num", digits=3,
               hint=f"Ø von {n_sur} Surrogaten · Spanne {_de(lam_lo, 3)} bis {_de(lam_hi, 3)}")
    out.metric("λ Chaos-Kontrolle", lam_logistic, "num", digits=3,
               hint="logistische Abbildung, theoretisch ln 2 ≈ 0,693")
    out.metric("Determinismus DET", det, "pct", digits=1,
               hint=f"{len(x)} {units} · Surrogate Ø {_de(100 * np.mean(det_s), 1)} % · {_rank_text(det, det_s)}")
    out.metric("Laminarität LAM", lamr, "pct", digits=1,
               hint=f"Surrogate Ø {_de(100 * np.mean(lam_s), 1)} % · {_rank_text(lamr, lam_s)}")
    out.metric("Schwelle ε", eps / float(np.std(x)), "num", digits=2,
               hint=f"in σ der {'Wochen' if weekly else 'Tages'}renditen · Rekurrenzrate "
                    f"{_de(100 * float(rqa['recurrence_rate']), 1)} %")

    # --- charts 1 + 2: recurrence plots, real and shuffled -----------------------------
    D, b = _blocks(R, HEAT_CELLS)
    D_sur, _ = _blocks(R_sur, HEAT_CELLS)
    centers = [min(len(R) - 1, i * b + b // 2) for i in range(D.shape[0])]
    ticks = [x_lab[c] for c in centers]
    hx = C.axis("", type="date") if data.dates else C.axis("Handelstag", fmt="int")
    top = float(max(np.nanmax(D), np.nanmax(D_sur)))
    z_ax = C.axis("Rekurrenzdichte", fmt="pct", digits=0, range=(0.0, top))   # same colour range for both plots
    dat = "Wochen" if weekly else "Tagen"
    lead = (f"Jede Zelle fasst {b} × {b} Paare von {dat} zusammen: Anteil der Paare, deren Zustände" if b > 1 else
            f"Jede Zelle ist ein Paar von {dat}: 100 %, wenn ihre Zustände")
    period = (f"{_date_de(x_lab[0])} – {_date_de(x_lab[-1])}" if data.dates else f"letzte {len(x)} {units}")
    scale_txt = "Wochenrenditen" if weekly else "Tagesrenditen"
    out.chart(C.heatmap(
        f"Recurrence-Plot der {scale_txt} ({period})", ticks, ticks, np.round(D, 3), x=hx, y=hx, z=z_ax,
        scale="sequential", height=560,
        subtitle=f"{lead} (m = {m}, τ = {tau}) näher als ε liegen. Dunkle Blöcke = ruhige Phasen, "
                 "die sich ähneln; helle Kreuze = Krisen, deren Zustände nie wiederkehren. Zeit läuft nach rechts und "
                 "nach unten.",
    ))
    out.chart(C.heatmap(
        f"Zum Vergleich: dieselben {scale_txt}, zufällig gemischt", ticks, ticks, np.round(D_sur, 3), x=hx, y=hx,
        z=z_ax, scale="sequential", height=560,
        subtitle=f"Gleiche Verteilung, aber keine zeitliche Ordnung (Seed {seed}). Die dichteste Zelle erreicht hier "
                 f"{_de(100 * float(np.nanmax(D_sur)), 0)} %, bei den echten Renditen "
                 f"{_de(100 * float(np.nanmax(D)), 0)} %. "
                 "Was oben an Blöcken und Kreuzen mehr zu sehen ist, ist zeitliche Struktur – bei Renditen das "
                 "Volatilitäts-Clustering.",
    ))
    out.chart(C.xy(
        f"{scale_txt} im Fenster des Recurrence-Plots",
        [C.trace("Wochenrendite" if weekly else "Tagesrendite", x_lab, x, type="bar", color=1)],
        x=hx, y=C.axis("Rendite (log)", fmt="pct", digits=1), height=260,
        subtitle="Große Ausschläge erscheinen im Recurrence-Plot als helle Kreuze: Ihre Zustände kehren nie wieder.",
    ))

    # --- chart 3: lambda versus embedding dimension -------------------------------------
    ms = list(M_CURVE)
    out.chart(C.xy(
        "λ hängt von m ab – außer bei echtem Chaos",
        [C.trace("Renditen", ms, curves["r"], color=1, width=2.5),
         C.trace(f"gemischte Renditen (Ø {min(CURVE_SURR, n_sur)})", ms, curves["s"], color=2, dash=True),
         C.trace("Gauß-Rauschen (gleiche σ)", ms, curves["g"], color="muted", dash=True),
         C.trace("logistische Abbildung (Chaos)", ms, curves["l"], color=4, width=2.5)],
        x=C.axis("Einbettungsdimension m", fmt="int"), y=C.axis("geschätztes λ"), height=360,
        hlines=[C.hline(np.log(2.0), "ln 2 = wahres λ der logistischen Abbildung"),
                C.hline(0.1, "Schwelle „chaotisch“ des Originals")],
        subtitle="Echtes Chaos hat ein λ, das nicht von der Einbettung abhängt (Plateau bei ln 2). Bei Renditen, "
                 "Surrogat und Rauschen fällt λ mit m – ein Artefakt des Schätzers, kein Chaos.",
    ))

    # --- chart 4: delay embedding of the daily series --------------------------------------
    emb = np.asarray(est.embedded, dtype=float)
    dl = list(data.dates[-n:]) if data.dates else None
    lab = [_date_de(d) for d in dl[:len(emb)]] if dl else None
    if m >= 3:
        out.chart(C.scatter3d(
            f"Rekonstruierter Phasenraum der Tagesrenditen (Delay-Einbettung, {len(emb)} Zustände)",
            [C.path3d("Zustand (r(t), r(t+τ), r(t+2τ))", emb[:, 0], emb[:, 1], emb[:, 2], color=1, mode="markers",
                      size=2, text=lab)],
            x=C.axis("r(t)", fmt="pct", digits=0), y=C.axis("r(t+τ)", fmt="pct", digits=0),
            z=C.axis("r(t+2τ)", fmt="pct", digits=0), height=500,
            subtitle="Ein chaotischer Attraktor läge auf einer Fläche oder Kurve. Renditen bilden eine Wolke: dichter "
                     "Kern (ruhige Tage) und einzelne weit entfernte Punkte, meist nahe einer Achse – ein Schocktag "
                     "taucht in m Zuständen auf, jedes Mal in einer anderen Koordinate. Rauschen mit fetten Rändern.",
        ))
    else:
        out.chart(C.xy(
            f"Rekonstruierter Phasenraum der Tagesrenditen (Delay-Einbettung, {len(emb)} Zustände)",
            [C.trace("Zustand (r(t), r(t+τ))", emb[:, 0], emb[:, 1], type="scatter", color=1, size=3,
                     decimate_to=None, text=lab)],
            x=C.axis("r(t)", fmt="pct", digits=0), y=C.axis("r(t+τ)", fmt="pct", digits=0), height=400,
            subtitle="Ein chaotisches System läge auf einer Kurve (die logistische Abbildung auf einer Parabel). "
                     "Renditen bilden eine Wolke mit Ausläufern entlang der Achsen.",
        ))

    # --- tables ---------------------------------------------------------------------------
    rows = [
        ["Lyapunov-Exponent λ (Tage)", "num", lam, float(np.nanmean(lam_sur)),
         f"{_de(lam_lo, 3)} … {_de(lam_hi, 3)}", _rank_text(lam, lam_sur)],
        [f"Determinismus DET ({units})", "pct", det, float(np.mean(det_s)),
         f"{_de(100 * det_s.min(), 1)} … {_de(100 * det_s.max(), 1)} %", _rank_text(det, det_s)],
        [f"Laminarität LAM ({units})", "pct", lamr, float(np.mean(lam_s)),
         f"{_de(100 * lam_s.min(), 1)} … {_de(100 * lam_s.max(), 1)} %", _rank_text(lamr, lam_s)],
        ["Stabilitätsindex DET/RR", "num", float(rqa["stability_index"]), float(np.mean(stab_s)),
         f"{_de(stab_s.min())} … {_de(stab_s.max())}", _rank_text(float(rqa["stability_index"]), stab_s)],
    ]
    out.table("Surrogat-Test: echte gegen zufällig gemischte Renditen",
              ["Kennzahl", "Renditen", "Surrogate Ø", "Surrogate Spanne", "Einordnung"],
              [[row[0], _fmt(row[2], row[1]), _fmt(row[3], row[1]), row[4], row[5]] for row in rows],
              formats=["text", "text", "text", "text", "text"],
              note=f"λ aus {n} Tagesrenditen, RQA aus den letzten {len(x)} {scale_txt} (N = {int(rqa['n_points'])} "
                   f"Zustände, Rekurrenzrate fest {_de(100 * q, 0)} %, Linien ab 3 Punkten, ohne Hauptdiagonale). "
                   f"Surrogate: {n_sur} zufällige Permutationen (Seed {seed}). Liegt ein Wert über allen Surrogaten, "
                   "steckt zeitliche Struktur in der Reihe.")
    out.table("λ je Einbettungsdimension", ["m", "Renditen", "gemischt", "Gauß-Rauschen", "logistische Abbildung"],
              [[mm] + [_de(curves[k][i], 3) for k in ("r", "s", "g", "l")] for i, mm in enumerate(ms)],
              formats=["int", "text", "text", "text", "text"],
              note="Rosenstein-artiger Schätzer des Originals: Ø von log(dₜ/d₀)/t für t = 1 … 9 Schritte, nächster "
                   "Nachbar mindestens 6 Schritte entfernt.")

    # --- the honest reading --------------------------------------------------------------
    if fallback:
        out.note(f"Für Wochenrenditen ist die Reihe zu kurz – der Recurrence-Plot nutzt die letzten {len(x)} "
                 "Tagesrenditen.", "warning")
    if lam <= lam_hi:
        out.note(f"**Kein Hinweis auf deterministisches Chaos.** λ = {_de(lam, 3)} liegt "
                 + ("im Bereich der" if lam >= lam_lo else "sogar unter allen")
                 + f" gemischten Renditen ({_de(lam_lo, 3)} … {_de(lam_hi, 3)}). Das „chaotisch“ des Originals "
                 "bekommt jede verrauschte Reihe – auch Gauß-Rauschen.", "success")
    else:
        out.note(f"λ = {_de(lam, 3)} liegt über allen Surrogaten. Das spricht für schnellere Trennung als beim "
                 "Rauschen – prüfe mit anderem m, τ und Seed, ob das stabil ist, bevor du es Chaos nennst.", "warning")
    if lam < lam_lo:
        out.note("Dass λ sogar etwas **unter** den Surrogaten liegt, ist Volatilitäts-Clustering: In ruhigen Phasen "
                 "liegen Nachbarn dicht beieinander und bleiben es ein paar Tage. Das macht die Schwankungsbreite "
                 "etwas vorhersagbar, nicht die Richtung.")
    above = [name for name, v, s in (("DET", det, det_s), ("LAM", lamr, lam_s)) if v > s.max()]
    if above:
        out.note(f"**Zeitliche Struktur im Recurrence-Plot:** {' und '.join(above)} "
                 f"{'liegen' if len(above) > 1 else 'liegt'} über allen {n_sur} Surrogaten. Ruhige und turbulente "
                 "Phasen wechseln sich in Blöcken ab (Volatilitäts-Clustering) – genau das, was GARCH- und "
                 "Regime-Modelle beschreiben. Für die Richtung der Kurse folgt daraus nichts.")
    else:
        out.note(f"DET und LAM liegen im Bereich der Surrogate: In diesen {len(x)} {dat} zeigt der Recurrence-Plot "
                 "keine Struktur, die gemischte Renditen nicht auch hätten. "
                 + ("Eine längere Reihe mit ruhigen und turbulenten Phasen zeigt Volatilitäts-Cluster deutlicher. "
                    if fallback else
                    "Wochenrenditen über mehrere Jahre zeigen Volatilitäts-Cluster meist deutlicher. "
                    if not weekly else "Ein Zeitraum mit Krise zeigt Volatilitäts-Cluster meist deutlicher. ")
                 + "Bei unabhängig simulierten Renditen ist genau dieses Ergebnis zu erwarten.")

    out.progress("Original-Grafik wird gezeichnet …", 0.9)
    fig = rpa.plot_recurrence(figsize=(7, 7), epsilon=eps)
    out.figure(fig, f"Original: RecurrencePlotAnalyzer.plot_recurrence – volle Auflösung, {len(R)} × {len(R)} "
                    f"Zustände der {scale_txt}, schwarz = Rekurrenz (Zeit läuft nach rechts und nach oben).", dpi=100)


def _fmt(v, kind):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{_de(100 * v, 1)} %" if kind == "pct" else _de(v, 3)
