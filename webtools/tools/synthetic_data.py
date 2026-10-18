"""Synthetische Daten: the framework's six return models side by side, the crisis scenario and the VIX proxy."""

import numpy as np
from scipy import stats

from src.utils.data_loader import (
    generate_crisis_scenario,
    generate_synthetic_returns,
    generate_vix_from_returns,
    load_sample_data,
)
from src.utils.statistics import compute_moments

from .. import charts as C

SPEC = {
    "id": "synthetic-data",
    "title": "Synthetische Märkte im Vergleich",
    "group": "data",
    "order": 60,
    "summary": "Die sechs Renditemodelle des Frameworks nebeneinander – Gauß, Student-t, Lévy, GARCH, "
               "Regimewechsel und Lévy-Sprünge –, dazu das Krisenszenario mit abgeleitetem VIX.",
    "description": (
        "Das Framework erzeugt künstliche Tagesrenditen mit `generate_synthetic_returns` in sechs Modellen: "
        "**Normalverteilung** (die Referenz), **Student-t** mit ν = 4, **Lévy-stabil** mit α = 1,7 (unendliche "
        "Varianz), **GARCH(1,1)** (Volatilitäts-Cluster), **Regimewechsel** zwischen ruhig und Stress und "
        "**Gauß + Lévy-Sprünge** (das realistischste). Alle nutzen denselben Seed und dieselbe Länge.\n\n"
        "So liest du die Diagramme: Die Kursverläufe zeigen, wie verschieden Pfade bei ähnlicher Tagesvola "
        "aussehen. Die Dichte ist **robust standardisiert** (Median und MAD), damit alle Modelle in der Mitte "
        "gleich breit sind – auf der logarithmischen Achse siehst du dann nur noch die Ränder: Je weiter eine "
        "Kurve über der Gauß-Kurve liegt, desto häufiger sind Crash-Tage. `generate_crisis_scenario` baut "
        "einen Markt aus ruhiger Phase, Krise (10 % Schock, 40 % Hauptkrise, 50 % Nachbeben) und Erholung; "
        "`generate_vix_from_returns` leitet daraus einen VIX ab.\n\n"
        "Jedes Modell kannst du als Datensatz für alle anderen Werkzeuge nutzen: oben bei **Datensatz → "
        "Ändern**, Reiter „Synthetisch“ bzw. „Krisenszenario“."
    ),
    "needs": [],
    "params": [
        {"name": "n", "type": "int", "default": 2520, "min": 250, "max": 10000, "step": 10,
         "label": "Länge je Modell", "unit": "Tage", "help": "2.520 Handelstage ≈ 10 Jahre."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 999999, "step": 1, "label": "Zufalls-Seed"},
        {"name": "n_pre", "type": "int", "default": 500, "min": 50, "max": 2000, "step": 10,
         "label": "Krisenszenario: ruhige Tage davor", "unit": "Tage"},
        {"name": "n_crisis", "type": "int", "default": 100, "min": 10, "max": 500, "step": 5,
         "label": "Krisenszenario: Krisentage", "unit": "Tage"},
        {"name": "n_post", "type": "int", "default": 400, "min": 50, "max": 2000, "step": 10,
         "label": "Krisenszenario: Tage danach", "unit": "Tage"},
    ],
    "covers": [
        "src.utils.data_loader.generate_synthetic_returns",
        "src.utils.data_loader.generate_crisis_scenario",
        "src.utils.data_loader.generate_vix_from_returns",
        "src.utils.data_loader.load_sample_data",
        "src.utils.statistics.compute_moments",
    ],
    "cost": "fast",
}
ORDER = ["gaussian", "student_t", "levy", "garch", "regime_switching", "levy_jump"]
SHORT = {"gaussian": "Normalverteilung", "student_t": "Student-t (ν = 4)", "levy": "Lévy-stabil (α = 1,7)",
         "garch": "GARCH(1,1)", "regime_switching": "Regimewechsel", "levy_jump": "Gauß + Lévy-Sprünge"}


def _max_dd(r):
    w = np.exp(np.cumsum(r))
    w = np.concatenate([[1.0], w])
    return float(np.min(w / np.maximum.accumulate(w) - 1))


def _row(name, r):
    m = compute_moments(r)
    sd = float(m["std"])
    return [name, sd * np.sqrt(252), float(m["skewness"]), float(m["kurtosis"]), float(np.min(r)), float(np.max(r)),
            _max_dd(r), int(np.sum(np.abs(r - m["mean"]) > 4 * sd))]


def run(params, data, out):
    n, seed = int(params["n"]), int(params["seed"])
    series = {m: np.asarray(generate_synthetic_returns(n, m, seed=seed), dtype=float) for m in ORDER}
    days = list(range(n))

    rows = [_row(SHORT[m], series[m]) for m in ORDER]
    kurt = {m: r[3] for m, r in zip(ORDER, rows)}
    fattest = max(kurt, key=kurt.get)
    out.metric("Modelle", len(ORDER), "int", hint=f"je {n} Handelstage, Seed {seed}")
    out.metric("Fetteste Ränder", SHORT[fattest], "text", hint=f"Exzess-Kurtosis {kurt[fattest]:.1f}".replace(".", ","))
    out.metric("Gauß: Exzess-Kurtosis", kurt["gaussian"], "num", digits=2, hint="Referenz ≈ 0")
    out.metric("Schlechtester Tag (alle)", min(r[4] for r in rows), "pct", digits=1)

    # ---- price paths --------------------------------------------------------------------------------
    paths = {m: 100 * np.exp(np.cumsum(series[m])) for m in ORDER}
    lo_p, hi_p = min(float(v.min()) for v in paths.values()), max(float(v.max()) for v in paths.values())
    out.chart(C.xy(
        "Kursverläufe der sechs Modelle",
        [C.trace(SHORT[m], days, paths[m], color="muted" if m == "gaussian" else i, width=1.2)
         for i, m in enumerate(ORDER)],
        x=C.axis("Handelstag", fmt="int"),
        y=C.axis("Kurs (Start 100)", type="log" if hi_p > 10 * lo_p else "linear"),
        subtitle="Gleiche Länge, gleicher Seed. Lévy- und Sprungmodelle springen, GARCH und Regimewechsel "
                 "haben unruhige Phasen.",
    ))

    # ---- robustly standardised densities on a log axis ------------------------------------------------
    edges = (-12, 12)
    traces, floor = [], np.inf
    for i, m in enumerate(ORDER):
        r = series[m]
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        z = (r - np.median(r)) / scale
        centers, h = C.histogram(z, bins=97, density=True, range=edges)
        floor = min(floor, float(np.min(h[h > 0])))
        traces.append(C.trace(SHORT[m], centers, np.where(h > 0, h, np.nan), color="muted" if m == "gaussian" else i,
                              width=1.4, dash=m == "gaussian"))
    grid = np.linspace(*edges, 400)
    pdf = stats.norm.pdf(grid)
    traces.append(C.trace("Gauß-Kurve (Theorie)", grid, np.where(pdf >= floor / 4, pdf, np.nan), color="ink",
                          width=1, dash=True))
    out.chart(C.xy(
        "Verteilung der Tagesrenditen – logarithmisch",
        traces, x=C.axis("Rendite in robusten Standardabweichungen (Median ± MAD)"),
        y=C.axis("Dichte", type="log", range=(floor / 4, 1.0)),
        subtitle="In der Mitte gleich breit, an den Rändern grundverschieden: Fette Ränder liegen weit über "
                 "der Gauß-Kurve.",
    ))
    out.table("Kennzahlen der Modelle",
              ["Modell", "Vola p. a.", "Schiefe", "Exzess-Kurtosis", "Schlechtester Tag", "Bester Tag",
               "Max. Drawdown", "Tage jenseits 4 σ"], rows,
              formats=["text", "pct", "num", "num", "pct", "pct", "pct", "int"],
              note=f"Eine Normalverteilung hätte bei {n} Tagen im Mittel "
                   f"{n * 2 * stats.norm.sf(4):.2f}".replace(".", ",") + " Tage jenseits von 4 σ.")

    # ---- crisis scenario + VIX proxy ------------------------------------------------------------------
    cr, meta = generate_crisis_scenario(n_pre=int(params["n_pre"]), n_crisis=int(params["n_crisis"]),
                                        n_post=int(params["n_post"]), seed=seed)
    cr = np.asarray(cr, dtype=float)
    a, b, peak = int(meta["crisis_start"]), int(meta["crisis_end"]), int(meta["crisis_peak"])
    cdays = list(range(len(cr)))
    vix = np.asarray(generate_vix_from_returns(cr), dtype=float)
    band = [C.band(a, b, "Krise", "critical")]
    out.chart(C.xy(
        "Krisenszenario: Kurs",
        [C.trace("Kurs (Start 100)", cdays, 100 * np.exp(np.cumsum(cr)), color=1, width=1.4)],
        x=C.axis("Handelstag", fmt="int"), y=C.axis("Kurs"), bands=[C.band(a, b, "", "critical")],
        vlines=[C.vline(peak, "Krisen-Peak laut Metadaten")],
        subtitle=f"Rot hinterlegt: Krise. Ruhig ({a} T) → Krise ({b - a} T) → Erholung ({len(cr) - b} T).",
    ))
    out.chart(C.xy(
        "Krisenszenario: abgeleiteter VIX",
        [C.trace("VIX aus Renditen", cdays, vix, color=2, width=1.4)],
        x=C.axis("Handelstag", fmt="int"), y=C.axis("VIX-Punkte"), bands=band,
        subtitle="generate_vix_from_returns: 15 + 100 × rollierende Jahresvola, Sprung bei Tagesverlusten "
                 "über 2 %, begrenzt auf 10–80.",
    ))
    phases = [("Vor der Krise", cr[:a]), ("Krise", cr[a:b]), ("Danach", cr[b:])]
    prow = [[name, len(r), float(np.expm1(r.sum())) if r.size else None, float(np.std(r) * np.sqrt(252)) if r.size else None,
             float(np.min(r)) if r.size else None, _max_dd(r) if r.size else None] for name, r in phases]
    out.table("Phasen des Krisenszenarios", ["Phase", "Tage", "Rendite", "Vola p. a.", "Schlechtester Tag",
                                             "Max. Drawdown"], prow, formats=["text", "int", "pct", "pct", "pct", "pct"])
    if prow[1][2] is not None and prow[1][2] > 0:
        out.note(f"Bei Seed {seed} steigt der Kurs in der Krisenphase ({prow[1][2] * 100:+.0f} %): Die "
                 "Hauptkrise zieht aus einer t-Verteilung mit ν = 3, deren Ränder auf beiden Seiten fett sind. "
                 "Probier einen anderen Seed für einen echten Crash.", "warning")

    # ---- the framework's fixed sample bundle -----------------------------------------------------------
    sample = load_sample_data()
    sm = sample["crisis_metadata"]
    srows = []
    for key, model, seed_s in (("normal", "GARCH(1,1)", 42), ("crisis", "Krisenszenario", 43),
                               ("fat_tails", "Lévy-stabil", 44), ("regime_switching", "Regimewechsel", 45)):
        r = np.asarray(sample[key], dtype=float)
        mom = compute_moments(r)
        hint = ""
        if key == "crisis":
            cret = float(np.expm1(r[sm["crisis_start"]:sm["crisis_end"]].sum()))
            hint = f"Krisenfenster Tag {sm['crisis_start']}–{sm['crisis_end']}: {cret * 100:+.0f} %"
        srows.append([key, f"{model} · Seed {seed_s}", len(r), float(mom["std"]) * np.sqrt(252), float(mom["kurtosis"]),
                      float(np.expm1(r.sum())), hint])
    for key, src in (("vix_normal", "aus „normal“"), ("vix_crisis", "aus „crisis“")):
        v = np.asarray(sample[key], dtype=float)
        srows.append([key, f"VIX {src}", len(v), None, None, None,
                      f"Mittel {v.mean():.1f}, Spitze {v.max():.1f}".replace(".", ",")])
    sample_crisis = float(np.expm1(np.asarray(sample["crisis"])[sm["crisis_start"]:sm["crisis_end"]].sum()))
    if sample_crisis > 0:
        out.note(f"Achtung: Die Beispielreihe „crisis“ (Seed 43) steigt in ihrem Krisenfenster um "
                 f"{sample_crisis * 100:.0f} %".replace(".", ",") + ". Die Hauptkrise zieht aus einer symmetrischen "
                 "t-Verteilung (ν = 3) mit sehr großer Skala, deren Schwankung die negative Drift überdeckt – bei "
                 "rund einem Fünftel der Seeds wird so aus der Krise eine Rally. Seed 43 ist auch die Vorgabe des "
                 "Datensatzes „Krisenszenario“; wähle dort einen anderen Seed (z. B. 42) für einen echten Crash.",
                 "warning")
    out.table("Beispieldaten des Frameworks (load_sample_data)",
              ["Reihe", "Modell", "Länge", "Vola p. a.", "Exzess-Kurtosis", "Rendite gesamt", "Hinweis"], srows,
              formats=["text", "text", "int", "pct", "num", "pct", "text"],
              note="Feste Seeds 42–45: dieselben Reihen, die die Demos und Tests des Frameworks verwenden.")
    out.note("Nutze ein Modell als Datensatz für alle anderen Werkzeuge: oben bei **Datensatz → Ändern**, "
             "Reiter „Synthetisch“ (Modell, Länge und Seed frei) oder „Krisenszenario“.", "success")
