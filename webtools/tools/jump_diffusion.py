"""Jump-diffusion risk: split returns into jumps and diffusion, decompose the VaR, OU coordinates of |r|."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import JumpDiffusionRiskAnalyzer, compute_ou_coordinates

from .. import charts as C
from ..core import ToolError
from ..data import index_axis

SPEC = {
    "id": "jump-diffusion",
    "title": "Sprünge vs. Diffusion: VaR-Zerlegung",
    "group": "risk",
    "order": 60,
    "summary": "Welcher Teil des Tail-Risikos kommt aus seltenen Sprüngen und welcher aus dem normalen "
               "Rauschen? Sprungerkennung, Monte-Carlo-VaR-Zerlegung und OU-Koordinaten.",
    "description": (
        "Das Modell zerlegt deine Renditen in zwei Teile: **Sprünge** sind alle Tage, die weiter als die "
        "Schwelle (in Standardabweichungen) vom Mittel entfernt liegen; der Rest ist die **Diffusion**. Auf die "
        "ruhigen Tage wird ein Ornstein-Uhlenbeck-Prozess gefittet, die Sprünge liefern Häufigkeit, Mittel und "
        "Streuung. Eine Monte-Carlo-Simulation des Modells mit und ohne Sprünge – mit **denselben Zufallszahlen** "
        "für die Diffusion – zeigt dann, wie viel des Tages-VaR auf die Sprünge entfällt.\n\n"
        "So liest du die Diagramme: Rote Punkte sind erkannte Sprünge. Das Balkendiagramm je Jahr zeigt, ob die "
        "Sprünge gleichmäßig kommen (wie es das Poisson-Modell annimmt) oder in Krisen gehäuft – meist "
        "Letzteres. Die Zerlegung je Schwelle zeigt, wie stark das Ergebnis von dieser Definition abhängt. "
        "Die OU-Koordinaten fitten in jedem rollierenden Fenster einen OU-Prozess auf |Rendite|: X ist das "
        "Vola-Niveau, Y der aktuelle Ausschlag, Z die Rückstellgeschwindigkeit.\n\n"
        "Hinweis: Tagesrenditen haben kaum Gedächtnis. Der OU-Fit landet deshalb meist im Grenzfall "
        "„weißes Rauschen“ (θ wird gedeckelt, Halbwertszeit ≈ 0,1 Tage) – die Diffusion entspricht dann einer "
        "Normalverteilung mit der Streuung der ruhigen Tage."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 3.0, "min": 2.0, "max": 6.0, "step": 0.1,
         "label": "Sprungschwelle", "unit": "σ", "help": "Tage jenseits von Mittel ± Schwelle·σ gelten als Sprung."},
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999, "step": 0.001,
         "label": "Konfidenzniveau"},
        {"name": "n_simulations", "type": "int", "default": 10000, "min": 2000, "max": 50000, "step": 1000,
         "label": "Monte-Carlo-Szenarien"},
        {"name": "window", "type": "int", "default": 30, "min": 10, "max": 120, "step": 5,
         "label": "Fenster der OU-Koordinaten", "unit": "Tage"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer",
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer.fit",
        "src.physics.ornstein_uhlenbeck.JumpDiffusionRiskAnalyzer.tail_risk_decomposition",
        "src.physics.ornstein_uhlenbeck.compute_ou_coordinates",
    ],
    "cost": "fast",
}

FIT_STATUS = {
    "ok": "regulär",
    "theta_capped": "weißes Rauschen (θ gedeckelt)",
    "no_mean_reversion": "keine Rückkehr zum Mittel",
    "too_few_points": "zu wenige ruhige Tage",
}
SWEEP = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _date(d):
    """ISO date -> German date; trading-day indices stay as they are."""
    return f"{d[8:10]}.{d[5:7]}.{d[:4]}" if isinstance(d, str) and len(d) >= 10 else d


def _level(conf):
    return f"{100 * conf:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


def run(params, data, out):
    r = data.returns
    n = data.n
    thr, conf, n_sim, seed = params["threshold"], params["confidence"], params["n_simulations"], params["seed"]
    if n < 2 * params["window"] + 20:
        raise ToolError(f"Zu wenige Beobachtungen ({n}) für ein Koordinaten-Fenster von {params['window']} Tagen.")

    ja = JumpDiffusionRiskAnalyzer()
    dec = ja.tail_risk_decomposition(r, confidence=conf, threshold=thr, n_simulations=n_sim, seed=seed)
    has_model = ja.ou_model is not None
    mean, std = float(np.mean(r)), float(np.std(r))
    is_jump = np.abs(r - mean) > thr * std  # the analyzer's own definition, for display
    var_hist = -float(np.percentile(r, (1 - conf) * 100))
    var_gauss = -(mean + stats.norm.ppf(1 - conf) * std)
    lvl = _level(conf)
    share = dec["var_jump"] / dec["var_total"] if dec["var_total"] > 0 else None

    out.metric(f"VaR {lvl} Modell gesamt", dec["var_total"], "pct", hint="OU + Sprünge, 1 Tag")
    out.metric("davon Diffusion", dec["var_continuous"], "pct", hint="dasselbe Modell ohne Sprünge")
    out.metric("davon Sprünge", dec["var_jump"], "pct",
               hint=f"{_de(100 * share, 0)} % des VaR" if share is not None else "",
               status="serious" if share is not None and share > 0.25 else None)
    out.metric(f"Historischer VaR {lvl}", var_hist, "pct", hint=f"Normalverteilung: {_de(100 * var_gauss)} %")
    out.metric("Erkannte Sprünge", int(dec["n_jumps"]), "int",
               hint=f"{_de(dec['jump_intensity'] * 252, 1)} pro Jahr ({_de(100 * dec['jump_intensity'], 2)} % der Tage)")
    out.metric("Ø-Sprung", dec["jump_mean"], "pct", hint=f"Streuung {_de(100 * dec['jump_std'])} %")
    out.metric("Anteil der Sprünge an Σ|r|", dec["jump_contribution"], "pct", digits=1,
               hint="so viel der gesamten Bewegung steckt in den Sprungtagen")
    if has_model:
        out.metric("Streuung der ruhigen Tage", ja.ou_model.stationary_std(), "pct",
                   hint=f"stationäre Std des OU · Fit: {FIT_STATUS[dec['ou_fit_status']]}")
    else:
        out.metric("OU-Fit", FIT_STATUS[dec["ou_fit_status"]], "text", status="warning",
                   hint="ohne Modell ist der VaR der historische")

    x, xa = index_axis(data)
    out.chart(C.xy(
        f"Tagesrenditen mit erkannten Sprüngen (jenseits von {_de(thr, 1)} σ)",
        [C.trace("Tagesrendite", x, r, color=1, width=1),
         C.trace("Sprung", [xi for xi, j in zip(x, is_jump) if j], r[is_jump], type="scatter", color="critical",
                 size=7, decimate_to=None)],
        x=xa, y=C.axis("Rendite", fmt="pct", digits=1),
        hlines=[C.hline(mean + thr * std, f"+{_de(thr, 1)} σ"), C.hline(mean - thr * std, f"−{_de(thr, 1)} σ")],
    ))

    # Jumps per (calendar) year: Poisson would spread them evenly
    if data.dates:
        years, inv = np.unique([d[:4] for d in data.dates], return_inverse=True)
        years = [str(y) for y in years]
    else:
        inv = np.arange(n) // 252
        years = [f"Jahr {i + 1}" for i in range(int(inv[-1]) + 1)]
    counts = np.bincount(inv, weights=is_jump.astype(float), minlength=len(years))
    days = np.bincount(inv, minlength=len(years))
    expected = dec["jump_intensity"] * days
    disp = float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0 else None
    out.chart(C.xy(
        "Sprünge je Jahr",
        [C.trace("Sprünge", years, counts, type="bar", color=1),
         C.trace("Poisson-Erwartung", years, expected, color="muted", dash=True)],
        x=C.axis("Jahr", type="category"), y=C.axis("Anzahl", fmt="int"), height=300,
        subtitle=(f"Streuungsindex Var/Mittel = {_de(disp, 1)} (Poisson: 1). Größer heißt: Sprünge kommen gehäuft."
                  if disp is not None else "Keine Sprünge erkannt."),
    ))

    # Decomposition across thresholds
    out.progress("VaR-Zerlegung für mehrere Schwellen …", 0.5)
    sweep = sorted(set(SWEEP + [round(thr, 2)]))
    sw = [JumpDiffusionRiskAnalyzer().tail_risk_decomposition(r, confidence=conf, threshold=t,
                                                              n_simulations=min(n_sim, 20000), seed=seed)
          for t in sweep]
    labels = [f"{_de(t, 1)} σ" + (" ◆" if abs(t - thr) < 1e-9 else "") for t in sweep]
    out.chart(C.xy(
        "VaR-Zerlegung je Sprungschwelle",
        [C.trace("Diffusion", labels, [d["var_continuous"] for d in sw], type="bar", color=1),
         C.trace("Sprung-Aufschlag", labels, [d["var_jump"] for d in sw], type="bar", color=2)],
        x=C.axis("Sprungschwelle (◆ = deine Wahl)", type="category"), y=C.axis(f"VaR {lvl}", fmt="pct", digits=1),
        hlines=[C.hline(var_hist, "historischer VaR")], stacked=True, height=320,
        subtitle="Niedrige Schwelle: mehr Tage gelten als Sprung, die Diffusion wird ruhiger – und umgekehrt.",
    ))

    # OU coordinates of |r|
    X, Y, Z = compute_ou_coordinates(r, window=params["window"])
    valid = np.arange(2 * params["window"], n)
    xs = [x[i] for i in valid]
    out.chart(C.xy(
        f"OU-Koordinaten von |Rendite| (Fenster {params['window']} Tage)",
        [C.trace("Z: Rückstellgeschwindigkeit θ", xs, Z[valid], color=2, width=1),
         C.trace("Y: aktueller Ausschlag |r|", xs, Y[valid], color="muted", width=1),
         C.trace("X: Vola-Niveau μ", xs, X[valid], color=1, width=2.5)],
        x=xa, y=C.axis("normiert (0 = Minimum, 1 = Maximum)", range=(0, 1.02)), height=320,
        subtitle="Fenster ohne gültigen OU-Fit (negative Autokorrelation) setzt das Framework auf θ = 1.",
    ))

    rows = [["Modell: OU + Sprünge", dec["var_total"]], ["nur Diffusion (OU)", dec["var_continuous"]],
            ["Sprung-Aufschlag", dec["var_jump"]], ["Historisch (empirisches Quantil)", var_hist],
            ["Normalverteilung (μ − z·σ)", var_gauss]]
    out.table(f"Tages-VaR {lvl} im Vergleich", ["Methode", "VaR"], rows, formats=["text", "pct"],
              note=f"{n_sim} Szenarien, Seed {seed}. OU-Fit: {FIT_STATUS[dec['ou_fit_status']]}.")
    order = np.argsort(-np.abs(r - mean))
    top = [i for i in order if is_jump[i]][:15]
    if top:
        out.table("Größte Sprünge", ["Datum" if data.dates else "Tag", "Rendite", "σ-Vielfaches"],
                  [[_date(x[i]), r[i], abs(r[i] - mean) / std] for i in top], formats=["text", "pct", "num"])

    if not has_model:
        out.note("Es gab zu wenige ruhige Tage oder keine Rückkehr zum Mittel – ohne OU-Modell ist der "
                 "Modell-VaR einfach der historische VaR.", "warning")
        return
    if var_hist > 0 and dec["var_total"] < 0.85 * var_hist:
        out.note(f"Das Modell liegt {_de(100 * (1 - dec['var_total'] / var_hist), 0)} % unter dem historischen VaR. "
                 "Es kennt keine Volatilitäts-Cluster: Ruhige und stürmische Phasen werden zu einer einzigen "
                 "Diffusion gemittelt, und Sprünge kommen gleichmäßig statt gehäuft.", "warning")
    if int(dec["n_jumps"]) == 0:
        out.set_verdict("good", f"Keine Sprünge jenseits von {_de(thr, 1)} σ",
                        "Das Tail-Risiko ist hier vollständig Diffusion.")
        return
    if share is None:
        return
    if share > 0.25:
        out.set_verdict("serious", "Ein großer Teil des Tail-Risikos steckt in Sprüngen",
                        f"{_de(100 * share, 0)} % des Tages-VaR entfallen auf Sprünge – ein reines Diffusionsmodell "
                        "unterschätzt das Risiko deutlich.")
    elif share > 0.1:
        out.set_verdict("warning", "Sprünge erhöhen den VaR spürbar",
                        f"{_de(100 * share, 0)} % des Tages-VaR entfallen auf Sprünge.")
    else:
        out.set_verdict("good", "Der VaR wird überwiegend von der Diffusion bestimmt",
                        f"Nur {_de(100 * max(share, 0), 0)} % des Tages-VaR entfallen auf Sprünge. Für "
                        "Expected Shortfall und längere Horizonte kann das anders aussehen.")
