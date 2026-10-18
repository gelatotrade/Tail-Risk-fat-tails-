"""OU process with compound-Poisson jumps: jump types and their effect on VaR / Expected Shortfall."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import OUWithJumps

from .. import charts as C

SPEC = {
    "id": "ou-jumps",
    "title": "OU mit Sprüngen: Crash-Risiko im VaR",
    "group": "physics",
    "order": 45,
    "summary": "Was seltene Sprünge mit VaR und Expected Shortfall machen – normal-, doppelt-exponential- oder "
               "Lévy-verteilt, im direkten Vergleich zum Prozess ohne Sprünge.",
    "description": (
        "Ein mean-revertierender Prozess **dx = θ(μ − x)dt + σ dW + dJ**: Zwischen den Sprüngen verhält er sich "
        "wie ein Ornstein-Uhlenbeck-Prozess, dazu kommen im Schnitt **λ Sprünge pro Jahr** (Poisson). Denk an "
        "die Abweichung einer Position vom fairen Wert – etwa ein Spread oder ein Rohstoffpreis –, die "
        "normalerweise ruhig zurückläuft, aber gelegentlich abrupt springt. x ist eine Log-Abweichung, "
        "0,05 heißt also +5 %.\n\n"
        "So liest du die Diagramme: Die Beispielpfade markieren jeden Sprung. Die Verteilung der Änderung über "
        "den VaR-Horizont ist logarithmisch aufgetragen – dort sieht man, wie die Sprünge den Rand anheben, "
        "während die Mitte fast gleich bleibt. Beide Simulationen nutzen **dieselben Zufallszahlen** für die "
        "Diffusion, der Unterschied ist also allein der Effekt der Sprünge.\n\n"
        "Sprungverteilungen: **Normal** N(Mittel, Std), **doppelt-exponential** (Kou-Modell: gleich wahrscheinlich "
        "nach oben und unten, exponentiell verteilt) und **Lévy** (Student-t mit ν = 1,7 als Näherung – unendliche "
        "Varianz, einzelne Sprünge können absurd groß werden). Der Expected Shortfall reagiert stärker als der "
        "VaR, weil er den ganzen Rand jenseits des VaR mittelt."
    ),
    "needs": [],
    "params": [
        {"name": "theta", "type": "float", "default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1,
         "label": "Rückstellgeschwindigkeit θ", "unit": "pro Jahr"},
        {"name": "sigma", "type": "float", "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
         "label": "Diffusions-Volatilität σ", "unit": "pro √Jahr", "help": "0,2 = 20 % pro Jahr."},
        {"name": "jump_intensity", "type": "float", "default": 10.0, "min": 0.0, "max": 252.0, "step": 1.0,
         "label": "Sprünge pro Jahr λ"},
        {"name": "jump_distribution", "type": "select", "default": "normal", "label": "Sprungverteilung",
         "options": [{"value": "normal", "label": "Normal"},
                     {"value": "double_exponential", "label": "Doppelt-exponential (Kou)"},
                     {"value": "levy", "label": "Lévy (Student-t, ν = 1,7)"}]},
        {"name": "jump_mean", "type": "float", "default": -0.02, "min": -0.2, "max": 0.2, "step": 0.005,
         "label": "Mittlere Sprunghöhe", "help": "−0,02 = im Schnitt −2 % je Sprung."},
        {"name": "jump_std", "type": "float", "default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005,
         "label": "Streuung der Sprünge", "help": "Std (normal) bzw. Skala (Kou, Lévy)."},
        {"name": "horizon_days", "type": "int", "default": 1, "min": 1, "max": 21, "step": 1,
         "label": "VaR-Horizont", "unit": "Handelstage"},
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999, "step": 0.001,
         "label": "Konfidenzniveau"},
        {"name": "n_simulations", "type": "int", "default": 10000, "min": 2000, "max": 50000, "step": 1000,
         "label": "Simulationen für VaR/ES"},
        {"name": "path_years", "type": "float", "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25,
         "label": "Länge der Beispielpfade", "unit": "Jahre"},
        {"name": "n_paths", "type": "int", "default": 200, "min": 10, "max": 500, "step": 10,
         "label": "Anzahl Pfade", "help": "für Sprungstatistik und Pfad-Fächer"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.OUWithJumps",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.sample_jump_size",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.simulate",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.compute_var",
        "src.physics.ornstein_uhlenbeck.OUWithJumps.compute_expected_shortfall",
    ],
    "cost": "fast",
}

LEVELS = [0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
N_EXAMPLES = 3


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _level(conf):
    """0.99 -> '99 %', 0.975 -> '97,5 %'."""
    return f"{100 * conf:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


def _var_es(change, conf):
    """Same estimator as OUWithJumps.compute_var / compute_expected_shortfall, on a given sample."""
    var = -np.percentile(change, (1 - conf) * 100)
    tail = change[change < -var]
    return var, (-np.mean(tail) if len(tail) else var)


def run(params, data, out):
    kw = dict(theta=params["theta"], mu=0.0, sigma=params["sigma"], jump_mean=params["jump_mean"],
              jump_std=params["jump_std"], jump_distribution=params["jump_distribution"])
    lam = params["jump_intensity"]
    model = OUWithJumps(jump_intensity=lam, **kw)
    plain = OUWithJumps(jump_intensity=0.0, **kw)
    h = params["horizon_days"] / 252
    conf, n_sim, seed = params["confidence"], params["n_simulations"], params["seed"]

    # VaR / ES from the framework (same seed -> the same sample for both, and common random numbers)
    var_j = model.compute_var(0.0, h, conf, n_sim, seed=seed)
    es_j = model.compute_expected_shortfall(0.0, h, conf, n_sim, seed=seed)
    var_0 = plain.compute_var(0.0, h, conf, n_sim, seed=seed)
    es_0 = plain.compute_expected_shortfall(0.0, h, conf, n_sim, seed=seed)
    change_j = model.simulate(0.0, h, n_steps=1, n_paths=n_sim, seed=seed)["paths"][:, -1]
    change_0 = plain.simulate(0.0, h, n_steps=1, n_paths=n_sim, seed=seed)["paths"][:, -1]

    hz = f"{params['horizon_days']} Tag" + ("" if params["horizon_days"] == 1 else "e")
    lvl = _level(conf)
    out.metric(f"VaR {lvl} mit Sprüngen", var_j, "pct", hint=f"Verlust über {hz}",
               status="serious" if var_j > 1.5 * var_0 else None)
    out.metric(f"Expected Shortfall {lvl} mit Sprüngen", es_j, "pct", hint="Ø-Verlust jenseits des VaR",
               status="critical" if es_j > 2 * es_0 else "serious" if es_j > 1.3 * es_0 else None)
    out.metric("VaR ohne Sprünge", var_0, "pct", hint="reiner OU, dieselben Zufallszahlen")
    out.metric("Expected Shortfall ohne Sprünge", es_0, "pct")
    out.metric("Sprung-Faktor VaR", var_j / var_0 if var_0 > 0 else None, "x")
    out.metric("Sprung-Faktor ES", es_j / es_0 if es_0 > 0 else None, "x")
    out.metric("Erwartete Sprünge im Horizont", lam * h, "num", digits=3, hint=f"λ·Horizont, λ = {_de(lam, 0)} pro Jahr")
    out.metric("Exzess-Kurtosis der Änderung", stats.kurtosis(change_j), "num", digits=1,
               hint=f"ohne Sprünge: {_de(stats.kurtosis(change_0), 1)}")

    out.table("Risiko mit und ohne Sprünge", ["Kennzahl", "mit Sprüngen", "ohne Sprünge", "Aufschlag", "Faktor"],
              [["VaR " + lvl, var_j, var_0, var_j - var_0, var_j / var_0 if var_0 > 0 else None],
               ["Expected Shortfall " + lvl, es_j, es_0, es_j - es_0, es_j / es_0 if es_0 > 0 else None]],
              formats=["text", "pct", "pct", "pct", "x"],
              note=f"{n_sim} Monte-Carlo-Szenarien über {hz}, Start auf dem Mittel (x₀ = μ = 0).")

    # Example paths with jump markers
    sim = model.simulate(0.0, params["path_years"], n_steps=int(round(params["path_years"] * 252)),
                         n_paths=params["n_paths"], seed=seed + 1)
    paths, times = sim["paths"], sim["times"]
    dt = times[1] - times[0]
    traces = []
    for k in range(min(N_EXAMPLES, len(paths))):
        traces.append(C.trace(f"Pfad {k + 1}", times, paths[k], color=k + 1, width=1.3))
    sel = sim["jump_paths"] < N_EXAMPLES
    if np.any(sel):
        steps = np.rint(sim["jump_times"][sel] / dt).astype(int)
        jp = sim["jump_paths"][sel]
        traces.append(C.trace("Sprünge", sim["jump_times"][sel], paths[jp, steps], type="scatter", color="critical",
                              size=8, decimate_to=None,
                              text=[f"Pfad {p + 1}: Sprung {_de(100 * s, 1)} %" for p, s in zip(jp, sim["jump_sizes"][sel])]))
    out.chart(C.xy(
        "Beispielpfade mit Sprüngen",
        traces, x=C.axis("Zeit (Jahre)"), y=C.axis("Abweichung x", fmt="pct", digits=0),
        hlines=[C.hline(0.0, "μ")], height=360,
        subtitle="Nach jedem Sprung zieht die Rückstellkraft den Pfad zurück – Halbwertszeit "
                 f"{_de(model.base_ou.half_life() * 252, 0)} Handelstage.",
    ))

    # Distribution of the change over the VaR horizon (log density)
    lo, hi = np.percentile(np.concatenate([change_j, change_0]), [0.05, 99.95])
    cj, hj = C.histogram(change_j, bins=80, range=(lo, hi))
    c0, h0 = C.histogram(change_0, bins=80, range=(lo, hi))
    positive = np.concatenate([hj[hj > 0], h0[h0 > 0]])
    floor = float(positive.min()) / 3 if len(positive) else 1e-3
    out.chart(C.xy(
        f"Änderung über {hz}: mit vs. ohne Sprünge",
        [C.trace("mit Sprüngen", cj, np.where(hj > 0, hj, np.nan), type="bar", color=1),
         C.trace("ohne Sprünge", c0, np.where(h0 > 0, h0, np.nan), color="muted", width=2)],
        x=C.axis("Änderung von x", fmt="pct", digits=1),
        y=C.axis("Dichte (log)", type="log", range=(floor, float(max(hj.max(), h0.max())) * 3)),
        vlines=[C.vline(-var_j, "VaR mit Sprüngen", color="critical"), C.vline(-var_0, "VaR ohne", color="muted")],
        height=340, subtitle="Logarithmisch aufgetragen: Die Sprünge heben den Rand an, die Mitte bleibt fast gleich.",
    ))

    # VaR and ES across confidence levels (same samples as above)
    labels = [_level(c) for c in LEVELS]
    vj, ej = zip(*[_var_es(change_j, c) for c in LEVELS])
    v0, e0 = zip(*[_var_es(change_0, c) for c in LEVELS])
    out.chart(C.xy(
        "VaR und Expected Shortfall nach Konfidenzniveau",
        [C.trace("ES mit Sprüngen", labels, ej, color=1, width=2.5),
         C.trace("VaR mit Sprüngen", labels, vj, color=2, width=2.5),
         C.trace("ES ohne Sprünge", labels, e0, color=1, dash=True),
         C.trace("VaR ohne Sprünge", labels, v0, color=2, dash=True)],
        x=C.axis("Konfidenzniveau", type="category"), y=C.axis(f"Verlust über {hz}", fmt="pct", digits=1),
        height=320, subtitle="Je weiter man in den Rand schaut, desto mehr dominieren die Sprünge.",
    ))

    sizes = sim["jump_sizes"]
    expected = lam * params["path_years"] * params["n_paths"]
    if len(sizes) >= 20:
        a, b = np.percentile(sizes, [0.5, 99.5])
        cs, hs = C.histogram(sizes, bins=50, range=(a, b))
        out.chart(C.xy(
            f"Sprunghöhen: {len(sizes)} simulierte Sprünge",
            [C.trace("Sprünge", cs, hs, type="bar", color=3)],
            x=C.axis("Sprunghöhe", fmt="pct", digits=0), y=C.axis("Dichte"), height=280,
            subtitle=f"Erwartet: {_de(expected, 0)} Sprünge in {params['n_paths']} Pfaden × "
                     f"{_de(params['path_years'], 2)} Jahren. Dargestellt: 0,5- bis 99,5-%-Quantil.",
        ))
    elif lam > 0:
        out.note(f"In den Beispielpfaden gab es nur {len(sizes)} Sprünge (erwartet {_de(expected, 1)}) – "
                 "erhöhe λ, die Pfadlänge oder die Anzahl der Pfade für ein Histogramm der Sprunghöhen.")
    if params["jump_distribution"] == "levy":
        big = float(np.max(np.abs(sizes))) if len(sizes) else 0.0
        out.note(f"Lévy-Sprünge haben unendliche Varianz: Der größte simulierte Sprung war {_de(100 * big, 0)} %. "
                 "VaR und ES schwanken deshalb stark mit dem Seed.", "warning")
