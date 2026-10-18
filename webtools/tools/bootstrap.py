"""Wie sicher sind die Zahlen? Bootstrap confidence intervals for VaR, Expected Shortfall, kurtosis and tail index."""

import numpy as np
from scipy import stats

from src.models.risk_metrics import TailRiskMetrics
from src.physics.levy_flight import estimate_tail_index
from src.utils.numerical import numerical_gradient, numerical_hessian
from src.utils.statistics import bootstrap_confidence_interval, compute_moments

from .. import charts as C

SPEC = {
    "id": "bootstrap",
    "title": "Bootstrap-Konfidenzintervalle",
    "group": "data",
    "order": 50,
    "summary": "Wie stark VaR, Expected Shortfall, Kurtosis und Tail-Index mit einer anderen, gleich "
               "plausiblen Geschichte schwanken würden – als Konfidenzintervall.",
    "description": (
        "Jede Risikokennzahl ist eine Schätzung aus einer einzigen Geschichte. Der **Bootstrap** "
        "(`bootstrap_confidence_interval`) zieht aus deinen Tagesrenditen viele neue Stichproben gleicher "
        "Länge mit Zurücklegen, rechnet die Kennzahl auf jeder neu und nimmt die mittleren 95 % (einstellbar) "
        "der Ergebnisse als Konfidenzintervall. Breite Intervalle heißen: Die Zahl ist unsicher, egal wie viele "
        "Nachkommastellen sie hat.\n\n"
        "Untersucht werden der historische **VaR** und der **Expected Shortfall** (mittlerer Verlust jenseits "
        "des VaR), die **Exzess-Kurtosis** und der **Hill-Tail-Index α** der absoluten Renditen (kleiner = "
        "fettere Ränder; unter 2 unendliche Varianz). Die Histogramme zeigen die Bootstrap-Verteilung jeder "
        "Kennzahl mit Schätzwert und Intervallgrenzen. Für α vergleicht die Tabelle zusätzlich das "
        "asymptotische Wald-Intervall aus der Krümmung der Log-Likelihood (`numerical_hessian`).\n\n"
        "Wichtig: Der einfache Bootstrap zieht Tage unabhängig voneinander. Volatilitäts-Cluster gehen dabei "
        "verloren – bei echten Märkten sind die wahren Unsicherheiten eher größer. Der **Seed** macht das "
        "Ergebnis reproduzierbar; mehr Stichproben glätten die Histogramme, kosten aber Rechenzeit."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "n_boot", "type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100,
         "label": "Bootstrap-Stichproben", "help": "Bei langen Reihen begrenzt das Werkzeug die Zahl, damit es im "
                                                    "Browser flüssig bleibt."},
        {"name": "confidence", "type": "float", "default": 0.95, "min": 0.80, "max": 0.99, "step": 0.01,
         "label": "Konfidenzniveau des Intervalls"},
        {"name": "level", "type": "float", "default": 0.99, "min": 0.90, "max": 0.995, "step": 0.005,
         "label": "VaR-/ES-Niveau"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.utils.statistics.bootstrap_confidence_interval",
        "src.utils.statistics.compute_moments",
        "src.utils.numerical.numerical_gradient",
        "src.utils.numerical.numerical_hessian",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.expected_shortfall",
        "src.physics.levy_flight.estimate_tail_index",
    ],
    "cost": "medium",
}
MAX_WORK = 4e6  # observations x resamples, keeps the browser responsive


def _pct(x, digits=2):
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def _lvl(p):
    """Confidence level as a short German percentage: 0.99 -> '99 %', 0.975 -> '97,5 %'."""
    return f"{p * 100:.1f}".rstrip("0").rstrip(".").replace(".", ",") + "\u00a0%"


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _hill_wald(r, alpha, z):
    """Wald interval for the Hill estimate: the Pareto log-likelihood of the k largest |r| (same k as
    ``estimate_tail_index``) has curvature -k/alpha^2 at its maximum; numerical_hessian measures it."""
    x = np.sort(np.abs(r[np.isfinite(r)]))[::-1]
    n, n_pos = len(x), int(np.count_nonzero(x > 0))
    k = min(max(10, int(0.1 * n)), n - 1, n_pos - 1)
    s = float(np.sum(np.log(x[:k]) - np.log(x[k])))

    def loglik(a):
        return k * np.log(a) - a * s

    if not np.isclose(k / s, alpha, rtol=1e-9):  # not the same estimator -> no comparison
        return None
    grad = numerical_gradient(loglik, alpha)
    hess = numerical_hessian(loglik, alpha)
    if not hess < 0:
        return None
    se = 1.0 / np.sqrt(-hess)
    return alpha - z * se, alpha + z * se, se, grad, k


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    level, conf, seed = float(params["level"]), float(params["confidence"]), int(params["seed"])
    n_boot = int(params["n_boot"])
    if n * n_boot > MAX_WORK:
        n_boot = max(100, int(MAX_WORK // n // 50 * 50))
        out.note(f"Für {n} Beobachtungen rechnet das Werkzeug {n_boot} statt {int(params['n_boot'])} Stichproben, "
                 "damit es im Browser flüssig bleibt.", "info")

    # TailRiskMetrics precomputes skew/kurtosis in __init__ (milliseconds per call). Its historical VaR and
    # ES only read .returns, so one instance is reused and handed each resample.
    trm = TailRiskMetrics(r)

    def var_hist(s):
        trm.returns = s
        return trm.var_historical(level)

    def es_hist(s):
        trm.returns = s
        return trm.expected_shortfall(level)

    stats_def = [
        ("var", f"VaR {_lvl(level)} (1 Tag)", var_hist, True),
        ("es", f"Expected Shortfall {_lvl(level)}", es_hist, True),
        ("kurt", "Exzess-Kurtosis", lambda s: compute_moments(s)["kurtosis"], False),
        ("hill", "Hill-Tail-Index α", lambda s: estimate_tail_index(s, "hill"), False),
    ]
    results = {}
    for i, (key, name, fn, _) in enumerate(stats_def):
        out.progress(f"Bootstrap: {name} …", 0.05 + 0.22 * i)
        point = float(fn(r))
        reps = []

        def recorded(s, fn=fn, reps=reps):
            v = float(fn(s))
            reps.append(v)
            return v

        np.random.seed(seed)  # same seed -> the same resamples for every statistic
        lo, hi = bootstrap_confidence_interval(r, recorded, n_bootstrap=n_boot, confidence=conf)
        results[key] = (point, float(lo), float(hi), np.asarray(reps))
    trm.returns = r

    cl = _lvl(conf)
    tail_obs = int(np.sum(r < -results["var"][0]))
    out.metric(f"VaR {_lvl(level)} (1 Tag)", results["var"][0], "pct", digits=2,
               hint=f"{cl}-Intervall {_pct(results['var'][1])} – {_pct(results['var'][2])}")
    out.metric(f"Expected Shortfall {_lvl(level)}", results["es"][0], "pct", digits=2,
               hint=f"{cl}-Intervall {_pct(results['es'][1])} – {_pct(results['es'][2])}")
    out.metric("Exzess-Kurtosis", results["kurt"][0], "num", digits=1,
               hint=f"{cl}-Intervall {_num(results['kurt'][1], 1)} – {_num(results['kurt'][2], 1)}")
    out.metric("Hill-Tail-Index α", results["hill"][0], "num", digits=2,
               hint=f"{cl}-Intervall {_num(results['hill'][1])} – {_num(results['hill'][2])}",
               status="critical" if results["hill"][2] < 2 else "serious" if results["hill"][0] < 3 else None)
    out.metric("Tage jenseits des VaR", tail_obs, "int", hint=f"von {n}: darauf beruht der Expected Shortfall")

    # ---- bootstrap distributions --------------------------------------------------------------------
    sub = {"var": "Größter Tagesverlust, der an {lv} der Tage nicht überschritten wird.",
           "es": "Mittlerer Verlust an den schlimmsten Tagen jenseits des VaR.",
           "kurt": "Wenige Extremtage bestimmen die Kurtosis – deshalb ist ihr Intervall so breit.",
           "hill": "Unter 2: unendliche Varianz (Lévy-Bereich). Typisch für Aktien: 3 bis 4."}
    for key, name, _, is_pct in stats_def:
        point, lo, hi, reps = results[key]
        centers, counts = C.histogram(reps, bins=40, density=False)
        out.chart(C.xy(
            name, [C.trace("Bootstrap-Stichproben", centers, counts, type="bar", color=1)],
            x=C.axis(name, fmt="pct" if is_pct else "num", digits=2), y=C.axis("Anzahl Stichproben", fmt="int"),
            vlines=[C.vline(lo, f"{cl}: {_pct(lo) if is_pct else _num(lo)}", 2),
                    C.vline(hi, _pct(hi) if is_pct else _num(hi), 2),
                    C.vline(point, "Schätzwert", "ink", dash=False)],
            subtitle=sub[key].format(lv=_lvl(level)), height=300,
        ))

    # ---- summary table --------------------------------------------------------------------------------
    rows = []
    for key, name, _, is_pct in stats_def:
        point, lo, hi, reps = results[key]
        f = _pct if is_pct else _num
        rel = (hi - lo) / 2 / abs(point) if point else None
        rows.append([name, "Bootstrap (Perzentil)", f(point), f(lo), f(hi), rel])
    z = stats.norm.ppf(0.5 + conf / 2)
    wald = _hill_wald(r, results["hill"][0], z)
    note = (f"± relativ = halbe Intervallbreite ÷ Schätzwert. {n_boot} Stichproben mit Seed {seed}; "
            "alle Kennzahlen nutzen dieselben Stichproben.")
    if wald:
        lo_w, hi_w, se, grad, k = wald
        rows.append(["Hill-Tail-Index α", f"Wald (Krümmung der Log-Likelihood, k = {k})", _num(results["hill"][0]),
                     _num(lo_w), _num(hi_w), (hi_w - lo_w) / 2 / results["hill"][0]])
        note += (f" Wald: Standardfehler α/√k = {_num(se, 3)} aus numerical_hessian; die Steigung der "
                 f"Log-Likelihood am Schätzwert ist {grad:.1e} (≈ 0, also ein Maximum).".replace(".", ",", 1)
                 .replace("e-0", "e−0"))
    out.table(f"Konfidenzintervalle ({cl})", ["Kennzahl", "Methode", "Schätzwert", "untere Grenze", "obere Grenze",
                                              "± relativ"], rows,
              formats=["text", "text", "text", "text", "text", "pct"], note=note)
    if tail_obs < 10:
        out.note(f"Nur {tail_obs} Tag{'e' if tail_obs != 1 else ''} jenseits des VaR: Der Expected Shortfall beruht "
                 "auf so wenigen Verlusten, dass auch das Bootstrap-Intervall die Unsicherheit unterschätzt – der "
                 "Bootstrap kann keine Verluste erfinden, die schlimmer sind als der schlimmste beobachtete Tag. "
                 "Nimm eine längere Reihe oder ein niedrigeres Niveau.", "warning")
    out.note("Der Bootstrap zieht Tage unabhängig. Volatilitäts-Cluster (ruhige und hektische Phasen) gehen dabei "
             "verloren – in echten Märkten sind die Intervalle eher breiter als hier.", "warning")
