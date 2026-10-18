"""Ornstein-Uhlenbeck process: mean reversion, path fan with the analytic band, and an AR(1) fit (VIX-like)."""

import numpy as np
from scipy import stats

from src.physics.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

from .. import charts as C

SPEC = {
    "id": "ou-process",
    "title": "Ornstein-Uhlenbeck: Rückkehr zum Mittel",
    "group": "physics",
    "order": 40,
    "summary": "Wie schnell kehrt eine Größe wie der VIX zu ihrem Mittel zurück? Pfad-Fächer mit exaktem "
               "Erwartungsband, Halbwertszeit und Parameter-Schätzung aus Daten.",
    "description": (
        "Der Ornstein-Uhlenbeck-Prozess **dx = θ(μ − x)dt + σ dW** stammt aus der Physik (Geschwindigkeit eines "
        "Teilchens mit Reibung) und beschreibt in der Finanzwelt alles, was zu einem Mittel zurückkehrt: "
        "Volatilität, Zinsen, Spreads. **θ** ist die Rückstellgeschwindigkeit pro Jahr, **μ** das langfristige "
        "Mittel, **σ** die Stärke der Zufallsstöße. Die **Halbwertszeit** ln 2/θ sagt, nach welcher Zeit die "
        "Hälfte einer Abweichung abgebaut ist.\n\n"
        "So liest du die Diagramme: Der Fächer zeigt simulierte Pfade (exakte Diskretisierung) ab dem Startwert "
        "x₀. Die durchgezogene Kurve ist der exakte Erwartungswert μ + (x₀ − μ)e^(−θt), die gestrichelten Linien "
        "das ±2σ(t)-Band. Die Streuung wächst nicht unbegrenzt wie bei einer Irrfahrt, sondern sättigt bei der "
        "stationären Standardabweichung σ/√(2θ). Die Voreinstellungen ähneln dem echten VIX (Mittel um 19, "
        "Halbwertszeit rund vier Wochen), Start im Stress bei 35.\n\n"
        "Der **Fit** schätzt θ, μ und σ per AR(1)-Regression x_{t+1} = a + b·x_t: aus dem echten VIX deines "
        "Datensatzes (Voreinstellung) oder aus einem simulierten Pfad mit deinen Parametern – so siehst du, wie "
        "genau die Schätzung bei gegebener Datenlänge ist (θ ist notorisch schwer zu treffen). Hat der Datensatz "
        "keinen VIX, nimmt der Fit automatisch den simulierten Pfad."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "theta", "type": "float", "default": 9.0, "min": 0.1, "max": 100.0, "step": 0.1,
         "label": "Rückstellgeschwindigkeit θ", "unit": "pro Jahr"},
        {"name": "mu", "type": "float", "default": 20.0, "min": -50.0, "max": 100.0, "step": 0.5,
         "label": "Langfristiges Mittel μ"},
        {"name": "sigma", "type": "float", "default": 30.0, "min": 0.1, "max": 200.0, "step": 0.5,
         "label": "Volatilität σ", "unit": "pro √Jahr"},
        {"name": "x0", "type": "float", "default": 35.0, "min": -50.0, "max": 150.0, "step": 0.5,
         "label": "Startwert x₀"},
        {"name": "horizon", "type": "float", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1,
         "label": "Horizont", "unit": "Jahre"},
        {"name": "n_steps", "type": "int", "default": 126, "min": 10, "max": 1260, "step": 1,
         "label": "Zeitschritte", "help": "126 Schritte auf ½ Jahr = ein Schritt je Handelstag."},
        {"name": "n_paths", "type": "int", "default": 500, "min": 10, "max": 2000, "step": 10,
         "label": "Simulierte Pfade"},
        {"name": "fit_source", "type": "select", "default": "vix", "label": "Parameter-Fit",
         "options": [{"value": "vix", "label": "VIX des Datensatzes"},
                     {"value": "sim", "label": "Simulierter Pfad mit deinen Parametern"},
                     {"value": "none", "label": "kein Fit"}]},
        {"name": "fit_years", "type": "float", "default": 4.0, "min": 0.5, "max": 40.0, "step": 0.5,
         "label": "Länge des simulierten Fit-Pfads", "unit": "Jahre",
         "help": "Nur für den Fit auf den simulierten Pfad. Ein Schritt je Handelstag (dt = 1/252)."},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.stationary_variance",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.stationary_std",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.half_life",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.expected_value",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.variance_at_time",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.simulate",
        "src.physics.ornstein_uhlenbeck.OrnsteinUhlenbeckProcess.fit",
    ],
    "cost": "fast",
}

N_SHOW = 15         # paths drawn individually
FORECAST_DAYS = 63  # forecast cone after the fitted series (one quarter)
DT = 1 / 252


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _gapped(times, paths):
    """Several paths as one trace, separated by gaps (NaN), so the legend stays short."""
    xs, ys = [], []
    for p in paths:
        xs.extend(times.tolist() + [None])
        ys.extend(p.tolist() + [np.nan])
    return xs, np.asarray(ys, dtype=float)


def run(params, data, out):
    ou = OrnsteinUhlenbeckProcess(theta=params["theta"], mu=params["mu"], sigma=params["sigma"])
    x0, T, steps = params["x0"], params["horizon"], params["n_steps"]
    paths = ou.simulate(x0, T, n_steps=steps, n_paths=params["n_paths"], seed=params["seed"])
    t = np.linspace(0, T, steps + 1)
    mean_t, sd_t = ou.expected_value(x0, t), np.sqrt(ou.variance_at_time(t))
    mc_mean, mc_sd = paths.mean(axis=0), paths.std(axis=0)
    half_days = ou.half_life() * 252

    out.metric("Halbwertszeit", half_days, "num", digits=0 if half_days >= 10 else 1, unit="Handelstage",
               hint=f"ln 2/θ = {_de(ou.half_life(), 3)} Jahre")
    out.metric("Stationäre Std σ/√(2θ)", ou.stationary_std(), "num",
               hint=f"stationäre Varianz {_de(ou.stationary_variance(), 1)}")
    out.metric("E[X] am Horizont (exakt)", mean_t[-1], "num", hint=f"Simulation: {_de(mc_mean[-1])}")
    out.metric("Std am Horizont (exakt)", sd_t[-1], "num", hint=f"Simulation: {_de(mc_sd[-1])}")
    out.metric("Rest der Startabweichung am Horizont", np.exp(-ou.theta * T), "pct", digits=1,
               hint="e^(−θT): so viel von x₀ − μ ist im Erwartungswert noch übrig")

    gx, gy = _gapped(t, paths[:N_SHOW])
    bx, by = _gapped(t, [mean_t + 2 * sd_t, mean_t - 2 * sd_t])
    fan = [C.trace(f"{min(N_SHOW, len(paths))} simulierte Pfade", gx, gy, color=1, width=1, decimate_to=None),
           C.trace("Erwartungswert (exakt)", t, mean_t, color=2, width=3),
           C.trace("±2σ(t) (exakt)", bx, by, color=2, dash=True, decimate_to=None),
           C.trace(f"Mittel aller {params['n_paths']} Pfade", t, mc_mean, color=3, dash=True)]
    if ou.half_life() < T:
        lo, hi = float(np.min(paths[:N_SHOW])), float(np.max(paths[:N_SHOW]))
        fan.append(C.trace("nach 1 Halbwertszeit", [ou.half_life()] * 2, [lo, hi], color="muted", dash=True,
                           width=1.5, decimate_to=None))
    out.chart(C.xy(
        "Pfad-Fächer mit exaktem Erwartungsband", fan,
        x=C.axis("Zeit (Jahre)"), y=C.axis("x"), hlines=[C.hline(ou.mu, "μ")], height=380,
        subtitle="Die Abweichung vom Mittel schrumpft exponentiell; ±2σ(t) enthält rund 95 % der Pfade.",
    ))
    out.chart(C.xy(
        "Streuung: wächst und sättigt",
        [C.trace("σ(t) exakt", t, sd_t, color=2, width=2.5),
         C.trace("σ(t) der Simulation", t, mc_sd, color=1, dash=True)],
        x=C.axis("Zeit (Jahre)"), y=C.axis("Standardabweichung"),
        hlines=[C.hline(ou.stationary_std(), "stationär σ/√(2θ)")], height=300,
        subtitle="Anders als bei einer Irrfahrt (√t) wird die Unsicherheit nicht beliebig groß.",
    ))
    end = paths[:, -1]
    centers, heights = C.histogram(end, bins=40)
    grid = np.linspace(centers[0], centers[-1], 300)
    out.chart(C.xy(
        "Verteilung am Horizont",
        [C.trace("Simulation", centers, heights, type="bar", color=1),
         C.trace("Exakt N(E[X_T], Var[X_T])", grid, stats.norm.pdf(grid, mean_t[-1], sd_t[-1]), color=2),
         C.trace("Stationär N(μ, σ²/2θ)", grid, stats.norm.pdf(grid, ou.mu, ou.stationary_std()),
                 color="muted", dash=True)],
        x=C.axis(f"x nach {_de(T, 1)} Jahren"), y=C.axis("Dichte"), height=300,
        subtitle="Die Verteilung ist normal. Nach einigen Halbwertszeiten ist sie von der stationären nicht "
                 "mehr zu unterscheiden.",
    ))
    if ou.mu > 0 and ou.mu - 2 * ou.stationary_std() < 0:
        out.note("Achtung bei VIX-Anwendungen: Der OU-Prozess ist normalverteilt und kann negativ werden – "
                 "die echte Volatilität nicht.", "warning")

    if params["fit_source"] != "none":
        _fit(params, data, out, ou)


def _fit(params, data, out, true_ou):
    fit_source = params["fit_source"]
    if fit_source == "vix" and (data is None or data.vix is None):
        out.note("Dein Datensatz enthält keinen VIX – der Fit nutzt deshalb einen simulierten Pfad mit deinen "
                 "Parametern. Für den echten VIX wähle als Datensatz „S&P 500“.")
        fit_source = "sim"
    if fit_source == "vix":
        series = np.asarray(data.vix, dtype=float)
        label, source = f"VIX ({data.name})", "dem VIX des Datensatzes"
    else:
        n_fit = int(round(params["fit_years"] * 252))
        series = true_ou.simulate(true_ou.mu, n_fit * DT, n_steps=n_fit, n_paths=1, seed=params["seed"] + 1)[0]
        label, source = "simulierter Pfad", "dem simulierten Pfad"

    try:
        fit = OrnsteinUhlenbeckProcess.fit(series, dt=DT)
    except ValueError as exc:
        msg = str(exc)
        reason = ("Die AR(1)-Steigung ist ≥ 1: Die Reihe zeigt keine Rückkehr zum Mittel (Irrfahrt oder Trend)."
                  if ">= 1" in msg else
                  "Die AR(1)-Steigung ist ≤ 0: Die Reihe springt eher hin und her, als dass sie zurückkehrt – "
                  "θ wäre unendlich." if "<= 0" in msg else msg)
        out.note(f"Kein OU-Fit auf {source} möglich. {reason}", "warning")
        return

    b = np.exp(-fit.theta * DT)
    rows = [["θ (pro Jahr)", fit.theta], ["μ", fit.mu], ["σ (pro √Jahr)", fit.sigma],
            ["Halbwertszeit (Handelstage)", fit.half_life() * 252], ["stationäre Std", fit.stationary_std()],
            ["AR(1)-Steigung b = e^(−θ·dt)", b]]
    truth = [true_ou.theta, true_ou.mu, true_ou.sigma, true_ou.half_life() * 252, true_ou.stationary_std(),
             np.exp(-true_ou.theta * DT)]
    if fit_source == "sim":
        rows = [r + [tv, r[1] / tv - 1 if tv else None] for r, tv in zip(rows, truth)]
        out.table(f"Parameter-Schätzung aus {len(series)} Tagen ({_de(len(series) / 252, 1)} Jahre)",
                  ["Parameter", "geschätzt", "wahr", "Fehler"], rows, formats=["text", "num", "num", "pct"],
                  note="θ braucht viele Halbwertszeiten Daten: Bei kurzen Reihen wird es systematisch überschätzt.")
    else:
        out.table(f"OU-Fit auf {label} ({len(series)} Tage)", ["Parameter", "geschätzt", "deine Simulation"],
                  [r + [tv] for r, tv in zip(rows, truth)], formats=["text", "num", "num"],
                  note="Rechts zum Vergleich die Parameter, mit denen du oben simulierst.")
    out.metric("Fit: Halbwertszeit", fit.half_life() * 252, "num", digits=0, unit="Handelstage",
               hint=f"geschätzt aus {source}")
    out.metric("Fit: μ", fit.mu, "num", hint=f"θ = {_de(fit.theta, 1)} pro Jahr, σ = {_de(fit.sigma, 1)}")

    hist = series[-252:]
    h_days = np.arange(-len(hist) + 1, 1)
    f_days = np.arange(0, FORECAST_DAYS + 1)
    tf = f_days * DT
    f_mean = fit.expected_value(series[-1], tf)
    f_sd = np.sqrt(fit.variance_at_time(tf))
    cx, cy = _gapped(f_days, [f_mean + 2 * f_sd, f_mean - 2 * f_sd])
    out.chart(C.xy(
        f"Fit und Prognosekegel – {label}",
        [C.trace(f"Beobachtet (letzte {len(hist)} Tage)", h_days, hist, color=1, width=1.5),
         C.trace("Prognose E[X]", f_days, f_mean, color=2, width=2.5),
         C.trace("±2σ", cx, cy, color=2, dash=True, decimate_to=None)],
        x=C.axis("Handelstage relativ zum letzten Wert (0 = heute)", fmt="int"), y=C.axis("x"),
        hlines=[C.hline(fit.mu, "geschätztes μ")], height=340,
        subtitle=f"Mit dem geschätzten Modell: nach {_de(fit.half_life() * 252, 0)} Handelstagen ist die "
                 "Hälfte der heutigen Abweichung abgebaut.",
    ))
