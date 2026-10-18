"""The 3D risk visualisers of src/visualization/risk_surface_3d.py: VaR surface, model surface, joint tails,
approach to a critical point and the comprehensive 3D dashboard."""

import numpy as np

from src.models.fat_tail_distributions import StudentTDistribution
from src.models.risk_metrics import TailRiskMetrics
from src.physics.levy_flight import estimate_tail_index, levy_flight_3d_coordinates
from src.physics.phase_transitions import CriticalSlowingDownDetector, compute_phase_space_coordinates
from src.physics.tsallis_statistics import compute_tsallis_coordinates
from src.visualization.risk_surface_3d import (CriticalPointVisualizer, TailRisk3DSurface, TailRiskDensityPlot,
                                               create_comprehensive_3d_dashboard)

from .. import charts as C
from ..core import ToolError

SPEC = {
    "id": "risk-surfaces",
    "title": "3D-Risikoflächen des Frameworks",
    "group": "surface",
    "order": 20,
    "summary": "Fünf Blickwinkel in 3D: VaR über Horizont und Konfidenz, eine Student-t-Crashfläche, gemeinsame "
               "Ränder, der Weg zum kritischen Punkt und das 3D-Phasenraum-Dashboard.",
    "description": (
        "Das Modul `risk_surface_3d` bündelt die 3D-Grafiken des Frameworks. Wähle oben die **Ansicht**: "
        "**VaR-Fläche** zeigt den historischen VaR für Horizont × Konfidenz – das Original skaliert mit der "
        "√t-Regel, daneben steht der tatsächlich gemessene Mehrtages-VaR. **Modellfläche** rechnet "
        "`create_risk_surface` mit Student-t-Rändern: P(mind. ein Tagesverlust über der Schwelle in 30 Tagen) über "
        "Volatilität und Tail-Index ν – anders als die README-Fläche bleibt das Risiko bei dünnen Rändern (ν groß) "
        "nicht bei null. **Gemeinsame Ränder** prüft, ob Extremtage gemeinsam auftreten (heute und gestern oder "
        "Rendite und VIX-Sprung). **Kritischer Punkt** verfolgt AC(1), Varianz und Schiefe auf dem Weg zu einem "
        "Kipppunkt. **Physik-Dashboard** zeigt Lévy-, Tsallis- und Phasenübergangsraum.\n\n"
        "In jeder Ansicht ist das interaktive Diagramm oben die Hauptdarstellung (ziehen = drehen, Mausrad = "
        "zoomen); darunter folgt die Original-Grafik aus `src/visualization`. Unter der Ansicht erscheinen nur die "
        "Parameter, die für sie gelten."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "view", "type": "select", "default": "var", "label": "Ansicht",
         "options": [{"value": "var", "label": "VaR-Fläche: Horizont × Konfidenz"},
                     {"value": "model", "label": "Modellfläche: Student-t-Crashrisiko"},
                     {"value": "joint", "label": "Gemeinsame Ränder"},
                     {"value": "critical", "label": "Annäherung an den kritischen Punkt"},
                     {"value": "physics", "label": "Physik-Dashboard (Lévy, Tsallis, Phase)"}]},
        {"name": "last_n", "type": "int", "default": 2520, "min": 250, "max": 5000, "step": 250,
         "label": "Zeitraum (letzte Handelstage)", "unit": "Tage"},
        {"name": "max_horizon", "type": "int", "default": 20, "min": 2, "max": 250, "step": 1,
         "label": "Längster Horizont", "unit": "Tage", "show_if": {"view": ["var"]}},
        {"name": "threshold", "type": "float", "default": 5.0, "min": 2.0, "max": 15.0, "step": 0.5,
         "label": "Crash-Schwelle (Tagesverlust)", "unit": "%", "show_if": {"view": ["model"]}},
        {"name": "partner", "type": "select", "default": "lag", "label": "Zweite Reihe",
         "show_if": {"view": ["joint"]},
         "options": [{"value": "lag", "label": "Rendite am Vortag"}, {"value": "vix", "label": "VIX-Veränderung"}]},
        {"name": "ews_window", "type": "int", "default": 50, "min": 20, "max": 250, "step": 5,
         "label": "Fenster für AC(1), Varianz, Schiefe", "unit": "Tage", "show_if": {"view": ["critical"]}},
        {"name": "ac_threshold", "type": "float", "default": 0.7, "min": 0.3, "max": 0.99, "step": 0.01,
         "label": "AC(1)-Schwelle „nahe kritisch“", "help": "Original: 0,7.",
         "show_if": {"view": ["critical"]}},
    ],
    "covers": [
        "src.visualization.risk_surface_3d.TailRisk3DSurface",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_risk_surface",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_phase_space_trajectory",
        "src.visualization.risk_surface_3d.TailRisk3DSurface.create_multi_coordinate_view",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot.create_tail_probability_surface",
        "src.visualization.risk_surface_3d.TailRiskDensityPlot.create_joint_tail_density",
        "src.visualization.risk_surface_3d.CriticalPointVisualizer",
        "src.visualization.risk_surface_3d.CriticalPointVisualizer.create_critical_surface",
        "src.visualization.risk_surface_3d.create_comprehensive_3d_dashboard",
        "src.models.risk_metrics.TailRiskMetrics.var_historical",
        "src.models.risk_metrics.TailRiskMetrics.tail_dependence",
        "src.models.fat_tail_distributions.StudentTDistribution.cdf",
        "src.physics.phase_transitions.CriticalSlowingDownDetector.compute_ews",
    ],
    "cost": "medium",
}

FIG_PX = 900       # target width of the original figures (they are shown at about 860 px)


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f} %".replace(".", ",").replace("-", "−")


def _date_de(d):
    s = str(d)
    return f"{s[8:10]}.{s[5:7]}.{s[:4]}" if len(s) == 10 and s[4] == "-" else s


def _x(data, k):
    if data.dates:
        return list(data.dates[-k:]), C.axis("", type="date")
    return list(range(data.n - k + 1, data.n + 1)), C.axis("Handelstag", fmt="int")


def _fig(out, fig, caption):
    dpi = int(np.clip(FIG_PX / fig.get_figwidth(), 48, 88))   # sharp 10-inch figures, light 20-inch dashboards
    fig.set_dpi(dpi)
    out.figure(fig, caption, dpi=dpi)


def run(params, data, out):
    k = int(min(params["last_n"], data.n))
    if k < 120:
        raise ToolError(f"Für die 3D-Flächen brauchst du mindestens 120 Tagesrenditen; der Datensatz hat {data.n}.")
    r = np.asarray(data.returns[-k:], dtype=float)
    {"var": _var_view, "model": _model_view, "joint": _joint_view, "critical": _critical_view,
     "physics": _physics_view}[params["view"]](params, data, out, r, k)


# --------------------------------------------------------------------------- #
# VaR surface: horizon x confidence (TailRiskDensityPlot.create_tail_probability_surface)
# --------------------------------------------------------------------------- #
def _var_view(params, data, out, r, k):
    hmax = int(min(params["max_horizon"], k // 4))
    horizons = np.arange(1, hmax + 1, dtype=float) if hmax <= 60 else np.unique(
        np.round(np.geomspace(1, hmax, 60))).astype(float)
    confs = np.linspace(0.90, 0.995, 20)
    tm = TailRiskMetrics(r)
    # identical to create_tail_probability_surface: VaR(conf) * sqrt(horizon)
    var1 = np.array([tm.var_historical(c) for c in confs])
    surface = var1[:, None] * np.sqrt(horizons)[None, :]

    # what the horizon really does: VaR of the overlapping h-day sums of log returns
    csum = np.concatenate([[0.0], np.cumsum(r)])

    def emp_var(h, conf):
        h = int(h)
        sums = csum[h:] - csum[:-h]
        return float(-np.percentile(sums, (1 - conf) * 100)) if len(sums) >= 50 else np.nan

    v99_1, v99_10 = float(tm.var_historical(0.99)), float(tm.var_historical(0.99) * np.sqrt(min(10, hmax)))
    e99_10 = emp_var(min(10, hmax), 0.99)
    out.metric("VaR 99 %, 1 Tag", v99_1, "pct", digits=2, hint="historisch, aus den gezeigten Tagen")
    out.metric(f"VaR 99 %, {min(10, hmax)} Tage (√t)", v99_10, "pct", digits=1, hint="so rechnet die Fläche")
    out.metric(f"VaR 99 %, {min(10, hmax)} Tage gemessen", e99_10, "pct", digits=1,
               hint="aus überlappenden Mehrtagesrenditen",
               status="warning" if np.isfinite(e99_10) and e99_10 > 1.15 * v99_10 else None)
    if np.isfinite(e99_10) and v99_10 > 0:
        out.metric("Gemessen ÷ √t", e99_10 / v99_10, "x",
                   hint="> 1: die √t-Regel unterschätzt das Mehrtagesrisiko (Vola-Cluster)")

    out.chart(C.surface(
        "VaR-Fläche: Horizont × Konfidenz (√t-Skalierung)", horizons, confs, surface,
        x=C.axis("Horizont (Handelstage)", fmt="int"), y=C.axis("Konfidenz", fmt="pct", digits=1),
        z=C.axis("VaR", fmt="pct", digits=0), scale="risk", height=540,
        subtitle="z = historischer 1-Tages-VaR × √Horizont – genau die Werte von create_tail_probability_surface.",
    ))
    hs = horizons[horizons <= hmax]
    traces = []
    for i, conf in enumerate((0.99, 0.95)):
        emp = [emp_var(h, conf) for h in hs]
        traces.append(C.trace(f"gemessen {int(conf * 100)} %", hs, emp, color=1 + i, width=2))
        traces.append(C.trace(f"√t-Regel {int(conf * 100)} %", hs, tm.var_historical(conf) * np.sqrt(hs),
                              color=1 + i, dash=True, width=1.5))
    out.chart(C.xy(
        "Trägt die √t-Regel? Gemessener Mehrtages-VaR vs. Skalierung",
        traces, x=C.axis("Horizont (Handelstage)", fmt="int"), y=C.axis("VaR", fmt="pct", digits=0),
        subtitle="Durchgezogen: VaR der tatsächlichen h-Tages-Renditen. Gestrichelt: 1-Tages-VaR × √h.",
    ))
    rows = []
    for h in [h for h in (1, 5, 10, 20, 60, 120, 250) if h <= hmax]:
        row = [h]
        for conf in (0.95, 0.99, 0.995):
            row += [tm.var_historical(conf) * np.sqrt(h), emp_var(h, conf)]
        rows.append(row)
    out.table("VaR nach Horizont: √t-Regel und gemessen", [
        "Horizont (Tage)", "95 % √t", "95 % gemessen", "99 % √t", "99 % gemessen", "99,5 % √t", "99,5 % gemessen"],
        rows, formats=["int", "pct", "pct", "pct", "pct", "pct", "pct"],
        note="Log-Renditen, Verlust positiv. Gemessen: Perzentil der überlappenden h-Tages-Summen – bei langen "
             "Horizonten wenige unabhängige Fälle.")
    fig = TailRiskDensityPlot(figsize=(10, 8)).create_tail_probability_surface(r, horizons, confs)
    _fig(out, fig, "Original: TailRiskDensityPlot.create_tail_probability_surface")


# --------------------------------------------------------------------------- #
# Model surface: TailRisk3DSurface.create_risk_surface with Student-t tails
# --------------------------------------------------------------------------- #
def _model_view(params, data, out, r, k):
    thr = float(params["threshold"]) / 100.0
    horizon = 30
    vols = np.linspace(0.05, 0.80, 28)          # annualised
    nus = np.linspace(2.2, 8.0, 28)              # tail index = degrees of freedom
    cache = {}

    def crash_prob(vol, nu):
        """P(at least one daily loss > thr in 30 days), Student-t with std = vol/sqrt(252)."""
        key = (round(float(vol), 10), round(float(nu), 10))
        if key not in cache:
            sd = float(vol) / np.sqrt(252.0)
            dist = StudentTDistribution(nu=float(nu), loc=0.0, scale=sd * np.sqrt((nu - 2.0) / nu))
            p_day = float(dist.cdf(np.log1p(-thr)))
            cache[key] = 1.0 - (1.0 - p_day) ** horizon
        return cache[key]

    Z = np.array([[crash_prob(v, a) for v in vols] for a in nus])
    w = min(60, k)
    vol_now = float(np.std(r[-w:]) * np.sqrt(252))
    try:
        nu_raw = float(estimate_tail_index(r[-w:]))
    except ValueError:
        nu_raw = np.nan
    nu_now = float(np.clip(nu_raw, nus[0], nus[-1])) if np.isfinite(nu_raw) else nus[-1]
    p_now = crash_prob(np.clip(vol_now, vols[0], vols[-1]), nu_now)
    gauss = 1 - (1 - float(StudentTDistribution(nu=1e6, scale=vol_now / np.sqrt(252)).cdf(np.log1p(-thr)))) ** horizon

    thr_txt = _pct(thr, 1 if thr * 1000 % 10 else 0)
    out.metric(f"Crash-Wahrscheinlichkeit ({horizon} T)", p_now, "pct", digits=1,
               hint=f"P(mind. ein Tagesverlust > {thr_txt}) bei heutiger Vola und ν = Hill-α")
    out.metric("Volatilität p. a. (60 T)", vol_now, "pct", digits=1)
    out.metric("Tail-Index ν (Hill, 60 T)", nu_raw, "num", digits=2,
               hint=f"auf der Fläche {_de(nu_now)}" if abs(nu_now - nu_raw) > 1e-9 else "kleiner = fettere Ränder")
    if gauss >= 1e-5:
        out.metric("Gleiche Vola, Normalverteilung", gauss, "pct", digits=1,
                   hint="ν → ∞: was ein Gauß-Modell sagen würde")
    else:
        exp = str(int(np.floor(np.log10(max(gauss, 1e-300))))).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        out.metric("Gleiche Vola, Normalverteilung", "< 0,001 %", "text",
                   hint=f"ν → ∞: was ein Gauß-Modell sagen würde (≈ 10{exp})")

    out.chart(C.surface(
        f"Student-t-Crashfläche: P(Tagesverlust > {thr_txt} in {horizon} Tagen)", vols, nus, Z,
        x=C.axis("Volatilität p. a.", fmt="pct", digits=0, range=(vols[-1], vols[0])),
        y=C.axis("Tail-Index ν", range=(nus[-1], nus[0])),
        z=C.axis("Wahrscheinlichkeit", fmt="pct", digits=0, range=(0, 1.05)),
        points=[C.path3d("Heute", [np.clip(vol_now, vols[0], vols[-1])], [nu_now], [p_now + 0.012], color="ink",
                         mode="markers", size=8, text=[f"{_pct(p_now)}"])],
        scale="risk", height=560,
        camera={"eye": {"x": 1.2, "y": 1.7, "z": 1.0}},   # same view as the README surface (same axes)
        subtitle="Gleiche Volatilität (Standardabweichung) auf der ganzen Fläche – ν verteilt sie nur anders: "
                 "kleines ν = seltener, aber heftiger.",
    ))
    rows = [[_pct(v, 0)] + [crash_prob(v, nu) for nu in (2.5, 3.0, 4.0, 6.0, 8.0)] for v in (0.10, 0.20, 0.30, 0.50)]
    out.table(f"Crash-Wahrscheinlichkeit ({horizon} Tage) nach Volatilität und ν",
              ["Vola p. a.", "ν = 2,5", "ν = 3", "ν = 4", "ν = 6", "ν = 8"], rows,
              formats=["text", "pct", "pct", "pct", "pct", "pct"],
              note="Bei niedriger Vola erhöhen fette Ränder (kleines ν) das Crashrisiko, bei sehr hoher Vola kann "
                   "ein großes ν sogar mehr mittelgroße Verluste bedeuten – die Varianz ist ja fest.")
    out.note("Unterschied zur README-Fläche („3D-Tail-Risk-Fläche“): Dort zählt nur der Potenzgesetz-Rand einer "
             "Lévy-Verteilung (α ≤ 2), der bei α = 2 verschwindet. Hier ist die Varianz fest und ν > 2 – auch "
             "Gauß-ähnliche Ränder erzeugen bei hoher Vola Crash-Tage.")
    fig = TailRisk3DSurface(figsize=(10, 8)).create_risk_surface(
        vols, nus, crash_prob, title=f"P(daily loss > {thr * 100:g}% within 30 days), Student-t")
    _fig(out, fig, "Original: TailRisk3DSurface.create_risk_surface (Student-t-Risikofunktion)")


# --------------------------------------------------------------------------- #
# Joint tails: TailRiskDensityPlot.create_joint_tail_density + TailRiskMetrics.tail_dependence
# --------------------------------------------------------------------------- #
def _joint_view(params, data, out, r, k):
    if params["partner"] == "vix":
        if data.vix is None:
            raise ToolError("Der aktive Datensatz enthält keinen VIX. Wähle „Rendite am Vortag“ oder einen "
                            "Datensatz mit VIX (z. B. S&P 500).")
        vix = np.asarray(data.vix[-k:], dtype=float)
        a, b = r[1:], np.diff(np.log(vix))
        names = ("Rendite", "VIX-Veränderung")
        lab_a, lab_b = "Tagesrendite (heute)", "VIX-Veränderung (log, heute)"
    else:
        a, b = r[1:], r[:-1]
        names = ("Heute", "Vortag")
        lab_a, lab_b = "Tagesrendite heute", "Tagesrendite am Vortag"
    vix_mode = params["partner"] == "vix"
    tm = TailRiskMetrics(a)
    # the VIX rises when stocks fall: its "crash tail" is the upper one, so the dependence is measured on -ΔVIX
    dep = tm.tail_dependence(-b if vix_mode else b, quantile=0.95)
    corr = -dep["standard_correlation"] if vix_mode else dep["standard_correlation"]
    qa, qb = np.percentile(a, 5), np.percentile(b, 5)
    qb_hi = np.percentile(b, 95)
    joint_low = (a < qa) & (b < qb)
    joint_cross = (a < qa) & (b > qb_hi)       # crash today with a VIX spike / a rally the day before
    out.metric("Korrelation", corr, "num", digits=2)
    if vix_mode:
        out.metric("Crash & VIX-Sprung (5 %)", dep["lower_tail_dependence"], "pct", digits=1,
                   hint="P(Rendite unter 5-%-Quantil und VIX-Veränderung über 95-%-Quantil) ÷ 5 % · unabhängig: 5 %")
        out.metric("Rallye & VIX-Rückgang (5 %)", dep["upper_tail_dependence"], "pct", digits=1,
                   hint="P(Rendite über 95-%-Quantil und VIX-Veränderung unter 5-%-Quantil) ÷ 5 %")
        out.metric("Asymmetrie (Crash − Rallye)", dep["tail_asymmetry"], "pct", digits=1,
                   hint="> 0: Crashs kommen öfter mit VIX-Sprung als Rallyes mit VIX-Rückgang")
    else:
        out.metric("Gemeinsamer linker Rand (5 %)", dep["lower_tail_dependence"], "pct", digits=1,
                   hint="P(beide unter ihrem 5-%-Quantil) ÷ 5 % · unabhängig: 5 %")
        out.metric("Gemeinsamer rechter Rand (5 %)", dep["upper_tail_dependence"], "pct", digits=1,
                   hint="P(beide über ihrem 95-%-Quantil) ÷ 5 %")
        out.metric("Asymmetrie (links − rechts)", dep["tail_asymmetry"], "pct", digits=1,
                   hint="> 0: Abstürze treten häufiger gemeinsam auf als Rallyes")

    dates = data.dates[-k:][1:] if data.dates else None
    normal = ~(joint_low | joint_cross)
    text = (lambda m: [dates[i] for i in np.flatnonzero(m)]) if dates else (lambda m: None)
    traces = [C.trace("alle Tage", a[normal], b[normal], type="scatter", color=1, size=4, decimate_to=None,
                      text=text(normal))]
    if joint_low.any():
        traces.append(C.trace("beide im linken 5-%-Rand", a[joint_low], b[joint_low], type="scatter",
                              color="critical", size=7, decimate_to=None, text=text(joint_low)))
    if joint_cross.any():
        traces.append(C.trace(f"{names[0]} links, {names[1]} rechts im Rand", a[joint_cross], b[joint_cross],
                              type="scatter", color=2, size=6, decimate_to=None, text=text(joint_cross)))
    out.chart(C.xy(
        f"Gemeinsame Ränder: {names[0]} gegen {names[1]}", traces,
        x=C.axis(lab_a, fmt="pct", digits=1), y=C.axis(lab_b, fmt="pct", digits=1),
        vlines=[C.vline(float(qa), "5-%-Quantil")], hlines=[C.hline(float(qb), "5-%-Quantil"),
                                                            C.hline(float(qb_hi), "95-%-Quantil")],
        height=440,
        subtitle=(f"Unabhängige Reihen hätten im linken oberen Eck (Crash und VIX-Sprung) nur 0,25 % der Tage – "
                  f"hier sind es {_pct(float(joint_cross.mean()), 2)}." if vix_mode else
                  f"Unabhängige Reihen hätten im linken unteren Eck (zwei Verlusttage in Folge) nur 0,25 % der Tage – "
                  f"hier sind es {_pct(float(joint_low.mean()), 2)}."),
    ))
    rows = []
    for q in (0.10, 0.05, 0.01):
        both = float(np.mean((a < np.percentile(a, 100 * q)) & (b < np.percentile(b, 100 * q))))
        cross = float(np.mean((a < np.percentile(a, 100 * q)) & (b > np.percentile(b, 100 * (1 - q)))))
        key = cross if vix_mode else both
        rows.append([_pct(q, 0), both, cross, q * q, key / (q * q) if q > 0 else None])
    out.table("Gemeinsame Extremtage gegen Unabhängigkeit", [
        "Rand", "beide links", f"{names[0]} links, {names[1]} rechts", "unabhängig erwartet",
        "Faktor (Crash & VIX-Sprung)" if vix_mode else "Faktor (beide links)"],
        rows, formats=["text", "pct", "pct", "pct", "x"],
        note=("Faktor = beobachtet ÷ unabhängig erwartet. Crash und VIX-Sprung am selben Tag sind die Regel, nicht die "
              "Ausnahme – deshalb taugt der VIX als Absicherungssignal." if vix_mode else
              "Faktor > 1: Extremtage kommen gemeinsam (Volatilitäts-Cluster); die Spalte links/rechts zählt "
              "Crash-Tage direkt nach einem starken Plus-Tag."))
    fig = TailRiskDensityPlot(figsize=(10, 8)).create_joint_tail_density(a, b, asset_names=names,
                                                                         title="Joint Tail Density")
    _fig(out, fig, "Original: TailRiskDensityPlot.create_joint_tail_density")


# --------------------------------------------------------------------------- #
# Critical point: CriticalPointVisualizer.create_critical_surface on the framework's EWS
# --------------------------------------------------------------------------- #
def _critical_view(params, data, out, r, k):
    w = int(params["ews_window"])
    if k < w + 60:
        raise ToolError(f"Für ein Fenster von {w} Tagen brauchst du mindestens {w + 60} Tage im Zeitraum.")
    ews = CriticalSlowingDownDetector(window=w).compute_ews(r)
    ac = np.asarray(ews["autocorrelation"], dtype=float).copy()
    var = np.asarray(ews["variance"], dtype=float).copy()
    skew = np.asarray(ews["skewness"], dtype=float).copy()
    ac[:w] = var[:w] = skew[:w] = np.nan                 # no full window yet
    thr = float(params["ac_threshold"])
    var_cut = float(np.nanpercentile(var, 80))
    crit = (ac > thr) & (var > var_cut)
    x, xa = _x(data, k)
    idx = np.flatnonzero(np.isfinite(ac))
    var_pct_now = float(np.mean(var[idx] <= var[-1]))
    ac_pct_now = float(np.mean(ac[idx] <= ac[-1]))
    above_thr = float(np.mean(ac[idx] > thr))
    crit_share = float(np.mean(crit[idx]))
    # AC(1) of a detrended price is above 0.7 almost always - only a high value *relative to its history* is a signal
    out.metric("AC(1) heute", ac[-1], "num", digits=2,
               hint=f"des trendbereinigten Log-Kurses · Perzentil {_de(100 * ac_pct_now, 0)} %",
               status="warning" if ac[-1] > thr and ac_pct_now >= 0.8 else None)
    out.metric("Varianz-Perzentil heute", var_pct_now, "pct", digits=0, hint="gegenüber dem gezeigten Zeitraum",
               status="warning" if var_pct_now > 0.8 else None)
    out.metric("Schiefe heute", skew[-1], "num", digits=2, hint=f"der Tagesrenditen über {w} Tage")
    out.metric("Tage „nahe kritisch“", int(crit.sum()), "int",
               hint=f"AC(1) > {_de(thr)} und Varianz > 80. Perzentil · {_pct(crit_share, 1)} der Tage")
    if crit.any():
        last = x[int(np.flatnonzero(crit)[-1])]
        out.metric("Zuletzt nahe kritisch", _date_de(last) if isinstance(last, str) else f"Tag {last}", "text")

    step = max(1, len(idx) // 900)
    keep = idx[::step]
    recent = idx[-min(60, len(idx)):]
    label = (lambda ii: [x[i] for i in ii]) if data.dates else (lambda ii: [f"Tag {x[i]}" for i in ii])
    traces = [C.path3d("Verlauf", ac[keep], var[keep], skew[keep], color=1, mode="markers", size=2, text=label(keep)),
              C.path3d("letzte 60 Tage", ac[recent], var[recent], skew[recent], color="ink", mode="lines+markers",
                       size=2, text=label(recent))]
    if crit.any():
        ci = np.flatnonzero(crit)
        traces.append(C.path3d("nahe kritisch", ac[ci], var[ci], skew[ci], color="critical", mode="markers", size=4,
                               text=label(ci)))
    out.chart(C.scatter3d(
        "Annäherung an den kritischen Punkt: AC(1) × Varianz × Schiefe", traces,
        x=C.axis("AC(1)"), y=C.axis("Varianz (trendbereinigt)"), z=C.axis("Schiefe"), height=560,
        subtitle="Kritisches Verlangsamen: AC(1) und Varianz steigen gemeinsam, die Schiefe kippt ins Negative. "
                 "Rot: Tage mit hoher AC(1) und Varianz im obersten Fünftel.",
    ))
    tr = [C.trace("AC(1)", x, ac, color=1, width=1.2)]
    if crit.any():
        ci = np.flatnonzero(crit)
        tr.append(C.trace("nahe kritisch", [x[i] for i in ci], ac[ci], type="scatter", color="critical", size=5,
                          decimate_to=None))
    out.chart(C.xy(f"AC(1) im Zeitverlauf ({w}-Tage-Fenster)", tr, x=xa, y=C.axis("AC(1)"),
                   hlines=[C.hline(thr, f"Schwelle {_de(thr)}", color="warning")],
                   subtitle="Rot: Tage, an denen zusätzlich die Varianz über ihrem 80. Perzentil liegt."))
    mech = (f" Hier liegt sie an {_pct(above_thr, 0)} der Tage über {_de(thr)} – „nahe kritisch“ heißt damit "
            "praktisch nur „Varianz im obersten Fünftel“, und das trifft per Definition 20 % der Tage."
            if above_thr > 0.9 else "")
    out.note("Die AC(1) wird auf dem trendbereinigten Log-Kurs gemessen (wie im Werkzeug „Kritisches "
             "Verlangsamen“); schon eine reine Irrfahrt hat dort hohe Werte." + mech + " Aussagekräftig ist der "
             "gemeinsame Anstieg von AC(1) und Varianz, nicht das Niveau allein.")
    fig = CriticalPointVisualizer(figsize=(10, 8)).create_critical_surface(ac, var, skew, np.arange(k, dtype=float),
                                                                           ac_threshold=thr)
    _fig(out, fig, "Original: CriticalPointVisualizer.create_critical_surface")


# --------------------------------------------------------------------------- #
# Physics dashboard: create_comprehensive_3d_dashboard and TailRisk3DSurface views
# --------------------------------------------------------------------------- #
def _physics_view(params, data, out, r, k):
    k = min(k, 2520)
    r = r[-k:]
    out.progress("Lévy-, Tsallis- und Phasenkoordinaten …", 0.1)
    levy = levy_flight_3d_coordinates(r)
    tsallis = compute_tsallis_coordinates(r)
    phase = compute_phase_space_coordinates(r)
    warm = min(100, k // 3)                       # the coordinates are 0 until their windows are full
    risk = (levy[2] + 0.5 * tsallis[0] + phase[0]) / 2.5   # composite of create_comprehensive_3d_dashboard
    rv = risk[warm:]
    lo, mid = np.percentile(rv, [50, 90])
    x, xa = _x(data, k)
    xs = x[warm:]
    out.metric("Composite heute", rv[-1], "num", digits=2, hint="(λ + 0,5·q + χ) / 2,5")
    out.metric("Perzentil heute", float(np.mean(rv <= rv[-1])), "pct", digits=0, hint="gegenüber dem Zeitraum")
    out.metric("Lévy-Vola heute", levy[0][-1], "num", digits=2, hint="σ ÷ Ø σ · 1 = durchschnittliche Volatilität")
    out.metric("Tsallis-q heute", tsallis[0][-1], "num", digits=2, hint="1 = Gauß, größer = fettere Ränder")

    traces = []
    for name, mask, col in (("Composite unter Median", rv <= lo, "good"),
                            ("Median bis 90. Perzentil", (rv > lo) & (rv <= mid), "warning"),
                            ("oberste 10 %", rv > mid, "critical")):
        ii = np.flatnonzero(mask)
        ii = ii[::max(1, len(ii) // 700)]
        traces.append(C.path3d(name, levy[0][warm:][ii], tsallis[1][warm:][ii], rv[ii], color=col,
                               mode="markers", size=2.5, text=[str(xs[i]) for i in ii]))
    traces.append(C.path3d("heute", [levy[0][-1]], [tsallis[1][-1]], [rv[-1]], color="ink", mode="markers", size=7))
    out.chart(C.scatter3d(
        "Composite-Risiko im Raum aus Volatilität und Entropie", traces,
        x=C.axis("Volatilität σ ÷ Ø σ (Lévy)"), y=C.axis("Tsallis-Entropie S_q"), z=C.axis("Composite-Risiko"),
        height=540,
        subtitle="Panel 4 des 3D-Dashboards, drehbar. Die Farbe ist das Composite-Perzentil (Median, 90 %).",
    ))
    out.chart(C.xy("Composite-Risiko im Zeitverlauf", [C.trace("Composite", xs, rv, color=1, width=1)],
                   x=xa, y=C.axis("Composite"), hlines=[C.hline(float(mid), "90. Perzentil", color="warning")],
                   subtitle="Der Composite folgt vor allem der heutigen Tagesbewegung |r|/σ (λ)."))
    out.note("Die interaktiven Einzelräume (Lévy, Tsallis, Fokker-Planck) mit Datums-Hover findest du im Werkzeug "
             "„Physikalische Phasenräume in 3D“. Hier stehen die Original-Grafiken des Moduls `risk_surface_3d`.")
    out.progress("Original-Grafiken werden gezeichnet …", 0.5)
    fig = create_comprehensive_3d_dashboard(r)    # no save_path: nothing is written
    _fig(out, fig, "Original: create_comprehensive_3d_dashboard")
    vis = TailRisk3DSurface(figsize=(10, 8))
    trim = slice(warm, None)
    fig = vis.create_multi_coordinate_view(tuple(c[trim] for c in levy), tuple(c[trim] for c in tsallis),
                                           tuple(c[trim] for c in phase))
    _fig(out, fig, "Original: TailRisk3DSurface.create_multi_coordinate_view (ohne Aufwärmphase)")
    m = min(250, k - warm)
    fig = vis.create_phase_space_trajectory(levy[0][-m:], levy[1][-m:], levy[2][-m:],
                                            time=np.arange(m, dtype=float),
                                            title=f"Lévy phase-space trajectory (last {m} days)",
                                            axis_labels=("σ / mean σ", "2 / α (tail heaviness)", "|r| / σ (jump)"))
    _fig(out, fig, f"Original: TailRisk3DSurface.create_phase_space_trajectory (letzte {m} Tage im Lévy-Raum)")
