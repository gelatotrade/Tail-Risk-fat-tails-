"""VIX-Sentiment: regime, percentile, spikes and mean reversion over time, plus the 3D sentiment phase space."""

import numpy as np

from src.analysis.sentiment import VIXAnalyzer, compute_sentiment_coordinates
from src.utils.data_loader import generate_vix_from_returns

from .. import charts as C
from ..data import index_axis

SPEC = {
    "id": "sentiment",
    "title": "VIX-Sentiment: Angst, Spikes, Umkehr",
    "group": "warning",
    "order": 30,
    "summary": "Wie viel Angst steckt im VIX? Regime, Perzentil, Spikes und Mean-Reversion-Signal über die Zeit – "
               "und was danach mit den Kursen geschah.",
    "description": (
        "Der VIX misst die für 30 Tage erwartete Schwankung des S&P 500 aus Optionspreisen – das bekannteste "
        "Angstbarometer. Der `VIXAnalyzer` des Frameworks ordnet ihn in Regime ein (unter 15 sorglos, bis 25 "
        "normal, bis 35 erhöht, bis 50 hohe Angst, darüber Panik), misst sein **Perzentil** im letzten Jahr, "
        "erkennt **Spikes** (Tagesanstieg größer als die Schwelle × übliche Tagesschwankung der letzten 19 "
        "Tage) und gibt ein **Mean-Reversion-Signal**: Ein z-Wert über 2 deutet auf eine nahe Umkehr der "
        "Angst, unter −1,5 auf Sorglosigkeit.\n\n"
        "Jeder Tageswert nutzt nur den VIX bis zu diesem Tag. Die Tabellen prüfen, was in den 20 "
        "Handelstagen nach einem Regime oder Spike mit den Kursen geschah. Der 3D-Phasenraum "
        "(`compute_sentiment_coordinates`) zeigt VIX-Niveau, VIX-Momentum und die Korrelation zwischen "
        "Renditen und VIX-Änderungen, jeweils auf 0–1 normiert.\n\n"
        "Enthält der Datensatz keinen VIX, leitet das Framework einen aus den Renditen ab "
        "(`generate_vix_from_returns`) – dann sind die Werte illustrativ, und es gibt keine Einschätzung."
    ),
    "needs": ["returns"],
    "params": [
        {"name": "threshold", "type": "float", "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1,
         "label": "Spike-Schwelle (z)", "help": "Ein Spike ist ein Tagesanstieg um mehr als z übliche Tagesschwankungen."},
        {"name": "pct_window", "type": "int", "default": 252, "min": 20, "max": 1260, "step": 21,
         "label": "Perzentil-Fenster", "unit": "Tage", "help": "252 Handelstage ≈ ein Jahr."},
        {"name": "coord_window", "type": "int", "default": 20, "min": 5, "max": 120, "step": 5,
         "label": "Fenster des Phasenraums", "unit": "Tage", "help": "Für VIX-Momentum und Rendite-VIX-Korrelation."},
        {"name": "lookback_3d", "type": "int", "default": 1000, "min": 100, "max": 20000, "step": 100,
         "label": "3D: letzte Handelstage", "unit": "Tage"},
    ],
    "covers": [
        "src.analysis.sentiment.VIXAnalyzer",
        "src.analysis.sentiment.VIXAnalyzer.add_data",
        "src.analysis.sentiment.VIXAnalyzer.current_regime",
        "src.analysis.sentiment.VIXAnalyzer.vix_percentile",
        "src.analysis.sentiment.VIXAnalyzer.vix_spike_detection",
        "src.analysis.sentiment.VIXAnalyzer.mean_reversion_signal",
        "src.analysis.sentiment.compute_sentiment_coordinates",
        "src.utils.data_loader.generate_vix_from_returns",
    ],
    "cost": "medium",
}

REGIME = {  # code -> (German label, status colour)
    "COMPLACENT": ("Sorglos", "good"),
    "NORMAL": ("Normal", "good"),
    "ELEVATED": ("Erhöht", "warning"),
    "HIGH_FEAR": ("Hohe Angst", "serious"),
    "PANIC": ("Panik", "critical"),
}
SIGNAL = {"HIGH_FEAR_REVERSAL_LIKELY": "Angst-Umkehr wahrscheinlich", "COMPLACENCY_WARNING": "Warnung: Sorglosigkeit",
          "NEUTRAL": "neutral"}
H = 20  # forward window of the outcome tables (trading days)


def run(params, data, out):
    r = np.asarray(data.returns, dtype=float)
    n = data.n
    x, xa = index_axis(data)
    label = (lambda i: x[i]) if data.dates else (lambda i: f"Tag {i}")
    real = data.vix is not None
    vix = np.asarray(data.vix, dtype=float) if real else np.asarray(generate_vix_from_returns(r), dtype=float)

    an = VIXAnalyzer()
    an.add_data(vix)
    th = an.thresholds
    regime = an.current_regime()
    pct = an.vix_percentile(int(params["pct_window"]))
    spike = an.vix_spike_detection(float(params["threshold"]))
    mr = an.mean_reversion_signal()

    # the same methods day by day, each on the VIX known at that day
    out.progress("Werte jeden Tag aus …", 0.2)
    regimes, z_spike, z_mr = [], np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        a = VIXAnalyzer(vix[:t + 1])
        regimes.append(a.current_regime())
        s = a.vix_spike_detection(float(params["threshold"]))
        if "z_score" in s:
            z_spike[t] = s["z_score"]
        m = a.mean_reversion_signal()
        if "z_score" in m:
            z_mr[t] = m["z_score"]
    regimes = np.array(regimes)
    spikes = np.flatnonzero(np.nan_to_num(z_spike) > params["threshold"])

    reg_de, reg_status = REGIME[regime]
    out.metric("VIX zuletzt", vix[-1], "num", digits=1, hint=f"Regime {reg_de}" + ("" if real else " · abgeleitet"),
               status=reg_status if real else None)
    out.metric("VIX-Perzentil", pct / 100, "pct", digits=0, hint=f"im Fenster von {int(params['pct_window'])} Tagen")
    out.metric("Spike-z heute", spike.get("z_score"), "num", digits=1,
               hint="Spike!" if spike.get("spike_detected") else f"Spike ab z > {params['threshold']:.1f}".replace(".", ","),
               status="serious" if spike.get("spike_detected") else None)
    out.metric("Mean-Reversion-z", mr.get("z_score"), "num", digits=2, hint=SIGNAL.get(mr["signal"], mr["signal"]),
               status="warning" if mr["signal"] != "NEUTRAL" else None)
    last_year = spikes[spikes >= n - 252]
    out.metric("Spikes im letzten Jahr", len(last_year), "int", hint=f"insgesamt {len(spikes)} in {n} Tagen")

    if real:
        if regime == "PANIC":
            level, title = "critical", "Panik im VIX"
        elif regime == "HIGH_FEAR":
            level, title = "serious", "Hohe Angst im VIX"
        elif regime == "ELEVATED" or spike.get("spike_detected"):
            level, title = "warning", "Erhöhte Angst" if regime == "ELEVATED" else "VIX-Spike heute"
        elif mr["signal"] == "COMPLACENCY_WARNING":
            level, title = "warning", "Auffällig niedrige Angst – Sorglosigkeit"
        else:
            level, title = "good", "Ruhiges VIX-Regime" if regime == "NORMAL" else "Niedrige Angst (sorglos)"
        out.set_verdict(level, title,
                        f"VIX {vix[-1]:.1f} im Regime „{reg_de}“, {pct:.0f}. Perzentil des letzten Fensters; "
                        f"Mean-Reversion: {SIGNAL.get(mr['signal'], mr['signal'])}.".replace(".", ",", 1))
    else:
        out.note("Der Datensatz enthält keinen VIX. Das Framework leitet ihn aus der realisierten Volatilität ab "
                 "(`generate_vix_from_returns`, Niveau 15 + 100 × Jahresvola) – die Werte sind illustrativ und meist "
                 "höher als der echte VIX.", "warning")

    # ---- charts ---------------------------------------------------------------------------------------
    out.chart(C.xy(
        "VIX mit Regimegrenzen und Spikes",
        [C.trace("VIX" if real else "VIX (aus Renditen abgeleitet)", x, vix, color=1, width=1.2),
         C.trace("Spike", [x[i] for i in spikes], vix[spikes], type="scatter", color="critical", size=6,
                 decimate_to=None)],
        x=xa, y=C.axis("VIX-Punkte"),
        hlines=[C.hline(th["complacent"], "sorglos < 15"), C.hline(th["normal"], "erhöht ab 25", "warning"),
                C.hline(th["elevated"], "hohe Angst ab 35", "serious"), C.hline(th["panic"], "Panik ab 50", "critical")],
        subtitle="Rote Punkte: Tage, an denen der VIX stärker stieg als die Spike-Schwelle erlaubt.",
    ))
    out.chart(C.xy(
        "Mean-Reversion-Signal (z-Wert des VIX)",
        [C.trace("z-Wert", x, z_mr, color=2, width=1.2)],
        x=xa, y=C.axis("z (gegen 252-Tage-Mittel)"),
        hlines=[C.hline(2.0, "Umkehr wahrscheinlich (z > 2)", "serious"),
                C.hline(-1.5, "Sorglosigkeit (z < −1,5)", "warning")],
        subtitle="Über 2: Angst extrem hoch, historisch oft nahe am Tief. Unter −1,5: ungewöhnlich ruhig.",
    ))

    out.progress("Phasenraum …", 0.7)
    X, Y, Z = compute_sentiment_coordinates(vix, r, window=int(params["coord_window"]))
    w = int(params["coord_window"])
    start = max(w, n - int(params["lookback_3d"]))
    idx = np.arange(start, n)
    traces = []
    for code in ("COMPLACENT", "NORMAL", "ELEVATED", "HIGH_FEAR", "PANIC"):
        m = idx[regimes[idx] == code]
        if m.size:
            traces.append(C.path3d(REGIME[code][0], X[m], Y[m], Z[m], color=REGIME[code][1] if code != "COMPLACENT"
                                   else 1, mode="markers", size=2.5,
                                   text=[str(label(i)) for i in m]))
    traces.append(C.path3d("Letzter Tag", [X[-1]], [Y[-1]], [Z[-1]], color="ink", mode="markers", size=8))
    out.chart(C.scatter3d(
        "Sentiment-Phasenraum", traces,
        x=C.axis("VIX-Niveau (0–1)"), y=C.axis("VIX-Momentum (0–1)"), z=C.axis("Rendite-VIX-Korrelation (0–1)"),
        subtitle=f"Letzte {len(idx)} Tage. Panik liegt rechts (hohes Niveau) und vorn (steigender VIX); "
                 "kleine Korrelation = Renditen und VIX laufen gegeneinander.",
    ))

    # ---- what happened next? ----------------------------------------------------------------------------
    c = np.concatenate([[0.0], np.cumsum(r)])
    fwd = np.full(n, np.nan)
    if n > H:
        fwd[:n - H] = np.expm1(c[1 + H:] - c[1:n + 1 - H])
    rows = []
    bounds = {"COMPLACENT": f"< {th['complacent']}", "NORMAL": f"{th['complacent']}–{th['normal']}",
              "ELEVATED": f"{th['normal']}–{th['elevated']}", "HIGH_FEAR": f"{th['elevated']}–{th['panic']}",
              "PANIC": f"≥ {th['panic']}"}
    for code in ("COMPLACENT", "NORMAL", "ELEVATED", "HIGH_FEAR", "PANIC"):
        m = regimes == code
        if not m.any():
            continue
        f = fwd[m][np.isfinite(fwd[m])]
        rows.append([REGIME[code][0], bounds[code], float(m.mean()), float(np.std(r[m]) * np.sqrt(252)),
                     float(np.mean(f)) if f.size else None, float(np.mean(f < 0)) if f.size else None])
    fa = fwd[np.isfinite(fwd)]
    rows.append(["Alle Tage", "", 1.0, float(np.std(r) * np.sqrt(252)), float(np.mean(fa)) if fa.size else None,
                 float(np.mean(fa < 0)) if fa.size else None])
    out.table("VIX-Regime und was danach geschah",
              ["Regime", "VIX", "Anteil der Tage", "Vola p. a. im Regime", f"Ø Rendite nächste {H} T",
               f"Anteil Verlust nach {H} T"], rows, formats=["text", "text", "pct", "pct", "pct", "pct"],
              note="Die Folgerendite beginnt am Tag nach der Einordnung. Hohe Angst ging historisch oft "
                   "überdurchschnittlichen Renditen voraus – das ist die Idee hinter dem Mean-Reversion-Signal.")

    if spikes.size:
        top = spikes[np.argsort(z_spike[spikes])[::-1][:10]]
        srows = [[label(i), float(vix[i]), float(vix[i] - vix[i - 1]), float(z_spike[i]), float(r[i]),
                  float(fwd[i]) if np.isfinite(fwd[i]) else None] for i in sorted(top)]
        out.table("Die zehn stärksten VIX-Spikes",
                  ["Datum", "VIX", "Anstieg (Punkte)", "z", "Rendite am Tag", f"Rendite nächste {H} T"], srows,
                  formats=["text", "num", "num", "num", "pct", "pct"],
                  note="z = Anstieg ÷ Standardabweichung der VIX-Tagesänderungen der 19 Tage davor.")
