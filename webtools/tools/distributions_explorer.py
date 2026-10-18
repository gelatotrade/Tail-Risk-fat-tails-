"""Fat-tail families side by side: Student-t, GH, NIG and Variance Gamma at the same volatility."""

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid, trapezoid

from src.models.fat_tail_distributions import (
    GeneralizedHyperbolic,
    NormalInverseGaussian,
    StudentTDistribution,
    VarianceGamma,
)

from .. import charts as C

LEVELS = [{"value": v, "label": l} for v, l in
          ((0.95, "95 %"), (0.975, "97,5 %"), (0.99, "99 %"), (0.995, "99,5 %"), (0.999, "99,9 %"))]

SPEC = {
    "id": "distributions-explorer",
    "title": "Fat-Tail-Verteilungen im Vergleich",
    "group": "data",
    "order": 30,
    "summary": "Student-t, Generalisierte Hyperbolische, NIG und Variance Gamma bei gleicher Volatilität: "
               "Wie viel dicker sind ihre Ränder, und wie stark steigen VaR und Expected Shortfall?",
    "description": (
        "Alle fünf Verteilungen haben hier **dieselbe Tagesvolatilität** und den Mittelwert 0. Sie unterscheiden "
        "sich nur in der Form: Die **Student-t** hat Ränder nach einem Potenzgesetz (Freiheitsgrade ν: je kleiner, "
        "desto dicker). Die **Generalisierte Hyperbolische (GH)** ist eine Normalverteilung, deren Varianz selbst "
        "zufällig ist (GIG-Mischung); λ und α·δ steuern Form und Randgewicht. **NIG** ist der Spezialfall λ = −½, "
        "**Variance Gamma** eine Brownsche Bewegung mit Gamma-Zeitwechsel (ν = Varianz der Zeit, ≈ Exzess-Kurtosis/3).\n\n"
        "So liest du die Diagramme: Auf der **logarithmischen** Dichte-Achse ist die Normalverteilung eine nach unten "
        "offene Parabel, die Fat-Tail-Familien fallen fast linear (exponentiell) oder noch langsamer (Potenzgesetz). "
        "Das Rand-Diagramm zeigt, wie oft ein Verlust jenseits von k σ vorkommt; bei 5 σ trennen die Modelle "
        "oft Größenordnungen. Gleiche Volatilität heißt also keineswegs gleiches Crash-Risiko.\n\n"
        "VaR und Expected Shortfall sind Tagesverluste als positive Dezimalzahl. Gauß und Student-t werden analytisch "
        "berechnet, GH, NIG und Variance Gamma per Monte Carlo aus den `sample`-Methoden des Frameworks. Der "
        "Schiefe-Parameter wirkt als β/α bei GH und NIG und als θ/σ bei Variance Gamma; negativ heißt: Der "
        "Verlust-Rand ist schwerer."
    ),
    "needs": [],
    "params": [
        {"name": "confidence", "type": "select", "default": 0.99, "label": "Konfidenzniveau", "options": LEVELS},
        {"name": "vol", "type": "float", "default": 0.01, "min": 0.002, "max": 0.05, "step": 0.001,
         "label": "Tagesvolatilität σ", "help": "Gemeinsame Standardabweichung aller Verteilungen (0,01 = 1 % pro Tag)."},
        {"name": "t_nu", "type": "float", "default": 4.0, "min": 2.2, "max": 50.0, "step": 0.1,
         "label": "Student-t: Freiheitsgrade ν", "help": "Kleiner = dickere Ränder; ab ≈ 30 fast normal."},
        {"name": "gh_lambda", "type": "float", "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1,
         "label": "GH: Form λ", "help": "λ = 1: Hyperbolisch, λ = −½: NIG."},
        {"name": "alpha_delta", "type": "float", "default": 0.6, "min": 0.15, "max": 20.0, "step": 0.05,
         "label": "GH/NIG: Randgewicht α·δ", "help": "Dimensionslos. Klein = dicke Ränder, groß = fast normal."},
        {"name": "skew", "type": "float", "default": -0.2, "min": -0.8, "max": 0.8, "step": 0.05,
         "label": "Schiefe", "help": "β/α bei GH und NIG, θ/σ bei Variance Gamma. Negativ = Crash-Rand schwerer."},
        {"name": "vg_nu", "type": "float", "default": 1.0, "min": 0.05, "max": 5.0, "step": 0.05,
         "label": "Variance Gamma: ν", "help": "Varianz des Gamma-Zeitwechsels; Exzess-Kurtosis ≈ 3ν."},
        {"name": "n_samples", "type": "int", "default": 200000, "min": 20000, "max": 1000000, "step": 20000,
         "label": "Monte-Carlo-Ziehungen"},
        {"name": "seed", "type": "int", "default": 42, "min": 0, "max": 99999, "step": 1, "label": "Zufalls-Seed"},
    ],
    "covers": [
        "src.models.fat_tail_distributions.StudentTDistribution",
        "src.models.fat_tail_distributions.StudentTDistribution.pdf",
        "src.models.fat_tail_distributions.StudentTDistribution.cdf",
        "src.models.fat_tail_distributions.StudentTDistribution.ppf",
        "src.models.fat_tail_distributions.StudentTDistribution.sample",
        "src.models.fat_tail_distributions.StudentTDistribution.var",
        "src.models.fat_tail_distributions.StudentTDistribution.expected_shortfall",
        "src.models.fat_tail_distributions.GeneralizedHyperbolic",
        "src.models.fat_tail_distributions.GeneralizedHyperbolic.pdf",
        "src.models.fat_tail_distributions.GeneralizedHyperbolic.sample",
        "src.models.fat_tail_distributions.GeneralizedHyperbolic.moments",
        "src.models.fat_tail_distributions.NormalInverseGaussian",
        "src.models.fat_tail_distributions.NormalInverseGaussian.pdf",
        "src.models.fat_tail_distributions.NormalInverseGaussian.sample",
        "src.models.fat_tail_distributions.VarianceGamma",
        "src.models.fat_tail_distributions.VarianceGamma.pdf",
        "src.models.fat_tail_distributions.VarianceGamma.sample",
        "src.models.fat_tail_distributions.VarianceGamma.characteristic_function",
    ],
    "cost": "fast",
}

SPAN = 400.0  # integration range in sigma; exponential tails are negligible beyond


def _de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def _sinh_grid(vol, n=40001):
    """Dense near 0, coarse in the far tails: x = vol * sinh(t)."""
    return vol * np.sinh(np.linspace(-np.arcsinh(SPAN), np.arcsinh(SPAN), n))


def _left_tail_from_pdf(pdf, xs, vol):
    """P(X < x) for x <= -1 sigma by integrating the density from the far left (-SPAN sigma)."""
    x = -np.geomspace(SPAN * vol, vol, 30000)
    f = np.nan_to_num(np.asarray(pdf(x), dtype=float), nan=0.0, posinf=0.0)
    cum = cumulative_trapezoid(f, x, initial=0.0)
    return np.interp(xs, x, cum)


def _moments_from_pdf(pdf, vol):
    """Mean, std and excess kurtosis by numerical integration (smooth densities only)."""
    x = _sinh_grid(vol)
    f = np.nan_to_num(np.asarray(pdf(x), dtype=float), nan=0.0, posinf=0.0)
    mass = trapezoid(f, x)
    mean = trapezoid(x * f, x) / mass
    var = trapezoid((x - mean) ** 2 * f, x) / mass
    m4 = trapezoid((x - mean) ** 4 * f, x) / mass
    return mass, mean, np.sqrt(var), m4 / var ** 2 - 3.0


def _kurtosis_from_cf(vg, std):
    """Excess kurtosis from the characteristic function: Re log phi(s/std) = -s^2/2 + k4 s^4/24 - ..."""
    s = np.linspace(-0.15, 0.15, 121)
    logphi = np.log(vg.characteristic_function(s / std))
    poly = np.polynomial.Polynomial.fit(s, logphi.real, 12).convert()
    c = np.zeros(13)
    c[:len(poly.coef)] = poly.coef
    k2, k4 = -2.0 * c[2], 24.0 * c[4]
    return k4 / k2 ** 2


def _var_es_from_sample(x, conf):
    var = -float(np.quantile(x, 1 - conf))
    tail = x[x <= -var]
    return var, (-float(np.mean(tail)) if tail.size else var)


def run(params, data, out):
    conf, vol = float(params["confidence"]), float(params["vol"])
    nu, lam, ad = params["t_nu"], params["gh_lambda"], params["alpha_delta"]
    skew, vg_nu = params["skew"], params["vg_nu"]
    n_s, seed = int(params["n_samples"]), int(params["seed"])
    alpha = 1 - conf

    # --- build every family with mean 0 and std = vol (all are location-scale families)
    t_dist = StudentTDistribution(nu=nu, loc=0.0, scale=vol * np.sqrt((nu - 2) / nu))

    def gh_like(lam_):
        base = GeneralizedHyperbolic(lam_, ad, skew * ad, 1.0, 0.0)
        mom = base.moments()
        c = vol / float(mom["std"])
        return c, float(mom["mean"])

    c_gh, m_gh = gh_like(lam)
    gh = GeneralizedHyperbolic(lam, ad / c_gh, skew * ad / c_gh, c_gh, -m_gh * c_gh)
    c_nig, m_nig = gh_like(-0.5)
    nig = NormalInverseGaussian(ad / c_nig, skew * ad / c_nig, c_nig, -m_nig * c_nig)
    c_vg = vol / np.sqrt(1.0 + skew ** 2 * vg_nu)
    vg = VarianceGamma(sigma=c_vg, theta=skew * c_vg, nu=vg_nu, mu=-skew * c_vg)
    gauss = stats.norm(0.0, vol)

    # --- Monte Carlo samples from the framework's samplers (global RNG reseeded per family)
    out.progress("Ziehe Stichproben …", 0.2)
    np.random.seed(seed)
    s_t = np.asarray(t_dist.sample(n_s), dtype=float)
    s_gh = np.asarray(gh.sample(n_s, seed=seed + 1), dtype=float)
    s_nig = np.asarray(nig.sample(n_s, seed=seed + 2), dtype=float)
    s_vg = np.asarray(vg.sample(n_s, seed=seed + 3), dtype=float)

    fams = [
        {"key": "gauss", "name": "Normalverteilung", "short": "Gauß", "color": "muted", "pdf": gauss.pdf,
         "cdf": gauss.cdf, "par": f"σ = {_de(100 * vol)} %", "sample": None},
        {"key": "t", "name": f"Student-t (ν = {_de(nu, 1)})", "short": "Student-t", "color": 1, "pdf": t_dist.pdf,
         "cdf": t_dist.cdf, "par": f"ν = {_de(nu, 1)}", "sample": s_t},
        {"key": "gh", "name": f"GH (λ = {_de(lam, 1)})", "short": "GH", "color": 2, "pdf": gh.pdf, "cdf": None,
         "par": f"λ = {_de(lam, 1)}, α·δ = {_de(ad)}, β/α = {_de(skew)}", "sample": s_gh},
        {"key": "nig", "name": "NIG (λ = −½)", "short": "NIG", "color": 3, "pdf": nig.pdf, "cdf": None,
         "par": f"α·δ = {_de(ad)}, β/α = {_de(skew)}", "sample": s_nig},
        {"key": "vg", "name": f"Variance Gamma (ν = {_de(vg_nu)})", "short": "Variance Gamma", "color": 4,
         "pdf": vg.pdf, "cdf": None, "par": f"ν = {_de(vg_nu)}, θ/σ = {_de(skew)}", "sample": s_vg},
    ]

    # --- risk numbers per family
    out.progress("Berechne VaR, ES und Randwahrscheinlichkeiten …", 0.5)
    ks = np.linspace(1.0, 10.0, 91)
    z_a = stats.norm.ppf(alpha)
    for f in fams:
        if f["key"] == "gauss":
            f["var"], f["es"] = -vol * z_a, vol * stats.norm.pdf(z_a) / alpha
            f["kurt"], f["sd"] = 0.0, vol
        elif f["key"] == "t":
            f["var"], f["es"] = float(t_dist.var(conf)), float(t_dist.expected_shortfall(conf))
            f["kurt"] = 6.0 / (nu - 4.0) if nu > 4 else np.inf
        else:
            f["var"], f["es"] = _var_es_from_sample(f["sample"], conf)
        if f["sample"] is not None:
            f["sd"] = float(np.std(f["sample"]))
        if f["cdf"] is not None:
            f["tail"] = np.asarray(f["cdf"](-ks * vol), dtype=float)
        else:
            f["tail"] = _left_tail_from_pdf(f["pdf"], -ks * vol, vol)
    for f in fams:
        if f["key"] in ("gh", "nig"):
            f["kurt"] = _moments_from_pdf(f["pdf"], vol)[3]
        elif f["key"] == "vg":
            f["kurt"] = _kurtosis_from_cf(vg, vol)

    g = fams[0]
    p3_g, p5_g = float(np.interp(3, ks, g["tail"])), float(np.interp(5, ks, g["tail"]))
    worst = max(fams[1:], key=lambda f: f["es"])
    worst5 = max(fams[1:], key=lambda f: float(np.interp(5, ks, f["tail"])))
    p5_worst = float(np.interp(5, ks, worst5["tail"]))
    lvl = next(o["label"] for o in LEVELS if o["value"] == params["confidence"])

    out.metric("Tagesvolatilität (alle gleich)", vol, "pct", hint="Mittelwert 0, gleiche Standardabweichung")
    out.metric(f"VaR {lvl} · Gauß", g["var"], "pct")
    out.metric(f"Expected Shortfall {lvl} · Gauß", g["es"], "pct")
    out.metric(f"Höchster Expected Shortfall {lvl}", worst["es"], "pct", hint=worst["short"],
               status="serious" if worst["es"] > 1.3 * g["es"] else "warning" if worst["es"] > 1.1 * g["es"] else None)
    out.metric("ES-Aufschlag ggü. Gauß", worst["es"] / g["es"], "x", hint=f"{worst['short']} ÷ Gauß")
    out.metric("5-σ-Verlust: häufiger als Gauß", p5_worst / p5_g if p5_g > 0 else None, "x",
               hint=f"{worst5['short']}: einmal in {1 / p5_worst / 252:,.0f} Jahren statt {1 / p5_g / 252:,.0f}".replace(",", ".")
               if p5_worst > 0 else "")

    # --- chart 1: densities on a log scale
    x = np.linspace(-8 * vol, 8 * vol, 641)
    peak = float(gauss.pdf(0.0))
    floor = peak * 1e-7
    traces = []
    for f in fams:
        y = np.asarray(f["pdf"](x), dtype=float)
        y = np.where(np.isfinite(y) & (y >= floor), np.minimum(y, 50 * peak), np.nan)
        traces.append(C.trace(f["name"], x, y, color=f["color"], dash=f["key"] == "gauss",
                              width=2.5 if f["key"] != "gauss" else 2))
    top = np.nanmax([np.nanmax(tr["y"]) for tr in traces])
    out.chart(C.xy(
        "Dichten bei gleicher Volatilität – logarithmisch", traces,
        x=C.axis("Tagesrendite", fmt="pct", digits=0), y=C.axis("Dichte", type="log", range=(floor, top * 2)),
        subtitle="Die Gauß-Parabel stürzt ab, die Fat-Tail-Familien fallen viel langsamer.",
        vlines=[C.vline(-g["var"], f"Gauß-VaR {lvl}")],
    ))

    # --- chart 2: left tail probability beyond k sigma
    floor_p = 1e-10
    tail_traces = [C.trace(f["name"], ks, np.where(f["tail"] >= floor_p, f["tail"], np.nan), color=f["color"],
                           dash=f["key"] == "gauss") for f in fams]
    out.chart(C.xy(
        "Wie oft ein Verlust jenseits von k σ vorkommt", tail_traces,
        x=C.axis("Verlust in Vielfachen von σ"), y=C.axis("P(Rendite < −k σ)", type="log", range=(floor_p, 0.5)),
        subtitle="Logarithmische Achse: Jede Gitterlinie ist ein Faktor 10.",
        hlines=[C.hline(1 / 252, "einmal pro Jahr"), C.hline(1 / 2520, "einmal in 10 Jahren")],
    ))

    # --- chart 3: VaR and ES per family
    names = [f["short"] for f in fams]
    out.chart(C.xy(
        f"VaR und Expected Shortfall {lvl} je Verteilung",
        [C.trace(f"VaR {lvl}", names, [f["var"] for f in fams], type="bar", color=1),
         C.trace(f"Expected Shortfall {lvl}", names, [f["es"] for f in fams], type="bar", color=2)],
        x=C.axis("", type="category"), y=C.axis("Tagesverlust", fmt="pct", digits=1), barmode="group",
        subtitle="Gleiche Volatilität, unterschiedliche Ränder: Der Expected Shortfall reagiert stärker als der VaR.",
        hlines=[C.hline(g["es"], "ES Gauß")],
    ))

    rows = []
    for f in fams:
        p3, p5 = float(np.interp(3, ks, f["tail"])), float(np.interp(5, ks, f["tail"]))
        rows.append([f["short"], f["par"], f["var"], f["es"], f["es"] / f["var"], p3, p5,
                     p5 / p5_g if p5_g > 0 else None, f["kurt"] if np.isfinite(f["kurt"]) else "∞", f["sd"]])
    out.table(
        "Verteilungen im Überblick", ["Verteilung", "Parameter", f"VaR {lvl}", f"ES {lvl}", "ES ÷ VaR",
                                      "P(< −3 σ)", "P(< −5 σ)", "5 σ vs. Gauß", "Exzess-Kurtosis", "σ (Stichprobe)"],
        rows, formats=["text", "text", "pct", "pct", "num", "pct", "pct", "x", "num", "pct"],
        note=(f"VaR/ES von Gauß und Student-t analytisch, sonst aus {n_s:,} Ziehungen. ".replace(",", ".")
              + "Randwahrscheinlichkeiten durch Integration der Dichten; Exzess-Kurtosis der t-Verteilung ist für ν ≤ 4 "
              "unendlich, bei Variance Gamma aus der charakteristischen Funktion berechnet."),
    )
    if p3_g > 0:
        out.note(f"Ein 3-σ-Verlust hat unter der Normalverteilung die Wahrscheinlichkeit {_de(100 * p3_g, 3)} % pro Tag "
                 f"(etwa einmal in {_de(1 / p3_g / 252, 1)} Jahren). Die Tabelle zeigt, wie viel häufiger er unter "
                 "den Fat-Tail-Familien ist.", "info")
