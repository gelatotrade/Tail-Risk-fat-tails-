"""Shared pieces of the early-warning tools: German labels for the framework's codes and readings."""

import numpy as np

from src.analysis.early_warning import NetGammaExposure

# composite level -> (status colour, German label)
LEVEL = {
    "NORMAL": ("good", "Normal"),
    "ELEVATED": ("warning", "Erhöht"),
    "HIGH": ("serious", "Hoch"),
    "EXTREME": ("critical", "Extrem"),
}
ACTION = {
    "MONITOR": "Beobachten",
    "INCREASE HEDGES": "Absicherungen erhöhen",
    "SIGNIFICANT HEDGING REQUIRED": "Deutlich absichern",
    "IMMEDIATE RISK REDUCTION": "Risiko sofort reduzieren",
}
# first line of CompositeEarlyWarningSystem.get_risk_assessment -> German rendering of its text
ASSESSMENT = {
    "EXTREME RISK - IMMEDIATE ACTION REQUIRED": (
        "Mehrere Indikatoren signalisieren eine unmittelbare Verwerfung: negative Gamma-Lage beschleunigt "
        "Bewegungen, Smart Money sichert sich teuer ab (hoher TDEX), die VIX-Kurve ist invertiert, "
        "Institutionelle verteilen (niedriger DIX), Profis steigen aus (SMFI). Empfehlung des Frameworks: "
        "Risiko maximal reduzieren – Vollabsicherung oder Cash erwägen."),
    "HIGH RISK - SIGNIFICANT HEDGING REQUIRED": (
        "Mehrere Warnindikatoren sind erhöht: Die Marktstruktur wird fragil, institutionelle Absicherung nimmt "
        "zu, eine rasche Verschlechterung ist möglich. Empfehlung des Frameworks: Absicherungen deutlich "
        "ausbauen, Positionen verkleinern."),
    "ELEVATED RISK - INCREASED CAUTION": (
        "Einige Warnsignale: Die Lage verdient Aufmerksamkeit, erste Verschiebungen in der Positionierung sind "
        "sichtbar. Empfehlung des Frameworks: Absicherungen ergänzen, Stopps enger setzen."),
    "NORMAL CONDITIONS - MONITOR": (
        "Indikatoren im Normalbereich: Die Marktstruktur stützt, keine nennenswerten Divergenzen. Empfehlung "
        "des Frameworks: normales Risikomanagement beibehalten."),
}
# NetGammaExposure.get_regime_description (English) -> German
GEX_DESCRIPTION = {
    "Dealers aggressively selling into decline. Maximum acceleration risk.":
        "Dealer verkaufen aggressiv in den Abverkauf – maximales Beschleunigungsrisiko.",
    "Dealers selling as market falls. Volatility amplification active.":
        "Dealer verkaufen in fallende Kurse – Schwankungen werden verstärkt.",
    "Mixed dealer positioning. Normal market dynamics.":
        "Gemischte Dealer-Positionierung – normale Marktdynamik.",
    "Dealers buying dips. Volatility suppression active.":
        "Dealer kaufen Rücksetzer – Schwankungen werden gedämpft.",
    "Strong dealer buying support. Market likely pinned to strikes.":
        "Starke Dealer-Unterstützung – der Markt klebt vermutlich an großen Strikes.",
}
GEX_REGIME = {"EXTREME_NEGATIVE": "Stark negativ", "NEGATIVE": "Negativ", "NEUTRAL": "Neutral",
              "POSITIVE": "Positiv", "EXTREME_POSITIVE": "Stark positiv"}
GEX_SIGNAL = {"CRITICAL_WARNING": "kritische Warnung", "HIGH_WARNING": "hohe Warnung", "MODERATE": "moderat",
              "LOW": "gering", "SUPPRESSED": "gedämpft"}
TDEX_LEVEL = {"COMPLACENT": "Sorglos", "NORMAL": "Normal", "ELEVATED": "Erhöht", "EXTREME": "Extrem"}
TDEX_SIGNAL = {"WARNING_LOW_FEAR": "kaum Tail-Angst", "NEUTRAL": "neutral",
               "SMART_MONEY_HEDGING": "Smart Money sichert ab", "PANIC_PROTECTION": "Panik-Absicherung"}
STRUCTURE = {"STEEP_CONTANGO": "Steiles Contango", "MILD_CONTANGO": "Leichtes Contango", "FLAT": "Flach",
             "BACKWARDATION": "Backwardation", "SEVERE_BACKWARDATION": "Starke Backwardation"}
STRUCTURE_SIGNAL = {"COMPLACENT": "Sorglosigkeit", "NORMAL": "normal", "ELEVATED_CAUTION": "erhöhte Vorsicht",
                    "HIGH_FEAR": "hohe Angst", "CRITICAL_PANIC": "Panik"}
DIX_LEVEL = {"STRONG_BUYING": "Starke Käufe", "NORMAL": "Normal", "WEAK_BUYING": "Schwache Käufe",
             "DISTRIBUTION": "Distribution"}
DIX_SIGNAL = {"BULLISH_FLOW": "bullischer Fluss", "NEUTRAL": "neutral", "CAUTION": "Vorsicht",
              "BEARISH_FLOW": "bärischer Fluss"}
SMFI_SIGNAL = {"SMART_MONEY_ACCUMULATING": "Smart Money kauft", "SMART_MONEY_DISTRIBUTING": "Smart Money verkauft",
               "MIXED": "gemischt"}
DIVERGENCE = {"BEARISH": "bärisch", "BULLISH": "bullisch", "NONE": "keine"}
WARNING_LEVEL = {"HIGH": "hoch", "LOW": "niedrig", "OPPORTUNITY": "Chance", "NEUTRAL": "neutral"}

NAMES = {"gex": "Net Gamma (GEX)", "tdex": "TailDex (TDEX)", "vix_term": "VIX-Kurve",
         "dix": "Dark Index (DIX)", "smfi": "Smart Money Flow (SMFI)"}
KEYS = ("gex", "tdex", "vix_term", "dix", "smfi")


def de(x, digits=2, sign=False):
    """German number string: 1234.5 -> '1.234,50'; sign=True prefixes '+' for positives."""
    s = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")


def status_of(score):
    """Status colour of a 0-100 warning score (composite level bands 50 / 75 / 90)."""
    return "critical" if score >= 90 else "serious" if score >= 75 else "warning" if score >= 50 else "good"


def assessment_de(text):
    head = text.strip().splitlines()[0].strip() if text and text.strip() else ""
    return ASSESSMENT.get(head, text.strip())


def gex_snapshot(net_gex, spot, flip):
    """``NetGammaExposure.calculate_gex`` on the smallest options chain with exactly this reading.

    Two strikes around the flip level carry per-strike GEX of -n and +2n (bn $), so the
    cumulative profile is [-n, +n]: net GEX = n and the zero crossing sits exactly at ``flip``
    (mirrored for negative GEX).  Gamma is 1, open interest is scaled to the spot.
    """
    gex = NetGammaExposure(spot_price=float(spot))
    n = float(net_gex) if abs(net_gex) > 1e-9 else 1e-9
    unit = float(spot) ** 2 * 0.01 * 100 / 1e9  # bn $ per unit of gamma x OI at this spot
    h = 0.01 * float(spot)
    per_strike = np.array([-n, 2.0 * n])
    ones = np.ones(2)
    res = gex.calculate_gex(ones, ones, np.clip(per_strike, 0, None) / unit, np.clip(-per_strike, 0, None) / unit,
                            np.array([flip - h, flip + h]))
    return res, gex
