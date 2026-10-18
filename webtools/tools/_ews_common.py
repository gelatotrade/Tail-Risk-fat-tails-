"""Shared pieces of the early-warning tools: English labels for the framework's codes and readings."""

import numpy as np

from src.analysis.early_warning import NetGammaExposure

# composite level -> (status colour, label)
LEVEL = {
    "NORMAL": ("good", "Normal"),
    "ELEVATED": ("warning", "Elevated"),
    "HIGH": ("serious", "High"),
    "EXTREME": ("critical", "Extreme"),
}
ACTION = {
    "MONITOR": "Monitor",
    "INCREASE HEDGES": "Increase hedges",
    "SIGNIFICANT HEDGING REQUIRED": "Hedge substantially",
    "IMMEDIATE RISK REDUCTION": "Reduce risk immediately",
}
# first line of CompositeEarlyWarningSystem.get_risk_assessment -> prose rendering of its text
ASSESSMENT = {
    "EXTREME RISK - IMMEDIATE ACTION REQUIRED": (
        "Several indicators point to an imminent dislocation: negative gamma accelerates moves, smart money is "
        "paying up for protection (high TDEX), the VIX curve is inverted, institutions are distributing (low DIX) "
        "and professional money is exiting (SMFI). The framework recommends maximum risk reduction: consider a "
        "full hedge or moving to cash."),
    "HIGH RISK - SIGNIFICANT HEDGING REQUIRED": (
        "Several warning indicators are elevated: market structure is becoming fragile, institutional hedging is "
        "increasing and a rapid deterioration is possible. The framework recommends increasing hedges "
        "substantially and reducing position sizes."),
    "ELEVATED RISK - INCREASED CAUTION": (
        "Some warning signals are present: conditions warrant attention and early shifts in positioning are "
        "visible. The framework recommends adding hedges and tightening stop losses."),
    "NORMAL CONDITIONS - MONITOR": (
        "Indicators are within normal ranges: market structure is supportive and there are no significant "
        "divergences. The framework recommends maintaining standard risk management."),
}
# NetGammaExposure.get_regime_description -> edited English
GEX_DESCRIPTION = {
    "Dealers aggressively selling into decline. Maximum acceleration risk.":
        "Dealers are selling aggressively into the decline – maximum acceleration risk.",
    "Dealers selling as market falls. Volatility amplification active.":
        "Dealers are selling into falling prices – volatility is being amplified.",
    "Mixed dealer positioning. Normal market dynamics.":
        "Mixed dealer positioning – normal market dynamics.",
    "Dealers buying dips. Volatility suppression active.":
        "Dealers are buying the dips – volatility is being dampened.",
    "Strong dealer buying support. Market likely pinned to strikes.":
        "Strong dealer support – the market is likely pinned to large strikes.",
}
GEX_REGIME = {"EXTREME_NEGATIVE": "Strongly negative", "NEGATIVE": "Negative", "NEUTRAL": "Neutral",
              "POSITIVE": "Positive", "EXTREME_POSITIVE": "Strongly positive"}
GEX_SIGNAL = {"CRITICAL_WARNING": "Critical warning", "HIGH_WARNING": "High warning", "MODERATE": "Moderate",
              "LOW": "Low", "SUPPRESSED": "Dampened"}
TDEX_LEVEL = {"COMPLACENT": "Complacent", "NORMAL": "Normal", "ELEVATED": "Elevated", "EXTREME": "Extreme"}
TDEX_SIGNAL = {"WARNING_LOW_FEAR": "Little tail fear", "NEUTRAL": "Neutral",
               "SMART_MONEY_HEDGING": "Smart money hedging", "PANIC_PROTECTION": "Panic hedging"}
STRUCTURE = {"STEEP_CONTANGO": "Steep contango", "MILD_CONTANGO": "Mild contango", "FLAT": "Flat",
             "BACKWARDATION": "Backwardation", "SEVERE_BACKWARDATION": "Severe backwardation"}
STRUCTURE_SIGNAL = {"COMPLACENT": "Complacency", "NORMAL": "Normal", "ELEVATED_CAUTION": "Heightened caution",
                    "HIGH_FEAR": "High fear", "CRITICAL_PANIC": "Panic"}
DIX_LEVEL = {"STRONG_BUYING": "Strong buying", "NORMAL": "Normal", "WEAK_BUYING": "Weak buying",
             "DISTRIBUTION": "Distribution"}
DIX_SIGNAL = {"BULLISH_FLOW": "Bullish flow", "NEUTRAL": "Neutral", "CAUTION": "Caution",
              "BEARISH_FLOW": "Bearish flow"}
SMFI_SIGNAL = {"SMART_MONEY_ACCUMULATING": "Smart money buying", "SMART_MONEY_DISTRIBUTING": "Smart money selling",
               "MIXED": "Mixed"}
DIVERGENCE = {"BEARISH": "Bearish", "BULLISH": "Bullish", "NONE": "None"}
WARNING_LEVEL = {"HIGH": "High", "LOW": "Low", "OPPORTUNITY": "Opportunity", "NEUTRAL": "Neutral"}

NAMES = {"gex": "Net Gamma (GEX)", "tdex": "TailDex (TDEX)", "vix_term": "VIX curve",
         "dix": "Dark Index (DIX)", "smfi": "Smart Money Flow (SMFI)"}
KEYS = ("gex", "tdex", "vix_term", "dix", "smfi")


def num(x, digits=2, sign=False):
    """English number string: 1234.5 -> '1,234.50'; sign=True prefixes '+' for positives (minus as U+2212)."""
    s = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    return s.replace("-", "−")


def bn(x, digits=1, sign=False):
    """US dollars in billions: 4.2 -> '$4.2bn'; sign=True gives '+$4.2bn' / '−$4.2bn'.

    A word joiner (U+2060) follows the sign so narrow table cells never wrap between sign and '$'.
    """
    body = f"${abs(x):,.{digits}f}bn"
    if x < 0 and round(abs(x), digits) != 0:
        return "−⁠" + body
    return ("+⁠" + body) if sign else body


def ordinal(x):
    """62.4 -> '62nd' (for percentiles); '–' for a missing value."""
    if x is None or not np.isfinite(float(x)):
        return "–"
    n = int(round(float(x)))
    suffix = "th" if 10 <= abs(n) % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")
    return f"{n:,}{suffix}"


def status_of(score):
    """Status colour of a 0-100 warning score (composite level bands 50 / 75 / 90)."""
    return "critical" if score >= 90 else "serious" if score >= 75 else "warning" if score >= 50 else "good"


def assessment_text(text):
    """Prose version of ``CompositeEarlyWarningSystem.get_risk_assessment`` (falls back to the raw text)."""
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
