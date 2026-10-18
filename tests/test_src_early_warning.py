"""Regression tests for src/analysis/early_warning.py."""

import numpy as np
import pytest
from scipy import stats

from src.analysis.early_warning import (
    CompositeEarlyWarningSystem,
    DarkIndex,
    NetGammaExposure,
    SmartMoneyFlowIndex,
    TailDex,
    VIXTermStructure,
)
from src.utils.data_loader import generate_crisis_scenario, generate_vix_from_returns

INVERTED_LABELS = {"FLAT", "BACKWARDATION", "SEVERE_BACKWARDATION"}


# ------------------------------------------------------------ VIX term structure
def test_term_structure_contango_is_labelled_contango():
    res = VIXTermStructure().calculate_term_structure(15, 16, 17.5)  # M2/M1 = 1.094
    assert res["structure"] == "STEEP_CONTANGO" and res["signal"] == "COMPLACENT"
    assert not res["inverted"]
    res = VIXTermStructure().calculate_term_structure(15, 16, 16.4)  # M2/M1 = 1.025
    assert res["structure"] == "MILD_CONTANGO" and res["signal"] == "NORMAL"


def test_term_structure_inversion_is_labelled_backwardation():
    res = VIXTermStructure().calculate_term_structure(50, 45, 38)  # M2/M1 = 0.84
    assert res["structure"] == "SEVERE_BACKWARDATION" and res["signal"] == "CRITICAL_PANIC"
    assert res["inverted"]
    res = VIXTermStructure().calculate_term_structure(30, 30, 28.8)  # M2/M1 = 0.96
    assert res["structure"] == "FLAT"


@pytest.mark.parametrize("m2", np.linspace(14.0, 26.0, 25))
def test_term_structure_label_agrees_with_inverted_flag(m2):
    res = VIXTermStructure().calculate_term_structure(20.0, 20.0, float(m2))
    assert res["ratio_m1_m2"] == pytest.approx(20.0 / m2)
    assert res["inverted"] == (res["structure"] in INVERTED_LABELS)
    assert (res["warning_score"] >= 30) == bool(res["inverted"])


def test_term_structure_zero_guards():
    # used to raise ZeroDivisionError in roll_yield / ratio_m1_m4
    res = VIXTermStructure().calculate_term_structure(18, 0, 0)
    assert np.isfinite(res["roll_yield"]) and res["ratio_m1_m2"] == 1.0
    res = VIXTermStructure().calculate_term_structure(18, 17, 18, 0)
    assert res["ratio_m1_m4"] == 1.0
    res = VIXTermStructure().calculate_term_structure(18, 17, 18.5)
    assert res["roll_yield"] == pytest.approx((18.5 - 17) / 17 * 12)


# ----------------------------------------------------------------- GEX gamma flip
def _flip(strikes, call_oi, put_oi, spot=100.0, gamma=None):
    k = np.asarray(strikes, dtype=float)
    g = np.ones_like(k) if gamma is None else gamma
    return NetGammaExposure(spot).calculate_gex(g, g, np.asarray(call_oi, float), np.asarray(put_oi, float), k)


def test_gamma_flip_is_the_zero_crossing_of_the_cumulative_profile():
    # net gamma*OI by strike [-2, -1, +5] -> cumulative [-2, -3, +2]: crosses 0 at 100 + 10*3/5
    res = _flip([90, 100, 110], [0, 0, 5], [2, 1, 0])
    assert res["flip_level"] == pytest.approx(106.0)
    assert res["distance_to_flip"] == pytest.approx(-6.0)


def test_gamma_flip_ignores_flat_deep_otm_start_and_unsorted_strikes():
    # cumulative ~0 over the deep-OTM wings: the old argmin(|cum|) returned the lowest strike
    k = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140.0])
    call_oi = np.array([0, 0, 0, 0, 1, 4, 0, 0, 0.0])
    put_oi = np.array([0, 0, 0, 2, 1, 0, 0, 0, 0.0])
    order = np.random.default_rng(0).permutation(len(k))
    res = _flip(k[order], call_oi[order], put_oi[order])
    assert res["flip_level"] == pytest.approx(105.0)  # cumulative -2, -2, +2


def test_gamma_flip_black_scholes_chain():
    S, T, sig = 5000.0, 30 / 365, 0.18
    K = np.arange(4000, 6001, 25.0)
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    gamma = stats.norm.pdf(d1) / (S * sig * np.sqrt(T))
    call_oi = 2e4 * np.exp(-((K - 5200) / 300) ** 2)
    put_oi = 1e4 * np.exp(-((K - 4800) / 300) ** 2)
    res = NetGammaExposure(S).calculate_gex(gamma, gamma, call_oi, put_oi, K)
    cum = np.cumsum(gamma * (call_oi - put_oi))
    j = np.flatnonzero(np.diff(np.sign(cum)))[-1] + 1
    expected = K[j - 1] + 25 * cum[j - 1] / (cum[j - 1] - cum[j])
    assert res["flip_level"] == pytest.approx(expected)
    assert K[0] < res["flip_level"] < K[-1]
    assert res["net_gex"] > 0


def test_gamma_flip_without_crossing_lies_outside_the_chain():
    k = [90, 100, 110]
    assert _flip(k, [1, 2, 1], [0, 0, 0])["flip_level"] == 90.0     # all positive: below
    assert _flip(k, [0, 0, 0], [1, 2, 1])["flip_level"] == 110.0    # all negative: above


def test_gex_empty_chain():
    # used to raise ValueError at np.argmin before the empty-chain guard
    e = np.array([])
    res = NetGammaExposure(5000.0).calculate_gex(e, e, e, e, e)
    assert res["net_gex"] == 0.0 and res["flip_level"] == 5000.0


# ---------------------------------------------------------------- DIX divergence
def test_dix_divergence_can_trigger():
    d = DarkIndex()
    up, down = np.linspace(100, 110, 20), np.linspace(48, 40, 20)
    res = d.detect_divergence(up, down)  # used to return NONE (trend capped at 0.17)
    assert res["type"] == "BEARISH" and res["divergence"]
    assert d.detect_divergence(up[::-1], down[::-1])["type"] == "BULLISH"


def test_dix_trend_is_correlation_with_time():
    rng = np.random.default_rng(1)
    p = 100 + np.cumsum(rng.normal(0, 1, 30))
    x = 45 + np.cumsum(rng.normal(0, 0.5, 30))
    res = DarkIndex().detect_divergence(p, x, window=30)
    t = np.arange(30)
    assert res["price_trend"] == pytest.approx(np.corrcoef(t, p)[0, 1], rel=1e-9)
    assert res["dix_trend"] == pytest.approx(np.corrcoef(t, x)[0, 1], rel=1e-9)


# ------------------------------------------------ component scores on 0-100
def test_dix_warning_score_monotone_and_on_0_100_scale():
    d = DarkIndex()
    grid = np.linspace(30, 50, 401)
    for pct in (0.0, 10.0, 50.0):
        s = np.array([d._compute_warning_score(x, pct) for x in grid])
        assert np.all(np.diff(s) <= 1e-12)  # lower DIX never lowers the warning
        assert s.min() >= 0 and s.max() <= 100
    assert d._compute_warning_score(30.0, 0.0) == pytest.approx(100.0)
    assert d._compute_warning_score(39.99, 50.0) >= d._compute_warning_score(40.0, 50.0)


def test_smfi_warning_score_on_0_100_scale():
    s = SmartMoneyFlowIndex()
    assert s._compute_warning_score(-20.0, -0.02) == pytest.approx(100.0)
    assert s._compute_warning_score(-7.5, -0.0075) == pytest.approx(100 * 0.75 * 0.75)
    assert s._compute_warning_score(3.0, -0.02) == 0


def test_composite_of_maximal_component_scores_is_extreme():
    # DIX/SMFI were pre-weighted (max 20 / 10) and then weighted again: composite <= 83
    ews = CompositeEarlyWarningSystem()
    scores = dict(
        gex_score=ews.gex._compute_warning_score(-10.0, ews.gex.spot_price),
        tdex_score=TailDex()._compute_warning_score(100.0),
        vix_term_score=ews.vix_term._compute_warning_score(1.2, 50.0),
        dix_score=ews.dix._compute_warning_score(30.0, 0.0),
        smfi_score=ews.smfi._compute_warning_score(-20.0, -0.02),
    )
    res = ews.compute_composite(**scores)
    assert res["composite_score"] == pytest.approx(100.0)
    assert res["level"] == "EXTREME"
    assert "EXTREME RISK" in ews.get_risk_assessment(res["composite_score"])


def test_simulated_scores_are_0_100_and_dix_score_monotone():
    r, _ = generate_crisis_scenario(n_pre=300, n_crisis=100, n_post=200, seed=2)
    vix = generate_vix_from_returns(r)
    np.random.seed(0)
    ind = CompositeEarlyWarningSystem().simulate_all_indicators(r, 100 * np.exp(np.cumsum(r)), vix)
    for k in ("gex_score", "tdex_score", "vix_term_score", "dix_score", "smfi_score", "composite_score"):
        assert ind[k].min() >= 0 and ind[k].max() <= 100 + 1e-9, k
    order = np.argsort(ind["dix"])
    assert ind["dix"].min() < 39 and ind["dix"].max() > 43
    assert np.all(np.diff(ind["dix_score"][order]) <= 1e-9)  # no jump down at DIX 40
    np.testing.assert_allclose(ind["dix_score"], np.clip((43 - ind["dix"]) * 12.5, 0, 100))
    w = {"gex": 0.35, "tdex": 0.25, "vix_term": 0.20, "dix": 0.10, "smfi": 0.10}
    np.testing.assert_allclose(ind["composite_score"], sum(w[k] * ind[k + "_score"] for k in w))
