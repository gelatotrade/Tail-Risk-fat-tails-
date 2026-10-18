"""Regression tests for src/analysis/sentiment.py."""

import warnings

import numpy as np
import pytest

from src.analysis.sentiment import (
    CompositeSentimentIndicator,
    MarketBreadthAnalyzer,
    VIXAnalyzer,
    compute_sentiment_coordinates,
)
from src.utils.data_loader import generate_crisis_scenario, generate_vix_from_returns


# ------------------------------------------------------ compute_sentiment_coordinates
def test_sentiment_coordinates_finite_with_pinned_vix():
    # Synthetic crisis VIX is clipped at 80 -> constant windows -> corrcoef NaN,
    # which used to turn ALL of Z into NaN
    r, _ = generate_crisis_scenario(seed=1)
    vix = generate_vix_from_returns(r)
    assert np.sum(vix == 80) > 20
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        X, Y, Z = compute_sentiment_coordinates(vix, r)
    for a in (X, Y, Z):
        assert a.shape == r.shape
        assert np.all(np.isfinite(a))
        assert a.min() >= 0 and a.max() <= 1 + 1e-9


def test_sentiment_coordinates_padding_does_not_set_the_minimum():
    # VIX never drops below 10: with padding zeros in the min/max, X never reached 0
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 300)
    vix = 15 + 10 * np.abs(np.sin(np.arange(300) / 17.0))
    window = 20
    X, Y, Z = compute_sentiment_coordinates(vix, r, window=window)
    lo = vix[window:].min()
    hi = vix[window:].max()
    np.testing.assert_allclose(X[window:], (vix[window:] - lo) / (hi - lo), atol=1e-8)
    for a in (X, Y, Z):
        assert a[window:].min() == pytest.approx(0.0, abs=1e-12)
        assert a[window:].max() == pytest.approx(1.0, abs=1e-8)
        np.testing.assert_array_equal(a[:window], 0.0)


def test_sentiment_coordinates_resampling_grid():
    # A shorter VIX series is resampled onto 0..n-1 (linspace(0, n, m) stretched it)
    vix_short = np.array([10.0, 20.0, 10.0, 20.0, 10.0])
    r = np.random.default_rng(1).normal(0, 0.01, 41)
    X, _, _ = compute_sentiment_coordinates(vix_short, r, window=3)
    assert X[10] == pytest.approx(1.0, abs=1e-8)
    assert X[30] == pytest.approx(1.0, abs=1e-8)
    assert X[20] == pytest.approx(0.0, abs=1e-8)
    assert X[40] == pytest.approx(0.0, abs=1e-8)


def test_sentiment_correlation_value():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 200)
    dv = np.where(np.arange(200) < 100, -300.0, 300.0) * r  # corr -1, then +1
    vix = 20 + np.cumsum(dv)
    _, _, Z = compute_sentiment_coordinates(vix, r, window=20)
    np.testing.assert_allclose(Z[20:100], 0.0, atol=1e-6)
    np.testing.assert_allclose(Z[120:], 1.0, atol=1e-6)


# ------------------------------------------------------ CompositeSentimentIndicator
def test_composite_vix_subscore_is_neutral_at_normal_vix():
    csi = CompositeSentimentIndicator()
    assert csi.compute_composite(vix=20)["composite"] == pytest.approx(0.0)
    # VIXAnalyzer regimes 15 / 25 / 35 land on the GREED / FEAR / EXTREME_FEAR boundaries
    assert csi.compute_composite(vix=15)["composite"] == pytest.approx(25.0)
    assert csi.compute_composite(vix=25)["composite"] == pytest.approx(-25.0)
    assert csi.compute_composite(vix=35)["composite"] == pytest.approx(-75.0)
    # VIX 45 (HIGH_FEAR) used to score as greed (+5)
    res = csi.compute_composite(vix=45)
    assert res["composite"] < -75 and res["level"] == "EXTREME_FEAR"
    assert csi.compute_composite(vix=12)["level"] == "GREED"


def test_composite_stays_within_documented_range():
    csi = CompositeSentimentIndicator()
    hi = csi.compute_composite(vix=-100, put_call=0.5, breadth=100, momentum=100)
    lo = csi.compute_composite(vix=200, put_call=2.0, breadth=-50, momentum=-100)
    assert hi["composite"] == pytest.approx(100.0) and hi["level"] == "EXTREME_GREED"
    assert lo["composite"] == pytest.approx(-100.0) and lo["level"] == "EXTREME_FEAR"
    for v in np.linspace(5, 90, 18):
        c = csi.compute_composite(vix=v, put_call=0.9, breadth=40, momentum=-2)["composite"]
        assert -100 <= c <= 100


def test_composite_weighted_average():
    csi = CompositeSentimentIndicator()
    res = csi.compute_composite(vix=18, breadth=70, momentum=3)
    comp = res["components"]
    assert comp == {"vix": pytest.approx(10.0), "breadth": pytest.approx(40.0), "momentum": pytest.approx(30.0)}
    expected = (0.3 * 10 + 0.25 * 40 + 0.2 * 30) / (0.3 + 0.25 + 0.2)
    assert res["composite"] == pytest.approx(expected)


# ------------------------------------------------------------- VIXAnalyzer spikes
def test_vix_spike_uses_std_of_daily_moves():
    rng = np.random.default_rng(0)
    # VIX trending from 20 to 60 (large level dispersion), then a 19-point jump
    vix = np.r_[np.linspace(20, 60, 19) + rng.normal(0, 1, 19), 78.97]
    res = VIXAnalyzer(vix).vix_spike_detection()
    sd = np.std(np.diff(vix[-20:-1]))
    assert res["z_score"] == pytest.approx((vix[-1] - vix[-2]) / (sd + 1e-10))
    assert res["spike_detected"]  # the level std (~12) gave z = 1.5: no spike


def test_vix_no_spike_on_ordinary_move():
    rng = np.random.default_rng(1)
    vix = 18 + np.cumsum(rng.normal(0, 0.5, 60))
    vix[-1] = vix[-2] + 0.3
    assert not VIXAnalyzer(vix).vix_spike_detection()["spike_detected"]


# ---------------------------------------------------------- MarketBreadthAnalyzer
def test_breadth_thrust_looks_back_ten_days():
    mb = MarketBreadthAnalyzer()
    for v in range(30):
        mb.add_data(pct_50ma=float(v))
    res = mb.breadth_thrust_signal()
    assert res["prior_breadth"] == 19.0  # 10 trading days before 29
    assert res["thrust_magnitude"] == 10.0
