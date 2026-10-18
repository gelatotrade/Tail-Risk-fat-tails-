"""Regression tests for src/models/risk_metrics.py."""

import numpy as np
import pytest
from scipy import stats

from src.models.risk_metrics import RollingRiskMetrics, TailRiskMetrics, TailRiskParity, compute_risk_3d_coordinates


def _t_returns(n=5000, seed=0, nu=4, scale=0.01, loc=0.0003):
    return loc + np.random.default_rng(seed).standard_t(nu, n) * scale


# --------------------------------------------------------------------------- Expected Shortfall


def test_historical_es_is_mean_loss_beyond_var():
    r = _t_returns()
    m = TailRiskMetrics(r)
    for conf in (0.95, 0.99):
        var = m.var_historical(conf)
        es = m.expected_shortfall(conf)
        assert es == pytest.approx(np.mean(-r[r < -var]), rel=1e-12)
        assert es > var > 0  # was negative: the mean of the gains above +VaR


def test_historical_es_matches_student_t_closed_form():
    nu, scale, loc = 4, 0.01, 0.0003
    m = TailRiskMetrics(_t_returns(200_000, seed=1, nu=nu, scale=scale, loc=loc))
    z = stats.t.ppf(0.01, nu)
    true_es = scale * (nu + z**2) / (nu - 1) * stats.t.pdf(z, nu) / 0.01 - loc
    assert m.expected_shortfall(0.99) == pytest.approx(true_es, rel=0.03)


def test_cornish_fisher_es_uses_the_loss_tail():
    r = _t_returns(seed=2, nu=6)
    m = TailRiskMetrics(r)
    var_cf = m.var_cornish_fisher(0.99)
    es_cf = m.expected_shortfall(0.99, "cornish_fisher")
    assert es_cf == pytest.approx(np.mean(-r[r < -var_cf]), rel=1e-12)
    assert es_cf > var_cf > 0


def test_expected_shortfall_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        TailRiskMetrics(_t_returns(500)).expected_shortfall(0.99, "evt")


# --------------------------------------------------------------------------- Drawdown


def test_maximum_drawdown_of_log_returns_matches_price_path():
    r = _t_returns(3000, seed=3, scale=0.015)
    prices = 100 * np.exp(np.cumsum(r))
    true_dd = 1 - np.min(prices / np.maximum.accumulate(prices))

    dd = TailRiskMetrics(r).maximum_drawdown()  # default: log returns (framework convention)
    assert dd["max_drawdown"] == pytest.approx(true_dd, rel=1e-12)
    np.testing.assert_allclose(dd["drawdown_series"], prices / np.maximum.accumulate(prices) - 1, atol=1e-12)


def test_maximum_drawdown_of_simple_returns_on_request():
    simple = np.random.default_rng(4).normal(0.0003, 0.02, 2000)
    wealth = np.cumprod(1 + simple)
    true_dd = 1 - np.min(wealth / np.maximum.accumulate(wealth))
    assert TailRiskMetrics(simple, log_returns=False).maximum_drawdown()["max_drawdown"] == pytest.approx(true_dd, rel=1e-12)


# --------------------------------------------------------------------------- Rolling metrics


def _rolling_reference(r, window):
    n = len(r)
    out = {k: np.full(n, np.nan) for k in ("var", "es", "kurt", "skew")}
    for i in range(window, n):
        w = r[i - window:i]
        var = -np.percentile(w, 1)
        tail = -w[w < -var]
        out["var"][i] = var
        out["es"][i] = np.mean(tail) if len(tail) else var
        out["kurt"][i] = stats.kurtosis(w)
        out["skew"][i] = stats.skew(w)
    return out


@pytest.mark.parametrize("window", [20, 60])
def test_vectorized_rolling_metrics_match_window_loop(window):
    r = _t_returns(400, seed=5)
    rolling = RollingRiskMetrics(r, window)
    ref = _rolling_reference(r, window)
    np.testing.assert_allclose(rolling.rolling_var(0.99), ref["var"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rolling.rolling_expected_shortfall(0.99), ref["es"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rolling.rolling_kurtosis(), ref["kurt"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rolling.rolling_skewness(), ref["skew"], rtol=1e-9, atol=1e-12)
    assert np.isnan(rolling.rolling_var()[:window]).all()


def test_rolling_metrics_shorter_than_window_are_all_nan():
    rolling = RollingRiskMetrics(_t_returns(30), window=60)
    for series in (rolling.rolling_var(), rolling.rolling_expected_shortfall(), rolling.rolling_kurtosis()):
        assert series.shape == (30,) and np.isnan(series).all()
    X, Y, Z = compute_risk_3d_coordinates(_t_returns(300), window=60)
    assert X.shape == (300,) and np.nanmax(X) <= 1 + 1e-9


def test_rolling_tail_index_rejects_unknown_method():
    rolling = RollingRiskMetrics(_t_returns(200), window=60)
    with pytest.raises(ValueError, match="Unknown method"):
        rolling.rolling_tail_index("hil")  # was silently all-NaN
    hill = rolling.rolling_tail_index("hill")
    assert np.isfinite(hill[60:]).all() and np.isnan(hill[:60]).all()


# --------------------------------------------------------------------------- Tail risk parity


def _assets(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.standard_t(4, n)
    noise = rng.standard_t(4, (n, 4))
    return (0.6 * common[:, None] + 0.8 * noise) * np.array([0.004, 0.008, 0.015, 0.025])


def test_tail_risk_parity_equalizes_es_contributions():
    trp = TailRiskParity(_assets(), confidence=0.95)
    w = trp.tail_risk_parity_weights()
    rc = trp.risk_contribution(w)
    shares = rc / rc.sum()

    assert w.sum() == pytest.approx(1.0, abs=1e-8) and np.all(w >= 0.01 - 1e-9)
    assert np.all(np.diff(w) < 0)  # lower-risk assets get more weight
    assert np.abs(shares - 0.25).max() < 0.02  # equal weights: shares 0.06 .. 0.55


def test_tail_risk_parity_is_scale_invariant():
    R = _assets(seed=1)
    w_daily = TailRiskParity(R, confidence=0.95).tail_risk_parity_weights()
    w_percent = TailRiskParity(100 * R, confidence=0.95).tail_risk_parity_weights()
    np.testing.assert_allclose(w_daily, w_percent, atol=1e-6)
    assert np.ptp(w_daily) > 0.2  # it did move away from the equal-weight start
