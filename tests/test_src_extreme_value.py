"""Regression tests for src/models/extreme_value.py (iid Student-t data with known quantiles)."""

import numpy as np
import pytest
from scipy import optimize, stats

from src.models.extreme_value import (
    EVTTailRiskAnalyzer,
    GeneralizedExtremeValue,
    GeneralizedParetoDistribution,
)

NU, SCALE = 4, 0.01  # daily-return scale


def _t_returns(n=8000, seed=1):
    return np.random.default_rng(seed).standard_t(NU, n) * SCALE


def _true_quantile(p):
    return stats.t.ppf(p, NU) * SCALE


@pytest.fixture(scope="module")
def evt():
    return EVTTailRiskAnalyzer(_t_returns())


# --------------------------------------------------------------------------- GEV block maxima


def _block_maxima(x, block=20):
    return x[: len(x) // block * block].reshape(-1, block).max(axis=1)


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_gev_fit_reaches_the_likelihood_maximum_on_daily_scale(seed):
    x = -_t_returns(2520, seed=seed)  # 126 monthly maxima of daily losses
    gev = GeneralizedExtremeValue.fit_block_maxima(x)
    m = _block_maxima(x)

    def nll(p):
        return -stats.genextreme.logpdf(m, p[0], p[1], p[2]).sum() if p[2] > 0 else 1e10

    start = [-gev.xi, gev.mu, gev.sigma]
    polished = optimize.minimize(nll, start, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000})
    assert nll(start) - polished.fun < 1e-3  # raw fit stopped 26-47 nats short (seeds 2, 5)
    assert -0.1 < gev.xi < 0.45  # t(4): xi = 1/4 in theory; the raw fit gave 0.75-0.83


@pytest.mark.parametrize("seed", [2, 5])
def test_gev_fit_is_scale_equivariant(seed):
    x = -_t_returns(2520, seed=seed)
    small, big = GeneralizedExtremeValue.fit_block_maxima(x), GeneralizedExtremeValue.fit_block_maxima(100 * x)
    assert small.xi == pytest.approx(big.xi, abs=1e-4)
    assert 100 * small.mu == pytest.approx(big.mu, rel=1e-3)
    assert 100 * small.sigma == pytest.approx(big.sigma, rel=1e-3)


# --------------------------------------------------------------------------- GPD tail probability


def test_gpd_tail_probability_conditional_and_unconditional():
    gpd = GeneralizedParetoDistribution(xi=0.25, sigma=0.006, threshold=0.015)
    x, p_u = 0.04, 0.05
    conditional = stats.genpareto.sf(x - 0.015, 0.25, scale=0.006)
    assert gpd.tail_probability(x) == pytest.approx(conditional, rel=1e-12)
    assert gpd.tail_probability(x, p_u) == pytest.approx(p_u * conditional, rel=1e-12)
    assert gpd.tail_probability(0.01) == 1.0
    assert np.isnan(gpd.tail_probability(0.01, p_u))


def test_tail_comparison_gpd_probability_is_unconditional(evt):
    for k in (3, 4):
        res = evt.tail_comparison_to_gaussian(k)
        x = k * np.std(evt.data)
        truth = 2 * stats.t.sf(x / SCALE, NU)
        # summing two conditional tails overstated this ~20x
        assert res["gpd_probability"] == pytest.approx(truth, rel=0.3)
        assert res["gpd_probability"] == pytest.approx(res["empirical_probability"], rel=0.5)


def test_tail_comparison_below_threshold_uses_empirical_tail(evt):
    res = evt.tail_comparison_to_gaussian(0.5)  # 0.5 sigma is below both 95 % thresholds
    assert res["gpd_probability"] == pytest.approx(res["empirical_probability"], rel=1e-12)


# --------------------------------------------------------------------------- VaR and return levels


def test_gev_var_is_a_positive_daily_loss_quantile(evt):
    true_var = -_true_quantile(0.01)
    assert evt.var(0.99, "gev") == pytest.approx(true_var, rel=0.1)  # was -0.049 (a 100-block level, wrong sign)
    assert evt.var(0.99, "gpd") == pytest.approx(true_var, rel=0.1)
    assert evt.var(0.995, "gev") > evt.var(0.99, "gev") > 0


def test_var_unknown_method_raises(evt):
    with pytest.raises(ValueError, match="Unknown method"):
        evt.var(0.99, "historical")


def test_return_level_period_is_in_trading_days(evt):
    level = _true_quantile(1 - 1 / 2520)  # once in 10 years of trading days
    assert -evt.return_level(2520) == pytest.approx(level, rel=0.25)
    assert evt.return_level(2520, tail="right") == pytest.approx(level, rel=0.25)
    # consistent with the GEV VaR and monotone in the period
    assert evt.return_level(500) == pytest.approx(-evt.var(1 - 1 / 500, "gev"), rel=1e-12)
    assert evt.return_level(252 * 100) < evt.return_level(2520) < 0

    report = evt.comprehensive_report()
    assert report["return_levels"]["10_year"] == pytest.approx(evt.return_level(252 * 10))
    with pytest.raises(ValueError):
        evt.return_level(1)


def test_block_size_is_configurable():
    x = _t_returns(4000, seed=5)
    evt5 = EVTTailRiskAnalyzer(x, block_size=5)
    assert evt5.block_size == 5
    assert evt5.var(0.99, "gev") == pytest.approx(-_true_quantile(0.01), rel=0.15)
