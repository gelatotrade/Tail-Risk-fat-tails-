"""Regression tests for src/utils/statistics.py (Kolmogorov-Smirnov with fitted families)."""

import numpy as np
import pytest
from scipy import stats

from src.utils.statistics import compute_moments, jarque_bera_test, kolmogorov_smirnov_test


@pytest.fixture(scope="module")
def t_returns():
    return stats.t.rvs(4, loc=0.0003, scale=0.01, size=2000, random_state=1)


def test_ks_normal_path_unchanged(t_returns):
    x = t_returns
    ref = stats.kstest(x, stats.norm(np.mean(x), np.std(x)).cdf)
    res = kolmogorov_smirnov_test(x)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert not res["matches_distribution"]


def test_ks_student_t_does_not_crash_and_fits(t_returns):
    # used to raise TypeError: _parse_args() missing 1 required positional argument: 'df'
    res = kolmogorov_smirnov_test(t_returns, "t")
    params = stats.t.fit(t_returns)
    ref = stats.kstest(t_returns, stats.t(*params).cdf)
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-9)
    assert res["p_value"] > 0.05 and res["matches_distribution"]


def test_ks_laplace_is_fitted_not_standard():
    # Raw returns against a standard Laplace (scale 1) always rejected with D ~ 0.48
    x = stats.laplace.rvs(loc=0.0005, scale=0.008, size=1500, random_state=2)
    res = kolmogorov_smirnov_test(x, "laplace")
    assert res["statistic"] < 0.05
    assert res["matches_distribution"]


def test_ks_fixed_params_reproduce_plain_kstest(t_returns):
    res = kolmogorov_smirnov_test(t_returns, "cauchy", params=())
    ref = stats.kstest(t_returns, "cauchy")
    assert res["statistic"] == pytest.approx(ref.statistic, rel=1e-12)


def test_ks_unknown_distribution_raises():
    with pytest.raises(ValueError):
        kolmogorov_smirnov_test(np.arange(10.0), "no_such_distribution")


def test_moments_and_jarque_bera_match_scipy(t_returns):
    m = compute_moments(t_returns)
    assert m["kurtosis"] == pytest.approx(stats.kurtosis(t_returns))
    jb = jarque_bera_test(t_returns)
    assert jb["statistic"] == pytest.approx(stats.jarque_bera(t_returns).statistic)
