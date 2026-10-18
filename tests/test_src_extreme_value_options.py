"""EVTTailRiskAnalyzer(threshold_quantile=...) as used by the 'evt' web tool."""

import numpy as np
import pytest

from src.models.extreme_value import EVTTailRiskAnalyzer, GeneralizedParetoDistribution


def _t_returns(n=3000, seed=11):
    return 0.007 * np.random.default_rng(seed).standard_t(3.5, n)


def test_threshold_quantile_controls_both_gpd_fits():
    x = _t_returns()
    for q in (0.9, 0.95, 0.98):
        evt = EVTTailRiskAnalyzer(x, threshold_quantile=q)
        gpd_l, st_l = GeneralizedParetoDistribution.fit_pot(-x, quantile=q)
        gpd_r, st_r = GeneralizedParetoDistribution.fit_pot(x, quantile=q)
        assert evt.stats_left["threshold"] == pytest.approx(st_l["threshold"])
        assert evt.stats_right["threshold"] == pytest.approx(st_r["threshold"])
        assert evt.stats_left["exceedance_probability"] == pytest.approx(1 - q, abs=2 / len(x))
        assert evt.tail_index("left") == pytest.approx(gpd_l.xi)


def test_default_threshold_unchanged():
    x = _t_returns()
    assert EVTTailRiskAnalyzer(x).stats_left["threshold"] == pytest.approx(np.percentile(-x, 95))


def test_var_and_es_ordering_across_thresholds():
    x = _t_returns()
    for q in (0.9, 0.95, 0.97):
        evt = EVTTailRiskAnalyzer(x, threshold_quantile=q)
        v99, v999 = evt.var(0.99), evt.var(0.999)
        assert v999 > v99 > 0
        assert evt.expected_shortfall(0.99) >= v99


@pytest.mark.parametrize("q", [0.0, 1.0, 1.5, -0.1])
def test_invalid_threshold_quantile(q):
    with pytest.raises(ValueError):
        EVTTailRiskAnalyzer(_t_returns(500), threshold_quantile=q)
