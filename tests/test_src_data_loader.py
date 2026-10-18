"""Regression tests for src/utils/data_loader.py (crisis length, GARCH edge case)."""

import numpy as np
import pytest
from scipy import stats

from src.utils.data_loader import (
    generate_crisis_scenario,
    generate_synthetic_returns,
    generate_vix_from_returns,
)


@pytest.mark.parametrize("n_crisis", [0, 1, 5, 30, 100, 250])
def test_crisis_length_follows_n_crisis(n_crisis):
    # n_crisis used to change only the metadata: the data always had 100 crisis days
    r, meta = generate_crisis_scenario(n_pre=100, n_crisis=n_crisis, n_post=80, seed=1)
    assert len(r) == 100 + n_crisis + 80 == meta["total_length"]
    assert meta["crisis_start"] == 100
    assert meta["crisis_end"] - meta["crisis_start"] == n_crisis
    assert meta["crisis_start"] <= meta["crisis_peak"] <= meta["crisis_end"]


def test_crisis_window_is_where_the_crisis_volatility_is():
    r, meta = generate_crisis_scenario(n_pre=300, n_crisis=40, n_post=300, seed=2)
    crisis = r[meta["crisis_start"]:meta["crisis_end"]]
    pre = r[:meta["crisis_start"]]
    assert np.std(crisis) > 3 * np.std(pre)


def test_default_crisis_scenario_is_unchanged():
    """The default 100-day crisis keeps the original 10/40/50-day phases and draws."""
    seed = 123
    r, meta = generate_crisis_scenario(seed=seed)
    np.random.seed(seed)
    pre = np.random.normal(0.0005, 0.006, 500)
    shock = np.random.normal(-0.02, 0.03, 10)
    main = stats.t.rvs(df=3, loc=-0.005, scale=0.04, size=40)
    late = np.random.normal(0.001, 0.025, 50)
    post = np.array([np.random.normal(0.0003, v) for v in np.linspace(0.02, 0.01, 400)])
    np.testing.assert_array_equal(r, np.concatenate([pre, shock, main, late, post]))
    assert meta == {"crisis_start": 500, "crisis_end": 600, "crisis_peak": 525, "total_length": 1000}


def test_garch_with_zero_length():
    # used to raise IndexError (sigma2[0] on an empty array)
    assert generate_synthetic_returns(0, "garch", seed=1).shape == (0,)
    assert generate_synthetic_returns(5, "garch", seed=1).shape == (5,)


def test_vix_from_returns_bounds():
    r, _ = generate_crisis_scenario(seed=1)
    vix = generate_vix_from_returns(r)
    assert vix.shape == r.shape
    assert vix.min() >= 10 and vix.max() <= 80
