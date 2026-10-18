"""compute_moments: NumPy moments equal scipy.stats.skew / kurtosis (fast enough for bootstraps)."""

import time
import warnings

import numpy as np
import pytest
from scipy import stats

from src.utils.statistics import compute_moments


def _cases():
    rng = np.random.default_rng(7)
    yield "student-t", stats.t.rvs(4, loc=0.0003, scale=0.01, size=2500, random_state=1)
    yield "lognormal", rng.lognormal(0, 1, 500)
    yield "shifted", 1e6 + rng.normal(0, 1, 300)          # large mean, small spread
    yield "two-points", np.array([0.0, 1.0])
    yield "integers", np.arange(10)
    yield "list", [0.01, -0.02, 0.005, 0.03, -0.04]


@pytest.mark.parametrize("name,x", list(_cases()), ids=[c[0] for c in _cases()])
def test_matches_scipy(name, x):
    m = compute_moments(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert m["skewness"] == pytest.approx(stats.skew(x), rel=1e-9, abs=1e-12)
        assert m["kurtosis"] == pytest.approx(stats.kurtosis(x), rel=1e-9, abs=1e-12)
    assert m["mean"] == pytest.approx(np.mean(x))
    assert m["variance"] == pytest.approx(np.var(x))
    assert m["std"] == pytest.approx(np.std(x))


@pytest.mark.parametrize("x", [np.full(50, 0.1), np.full(50, 123.456), np.zeros(10), np.full(7, -3.0)])
def test_constant_data_behaves_like_scipy(x):
    # exact constants give NaN; a mean with round-off (0.1) slips past scipy's guard too - same result
    m = compute_moments(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_s, want_k = stats.skew(x), stats.kurtosis(x)
    np.testing.assert_allclose([m["skewness"], m["kurtosis"]], [want_s, want_k], rtol=1e-9, equal_nan=True)
    if np.all(x == x[0]) and np.mean(x) == x[0]:
        assert np.isnan(m["skewness"]) and np.isnan(m["kurtosis"])


def test_nan_propagates():
    x = np.array([0.01, np.nan, -0.02, 0.03])
    m = compute_moments(x)
    assert np.isnan(m["mean"]) and np.isnan(m["skewness"]) and np.isnan(m["kurtosis"])


def test_returns_numpy_floats():
    m = compute_moments(np.random.default_rng(0).normal(size=100))
    assert all(isinstance(v, np.floating) for v in m.values())


def test_fast_enough_for_a_bootstrap():
    x = np.random.default_rng(1).standard_t(4, 2500)
    compute_moments(x)
    t0 = time.perf_counter()
    for _ in range(200):
        compute_moments(x)
    assert (time.perf_counter() - t0) / 200 < 1e-3  # SciPy's skew + kurtosis took ~3 ms per call
