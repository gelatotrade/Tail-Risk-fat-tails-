"""Regression tests for src/utils/numerical.py (scalar safety, EWMA bias, Hessian step)."""

import numpy as np
import pytest

from src.utils.numerical import (
    exponential_weighted_mean,
    numerical_gradient,
    numerical_hessian,
    rolling_window,
    safe_divide,
)


# ------------------------------------------------------------------ safe_divide
def test_safe_divide_accepts_scalars():
    # used to raise TypeError: 'numpy.float64' object does not support item assignment
    assert safe_divide(1.0, 0.0) == 0.0
    assert safe_divide(1.0, 0.0, fill=-1.0) == -1.0
    assert safe_divide(6.0, 3.0) == pytest.approx(2.0)
    assert np.ndim(safe_divide(6.0, 3.0)) == 0


def test_safe_divide_arrays_unchanged():
    out = safe_divide(np.array([1.0, 2.0, 0.0, 3]), np.array([0.0, 4.0, 0.0, 2]), fill=7.0)
    np.testing.assert_array_equal(out, [7.0, 0.5, 7.0, 1.5])
    np.testing.assert_array_equal(safe_divide(np.array([1, 2]), np.array([2, 4])), [0.5, 0.5])


# --------------------------------------------------------------- rolling_window
def test_rolling_window_values_and_read_only():
    x = np.arange(6.0)
    w = rolling_window(x, 3)
    np.testing.assert_array_equal(w, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    # A writeable as_strided view aliased the source and the neighbouring windows
    assert not w.flags.writeable
    with pytest.raises(ValueError):
        w[0, 1] = 99.0
    np.testing.assert_array_equal(x, np.arange(6.0))


def test_rolling_window_2d_and_too_long_window():
    x = np.arange(10.0).reshape(2, 5)
    w = rolling_window(x, 2)
    assert w.shape == (2, 4, 2)
    np.testing.assert_array_equal(w[1, 3], [8.0, 9.0])
    with pytest.raises(ValueError, match="window"):
        rolling_window(np.arange(3.0), 5)


# ---------------------------------------------------- exponential_weighted_mean
def _ewm_reference(x, alpha):
    """Direct evaluation of the definition (= pandas ewm(adjust=True))."""
    out = np.empty(len(x))
    for t in range(len(x)):
        w = (1 - alpha) ** np.arange(t, -1, -1)
        out[t] = np.sum(w * x[: t + 1]) / np.sum(w)
    return out


def test_ewm_of_constant_is_constant():
    # Weights normalised over the full length gave 0.06 at t=0 and 0.49 at t=10
    np.testing.assert_allclose(exponential_weighted_mean(np.ones(100)), 1.0, rtol=1e-12)


@pytest.mark.parametrize("alpha", [0.06, 0.3, 1.0])
def test_ewm_matches_definition(alpha):
    x = np.random.default_rng(3).normal(0.001, 0.01, 300)
    out = exponential_weighted_mean(x, alpha)
    np.testing.assert_allclose(out, _ewm_reference(x, alpha), rtol=1e-10, atol=1e-15)
    assert out[0] == pytest.approx(x[0])


def test_ewm_alpha_zero_is_expanding_mean():
    x = np.random.default_rng(4).normal(size=50)
    np.testing.assert_allclose(exponential_weighted_mean(x, 0.0), np.cumsum(x) / np.arange(1, 51), rtol=1e-12)


# ---------------------------------------------------------- numerical_hessian
def test_hessian_default_step_is_accurate_for_large_function_values():
    # |f| ~ 1e4 (log-likelihood scale): h = 1e-6 had an error of ~0.7 on f'' = 12
    f = lambda x: 1e4 + x ** 3
    assert numerical_hessian(f, 2.0) == pytest.approx(12.0, rel=1e-5)
    g = lambda x: -5000.0 * np.log(x) + 2e4
    assert numerical_hessian(g, 3.0) == pytest.approx(5000.0 / 9.0, rel=1e-5)


def test_hessian_default_step_scales_with_x():
    f = lambda x: np.exp(x / 1000.0)
    assert numerical_hessian(f, 5000.0) == pytest.approx(np.exp(5.0) / 1e6, rel=1e-4)


def test_hessian_and_gradient_explicit_step_still_honoured():
    f = lambda x: x ** 4
    h = 1e-3
    assert numerical_hessian(f, 1.0, h) == pytest.approx((f(1 + h) - 2 * f(1) + f(1 - h)) / h ** 2, rel=1e-12)
    assert numerical_gradient(np.sin, 0.3) == pytest.approx(np.cos(0.3), rel=1e-8)
