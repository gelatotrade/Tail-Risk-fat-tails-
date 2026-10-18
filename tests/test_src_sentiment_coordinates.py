"""compute_sentiment_coordinates: the vectorised version equals the original per-day loop."""

import numpy as np
import pytest

from src.analysis.sentiment import compute_sentiment_coordinates
from src.utils.data_loader import generate_crisis_scenario, generate_synthetic_returns, generate_vix_from_returns


def _reference(vix_series, returns, window=20):
    """The per-day loop of the original implementation (with the NaN-safe correlation)."""
    n = len(returns)
    vix_series = np.asarray(vix_series, dtype=float)
    vix = vix_series if len(vix_series) == n else np.interp(
        np.arange(n), np.linspace(0, n - 1, len(vix_series)), vix_series)
    X, Y, Z = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(window, n):
        X[i] = vix[i]
        Y[i] = vix[i] - np.mean(vix[i - window:i])
        vix_changes = np.diff(vix[i - window:i + 1])
        ret_window = returns[i - window + 1:i + 1]
        if len(vix_changes) == len(ret_window) and len(vix_changes) > 2:
            with np.errstate(invalid='ignore', divide='ignore'):
                c = np.corrcoef(ret_window, vix_changes)[0, 1]
            Z[i] = c if np.isfinite(c) else 0.0

    def _normalize(a):
        out = np.zeros(n)
        valid = a[window:]
        if valid.size == 0 or not np.any(np.isfinite(valid)):
            return out
        lo, hi = np.nanmin(valid), np.nanmax(valid)
        out[window:] = (valid - lo) / (hi - lo + 1e-10)
        return out

    return _normalize(X), _normalize(Y), _normalize(Z)


def _cases():
    r = generate_synthetic_returns(800, "levy_jump", seed=3)
    yield "levy_jump", generate_vix_from_returns(r), r, 20
    rc, _ = generate_crisis_scenario(seed=1)  # VIX pinned at 80 for weeks: constant windows
    yield "crisis-clipped", generate_vix_from_returns(rc), rc, 20
    yield "short-window", generate_vix_from_returns(r), r, 3
    yield "window-2-no-correlation", generate_vix_from_returns(r), r, 2
    yield "resampled-vix", generate_vix_from_returns(r)[::2], r, 15
    yield "long-window", generate_vix_from_returns(r), r, 250


@pytest.mark.parametrize("name,vix,r,window", list(_cases()), ids=[c[0] for c in _cases()])
def test_vectorised_equals_loop(name, vix, r, window):
    got = compute_sentiment_coordinates(vix, r, window=window)
    want = _reference(vix, r, window=window)
    for g, w, axis in zip(got, want, "XYZ"):
        assert np.all(np.isfinite(g)), axis
        np.testing.assert_allclose(g, w, rtol=1e-9, atol=1e-9, err_msg=f"{name}: {axis}")


def test_series_not_longer_than_window_gives_zeros():
    r = np.random.default_rng(0).normal(0, 0.01, 15)
    X, Y, Z = compute_sentiment_coordinates(15 + np.cumsum(r), r, window=20)
    assert not X.any() and not Y.any() and not Z.any()


def test_invalid_window_is_rejected():
    r = np.zeros(50)
    with pytest.raises(ValueError):
        compute_sentiment_coordinates(np.full(50, 20.0), r, window=0)
