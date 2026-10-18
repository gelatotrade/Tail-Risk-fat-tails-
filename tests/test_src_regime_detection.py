"""Regression tests for src/models/regime_detection.py (exact HMM recursions, simulated regimes)."""

import itertools
import time

import numpy as np
import pytest
from scipy import stats

from src.models.regime_detection import (
    MarkovRegimeSwitching,
    RegimeAwareTailRisk,
    VolatilityClusteringDetector,
    compute_regime_coordinates,
)


def _simulate_hmm(n, P, means, stds, seed):
    rng = np.random.default_rng(seed)
    k = len(stds)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(k, p=P[states[t - 1]])
    return np.asarray(means)[states] + np.asarray(stds)[states] * rng.standard_normal(n), states


P2 = np.array([[0.98, 0.02], [0.05, 0.95]])
STDS2 = [0.005, 0.02]


@pytest.fixture(scope="module")
def two_state_path():
    return _simulate_hmm(1500, P2, [0.0005, -0.001], STDS2, seed=11)


@pytest.fixture(scope="module")
def two_state_returns(two_state_path):
    return two_state_path[0]


# --------------------------------------------------------------------------- exact E-step


def _brute_force(y, P, means, stds):
    """Enumerate all state paths (uniform initial distribution)."""
    n, k = len(y), len(stds)
    lik = stats.norm.pdf(y[:, None], means, stds)
    total = 0.0
    smoothed = np.zeros((n, k))
    pair = np.zeros((k, k))
    filtered_last = np.zeros(k)
    filtered_first = np.zeros(k)
    for path in itertools.product(range(k), repeat=n):
        prob = (1 / k) * np.prod([lik[t, s] for t, s in enumerate(path)])
        prob *= np.prod([P[path[t - 1], path[t]] for t in range(1, n)])
        total += prob
        for t, s in enumerate(path):
            smoothed[t, s] += prob
        for t in range(1, n):
            pair[path[t - 1], path[t]] += prob
        filtered_last[path[-1]] += prob
    filtered_first = lik[0] / lik[0].sum()
    return np.log(total), smoothed / total, pair / total, filtered_first, filtered_last / total


def test_filter_and_smoother_match_brute_force_enumeration():
    y = np.array([0.001, -0.02, 0.004, 0.03, -0.001, 0.0005, -0.012])
    model = MarkovRegimeSwitching(3)
    model.state_means = np.array([0.0005, 0.0, -0.002])
    model.state_stds = np.array([0.004, 0.01, 0.025])
    model.transition_matrix = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]])

    ll, smoothed, pairs, first, last = _brute_force(y, model.transition_matrix, model.state_means, model.state_stds)

    filtered = model._filter(y)
    np.testing.assert_allclose(filtered[0], first, rtol=1e-10)  # y_0 must update the prior
    np.testing.assert_allclose(filtered[-1], last, rtol=1e-10)
    np.testing.assert_allclose(model._smooth(y), smoothed, rtol=1e-10, atol=1e-14)
    assert model._log_likelihood(y) == pytest.approx(ll, rel=1e-12)

    f, p, _ = model._forward(y)
    _, transitions = model._backward(f, p)
    np.testing.assert_allclose(transitions, pairs, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------- EM


def test_em_never_decreases_the_likelihood(two_state_returns):
    y = two_state_returns[:600]
    model = MarkovRegimeSwitching(2)
    model.fit(y, max_iter=0)  # initial parameters only
    lls = [model._log_likelihood(y)]
    for _ in range(15):
        model._update_parameters(y)
        lls.append(model._log_likelihood(y))
    assert np.all(np.diff(lls) > -1e-8), np.diff(lls)
    assert lls[-1] > lls[0] + 1


def test_fit_recovers_simulated_two_state_regimes(two_state_path):
    y, states = two_state_path
    res = MarkovRegimeSwitching(2).fit(y)

    # reference: what the simulated path actually realized
    counts = np.zeros((2, 2))
    np.add.at(counts, (states[:-1], states[1:]), 1)
    realized_P = counts / counts.sum(axis=1, keepdims=True)
    realized_stds = [y[states == k].std() for k in (0, 1)]

    np.testing.assert_allclose(res["state_stds"], realized_stds, rtol=0.05)
    np.testing.assert_allclose(res["transition_matrix"], realized_P, atol=0.01)
    np.testing.assert_allclose(res["transition_matrix"].sum(axis=1), 1.0, atol=1e-12)
    assert res["n_iter"] < 100


def test_states_are_ordered_by_volatility():
    y, _ = _simulate_hmm(1200, np.array([[0.97, 0.02, 0.01], [0.03, 0.95, 0.02], [0.02, 0.08, 0.9]]),
                         [0.0, 0.0, -0.002], [0.004, 0.01, 0.03], seed=5)
    model = MarkovRegimeSwitching(3)
    res = model.fit(y, max_iter=60)
    assert np.all(np.diff(res["state_stds"]) > 0)
    # filtered probabilities follow the relabelling: high-vol days load on state 2
    big = np.abs(y) > 0.06
    assert model.filtered_probs[big, 2].mean() > 0.9


def test_regime_aware_reports_the_current_high_vol_regime():
    rng = np.random.default_rng(8)
    y = np.concatenate([rng.normal(0.0005, 0.006, 700), rng.normal(-0.001, 0.03, 80)])
    start = time.perf_counter()
    analyzer = RegimeAwareTailRisk(y)
    elapsed = time.perf_counter() - start

    info = analyzer.current_regime()
    assert info["markov_regime"] == "HIGH_VOL"  # P collapsed to identity before the fix
    assert info["simple_regime"] == "HIGH_VOL"
    assert info["regime_probabilities"]["high_vol"] > 0.9
    assert elapsed < 6  # ~9 s for 500 observations before vectorizing


# --------------------------------------------------------------------------- simple regimes


def test_vol_regime_warmup_takes_the_first_full_window_label():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 0.05, 40), rng.normal(0, 0.005, 500)])
    regimes = VolatilityClusteringDetector(y).detect_vol_regime()
    assert regimes[20] == 2
    assert np.all(regimes[:20] == 2)  # were always 0 (LOW_VOL) because vol is NaN there
    assert np.all(VolatilityClusteringDetector(y[:10]).detect_vol_regime() == 0)


# --------------------------------------------------------------------------- regime coordinates


def _coordinates_reference(returns, window):
    """The original O(n^2) loop (raw values, before normalization)."""
    n = len(returns)
    X, Y, Z = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(2 * window, n):
        X[i] = np.std(returns[i - window:i]) / np.std(returns[:i])
        vol_series = [np.std(returns[j - window // 2:j]) for j in range(i - window, i, window // 4)]
        if len(vol_series) > 2:
            Y[i] = np.abs(np.corrcoef(vol_series[:-1], vol_series[1:])[0, 1])
        vols = [np.std(returns[j - window // 4:j]) for j in range(i - window, i, window // 8)]
        Z[i] = np.mean(np.abs(np.diff(vols))) / (np.mean(vols) + 1e-10)
    return X, Y, Z


@pytest.mark.parametrize("window", [8, 13, 30])
def test_regime_coordinates_match_reference_and_normalize_estimated_part(window):
    r = np.random.default_rng(window).standard_t(4, 500) * 0.01
    X, Y, Z = compute_regime_coordinates(r, window)
    Xr, Yr, Zr = _coordinates_reference(r, window)
    v = slice(2 * window, None)

    def min_max(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-10)

    np.testing.assert_allclose(X[v], min_max(Xr[v]), atol=1e-9)
    np.testing.assert_allclose(Y[v], np.clip(Yr[v], 0, 1), atol=1e-9)
    np.testing.assert_allclose(Z[v], min_max(Zr[v]), atol=1e-9)
    assert X[v].min() == pytest.approx(0, abs=1e-12) and X[v].max() == pytest.approx(1, abs=1e-6)
    assert not X[:2 * window].any() and not Z[:2 * window].any()


def test_regime_coordinates_small_window_raises_clear_error():
    with pytest.raises(ValueError, match="window must be >= 8"):
        compute_regime_coordinates(np.zeros(200), window=6)
    X, Y, Z = compute_regime_coordinates(np.random.default_rng(1).normal(0, 0.01, 40), window=30)
    assert not X.any() and not Y.any() and not Z.any()
