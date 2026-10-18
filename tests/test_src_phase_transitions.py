"""Regression tests for src/physics/phase_transitions.py."""

import numpy as np
import pytest
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from src.physics import phase_transitions as pt
from src.physics.phase_transitions import (
    CriticalSlowingDownDetector,
    IsingMarketModel,
    MarketPhaseClassifier,
    PowerLawAnalyzer,
)
from src.utils.data_loader import generate_crisis_scenario, generate_synthetic_returns

# --------------------------------------------------------------------------- Ising


def test_energy_change_of_a_flip_equals_metropolis_delta_e():
    """energy() and monte_carlo_step() must describe the same periodic ring."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 30):
        model = IsingMarketModel(n_agents=n, J=1.3, h=0.4, T=2.0)
        model.spins = rng.choice([-1, 1], size=n)
        for i in range(n):
            s = model.spins
            d_e = 2 * s[i] * (model.J * (s[(i - 1) % n] + s[(i + 1) % n]) + model.h)
            e0 = model.energy()
            model.spins[i] *= -1
            assert model.energy() - e0 == pytest.approx(d_e), (n, i)


def test_energy_of_ordered_ring():
    model = IsingMarketModel(n_agents=10, J=1.0, h=0.5)
    model.spins = np.ones(10, dtype=int)
    assert model.energy() == pytest.approx(-10 * 1.0 - 0.5 * 10)  # N bonds on a ring


def test_phase_diagram_has_no_side_effects():
    np.random.seed(0)
    model = IsingMarketModel(n_agents=20, J=1.0, T=1.7)
    spins = model.spins.copy()
    out = model.phase_diagram([0.5, 1.0], n_samples=5)
    assert model.T == 1.7
    np.testing.assert_array_equal(model.spins, spins)
    assert isinstance(out["temperature"], np.ndarray) and out["temperature"].dtype == float
    assert out["magnetization"].shape == (2,)


# --------------------------------------------------------------------------- EWS


def _reference_ews(data, window, sigma):
    """The original per-window loops (np.corrcoef / np.var / stats.skew)."""
    n = len(data)
    ac, var, skew = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(window, n):
        w = data[i - window:i]
        d = w - gaussian_filter1d(w, sigma)
        if np.std(d) > 0:
            ac[i] = np.corrcoef(d[:-1], d[1:])[0, 1]
        var[i] = np.var(d)
        skew[i] = stats.skew(w)
    return ac, var, skew


@pytest.mark.parametrize("window,sigma", [(50, 100), (20, 3), (10, 0.5)])
def test_vectorised_ews_equals_reference_loops(window, sigma, monkeypatch):
    monkeypatch.setattr(pt, "_EWS_BLOCK_CELLS", 7 * window)  # force several blocks
    x = np.cumsum(generate_synthetic_returns(400, "garch", seed=3))
    det = CriticalSlowingDownDetector(window, sigma)
    ac, var, skew = _reference_ews(x, window, sigma)
    np.testing.assert_allclose(det.autocorrelation_lag1(x), ac, rtol=0, atol=1e-12)
    np.testing.assert_allclose(det.rolling_variance(x), var, rtol=1e-12, atol=0)
    np.testing.assert_allclose(det.rolling_skewness(x), skew, rtol=0, atol=1e-12)


def test_ews_short_series_is_all_padding():
    det = CriticalSlowingDownDetector(window=50)
    for n in (5, 50):
        x = np.random.default_rng(n).normal(0, 0.01, n)
        for f in (det.autocorrelation_lag1, det.rolling_variance, det.rolling_skewness):
            np.testing.assert_array_equal(f(x), np.zeros(n))


def test_composite_is_zscore_of_valid_windows_only():
    r, _ = generate_crisis_scenario(seed=123)
    w = 50
    ews = CriticalSlowingDownDetector(window=w).compute_ews(r[:400])
    ac, var = ews["autocorrelation"][w:], ews["variance"][w:]
    expected = 0.5 * (ac - ac.mean()) / (ac.std() + 1e-10) + 0.5 * (var - var.mean()) / (var.std() + 1e-10)
    np.testing.assert_array_equal(ews["composite"][:w], 0.0)
    np.testing.assert_allclose(ews["composite"][w:], expected, atol=1e-12)
    assert abs(ews["composite"][w:].mean()) < 1e-9


def test_classifier_runs_on_crisis_scenario():
    r, _ = generate_crisis_scenario(seed=123)
    out = MarketPhaseClassifier().classify(r)
    assert out["phase"] in {"STABLE", "TRENDING", "CRITICAL", "CRASH", "RECOVERY"}
    assert 0 <= out["volatility_percentile"] <= 100


# --------------------------------------------------------------------------- power law


def _pareto(n, alpha, x_min, seed):
    u = np.random.default_rng(seed).uniform(0, 1, n)
    return x_min * (1 - u) ** (-1 / (alpha - 1))  # density ∝ x^-alpha


def test_clauset_finds_x_min_above_the_median():
    """Uniform body (80 %) + Pareto tail (20 %, alpha = 3, x_min = 1)."""
    body = np.random.default_rng(1).uniform(0, 1, 2000)
    x = np.r_[body, _pareto(500, 3.0, 1.0, seed=2)]
    res = PowerLawAnalyzer().fit_power_law(x, seed=0)
    assert res["x_min"] > np.median(x)
    assert 0.9 < res["x_min"] < 1.6
    assert res["alpha"] == pytest.approx(3.0, abs=0.3)
    assert res["n_tail"] < 600


def test_clauset_x_min_is_global_ks_minimum():
    x = np.abs(stats.t.rvs(3, size=600, random_state=4))
    res = PowerLawAnalyzer().fit_power_law(x, seed=0)
    data = np.sort(x)
    best = (np.inf, None, None)
    for xmin in np.unique(data[len(data) // 20:]):
        tail = data[data >= xmin]
        if len(tail) < 10:
            continue
        alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
        ks = np.max(np.abs(np.arange(1, len(tail) + 1) / len(tail) - (1 - (xmin / tail) ** (alpha - 1))))
        if ks < best[0]:
            best = (ks, xmin, alpha)
    assert res["ks_statistic"] == pytest.approx(best[0], rel=1e-9)
    assert res["x_min"] == best[1]
    assert res["alpha"] == pytest.approx(best[2], rel=1e-9)


def test_gaussian_data_is_not_reported_as_infinite_variance():
    g = np.random.default_rng(42).normal(0, 0.01, 2520)
    assert PowerLawAnalyzer().fit_power_law(g, seed=0)["alpha"] > 4


def test_fixed_method_is_closed_form_mle():
    x = _pareto(3000, 2.5, 0.01, seed=5)
    res = PowerLawAnalyzer().fit_power_law(x, x_min_method="fixed", seed=0)
    x_min = np.percentile(x, 90)
    tail = x[x >= x_min]
    assert res["alpha"] == pytest.approx(1 + len(tail) / np.sum(np.log(tail / x_min)))
    assert res["alpha"] == pytest.approx(2.5, abs=0.15)


def test_bootstrap_is_seeded_and_configurable():
    x = _pareto(1000, 3.0, 1.0, seed=6)
    a = PowerLawAnalyzer().fit_power_law(x, n_bootstrap=50, seed=7)
    b = PowerLawAnalyzer().fit_power_law(x, n_bootstrap=50, seed=7)
    assert a["p_value"] == b["p_value"]
    assert a["p_value"] * 50 == pytest.approx(round(a["p_value"] * 50))
    assert a["p_value"] > 0.05  # true power law is not rejected


def test_power_law_rejects_empty_input():
    for data in (np.zeros(100), np.array([]), np.r_[np.zeros(50), np.ones(5)]):
        with pytest.raises(ValueError):
            PowerLawAnalyzer().fit_power_law(data)
