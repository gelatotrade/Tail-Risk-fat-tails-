"""Regression tests for src/physics/ornstein_uhlenbeck.py (closed forms, no plotting)."""

import warnings

import numpy as np
import pytest
from scipy import optimize, stats

from src.physics.ornstein_uhlenbeck import (
    JumpDiffusionRiskAnalyzer,
    OrnsteinUhlenbeckProcess,
    OUWithJumps,
    VolatilityOUModel,
    compute_ou_coordinates,
)
from src.utils.data_loader import generate_crisis_scenario, generate_synthetic_returns

DT = 1 / 252


def _ols_slope(x):
    return np.polyfit(x[:-1], x[1:], 1)[0]


def _white_noise_with_negative_slope(n=500):
    """i.i.d. normal returns whose AR(1) slope happens to be negative."""
    for seed in range(100):
        r = np.random.default_rng(seed).normal(0.0003, 0.01, n)
        if _ols_slope(r) < 0:
            return r
    raise AssertionError("no seed with negative slope")


def _one_step(model, dt=DT):
    """Mean and variance of X_dt | X_0 = 0 (exact OU transition)."""
    b = np.exp(-model.theta * dt)
    return model.mu * (1 - b), model.variance_at_time(dt)


# --------------------------------------------------------------------------- OU fit


def test_fit_recovers_parameters_of_simulated_path():
    ou = OrnsteinUhlenbeckProcess(theta=5.0, mu=0.2, sigma=0.3)
    path = ou.simulate(0.2, T=40, n_steps=40 * 252, seed=1)[0]
    fit = OrnsteinUhlenbeckProcess.fit(path, dt=DT)
    assert fit.theta == pytest.approx(5.0, rel=0.25)
    assert fit.mu == pytest.approx(0.2, abs=0.03)
    assert fit.sigma == pytest.approx(0.3, rel=0.05)


def test_fit_negative_slope_raises_instead_of_returning_nan():
    r = _white_noise_with_negative_slope()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no 'invalid value encountered in log'
        with pytest.raises(ValueError, match="b ="):
            OrnsteinUhlenbeckProcess.fit(r)


def test_fit_unit_root_and_constant_data_raise():
    explosive = 1.01 ** np.arange(300) + np.random.default_rng(0).normal(0, 1e-3, 300)
    assert _ols_slope(explosive) > 1
    with pytest.raises(ValueError, match="no mean reversion"):
        OrnsteinUhlenbeckProcess.fit(explosive)
    with pytest.raises(ValueError):
        OrnsteinUhlenbeckProcess.fit(np.full(50, 0.3))


def test_fit_max_theta_gives_white_noise_limit():
    r = _white_noise_with_negative_slope()
    cap = np.log(1000) / DT
    m = OrnsteinUhlenbeckProcess.fit(r, dt=DT, max_theta=cap)
    assert m.theta == pytest.approx(cap)
    # b -> 0: the stationary law is the marginal law of the data
    assert m.stationary_variance() == pytest.approx(np.var(r[1:]), rel=0.01)
    assert m.mu == pytest.approx(np.mean(r), abs=1e-4)


def test_constructor_rejects_nan_and_inf():
    for theta, mu, sigma in [(np.nan, 0, 0.1), (np.inf, 0, 0.1), (1, 0, np.nan), (1, np.nan, 0.1)]:
        with pytest.raises(ValueError):
            OrnsteinUhlenbeckProcess(theta, mu, sigma)


def test_ou_coordinates_are_finite():
    r = generate_synthetic_returns(1500, "levy_jump", seed=1)
    # record (not raise) warnings: an exception inside would hit the fallback branch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X, Y, Z = compute_ou_coordinates(r)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    for c in (X, Y, Z):
        assert np.all(np.isfinite(c))
        assert c.min() >= 0 and c.max() <= 1 + 1e-12
    assert np.ptp(Z[60:]) > 0.5


# --------------------------------------------------------------------------- OU with jumps


def test_jump_records_are_aligned_with_paths():
    theta, n_steps, n_paths = 2.0, 50, 3
    proc = OUWithJumps(theta=theta, mu=0.0, sigma=1e-12, jump_intensity=200, jump_std=0.5)
    res = proc.simulate(0.0, 1.0, n_steps, n_paths, seed=0)
    times, sizes, paths_idx = res["jump_times"], res["jump_sizes"], res["jump_paths"]
    assert len(times) == len(sizes) == len(paths_idx) > 0
    # Rebuild the (noise-free) paths from the jump records alone
    dt = 1.0 / n_steps
    steps = np.rint(times / dt).astype(int)
    rebuilt = np.zeros((n_paths, n_steps + 1))
    for i in range(1, n_steps + 1):
        rebuilt[:, i] = rebuilt[:, i - 1] * np.exp(-theta * dt)
        for p in range(n_paths):
            rebuilt[p, i] += sizes[(steps == i) & (paths_idx == p)].sum()
    np.testing.assert_allclose(res["paths"], rebuilt, atol=1e-9)


def test_var_es_seeded_and_match_closed_form_without_jumps():
    proc = OUWithJumps(theta=5.0, mu=0.01, sigma=0.2, jump_intensity=0.0)
    var1 = proc.compute_var(0.0, T=DT, confidence=0.99, n_simulations=40000, seed=3)
    var2 = proc.compute_var(0.0, T=DT, confidence=0.99, n_simulations=40000, seed=3)
    es = proc.compute_expected_shortfall(0.0, T=DT, confidence=0.99, n_simulations=40000, seed=3)
    assert var1 == var2
    m, v = _one_step(proc.base_ou)
    z = stats.norm.ppf(0.01)
    assert var1 == pytest.approx(-(m + z * np.sqrt(v)), rel=0.03)
    assert es == pytest.approx(-(m - np.sqrt(v) * stats.norm.pdf(z) / 0.01), rel=0.03)
    assert es > var1


def test_volatility_model_seed_is_reproducible():
    vm = VolatilityOUModel()
    a = vm.simulate_price(100, 0.2, 1.0, 252, seed=5)
    b = vm.simulate_price(100, 0.2, 1.0, 252, seed=5)
    np.testing.assert_array_equal(a["prices"], b["prices"])
    np.testing.assert_array_equal(vm.simulate_volatility(0.2, 1.0, 100, seed=2),
                                  vm.simulate_volatility(0.2, 1.0, 100, seed=2))


# --------------------------------------------------------------------------- jump-diffusion analyzer


def test_decomposition_finite_for_white_noise_returns():
    r = _white_noise_with_negative_slope(1000)
    res = JumpDiffusionRiskAnalyzer().tail_risk_decomposition(r, seed=0)
    assert res["ou_fit_status"] == "theta_capped"
    for key in ("theta", "sigma", "var_total", "var_continuous", "var_jump", "jump_contribution_pct"):
        assert np.isfinite(res[key]), key
    cont = r[np.abs(r - r.mean()) <= 3 * r.std()]
    assert res["var_continuous"] == pytest.approx(-(cont.mean() + stats.norm.ppf(0.01) * cont.std()), rel=0.05)


def test_decomposition_matches_poisson_normal_mixture():
    r, _ = generate_crisis_scenario(seed=123)
    ja = JumpDiffusionRiskAnalyzer()
    res = ja.tail_risk_decomposition(r, confidence=0.99, n_simulations=100000, seed=0)

    m, v = _one_step(ja.ou_model)
    jumps = r[np.abs(r - r.mean()) > 3 * r.std()]
    lam_dt = len(jumps) / len(r)  # expected jumps per day
    jm, js = jumps.mean(), jumps.std()
    k = np.arange(10)
    w = stats.poisson.pmf(k, lam_dt)

    def cdf(x):
        return np.sum(w * stats.norm.cdf((x - m - k * jm) / np.sqrt(v + k * js**2))) - 0.01

    var_total = -optimize.brentq(cdf, -1, 1)
    var_cont = -(m + stats.norm.ppf(0.01) * np.sqrt(v))
    assert res["var_continuous"] == pytest.approx(var_cont, rel=0.02)
    assert res["var_total"] == pytest.approx(var_total, rel=0.06)
    assert res["var_jump"] == pytest.approx(res["var_total"] - res["var_continuous"])
    assert res["var_jump"] > 0.1 * res["var_continuous"]  # jumps actually simulated
    assert res["jump_mean"] == pytest.approx(jm)


def test_decomposition_threshold_and_common_random_numbers():
    r, _ = generate_crisis_scenario(seed=123)
    res = JumpDiffusionRiskAnalyzer().tail_risk_decomposition(r, threshold=50.0, seed=1)
    assert res["n_jumps"] == 0
    assert res["var_jump"] == 0.0  # same diffusion draws in both runs


def test_fit_does_not_keep_stale_model():
    r = generate_synthetic_returns(1000, "levy_jump", seed=2)
    ja = JumpDiffusionRiskAnalyzer()
    first = ja.fit(r)
    assert first["theta"] > 0
    second = ja.fit(r[:8])
    assert ja.ou_model is None
    assert second["theta"] == 0 and second["ou_fit_status"] == "too_few_points"
