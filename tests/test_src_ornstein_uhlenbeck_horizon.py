"""Regression tests: OUWithJumps VaR / ES over multi-day horizons (src/physics/ornstein_uhlenbeck.py).

compute_var used one ``simulate`` step for the whole horizon, which adds every jump of the
horizon undecayed at its end.  Mean reversion also pulls jumps back, so for θT not small the
jump risk was overstated (θ = 100/yr over 21 days: 18 % instead of about 6 %).
"""

import numpy as np
import pytest

from src.physics.ornstein_uhlenbeck import OUWithJumps

DT = 1 / 252


def _model(theta, lam=10.0, sigma=0.2):
    return OUWithJumps(theta=theta, mu=0.0, sigma=sigma, jump_intensity=lam, jump_mean=-0.02, jump_std=0.05)


@pytest.mark.parametrize("theta", [5.0, 100.0])
def test_var_over_horizon_matches_fine_step_simulation(theta):
    m = _model(theta)
    T = 21 * DT
    var = m.compute_var(0.0, T, 0.99, 40000, seed=1)
    # the process itself, with small steps (jumps decay between steps)
    fine = m.simulate(0.0, T, n_steps=210, n_paths=6000, seed=2)["paths"][:, -1]
    assert var == pytest.approx(-np.percentile(fine, 1), rel=0.12)


def test_terminal_moments_follow_campbell_formula():
    """Jumps only: E = λm(1-e^{-θT})/θ, Var = λ(m²+s²)(1-e^{-2θT})/(2θ) plus the exact OU variance."""
    theta, lam, mean, std, T = 30.0, 40.0, -0.02, 0.05, 21 * DT
    m = OUWithJumps(theta=theta, mu=0.0, sigma=0.1, jump_intensity=lam, jump_mean=mean, jump_std=std)
    x = m.sample_terminal(0.0, T, 400000, seed=3)
    exp_mean = lam * mean * (1 - np.exp(-theta * T)) / theta
    exp_var = lam * (mean ** 2 + std ** 2) * (1 - np.exp(-2 * theta * T)) / (2 * theta) + m.base_ou.variance_at_time(T)
    assert np.mean(x) == pytest.approx(exp_mean, abs=4 * np.sqrt(exp_var / len(x)))
    assert np.var(x) == pytest.approx(exp_var, rel=0.02)


def test_var_and_es_share_one_sample_and_common_random_numbers():
    jumps, plain = _model(5.0), _model(5.0, lam=0.0)
    T = 5 * DT
    assert jumps.compute_var(0.0, T, 0.99, 20000, seed=4) == jumps.compute_var(0.0, T, 0.99, 20000, seed=4)
    a = jumps.sample_terminal(0.0, T, 20000, seed=4)
    b = plain.sample_terminal(0.0, T, 20000, seed=4)
    differ = np.mean(a != b)  # only scenarios with at least one jump differ
    assert differ == pytest.approx(1 - np.exp(-10.0 * T), abs=0.01)
    es = jumps.compute_expected_shortfall(0.0, T, 0.99, 20000, seed=4)
    var = jumps.compute_var(0.0, T, 0.99, 20000, seed=4)
    assert es == pytest.approx(-np.mean(a[a < -var]))
    assert es > var > 0


def test_one_day_horizon_close_to_single_step_simulation():
    """For θ·dt small, decaying jumps within the day changes almost nothing."""
    m = _model(5.0)
    new = m.compute_var(0.0, DT, 0.99, 100000, seed=5)
    old = -np.percentile(m.simulate(0.0, DT, n_steps=1, n_paths=100000, seed=6)["paths"][:, -1], 1)
    assert new == pytest.approx(old, rel=0.05)
