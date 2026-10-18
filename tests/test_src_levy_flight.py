"""Regression tests for src/physics/levy_flight.py (NumPy 2.x / Pyodide and math fixes)."""

import numpy as np
import pytest
from scipy import stats

from src.physics.levy_flight import (
    LevyFlightProcess,
    LevyStableDistribution,
    estimate_tail_index,
    levy_flight_3d_coordinates,
)


# --------------------------------------------------------------------------- pdf
def test_pdf_runs_without_np_trapz_and_matches_scipy_s1():
    # np.trapz no longer exists in NumPy >= 2.4 (Pyodide ships 2.4.6)
    d = LevyStableDistribution(1.7, -0.1, 1.0, 0.0)
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    ref = stats.levy_stable.pdf(x, 1.7, -0.1)  # scipy default = S1
    np.testing.assert_allclose(d.pdf(x), ref, atol=3e-4)


def test_pdf_keeps_shape_and_scalar_input():
    d = LevyStableDistribution(1.5, 0.3, 2.0, 0.5)
    grid = np.linspace(-5, 5, 12).reshape(3, 4)
    out = d.pdf(grid)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out.ravel(), d.pdf(grid.ravel()), rtol=0, atol=1e-15)
    assert np.ndim(d.pdf(0.0)) == 0


def test_pdf_integrates_to_one_for_gaussian_limit():
    d = LevyStableDistribution(2.0, 0.0, 1.0, 0.0)  # = N(0, 2 gamma^2)
    x = np.linspace(-6, 6, 61)
    np.testing.assert_allclose(d.pdf(x), stats.norm.pdf(x, 0, np.sqrt(2)), atol=1e-6)


# ------------------------------------------------------------------------ sample
def test_sample_alpha_one_is_consistent_with_pdf():
    # S1, alpha = 1: gamma*Z + delta needs the location term 2/pi*beta*gamma*ln(gamma)
    d = LevyStableDistribution(1.0, 0.5, 10.0, 0.0)
    s = d.sample(200_000, seed=3)
    centres = np.array([-10.0, 0.0, 10.0, 20.0])
    half = 1.0
    emp = np.array([np.mean(np.abs(s - c) < half) / (2 * half) for c in centres])
    fine = np.linspace(-1, 1, 21)
    model = np.array([np.mean(d.pdf(c + fine, n_points=4000)) for c in centres])
    np.testing.assert_allclose(emp, model, rtol=0.05)


# -------------------------------------------------------------- tail_probability
def test_tail_probability_matches_scipy_asymptote():
    d = LevyStableDistribution(1.7, -0.1, 1.0, 0.0)
    for x in (50.0, 100.0):
        ratio = stats.levy_stable.sf(x, 1.7, -0.1) / d.tail_probability(x)
        assert ratio == pytest.approx(1.0, abs=0.03)  # was exactly 2x off


def test_tail_probability_left_both_delta_and_zero():
    d = LevyStableDistribution(1.5, 0.4, 2.0, 0.0)
    right, left = d.tail_probability(30.0, 'right'), d.tail_probability(-30.0, 'left')
    assert left / right == pytest.approx((1 - 0.4) / (1 + 0.4))
    assert d.tail_probability(30.0, 'both') == pytest.approx(right + left)
    # location shift: threshold measured from delta
    shifted = LevyStableDistribution(1.5, 0.4, 2.0, 5.0)
    assert shifted.tail_probability(35.0, 'right') == pytest.approx(right)
    # x = delta no longer divides by zero, probabilities are capped at 1
    assert d.tail_probability(0.0) == 1.0
    assert np.all(d.tail_probability(np.array([0.01, 0.1, 1.0])) <= 1.0)
    with pytest.raises(ValueError):
        d.tail_probability(3.0, 'upper')


# --------------------------------------------------------------------------- fit
def test_fit_gaussian_scale_constant():
    x = np.random.default_rng(0).normal(0.0, 0.01, 20_000)
    f = LevyStableDistribution.fit(x)
    assert f.alpha == pytest.approx(2.0, abs=0.03)
    # S1 at alpha = 2 is N(delta, 2 gamma^2) -> gamma = sigma / sqrt(2) (was IQR/2.63)
    assert f.gamma == pytest.approx(0.01 / np.sqrt(2), rel=0.03)


@pytest.mark.parametrize("alpha,beta", [(1.5, 0.5), (1.7, -0.1), (1.2, -0.6)])
def test_fit_recovers_stable_parameters(alpha, beta):
    true = LevyStableDistribution(alpha, beta, 0.01, 0.0005)
    f = LevyStableDistribution.fit(true.sample(50_000, seed=7))
    assert f.alpha == pytest.approx(alpha, abs=0.06)  # kurtosis proxy gave 1.1
    assert f.beta == pytest.approx(beta, abs=0.12)
    assert f.gamma == pytest.approx(0.01, rel=0.04)


def test_fit_matches_scipy_mcculloch_reference():
    fitstart = pytest.importorskip("scipy.stats._levy_stable")._fitstart_S1
    data = LevyStableDistribution(1.6, 0.3, 0.02, 0.001).sample(5_000, seed=11)
    f = LevyStableDistribution.fit(data)
    a, b, delta, c = fitstart(data)
    np.testing.assert_allclose([f.alpha, f.beta, f.gamma, f.delta], [a, b, c, delta],
                               rtol=1e-9, atol=1e-12)


def test_fit_constant_data_raises_clear_error():
    with pytest.raises(ValueError, match="interquartile"):
        LevyStableDistribution.fit(np.ones(200))


# ----------------------------------------------------------------- VaR / CVaR
def test_var_uses_stable_time_scaling_and_seed():
    d = LevyStableDistribution(1.7, 0.0, 1.0, 0.0)
    p = LevyFlightProcess(d)
    v1 = p.compute_var(0.99, 1, 20_000, seed=5)
    v10 = p.compute_var(0.99, 10, 20_000, seed=5)
    assert v10 / v1 == pytest.approx(10 ** (1 / 1.7), rel=1e-9)  # not sqrt(10)
    assert p.compute_var(0.99, 10, 20_000, seed=5) == v10  # reproducible


def test_var_cvar_against_exact_stable_quantile():
    d = LevyStableDistribution(1.7, 0.0, 1.0, 0.0)
    p = LevyFlightProcess(d)
    h = 10
    exact = -stats.levy_stable.ppf(0.01, 1.7, 0.0, scale=h ** (1 / 1.7))
    var = p.compute_var(0.99, h, 200_000, seed=1)
    cvar = p.compute_cvar(0.99, h, 200_000, seed=1)
    assert var == pytest.approx(exact, rel=0.05)
    assert cvar > var


# ---------------------------------------------------------------- simulate_path
def test_simulate_path_no_nan_for_heavy_jumps():
    p = LevyFlightProcess(LevyStableDistribution(0.5, 0.0, 1.0, 0.0), jump_intensity=50)
    with np.errstate(all="ignore"):
        paths = p.simulate_path(100.0, 1.0, 252, 200, seed=1)
    assert paths.shape == (200, 253)
    assert not np.isnan(paths).any()
    assert np.all(paths[:, 0] == 100.0)


def test_simulate_path_single_path_matches_reference_loop():
    # One path: identical random stream as the original per-path loop
    levy = LevyStableDistribution(1.7, -0.1, 1.0, 0.0)
    p = LevyFlightProcess(levy, mu=0.05, sigma_gaussian=0.2, jump_intensity=20)
    got = p.simulate_path(100.0, 1.0, 252, 1, seed=4)

    np.random.seed(4)
    dt = 1.0 / 252
    ref = [100.0]
    for _ in range(252):
        dW = np.random.normal(0, np.sqrt(dt), 1)
        nj = np.random.poisson(20 * dt, 1)
        jump = np.sum(levy.sample(nj[0])) * 0.01 if nj[0] > 0 else 0.0
        lr = (0.05 - 0.5 * 0.2**2) * dt + 0.2 * dW[0] + jump
        ref.append(ref[-1] * np.exp(lr))
    np.testing.assert_allclose(got[0], ref, rtol=1e-10)


def test_simulate_path_multi_path_jump_statistics():
    # sigma = 0, mu = 0: log(S_T/S0) is the sum of 0.01-scaled jumps per path
    levy = LevyStableDistribution(2.0, 0.0, 1.0, 0.0)  # N(0, 2) jumps
    p = LevyFlightProcess(levy, mu=0.0, sigma_gaussian=0.0, jump_intensity=50)
    paths = p.simulate_path(1.0, 1.0, 50, 4000, seed=2)
    log_ret = np.log(paths[:, -1])
    # compound Poisson: Var = lambda * T * E[J^2] = 50 * (0.01^2 * 2)
    assert np.var(log_ret) == pytest.approx(50 * 2e-4, rel=0.08)


# ------------------------------------------------------------ estimate_tail_index
def test_hill_small_samples_and_zeros_do_not_crash():
    rng = np.random.default_rng(3)
    assert np.isfinite(estimate_tail_index(rng.standard_normal(5)))
    with_zeros = np.r_[np.zeros(950), rng.standard_normal(50)]
    assert np.isfinite(estimate_tail_index(with_zeros))
    assert np.isfinite(estimate_tail_index(np.r_[rng.standard_t(3, 500), np.nan]))
    with pytest.raises(ValueError):
        estimate_tail_index(np.zeros(20))
    with pytest.raises(ValueError):
        estimate_tail_index(rng.standard_normal(5), 'pickands')


def test_hill_on_pareto_tail():
    rng = np.random.default_rng(9)
    x = (rng.pareto(3.0, 20_000) + 1) * rng.choice([-1, 1], 20_000)
    assert estimate_tail_index(x) == pytest.approx(3.0, rel=0.08)


def test_levy_coordinates_still_work():
    r = LevyStableDistribution(1.7, 0.0, 0.01, 0.0).sample(300, seed=1)
    X, Y, Z = levy_flight_3d_coordinates(r, window=20)
    assert X.shape == Y.shape == Z.shape == (300,)
    assert np.all((Y >= 0) & (Y <= 2))
