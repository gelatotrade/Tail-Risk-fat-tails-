"""Regression tests for src/physics/fokker_planck.py (forward FPE, fractional operator, NumPy 2.x)."""

import numpy as np
import pytest
from scipy import stats

from src.physics.fokker_planck import (
    FokkerPlanckSolver,
    FokkerPlanckTailRisk,
    compute_fokker_planck_coordinates,
)
from src.physics.levy_flight import LevyStableDistribution


def _gauss(x, m, s):
    p = np.exp(-0.5 * ((x - m) / s) ** 2)
    return p / (p.sum() * (x[1] - x[0]))


# ------------------------------------------------------------ standard (local) FPE
def test_ou_mean_and_variance_follow_forward_equation():
    s = FokkerPlanckSolver()
    P0 = _gauss(s.x, 2.0, 0.3)
    params = {'theta': 1.0, 'D0': 0.001, 'lambda': 0.0}
    _, Pe = s.solve(P0, 1.0, 0.005, params)
    m = s.compute_moments(Pe[-1])
    # OU: E = m0 e^{-theta T}; Var = v0 e^{-2 theta T} + D/theta (1 - e^{-2 theta T})
    assert m['mean'] == pytest.approx(2.0 * np.exp(-1.0), abs=0.01)  # was 5.41
    var = 0.09 * np.exp(-2.0) + 0.001 * (1 - np.exp(-2.0))
    assert m['variance'] == pytest.approx(var, rel=0.05)


def test_operator_conserves_mass():
    s = FokkerPlanckSolver()
    L = s.build_operator_matrix(0, {'theta': 0.5, 'D0': 0.02, 'lambda': 0.5})
    col_sums = np.asarray(L.sum(axis=0)).ravel()
    np.testing.assert_allclose(col_sums[1:-1], 0.0, atol=1e-9)


def test_last_frame_is_at_final_time():
    s = FokkerPlanckSolver()
    P0 = _gauss(s.x, 0.0, 1.0)
    for T in (1.0, 60 / 252):
        times, Pe = s.solve(P0, T, 0.001)
        assert times[-1] == pytest.approx(T, abs=1e-12)  # was 0.991 / 0.237
        assert len(times) == len(Pe)


# --------------------------------------------------------------- fractional FPE
def test_fractional_operator_integer_alpha_and_laplacian_limit():
    s = FokkerPlanckSolver(n_grid=200)
    for a in (1.0, 2.0):
        assert np.isfinite(s.build_fractional_operator(a)).all()  # gamma poles before
    L2 = s.build_fractional_operator(2.0, {'D_alpha': 0.3})
    lap = s.build_operator_matrix(0, {'theta': 0.0, 'D0': 0.3, 'lambda': 0.0}).toarray()
    np.testing.assert_allclose(L2, lap, atol=1e-9)


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5, 1.7, 1.95])
def test_fractional_operator_is_diffusive(alpha):
    s = FokkerPlanckSolver(n_grid=200)
    L = s.build_fractional_operator(alpha, {'D_alpha': 0.05})
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    assert np.linalg.eigvalsh(L).max() <= 1e-10  # was up to +7.7 (anti-diffusive)


def test_fractional_evolution_matches_stable_convolution():
    # dP/dt = D d^a P/d|x|^a : a stable S(a, 0, gamma) evolves into
    # S(a, 0, (gamma^a + D T)^(1/a))
    a, g0, D, T = 1.5, 0.5, 0.5, 0.5
    s = FokkerPlanckSolver(x_min=-10, x_max=10, n_grid=400, alpha=a)
    P0 = LevyStableDistribution(a, 0.0, g0).pdf(s.x)
    P0 /= P0.sum() * s.dx
    _, Pe = s.solve(P0, T, 0.005, {'theta': 0.0, 'D_alpha': D}, use_fractional=True)
    g1 = (g0**a + D * T) ** (1 / a)
    ref = LevyStableDistribution(a, 0.0, g1).pdf(s.x)
    core = np.abs(s.x) < 3
    np.testing.assert_allclose(Pe[-1][core], ref[core], rtol=0.03, atol=2e-3)


def test_fractional_branch_keeps_drift():
    s = FokkerPlanckSolver(alpha=1.5)
    P0 = _gauss(s.x, 2.0, 0.3)
    _, Pe = s.solve(P0, 1.0, 0.01, {'theta': 1.0, 'D_alpha': 0.001}, use_fractional=True)
    assert s.compute_moments(Pe[-1])['mean'] == pytest.approx(2.0 * np.exp(-1.0), abs=0.03)


# -------------------------------------------------------------- tail risk metrics
def test_tail_risk_metrics_without_np_math():
    fr = FokkerPlanckTailRisk()
    metrics = fr.compute_tail_risk_metrics(fr.initial_distribution())  # np.math removed
    three = metrics['tail_probabilities']['3sigma']
    assert three['total'] == pytest.approx(2 * stats.norm.sf(3), rel=0.03)
    assert three['ratio_to_gaussian'] == pytest.approx(1.0, abs=0.03)


def test_tail_probability_is_continuous_in_threshold():
    s = FokkerPlanckSolver()
    P = _gauss(s.x, 0.0, 1.0)
    thr = np.linspace(2.9, 3.1, 201)
    left = np.array([s.tail_probability(P, t)[0] for t in thr])
    steps = np.diff(left)
    assert np.all(steps < 0)  # strictly decreasing (hard mask was flat between nodes)
    assert np.max(np.abs(steps)) < 2e-5  # ~pdf*0.001; hard mask jumped by ~P*dx = 1.8e-4
    assert left[100] == pytest.approx(stats.norm.sf(3), rel=0.01)


def test_forecast_alpha_one_is_finite_and_normalised():
    fr = FokkerPlanckTailRisk(alpha=1.0)
    P = fr.forecast_distribution(fr.initial_distribution(), 20)
    assert np.isfinite(P).all()  # was 498/500 NaN
    assert P.sum() * fr.solver.dx == pytest.approx(1.0, abs=1e-6)


def test_forecast_develops_fat_tails_and_surface_shape():
    fr = FokkerPlanckTailRisk(alpha=1.7)
    P = fr.forecast_distribution(fr.initial_distribution(), 60, vol_regime=2.0)
    assert fr.solver.compute_moments(P)['kurtosis'] > 0.05
    Z = fr.generate_3d_risk_surface(np.array([1.0, 2.0]), np.array([5, 20]))
    assert Z.shape == (2, 2)
    assert np.all(np.diff(Z, axis=0) > 0)  # higher vol regime -> more tail mass


def test_coordinates_unchanged_api():
    r = np.random.default_rng(0).standard_t(4, 200) * 0.01
    X, Y, Z = compute_fokker_planck_coordinates(r, window=30)
    assert X.shape == Y.shape == Z.shape == (200,)
