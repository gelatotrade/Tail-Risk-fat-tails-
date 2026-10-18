"""Regression tests for src/physics/tsallis_statistics.py (q-Gaussian math fixes)."""

import numpy as np
import pytest
from scipy import integrate, stats

from src.physics.tsallis_statistics import (
    TsallisDistribution,
    TsallisEntropyAnalyzer,
    TsallisTailRiskModel,
    compute_tsallis_coordinates,
)


def _numeric_sf(d, x):
    return integrate.quad(d.pdf, x, np.inf)[0]


@pytest.mark.parametrize("q,beta", [(1.2, 1.0), (1.4, 1.0), (1.6, 2.5)])
def test_variance_matches_integral(q, beta):
    d = TsallisDistribution(q, beta)
    num = integrate.quad(lambda z: z * z * d.pdf(z), -np.inf, np.inf, limit=200)[0]
    assert d.variance() == pytest.approx(num, rel=1e-6)  # 1/(beta(5-3q)), not 1/(beta(3-q))


def test_variance_gaussian_limit_and_infinite():
    assert TsallisDistribution(1.0, 2.0).variance() == pytest.approx(0.25)
    assert np.isinf(TsallisDistribution(1.7, 1.0).variance())


@pytest.mark.parametrize("q", [1.1, 1.4, 1.8])
def test_sample_follows_pdf(q):
    d = TsallisDistribution(q, 2.0, 0.3)
    s = d.sample(50_000, seed=1)
    ks = stats.kstest(s, d.cdf)
    assert ks.statistic < 0.01  # was 0.14 (samples too wide)


@pytest.mark.parametrize("q,x", [(1.4, 1.5), (2.5, 50.0), (2.5, -3.0)])
def test_cdf_matches_integral_of_pdf(q, x):
    d = TsallisDistribution(q, 1.0)
    num = integrate.quad(d.pdf, -np.inf, x, limit=200)[0]
    assert float(d.cdf(x)) == pytest.approx(num, abs=1e-7)  # lost mass below -50 before


def test_ppf_inverts_cdf_and_edges():
    d = TsallisDistribution(1.4, 3.0, -0.2)
    p = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    np.testing.assert_allclose(d.cdf(d.ppf(p)), p, atol=1e-10)
    edges = d.ppf(np.array([0.0, 1.0]))
    assert edges[0] == -np.inf and edges[1] == np.inf  # was -50 / NaN
    assert d.ppf(np.array([[0.5]])).shape == (1, 1)


def test_q_exponential_accepts_python_float():
    d = TsallisDistribution(1.4, 1.0)
    assert d.q_exponential(1.0) == pytest.approx((1 - 0.4) ** (1 / -0.4))
    np.testing.assert_allclose(d.q_exponential(np.array([0.0, -1.0])), [1.0, 1.4 ** (-2.5)])


def test_tail_probability_asymptote_and_exact():
    d = TsallisDistribution(1.4, 1.0)
    # asymptote must converge to the true tail (prefactor A((q-1)beta)^(-1/(q-1)))
    assert d.tail_probability(200.0, 'right') / _numeric_sf(d, 200.0) == pytest.approx(1, rel=0.02)
    # exact tail = 1 - cdf
    assert d.tail_probability(3.0, 'right', exact=True) == pytest.approx(1 - float(d.cdf(3.0)),
                                                                        rel=1e-8)
    assert d.tail_probability(3.0, 'both', exact=True) == pytest.approx(2 * (1 - float(d.cdf(3.0))))
    assert d.tail_probability(0.01, 'both') <= 1.0


def test_tail_probability_scale_invariant_and_gaussian_branch():
    s = 0.01
    z = TsallisDistribution(1.5, 1.0)
    x = TsallisDistribution(1.5, 1.0 / s**2)
    assert x.tail_probability(4 * s) == pytest.approx(z.tail_probability(4.0), rel=1e-10)
    g = TsallisDistribution(1.0, 2.0, 1.0)  # N(1, 0.5^2)
    assert g.tail_probability(2.5, 'right') == pytest.approx(stats.norm.sf(3.0))
    assert g.tail_probability(-0.5, 'left') == pytest.approx(stats.norm.sf(3.0))  # was Phi(x)


def test_fit_moments_kurtosis_inversion():
    rng = np.random.default_rng(5)
    gauss = TsallisDistribution.fit(rng.normal(0, 0.01, 50_000))
    assert gauss.q < 1.05  # was ~2.0 (Cauchy-like) for Gaussian data
    # Student-t(10) = q-Gaussian with q = 13/11 and excess kurtosis 1
    t10 = TsallisDistribution.fit(rng.standard_t(10, 400_000))
    assert t10.q == pytest.approx(13 / 11, abs=0.04)
    assert t10.variance() == pytest.approx(10 / 8, rel=0.03)


def test_fit_mle_is_scale_invariant():
    z = TsallisDistribution(1.4, 1.0).sample(5_000, seed=3)
    fz = TsallisDistribution.fit(z, 'mle')
    fx = TsallisDistribution.fit(0.01 * z, 'mle')  # raw daily-return scale
    assert fx.q == pytest.approx(fz.q, abs=1e-3)  # was pinned at the bounds (1.01, 100)
    assert fx.beta == pytest.approx(fz.beta * 1e4, rel=1e-2)
    assert fz.q == pytest.approx(1.4, abs=0.06)


def test_estimate_optimal_q_uses_density_exponent_and_small_n():
    an = TsallisEntropyAnalyzer()
    z = TsallisDistribution(1.4, 1.0).sample(200_000, seed=2)
    assert an.estimate_optimal_q(z) == pytest.approx(1.4, abs=0.1)  # 1 + 2/(alpha_H + 1)
    assert np.isfinite(an.estimate_optimal_q(np.random.default_rng(0).standard_normal(8)))
    e = an.rolling_entropy(np.random.default_rng(1).standard_t(4, 60) * 0.01, window=8)
    assert np.isfinite(e).all()


def test_tail_risk_model_probabilities_on_raw_returns():
    r = TsallisDistribution(1.35, 1.0).sample(3_000, seed=4) * 0.01
    out = TsallisTailRiskModel().update(r)
    probs = np.array(list(out['extreme_probabilities'].values()))
    assert np.all((probs > 0) & (probs <= 1))  # was ~1e19
    assert np.all(np.diff(probs) < 0)
    emp = np.mean(np.abs(r - np.median(r)) > 3 * r.std())
    assert out['extreme_probabilities']['3sigma'] == pytest.approx(emp, rel=0.5)


def test_tsallis_coordinates_api():
    r = np.random.default_rng(2).standard_t(4, 200) * 0.01
    X, Y, Z = compute_tsallis_coordinates(r, window=30)
    assert X.shape == Y.shape == Z.shape == (200,)
    assert np.all(X[30:] >= 1.0) and np.all(X[30:] < 1.4)
