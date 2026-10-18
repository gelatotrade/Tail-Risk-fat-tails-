"""Regression tests for src/models/fat_tail_distributions.py (closed forms, scipy references)."""

import warnings

import numpy as np
import pytest
from scipy import integrate, stats

from src.models.fat_tail_distributions import (
    GeneralizedHyperbolic,
    NormalInverseGaussian,
    StudentTDistribution,
    VarianceGamma,
    _normal_mixture_cdf,
    compare_distributions,
)


def _skewed_heavy_returns(n=3000, seed=0):
    """Daily-scale, left-skewed, fat-tailed sample (NIG draws, scipy reference)."""
    dist = stats.norminvgauss(1.2, -0.4, loc=0.0008, scale=0.008)
    return dist.rvs(size=n, random_state=np.random.default_rng(seed))


def _quad(f, lo, hi, points=None):
    return integrate.quad(f, lo, hi, points=points, limit=500)[0]


# --------------------------------------------------------------------------- Student-t


@pytest.mark.parametrize("nu,loc,scale", [(4, 0.001, 0.01), (5.2, 0.0006, 0.007), (3, 0.0, 1.0), (8, -0.2, 2.5)])
def test_student_t_expected_shortfall_matches_integral(nu, loc, scale):
    t = StudentTDistribution(nu, loc=loc, scale=scale)
    for conf in (0.95, 0.99):
        alpha = 1 - conf
        dist = stats.t(nu, loc=loc, scale=scale)
        reference = -dist.expect(lambda x: x, ub=dist.ppf(alpha)) / alpha
        assert t.expected_shortfall(conf) == pytest.approx(reference, rel=1e-8)
        assert t.expected_shortfall(conf) > t.var(conf)


def test_student_t_expected_shortfall_infinite_for_nu_le_1():
    assert StudentTDistribution(1.0, scale=0.01).expected_shortfall(0.99) == np.inf


# --------------------------------------------------------------------------- Generalized hyperbolic

GH_PARAMS = [(-0.5, 2.0, -0.5, 1.0, 0.1), (1.0, 1.5, -0.3, 1.0, 0.0), (2.5, 3.0, 1.0, 0.5, -0.2), (1.0, 300.0, -20.0, 0.004, 0.0005)]


@pytest.mark.parametrize("lam,alpha,beta,delta,mu", GH_PARAMS)
def test_gh_pdf_is_normalized_and_matches_scipy(lam, alpha, beta, delta, mu):
    gh = GeneralizedHyperbolic(lam, alpha, beta, delta, mu)
    sd = gh.moments()["std"]
    total = _quad(gh.pdf, mu - 60 * sd, mu + 60 * sd, points=[mu])
    assert total == pytest.approx(1.0, abs=1e-8)

    ref = stats.genhyperbolic(lam, alpha * delta, beta * delta, loc=mu, scale=delta)
    x = mu + sd * np.linspace(-8, 8, 41)
    np.testing.assert_allclose(gh.pdf(x), ref.pdf(x), rtol=1e-9)


def test_gh_pdf_is_finite_far_in_the_tails():
    gh = GeneralizedHyperbolic(1.0, 300.0, -20.0, 0.004, 0.0005)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = gh.pdf(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
    assert np.all(np.isfinite(values)) and values[2] > 0


@pytest.mark.parametrize("lam,alpha,beta,delta,mu", [(1.0, 1.0, 0.0, 1.0, 0.0), (-0.5, 2.0, -0.5, 1.0, 0.1), (2.0, 3.0, 1.0, 0.5, 0.0)])
def test_gh_sample_matches_moments(lam, alpha, beta, delta, mu):
    gh = GeneralizedHyperbolic(lam, alpha, beta, delta, mu)
    x = gh.sample(200_000, seed=1)
    m = gh.moments()
    assert x.mean() == pytest.approx(m["mean"], abs=0.02 * m["std"] + 1e-3)
    assert x.var() == pytest.approx(m["variance"], rel=0.03)
    np.testing.assert_array_equal(x, gh.sample(200_000, seed=1))


# --------------------------------------------------------------------------- NIG


def _scipy_nig(nig):
    return stats.norminvgauss(nig.alpha * nig.delta, nig.beta * nig.delta, loc=nig.mu, scale=nig.delta)


def test_nig_pdf_matches_scipy_and_has_no_overflow():
    for nig in (NormalInverseGaussian(2.0, -0.5, 0.5, 0.1), NormalInverseGaussian(800.0, 50.0, 2.0, 0.0)):
        x = nig.mu + np.linspace(-0.3, 0.3, 31)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = nig.pdf(x)
        np.testing.assert_allclose(values, _scipy_nig(nig).pdf(x), rtol=1e-9)


@pytest.mark.parametrize("alpha,beta,delta,mu", [(2.0, -0.5, 0.5, 0.0), (1.0, 0.3, 3.0, 1.0)])
def test_nig_sample_matches_scipy_moments(alpha, beta, delta, mu):
    nig = NormalInverseGaussian(alpha, beta, delta, mu)
    ref_mean, ref_var = _scipy_nig(nig).stats("mv")
    x = nig.sample(200_000, seed=1)
    assert x.var() == pytest.approx(ref_var, rel=0.03)
    assert x.mean() == pytest.approx(ref_mean, abs=0.02 * np.sqrt(ref_var))


def test_nig_fit_matches_all_four_sample_moments():
    data = _skewed_heavy_returns()
    nig = NormalInverseGaussian.fit(data)
    mean, var, skew, kurt = _scipy_nig(nig).stats("mvsk")
    assert mean == pytest.approx(np.mean(data), rel=1e-9)
    assert var == pytest.approx(np.var(data), rel=1e-9)
    assert skew == pytest.approx(stats.skew(data), rel=1e-9)
    assert kurt == pytest.approx(stats.kurtosis(data), rel=1e-9)
    # and the fitted sampler reproduces the data scale
    assert nig.sample(50_000, seed=2).std() == pytest.approx(np.std(data), rel=0.05)


def test_nig_fit_on_near_gaussian_data_is_valid():
    data = np.random.default_rng(3).normal(0.0005, 0.01, 2000)
    nig = NormalInverseGaussian.fit(data)
    assert abs(nig.beta) < nig.alpha and nig.delta > 0
    x = np.linspace(-0.05, 0.05, 11)
    values = nig.pdf(x)
    assert np.all(np.isfinite(values))
    assert _quad(nig.pdf, -0.2, 0.2) == pytest.approx(1.0, abs=1e-6)
    assert _scipy_nig(nig).std() == pytest.approx(np.std(data), rel=1e-6)


# --------------------------------------------------------------------------- Variance Gamma

VG_PARAMS = [(0.2, -0.1, 0.5, 0.05), (0.01, -0.002, 1.5, 0.0005), (0.012, -0.001, 0.4, 0.0)]


@pytest.mark.parametrize("sigma,theta,nu,mu", VG_PARAMS)
def test_vg_pdf_moments_and_orientation(sigma, theta, nu, mu):
    vg = VarianceGamma(sigma, theta, nu, mu)
    sd = np.sqrt(sigma**2 + theta**2 * nu)
    lo, hi = mu - 60 * sd, mu + 60 * sd
    f = lambda x: vg.pdf(np.array([x]))[0]
    assert _quad(f, lo, hi, points=[mu]) == pytest.approx(1.0, abs=1e-7)
    # E[X] = mu + theta (the old FFT pdf was mirrored: mean mu - theta)
    assert _quad(lambda x: x * f(x), lo, hi, points=[mu]) == pytest.approx(mu + theta, abs=1e-6 * sd + 1e-12)
    assert _quad(lambda x: (x - mu - theta) ** 2 * f(x), lo, hi, points=[mu]) == pytest.approx(sd**2, rel=1e-6)


def test_vg_pdf_is_fourier_inverse_of_characteristic_function():
    vg = VarianceGamma(0.01, -0.002, 1.5, 0.0005)
    sd = np.sqrt(vg.sigma**2 + vg.theta**2 * vg.nu)
    f = lambda x: vg.pdf(np.array([x]))[0]
    lo, hi = vg.mu - 60 * sd, vg.mu + 60 * sd
    for u in (0.5 / sd, 2 / sd):
        re = _quad(lambda x: f(x) * np.cos(u * x), lo, hi, points=[vg.mu])
        im = _quad(lambda x: f(x) * np.sin(u * x), lo, hi, points=[vg.mu])
        assert abs(re + 1j * im - vg.characteristic_function(np.array([u]))[0]) < 1e-8


def test_vg_pdf_at_mu_is_limit_or_inf():
    vg = VarianceGamma(0.2, -0.1, 0.5, 0.0)  # 1/nu - 1/2 > 0: finite peak
    assert vg.pdf(np.array([0.0]))[0] == pytest.approx(vg.pdf(np.array([1e-9]))[0], rel=1e-6)
    assert np.isinf(VarianceGamma(0.2, -0.1, 3.0, 0.0).pdf(np.array([0.0]))[0])


def test_vg_fit_location_reproduces_mean():
    data = _skewed_heavy_returns()
    vg = VarianceGamma.fit(data)
    assert vg.mu + vg.theta == pytest.approx(np.mean(data), rel=1e-12)


# --------------------------------------------------------------------------- compare_distributions


def test_mixture_cdf_matches_scipy_nig_cdf():
    nig = NormalInverseGaussian.fit(_skewed_heavy_returns())
    mixing = stats.invgauss(mu=1 / (nig.delta * nig.gamma), scale=nig.delta**2)
    x = np.linspace(-0.05, 0.04, 19)
    np.testing.assert_allclose(_normal_mixture_cdf(x, nig.mu, nig.beta, 1.0, mixing), _scipy_nig(nig).cdf(x), atol=1e-4)


def test_compare_distributions_scores_every_model():
    data = _skewed_heavy_returns(1500, seed=4)
    res = compare_distributions(data)  # crashed on SciPy 1.18 (kstest 'norm' with args)

    names = ["gaussian", "student_t", "nig", "variance_gamma"]
    for name in names:
        assert "error" not in res[name], res[name]
        assert 0 < res[name]["ks_statistic"] < 1
        assert 0 <= res[name]["ks_pvalue"] <= 1

    gauss = stats.norm(*stats.norm.fit(data))
    assert res["gaussian"]["ks_statistic"] == pytest.approx(stats.kstest(data, gauss.cdf).statistic)

    nig = NormalInverseGaussian.fit(data)
    exact = stats.kstest(data, _scipy_nig(nig).cdf).statistic
    assert res["nig"]["ks_statistic"] == pytest.approx(exact, abs=2e-4)

    assert res["best_fit"] == min(names, key=lambda k: res[k]["ks_statistic"])
    # data are NIG draws: the moment-fitted NIG fits far better than the Gaussian
    assert res["nig"]["ks_statistic"] < 0.5 * res["gaussian"]["ks_statistic"]
