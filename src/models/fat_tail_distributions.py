"""
Fat-Tail Distributions for Financial Modeling
==============================================

Traditional finance assumes returns follow Gaussian distributions.
Reality shows:
- Fat tails (excess kurtosis)
- Skewness (asymmetry)
- Clustering (time-varying volatility)

This module provides alternative distributions that capture these features:
1. Student-t: Simple fat tails
2. Generalized Hyperbolic: Flexible skew and kurtosis
3. Normal Inverse Gaussian (NIG): Semi-heavy tails
4. Variance Gamma: Pure jump process limit
5. Generalized Pareto Distribution (GPD): Extreme tail modeling
"""

import numpy as np
from scipy import stats, optimize, special
from scipy.special import kv as bessel_k  # Modified Bessel function
from typing import Tuple, Optional, Dict, List
import warnings


class StudentTDistribution:
    """
    Student-t distribution for fat-tailed returns.

    The simplest fat-tail extension of Gaussian:
    - ν (nu): Degrees of freedom
    - ν → ∞: Gaussian
    - ν = 1: Cauchy (extremely fat tails)
    - ν ≈ 4-6: Typical for stock returns

    Tail behavior: P(X > x) ~ x^(-ν) for large x
    """

    def __init__(self, nu: float = 5.0, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize Student-t distribution.

        Args:
            nu: Degrees of freedom (lower = fatter tails)
            loc: Location parameter
            scale: Scale parameter
        """
        if nu <= 0:
            raise ValueError("nu must be positive")

        self.nu = nu
        self.loc = loc
        self.scale = scale
        self._dist = stats.t(df=nu, loc=loc, scale=scale)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.cdf(x)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        return self._dist.ppf(p)

    def sample(self, size: int = 1000) -> np.ndarray:
        return self._dist.rvs(size=size)

    def var(self, confidence: float = 0.99) -> float:
        """Value at Risk at given confidence level."""
        return -self.ppf(1 - confidence)

    def expected_shortfall(self, confidence: float = 0.99) -> float:
        """
        Expected Shortfall (CVaR) as a positive loss.

        For Student-t (z_α and t_ν of the *standard* t, α = 1 - confidence):
        ES_α = scale * (ν + z_α²)/(ν-1) * t_ν(z_α)/α - loc,  z_α = t_ν⁻¹(α)
        """
        if self.nu <= 1:
            return np.inf

        alpha = 1 - confidence
        z_alpha = stats.t.ppf(alpha, self.nu)
        t_pdf = stats.t.pdf(z_alpha, self.nu)

        es_standard = (self.nu + z_alpha**2) / (self.nu - 1) * t_pdf / alpha
        return self.scale * es_standard - self.loc

    @staticmethod
    def fit(data: np.ndarray) -> 'StudentTDistribution':
        """Fit Student-t to data using MLE."""
        # Use scipy's built-in fitting
        params = stats.t.fit(data)
        return StudentTDistribution(nu=params[0], loc=params[1], scale=params[2])


class GeneralizedHyperbolic:
    """
    Generalized Hyperbolic (GH) Distribution.

    A very flexible family that includes:
    - λ = 1: Hyperbolic
    - λ = -1/2: Normal Inverse Gaussian (NIG)
    - λ → -∞: Variance Gamma
    - λ = -(ν+1)/2: Generalized Student-t

    Parameters:
    - λ (lambda): Shape parameter
    - α: Tail heaviness (α > 0)
    - β: Skewness (-α < β < α)
    - δ: Scale (δ > 0)
    - μ: Location

    This is the most flexible distribution for financial returns.
    """

    def __init__(self, lam: float = 1.0, alpha: float = 1.0,
                 beta: float = 0.0, delta: float = 1.0, mu: float = 0.0):
        """
        Initialize GH distribution.

        Args:
            lam: Shape parameter (λ)
            alpha: Tail decay parameter
            beta: Skewness parameter
            delta: Scale parameter
            mu: Location parameter
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")
        if abs(beta) >= alpha:
            raise ValueError("|beta| must be < alpha")

        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu

        # Precompute normalization
        self._compute_normalization()

    def _compute_normalization(self):
        """Compute normalization constant."""
        gamma = np.sqrt(self.alpha**2 - self.beta**2)
        self.gamma = gamma

        # Normalization constant
        # a(λ,α,β,δ) = (γ/δ)^λ / (√(2π) α^(λ-1/2) K_λ(δγ))
        self.norm_const = (
            (gamma / self.delta)**self.lam
            / (np.sqrt(2 * np.pi) * self.alpha**(self.lam - 0.5)
               * bessel_k(self.lam, self.delta * gamma))
        )
        # Same constant in log form, with the exponentially scaled Bessel
        # function (K_λ(z) = kve(λ, z) e^{-z}), so pdf() cannot overflow.
        self._log_norm_const = (
            self.lam * np.log(gamma / self.delta)
            - 0.5 * np.log(2 * np.pi)
            - (self.lam - 0.5) * np.log(self.alpha)
            - np.log(special.kve(self.lam, self.delta * gamma))
            + self.delta * gamma
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function.

        f(x) = a(λ,α,β,δ) * (δ² + (x-μ)²)^((λ-1/2)/2) * K_{λ-1/2}(α√(δ²+(x-μ)²)) * exp(β(x-μ))
        """
        x = np.asarray(x, dtype=float)
        z = x - self.mu

        q = np.sqrt(self.delta**2 + z**2)
        aq = self.alpha * q

        # log f = log a + (λ-1/2) log q + log K_{λ-1/2}(αq) + βz, with
        # K_v(αq) = kve(v, αq) e^{-αq}; βz - αq <= 0 keeps exp() finite.
        log_pdf = (
            self._log_norm_const
            + (self.lam - 0.5) * np.log(q)
            + np.log(special.kve(self.lam - 0.5, aq))
            - aq
            + self.beta * z
        )

        return np.exp(log_pdf)

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample using Normal Variance-Mean mixture representation.

        X = μ + βW + √W * Z where:
        - W ~ GIG(λ, δ², γ²) (Generalized Inverse Gaussian)
        - Z ~ N(0, 1)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample the GIG mixing variable (exact, scipy.stats.geninvgauss)
        W = self._sample_gig(size)

        # Sample standard normal
        Z = np.random.normal(0, 1, size)

        # Construct GH samples
        X = self.mu + self.beta * W + np.sqrt(W) * Z

        return X

    def _sample_gig(self, size: int) -> np.ndarray:
        """
        Sample W ~ GIG(λ, χ=δ², ψ=γ²), density ∝ w^(λ-1) exp(-(δ²/w + γ²w)/2).

        Equals (δ/γ) * Y with Y ~ scipy.stats.geninvgauss(p=λ, b=δγ).
        Draws from the global NumPy random state (seeded by sample()).
        """
        lam, delta, gamma = self.lam, self.delta, self.gamma

        samples = stats.geninvgauss.rvs(lam, delta * gamma, scale=delta / gamma,
                                        size=size)
        return np.maximum(samples, 1e-300)

    def moments(self) -> Dict[str, float]:
        """Compute moments of the GH distribution."""
        lam, alpha, beta, delta = self.lam, self.alpha, self.beta, self.delta
        gamma = self.gamma

        # Mean
        mean = self.mu + delta * beta / gamma * bessel_k(lam+1, delta*gamma) / bessel_k(lam, delta*gamma)

        # Variance (approximate)
        var = delta / gamma * bessel_k(lam+1, delta*gamma) / bessel_k(lam, delta*gamma)
        var += (beta / gamma)**2 * delta**2 * (
            bessel_k(lam+2, delta*gamma) / bessel_k(lam, delta*gamma)
            - (bessel_k(lam+1, delta*gamma) / bessel_k(lam, delta*gamma))**2
        )

        return {
            'mean': mean,
            'variance': var,
            'std': np.sqrt(var)
        }


class NormalInverseGaussian:
    """
    Normal Inverse Gaussian (NIG) Distribution.

    Special case of GH with λ = -1/2.

    Well-suited for financial returns due to:
    - Semi-heavy tails (lighter than Cauchy, heavier than Gaussian)
    - Flexible skewness
    - Closed-form density

    Parameters:
    - α: Tail heaviness (α > 0)
    - β: Skewness (-α < β < α)
    - δ: Scale (δ > 0)
    - μ: Location
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0,
                 delta: float = 1.0, mu: float = 0.0):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu
        self.gamma = np.sqrt(alpha**2 - beta**2)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """NIG density function."""
        x = np.asarray(x, dtype=float)
        z = x - self.mu
        q = np.sqrt(self.delta**2 + z**2)
        aq = self.alpha * q

        # K_1(αq) = kve(1, αq) e^{-αq}; δγ + βz - αq <= 0, so exp() cannot overflow
        pdf = (
            self.alpha * self.delta / np.pi
            * np.exp(self.delta * self.gamma + self.beta * z - aq)
            * special.kve(1, aq) / q
        )

        return pdf

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Sample from NIG using inverse Gaussian mixing."""
        if seed is not None:
            np.random.seed(seed)

        # Mixing variable W ~ IG(mean δ/γ, shape δ²). scipy's invgauss(mu, scale)
        # has mean mu*scale and shape scale, hence mu = 1/(δγ), scale = δ².
        from scipy.stats import invgauss
        W = invgauss.rvs(mu=1 / (self.delta * self.gamma), scale=self.delta**2, size=size)

        # Sample standard normal
        Z = np.random.normal(0, 1, size)

        return self.mu + self.beta * W + np.sqrt(W) * Z

    @staticmethod
    def fit(data: np.ndarray) -> 'NormalInverseGaussian':
        """
        Fit NIG to data using method of moments.

        Solves the NIG moment equations (γ = √(α²-β²), ρ = β/α)
            mean = μ + δβ/γ,  var = δα²/γ³,
            skew = 3ρ/√(δγ),  excess kurtosis = 3(1 + 4ρ²)/(δγ)
        for (α, β, δ, μ). NIG requires 3·kurt > 5·skew²; sample moments
        outside that region (e.g. near-Gaussian or platykurtic data) are moved
        to its boundary, which gives a nearly Gaussian NIG.
        """
        data = np.asarray(data, dtype=float)

        mean = np.mean(data)
        var = np.var(data)
        skew = float(stats.skew(data))
        kurt = float(stats.kurtosis(data))

        if not (np.isfinite(var) and var > 0):
            raise ValueError("NIG fit needs data with positive, finite variance")

        # Admissible region of the NIG: kurt > 5/3 * skew^2 (and kurt > 0)
        kurt = max(kurt, 5.0 / 3.0 * skew**2 * 1.01, 1e-2)

        rho2 = skew**2 / (3 * kurt - 4 * skew**2)        # (β/α)², in [0, 1)
        delta_gamma = 3 * (1 + 4 * rho2) / kurt           # δγ
        gamma = np.sqrt(delta_gamma / (var * (1 - rho2)))
        delta = delta_gamma / gamma
        alpha = gamma / np.sqrt(1 - rho2)
        beta = np.sign(skew) * np.sqrt(rho2) * alpha
        mu = mean - delta * beta / gamma

        return NormalInverseGaussian(alpha, beta, delta, mu)


class VarianceGamma:
    """
    Variance Gamma Distribution.

    Limit of GH as λ → -∞ and δ → 0 appropriately.
    Pure jump process (no diffusion component).

    Parameters:
    - σ: Volatility of Brownian motion
    - θ: Drift of Brownian motion (controls skewness)
    - ν: Variance of gamma time change (controls kurtosis)
    - μ: Location

    Arises from time-changing Brownian motion by a gamma process.
    """

    def __init__(self, sigma: float = 0.2, theta: float = -0.1,
                 nu: float = 0.5, mu: float = 0.0):
        """
        Initialize Variance Gamma distribution.

        Args:
            sigma: Volatility parameter
            theta: Drift parameter (negative for left skew)
            nu: Variance rate (higher = fatter tails)
            mu: Location parameter
        """
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self.mu = mu

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample VG by subordinating Brownian motion.

        X = μ + θG + σ√G * Z
        where G ~ Gamma(1/ν, ν) and Z ~ N(0,1)
        """
        if seed is not None:
            np.random.seed(seed)

        # Gamma time change
        G = np.random.gamma(1/self.nu, self.nu, size)

        # Standard normal
        Z = np.random.normal(0, 1, size)

        return self.mu + self.theta * G + self.sigma * np.sqrt(G) * Z

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Characteristic function of VG.

        φ(u) = exp(iμu) * (1 - iuθν + σ²νu²/2)^(-1/ν)
        """
        u = np.asarray(u)
        inner = 1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u**2
        return np.exp(1j * self.mu * u) * inner**(-1/self.nu)

    def pdf(self, x: np.ndarray, n_fft: int = 2048) -> np.ndarray:
        """
        Closed-form VG density (Madan, Carr & Chang 1998), z = x - μ:

        f(x) = 2 e^{θz/σ²} / (ν^{1/ν} √(2π) σ Γ(1/ν))
               · (|z|/c)^{1/ν-1/2} · K_{1/ν-1/2}(|z|·c/σ²),   c = √(2σ²/ν + θ²)

        It is the exact inverse Fourier transform of characteristic_function()
        at any scale (an FFT grid cannot resolve daily-return widths). For
        ν >= 2 the density has an integrable singularity at x = μ (inf there).

        Args:
            x: Evaluation points
            n_fft: Unused; kept for backward compatibility
        """
        x = np.asarray(x, dtype=float)
        sigma, theta, nu = self.sigma, self.theta, self.nu

        z = x - self.mu
        abs_z = np.abs(z)
        c = np.sqrt(2 * sigma**2 / nu + theta**2)
        v = 1 / nu - 0.5

        log_const = (
            np.log(2) - np.log(nu) / nu - 0.5 * np.log(2 * np.pi)
            - np.log(sigma) - special.gammaln(1 / nu)
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            w = abs_z * c / sigma**2
            # K_v(w) = kve(v, w) e^{-w};  θz/σ² - w <= 0 keeps exp() finite
            log_pdf = (
                log_const + theta * z / sigma**2
                + v * (np.log(abs_z) - np.log(c))
                + np.log(special.kve(v, w)) - w
            )
            pdf = np.exp(log_pdf)

        # x = μ: lim (|z|/c)^v K_v(w) = (σ²/c²)^v Γ(v) 2^(v-1) for v > 0, else inf
        at_mu = abs_z == 0
        if np.any(at_mu):
            if v > 0:
                peak = np.exp(log_const + v * np.log(sigma**2 / c**2)
                              + special.gammaln(v) + (v - 1) * np.log(2))
            else:
                peak = np.inf
            pdf = np.where(at_mu, peak, pdf)

        return pdf

    @staticmethod
    def fit(data: np.ndarray) -> 'VarianceGamma':
        """Fit VG parameters using method of moments."""
        mean = np.mean(data)
        var = np.var(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Method of moments (small-θ approximation: kurt ≈ 3ν, skew·sd³ ≈ 3θνσ²)
        nu = kurt / 3 if kurt > 0 else 0.5
        theta = skew * np.sqrt(var) / (3 * nu) if nu > 0 else 0
        sigma = np.sqrt(var - theta**2 * nu) if var > theta**2 * nu else np.sqrt(var)
        mu = mean - theta  # E[X] = μ + θ

        return VarianceGamma(sigma, theta, nu, mu)


def _normal_mixture_cdf(x: np.ndarray, mu: float, drift: float, vol: float,
                        mixing_dist, n_nodes: int = 2000) -> np.ndarray:
    """
    CDF of the normal variance-mean mixture X = μ + drift·W + vol·√W·Z.

    F(x) = E_W[Φ((x - μ - drift·W) / (vol·√W))], evaluated with W at
    n_nodes equal-probability quantiles of ``mixing_dist`` (a frozen scipy
    distribution). Deterministic, vectorized, accuracy about 1/n_nodes.
    """
    x = np.asarray(x, dtype=float)
    w = mixing_dist.ppf((np.arange(n_nodes) + 0.5) / n_nodes)
    w = np.maximum(w, 1e-300)
    sqrt_w = vol * np.sqrt(w)

    flat = x.ravel()
    cdf = np.empty(flat.shape)
    chunk = 256  # bounds memory to chunk * n_nodes doubles
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for i in range(0, flat.size, chunk):
            xi = flat[i:i + chunk, None]
            cdf[i:i + chunk] = special.ndtr((xi - mu - drift * w) / sqrt_w).mean(axis=1)
    return cdf.reshape(x.shape)


def compare_distributions(data: np.ndarray) -> Dict[str, Dict]:
    """
    Fit and compare multiple fat-tail distributions.

    Returns fit quality metrics for each distribution. Every model gets a
    Kolmogorov-Smirnov statistic against its own fitted CDF (NIG and VG via
    their normal-mixture representation); 'best_fit' is the smallest one.
    The KS p-values ignore that the parameters were estimated from the same
    data (they are optimistic).
    """
    data = np.asarray(data, dtype=float)
    results = {}

    # Gaussian (baseline)
    gauss_params = stats.norm.fit(data)
    gauss_ks = stats.kstest(data, stats.norm(*gauss_params).cdf)
    results['gaussian'] = {
        'params': {'loc': gauss_params[0], 'scale': gauss_params[1]},
        'ks_statistic': gauss_ks.statistic,
        'ks_pvalue': gauss_ks.pvalue
    }

    # Student-t
    t_dist = StudentTDistribution.fit(data)
    t_ks = stats.kstest(data, t_dist.cdf)
    results['student_t'] = {
        'params': {'nu': t_dist.nu, 'loc': t_dist.loc, 'scale': t_dist.scale},
        'ks_statistic': t_ks.statistic,
        'ks_pvalue': t_ks.pvalue
    }

    # NIG: X = μ + βW + √W·Z, W ~ IG(mean δ/γ, shape δ²)
    try:
        nig = NormalInverseGaussian.fit(data)
        mixing = stats.invgauss(mu=1 / (nig.delta * nig.gamma), scale=nig.delta**2)
        nig_ks = stats.kstest(
            data, lambda x: _normal_mixture_cdf(x, nig.mu, nig.beta, 1.0, mixing)
        )
        results['nig'] = {
            'params': {'alpha': nig.alpha, 'beta': nig.beta, 'delta': nig.delta, 'mu': nig.mu},
            'ks_statistic': nig_ks.statistic,
            'ks_pvalue': nig_ks.pvalue
        }
    except Exception as exc:
        results['nig'] = {'error': f'Fitting failed: {exc}'}

    # Variance Gamma: X = μ + θG + σ√G·Z, G ~ Gamma(1/ν, scale ν)
    try:
        vg = VarianceGamma.fit(data)
        mixing = stats.gamma(1 / vg.nu, scale=vg.nu)
        vg_ks = stats.kstest(
            data, lambda x: _normal_mixture_cdf(x, vg.mu, vg.theta, vg.sigma, mixing)
        )
        results['variance_gamma'] = {
            'params': {'sigma': vg.sigma, 'theta': vg.theta, 'nu': vg.nu, 'mu': vg.mu},
            'ks_statistic': vg_ks.statistic,
            'ks_pvalue': vg_ks.pvalue
        }
    except Exception as exc:
        results['variance_gamma'] = {'error': f'Fitting failed: {exc}'}

    # Determine best fit
    ks_stats = {k: v.get('ks_statistic', np.inf) for k, v in results.items() if 'error' not in v}
    best_fit = min(ks_stats, key=ks_stats.get)
    results['best_fit'] = best_fit

    return results
