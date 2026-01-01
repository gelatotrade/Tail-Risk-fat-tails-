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
        Expected Shortfall (CVaR).

        For Student-t:
        ES_α = (ν + z_α²)/(ν-1) * t_pdf(z_α)/α * scale + loc
        """
        alpha = 1 - confidence
        z_alpha = self.ppf(alpha)
        t_pdf = self.pdf(z_alpha)

        if self.nu > 1:
            es = (self.nu + z_alpha**2) / (self.nu - 1) * t_pdf / alpha * self.scale
            return es - self.loc
        else:
            return np.inf

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
        self.norm_const = (
            (gamma / self.delta)**self.lam
            / (np.sqrt(2 * np.pi) * bessel_k(self.lam, self.delta * gamma))
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function.

        f(x) = a(λ,α,β,δ) * (δ² + (x-μ)²)^((λ-1/2)/2) * K_{λ-1/2}(α√(δ²+(x-μ)²)) * exp(β(x-μ))
        """
        x = np.asarray(x)
        z = x - self.mu

        q = np.sqrt(self.delta**2 + z**2)
        bessel_term = bessel_k(self.lam - 0.5, self.alpha * q)

        pdf = (
            self.norm_const
            * q**(self.lam - 0.5)
            * bessel_term
            * np.exp(self.beta * z)
        )

        return pdf

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample using Normal Variance-Mean mixture representation.

        X = μ + βW + √W * Z where:
        - W ~ GIG(λ, δ², γ²) (Generalized Inverse Gaussian)
        - Z ~ N(0, 1)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample from GIG (approximate)
        # Use rejection sampling with inverse Gaussian proposal
        W = self._sample_gig(size)

        # Sample standard normal
        Z = np.random.normal(0, 1, size)

        # Construct GH samples
        X = self.mu + self.beta * W + np.sqrt(W) * Z

        return X

    def _sample_gig(self, size: int) -> np.ndarray:
        """Sample from Generalized Inverse Gaussian distribution."""
        # Simplified sampling using mode and variance matching
        lam, delta, gamma = self.lam, self.delta, self.gamma

        # Mode of GIG
        mode = (delta * np.sqrt(lam**2 + delta**2 * gamma**2) + lam * delta) / (delta * gamma)

        # Use gamma distribution as approximation
        if lam >= 0:
            shape = lam + 0.5
            scale = delta**2 / (2 * mode)
        else:
            shape = -lam + 0.5
            scale = delta**2 / (2 * mode)

        samples = np.random.gamma(shape, scale, size)
        return np.maximum(samples, 1e-10)

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
        x = np.asarray(x)
        z = x - self.mu
        q = np.sqrt(self.delta**2 + z**2)

        pdf = (
            self.alpha * self.delta / np.pi
            * np.exp(self.delta * self.gamma + self.beta * z)
            * bessel_k(1, self.alpha * q) / q
        )

        return pdf

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Sample from NIG using inverse Gaussian mixing."""
        if seed is not None:
            np.random.seed(seed)

        # Sample from Inverse Gaussian
        from scipy.stats import invgauss
        W = invgauss.rvs(mu=self.delta/self.gamma, scale=self.delta**2, size=size)

        # Sample standard normal
        Z = np.random.normal(0, 1, size)

        return self.mu + self.beta * W + np.sqrt(W) * Z

    @staticmethod
    def fit(data: np.ndarray) -> 'NormalInverseGaussian':
        """Fit NIG to data using method of moments."""
        data = np.asarray(data)

        mean = np.mean(data)
        var = np.var(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Method of moments estimators
        rho = 3 * kurt / skew**2 - 4 if abs(skew) > 0.01 else 10
        zeta = np.sign(skew) * np.sqrt(1 / rho)

        alpha = 3 * np.sqrt(1 + 1/rho) / (np.sqrt(var) * abs(kurt + 3))
        beta = alpha * zeta / np.sqrt(1 + zeta**2)
        delta = 3 * np.sqrt(var) * np.sqrt(1 + zeta**2) / abs(kurt + 3)
        mu = mean - delta * beta / np.sqrt(alpha**2 - beta**2)

        # Clamp to valid range
        alpha = max(abs(beta) + 0.01, alpha)

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
        """Compute PDF via FFT of characteristic function."""
        x = np.asarray(x)

        # FFT grid
        du = 0.1
        u = np.arange(-n_fft//2, n_fft//2) * du

        # Characteristic function values
        cf = self.characteristic_function(u)

        # Inverse FFT
        pdf_fft = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(cf))) * n_fft * du / (2 * np.pi)

        # Interpolate to x values
        x_fft = np.arange(-n_fft//2, n_fft//2) * 2 * np.pi / (n_fft * du)
        pdf = np.interp(x, x_fft, np.real(pdf_fft))

        return np.maximum(pdf, 0)

    @staticmethod
    def fit(data: np.ndarray) -> 'VarianceGamma':
        """Fit VG parameters using method of moments."""
        mean = np.mean(data)
        var = np.var(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        # Method of moments
        nu = kurt / 3 if kurt > 0 else 0.5
        theta = skew * np.sqrt(var) / (3 * nu) if nu > 0 else 0
        sigma = np.sqrt(var - theta**2 * nu) if var > theta**2 * nu else np.sqrt(var)
        mu = mean - theta / nu if nu > 0 else mean

        return VarianceGamma(sigma, theta, nu, mu)


def compare_distributions(data: np.ndarray) -> Dict[str, Dict]:
    """
    Fit and compare multiple fat-tail distributions.

    Returns fit quality metrics for each distribution.
    """
    results = {}

    # Gaussian (baseline)
    gauss_params = stats.norm.fit(data)
    gauss_ks = stats.kstest(data, 'norm', args=gauss_params)
    results['gaussian'] = {
        'params': {'loc': gauss_params[0], 'scale': gauss_params[1]},
        'ks_statistic': gauss_ks.statistic,
        'ks_pvalue': gauss_ks.pvalue
    }

    # Student-t
    t_dist = StudentTDistribution.fit(data)
    t_ks = stats.kstest(data, 't', args=(t_dist.nu, t_dist.loc, t_dist.scale))
    results['student_t'] = {
        'params': {'nu': t_dist.nu, 'loc': t_dist.loc, 'scale': t_dist.scale},
        'ks_statistic': t_ks.statistic,
        'ks_pvalue': t_ks.pvalue
    }

    # NIG
    try:
        nig = NormalInverseGaussian.fit(data)
        # Compute KS manually
        nig_cdf = np.array([np.mean(nig.sample(1000) <= x) for x in np.sort(data)])
        emp_cdf = np.arange(1, len(data) + 1) / len(data)
        nig_ks = np.max(np.abs(nig_cdf - emp_cdf))
        results['nig'] = {
            'params': {'alpha': nig.alpha, 'beta': nig.beta, 'delta': nig.delta, 'mu': nig.mu},
            'ks_statistic': nig_ks,
            'ks_pvalue': None
        }
    except:
        results['nig'] = {'error': 'Fitting failed'}

    # Variance Gamma
    try:
        vg = VarianceGamma.fit(data)
        results['variance_gamma'] = {
            'params': {'sigma': vg.sigma, 'theta': vg.theta, 'nu': vg.nu, 'mu': vg.mu}
        }
    except:
        results['variance_gamma'] = {'error': 'Fitting failed'}

    # Determine best fit
    ks_stats = {k: v.get('ks_statistic', np.inf) for k, v in results.items() if 'error' not in v}
    best_fit = min(ks_stats, key=ks_stats.get)
    results['best_fit'] = best_fit

    return results
