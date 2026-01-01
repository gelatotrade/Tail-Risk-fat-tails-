"""
Tsallis Statistics for Non-Extensive Financial Systems
=======================================================

Tsallis statistics extends Boltzmann-Gibbs thermodynamics to systems with
long-range correlations, memory effects, and fat tails - exactly what we
observe in financial markets.

Physics Background:
- Developed by Constantino Tsallis (1988)
- Generalizes entropy: S_q = k * (1 - Σp_i^q) / (q - 1)
- q = 1 recovers standard Boltzmann-Gibbs entropy
- q > 1: Fat tails (sub-extensive systems)
- q < 1: Compact support (super-extensive systems)

The Tsallis or q-Gaussian distribution:
P_q(x) ∝ [1 - β(1-q)x²]^(1/(1-q))

For financial markets:
- q ≈ 1.4-1.5 fits stock returns well
- The distribution has power-law tails: P(x) ~ |x|^(-2/(q-1))
- This naturally produces Black Swan events

Key Advantages:
1. Theoretical foundation from statistical mechanics
2. Maximum entropy principle preserved
3. Accounts for correlations and memory
4. Analytically tractable
"""

import numpy as np
from scipy import optimize, integrate
from scipy.special import gamma as gamma_func, beta as beta_func
from typing import Tuple, Optional, Dict
import warnings


class TsallisDistribution:
    """
    Tsallis (q-Gaussian) distribution for fat-tailed financial returns.

    The q-Gaussian is the maximum entropy distribution under Tsallis entropy:
    P_q(x) = A_q * exp_q(-β * x²)

    where exp_q(x) = [1 + (1-q)x]^(1/(1-q)) is the q-exponential

    For q > 1:
    - Distribution has power-law tails
    - Tail exponent: α = 2/(q-1)
    - Variance is infinite for q ≥ 5/3
    """

    def __init__(self, q: float = 1.4, beta: float = 1.0, mu: float = 0.0):
        """
        Initialize Tsallis distribution.

        Args:
            q: Entropic index (q > 1 for fat tails, q = 1 for Gaussian)
            beta: Inverse temperature (relates to variance)
            mu: Location parameter
        """
        if q < 1:
            raise ValueError("q must be >= 1 for fat tails")
        if q >= 3:
            raise ValueError("q must be < 3 for normalizable distribution")
        if beta <= 0:
            raise ValueError("beta must be positive")

        self.q = q
        self.beta = beta
        self.mu = mu

        # Compute normalization constant
        self._compute_normalization()

    def _compute_normalization(self):
        """Compute normalization constant A_q."""
        q = self.q
        beta = self.beta

        if q == 1:
            self.A = np.sqrt(beta / np.pi)
        else:
            # For q > 1: A_q = Γ(1/(q-1)) / [Γ((3-q)/(2(q-1))) * √(π/((q-1)β))]
            gamma_num = gamma_func(1 / (q - 1))
            gamma_den = gamma_func((3 - q) / (2 * (q - 1)))
            self.A = gamma_num / (gamma_den * np.sqrt(np.pi / ((q - 1) * beta)))

    def q_exponential(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the q-exponential function.

        exp_q(x) = [1 + (1-q)x]^(1/(1-q)) if 1 + (1-q)x > 0
                 = 0 otherwise (for q > 1)

        For q -> 1, this converges to the standard exponential.
        """
        q = self.q
        if q == 1:
            return np.exp(x)

        arg = 1 + (1 - q) * x
        result = np.zeros_like(x, dtype=float)
        mask = arg > 0
        result[mask] = arg[mask] ** (1 / (1 - q))

        return result

    def q_logarithm(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the q-logarithm function.

        ln_q(x) = (x^(1-q) - 1) / (1 - q)

        This is the inverse of the q-exponential.
        """
        q = self.q
        if q == 1:
            return np.log(x)

        return (x**(1 - q) - 1) / (1 - q)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function.

        P_q(x) = A_q * exp_q(-β * (x - μ)²)
        """
        x = np.asarray(x)
        z = x - self.mu
        return self.A * self.q_exponential(-self.beta * z**2)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability density (using q-logarithm for consistency)."""
        return np.log(self.pdf(x) + 1e-10)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function (numerical integration)."""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i, xi in enumerate(x.flat):
            result.flat[i], _ = integrate.quad(self.pdf, -50, xi)

        return result

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF)."""
        p = np.asarray(p)
        result = np.zeros_like(p, dtype=float)

        for i, pi in enumerate(p.flat):
            # Binary search for inverse
            def objective(x):
                return self.cdf(np.array([x]))[0] - pi

            try:
                result.flat[i] = optimize.brentq(objective, -50, 50)
            except:
                result.flat[i] = np.nan

        return result

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate samples using acceptance-rejection method.

        Uses a Cauchy proposal distribution for efficient sampling.
        """
        if seed is not None:
            np.random.seed(seed)

        q = self.q
        beta = self.beta

        if q == 1:
            # Standard Gaussian
            return np.random.normal(self.mu, 1 / np.sqrt(2 * beta), size)

        # For q > 1, use generalized Box-Muller
        # Sample from equivalent Student-t distribution
        nu = 2 / (q - 1) - 1  # Degrees of freedom

        if nu > 0:
            # Use Student-t sampling
            samples = np.random.standard_t(nu, size)
            samples = samples / np.sqrt(beta * (q - 1))
        else:
            # Fallback to acceptance-rejection
            samples = self._sample_acceptance_rejection(size)

        return samples + self.mu

    def _sample_acceptance_rejection(self, size: int) -> np.ndarray:
        """Acceptance-rejection sampling with Cauchy proposal."""
        samples = []
        M = self.A / (1 / np.pi)  # Proposal scaling

        while len(samples) < size:
            # Propose from Cauchy
            x = np.random.standard_cauchy(size)

            # Compute acceptance probability
            target = self.pdf(x)
            proposal = 1 / (np.pi * (1 + x**2))

            u = np.random.uniform(0, 1, size)
            accept = u < target / (M * proposal)

            samples.extend(x[accept])

        return np.array(samples[:size])

    def variance(self) -> float:
        """
        Compute variance (if finite).

        For q-Gaussian:
        Var = 1 / (β(3-q)) if q < 5/3
        Var = ∞ if q ≥ 5/3
        """
        if self.q >= 5/3:
            return np.inf

        return 1 / (self.beta * (3 - self.q))

    def tail_exponent(self) -> float:
        """
        Return the power-law tail exponent.

        For large |x|: P(x) ~ |x|^(-α) where α = 2/(q-1)
        """
        if self.q == 1:
            return np.inf  # Gaussian (exponential tails)

        return 2 / (self.q - 1)

    def tail_probability(self, x: float, tail: str = 'both') -> float:
        """
        Compute tail probability.

        For large |x|, use the asymptotic formula:
        P(X > x) ~ C * x^(1 - 2/(q-1)) / (2/(q-1) - 1)
        """
        q = self.q
        if q == 1:
            from scipy.stats import norm
            if tail == 'right':
                return 1 - norm.cdf(x)
            elif tail == 'left':
                return norm.cdf(x)
            else:
                return 2 * (1 - norm.cdf(abs(x)))

        # Power law asymptotic
        alpha = 2 / (q - 1)
        C = self.A / self.beta

        prob = C * abs(x)**(1 - alpha) / (alpha - 1)

        if tail == 'right':
            return prob
        elif tail == 'left':
            return prob
        else:
            return 2 * prob

    @staticmethod
    def fit(data: np.ndarray, method: str = 'moments') -> 'TsallisDistribution':
        """
        Fit Tsallis parameters to data.

        Args:
            data: Array of returns
            method: 'moments' or 'mle'

        Returns:
            Fitted TsallisDistribution
        """
        data = np.asarray(data)
        mu = np.median(data)
        data_centered = data - mu

        if method == 'moments':
            # Estimate q from kurtosis
            var = np.var(data_centered)
            kurt = np.mean(data_centered**4) / var**2

            # For q-Gaussian: kurtosis = 3(5-3q)/(7-5q) for q < 7/5
            # Approximate: q ≈ 1 + 6/(kurt + 3)
            q = np.clip(1 + 6 / (kurt + 3), 1.01, 2.5)

            # Estimate beta from variance
            if q < 5/3:
                beta = 1 / (var * (3 - q))
            else:
                beta = 1 / var  # Fallback

        elif method == 'mle':
            def neg_log_likelihood(params):
                q, beta = params
                if q <= 1 or q >= 3 or beta <= 0:
                    return np.inf
                try:
                    dist = TsallisDistribution(q, beta, mu)
                    return -np.sum(dist.log_pdf(data))
                except:
                    return np.inf

            # Optimize
            result = optimize.minimize(
                neg_log_likelihood,
                x0=[1.5, 1.0],
                bounds=[(1.01, 2.9), (0.01, 100)]
            )
            q, beta = result.x

        else:
            raise ValueError(f"Unknown method: {method}")

        return TsallisDistribution(q, beta, mu)


class TsallisEntropyAnalyzer:
    """
    Analyze market entropy using Tsallis framework.

    High Tsallis entropy indicates:
    - Increased uncertainty
    - Fat-tail regime activation
    - Potential for Black Swan events
    """

    def __init__(self, q_range: Tuple[float, float] = (1.0, 2.0)):
        """
        Initialize entropy analyzer.

        Args:
            q_range: Range of q values to analyze
        """
        self.q_range = q_range

    def compute_tsallis_entropy(self, P: np.ndarray, q: float) -> float:
        """
        Compute Tsallis entropy.

        S_q = (1 - Σ p_i^q) / (q - 1)

        For q -> 1: S_q -> -Σ p_i ln(p_i) (Shannon entropy)
        """
        P = np.asarray(P)
        P = P[P > 0]  # Remove zeros
        P = P / P.sum()  # Normalize

        if q == 1:
            return -np.sum(P * np.log(P))
        else:
            return (1 - np.sum(P**q)) / (q - 1)

    def estimate_optimal_q(self, returns: np.ndarray) -> float:
        """
        Estimate optimal q for the data.

        Uses the relationship between tail exponent and q.
        """
        # Estimate tail exponent using Hill estimator
        abs_returns = np.abs(returns)
        sorted_returns = np.sort(abs_returns)[::-1]

        k = max(10, len(returns) // 20)
        log_ratios = np.log(sorted_returns[:k]) - np.log(sorted_returns[k])
        hill_estimate = 1 / np.mean(log_ratios)

        # Convert to q: α = 2/(q-1) => q = 1 + 2/α
        q = 1 + 2 / hill_estimate

        return np.clip(q, 1.01, 2.5)

    def rolling_entropy(self, returns: np.ndarray, window: int = 50,
                       n_bins: int = 50) -> np.ndarray:
        """
        Compute rolling Tsallis entropy.

        Rising entropy often precedes market turbulence.
        """
        n = len(returns)
        entropy = np.zeros(n)

        for i in range(window, n):
            window_data = returns[i-window:i]

            # Estimate q for this window
            q = self.estimate_optimal_q(window_data)

            # Histogram to estimate probability
            hist, _ = np.histogram(window_data, bins=n_bins, density=True)
            hist = hist[hist > 0]
            hist = hist / hist.sum()

            entropy[i] = self.compute_tsallis_entropy(hist, q)

        return entropy


def compute_tsallis_coordinates(returns: np.ndarray,
                               window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D Tsallis phase space.

    Coordinates:
    - X: q parameter (entropic index / fat-tail measure)
    - Y: Tsallis entropy (uncertainty measure)
    - Z: Effective temperature 1/β (volatility proxy)

    This creates a thermodynamic phase space for market states.
    """
    n = len(returns)
    analyzer = TsallisEntropyAnalyzer()

    X = np.zeros(n)  # q parameter
    Y = np.zeros(n)  # Entropy
    Z = np.zeros(n)  # Temperature

    for i in range(window, n):
        window_data = returns[i-window:i]

        try:
            # Fit Tsallis distribution
            dist = TsallisDistribution.fit(window_data)

            X[i] = dist.q
            Z[i] = 1 / dist.beta  # Temperature

            # Compute entropy
            hist, _ = np.histogram(window_data, bins=30, density=True)
            hist = hist[hist > 0]
            hist = hist / hist.sum()
            Y[i] = analyzer.compute_tsallis_entropy(hist, dist.q)

        except:
            X[i] = 1.5
            Y[i] = 0
            Z[i] = np.var(window_data)

    return X, Y, Z


class TsallisTailRiskModel:
    """
    Comprehensive tail risk model using Tsallis statistics.

    Combines:
    1. q-Gaussian distribution fitting
    2. Non-extensive entropy analysis
    3. Critical slowing down detection (phase transitions)
    """

    def __init__(self):
        self.entropy_analyzer = TsallisEntropyAnalyzer()
        self.current_q = 1.5
        self.current_beta = 1.0

    def update(self, returns: np.ndarray) -> Dict:
        """
        Update model with new returns data.

        Returns comprehensive tail risk metrics.
        """
        # Fit distribution
        dist = TsallisDistribution.fit(returns)
        self.current_q = dist.q
        self.current_beta = dist.beta

        # Compute metrics
        tail_exponent = dist.tail_exponent()

        # Probability of extreme events
        thresholds = [3, 4, 5, 6]  # Standard deviations
        std = np.std(returns)
        extreme_probs = {
            f'{t}sigma': dist.tail_probability(t * std, 'both')
            for t in thresholds
        }

        # Entropy
        q = dist.q
        hist, _ = np.histogram(returns, bins=50, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        entropy = self.entropy_analyzer.compute_tsallis_entropy(hist, q)

        # Fat tail ratio (vs Gaussian)
        from scipy.stats import norm
        fat_tail_ratio = {
            f'{t}sigma': extreme_probs[f'{t}sigma'] / (2 * (1 - norm.cdf(t)))
            for t in thresholds
        }

        return {
            'q': dist.q,
            'beta': dist.beta,
            'tail_exponent': tail_exponent,
            'extreme_probabilities': extreme_probs,
            'fat_tail_ratio': fat_tail_ratio,
            'tsallis_entropy': entropy,
            'variance': dist.variance(),
            'risk_level': self._compute_risk_level(dist.q, entropy)
        }

    def _compute_risk_level(self, q: float, entropy: float) -> str:
        """Classify current risk level."""
        if q > 1.8 and entropy > 2:
            return 'EXTREME'
        elif q > 1.5 and entropy > 1.5:
            return 'HIGH'
        elif q > 1.3 and entropy > 1:
            return 'ELEVATED'
        else:
            return 'NORMAL'
