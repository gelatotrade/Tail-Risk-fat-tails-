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
from scipy import optimize, integrate, stats
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
    - It is a scaled Student-t: X = μ + T / sqrt(β(3-q)) with
      T ~ t(ν), ν = (3-q)/(q-1). cdf, ppf and sample use this exact identity.
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
        x = np.asarray(x, dtype=float)
        if q == 1:
            return np.exp(x)

        arg = 1 + (1 - q) * x
        result = np.zeros_like(x, dtype=float)
        mask = arg > 0
        result[mask] = arg[mask] ** (1 / (1 - q))

        return result[()] if result.ndim == 0 else result

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

    def _scale_and_df(self) -> Tuple[float, float]:
        """Scale s and Student-t degrees of freedom ν with X = μ + s·T, T ~ t(ν).

        For q = 1 (Gaussian) ν = inf and s is the standard deviation 1/sqrt(2β).
        """
        q, beta = self.q, self.beta
        if q == 1:
            return 1 / np.sqrt(2 * beta), np.inf
        return 1 / np.sqrt(beta * (3 - q)), (3 - q) / (q - 1)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function (closed form via Student-t)."""
        x = np.asarray(x, dtype=float)
        s, nu = self._scale_and_df()
        z = (x - self.mu) / s
        if np.isinf(nu):
            result = stats.norm.cdf(z)
        else:
            result = stats.t.cdf(z, nu)
        return np.asarray(result, dtype=float).reshape(x.shape)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF, closed form via Student-t).

        ppf(0) = -inf, ppf(1) = +inf, NaN outside [0, 1].
        """
        p = np.asarray(p, dtype=float)
        s, nu = self._scale_and_df()
        if np.isinf(nu):
            z = stats.norm.ppf(p)
        else:
            z = stats.t.ppf(p, nu)
        result = self.mu + s * np.asarray(z, dtype=float)
        return result.reshape(p.shape)

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
            # Use Student-t sampling: X = T / sqrt(β(3-q)) with T ~ t(ν)
            samples = np.random.standard_t(nu, size)
            samples = samples / np.sqrt(beta * (3 - q))
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
        Var = 1 / (β(5-3q)) if q < 5/3
        Var = ∞ if q ≥ 5/3
        (q = 1 gives the Gaussian variance 1/(2β).)
        """
        if self.q >= 5/3:
            return np.inf

        return 1 / (self.beta * (5 - 3 * self.q))

    def tail_exponent(self) -> float:
        """
        Return the power-law tail exponent.

        For large |x|: P(x) ~ |x|^(-α) where α = 2/(q-1)
        """
        if self.q == 1:
            return np.inf  # Gaussian (exponential tails)

        return 2 / (self.q - 1)

    def tail_probability(self, x: float, tail: str = 'both',
                         exact: bool = False) -> float:
        """
        Compute tail probability.

        The distance d = |x - μ| to the centre is used:
        'right' = P(X > μ + d), 'left' = P(X < μ - d), 'both' = P(|X - μ| > d).
        (For x above / below μ this is P(X > x) / P(X < x).)

        For large d, use the asymptotic formula (α = 2/(q-1)):
        P(X - μ > d) ~ C * d^(1 - α) / (α - 1),  C = A ((q-1)β)^(-1/(q-1))
        which follows from pdf(x) ~ C |x - μ|^(-α). The asymptote is capped at 1.

        Args:
            x: Threshold (scalar or array)
            tail: 'right', 'left' or 'both'
            exact: True uses the exact Student-t tail instead of the asymptote
        """
        if tail not in ('right', 'left', 'both'):
            raise ValueError("tail must be 'right', 'left' or 'both'")
        q = self.q
        d = np.abs(np.asarray(x, dtype=float) - self.mu)
        factor = 2.0 if tail == 'both' else 1.0

        if q == 1 or exact:
            # Exact tail (q = 1: Gaussian with variance 1/(2β))
            s, nu = self._scale_and_df()
            if np.isinf(nu):
                one_tail = stats.norm.sf(d / s)
            else:
                one_tail = stats.t.sf(d / s, nu)
            prob = factor * np.asarray(one_tail, dtype=float)
        else:
            # Power law asymptotic
            alpha = 2 / (q - 1)
            C = self.A * ((q - 1) * self.beta) ** (-1 / (q - 1))

            with np.errstate(divide='ignore', over='ignore'):
                prob = factor * C * d**(1 - alpha) / (alpha - 1)
            prob = np.minimum(prob, 1.0)

        return float(prob) if np.ndim(prob) == 0 else prob

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

            # For q-Gaussian: kurtosis = 3(5-3q)/(7-5q) for q < 7/5,
            # i.e. excess kurtosis k = 6(q-1)/(7-5q)  =>  q = (7k+6)/(5k+6).
            # (Moments can only identify q < 7/5; heavier tails -> q -> 7/5.)
            excess = max(kurt - 3, 0.0)
            q = np.clip((7 * excess + 6) / (5 * excess + 6), 1.01, 2.5)

            # Estimate beta from variance: Var = 1/(β(5-3q))
            if q < 5/3:
                beta = 1 / (var * (5 - 3 * q))
            else:
                beta = 1 / var  # Fallback

        elif method == 'mle':
            # Fit on standardised data z = (x - μ)/s and rescale afterwards:
            # if Z ~ qG(q, β_z) then sZ ~ qG(q, β_z / s²). This keeps the
            # optimiser bounds meaningful for any data scale (raw daily
            # returns need β ~ 1e3-1e4).
            scale = np.std(data_centered)
            if not scale > 0:
                raise ValueError("Tsallis MLE fit needs non-constant data")
            z = data_centered / scale

            def neg_log_likelihood(params):
                q, beta = params
                if q <= 1 or q >= 3 or beta <= 0:
                    return np.inf
                try:
                    dist = TsallisDistribution(q, beta, 0.0)
                    return -np.sum(dist.log_pdf(z))
                except:
                    return np.inf

            # Optimize
            result = optimize.minimize(
                neg_log_likelihood,
                x0=[1.5, 1.0],
                bounds=[(1.01, 2.9), (0.01, 100)]
            )
            q, beta_z = result.x
            beta = beta_z / scale**2

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
        returns = np.asarray(returns, dtype=float).ravel()
        abs_returns = np.abs(returns[np.isfinite(returns)])
        sorted_returns = np.sort(abs_returns)[::-1]
        n_pos = int(np.count_nonzero(sorted_returns > 0))
        if n_pos < 2:
            raise ValueError("estimate_optimal_q needs at least 2 non-zero returns")

        k = max(10, len(sorted_returns) // 20)
        k = min(k, n_pos - 1)  # threshold X_(k+1) must exist and be > 0
        log_ratios = np.log(sorted_returns[:k]) - np.log(sorted_returns[k])
        hill_estimate = 1 / np.mean(log_ratios)

        # Convert to q: Hill estimates the CCDF exponent α_H; the density
        # decays as |x|^-(α_H+1) = |x|^(-2/(q-1))  =>  q = 1 + 2/(α_H + 1)
        q = 1 + 2 / (hill_estimate + 1)

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
        # P(|X - μ| > t·σ) of the fitted q-Gaussian (exact Student-t tail)
        extreme_probs = {
            f'{t}sigma': dist.tail_probability(dist.mu + t * std, 'both', exact=True)
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
