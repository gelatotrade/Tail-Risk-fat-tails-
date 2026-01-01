"""
Extreme Value Theory (EVT) for Tail Risk Modeling
==================================================

EVT provides the mathematical foundation for modeling extreme events.
Unlike fitting a distribution to all data, EVT focuses specifically
on the tail behavior.

Key Theorems:

1. Fisher-Tippett-Gnedenko Theorem:
   The maximum of n iid random variables, properly normalized,
   converges to one of three distributions:
   - Gumbel (ξ = 0): Thin tails (Gaussian, exponential)
   - Fréchet (ξ > 0): Fat tails (Pareto, Cauchy)
   - Weibull (ξ < 0): Bounded tails

   These unify into the Generalized Extreme Value (GEV) distribution.

2. Pickands-Balkema-de Haan Theorem:
   For large thresholds u, exceedances follow the
   Generalized Pareto Distribution (GPD):
   P(X - u ≤ y | X > u) ≈ G_{ξ,σ}(y)

   This is the foundation of Peak-Over-Threshold (POT) methods.

For financial tail risk:
- ξ > 0 (Fréchet domain) is typical
- ξ ≈ 0.2-0.4 for stock returns
- Higher ξ = fatter tails = higher tail risk
"""

import numpy as np
from scipy import stats, optimize
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional, Dict, List
import warnings


class GeneralizedExtremeValue:
    """
    Generalized Extreme Value (GEV) Distribution.

    F(x; μ, σ, ξ) = exp{-[1 + ξ(x-μ)/σ]^(-1/ξ)}

    Parameters:
    - μ: Location parameter
    - σ: Scale parameter (σ > 0)
    - ξ: Shape parameter (tail index)
      - ξ > 0: Fréchet (fat tails, heavy-tailed)
      - ξ = 0: Gumbel (light tails, exponential decay)
      - ξ < 0: Weibull (bounded upper tail)
    """

    def __init__(self, xi: float = 0.0, mu: float = 0.0, sigma: float = 1.0):
        """
        Initialize GEV distribution.

        Args:
            xi: Shape parameter (tail index)
            mu: Location parameter
            sigma: Scale parameter
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.xi = xi
        self.mu = mu
        self.sigma = sigma

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        x = np.asarray(x)
        z = (x - self.mu) / self.sigma

        if abs(self.xi) < 1e-10:
            # Gumbel case
            return np.exp(-np.exp(-z))
        else:
            t = 1 + self.xi * z
            t = np.maximum(t, 1e-10)  # Ensure positivity
            return np.exp(-t**(-1/self.xi))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        x = np.asarray(x)
        z = (x - self.mu) / self.sigma

        if abs(self.xi) < 1e-10:
            # Gumbel case
            return np.exp(-z - np.exp(-z)) / self.sigma
        else:
            t = 1 + self.xi * z
            mask = t > 0
            pdf = np.zeros_like(x, dtype=float)
            pdf[mask] = (
                t[mask]**(-1 - 1/self.xi) * np.exp(-t[mask]**(-1/self.xi))
                / self.sigma
            )
            return pdf

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF)."""
        p = np.asarray(p)

        if abs(self.xi) < 1e-10:
            return self.mu - self.sigma * np.log(-np.log(p))
        else:
            return self.mu + self.sigma * ((-np.log(p))**(-self.xi) - 1) / self.xi

    def return_level(self, return_period: float) -> float:
        """
        Compute return level for a given return period.

        The return level z_T is the quantile that is exceeded
        on average once every T time periods.

        z_T = F^(-1)(1 - 1/T)
        """
        p = 1 - 1 / return_period
        return self.ppf(p)

    @staticmethod
    def fit_block_maxima(data: np.ndarray, block_size: int = 20) -> 'GeneralizedExtremeValue':
        """
        Fit GEV using block maxima method.

        Divides data into blocks and fits GEV to block maxima.

        Args:
            data: Time series of returns
            block_size: Number of observations per block

        Returns:
            Fitted GEV distribution
        """
        data = np.asarray(data)
        n = len(data)
        n_blocks = n // block_size

        # Compute block maxima
        maxima = []
        for i in range(n_blocks):
            block = data[i*block_size:(i+1)*block_size]
            maxima.append(np.max(block))

        maxima = np.array(maxima)

        # Fit using scipy's genextreme (note: scipy uses c = -ξ)
        params = stats.genextreme.fit(maxima)
        c, loc, scale = params

        return GeneralizedExtremeValue(xi=-c, mu=loc, sigma=scale)


class GeneralizedParetoDistribution:
    """
    Generalized Pareto Distribution (GPD) for tail modeling.

    Used in Peak-Over-Threshold (POT) method.

    G(y; ξ, σ) = 1 - (1 + ξy/σ)^(-1/ξ)  for ξ ≠ 0
               = 1 - exp(-y/σ)           for ξ = 0

    Parameters:
    - ξ: Shape parameter (same as GEV)
    - σ: Scale parameter
    - u: Threshold (exceedances are x - u where x > u)
    """

    def __init__(self, xi: float = 0.1, sigma: float = 1.0, threshold: float = 0.0):
        """
        Initialize GPD.

        Args:
            xi: Shape parameter (tail index)
            sigma: Scale parameter
            threshold: Threshold for exceedances
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.xi = xi
        self.sigma = sigma
        self.threshold = threshold

    def cdf(self, y: np.ndarray) -> np.ndarray:
        """CDF for exceedances y = x - threshold."""
        y = np.asarray(y)

        if abs(self.xi) < 1e-10:
            return 1 - np.exp(-y / self.sigma)
        else:
            t = 1 + self.xi * y / self.sigma
            t = np.maximum(t, 1e-10)
            return 1 - t**(-1/self.xi)

    def pdf(self, y: np.ndarray) -> np.ndarray:
        """PDF for exceedances."""
        y = np.asarray(y)

        if abs(self.xi) < 1e-10:
            return np.exp(-y / self.sigma) / self.sigma
        else:
            t = 1 + self.xi * y / self.sigma
            mask = t > 0
            pdf = np.zeros_like(y, dtype=float)
            pdf[mask] = t[mask]**(-1 - 1/self.xi) / self.sigma
            return pdf

    def ppf(self, p: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF)."""
        p = np.asarray(p)

        if abs(self.xi) < 1e-10:
            return -self.sigma * np.log(1 - p)
        else:
            return self.sigma / self.xi * ((1 - p)**(-self.xi) - 1)

    def tail_probability(self, x: float) -> float:
        """
        Compute P(X > x) for the original variable.

        Uses the POT estimator:
        P(X > x) = P(X > u) * P(X > x | X > u)
        """
        if x <= self.threshold:
            return 1.0  # Exceedance is certain

        y = x - self.threshold
        return 1 - self.cdf(np.array([y]))[0]

    def var(self, p: float, exceedance_prob: float) -> float:
        """
        Compute VaR at confidence level p.

        Args:
            p: Confidence level (e.g., 0.99)
            exceedance_prob: P(X > threshold)

        Returns:
            VaR estimate
        """
        # Convert to exceedance probability
        # P(X > VaR) = 1 - p = P(X > u) * P(X - u > VaR - u | X > u)
        # => P(X - u > VaR - u | X > u) = (1 - p) / P(X > u)

        conditional_exceed_prob = (1 - p) / exceedance_prob
        if conditional_exceed_prob >= 1:
            return self.threshold  # VaR is below threshold

        var_exceedance = self.ppf(1 - conditional_exceed_prob)
        return self.threshold + var_exceedance

    def expected_shortfall(self, p: float, exceedance_prob: float) -> float:
        """
        Compute Expected Shortfall at confidence level p.

        ES = VaR + E[X - VaR | X > VaR]

        For GPD:
        ES = VaR / (1 - ξ) + (σ - ξu) / (1 - ξ)
        """
        if self.xi >= 1:
            return np.inf

        var = self.var(p, exceedance_prob)
        es = var / (1 - self.xi) + (self.sigma - self.xi * self.threshold) / (1 - self.xi)

        return es

    @staticmethod
    def fit_pot(data: np.ndarray, threshold: Optional[float] = None,
                quantile: float = 0.95) -> Tuple['GeneralizedParetoDistribution', Dict]:
        """
        Fit GPD using Peak-Over-Threshold method.

        Args:
            data: Time series data
            threshold: Explicit threshold (overrides quantile)
            quantile: Quantile for automatic threshold selection

        Returns:
            Fitted GPD and fit statistics
        """
        data = np.asarray(data)

        # Set threshold
        if threshold is None:
            threshold = np.percentile(data, quantile * 100)

        # Get exceedances
        exceedances = data[data > threshold] - threshold
        n_exceed = len(exceedances)
        n_total = len(data)

        if n_exceed < 10:
            warnings.warn("Few exceedances; results may be unreliable")

        # Fit using MLE (scipy's genpareto)
        params = stats.genpareto.fit(exceedances, floc=0)
        c, _, scale = params

        gpd = GeneralizedParetoDistribution(xi=c, sigma=scale, threshold=threshold)

        fit_stats = {
            'n_exceedances': n_exceed,
            'n_total': n_total,
            'exceedance_probability': n_exceed / n_total,
            'threshold': threshold,
            'mean_excess': np.mean(exceedances),
            'xi': c,
            'sigma': scale
        }

        return gpd, fit_stats


class EVTTailRiskAnalyzer:
    """
    Comprehensive EVT-based tail risk analyzer.

    Combines GEV (block maxima) and GPD (POT) methods
    for robust tail risk estimation.
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize with historical data.

        Args:
            data: Array of returns (or losses)
        """
        self.data = np.asarray(data)
        self.n = len(data)

        # Fit both tails
        self._fit_models()

    def _fit_models(self):
        """Fit GEV and GPD models to both tails."""
        # Right tail (large positive returns - gains)
        self.gpd_right, self.stats_right = GeneralizedParetoDistribution.fit_pot(
            self.data, quantile=0.95
        )

        # Left tail (large negative returns - losses)
        # Negate to work with maxima
        neg_data = -self.data
        self.gpd_left, self.stats_left = GeneralizedParetoDistribution.fit_pot(
            neg_data, quantile=0.95
        )

        # Block maxima for GEV
        self.gev_maxima = GeneralizedExtremeValue.fit_block_maxima(self.data)
        self.gev_minima = GeneralizedExtremeValue.fit_block_maxima(-self.data)

    def tail_index(self, tail: str = 'left') -> float:
        """Get tail index (ξ) for specified tail."""
        if tail == 'left':
            return self.gpd_left.xi
        else:
            return self.gpd_right.xi

    def var(self, confidence: float = 0.99, method: str = 'gpd') -> float:
        """
        Compute Value at Risk.

        Args:
            confidence: Confidence level
            method: 'gpd' (POT) or 'gev' (block maxima)

        Returns:
            VaR (positive number representing loss)
        """
        if method == 'gpd':
            exceed_prob = self.stats_left['exceedance_probability']
            var = self.gpd_left.var(confidence, exceed_prob)
            # Transform back (we negated for fitting)
            return var

        elif method == 'gev':
            # Use minima GEV
            return_period = 1 / (1 - confidence)
            return -self.gev_minima.return_level(return_period)

    def expected_shortfall(self, confidence: float = 0.99) -> float:
        """Compute Expected Shortfall (CVaR)."""
        exceed_prob = self.stats_left['exceedance_probability']
        return self.gpd_left.expected_shortfall(confidence, exceed_prob)

    def return_level(self, return_period: float, tail: str = 'left') -> float:
        """
        Compute return level for extreme events.

        Args:
            return_period: Expected time between events of this magnitude
            tail: 'left' (losses) or 'right' (gains)

        Returns:
            Return level (magnitude of extreme event)
        """
        if tail == 'left':
            return -self.gev_minima.return_level(return_period)
        else:
            return self.gev_maxima.return_level(return_period)

    def tail_comparison_to_gaussian(self, threshold: float = 3.0) -> Dict:
        """
        Compare tail probabilities to Gaussian assumption.

        Shows how much the Gaussian underestimates tail risk.
        """
        std = np.std(self.data)
        gaussian_prob = 2 * (1 - stats.norm.cdf(threshold))

        # Empirical tail probability
        empirical_prob = np.mean(np.abs(self.data) > threshold * std)

        # GPD-based estimate
        x = threshold * std
        gpd_prob_left = self.gpd_left.tail_probability(x)
        gpd_prob_right = self.gpd_right.tail_probability(x)
        gpd_prob = gpd_prob_left + gpd_prob_right

        return {
            'threshold_sigma': threshold,
            'gaussian_probability': gaussian_prob,
            'empirical_probability': empirical_prob,
            'gpd_probability': gpd_prob,
            'underestimation_factor_empirical': empirical_prob / gaussian_prob,
            'underestimation_factor_gpd': gpd_prob / gaussian_prob
        }

    def comprehensive_report(self) -> Dict:
        """Generate comprehensive tail risk report."""
        return {
            'n_observations': self.n,
            'tail_indices': {
                'left': self.gpd_left.xi,
                'right': self.gpd_right.xi
            },
            'thresholds': {
                'left': self.stats_left['threshold'],
                'right': self.stats_right['threshold']
            },
            'var_99': self.var(0.99),
            'var_995': self.var(0.995),
            'var_999': self.var(0.999),
            'es_99': self.expected_shortfall(0.99),
            'return_levels': {
                '10_year': self.return_level(252 * 10),
                '100_year': self.return_level(252 * 100)
            },
            'gaussian_comparison': {
                '3sigma': self.tail_comparison_to_gaussian(3),
                '4sigma': self.tail_comparison_to_gaussian(4),
                '5sigma': self.tail_comparison_to_gaussian(5)
            }
        }


def mean_excess_plot(data: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Dict:
    """
    Compute mean excess function for threshold selection.

    The mean excess function e(u) = E[X - u | X > u] is:
    - Linear in u for GPD
    - Slope = ξ/(1-ξ)
    - Intercept = σ/(1-ξ)

    A linear region indicates where GPD is appropriate.
    """
    data = np.sort(data)

    if thresholds is None:
        thresholds = np.percentile(data, np.linspace(50, 99, 50))

    mean_excesses = []
    n_exceedances = []

    for u in thresholds:
        exceedances = data[data > u] - u
        if len(exceedances) > 5:
            mean_excesses.append(np.mean(exceedances))
            n_exceedances.append(len(exceedances))
        else:
            mean_excesses.append(np.nan)
            n_exceedances.append(0)

    return {
        'thresholds': thresholds,
        'mean_excess': np.array(mean_excesses),
        'n_exceedances': np.array(n_exceedances)
    }


def hill_plot(data: np.ndarray, k_range: Optional[np.ndarray] = None) -> Dict:
    """
    Compute Hill estimator for various k values.

    The Hill estimator uses the k largest order statistics:
    α_Hill = (1/k) * Σ log(X_{(i)} / X_{(k+1)})

    where X_{(1)} ≥ X_{(2)} ≥ ... are order statistics.

    Plot should stabilize at true tail index.
    """
    data = np.sort(np.abs(data))[::-1]  # Descending order
    n = len(data)

    if k_range is None:
        k_range = np.arange(10, n // 5)

    hill_estimates = []

    for k in k_range:
        if k < n:
            log_ratios = np.log(data[:k] / data[k])
            hill = 1 / np.mean(log_ratios)
            hill_estimates.append(hill)
        else:
            hill_estimates.append(np.nan)

    return {
        'k_values': k_range,
        'hill_estimates': np.array(hill_estimates)
    }
