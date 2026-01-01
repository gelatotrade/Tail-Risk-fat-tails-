"""
Comprehensive Risk Metrics with Fat-Tail Adjustments
=====================================================

Standard risk metrics (VaR, volatility) assume Gaussian returns.
This module provides fat-tail adjusted versions that properly
account for extreme event probabilities.

Key Metrics:
1. Fat-Tail VaR: Using EVT/GPD
2. Expected Shortfall: Average loss beyond VaR
3. Tail Risk Parity: Allocate based on tail risk, not volatility
4. Maximum Drawdown Analytics
5. Omega Ratio: Gain/loss probability ratio
6. Tail Dependence: Correlation in extremes
"""

import numpy as np
from scipy import stats, optimize
from typing import Tuple, Optional, Dict, List
import warnings


class TailRiskMetrics:
    """
    Comprehensive tail risk metrics calculator.

    Provides both parametric (EVT-based) and non-parametric
    estimates of tail risk measures.
    """

    def __init__(self, returns: np.ndarray, annualization: int = 252):
        """
        Initialize with return series.

        Args:
            returns: Array of returns
            annualization: Trading days per year
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)
        self.annualization = annualization

        # Precompute basic statistics
        self.mean = np.mean(returns)
        self.std = np.std(returns)
        self.skew = stats.skew(returns)
        self.kurtosis = stats.kurtosis(returns)

    def var_historical(self, confidence: float = 0.99) -> float:
        """Historical simulation VaR."""
        return -np.percentile(self.returns, (1 - confidence) * 100)

    def var_parametric_gaussian(self, confidence: float = 0.99) -> float:
        """Parametric VaR assuming Gaussian."""
        z = stats.norm.ppf(1 - confidence)
        return -(self.mean + z * self.std)

    def var_cornish_fisher(self, confidence: float = 0.99) -> float:
        """
        Cornish-Fisher VaR with skewness/kurtosis adjustment.

        Expands the Gaussian quantile to account for higher moments.
        """
        z = stats.norm.ppf(1 - confidence)
        S = self.skew
        K = self.kurtosis

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * S / 6
            + (z**3 - 3*z) * K / 24
            - (2*z**3 - 5*z) * S**2 / 36
        )

        return -(self.mean + z_cf * self.std)

    def var_evt(self, confidence: float = 0.99, threshold_quantile: float = 0.95) -> float:
        """
        EVT-based VaR using GPD for tail.

        More accurate for high confidence levels.
        """
        from .extreme_value import GeneralizedParetoDistribution

        # Fit GPD to left tail
        neg_returns = -self.returns
        gpd, stats_dict = GeneralizedParetoDistribution.fit_pot(
            neg_returns, quantile=threshold_quantile
        )

        return gpd.var(confidence, stats_dict['exceedance_probability'])

    def expected_shortfall(self, confidence: float = 0.99,
                          method: str = 'historical') -> float:
        """
        Expected Shortfall (Conditional VaR).

        ES = E[Loss | Loss > VaR]

        More coherent than VaR for fat-tailed distributions.
        """
        if method == 'historical':
            var = self.var_historical(confidence)
            tail_losses = -self.returns[-self.returns < -var]
            return np.mean(tail_losses) if len(tail_losses) > 0 else var

        elif method == 'gaussian':
            alpha = 1 - confidence
            z = stats.norm.ppf(alpha)
            es = -self.mean + self.std * stats.norm.pdf(z) / alpha
            return es

        elif method == 'cornish_fisher':
            # Approximate using CF-adjusted quantile
            var = self.var_cornish_fisher(confidence)
            # Use kernel density for tail
            tail_losses = -self.returns[-self.returns < -var]
            return np.mean(tail_losses) if len(tail_losses) > 0 else var

    def tail_ratio(self, threshold_sigma: float = 2.0) -> Dict:
        """
        Compute tail probability ratios vs Gaussian.

        Shows how much fatter the tails are compared to normal.
        """
        threshold = threshold_sigma * self.std

        # Empirical probabilities
        left_tail = np.mean(self.returns < -threshold)
        right_tail = np.mean(self.returns > threshold)

        # Gaussian probabilities
        gauss_tail = 1 - stats.norm.cdf(threshold_sigma)

        return {
            'threshold_sigma': threshold_sigma,
            'left_tail_prob': left_tail,
            'right_tail_prob': right_tail,
            'gaussian_tail_prob': gauss_tail,
            'left_ratio': left_tail / gauss_tail if gauss_tail > 0 else np.inf,
            'right_ratio': right_tail / gauss_tail if gauss_tail > 0 else np.inf,
            'asymmetry': left_tail / right_tail if right_tail > 0 else np.inf
        }

    def maximum_drawdown(self) -> Dict:
        """
        Compute maximum drawdown and related metrics.

        Drawdown is the peak-to-trough decline.
        """
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        max_dd = np.min(drawdowns)
        max_dd_end = np.argmin(drawdowns)
        max_dd_start = np.argmax(cumulative[:max_dd_end+1])

        # Time to recovery
        if max_dd_end < len(cumulative) - 1:
            recovery_idx = np.argmax(cumulative[max_dd_end:] >= running_max[max_dd_end])
            recovery_time = recovery_idx if recovery_idx > 0 else np.inf
        else:
            recovery_time = np.inf

        return {
            'max_drawdown': abs(max_dd),
            'max_dd_start': max_dd_start,
            'max_dd_end': max_dd_end,
            'max_dd_duration': max_dd_end - max_dd_start,
            'recovery_time': recovery_time,
            'current_drawdown': abs(drawdowns[-1]),
            'drawdown_series': drawdowns
        }

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega ratio: probability-weighted gain/loss ratio.

        Ω(r) = ∫_{r}^{∞} (1 - F(x)) dx / ∫_{-∞}^{r} F(x) dx

        Unlike Sharpe, captures full distribution shape.
        """
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]

        if np.sum(losses) == 0:
            return np.inf

        return np.sum(gains) / np.sum(losses)

    def sortino_ratio(self, target: float = 0.0) -> float:
        """
        Sortino ratio: like Sharpe but using downside deviation.

        Only penalizes negative volatility.
        """
        excess_return = np.mean(self.returns) - target
        downside_returns = self.returns[self.returns < target]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = np.sqrt(np.mean((downside_returns - target)**2))

        return excess_return / downside_std * np.sqrt(self.annualization)

    def calmar_ratio(self) -> float:
        """
        Calmar ratio: annual return / max drawdown.

        Measures return per unit of maximum loss.
        """
        annual_return = np.mean(self.returns) * self.annualization
        max_dd = self.maximum_drawdown()['max_drawdown']

        return annual_return / max_dd if max_dd > 0 else np.inf

    def tail_dependence(self, other_returns: np.ndarray,
                       quantile: float = 0.95) -> Dict:
        """
        Compute tail dependence between two return series.

        Measures how correlated assets are in extreme scenarios.
        Standard correlation underestimates this.
        """
        other = np.asarray(other_returns)
        if len(other) != len(self.returns):
            raise ValueError("Arrays must have same length")

        # Upper tail dependence
        threshold_self = np.percentile(self.returns, quantile * 100)
        threshold_other = np.percentile(other, quantile * 100)

        both_upper = np.mean(
            (self.returns > threshold_self) & (other > threshold_other)
        )
        marginal_upper = 1 - quantile

        upper_dependence = both_upper / marginal_upper if marginal_upper > 0 else 0

        # Lower tail dependence
        threshold_self_low = np.percentile(self.returns, (1 - quantile) * 100)
        threshold_other_low = np.percentile(other, (1 - quantile) * 100)

        both_lower = np.mean(
            (self.returns < threshold_self_low) & (other < threshold_other_low)
        )

        lower_dependence = both_lower / marginal_upper if marginal_upper > 0 else 0

        # Standard correlation for comparison
        correlation = np.corrcoef(self.returns, other)[0, 1]

        return {
            'upper_tail_dependence': upper_dependence,
            'lower_tail_dependence': lower_dependence,
            'standard_correlation': correlation,
            'tail_asymmetry': lower_dependence - upper_dependence
        }


class RollingRiskMetrics:
    """
    Compute rolling window risk metrics to detect regime changes.
    """

    def __init__(self, returns: np.ndarray, window: int = 60):
        """
        Initialize rolling calculator.

        Args:
            returns: Array of returns
            window: Rolling window size
        """
        self.returns = np.asarray(returns)
        self.window = window
        self.n = len(returns)

    def rolling_var(self, confidence: float = 0.99) -> np.ndarray:
        """Compute rolling VaR."""
        var = np.full(self.n, np.nan)

        for i in range(self.window, self.n):
            window_returns = self.returns[i-self.window:i]
            var[i] = -np.percentile(window_returns, (1 - confidence) * 100)

        return var

    def rolling_expected_shortfall(self, confidence: float = 0.99) -> np.ndarray:
        """Compute rolling Expected Shortfall."""
        es = np.full(self.n, np.nan)

        for i in range(self.window, self.n):
            window_returns = self.returns[i-self.window:i]
            var = -np.percentile(window_returns, (1 - confidence) * 100)
            tail_losses = -window_returns[window_returns < -var]
            es[i] = np.mean(tail_losses) if len(tail_losses) > 0 else var

        return es

    def rolling_kurtosis(self) -> np.ndarray:
        """Compute rolling excess kurtosis."""
        kurt = np.full(self.n, np.nan)

        for i in range(self.window, self.n):
            window_returns = self.returns[i-self.window:i]
            kurt[i] = stats.kurtosis(window_returns)

        return kurt

    def rolling_skewness(self) -> np.ndarray:
        """Compute rolling skewness."""
        skew = np.full(self.n, np.nan)

        for i in range(self.window, self.n):
            window_returns = self.returns[i-self.window:i]
            skew[i] = stats.skew(window_returns)

        return skew

    def rolling_tail_index(self, method: str = 'hill') -> np.ndarray:
        """
        Compute rolling tail index estimate.

        Lower values = fatter tails = higher tail risk.
        """
        from ..physics.levy_flight import estimate_tail_index

        alpha = np.full(self.n, np.nan)

        for i in range(self.window, self.n):
            window_returns = self.returns[i-self.window:i]
            try:
                alpha[i] = estimate_tail_index(window_returns, method)
            except:
                alpha[i] = np.nan

        return alpha

    def compute_all(self) -> Dict[str, np.ndarray]:
        """Compute all rolling metrics."""
        return {
            'var_99': self.rolling_var(0.99),
            'var_95': self.rolling_var(0.95),
            'es_99': self.rolling_expected_shortfall(0.99),
            'kurtosis': self.rolling_kurtosis(),
            'skewness': self.rolling_skewness(),
            'tail_index': self.rolling_tail_index()
        }


class TailRiskParity:
    """
    Portfolio allocation based on tail risk contribution.

    Unlike standard risk parity (volatility-based), this uses
    Expected Shortfall to properly account for fat tails.
    """

    def __init__(self, returns: np.ndarray, confidence: float = 0.99):
        """
        Initialize with multi-asset return matrix.

        Args:
            returns: Array of shape (n_obs, n_assets)
            confidence: Confidence level for ES calculation
        """
        self.returns = np.asarray(returns)
        if self.returns.ndim == 1:
            self.returns = self.returns.reshape(-1, 1)

        self.n_obs, self.n_assets = self.returns.shape
        self.confidence = confidence

    def marginal_es(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute marginal Expected Shortfall for each asset.

        Measures how much each asset contributes to portfolio ES.
        """
        weights = np.asarray(weights)
        portfolio_returns = self.returns @ weights

        # Find tail observations
        var = -np.percentile(portfolio_returns, (1 - self.confidence) * 100)
        tail_mask = portfolio_returns < -var

        if np.sum(tail_mask) == 0:
            return np.zeros(self.n_assets)

        # Marginal contribution: average asset return in tail scenarios
        marginal = -np.mean(self.returns[tail_mask], axis=0)

        return marginal

    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute risk contribution of each asset.

        RC_i = w_i * ∂ES/∂w_i
        """
        marginal = self.marginal_es(weights)
        return weights * marginal

    def tail_risk_parity_weights(self, target_risk: Optional[float] = None) -> np.ndarray:
        """
        Compute tail risk parity weights.

        Each asset contributes equally to portfolio ES.

        Args:
            target_risk: Optional target portfolio ES

        Returns:
            Optimal weights
        """
        def objective(w):
            rc = self.risk_contribution(w)
            # Penalize deviation from equal risk contribution
            avg_rc = np.mean(rc)
            return np.sum((rc - avg_rc)**2)

        # Constraints: weights sum to 1, all positive
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0.01, 1.0) for _ in range(self.n_assets)]

        # Initial guess: equal weight
        w0 = np.ones(self.n_assets) / self.n_assets

        result = optimize.minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x


def compute_risk_3d_coordinates(returns: np.ndarray,
                               window: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D risk phase space.

    Coordinates:
    - X: VaR level (normalized)
    - Y: Expected Shortfall / VaR ratio (tail heaviness)
    - Z: Kurtosis (distribution shape)

    High X, high Y, high Z = extreme tail risk regime
    """
    n = len(returns)
    rolling = RollingRiskMetrics(returns, window)

    var = rolling.rolling_var(0.99)
    es = rolling.rolling_expected_shortfall(0.99)
    kurt = rolling.rolling_kurtosis()

    # Normalize
    X = (var - np.nanmin(var)) / (np.nanmax(var) - np.nanmin(var) + 1e-10)
    Y = es / (var + 1e-10)  # ES/VaR ratio
    Y = (Y - np.nanmin(Y)) / (np.nanmax(Y) - np.nanmin(Y) + 1e-10)
    Z = (kurt - np.nanmin(kurt)) / (np.nanmax(kurt) - np.nanmin(kurt) + 1e-10)

    return X, Y, Z
