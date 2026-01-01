"""
Volatility Regime Detection for Tail Risk
==========================================

Financial markets exhibit distinct volatility regimes:
1. Low volatility (calm): Normal distribution, low tail risk
2. High volatility (turbulent): Fat tails, high tail risk
3. Crisis (extreme): Very fat tails, Black Swan territory

This module provides methods to:
- Detect current regime
- Predict regime transitions
- Adjust risk metrics for regime
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, List
import warnings


class MarkovRegimeSwitching:
    """
    Markov Regime Switching Model for volatility.

    States:
    - State 0: Low volatility regime
    - State 1: High volatility regime

    Transition probabilities:
    P(S_t = j | S_{t-1} = i) = p_ij
    """

    def __init__(self, n_states: int = 2):
        """
        Initialize regime switching model.

        Args:
            n_states: Number of regimes (default 2: low/high vol)
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.state_means = None
        self.state_stds = None
        self.filtered_probs = None

    def fit(self, returns: np.ndarray, max_iter: int = 100,
            tol: float = 1e-4) -> Dict:
        """
        Fit regime switching model using EM algorithm.

        Args:
            returns: Array of returns
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Dictionary with fitted parameters
        """
        returns = np.asarray(returns)
        n = len(returns)
        k = self.n_states

        # Initialize parameters
        # Means: sorted by volatility
        sorted_returns = np.sort(returns)
        quantiles = np.linspace(0, 1, k + 1)
        self.state_means = np.array([
            np.mean(sorted_returns[int(quantiles[i]*n):int(quantiles[i+1]*n)])
            for i in range(k)
        ])

        # Stds: increasing
        self.state_stds = np.array([
            np.std(returns) * (0.5 + i * 0.5)
            for i in range(k)
        ])

        # Transition matrix: high persistence
        self.transition_matrix = np.full((k, k), 0.1 / (k - 1))
        np.fill_diagonal(self.transition_matrix, 0.9)

        # EM algorithm
        log_likelihood_old = -np.inf

        for iteration in range(max_iter):
            # E-step: compute filtered probabilities
            self.filtered_probs = self._filter(returns)

            # M-step: update parameters
            self._update_parameters(returns)

            # Check convergence
            log_likelihood = self._log_likelihood(returns)
            if abs(log_likelihood - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood

        return {
            'transition_matrix': self.transition_matrix,
            'state_means': self.state_means,
            'state_stds': self.state_stds,
            'log_likelihood': log_likelihood,
            'n_iter': iteration + 1
        }

    def _filter(self, returns: np.ndarray) -> np.ndarray:
        """Hamilton filter for state probabilities."""
        n = len(returns)
        k = self.n_states

        # Filtered probabilities
        probs = np.zeros((n, k))

        # Initial distribution: stationary
        probs[0] = 1 / k

        for t in range(1, n):
            # Prediction: P(S_t | Y_{t-1})
            pred = self.transition_matrix.T @ probs[t-1]

            # Update: P(S_t | Y_t)
            likelihoods = np.array([
                stats.norm.pdf(returns[t], self.state_means[j], self.state_stds[j])
                for j in range(k)
            ])

            probs[t] = pred * likelihoods
            probs[t] /= np.sum(probs[t]) + 1e-10

        return probs

    def _update_parameters(self, returns: np.ndarray):
        """M-step: update model parameters."""
        n = len(returns)
        k = self.n_states

        # Smooth probabilities
        smoothed = self._smooth(returns)

        # Update means
        for j in range(k):
            weight = smoothed[:, j]
            self.state_means[j] = np.sum(weight * returns) / (np.sum(weight) + 1e-10)

        # Update stds
        for j in range(k):
            weight = smoothed[:, j]
            variance = np.sum(weight * (returns - self.state_means[j])**2) / (np.sum(weight) + 1e-10)
            self.state_stds[j] = np.sqrt(max(variance, 1e-6))

        # Update transition matrix
        for i in range(k):
            for j in range(k):
                num = 0
                den = 0
                for t in range(1, n):
                    num += smoothed[t-1, i] * self.transition_matrix[i, j] * \
                           stats.norm.pdf(returns[t], self.state_means[j], self.state_stds[j]) * \
                           smoothed[t, j]
                    den += smoothed[t-1, i]

                self.transition_matrix[i, j] = num / (den + 1e-10)

            # Normalize
            self.transition_matrix[i] /= np.sum(self.transition_matrix[i]) + 1e-10

    def _smooth(self, returns: np.ndarray) -> np.ndarray:
        """Kim smoother for smoothed probabilities."""
        n = len(returns)
        k = self.n_states

        filtered = self._filter(returns)
        smoothed = np.zeros((n, k))
        smoothed[-1] = filtered[-1]

        for t in range(n - 2, -1, -1):
            for j in range(k):
                smoothed[t, j] = filtered[t, j] * np.sum(
                    self.transition_matrix[j] * smoothed[t+1] /
                    (self.transition_matrix.T @ filtered[t] + 1e-10)
                )

        return smoothed

    def _log_likelihood(self, returns: np.ndarray) -> float:
        """Compute log-likelihood."""
        n = len(returns)
        k = self.n_states
        ll = 0

        probs = np.ones(k) / k

        for t in range(n):
            likelihoods = np.array([
                stats.norm.pdf(returns[t], self.state_means[j], self.state_stds[j])
                for j in range(k)
            ])

            ll += np.log(np.sum(probs * likelihoods) + 1e-10)
            probs = self.transition_matrix.T @ (probs * likelihoods)
            probs /= np.sum(probs) + 1e-10

        return ll

    def current_regime(self) -> int:
        """Return most likely current regime."""
        if self.filtered_probs is None:
            return 0
        return np.argmax(self.filtered_probs[-1])

    def regime_probability(self) -> np.ndarray:
        """Return current regime probabilities."""
        if self.filtered_probs is None:
            return np.ones(self.n_states) / self.n_states
        return self.filtered_probs[-1]

    def forecast_regime(self, steps: int = 5) -> np.ndarray:
        """
        Forecast regime probabilities.

        Args:
            steps: Number of steps ahead

        Returns:
            Array of shape (steps, n_states) with probabilities
        """
        if self.filtered_probs is None:
            return np.ones((steps, self.n_states)) / self.n_states

        forecast = np.zeros((steps, self.n_states))
        current = self.filtered_probs[-1]

        for s in range(steps):
            current = self.transition_matrix.T @ current
            forecast[s] = current

        return forecast


class VolatilityClusteringDetector:
    """
    Detect volatility clustering using GARCH-style analysis.

    Volatility clustering: large changes tend to follow large changes.
    This is captured by high autocorrelation in squared returns.
    """

    def __init__(self, returns: np.ndarray):
        """
        Initialize with return series.

        Args:
            returns: Array of returns
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)

    def ljung_box_test(self, lags: int = 20) -> Dict:
        """
        Ljung-Box test for autocorrelation in squared returns.

        High test statistic / low p-value indicates clustering.
        """
        squared = self.returns**2

        # Compute autocorrelations
        autocorr = np.correlate(squared - np.mean(squared),
                               squared - np.mean(squared), mode='full')
        autocorr = autocorr[self.n-1:] / autocorr[self.n-1]

        # Ljung-Box statistic
        Q = self.n * (self.n + 2) * np.sum(
            autocorr[1:lags+1]**2 / (self.n - np.arange(1, lags+1))
        )

        # p-value (chi-squared with 'lags' degrees of freedom)
        p_value = 1 - stats.chi2.cdf(Q, lags)

        return {
            'test_statistic': Q,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'autocorrelations': autocorr[1:lags+1]
        }

    def rolling_realized_volatility(self, window: int = 20) -> np.ndarray:
        """Compute rolling realized volatility."""
        vol = np.full(self.n, np.nan)

        for i in range(window, self.n):
            vol[i] = np.std(self.returns[i-window:i]) * np.sqrt(252)

        return vol

    def detect_vol_regime(self, vol_percentile: float = 75) -> np.ndarray:
        """
        Classify each period into volatility regime.

        Returns:
            Array with regime labels (0 = low, 1 = medium, 2 = high)
        """
        vol = self.rolling_realized_volatility()

        # Thresholds
        low_threshold = np.nanpercentile(vol, 100 - vol_percentile)
        high_threshold = np.nanpercentile(vol, vol_percentile)

        regimes = np.zeros(self.n, dtype=int)
        regimes[vol > high_threshold] = 2
        regimes[(vol > low_threshold) & (vol <= high_threshold)] = 1

        return regimes


class RegimeAwareTailRisk:
    """
    Tail risk metrics that adapt to current volatility regime.

    The key insight: tail risk in high-vol regime is MUCH higher
    than unconditional estimates suggest.
    """

    def __init__(self, returns: np.ndarray, window: int = 60):
        """
        Initialize regime-aware analyzer.

        Args:
            returns: Array of returns
            window: Window for regime detection
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)
        self.window = window

        # Fit regime model
        self._detect_regimes()

    def _detect_regimes(self):
        """Detect volatility regimes."""
        # Use simple volatility-based detection
        detector = VolatilityClusteringDetector(self.returns)
        self.regimes = detector.detect_vol_regime()

        # Also fit Markov model
        self.markov = MarkovRegimeSwitching(n_states=3)
        try:
            self.markov.fit(self.returns)
        except:
            pass

    def current_regime(self) -> Dict:
        """Get current regime information."""
        regime_labels = ['LOW_VOL', 'MEDIUM_VOL', 'HIGH_VOL']

        simple_regime = self.regimes[-1]
        markov_regime = self.markov.current_regime() if self.markov.filtered_probs is not None else 0
        markov_probs = self.markov.regime_probability() if self.markov.filtered_probs is not None else np.array([0.33, 0.34, 0.33])

        return {
            'simple_regime': regime_labels[simple_regime],
            'markov_regime': regime_labels[markov_regime],
            'regime_probabilities': {
                'low_vol': markov_probs[0],
                'medium_vol': markov_probs[1] if len(markov_probs) > 1 else 0,
                'high_vol': markov_probs[2] if len(markov_probs) > 2 else markov_probs[-1]
            }
        }

    def regime_conditional_var(self, confidence: float = 0.99) -> Dict:
        """
        Compute VaR conditional on each regime.

        Shows how tail risk varies dramatically by regime.
        """
        results = {}
        regime_labels = ['LOW_VOL', 'MEDIUM_VOL', 'HIGH_VOL']

        for r in range(3):
            regime_returns = self.returns[self.regimes == r]
            if len(regime_returns) > 10:
                var = -np.percentile(regime_returns, (1 - confidence) * 100)
                results[regime_labels[r]] = {
                    'var': var,
                    'n_observations': len(regime_returns)
                }
            else:
                results[regime_labels[r]] = {'var': np.nan, 'n_observations': len(regime_returns)}

        # Unconditional for comparison
        results['UNCONDITIONAL'] = {
            'var': -np.percentile(self.returns, (1 - confidence) * 100),
            'n_observations': len(self.returns)
        }

        return results

    def regime_transition_risk(self) -> Dict:
        """
        Analyze risk around regime transitions.

        Transitions from low to high vol are particularly dangerous.
        """
        transitions = np.diff(self.regimes)

        # Find transitions
        up_transitions = np.where(transitions > 0)[0]  # Increasing vol
        down_transitions = np.where(transitions < 0)[0]  # Decreasing vol

        # Returns following transitions
        up_returns = []
        down_returns = []

        for idx in up_transitions:
            if idx + 5 < self.n:
                up_returns.extend(self.returns[idx+1:idx+6])

        for idx in down_transitions:
            if idx + 5 < self.n:
                down_returns.extend(self.returns[idx+1:idx+6])

        return {
            'n_up_transitions': len(up_transitions),
            'n_down_transitions': len(down_transitions),
            'post_up_mean_return': np.mean(up_returns) if up_returns else np.nan,
            'post_up_volatility': np.std(up_returns) if up_returns else np.nan,
            'post_down_mean_return': np.mean(down_returns) if down_returns else np.nan,
            'post_down_volatility': np.std(down_returns) if down_returns else np.nan
        }


def compute_regime_coordinates(returns: np.ndarray,
                              window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D regime phase space.

    Coordinates:
    - X: Current volatility level (normalized)
    - Y: Regime persistence (autocorrelation of vol)
    - Z: Regime transition probability

    High X, low Y, high Z = unstable high-vol regime (danger)
    """
    n = len(returns)

    X = np.zeros(n)  # Volatility level
    Y = np.zeros(n)  # Persistence
    Z = np.zeros(n)  # Transition probability

    for i in range(2 * window, n):
        # Current volatility
        current_vol = np.std(returns[i-window:i])
        historical_vol = np.std(returns[:i])
        X[i] = current_vol / historical_vol

        # Volatility persistence
        vol_series = [np.std(returns[j-window//2:j]) for j in range(i-window, i, window//4)]
        if len(vol_series) > 2:
            Y[i] = np.abs(np.corrcoef(vol_series[:-1], vol_series[1:])[0, 1])

        # Transition probability (volatility change likelihood)
        vol_changes = np.diff([np.std(returns[j-window//4:j]) for j in range(i-window, i, window//8)])
        Z[i] = np.mean(np.abs(vol_changes)) / (np.mean([np.std(returns[j-window//4:j])
                                                        for j in range(i-window, i, window//8)]) + 1e-10)

    # Normalize
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-10)
    Y = np.clip(Y, 0, 1)
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-10)

    return X, Y, Z
