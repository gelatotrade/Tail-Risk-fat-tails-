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

import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

        # EM algorithm (Baum-Welch / Hamilton-Kim)
        log_likelihood_old = -np.inf
        filtered, predicted, log_likelihood = self._forward(returns)
        iteration = -1

        for iteration in range(max_iter):
            # E-step: filtered/smoothed probabilities with current parameters
            self.filtered_probs = filtered
            smoothed, transitions = self._backward(filtered, predicted)

            # M-step: update parameters
            self._m_step(returns, smoothed, transitions)

            # Check convergence (log-likelihood of the updated parameters)
            filtered, predicted, log_likelihood = self._forward(returns)
            if abs(log_likelihood - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood

        self.filtered_probs = filtered
        self._sort_states_by_volatility()

        return {
            'transition_matrix': self.transition_matrix,
            'state_means': self.state_means,
            'state_stds': self.state_stds,
            'log_likelihood': log_likelihood,
            'n_iter': iteration + 1
        }

    def _state_likelihoods(self, returns: np.ndarray) -> np.ndarray:
        """Gaussian state densities f_j(y_t) as an (n, k) array."""
        returns = np.asarray(returns, dtype=float)
        return stats.norm.pdf(returns[:, None], self.state_means[None, :],
                              self.state_stds[None, :])

    def _forward(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Hamilton filter.

        Returns:
            filtered P(S_t | Y_t), predicted P(S_t | Y_{t-1}) (both (n, k))
            and the log-likelihood log p(Y_n).
        """
        lik = self._state_likelihoods(returns)
        n, k = lik.shape
        # The recursion runs on Python floats: for k = 2..3 states this is
        # several times faster than per-step NumPy calls on tiny arrays.
        columns = self.transition_matrix.T.tolist()  # columns[j][i] = P_ij
        lik_rows = lik.tolist()
        log = math.log

        filtered = [None] * n
        predicted = [None] * n
        log_likelihood = 0.0

        # Initial distribution: uniform
        pred = [1.0 / k] * k
        for t in range(n):
            predicted[t] = pred
            joint = [p * l for p, l in zip(pred, lik_rows[t])]
            density = sum(joint)
            if 0.0 < density < math.inf:
                filt = [x / density for x in joint]
            else:
                filt = pred  # observation impossible under every state
            filtered[t] = filt
            log_likelihood += log(max(density, 1e-300))
            pred = [sum([f * p for f, p in zip(filt, col)]) for col in columns]

        return np.array(filtered), np.array(predicted), float(log_likelihood)

    def _backward(self, filtered: np.ndarray,
                  predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kim smoother.

        Returns:
            smoothed P(S_t | Y_n) as (n, k), and the expected transition
            counts sum_t P(S_{t-1}=i, S_t=j | Y_n) as (k, k).
        """
        n, k = filtered.shape
        P = self.transition_matrix
        rows = P.tolist()
        filt = filtered.tolist()
        pred = predicted.tolist()

        smoothed = [None] * n
        smoothed[-1] = filt[-1]
        ratio = [[0.0] * k for _ in range(n)]  # smoothed[t] / predicted[t]

        for t in range(n - 1, 0, -1):
            r = [s / p if p > 0 else 0.0 for s, p in zip(smoothed[t], pred[t])]
            ratio[t] = r
            prev = [f * sum([a * b for a, b in zip(row, r)])
                    for f, row in zip(filt[t - 1], rows)]
            total = sum(prev) or 1.0
            smoothed[t - 1] = [x / total for x in prev]

        ratio = np.array(ratio)
        # P(S_{t-1}=i, S_t=j | Y_n) = filtered[t-1,i] P_ij smoothed[t,j] / predicted[t,j]
        transitions = P * (filtered[:-1].T @ ratio[1:])

        return np.array(smoothed), transitions

    def _m_step(self, returns: np.ndarray, smoothed: np.ndarray,
                transitions: np.ndarray):
        """Update means, stds and transition matrix from E-step quantities."""
        returns = np.asarray(returns, dtype=float)
        weight = smoothed.sum(axis=0)

        self.state_means = (smoothed * returns[:, None]).sum(axis=0) / (weight + 1e-10)
        variance = ((smoothed * (returns[:, None] - self.state_means)**2).sum(axis=0)
                    / (weight + 1e-10))
        self.state_stds = np.sqrt(np.maximum(variance, 1e-6))

        row_sums = transitions.sum(axis=1, keepdims=True)
        k = self.n_states
        self.transition_matrix = np.where(row_sums > 0,
                                          transitions / np.where(row_sums > 0, row_sums, 1.0),
                                          1.0 / k)

    def _sort_states_by_volatility(self):
        """Relabel states so that state 0 has the lowest std (documented order)."""
        order = np.argsort(self.state_stds, kind='stable')
        if np.array_equal(order, np.arange(self.n_states)):
            return
        self.state_means = self.state_means[order]
        self.state_stds = self.state_stds[order]
        self.transition_matrix = self.transition_matrix[np.ix_(order, order)]
        if self.filtered_probs is not None:
            self.filtered_probs = self.filtered_probs[:, order]

    def _filter(self, returns: np.ndarray) -> np.ndarray:
        """Hamilton filter for state probabilities."""
        filtered, _, _ = self._forward(returns)
        return filtered

    def _update_parameters(self, returns: np.ndarray):
        """M-step: update model parameters (E-step with current parameters)."""
        filtered, predicted, _ = self._forward(returns)
        smoothed, transitions = self._backward(filtered, predicted)
        self._m_step(returns, smoothed, transitions)

    def _smooth(self, returns: np.ndarray) -> np.ndarray:
        """Kim smoother for smoothed probabilities."""
        filtered, predicted, _ = self._forward(returns)
        smoothed, _ = self._backward(filtered, predicted)
        return smoothed

    def _log_likelihood(self, returns: np.ndarray) -> float:
        """Compute log-likelihood."""
        _, _, log_likelihood = self._forward(returns)
        return log_likelihood

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
        valid = ~np.isnan(vol)

        regimes = np.zeros(self.n, dtype=int)
        if not np.any(valid):
            return regimes  # series shorter than the volatility window

        # Thresholds
        low_threshold = np.nanpercentile(vol, 100 - vol_percentile)
        high_threshold = np.nanpercentile(vol, vol_percentile)

        regimes[vol > high_threshold] = 2
        regimes[(vol > low_threshold) & (vol <= high_threshold)] = 1

        # Warm-up (no full window yet): use the regime of the first full window
        first = int(np.argmax(valid))
        regimes[:first] = regimes[first]

        return regimes


class RegimeAwareTailRisk:
    """
    Tail risk metrics that adapt to current volatility regime.

    The key insight: tail risk in high-vol regime is MUCH higher
    than unconditional estimates suggest.
    """

    def __init__(self, returns: np.ndarray, window: int = 60,
                 vol_percentile: float = 75):
        """
        Initialize regime-aware analyzer.

        Args:
            returns: Array of returns
            window: Window for regime detection (stored; the realized-vol
                classification uses its own 20-day window)
            vol_percentile: Percentile of realized volatility above which a
                day counts as HIGH_VOL (below 100 - vol_percentile: LOW_VOL),
                passed to VolatilityClusteringDetector.detect_vol_regime
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)
        self.window = window
        self.vol_percentile = vol_percentile

        # Fit regime model
        self._detect_regimes()

    def _detect_regimes(self):
        """Detect volatility regimes."""
        # Use simple volatility-based detection
        detector = VolatilityClusteringDetector(self.returns)
        self.regimes = detector.detect_vol_regime(self.vol_percentile)

        # Also fit Markov model (states are sorted by volatility after fit)
        self.markov = MarkovRegimeSwitching(n_states=3)
        try:
            self.markov.fit(self.returns)
        except Exception as exc:
            warnings.warn(f"Markov regime fit failed: {exc}")
            self.markov.filtered_probs = None

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


def _trailing_std(returns: np.ndarray, length: int) -> np.ndarray:
    """out[j] = np.std(returns[j-length:j]) for j >= length, NaN before."""
    out = np.full(len(returns) + 1, np.nan)
    if len(returns) >= length:
        out[length:] = sliding_window_view(returns, length).std(axis=1)
    return out


def _rowwise_abs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|Pearson correlation| of matching rows (NaN for constant rows, like np.corrcoef)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.abs(np.clip(corr, -1, 1))


def compute_regime_coordinates(returns: np.ndarray,
                              window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D regime phase space.

    Coordinates:
    - X: Current volatility level (normalized)
    - Y: Regime persistence (autocorrelation of vol)
    - Z: Regime transition probability

    High X, low Y, high Z = unstable high-vol regime (danger)

    The first 2 * window entries have no estimate and are 0. X and Z are
    min-max normalized over the estimated entries only. window must be >= 8.
    """
    window = int(window)
    if window < 8:
        raise ValueError("window must be >= 8 (sub-windows use window // 8 steps)")

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    X = np.zeros(n)  # Volatility level
    Y = np.zeros(n)  # Persistence
    Z = np.zeros(n)  # Transition probability

    start = 2 * window
    if start >= n:
        return X, Y, Z

    idx = np.arange(start, n)

    # Current volatility relative to the expanding (all-history) volatility
    centered = returns - returns.mean()
    count = np.arange(1, n + 1)
    mean_c = np.cumsum(centered) / count
    var_c = np.cumsum(centered**2) / count - mean_c**2
    historical_vol = np.sqrt(np.maximum(var_c[idx - 1], 0))  # std(returns[:i])
    X[idx] = _trailing_std(returns, window)[idx] / historical_vol

    # Volatility persistence: lag-1 correlation of half-window vols sampled
    # every window // 4 days over the last window days
    step = window // 4
    offsets = np.arange(0, window, step)
    if len(offsets) > 2:
        vol_series = _trailing_std(returns, window // 2)[idx[:, None] - window + offsets]
        Y[idx] = _rowwise_abs_corr(vol_series[:, :-1], vol_series[:, 1:])

    # Transition probability (volatility change likelihood): mean absolute
    # change of quarter-window vols sampled every window // 8 days
    step = window // 8
    offsets = np.arange(0, window, step)
    vols = _trailing_std(returns, window // 4)[idx[:, None] - window + offsets]
    Z[idx] = np.mean(np.abs(np.diff(vols, axis=1)), axis=1) / (np.mean(vols, axis=1) + 1e-10)

    # Normalize over the estimated entries
    def _min_max(values):
        valid = values[idx]
        return (values - np.min(valid)) / (np.max(valid) - np.min(valid) + 1e-10)

    X[idx] = _min_max(X)[idx]
    Y = np.clip(Y, 0, 1)
    Z[idx] = _min_max(Z)[idx]

    return X, Y, Z
