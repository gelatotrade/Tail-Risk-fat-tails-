"""
Phase Transition Theory for Market Regime Changes
==================================================

Financial markets exhibit phase transitions similar to physical systems:
- Liquid water → Ice (first-order transition with latent heat)
- Ferromagnet losing magnetization (second-order transition)

Market equivalent:
- Stable regime → Crisis regime (crash)
- Low volatility → High volatility clustering
- Trending → Mean-reverting

Physics Background:
1. Order Parameter: Variable that distinguishes phases
   - In magnets: Magnetization M
   - In markets: Volatility σ, correlation ρ, or return regime

2. Control Parameter: External variable driving transition
   - In magnets: Temperature T
   - In markets: Investor sentiment, leverage, liquidity

3. Critical Point: Where transition occurs
   - Characterized by power-law correlations
   - "Critical slowing down" - system responds slowly near transition

4. Universality: Different systems share critical exponents
   - Ising model exponents appear in many systems
   - Market crashes may belong to same universality class

Key Insight:
Before crashes, markets show "critical slowing down" - autocorrelation
increases, and variance rises. This is a precursor signal!
"""

import numpy as np
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional, Dict, List
import warnings


class IsingMarketModel:
    """
    Ising model adapted for financial markets.

    In the Ising model:
    - Spins (σ_i = ±1) represent trader decisions (buy/sell)
    - Nearest-neighbor coupling J represents herding
    - External field h represents market-wide sentiment
    - Temperature T represents noise/uncertainty

    The partition function:
    Z = Σ exp(-H/T) where H = -J Σ σ_i σ_j - h Σ σ_i

    Phase transition occurs at critical temperature T_c = 2J/k
    Below T_c: Ordered phase (trending market, herding)
    Above T_c: Disordered phase (random walk, efficient market)
    """

    def __init__(self, n_agents: int = 100, J: float = 1.0,
                 h: float = 0.0, T: float = 2.5):
        """
        Initialize Ising market model.

        Args:
            n_agents: Number of traders (spins)
            J: Coupling strength (herding tendency)
            h: External field (market-wide sentiment)
            T: Temperature (noise level)
        """
        self.n_agents = n_agents
        self.J = J
        self.h = h
        self.T = T

        # Critical temperature for 2D Ising
        self.T_c = 2 * J / np.log(1 + np.sqrt(2))

        # Initialize random spins
        self.spins = np.random.choice([-1, 1], size=n_agents)

    def magnetization(self) -> float:
        """
        Compute magnetization (order parameter).

        M = <σ> = mean of spins

        |M| ≈ 1: Strong trend (all buying or selling)
        |M| ≈ 0: No trend (balanced market)
        """
        return np.mean(self.spins)

    def energy(self) -> float:
        """
        Compute Hamiltonian (energy).

        H = -J Σ σ_i σ_{i+1} - h Σ σ_i
        """
        # Nearest neighbor interaction (1D chain)
        interaction = -self.J * np.sum(self.spins[:-1] * self.spins[1:])
        field = -self.h * np.sum(self.spins)
        return interaction + field

    def susceptibility(self) -> float:
        """
        Compute magnetic susceptibility.

        χ = dM/dh = <M²> - <M>² / T

        High χ indicates sensitivity to perturbations (unstable market).
        This diverges at the critical point.
        """
        M = self.magnetization()
        return (1 - M**2) * self.n_agents / self.T

    def monte_carlo_step(self):
        """
        Perform one Monte Carlo step (Metropolis algorithm).

        This simulates market dynamics through opinion changes.
        """
        for _ in range(self.n_agents):
            # Random agent
            i = np.random.randint(self.n_agents)

            # Compute energy change if spin flips
            # Neighbors: periodic boundary conditions
            left = self.spins[(i - 1) % self.n_agents]
            right = self.spins[(i + 1) % self.n_agents]

            dE = 2 * self.spins[i] * (self.J * (left + right) + self.h)

            # Metropolis criterion
            if dE <= 0 or np.random.random() < np.exp(-dE / self.T):
                self.spins[i] *= -1

    def simulate(self, n_steps: int = 1000,
                 record_interval: int = 10) -> Dict[str, np.ndarray]:
        """
        Run simulation and record observables.

        Returns time series of magnetization, energy, returns.
        """
        magnetization = []
        energy = []
        returns = []

        prev_price = 100  # Initial price

        for step in range(n_steps):
            self.monte_carlo_step()

            if step % record_interval == 0:
                M = self.magnetization()
                E = self.energy()

                # Price change proportional to magnetization
                price_change = M * 0.01  # 1% max daily move
                new_price = prev_price * (1 + price_change)
                ret = np.log(new_price / prev_price)

                magnetization.append(M)
                energy.append(E)
                returns.append(ret)

                prev_price = new_price

        return {
            'magnetization': np.array(magnetization),
            'energy': np.array(energy),
            'returns': np.array(returns),
            'prices': 100 * np.exp(np.cumsum(returns))
        }

    def phase_diagram(self, T_range: np.ndarray,
                     n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute phase diagram (magnetization vs temperature).

        Shows the phase transition at T_c.
        """
        magnetizations = []
        susceptibilities = []

        for T in T_range:
            self.T = T
            self.spins = np.random.choice([-1, 1], size=self.n_agents)

            # Equilibrate
            for _ in range(100):
                self.monte_carlo_step()

            # Sample
            M_samples = []
            for _ in range(n_samples):
                self.monte_carlo_step()
                M_samples.append(abs(self.magnetization()))

            magnetizations.append(np.mean(M_samples))
            susceptibilities.append(np.var(M_samples) * self.n_agents / T)

        return {
            'temperature': T_range,
            'magnetization': np.array(magnetizations),
            'susceptibility': np.array(susceptibilities),
            'T_c': self.T_c
        }


class CriticalSlowingDownDetector:
    """
    Detect critical slowing down as a crash precursor.

    Near phase transitions, systems exhibit:
    1. Increased autocorrelation (memory lengthens)
    2. Increased variance (fluctuations grow)
    3. Skewness changes (asymmetry emerges)

    These are Early Warning Signals (EWS) for regime change.
    """

    def __init__(self, window: int = 50, detrend_window: int = 100):
        """
        Initialize detector.

        Args:
            window: Rolling window for EWS calculation
            detrend_window: Window for detrending
        """
        self.window = window
        self.detrend_window = detrend_window

    def detrend(self, data: np.ndarray) -> np.ndarray:
        """Remove trend using Gaussian smoothing."""
        trend = gaussian_filter1d(data, self.detrend_window)
        return data - trend

    def autocorrelation_lag1(self, data: np.ndarray) -> np.ndarray:
        """
        Compute rolling lag-1 autocorrelation.

        AC(1) → 1 near critical point (critical slowing down)
        """
        n = len(data)
        ac = np.zeros(n)

        for i in range(self.window, n):
            window_data = data[i-self.window:i]
            detrended = self.detrend(window_data)

            if np.std(detrended) > 0:
                ac[i] = np.corrcoef(detrended[:-1], detrended[1:])[0, 1]

        return ac

    def rolling_variance(self, data: np.ndarray) -> np.ndarray:
        """
        Compute rolling variance.

        Variance increases near critical point.
        """
        n = len(data)
        var = np.zeros(n)

        for i in range(self.window, n):
            window_data = data[i-self.window:i]
            detrended = self.detrend(window_data)
            var[i] = np.var(detrended)

        return var

    def rolling_skewness(self, data: np.ndarray) -> np.ndarray:
        """
        Compute rolling skewness.

        Skewness often becomes more negative before crashes.
        """
        n = len(data)
        skew = np.zeros(n)

        for i in range(self.window, n):
            window_data = data[i-self.window:i]
            skew[i] = stats.skew(window_data)

        return skew

    def kendall_tau(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Compute Kendall's tau to detect trend in EWS.

        Significant positive tau indicates approaching transition.
        """
        signal = signal[signal != 0]  # Remove zeros (padding)
        n = len(signal)
        time = np.arange(n)

        tau, p_value = stats.kendalltau(time, signal)
        return tau, p_value

    def compute_ews(self, returns: np.ndarray) -> Dict:
        """
        Compute all Early Warning Signals.

        Returns:
            Dictionary with AC(1), variance, skewness, and trend statistics
        """
        # Convert to cumulative (integrated) signal if returns
        if np.mean(np.abs(returns)) < 0.1:  # Likely returns
            integrated = np.cumsum(returns)
        else:
            integrated = returns

        # Compute EWS
        ac1 = self.autocorrelation_lag1(integrated)
        var = self.rolling_variance(integrated)
        skew = self.rolling_skewness(returns)

        # Trend tests
        ac1_trend = self.kendall_tau(ac1)
        var_trend = self.kendall_tau(var)

        # Composite indicator
        # Normalize and combine
        ac1_norm = (ac1 - np.nanmean(ac1)) / (np.nanstd(ac1) + 1e-10)
        var_norm = (var - np.nanmean(var)) / (np.nanstd(var) + 1e-10)
        composite = 0.5 * ac1_norm + 0.5 * var_norm

        return {
            'autocorrelation': ac1,
            'variance': var,
            'skewness': skew,
            'composite': composite,
            'ac1_trend': {'tau': ac1_trend[0], 'p_value': ac1_trend[1]},
            'var_trend': {'tau': var_trend[0], 'p_value': var_trend[1]},
            'warning_level': self._classify_warning(ac1_trend, var_trend)
        }

    def _classify_warning(self, ac1_trend: Tuple, var_trend: Tuple) -> str:
        """Classify warning level based on EWS trends."""
        ac1_tau, ac1_p = ac1_trend
        var_tau, var_p = var_trend

        if ac1_tau > 0.3 and ac1_p < 0.05 and var_tau > 0.3 and var_p < 0.05:
            return 'CRITICAL'
        elif (ac1_tau > 0.2 and ac1_p < 0.1) or (var_tau > 0.2 and var_p < 0.1):
            return 'ELEVATED'
        elif ac1_tau > 0.1 or var_tau > 0.1:
            return 'MODERATE'
        else:
            return 'NORMAL'


class PowerLawAnalyzer:
    """
    Analyze power-law behavior in financial data.

    Power laws are signatures of critical phenomena:
    P(x) ∝ x^(-α)

    In finance:
    - Return distributions: α ≈ 3 (cubic law)
    - Firm sizes: Zipf's law
    - Volatility clustering: long memory
    """

    def __init__(self):
        self.fitted_alpha = None
        self.x_min = None

    def fit_power_law(self, data: np.ndarray,
                      x_min_method: str = 'clauset') -> Dict:
        """
        Fit power-law distribution using MLE.

        Uses Clauset et al. (2009) methodology for rigorous fitting.

        Returns:
            Dictionary with alpha, x_min, and goodness-of-fit
        """
        data = np.abs(data)
        data = data[data > 0]
        data = np.sort(data)

        if x_min_method == 'clauset':
            # Find optimal x_min via KS minimization
            best_ks = np.inf
            best_xmin = data[0]
            best_alpha = 2.0

            for xmin in data[len(data)//20:len(data)//2]:
                tail = data[data >= xmin]
                if len(tail) < 10:
                    continue

                # MLE for alpha
                alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))

                # KS statistic
                cdf_empirical = np.arange(1, len(tail) + 1) / len(tail)
                cdf_theoretical = 1 - (xmin / tail)**(alpha - 1)
                ks = np.max(np.abs(cdf_empirical - cdf_theoretical))

                if ks < best_ks:
                    best_ks = ks
                    best_xmin = xmin
                    best_alpha = alpha

            self.fitted_alpha = best_alpha
            self.x_min = best_xmin

        else:  # Simple MLE with fixed x_min
            x_min = np.percentile(data, 90)  # Top 10%
            tail = data[data >= x_min]
            alpha = 1 + len(tail) / np.sum(np.log(tail / x_min))

            self.fitted_alpha = alpha
            self.x_min = x_min

        # Compute p-value via bootstrap
        p_value = self._bootstrap_pvalue(data)

        return {
            'alpha': self.fitted_alpha,
            'x_min': self.x_min,
            'ks_statistic': best_ks if x_min_method == 'clauset' else None,
            'p_value': p_value,
            'n_tail': len(data[data >= self.x_min])
        }

    def _bootstrap_pvalue(self, data: np.ndarray, n_bootstrap: int = 100) -> float:
        """Estimate p-value via bootstrap."""
        # Simplified bootstrap
        tail = data[data >= self.x_min]
        n_tail = len(tail)

        # Generate synthetic power-law data
        ks_synthetic = []
        for _ in range(n_bootstrap):
            # Sample from fitted power law
            u = np.random.uniform(0, 1, n_tail)
            synthetic = self.x_min * (1 - u)**(1 / (1 - self.fitted_alpha))

            # Fit to synthetic
            alpha_syn = 1 + n_tail / np.sum(np.log(synthetic / self.x_min))

            cdf_emp = np.arange(1, n_tail + 1) / n_tail
            cdf_theo = 1 - (self.x_min / np.sort(synthetic))**(alpha_syn - 1)
            ks_synthetic.append(np.max(np.abs(cdf_emp - cdf_theo)))

        # p-value
        ks_data = np.max(np.abs(
            np.arange(1, n_tail + 1) / n_tail -
            (1 - (self.x_min / np.sort(tail))**(self.fitted_alpha - 1))
        ))

        return np.mean(np.array(ks_synthetic) >= ks_data)


def compute_phase_space_coordinates(returns: np.ndarray,
                                   window: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D phase transition coordinates.

    Coordinates:
    - X: Susceptibility proxy (sensitivity to shocks)
    - Y: Order parameter (trend strength / herding)
    - Z: Distance from criticality (T - T_c proxy)

    Near crashes, the system moves toward critical point (X→∞, Y→0, Z→0).
    """
    n = len(returns)
    detector = CriticalSlowingDownDetector(window)

    X = np.zeros(n)  # Susceptibility
    Y = np.zeros(n)  # Order parameter
    Z = np.zeros(n)  # Distance from criticality

    for i in range(2 * window, n):
        window_returns = returns[i-window:i]

        # Susceptibility: variance of cumulative returns
        cum_returns = np.cumsum(window_returns)
        X[i] = np.var(cum_returns) / (np.var(window_returns) + 1e-10)

        # Order parameter: autocorrelation (herding)
        if np.std(window_returns) > 0:
            Y[i] = np.abs(np.corrcoef(window_returns[:-1], window_returns[1:])[0, 1])

        # Distance from criticality: inverse of variance ratio
        # High variance ratio = near critical point = small Z
        Z[i] = 1 / (X[i] + 0.1)

    # Normalize
    X = X / (np.max(X) + 1e-10)
    Y = Y / (np.max(Y) + 1e-10)
    Z = Z / (np.max(Z) + 1e-10)

    return X, Y, Z


class MarketPhaseClassifier:
    """
    Classify market into distinct phases using phase transition theory.

    Phases:
    1. STABLE (Normal): Low volatility, efficient price discovery
    2. TRENDING (Ordered): Strong momentum, herding behavior
    3. CRITICAL (Pre-crash): High susceptibility, slowing down
    4. CRASH (Phase transition): Rapid regime change
    5. RECOVERY (Post-crash): Return to stability
    """

    def __init__(self):
        self.ews_detector = CriticalSlowingDownDetector()
        self.phases = ['STABLE', 'TRENDING', 'CRITICAL', 'CRASH', 'RECOVERY']

    def classify(self, returns: np.ndarray, window: int = 50) -> Dict:
        """
        Classify current market phase.

        Returns:
            Phase classification and supporting metrics
        """
        if len(returns) < 2 * window:
            return {'phase': 'UNKNOWN', 'confidence': 0}

        # Compute EWS
        ews = self.ews_detector.compute_ews(returns)

        # Recent metrics
        recent_returns = returns[-window:]
        recent_vol = np.std(recent_returns)
        recent_return = np.mean(recent_returns)
        recent_ac1 = ews['autocorrelation'][-1] if len(ews['autocorrelation']) > 0 else 0

        # Classification logic
        vol_percentile = stats.percentileofscore(
            [np.std(returns[i:i+window]) for i in range(len(returns) - window)],
            recent_vol
        )

        if vol_percentile > 95 and recent_return < -0.02:
            phase = 'CRASH'
            confidence = min(vol_percentile / 100, 1.0)

        elif ews['warning_level'] == 'CRITICAL':
            phase = 'CRITICAL'
            confidence = 0.8

        elif vol_percentile > 90 and recent_return > 0:
            phase = 'RECOVERY'
            confidence = 0.7

        elif recent_ac1 > 0.2 and abs(recent_return) > 0.01:
            phase = 'TRENDING'
            confidence = min(recent_ac1, 1.0)

        else:
            phase = 'STABLE'
            confidence = 1 - vol_percentile / 100

        return {
            'phase': phase,
            'confidence': confidence,
            'volatility_percentile': vol_percentile,
            'autocorrelation': recent_ac1,
            'ews': ews,
            'susceptibility': ews['variance'][-1] if len(ews['variance']) > 0 else 0
        }
