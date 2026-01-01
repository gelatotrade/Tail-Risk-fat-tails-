"""
Lévy Flight Model for Financial Fat Tails
==========================================

Lévy flights originate from particle physics and describe random walks where
step sizes follow a heavy-tailed distribution. Unlike Brownian motion (Gaussian),
Lévy flights can produce large jumps - perfectly modeling market crashes.

Physics Background:
- Originally described anomalous diffusion in physics
- Characterized by infinite variance (for α < 2)
- Step size distribution: P(x) ~ |x|^(-1-α) for large |x|
- α parameter (stability index): 0 < α ≤ 2
  - α = 2: Gaussian (normal diffusion)
  - α < 2: Fat tails (anomalous superdiffusion)
  - α ≈ 1.7: Typical for financial markets

The connection to finance:
- Market returns exhibit Lévy-stable behavior
- Large price movements (Black Swans) are Lévy jumps
- Mandelbrot's "cotton prices" were first modeled with Lévy distributions
"""

import numpy as np
from scipy import special, optimize, stats
from scipy.integrate import quad
from typing import Tuple, Optional, Dict, Any
import warnings


class LevyStableDistribution:
    """
    Lévy Stable Distribution for modeling fat-tailed financial returns.

    Parameters (Nolan's S0 parameterization):
    - alpha (α): Stability parameter, 0 < α ≤ 2 (tail index)
    - beta (β): Skewness parameter, -1 ≤ β ≤ 1
    - gamma (γ): Scale parameter, γ > 0
    - delta (δ): Location parameter

    For financial applications:
    - α < 2 indicates fat tails (typically 1.5-1.9 for stock returns)
    - β < 0 indicates left skewness (crash tendency)
    - γ represents volatility
    - δ represents mean drift
    """

    def __init__(self, alpha: float = 1.7, beta: float = -0.1,
                 gamma: float = 1.0, delta: float = 0.0):
        """
        Initialize Lévy Stable Distribution.

        Args:
            alpha: Stability index (0 < α ≤ 2). Lower = fatter tails
            beta: Skewness (-1 ≤ β ≤ 1). Negative = left skew (crashes)
            gamma: Scale parameter (> 0). Analogous to standard deviation
            delta: Location parameter. Analogous to mean
        """
        if not 0 < alpha <= 2:
            raise ValueError("alpha must be in (0, 2]")
        if not -1 <= beta <= 1:
            raise ValueError("beta must be in [-1, 1]")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def characteristic_function(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the characteristic function φ(t) of the Lévy stable distribution.

        The characteristic function is defined as:
        φ(t) = exp(iδt - γ^α|t|^α [1 - iβ sign(t) Φ])

        where Φ = tan(πα/2) for α ≠ 1, and Φ = -(2/π)log|t| for α = 1
        """
        t = np.asarray(t)
        alpha, beta, gamma, delta = self.alpha, self.beta, self.gamma, self.delta

        if alpha == 1:
            Phi = -(2 / np.pi) * np.log(np.abs(t) + 1e-10)
        else:
            Phi = np.tan(np.pi * alpha / 2)

        # Compute characteristic function
        phi = np.exp(
            1j * delta * t
            - gamma**alpha * np.abs(t)**alpha * (1 - 1j * beta * np.sign(t) * Phi)
        )

        return phi

    def pdf(self, x: np.ndarray, n_points: int = 1000) -> np.ndarray:
        """
        Compute PDF via numerical Fourier inversion of characteristic function.

        Uses the Lévy inversion formula:
        f(x) = (1/2π) ∫ exp(-itx) φ(t) dt
        """
        x = np.asarray(x)
        t_max = 50 / self.gamma  # Integration limit
        t = np.linspace(-t_max, t_max, n_points)
        dt = t[1] - t[0]

        pdf_vals = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x.flat):
            integrand = np.exp(-1j * t * xi) * self.characteristic_function(t)
            pdf_vals.flat[i] = np.real(np.trapz(integrand, t)) / (2 * np.pi)

        return np.maximum(pdf_vals, 0)  # Ensure non-negative

    def sample(self, size: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate samples using the Chambers-Mallows-Stuck algorithm.

        This is the standard method for simulating Lévy stable random variables.
        """
        if seed is not None:
            np.random.seed(seed)

        alpha, beta = self.alpha, self.beta

        # Generate uniform and exponential random variables
        V = np.random.uniform(-np.pi / 2, np.pi / 2, size)
        W = np.random.exponential(1, size)

        if alpha == 1:
            # Special case: Cauchy-like
            X = (2 / np.pi) * (
                (np.pi / 2 + beta * V) * np.tan(V)
                - beta * np.log((np.pi / 2 * W * np.cos(V)) / (np.pi / 2 + beta * V))
            )
        else:
            # General case
            zeta = beta * np.tan(np.pi * alpha / 2)
            xi = np.arctan(zeta) / alpha

            X = (
                (1 + zeta**2)**(1 / (2 * alpha))
                * np.sin(alpha * (V + xi)) / np.cos(V)**(1 / alpha)
                * (np.cos(V - alpha * (V + xi)) / W)**((1 - alpha) / alpha)
            )

        # Scale and shift
        return self.gamma * X + self.delta

    def tail_probability(self, x: float, tail: str = 'right') -> float:
        """
        Compute tail probability P(X > x) or P(X < x).

        For large |x|, the tail follows a power law:
        P(X > x) ~ C_α (1 + β) / 2 * γ^α * x^(-α)

        Args:
            x: Threshold value
            tail: 'right' for P(X > x), 'left' for P(X < x)
        """
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Power law coefficient
        C_alpha = (
            special.gamma(alpha) * np.sin(np.pi * alpha / 2)
            / np.pi
        )

        if tail == 'right':
            return C_alpha * (1 + beta) / 2 * gamma**alpha * abs(x)**(-alpha)
        else:
            return C_alpha * (1 - beta) / 2 * gamma**alpha * abs(x)**(-alpha)

    @staticmethod
    def fit(data: np.ndarray, method: str = 'quantile') -> 'LevyStableDistribution':
        """
        Fit Lévy stable parameters to data.

        Args:
            data: Array of financial returns
            method: 'quantile' (McCulloch) or 'mle' (Maximum Likelihood)

        Returns:
            Fitted LevyStableDistribution instance
        """
        data = np.asarray(data)

        if method == 'quantile':
            # McCulloch's quantile-based estimator
            # Fast and reasonably accurate
            q = np.percentile(data, [5, 25, 50, 75, 95])

            # Estimate scale from interquartile range
            gamma = (q[3] - q[1]) / 2.63  # For alpha ~ 2

            # Estimate location
            delta = q[2]  # Median

            # Estimate alpha from tail behavior
            tail_ratio = (q[4] - q[2]) / (q[2] - q[0])
            # Higher ratio = more right skewness

            # Estimate beta from skewness
            beta = np.clip((tail_ratio - 1) / (tail_ratio + 1), -1, 1)

            # Estimate alpha from kurtosis proxy
            kurtosis = stats.kurtosis(data)
            alpha = np.clip(2 / (1 + 0.1 * kurtosis), 1.1, 2.0)

        else:
            raise NotImplementedError(f"Method {method} not implemented")

        return LevyStableDistribution(alpha, beta, gamma, delta)


class LevyFlightProcess:
    """
    Lévy Flight Process for modeling price dynamics with jumps.

    This extends geometric Brownian motion to include Lévy jumps:
    dS/S = μdt + σdW + dL

    where L is a Lévy process generating large jumps (Black Swans).
    """

    def __init__(self, levy_dist: LevyStableDistribution,
                 mu: float = 0.0, sigma_gaussian: float = 0.1,
                 jump_intensity: float = 0.1):
        """
        Initialize Lévy flight price process.

        Args:
            levy_dist: Lévy stable distribution for jump sizes
            mu: Drift rate
            sigma_gaussian: Gaussian volatility component
            jump_intensity: Expected jumps per unit time (λ)
        """
        self.levy_dist = levy_dist
        self.mu = mu
        self.sigma_gaussian = sigma_gaussian
        self.jump_intensity = jump_intensity

    def simulate_path(self, S0: float, T: float, n_steps: int = 252,
                      n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate price paths with Lévy jumps.

        Uses compound Poisson process for jumps:
        - Number of jumps N(t) ~ Poisson(λt)
        - Jump sizes from Lévy stable distribution

        Args:
            S0: Initial price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            seed: Random seed

        Returns:
            Array of shape (n_paths, n_steps + 1) with price paths
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for i in range(1, n_steps + 1):
            # Gaussian component
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            # Lévy jump component
            n_jumps = np.random.poisson(self.jump_intensity * dt, n_paths)
            jumps = np.zeros(n_paths)
            for j in range(n_paths):
                if n_jumps[j] > 0:
                    jump_sizes = self.levy_dist.sample(n_jumps[j])
                    jumps[j] = np.sum(jump_sizes) * 0.01  # Scale down

            # Update price
            log_return = (
                (self.mu - 0.5 * self.sigma_gaussian**2) * dt
                + self.sigma_gaussian * dW
                + jumps
            )
            paths[:, i] = paths[:, i-1] * np.exp(log_return)

        return paths

    def compute_var(self, confidence: float = 0.99, horizon: int = 1,
                    n_simulations: int = 10000) -> float:
        """
        Compute Value at Risk using Lévy flight model.

        Unlike Gaussian VaR, this properly accounts for fat tails.

        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR)
            horizon: Time horizon in days
            n_simulations: Number of simulations

        Returns:
            VaR as percentage loss
        """
        # Simulate returns
        returns = self.levy_dist.sample(n_simulations)
        returns *= np.sqrt(horizon)  # Scale by time

        # VaR is the negative of the (1-confidence) quantile
        var = -np.percentile(returns, (1 - confidence) * 100)

        return var

    def compute_cvar(self, confidence: float = 0.99, horizon: int = 1,
                     n_simulations: int = 10000) -> float:
        """
        Compute Conditional Value at Risk (Expected Shortfall).

        CVaR = E[Loss | Loss > VaR]

        This is a more coherent risk measure than VaR for fat-tailed distributions.
        """
        returns = self.levy_dist.sample(n_simulations)
        returns *= np.sqrt(horizon)

        var = -np.percentile(returns, (1 - confidence) * 100)
        cvar = -np.mean(returns[returns < -var])

        return cvar


def estimate_tail_index(returns: np.ndarray, method: str = 'hill') -> float:
    """
    Estimate the tail index (α) from return data.

    The tail index determines how heavy the tails are:
    - α = 2: Gaussian (thin tails)
    - α ≈ 3: Finite variance, fat tails
    - α < 2: Infinite variance (Lévy stable regime)

    Args:
        returns: Array of returns
        method: 'hill' (Hill estimator) or 'pickands'

    Returns:
        Estimated tail index
    """
    returns = np.asarray(returns)

    # Use absolute values for tail analysis
    abs_returns = np.abs(returns)
    sorted_returns = np.sort(abs_returns)[::-1]  # Descending

    if method == 'hill':
        # Hill estimator: uses k largest order statistics
        n = len(sorted_returns)
        k = int(0.1 * n)  # Use top 10%
        k = max(10, min(k, n - 1))

        # Hill estimator formula
        log_ratios = np.log(sorted_returns[:k]) - np.log(sorted_returns[k])
        hill_estimate = 1 / np.mean(log_ratios)

        return hill_estimate

    elif method == 'pickands':
        # Pickands estimator
        n = len(sorted_returns)
        k = int(0.1 * n)

        # Pickands formula using order statistics
        X_k = sorted_returns[k]
        X_2k = sorted_returns[2*k]
        X_4k = sorted_returns[min(4*k, n-1)]

        pickands = 1 / np.log(2) * np.log((X_k - X_2k) / (X_2k - X_4k))

        return 1 / pickands

    else:
        raise ValueError(f"Unknown method: {method}")


def levy_flight_3d_coordinates(returns: np.ndarray,
                               window: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D Lévy flight phase space coordinates.

    The 3D space represents:
    - X: Volatility regime (rolling std)
    - Y: Tail heaviness (local tail index)
    - Z: Jump intensity (absolute return magnitude)

    This creates a "risk phase space" where:
    - Normal regimes cluster near origin
    - Crisis regimes extend outward
    - Black Swans appear as far outliers

    Args:
        returns: Time series of returns
        window: Rolling window size

    Returns:
        Tuple of (X, Y, Z) coordinate arrays
    """
    returns = np.asarray(returns)
    n = len(returns)

    # X: Rolling volatility (normalized)
    vol = np.array([np.std(returns[max(0, i-window):i+1])
                    for i in range(n)])
    X = vol / np.mean(vol)  # Normalize to mean = 1

    # Y: Local tail index (inverse, so higher = heavier tails)
    Y = np.zeros(n)
    for i in range(window, n):
        try:
            alpha = estimate_tail_index(returns[i-window:i])
            Y[i] = 2 / alpha  # Transform so higher = heavier tails
        except:
            Y[i] = 0.5
    Y = np.clip(Y, 0, 2)

    # Z: Jump intensity (scaled absolute returns)
    Z = np.abs(returns) / np.std(returns)

    return X, Y, Z
