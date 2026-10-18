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

try:  # NumPy >= 2.0 (np.trapz was removed in NumPy 2.4)
    from numpy import trapezoid as _trapezoid
except ImportError:  # pragma: no cover - NumPy < 2.0
    from numpy import trapz as _trapezoid


# ---------------------------------------------------------------------------
# McCulloch (1986) quantile estimator tables, "Simple consistent estimators of
# stable distribution parameters", Commun. Statist. - Simula. 15(4), 1109-1136.
# Same values as used by scipy.stats.levy_stable for its start values.
# ---------------------------------------------------------------------------
_MC_NU_ALPHA = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4,
                         5, 6, 8, 10, 15, 25])
_MC_NU_BETA = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1])
# Table III: alpha = psi_1(nu_alpha, nu_beta); rows nu_alpha, columns nu_beta
_MC_PSI1 = np.array([
    [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
    [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
    [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
    [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
    [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
    [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
    [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
    [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
    [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
    [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
    [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
    [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
    [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
    [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
    [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]])
# Table IV: beta = psi_2(nu_alpha, nu_beta); rows nu_alpha, columns nu_beta
_MC_PSI2 = np.array([
    [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
    [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
    [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
    [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
    [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
    [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
    [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
    [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
    [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
    [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
    [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
    [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
    [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
    [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
    [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]])
_MC_ALPHA = np.array([2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                      1, 0.9, 0.8, 0.7, 0.6, 0.5])[::-1]
_MC_BETA = np.array([0, 0.25, 0.5, 0.75, 1])
# Table V: nu_c = phi_3(alpha, beta) = IQR / gamma; rows alpha (ascending)
_MC_PHI3 = np.array([
    [1.908, 1.908, 1.908, 1.908, 1.908],
    [1.914, 1.915, 1.916, 1.918, 1.921],
    [1.921, 1.922, 1.927, 1.936, 1.947],
    [1.927, 1.930, 1.943, 1.961, 1.987],
    [1.933, 1.940, 1.962, 1.997, 2.043],
    [1.939, 1.952, 1.988, 2.045, 2.116],
    [1.946, 1.967, 2.022, 2.106, 2.211],
    [1.955, 1.984, 2.067, 2.188, 2.333],
    [1.965, 2.007, 2.125, 2.294, 2.491],
    [1.980, 2.040, 2.205, 2.435, 2.696],
    [2.000, 2.085, 2.311, 2.624, 2.973],
    [2.040, 2.149, 2.461, 2.886, 3.356],
    [2.098, 2.244, 2.676, 3.265, 3.912],
    [2.189, 2.392, 3.004, 3.844, 4.775],
    [2.337, 2.634, 3.542, 4.808, 6.247],
    [2.588, 3.073, 4.534, 6.636, 9.144]])[::-1]
# Table VII: nu_zeta = phi_5(alpha, beta) = (zeta - median) / gamma
_MC_PHI5 = np.array([
    [0, 0.000, 0.000, 0.000, 0.000],
    [0, -0.017, -0.032, -0.049, -0.064],
    [0, -0.030, -0.061, -0.092, -0.123],
    [0, -0.043, -0.088, -0.132, -0.179],
    [0, -0.056, -0.111, -0.170, -0.232],
    [0, -0.066, -0.134, -0.206, -0.283],
    [0, -0.075, -0.154, -0.241, -0.335],
    [0, -0.084, -0.173, -0.276, -0.390],
    [0, -0.090, -0.192, -0.310, -0.447],
    [0, -0.095, -0.208, -0.346, -0.508],
    [0, -0.098, -0.223, -0.380, -0.576],
    [0, -0.099, -0.237, -0.424, -0.652],
    [0, -0.096, -0.250, -0.469, -0.742],
    [0, -0.089, -0.262, -0.520, -0.853],
    [0, -0.078, -0.272, -0.581, -0.997],
    [0, -0.061, -0.279, -0.659, -1.198]])[::-1]


def _bilinear(row_grid: np.ndarray, col_grid: np.ndarray, table: np.ndarray,
              r: float, c: float) -> float:
    """Bilinear interpolation in a table (clamped to the table edges)."""
    along_cols = np.array([np.interp(c, col_grid, row) for row in table])
    return float(np.interp(r, row_grid, along_cols))


def _mcculloch_fit(data: np.ndarray) -> Tuple[float, float, float, float]:
    """McCulloch (1986) quantile estimator, S1 parameterization.

    Returns (alpha, beta, gamma, delta).
    """
    p05, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
    iqr = p75 - p25
    if not iqr > 0 or not (p95 - p05) > 0:
        raise ValueError(
            "Lévy fit needs data with a positive interquartile range "
            "(the data are (nearly) constant)"
        )

    nu_alpha = (p95 - p05) / iqr
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    if nu_alpha >= _MC_NU_ALPHA[0]:
        a_nb = abs(nu_beta)
        alpha = _bilinear(_MC_NU_ALPHA, _MC_NU_BETA, _MC_PSI1, nu_alpha, a_nb)
        beta = np.sign(nu_beta) * _bilinear(_MC_NU_ALPHA, _MC_NU_BETA, _MC_PSI2,
                                            nu_alpha, a_nb)
        alpha = float(np.clip(alpha, np.finfo(float).eps, 2.0))
        beta = float(np.clip(beta, -1.0, 1.0))
    else:
        # Lighter than Gaussian quantile spread -> Gaussian limit. At alpha = 2
        # the distribution does not depend on beta, so report beta = 0.
        alpha = 2.0
        beta = 0.0

    gamma = iqr / _bilinear(_MC_ALPHA, _MC_BETA, _MC_PHI3, alpha, abs(beta))
    zeta = p50 + gamma * np.sign(beta) * _bilinear(_MC_ALPHA, _MC_BETA, _MC_PHI5,
                                                   alpha, abs(beta))
    # McCulloch's location zeta -> S1 location delta
    if alpha != 1:
        delta = zeta - beta * gamma * np.tan(np.pi * alpha / 2)
    else:
        delta = zeta
    return alpha, beta, float(gamma), float(delta)


class LevyStableDistribution:
    """
    Lévy Stable Distribution for modeling fat-tailed financial returns.

    Parameters (S1 parameterization of Samorodnitsky & Taqqu / Nolan's S1,
    the default of ``scipy.stats.levy_stable``):
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
        phi = self.characteristic_function(t)  # independent of x: compute once

        x_flat = np.asarray(x, dtype=float).ravel()
        pdf_vals = np.zeros(x_flat.shape, dtype=float)
        chunk = max(1, 2_000_000 // max(1, n_points))  # bound memory (~32 MB)
        for start in range(0, x_flat.size, chunk):
            xc = x_flat[start:start + chunk]
            integrand = np.exp(-1j * np.outer(xc, t)) * phi
            pdf_vals[start:start + chunk] = (
                np.real(_trapezoid(integrand, t, axis=1)) / (2 * np.pi)
            )

        pdf_vals = pdf_vals.reshape(x.shape)
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
        if alpha == 1:
            # S1 with alpha = 1 is not closed under pure scaling:
            # gamma*Z + delta ~ S1(1, beta, gamma, delta - 2/pi*beta*gamma*ln(gamma)),
            # so the location has to be corrected to match characteristic_function().
            return (self.gamma * X + self.delta
                    + (2 / np.pi) * beta * self.gamma * np.log(self.gamma))
        return self.gamma * X + self.delta

    def tail_probability(self, x: float, tail: str = 'right') -> float:
        """
        Compute tail probability P(X > x) or P(X < x).

        For large |x - δ|, the tail follows a power law
        (Samorodnitsky & Taqqu 1994, Property 1.2.15):
        P(X > x) ~ C_α (1 + β) γ^α (x - δ)^(-α)    for x -> +∞
        P(X < x) ~ C_α (1 - β) γ^α |x - δ|^(-α)    for x -> -∞
        with C_α = Γ(α) sin(πα/2) / π.

        The distance to the location δ is used, so ``tail='left'`` with a
        threshold below δ gives the left tail. The asymptote is only valid far
        in the tail; the result is capped at 1.

        Args:
            x: Threshold value (scalar or array)
            tail: 'right' for P(X > x), 'left' for P(X < x),
                  'both' for P(|X - δ| > |x - δ|)
        """
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Power law coefficient
        C_alpha = (
            special.gamma(alpha) * np.sin(np.pi * alpha / 2)
            / np.pi
        )

        if tail == 'right':
            weight = 1 + beta
        elif tail == 'left':
            weight = 1 - beta
        elif tail == 'both':
            weight = 2.0
        else:
            raise ValueError("tail must be 'right', 'left' or 'both'")

        dist = np.abs(np.asarray(x, dtype=float) - self.delta)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            prob = C_alpha * weight * gamma**alpha * dist**(-alpha)
        prob = np.where(dist == 0, 1.0 if weight > 0 else 0.0, prob)
        prob = np.minimum(prob, 1.0)
        return float(prob) if prob.ndim == 0 else prob

    @staticmethod
    def fit(data: np.ndarray, method: str = 'quantile') -> 'LevyStableDistribution':
        """
        Fit Lévy stable parameters to data.

        Args:
            data: Array of financial returns
            method: 'quantile' (McCulloch 1986 quantile estimator).
                    'mle' (Maximum Likelihood) is not implemented and raises
                    NotImplementedError.

        Returns:
            Fitted LevyStableDistribution instance (S1 parameterization)
        """
        data = np.asarray(data, dtype=float).ravel()
        data = data[np.isfinite(data)]

        if method == 'quantile':
            # McCulloch's quantile-based estimator: alpha and beta from the
            # quantile ratios nu_alpha = (q95-q05)/(q75-q25) and
            # nu_beta = (q95+q05-2 q50)/(q95-q05), scale gamma = IQR/phi_3,
            # location from the median. Fast and consistent; tables cover 0.5 <= alpha <= 2.
            if data.size < 5:
                raise ValueError("Lévy fit needs at least 5 finite observations")
            alpha, beta, gamma, delta = _mcculloch_fit(data)

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
        # Accumulate log-prices: S_i = S0 * exp(sum of log-returns). Multiplying
        # prices step by step turns an overflowed path (inf) into NaN at the
        # next large down-jump (inf * 0); the log-space sum cannot.
        log_paths = np.zeros((n_paths, n_steps + 1))
        path_index = np.arange(n_paths)

        for i in range(1, n_steps + 1):
            # Gaussian component
            dW = np.random.normal(0, np.sqrt(dt), n_paths)

            # Lévy jump component: all jumps of this step in one draw,
            # then summed per path (same random stream as before for one path)
            n_jumps = np.random.poisson(self.jump_intensity * dt, n_paths)
            total_jumps = int(n_jumps.sum())
            if total_jumps > 0:
                jump_sizes = self.levy_dist.sample(total_jumps)
                owner = np.repeat(path_index, n_jumps)
                jumps = np.bincount(owner, weights=jump_sizes,
                                    minlength=n_paths) * 0.01  # Scale down
            else:
                jumps = np.zeros(n_paths)

            # Update log-price
            log_return = (
                (self.mu - 0.5 * self.sigma_gaussian**2) * dt
                + self.sigma_gaussian * dW
                + jumps
            )
            log_paths[:, i] = log_paths[:, i-1] + log_return

        with np.errstate(over='ignore'):
            paths = S0 * np.exp(log_paths)
        paths[:, 0] = S0

        return paths

    def _horizon_distribution(self, horizon: float) -> LevyStableDistribution:
        """
        Distribution of the sum of `horizon` i.i.d. one-period returns.

        Stable laws aggregate exactly (Samorodnitsky & Taqqu, Property 1.2.1):
        the h-period sum of S1(α, β, γ, δ) is S1(α, β, h^(1/α) γ, h δ).
        The square-root-of-time rule is the special case α = 2.
        """
        if not horizon > 0:
            raise ValueError("horizon must be positive")
        d = self.levy_dist
        return LevyStableDistribution(d.alpha, d.beta,
                                      d.gamma * horizon**(1 / d.alpha),
                                      d.delta * horizon)

    def _simulate_horizon_returns(self, horizon: float, n_simulations: int,
                                  seed: Optional[int]) -> np.ndarray:
        return self._horizon_distribution(horizon).sample(n_simulations, seed=seed)

    def compute_var(self, confidence: float = 0.99, horizon: int = 1,
                    n_simulations: int = 10000,
                    seed: Optional[int] = None) -> float:
        """
        Compute Value at Risk using Lévy flight model.

        Unlike Gaussian VaR, this properly accounts for fat tails. The returns
        are drawn from ``levy_dist`` (the one-period return law); the horizon
        is aggregated with the stable scaling h^(1/α) instead of sqrt(h).

        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR)
            horizon: Time horizon in periods (days if levy_dist is daily)
            n_simulations: Number of simulations
            seed: Optional random seed (reseeds NumPy's global RNG)

        Returns:
            VaR as a positive loss in the units of ``levy_dist``
            (a decimal return if it was fitted to decimal returns)
        """
        # Simulate horizon returns
        returns = self._simulate_horizon_returns(horizon, n_simulations, seed)

        # VaR is the negative of the (1-confidence) quantile
        var = -np.percentile(returns, (1 - confidence) * 100)

        return var

    def compute_cvar(self, confidence: float = 0.99, horizon: int = 1,
                     n_simulations: int = 10000,
                     seed: Optional[int] = None) -> float:
        """
        Compute Conditional Value at Risk (Expected Shortfall).

        CVaR = E[Loss | Loss >= VaR]

        This is a more coherent risk measure than VaR for fat-tailed distributions.
        Same units, horizon scaling and ``seed`` semantics as :meth:`compute_var`.
        """
        returns = self._simulate_horizon_returns(horizon, n_simulations, seed)

        var = -np.percentile(returns, (1 - confidence) * 100)
        cvar = -np.mean(returns[returns <= -var])

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
    returns = np.asarray(returns, dtype=float).ravel()
    returns = returns[np.isfinite(returns)]

    # Use absolute values for tail analysis
    abs_returns = np.abs(returns)
    sorted_returns = np.sort(abs_returns)[::-1]  # Descending

    if method == 'hill':
        # Hill estimator: uses k largest order statistics
        n = len(sorted_returns)
        # Only strictly positive values carry tail information (log(0) = -inf)
        n_pos = int(np.count_nonzero(sorted_returns > 0))
        if n_pos < 2:
            raise ValueError("Hill estimator needs at least 2 non-zero returns")
        k = int(0.1 * n)  # Use top 10%
        k = max(10, k)
        k = min(k, n - 1, n_pos - 1)  # threshold X_(k+1) must exist and be > 0

        # Hill estimator formula
        log_ratios = np.log(sorted_returns[:k]) - np.log(sorted_returns[k])
        hill_estimate = 1 / np.mean(log_ratios)

        return hill_estimate

    elif method == 'pickands':
        # Pickands estimator
        n = len(sorted_returns)
        k = int(0.1 * n)
        if k < 1:
            raise ValueError("Pickands estimator needs at least 10 observations")

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
