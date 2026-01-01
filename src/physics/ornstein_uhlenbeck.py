"""
Ornstein-Uhlenbeck Process with Jumps for Volatility Modeling
=============================================================

The Ornstein-Uhlenbeck (OU) process is a mean-reverting stochastic process
originally from physics (thermal fluctuations). In finance, it models
volatility and interest rates.

Physics Background:
- Describes velocity of a Brownian particle with friction
- Mean-reversion: dx = θ(μ - x)dt + σdW
- Stationary distribution: Gaussian with variance σ²/(2θ)

Financial Extensions:
- Vasicek model (interest rates)
- Volatility mean-reversion
- Jump-diffusion for fat tails

Key Innovation - OU with Lévy Jumps:
dx = θ(μ - x)dt + σdW + dJ

where J is a compound Poisson process with Lévy-distributed jumps.
This captures:
1. Mean-reversion (volatility clustering decay)
2. Jumps (sudden volatility spikes / Black Swans)
3. Fat tails (from jump component)
"""

import numpy as np
from scipy import stats, optimize
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional, Dict, List
import warnings


class OrnsteinUhlenbeckProcess:
    """
    Standard Ornstein-Uhlenbeck process.

    dx = θ(μ - x)dt + σdW

    Parameters:
    - θ (theta): Speed of mean reversion
    - μ (mu): Long-term mean
    - σ (sigma): Volatility of volatility
    """

    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.1):
        """
        Initialize OU process.

        Args:
            theta: Mean reversion speed (larger = faster reversion)
            mu: Long-term mean level
            sigma: Volatility parameter
        """
        if theta <= 0:
            raise ValueError("theta must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def stationary_variance(self) -> float:
        """Variance of stationary distribution: σ²/(2θ)"""
        return self.sigma**2 / (2 * self.theta)

    def stationary_std(self) -> float:
        """Standard deviation of stationary distribution."""
        return np.sqrt(self.stationary_variance())

    def half_life(self) -> float:
        """Half-life of mean reversion: ln(2)/θ"""
        return np.log(2) / self.theta

    def expected_value(self, x0: float, t: float) -> float:
        """
        Expected value at time t given initial value x0.

        E[X_t | X_0 = x0] = μ + (x0 - μ)exp(-θt)
        """
        return self.mu + (x0 - self.mu) * np.exp(-self.theta * t)

    def variance_at_time(self, t: float) -> float:
        """
        Variance at time t (starting from deterministic point).

        Var[X_t] = (σ²/2θ)(1 - exp(-2θt))
        """
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t))

    def simulate(self, x0: float, T: float, n_steps: int = 1000,
                 n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate OU process paths using exact discretization.

        Uses the exact transition density rather than Euler-Maruyama.

        Args:
            x0: Initial value
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            seed: Random seed

        Returns:
            Array of shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0

        # Exact transition parameters
        exp_theta_dt = np.exp(-self.theta * dt)
        var_dt = self.variance_at_time(dt)
        std_dt = np.sqrt(var_dt)

        for i in range(1, n_steps + 1):
            # Exact update
            mean = self.mu + (paths[:, i-1] - self.mu) * exp_theta_dt
            paths[:, i] = mean + std_dt * np.random.normal(0, 1, n_paths)

        return paths

    @staticmethod
    def fit(data: np.ndarray, dt: float = 1/252) -> 'OrnsteinUhlenbeckProcess':
        """
        Fit OU parameters using maximum likelihood.

        For discrete observations: X_{t+dt} = a + b*X_t + ε

        Where:
        - a = μ(1 - exp(-θdt))
        - b = exp(-θdt)
        - ε ~ N(0, σ²(1-exp(-2θdt))/(2θ))
        """
        data = np.asarray(data)
        x = data[:-1]
        y = data[1:]

        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)

        # OLS estimates
        b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        a = (sum_y - b * sum_x) / n

        # Transform to OU parameters
        theta = -np.log(b) / dt
        mu = a / (1 - b)

        # Estimate sigma from residuals
        residuals = y - (a + b * x)
        var_residuals = np.var(residuals)
        sigma = np.sqrt(2 * theta * var_residuals / (1 - b**2))

        return OrnsteinUhlenbeckProcess(theta, mu, sigma)


class OUWithJumps:
    """
    Ornstein-Uhlenbeck process with compound Poisson jumps.

    dx = θ(μ - x)dt + σdW + dJ

    where J is a compound Poisson process:
    - Jump times: Poisson(λ)
    - Jump sizes: Lévy stable or double exponential

    This creates fat tails through the jump component while maintaining
    mean reversion for the base process.
    """

    def __init__(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 0.1,
                 jump_intensity: float = 0.1, jump_mean: float = 0.0,
                 jump_std: float = 0.5, jump_distribution: str = 'normal'):
        """
        Initialize OU with jumps.

        Args:
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Gaussian volatility
            jump_intensity: Expected jumps per unit time (λ)
            jump_mean: Mean jump size
            jump_std: Jump size volatility
            jump_distribution: 'normal', 'double_exponential', or 'levy'
        """
        self.base_ou = OrnsteinUhlenbeckProcess(theta, mu, sigma)
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.jump_distribution = jump_distribution

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def sample_jump_size(self, n: int = 1) -> np.ndarray:
        """Sample jump sizes from specified distribution."""
        if self.jump_distribution == 'normal':
            return np.random.normal(self.jump_mean, self.jump_std, n)

        elif self.jump_distribution == 'double_exponential':
            # Double exponential (Kou model)
            p = 0.5  # Probability of upward jump
            eta_up = 1 / self.jump_std
            eta_down = 1 / self.jump_std

            u = np.random.uniform(0, 1, n)
            jumps = np.where(
                u < p,
                np.random.exponential(1/eta_up, n),
                -np.random.exponential(1/eta_down, n)
            )
            return jumps + self.jump_mean

        elif self.jump_distribution == 'levy':
            # Lévy stable with α < 2
            alpha = 1.7
            # Use Student-t as approximation
            df = alpha
            jumps = stats.t.rvs(df, loc=self.jump_mean, scale=self.jump_std, size=n)
            return jumps

        else:
            raise ValueError(f"Unknown distribution: {self.jump_distribution}")

    def simulate(self, x0: float, T: float, n_steps: int = 1000,
                 n_paths: int = 1, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate OU-jump process.

        Returns paths and jump information.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0

        jump_times = []
        jump_sizes = []

        # OU parameters
        exp_theta_dt = np.exp(-self.theta * dt)
        var_dt = self.base_ou.variance_at_time(dt)
        std_dt = np.sqrt(var_dt)

        for i in range(1, n_steps + 1):
            # OU update
            mean = self.mu + (paths[:, i-1] - self.mu) * exp_theta_dt
            paths[:, i] = mean + std_dt * np.random.normal(0, 1, n_paths)

            # Jump component
            for path_idx in range(n_paths):
                n_jumps = np.random.poisson(self.jump_intensity * dt)
                if n_jumps > 0:
                    jumps = self.sample_jump_size(n_jumps)
                    paths[path_idx, i] += np.sum(jumps)
                    jump_times.append(i * dt)
                    jump_sizes.extend(jumps)

        return {
            'paths': paths,
            'jump_times': np.array(jump_times),
            'jump_sizes': np.array(jump_sizes),
            'times': np.linspace(0, T, n_steps + 1)
        }

    def compute_var(self, x0: float, T: float = 1/252,
                    confidence: float = 0.99,
                    n_simulations: int = 10000) -> float:
        """
        Compute VaR for the jump-diffusion process.

        Accounts for both continuous and jump risk.
        """
        result = self.simulate(x0, T, n_steps=1, n_paths=n_simulations)
        terminal_values = result['paths'][:, -1]

        returns = terminal_values - x0
        var = -np.percentile(returns, (1 - confidence) * 100)

        return var

    def compute_expected_shortfall(self, x0: float, T: float = 1/252,
                                   confidence: float = 0.99,
                                   n_simulations: int = 10000) -> float:
        """Compute Expected Shortfall (CVaR)."""
        result = self.simulate(x0, T, n_steps=1, n_paths=n_simulations)
        terminal_values = result['paths'][:, -1]

        returns = terminal_values - x0
        var = -np.percentile(returns, (1 - confidence) * 100)
        es = -np.mean(returns[returns < -var])

        return es


class VolatilityOUModel:
    """
    Model volatility using OU process (like Vasicek for vol).

    Volatility follows:
    dσ = κ(θ_σ - σ)dt + η√σ dW + dJ

    This is a CIR-like process with jumps for volatility.
    """

    def __init__(self, kappa: float = 2.0, theta: float = 0.2,
                 eta: float = 0.3, jump_intensity: float = 0.05):
        """
        Initialize volatility model.

        Args:
            kappa: Volatility mean reversion speed
            theta: Long-term volatility level
            eta: Vol-of-vol
            jump_intensity: Jump frequency
        """
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.jump_intensity = jump_intensity

    def simulate_volatility(self, sigma0: float, T: float,
                           n_steps: int = 252) -> np.ndarray:
        """Simulate volatility path."""
        dt = T / n_steps
        sigma = np.zeros(n_steps + 1)
        sigma[0] = sigma0

        for i in range(1, n_steps + 1):
            # Drift and diffusion
            drift = self.kappa * (self.theta - sigma[i-1]) * dt
            diffusion = self.eta * np.sqrt(max(sigma[i-1], 0) * dt) * np.random.normal()

            # Jumps (always positive for volatility)
            jump = 0
            if np.random.random() < self.jump_intensity * dt:
                jump = np.random.exponential(0.1)  # Vol spike

            sigma[i] = max(sigma[i-1] + drift + diffusion + jump, 0.01)

        return sigma

    def simulate_price(self, S0: float, sigma0: float, T: float,
                       n_steps: int = 252, r: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Simulate price with stochastic volatility.

        dS/S = r dt + σ dW_S
        dσ = κ(θ - σ)dt + η√σ dW_σ
        """
        dt = T / n_steps
        sigma = self.simulate_volatility(sigma0, T, n_steps)

        S = np.zeros(n_steps + 1)
        S[0] = S0

        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            S[i] = S[i-1] * np.exp((r - 0.5 * sigma[i-1]**2) * dt + sigma[i-1] * dW)

        returns = np.diff(np.log(S))

        return {
            'prices': S,
            'volatility': sigma,
            'returns': returns,
            'times': np.linspace(0, T, n_steps + 1)
        }


def compute_ou_coordinates(returns: np.ndarray,
                          window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D OU phase space.

    Coordinates represent volatility dynamics:
    - X: Mean reversion level (θ)
    - Y: Volatility level (σ)
    - Z: Reversion speed (κ)

    High X, high Y, low Z = trending volatility regime (danger)
    Low X, low Y, high Z = stable regime
    """
    n = len(returns)

    X = np.zeros(n)  # Mean reversion level
    Y = np.zeros(n)  # Current volatility
    Z = np.zeros(n)  # Reversion speed

    # Use absolute returns as volatility proxy
    abs_returns = np.abs(returns)

    for i in range(2 * window, n):
        window_data = abs_returns[i-window:i]

        try:
            # Fit OU to volatility
            ou = OrnsteinUhlenbeckProcess.fit(window_data, dt=1)

            X[i] = ou.mu  # Long-term volatility
            Y[i] = window_data[-1]  # Current volatility
            Z[i] = ou.theta  # Mean reversion speed

        except:
            X[i] = np.mean(window_data)
            Y[i] = window_data[-1]
            Z[i] = 1.0

    # Normalize
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-10)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-10)
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-10)

    return X, Y, Z


class JumpDiffusionRiskAnalyzer:
    """
    Analyze tail risk using jump-diffusion framework.

    Combines OU dynamics with jump risk for comprehensive
    tail risk assessment.
    """

    def __init__(self):
        self.ou_model = None
        self.jump_intensity = 0
        self.jump_std = 0

    def fit(self, returns: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        Fit jump-diffusion model to returns.

        Separates continuous and jump components.
        """
        std = np.std(returns)
        mean = np.mean(returns)

        # Identify jumps (returns beyond threshold std)
        is_jump = np.abs(returns - mean) > threshold * std
        jumps = returns[is_jump]
        continuous = returns[~is_jump]

        # Fit OU to continuous part
        if len(continuous) > 10:
            self.ou_model = OrnsteinUhlenbeckProcess.fit(continuous)

        # Jump statistics
        self.jump_intensity = np.sum(is_jump) / len(returns)
        self.jump_std = np.std(jumps) if len(jumps) > 0 else 0

        return {
            'theta': self.ou_model.theta if self.ou_model else 0,
            'mu': self.ou_model.mu if self.ou_model else 0,
            'sigma': self.ou_model.sigma if self.ou_model else std,
            'jump_intensity': self.jump_intensity,
            'jump_std': self.jump_std,
            'n_jumps': np.sum(is_jump),
            'jump_contribution': np.sum(np.abs(jumps)) / np.sum(np.abs(returns)) if len(jumps) > 0 else 0
        }

    def tail_risk_decomposition(self, returns: np.ndarray,
                                confidence: float = 0.99) -> Dict:
        """
        Decompose tail risk into continuous and jump components.
        """
        # Fit model
        fit_result = self.fit(returns)

        # Simulate to estimate risks
        if self.ou_model:
            ou_jump = OUWithJumps(
                theta=fit_result['theta'],
                mu=fit_result['mu'],
                sigma=fit_result['sigma'],
                jump_intensity=fit_result['jump_intensity'],
                jump_std=fit_result['jump_std']
            )

            # VaR decomposition
            result = ou_jump.simulate(0, 1/252, n_steps=1, n_paths=10000)
            paths = result['paths'][:, -1]

            var_total = -np.percentile(paths, (1 - confidence) * 100)

            # Continuous-only VaR
            ou_only = OrnsteinUhlenbeckProcess(
                fit_result['theta'],
                fit_result['mu'],
                fit_result['sigma']
            )
            continuous_paths = ou_only.simulate(0, 1/252, n_steps=1, n_paths=10000)
            var_continuous = -np.percentile(continuous_paths[:, -1], (1 - confidence) * 100)

            var_jump = var_total - var_continuous

        else:
            var_total = np.percentile(-returns, confidence * 100)
            var_continuous = 0
            var_jump = var_total

        return {
            'var_total': var_total,
            'var_continuous': var_continuous,
            'var_jump': var_jump,
            'jump_contribution_pct': var_jump / var_total * 100 if var_total > 0 else 0,
            **fit_result
        }
