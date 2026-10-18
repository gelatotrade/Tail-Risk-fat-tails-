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
        # ``not x > 0`` also rejects NaN, which slips through ``x <= 0``.
        if not (theta > 0 and np.isfinite(theta)):
            raise ValueError(f"theta must be positive and finite (got {theta})")
        if not (sigma > 0 and np.isfinite(sigma)):
            raise ValueError(f"sigma must be positive and finite (got {sigma})")
        if not np.isfinite(mu):
            raise ValueError(f"mu must be finite (got {mu})")

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
    def fit(data: np.ndarray, dt: float = 1/252,
            max_theta: Optional[float] = None) -> 'OrnsteinUhlenbeckProcess':
        """
        Fit OU parameters using maximum likelihood.

        For discrete observations: X_{t+dt} = a + b*X_t + ε

        Where:
        - a = μ(1 - exp(-θdt))
        - b = exp(-θdt)
        - ε ~ N(0, σ²(1-exp(-2θdt))/(2θ))

        An OU process can only produce AR(1) slopes 0 < b < 1:

        - b >= 1 (random walk or trend) means no mean reversion (θ <= 0)
          and raises ValueError.
        - b <= 0 (negative lag-1 autocorrelation) would need θ = ∞ and
          raises ValueError, unless ``max_theta`` is given.  Then θ is capped
          at ``max_theta`` (b floored at exp(-max_theta·dt), the white-noise
          limit) and a and the residual variance are re-estimated for that
          slope.  The cap also applies to valid fits with θ > max_theta.
        - Constant, too short or non-finite data raises ValueError.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("OU fit needs a 1-D series with at least 3 observations")
        if not np.all(np.isfinite(data)):
            raise ValueError("OU fit: data contains NaN or inf")
        if max_theta is not None and not (max_theta > 0 and np.isfinite(max_theta)):
            raise ValueError(f"max_theta must be positive and finite (got {max_theta})")

        x = data[:-1]
        y = data[1:]

        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)

        denom = n * sum_x2 - sum_x**2
        if np.ptp(x) == 0 or not denom > 0:
            raise ValueError("OU fit: data is constant, the AR(1) slope is undefined")

        # OLS estimates
        b = (n * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n

        if not np.isfinite(b):
            raise ValueError("OU fit: AR(1) slope is not finite")
        if b >= 1:
            raise ValueError(
                f"OU fit: AR(1) slope b = {b:.4f} >= 1, no mean reversion "
                "(random walk or trend), theta would be <= 0")

        b_floor = np.exp(-max_theta * dt) if max_theta is not None else 0.0
        if b <= b_floor:
            if max_theta is None:
                raise ValueError(
                    f"OU fit: AR(1) slope b = {b:.4f} <= 0 (negative lag-1 "
                    "autocorrelation); an OU process needs 0 < b < 1, theta "
                    "would be infinite. Pass max_theta to cap theta.")
            # Constrained least squares on the boundary b = b_floor
            b = b_floor
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

        Returns paths and jump information.  ``jump_times``, ``jump_sizes``
        and ``jump_paths`` have one entry per jump (aligned element-wise):
        time of the step, jump size and index of the path it hit.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0

        jump_times = []
        jump_sizes = []
        jump_paths = []

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
                    jump_times.extend([i * dt] * n_jumps)
                    jump_sizes.extend(jumps)
                    jump_paths.extend([path_idx] * n_jumps)

        return {
            'paths': paths,
            'jump_times': np.array(jump_times, dtype=float),
            'jump_sizes': np.array(jump_sizes, dtype=float),
            'jump_paths': np.array(jump_paths, dtype=int),
            'times': np.linspace(0, T, n_steps + 1)
        }

    def sample_terminal(self, x0: float, T: float, n_simulations: int = 10000,
                        seed: Optional[int] = None) -> np.ndarray:
        """
        Exact sample of X_T given X_0 = x0, for any horizon T.

        The diffusion uses the exact OU transition over T.  Jumps arrive as
        a Poisson(λT) number at uniform times U in [0, T]; mean reversion
        acts on them too, so a jump J that arrives at U contributes
        J·exp(-θ(T - U)) at T.  (A single ``simulate`` step instead adds every
        jump of the step undecayed at its end, which overstates the jump risk
        when θT is not small.)

        With a seed, the diffusion draws come first, so a model without jumps
        and the same seed has exactly the same diffusion part (common random
        numbers).  Vectorised over the scenarios.
        """
        if seed is not None:
            np.random.seed(seed)
        n_simulations = int(n_simulations)
        x = (self.base_ou.expected_value(x0, T)
             + np.sqrt(self.base_ou.variance_at_time(T)) * np.random.normal(0, 1, n_simulations))
        counts = np.random.poisson(self.jump_intensity * T, n_simulations)
        total = int(counts.sum())
        if total:
            sizes = np.asarray(self.sample_jump_size(total), dtype=float)
            arrival = np.random.uniform(0, T, total)
            owner = np.repeat(np.arange(n_simulations), counts)
            np.add.at(x, owner, sizes * np.exp(-self.theta * (T - arrival)))
        return x

    def compute_var(self, x0: float, T: float = 1/252,
                    confidence: float = 0.99,
                    n_simulations: int = 10000,
                    seed: Optional[int] = None) -> float:
        """
        Compute VaR for the jump-diffusion process.

        Accounts for both continuous and jump risk over the horizon T
        (exact terminal law, see sample_terminal).  With the same ``seed``,
        compute_var and compute_expected_shortfall use the same sample.
        """
        terminal_values = self.sample_terminal(x0, T, n_simulations, seed=seed)

        returns = terminal_values - x0
        var = -np.percentile(returns, (1 - confidence) * 100)

        return var

    def compute_expected_shortfall(self, x0: float, T: float = 1/252,
                                   confidence: float = 0.99,
                                   n_simulations: int = 10000,
                                   seed: Optional[int] = None) -> float:
        """Compute Expected Shortfall (CVaR) over the horizon T (same sample as compute_var)."""
        terminal_values = self.sample_terminal(x0, T, n_simulations, seed=seed)

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
                           n_steps: int = 252,
                           seed: Optional[int] = None) -> np.ndarray:
        """Simulate volatility path."""
        if seed is not None:
            np.random.seed(seed)
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
                       n_steps: int = 252, r: float = 0.0,
                       seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate price with stochastic volatility.

        dS/S = r dt + σ dW_S
        dσ = κ(θ - σ)dt + η√σ dW_σ
        """
        if seed is not None:
            np.random.seed(seed)
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

    An OU process is fitted (dt = 1 observation) to |returns| in each
    rolling window returns[i-window:i] (i >= 2·window; earlier entries are 0
    before normalisation).  Coordinates represent volatility dynamics:

    - X: long-term level μ of |r| (volatility level)
    - Y: current |r| (last value in the window)
    - Z: mean-reversion speed θ per observation

    |r| over a short window is often close to white noise, so the AR(1)
    slope b is frequently <= 0, which no OU process can produce (θ = ∞).
    Such windows get the white-noise limit θ = ln(1000) (b = 0.001, the
    same cap as JumpDiffusionRiskAnalyzer); windows without mean reversion
    (b >= 1) or with constant |r| get θ = 0.  Each coordinate is min-max
    normalised to [0, 1].

    High X, high Y, low Z = persistent high-volatility regime (danger)
    Low X, low Y, high Z = stable regime
    """
    n = len(returns)

    X = np.zeros(n)  # Mean reversion level
    Y = np.zeros(n)  # Current volatility
    Z = np.zeros(n)  # Reversion speed

    # Use absolute returns as volatility proxy
    abs_returns = np.abs(returns)
    max_theta = JumpDiffusionRiskAnalyzer.MAX_THETA_DT  # per observation (dt = 1)

    for i in range(2 * window, n):
        window_data = abs_returns[i-window:i]

        try:
            # Fit OU to volatility (b <= 0 is capped at the white-noise limit)
            ou = OrnsteinUhlenbeckProcess.fit(window_data, dt=1, max_theta=max_theta)

            X[i] = ou.mu  # Long-term volatility
            Y[i] = window_data[-1]  # Current volatility
            Z[i] = ou.theta  # Mean reversion speed

        except (ValueError, ArithmeticError):
            # No mean reversion (AR(1) slope >= 1) or constant window
            X[i] = np.mean(window_data)
            Y[i] = window_data[-1]
            Z[i] = 0.0

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

    #: Returns are daily observations; OU parameters are per year (dt = 1/252).
    DT = 1 / 252
    #: Cap on θ·dt for the OU fit of the continuous returns.  Daily returns are
    #: close to white noise, so the AR(1) slope b is often <= 0, which no OU
    #: process can produce (θ = ∞).  The fit then uses the white-noise limit
    #: b = 0.001 (θ = ln(1000)/dt ≈ 1741 per year, half-life ≈ 0.1 days).
    MAX_THETA_DT = np.log(1000.0)

    def __init__(self):
        self.ou_model = None
        self.jump_intensity = 0
        self.jump_mean = 0
        self.jump_std = 0

    def fit(self, returns: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        Fit jump-diffusion model to returns.

        Separates continuous and jump components.  ``jump_intensity`` is the
        jump frequency per observation (n_jumps / n); ``ou_fit_status`` is
        'ok', 'theta_capped' (b <= 0.001, white-noise limit), 'no_mean_reversion'
        (b >= 1 or constant data) or 'too_few_points' (<= 10 continuous points).
        """
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            raise ValueError("returns must not be empty")
        std = np.std(returns)
        mean = np.mean(returns)

        # Identify jumps (returns beyond threshold std)
        is_jump = np.abs(returns - mean) > threshold * std
        jumps = returns[is_jump]
        continuous = returns[~is_jump]

        # Fit OU to continuous part (never keep a model from an earlier call)
        self.ou_model = None
        ou_fit_status = 'too_few_points'
        if len(continuous) > 10:
            max_theta = self.MAX_THETA_DT / self.DT
            try:
                self.ou_model = OrnsteinUhlenbeckProcess.fit(
                    continuous, dt=self.DT, max_theta=max_theta)
                capped = self.ou_model.theta >= max_theta * (1 - 1e-9)
                ou_fit_status = 'theta_capped' if capped else 'ok'
            except ValueError:
                ou_fit_status = 'no_mean_reversion'

        # Jump statistics
        self.jump_intensity = np.sum(is_jump) / len(returns)
        self.jump_mean = np.mean(jumps) if len(jumps) > 0 else 0
        self.jump_std = np.std(jumps) if len(jumps) > 0 else 0

        return {
            'theta': self.ou_model.theta if self.ou_model else 0,
            'mu': self.ou_model.mu if self.ou_model else 0,
            'sigma': self.ou_model.sigma if self.ou_model else std,
            'jump_intensity': self.jump_intensity,
            'jump_mean': self.jump_mean,
            'jump_std': self.jump_std,
            'n_jumps': np.sum(is_jump),
            'jump_contribution': np.sum(np.abs(jumps)) / np.sum(np.abs(returns)) if len(jumps) > 0 else 0,
            'ou_fit_status': ou_fit_status,
        }

    def tail_risk_decomposition(self, returns: np.ndarray,
                                confidence: float = 0.99,
                                threshold: float = 3.0,
                                n_simulations: int = 10000,
                                seed: Optional[int] = None) -> Dict:
        """
        Decompose tail risk into continuous and jump components.

        One-day Monte Carlo VaR of the fitted OU-with-jumps model versus the
        same OU without jumps.  Both runs use the same random numbers for the
        diffusion part, so ``var_jump = var_total - var_continuous`` is the
        effect of the jumps, not Monte Carlo noise.  The per-observation jump
        frequency is converted to the per-year intensity λ = p / dt that
        OUWithJumps expects, and jumps are N(jump_mean, jump_std).
        """
        # Fit model
        fit_result = self.fit(returns, threshold=threshold)

        # Simulate to estimate risks
        if self.ou_model:
            dt = self.DT
            ou_jump = OUWithJumps(
                theta=fit_result['theta'],
                mu=fit_result['mu'],
                sigma=fit_result['sigma'],
                jump_intensity=fit_result['jump_intensity'] / dt,
                jump_mean=fit_result['jump_mean'],
                jump_std=fit_result['jump_std']
            )
            if seed is None:
                seed = int(np.random.randint(0, 2**31 - 1))

            # VaR decomposition
            result = ou_jump.simulate(0, dt, n_steps=1, n_paths=n_simulations, seed=seed)
            paths = result['paths'][:, -1]

            var_total = -np.percentile(paths, (1 - confidence) * 100)

            # Continuous-only VaR (common random numbers: same seed, so the
            # diffusion draws are identical to the run with jumps)
            ou_only = OrnsteinUhlenbeckProcess(
                fit_result['theta'],
                fit_result['mu'],
                fit_result['sigma']
            )
            continuous_paths = ou_only.simulate(0, dt, n_steps=1, n_paths=n_simulations, seed=seed)
            var_continuous = -np.percentile(continuous_paths[:, -1], (1 - confidence) * 100)

            var_jump = var_total - var_continuous

        else:
            var_total = np.percentile(-np.asarray(returns, dtype=float), confidence * 100)
            var_continuous = 0.0
            var_jump = var_total

        return {
            'var_total': var_total,
            'var_continuous': var_continuous,
            'var_jump': var_jump,
            'jump_contribution_pct': var_jump / var_total * 100 if var_total > 0 else 0,
            **fit_result
        }
