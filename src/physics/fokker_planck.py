"""
Fokker-Planck Equation for Financial Probability Evolution
==========================================================

The Fokker-Planck equation (FPE) describes the time evolution of probability
density functions. In physics, it models diffusion processes; in finance,
it models how the distribution of asset prices evolves over time.

Physics Background:
- Also known as the Kolmogorov Forward Equation
- Describes probability density evolution: ∂P/∂t = -∂(μP)/∂x + ∂²(DP)/∂x²
- μ(x,t): Drift coefficient (deterministic trend)
- D(x,t): Diffusion coefficient (volatility/uncertainty)

Financial Interpretation:
- P(x,t): Probability of price being at x at time t
- μ: Expected return (drift)
- D: Volatility (diffusion)

Key Innovation for Fat Tails:
We extend the standard FPE with:
1. State-dependent diffusion (volatility clustering)
2. Lévy noise term (fat tails from jumps)
3. Non-local operators (memory effects)

This creates a "Fractional Fokker-Planck Equation" that generates fat tails.
"""

import numpy as np
from scipy import sparse, linalg
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional, Callable
import warnings


class FokkerPlanckSolver:
    """
    Numerical solver for the Fokker-Planck equation with fat-tail extensions.

    Standard FPE:
    ∂P/∂t = -∂(μ(x)P)/∂x + ∂²(D(x)P)/∂x²

    Fractional FPE (for fat tails):
    ∂P/∂t = -∂(μ(x)P)/∂x + D_α ∂^α P/∂|x|^α

    where α < 2 produces fat tails (Lévy-type diffusion).
    """

    def __init__(self, x_min: float = -10, x_max: float = 10,
                 n_grid: int = 500, alpha: float = 2.0):
        """
        Initialize the Fokker-Planck solver.

        Args:
            x_min: Minimum x value (log-return space)
            x_max: Maximum x value
            n_grid: Number of grid points
            alpha: Fractional order (2 = standard, <2 = fat tails)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_grid = n_grid
        self.alpha = alpha

        self.x = np.linspace(x_min, x_max, n_grid)
        self.dx = self.x[1] - self.x[0]

    def drift_coefficient(self, x: np.ndarray, t: float = 0,
                          params: Optional[dict] = None) -> np.ndarray:
        """
        Drift coefficient μ(x,t).

        Default: Mean-reverting Ornstein-Uhlenbeck drift
        μ(x) = -θ(x - μ_∞)

        This models returns reverting to a long-term mean.
        """
        params = params or {}
        theta = params.get('theta', 0.1)  # Reversion speed
        mu_inf = params.get('mu_inf', 0.0)  # Long-term mean

        return -theta * (x - mu_inf)

    def diffusion_coefficient(self, x: np.ndarray, t: float = 0,
                              params: Optional[dict] = None) -> np.ndarray:
        """
        Diffusion coefficient D(x,t).

        For volatility clustering, use state-dependent diffusion:
        D(x) = D_0 * (1 + λ|x|^β)

        When |x| is large (extreme returns), volatility increases.
        This creates volatility clustering and fat tails.
        """
        params = params or {}
        D0 = params.get('D0', 0.01)  # Base diffusion
        lam = params.get('lambda', 0.5)  # Clustering strength
        beta = params.get('beta', 1.0)  # Nonlinearity

        return D0 * (1 + lam * np.abs(x)**beta)

    def build_operator_matrix(self, t: float = 0,
                              params: Optional[dict] = None) -> sparse.csr_matrix:
        """
        Build the discrete Fokker-Planck operator as a sparse matrix.

        Uses finite differences:
        - Drift term: Central differences
        - Diffusion term: Second-order central differences

        Returns:
            Sparse matrix L such that dP/dt = L @ P
        """
        n = self.n_grid
        dx = self.dx

        mu = self.drift_coefficient(self.x, t, params)
        D = self.diffusion_coefficient(self.x, t, params)

        # Build tridiagonal components
        # dP/dt = -d(μP)/dx + d²(DP)/dx²

        # Main diagonal
        main_diag = -2 * D / dx**2

        # Upper diagonal (j+1 terms)
        upper_diag = D[:-1] / dx**2 + mu[:-1] / (2 * dx)

        # Lower diagonal (j-1 terms)
        lower_diag = D[1:] / dx**2 - mu[1:] / (2 * dx)

        # Create sparse matrix
        diagonals = [lower_diag, main_diag, upper_diag]
        L = sparse.diags(diagonals, [-1, 0, 1], format='csr')

        return L

    def build_fractional_operator(self, alpha: float,
                                  params: Optional[dict] = None) -> np.ndarray:
        """
        Build fractional diffusion operator for fat-tail modeling.

        The fractional Laplacian -(-Δ)^(α/2) is defined via its Fourier transform.
        In real space, it's a non-local operator capturing long-range correlations.

        For α < 2, this produces Lévy-type fat tails.
        """
        n = self.n_grid
        dx = self.dx

        params = params or {}
        D_alpha = params.get('D_alpha', 0.01)

        # Build fractional Laplacian matrix (Grünwald-Letnikov approximation)
        L_frac = np.zeros((n, n))

        # Coefficients for fractional derivative
        def grunwald_coeff(k, alpha):
            """Grünwald-Letnikov coefficients."""
            if k == 0:
                return 1.0
            return (-1)**k * gamma_func(alpha + 1) / (gamma_func(k + 1) * gamma_func(alpha - k + 1))

        # Number of terms in approximation
        m = min(20, n // 2)

        for i in range(n):
            for k in range(m):
                if i - k >= 0:
                    L_frac[i, i-k] += grunwald_coeff(k, alpha)
                if i + k < n:
                    L_frac[i, i+k] += grunwald_coeff(k, alpha)

        L_frac *= D_alpha / (2 * dx**alpha)

        return L_frac

    def solve(self, P0: np.ndarray, T: float, dt: float = 0.001,
              params: Optional[dict] = None,
              use_fractional: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Fokker-Planck equation forward in time.

        Uses Crank-Nicolson scheme for stability:
        (I - dt/2 * L) P^{n+1} = (I + dt/2 * L) P^n

        Args:
            P0: Initial probability distribution
            T: Final time
            dt: Time step
            params: Parameters for drift and diffusion
            use_fractional: Use fractional operator for fat tails

        Returns:
            Tuple of (time_points, probability_evolution)
            probability_evolution has shape (n_times, n_grid)
        """
        n_steps = int(T / dt)
        P = P0.copy()

        # Storage for evolution
        times = [0]
        P_evolution = [P.copy()]

        # Build operator
        if use_fractional and self.alpha < 2:
            L = self.build_fractional_operator(self.alpha, params)
            L = sparse.csr_matrix(L)
        else:
            L = self.build_operator_matrix(0, params)

        # Identity matrix
        I = sparse.eye(self.n_grid)

        # Crank-Nicolson matrices
        A = I - dt/2 * L  # Implicit part
        B = I + dt/2 * L  # Explicit part

        # Time stepping
        for step in range(n_steps):
            # Solve linear system
            rhs = B @ P
            P = sparse.linalg.spsolve(A, rhs)

            # Ensure non-negativity and normalization
            P = np.maximum(P, 0)
            P = P / (np.sum(P) * self.dx + 1e-10)

            # Apply boundary conditions (zero at boundaries)
            P[0] = P[-1] = 0

            # Store periodically
            if step % max(1, n_steps // 100) == 0:
                times.append((step + 1) * dt)
                P_evolution.append(P.copy())

        return np.array(times), np.array(P_evolution)

    def compute_moments(self, P: np.ndarray) -> dict:
        """
        Compute moments of the distribution.

        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        x = self.x
        dx = self.dx

        # Normalize
        norm = np.sum(P) * dx
        P_norm = P / (norm + 1e-10)

        # Mean
        mean = np.sum(x * P_norm) * dx

        # Variance
        var = np.sum((x - mean)**2 * P_norm) * dx

        # Skewness
        std = np.sqrt(var)
        skew = np.sum(((x - mean) / (std + 1e-10))**3 * P_norm) * dx

        # Kurtosis (excess)
        kurt = np.sum(((x - mean) / (std + 1e-10))**4 * P_norm) * dx - 3

        return {
            'mean': mean,
            'variance': var,
            'std': std,
            'skewness': skew,
            'kurtosis': kurt
        }

    def tail_probability(self, P: np.ndarray, threshold: float) -> Tuple[float, float]:
        """
        Compute left and right tail probabilities.

        Args:
            P: Probability distribution
            threshold: Number of standard deviations

        Returns:
            Tuple of (left_tail_prob, right_tail_prob)
        """
        moments = self.compute_moments(P)
        mean = moments['mean']
        std = moments['std']

        x_left = mean - threshold * std
        x_right = mean + threshold * std

        # Integrate tails
        left_mask = self.x < x_left
        right_mask = self.x > x_right

        left_prob = np.sum(P[left_mask]) * self.dx
        right_prob = np.sum(P[right_mask]) * self.dx

        return left_prob, right_prob


class FokkerPlanckTailRisk:
    """
    Fokker-Planck based tail risk analyzer.

    Uses the FPE framework to:
    1. Model probability evolution with fat tails
    2. Compute forward-looking risk measures
    3. Detect regime changes through distribution dynamics
    """

    def __init__(self, alpha: float = 1.7, theta: float = 0.1):
        """
        Initialize tail risk analyzer.

        Args:
            alpha: Fractional order (< 2 for fat tails)
            theta: Mean reversion speed
        """
        self.alpha = alpha
        self.theta = theta
        self.solver = FokkerPlanckSolver(x_min=-10, x_max=10, alpha=alpha)

    def initial_distribution(self, current_return: float = 0,
                            current_vol: float = 1.0) -> np.ndarray:
        """
        Create initial distribution centered at current state.
        """
        x = self.solver.x
        P0 = np.exp(-0.5 * ((x - current_return) / current_vol)**2)
        P0 /= np.sum(P0) * self.solver.dx
        return P0

    def forecast_distribution(self, P0: np.ndarray, horizon: int = 20,
                             vol_regime: float = 1.0) -> np.ndarray:
        """
        Forecast the probability distribution forward in time.

        Args:
            P0: Initial distribution
            horizon: Forecast horizon (days)
            vol_regime: Current volatility regime multiplier

        Returns:
            Forecast distribution at horizon
        """
        params = {
            'theta': self.theta,
            'mu_inf': 0.0,
            'D0': 0.01 * vol_regime**2,
            'lambda': 0.5,
            'D_alpha': 0.01 * vol_regime**2
        }

        times, P_evolution = self.solver.solve(
            P0, T=horizon/252, dt=0.001,
            params=params, use_fractional=True
        )

        return P_evolution[-1]

    def compute_tail_risk_metrics(self, P: np.ndarray) -> dict:
        """
        Compute comprehensive tail risk metrics from distribution.
        """
        moments = self.solver.compute_moments(P)
        std = moments['std']

        # Tail probabilities at various thresholds
        tail_probs = {}
        for sigma in [2, 3, 4, 5]:
            left, right = self.solver.tail_probability(P, sigma)
            tail_probs[f'{sigma}sigma'] = {
                'left': left,
                'right': right,
                'total': left + right,
                'ratio_to_gaussian': (left + right) / (2 * (1 - 0.5 * (1 + np.math.erf(sigma / np.sqrt(2)))))
            }

        return {
            'moments': moments,
            'tail_probabilities': tail_probs,
            'tail_index': self.alpha,
            'fat_tail_factor': 2 / self.alpha  # Higher = fatter tails
        }

    def generate_3d_risk_surface(self, vol_range: np.ndarray,
                                 horizon_range: np.ndarray,
                                 threshold_sigma: float = 3) -> np.ndarray:
        """
        Generate 3D tail risk surface.

        Axes:
        - X: Volatility regime
        - Y: Time horizon
        - Z: Tail probability (risk)

        This creates a "risk landscape" showing how tail risk varies
        with volatility and time.
        """
        Z = np.zeros((len(vol_range), len(horizon_range)))

        P0 = self.initial_distribution(0, 1)

        for i, vol in enumerate(vol_range):
            for j, horizon in enumerate(horizon_range):
                P_forecast = self.forecast_distribution(P0, int(horizon), vol)
                left, right = self.solver.tail_probability(P_forecast, threshold_sigma)
                Z[i, j] = left + right  # Total tail probability

        return Z


def compute_fokker_planck_coordinates(returns: np.ndarray,
                                     window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform returns into 3D Fokker-Planck phase space.

    Coordinates represent the probability evolution state:
    - X: Drift magnitude (μ estimate)
    - Y: Diffusion magnitude (D estimate)
    - Z: Fractional order (α estimate from tail behavior)

    Args:
        returns: Time series of returns
        window: Rolling window for estimation

    Returns:
        Tuple of (X, Y, Z) coordinates
    """
    n = len(returns)

    X = np.zeros(n)  # Drift
    Y = np.zeros(n)  # Diffusion
    Z = np.zeros(n)  # Fractional order

    for i in range(window, n):
        window_data = returns[i-window:i]

        # Estimate drift (mean)
        X[i] = np.mean(window_data)

        # Estimate diffusion (variance)
        Y[i] = np.var(window_data)

        # Estimate fractional order from kurtosis
        kurt = np.mean((window_data - np.mean(window_data))**4) / np.var(window_data)**2
        # Higher kurtosis -> lower alpha (fatter tails)
        Z[i] = np.clip(2 * 3 / kurt, 1.0, 2.0) if kurt > 0 else 2.0

    return X, Y, Z
