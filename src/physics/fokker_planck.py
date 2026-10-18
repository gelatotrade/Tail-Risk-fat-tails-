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

import math

import numpy as np
from scipy import sparse, linalg
import scipy.sparse.linalg  # noqa: F401  (explicit: do not rely on lazy loading)
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
                              params: Optional[dict] = None,
                              include_diffusion: bool = True) -> sparse.csr_matrix:
        """
        Build the discrete Fokker-Planck operator as a sparse matrix.

        Uses finite differences of the forward (Kolmogorov) equation in
        conservative form:
        - Drift term: Central differences of the flux μP
        - Diffusion term: Second-order central differences of DP

        Row j:  -(μ_{j+1}P_{j+1} - μ_{j-1}P_{j-1}) / (2dx)
                + (D_{j+1}P_{j+1} - 2 D_j P_j + D_{j-1}P_{j-1}) / dx²
        so every interior column sums to zero (probability is conserved).

        Args:
            t: Time (passed to the coefficient functions)
            params: Parameters for drift and diffusion
            include_diffusion: False builds only the drift part -∂(μP)/∂x
                (used together with the fractional operator)

        Returns:
            Sparse matrix L such that dP/dt = L @ P
        """
        n = self.n_grid
        dx = self.dx

        mu = self.drift_coefficient(self.x, t, params)
        D = self.diffusion_coefficient(self.x, t, params)
        if not include_diffusion:
            D = np.zeros_like(self.x, dtype=float)

        # Build tridiagonal components
        # dP/dt = -d(μP)/dx + d²(DP)/dx²

        # Main diagonal: L[j, j]
        main_diag = -2 * D / dx**2

        # Upper diagonal: L[j, j+1] acts on P_{j+1} -> coefficients at j+1
        upper_diag = D[1:] / dx**2 - mu[1:] / (2 * dx)

        # Lower diagonal: L[j, j-1] acts on P_{j-1} -> coefficients at j-1
        lower_diag = D[:-1] / dx**2 + mu[:-1] / (2 * dx)

        # Create sparse matrix
        diagonals = [lower_diag, main_diag, upper_diag]
        L = sparse.diags(diagonals, [-1, 0, 1], format='csr')

        return L

    def build_fractional_operator(self, alpha: float,
                                  params: Optional[dict] = None) -> np.ndarray:
        """
        Build fractional diffusion operator for fat-tail modeling.

        The fractional Laplacian -(-Δ)^(α/2) (= Riesz derivative ∂^α/∂|x|^α)
        is defined via its Fourier transform: symbol -|k|^α.
        In real space, it's a non-local operator capturing long-range correlations.

        Discretisation: fractional centred difference (Ortigueira 2006;
        Çelik & Duman 2012), valid for 0 < α ≤ 2:
            ∂^α P/∂|x|^α (x_i) ≈ -dx^(-α) Σ_k g_k P_{i-k},
            g_k = (-1)^k Γ(α+1) / (Γ(α/2-k+1) Γ(α/2+k+1)).
        g_0 > 0, g_k ≤ 0 for k ≠ 0 and Σ g_k = 0, so the operator is symmetric,
        negative semi-definite (diffusive) and mass conserving. α = 2 gives the
        standard Laplacian (P_{i+1} - 2P_i + P_{i-1}) / dx². The full kernel is
        used (no truncation), so long Lévy jumps across the grid are kept.

        For α < 2, this produces Lévy-type fat tails.

        Returns:
            Dense (n_grid, n_grid) matrix D_alpha * ∂^α/∂|x|^α
        """
        if not 0 < alpha <= 2:
            raise ValueError("alpha must be in (0, 2]")
        n = self.n_grid
        dx = self.dx

        params = params or {}
        D_alpha = params.get('D_alpha', 0.01)

        # Coefficients via the recurrence g_{k+1} = g_k (k - α/2) / (k + α/2 + 1)
        # (exact zeros for k > 1 at α = 2, no Gamma-function poles)
        g = np.empty(n)
        g[0] = gamma_func(alpha + 1) / gamma_func(alpha / 2 + 1)**2
        k = np.arange(n - 1, dtype=float)
        g[1:] = g[0] * np.cumprod((k - alpha / 2) / (k + alpha / 2 + 1))

        # Symmetric Toeplitz matrix: L[i, j] = -g_|i-j| / dx^α
        L_frac = -linalg.toeplitz(g)

        L_frac *= D_alpha / dx**alpha

        return L_frac

    def solve(self, P0: np.ndarray, T: float, dt: float = 0.001,
              params: Optional[dict] = None,
              use_fractional: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Fokker-Planck equation forward in time.

        Uses Crank-Nicolson scheme for stability:
        (I - dt/2 * L) P^{n+1} = (I + dt/2 * L) P^n

        Standard mode: L = -∂(μ·)/∂x + ∂²(D·)/∂x².
        Fractional mode (use_fractional and alpha < 2):
        L = -∂(μ·)/∂x + D_α ∂^α/∂|x|^α.

        The number of steps is round(T / dt) and dt is adjusted slightly so
        that the last step lands exactly on T. Frames are stored about every
        n_steps/100 steps; the final state at time T is always the last frame.

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
        n_steps = int(round(T / dt)) if dt > 0 else 0
        if n_steps > 0:
            dt = T / n_steps
        P = np.asarray(P0, dtype=float).copy()

        # Storage for evolution
        times = [0]
        P_evolution = [P.copy()]

        if n_steps <= 0:
            return np.array(times, dtype=float), np.array(P_evolution)

        # Build operator
        if use_fractional and self.alpha < 2:
            L_drift = self.build_operator_matrix(0, params, include_diffusion=False)
            L = self.build_fractional_operator(self.alpha, params) + L_drift.toarray()
        else:
            L = self.build_operator_matrix(0, params)

        # Crank-Nicolson matrices; the implicit matrix is factorised once
        # (SuperLU, also for the dense fractional operator: no BLAS threads)
        L = sparse.csc_matrix(L)
        I = sparse.eye(self.n_grid, format='csc')
        A = (I - dt/2 * L).tocsc()  # Implicit part
        B = (I + dt/2 * L).tocsr()  # Explicit part
        solve_A = sparse.linalg.splu(A).solve

        stride = max(1, n_steps // 100)

        # Time stepping
        for step in range(n_steps):
            # Solve linear system
            rhs = B @ P
            P = solve_A(rhs)

            # Ensure non-negativity and normalization
            P = np.maximum(P, 0)
            P = P / (np.sum(P) * self.dx + 1e-10)

            # Apply boundary conditions (zero at boundaries)
            P[0] = P[-1] = 0

            # Store periodically (and always the final state at t = T)
            if step % stride == 0 or step == n_steps - 1:
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

        The tail mass is the integral of the (piecewise linear) density beyond
        mean ± threshold·std, interpolated inside the grid cell that contains
        the threshold, so it varies continuously with the threshold.

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

        # Integrate tails: cumulative trapezoid of P, linearly interpolated
        P = np.asarray(P, dtype=float)
        cum = np.concatenate(([0.0], np.cumsum((P[1:] + P[:-1]) * 0.5 * self.dx)))
        total = cum[-1]

        left_prob = float(np.interp(x_left, self.x, cum))
        right_prob = float(total - np.interp(x_right, self.x, cum))

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
                'ratio_to_gaussian': (left + right) / (2 * (1 - 0.5 * (1 + math.erf(sigma / math.sqrt(2)))))
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
