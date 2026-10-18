"""
Data Loading and Generation Utilities
======================================

Functions for loading financial data and generating synthetic
data for testing tail risk models.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings


def generate_synthetic_returns(n: int = 2520, model: str = 'levy_jump',
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic return data with realistic fat tails.

    Models available:
    - 'gaussian': Standard normal (baseline)
    - 'student_t': Fat-tailed Student-t
    - 'levy': Lévy stable (very fat tails)
    - 'garch': GARCH(1,1) volatility clustering
    - 'levy_jump': Gaussian + Lévy jumps (most realistic)
    - 'regime_switching': Alternating volatility regimes

    Args:
        n: Number of observations
        model: Model type for generation
        seed: Random seed

    Returns:
        Array of synthetic returns
    """
    if seed is not None:
        np.random.seed(seed)

    daily_vol = 0.01  # 1% daily vol (16% annualized)
    daily_drift = 0.0003  # ~8% annual return

    if model == 'gaussian':
        returns = np.random.normal(daily_drift, daily_vol, n)

    elif model == 'student_t':
        from scipy import stats
        returns = stats.t.rvs(df=4, loc=daily_drift, scale=daily_vol * 0.7, size=n)

    elif model == 'levy':
        from ..physics.levy_flight import LevyStableDistribution
        levy = LevyStableDistribution(alpha=1.7, beta=-0.1,
                                      gamma=daily_vol * 0.3, delta=daily_drift)
        returns = levy.sample(n)

    elif model == 'garch':
        # GARCH(1,1)
        omega = daily_vol**2 * 0.05
        alpha = 0.10
        beta = 0.85

        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        if n > 0:
            sigma2[0] = daily_vol**2

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = daily_drift + np.sqrt(sigma2[t]) * np.random.normal()

    elif model == 'levy_jump':
        # Gaussian base + Lévy jumps (most realistic)
        base_returns = np.random.normal(daily_drift, daily_vol * 0.8, n)

        # Add jumps
        jump_intensity = 0.02  # 2% chance of jump per day
        jump_times = np.random.random(n) < jump_intensity

        # Jump sizes from heavy-tailed distribution
        from scipy import stats
        jump_sizes = stats.t.rvs(df=3, scale=daily_vol * 3, size=n)
        # Bias jumps slightly negative (crashes more common than rallies)
        jump_sizes = jump_sizes - 0.003

        returns = base_returns + jump_times * jump_sizes

    elif model == 'regime_switching':
        # Two regimes: low vol and high vol
        returns = np.zeros(n)
        regime = 0  # Start in low vol

        low_vol = daily_vol * 0.7
        high_vol = daily_vol * 2.5

        p_low_to_high = 0.02
        p_high_to_low = 0.05

        for t in range(n):
            if regime == 0:  # Low vol
                returns[t] = np.random.normal(daily_drift, low_vol)
                if np.random.random() < p_low_to_high:
                    regime = 1
            else:  # High vol
                returns[t] = np.random.normal(daily_drift * 0.5, high_vol)
                if np.random.random() < p_high_to_low:
                    regime = 0

    else:
        raise ValueError(f"Unknown model: {model}")

    return returns


def generate_crisis_scenario(n_pre: int = 500, n_crisis: int = 100,
                            n_post: int = 400, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    Generate synthetic data with a market crisis.

    Creates realistic pre-crisis, crisis, and post-crisis dynamics:
    - Pre-crisis: Low volatility, complacency
    - Crisis: Sharp selloff, volatility spike, fat tails
    - Post-crisis: Elevated volatility, gradual recovery

    Args:
        n_pre: Days before crisis
        n_crisis: Days during crisis
        n_post: Days after crisis
        seed: Random seed

    Returns:
        Tuple of (returns, metadata)
    """
    if seed is not None:
        np.random.seed(seed)

    returns = []

    # Pre-crisis: Low volatility, slight uptrend
    pre_crisis = np.random.normal(0.0005, 0.006, n_pre)
    returns.extend(pre_crisis)

    # Crisis: Sharp selloff with volatility spike. The three phases keep the
    # proportions 10% / 40% / 50% of n_crisis (= 10 / 40 / 50 days for the default 100).
    n_crisis = max(int(n_crisis), 0)
    n_shock = min(n_crisis, max(1, int(round(0.1 * n_crisis)))) if n_crisis > 0 else 0
    n_main = min(n_crisis - n_shock, int(round(0.4 * n_crisis)))
    n_late = n_crisis - n_shock - n_main

    # Day 1-10: Initial shock
    crisis_start = np.random.normal(-0.02, 0.03, n_shock)

    # Day 11-50: Main crisis (very high vol, negative skew)
    from scipy import stats
    crisis_main = stats.t.rvs(df=3, loc=-0.005, scale=0.04, size=n_main)

    # Day 51-100: Volatility remains high but mean recovers
    crisis_late = np.random.normal(0.001, 0.025, n_late)

    crisis = np.concatenate([crisis_start, crisis_main, crisis_late])
    returns.extend(crisis)

    # Post-crisis: Gradual return to normalcy
    post_vols = np.linspace(0.02, 0.01, n_post)
    post_crisis = np.array([np.random.normal(0.0003, vol) for vol in post_vols])
    returns.extend(post_crisis)

    returns = np.array(returns)

    metadata = {
        'crisis_start': n_pre,
        'crisis_end': n_pre + n_crisis,
        'crisis_peak': n_pre + int(round(0.25 * n_crisis)),  # Maximum drawdown point (day 25 of 100)
        'total_length': len(returns)
    }

    return returns, metadata


def generate_vix_from_returns(returns: np.ndarray, window: int = 20,
                              base_level: float = 15,
                              sensitivity: float = 100) -> np.ndarray:
    """
    Generate synthetic VIX-like index from returns.

    VIX tends to:
    - Spike during selloffs
    - Mean-revert during calm periods
    - Be negatively correlated with returns

    Args:
        returns: Return series
        window: Window for volatility calculation
        base_level: Base VIX level
        sensitivity: How much VIX responds to volatility

    Returns:
        Synthetic VIX series
    """
    n = len(returns)
    vix = np.zeros(n)

    # Rolling realized volatility (annualized)
    for i in range(n):
        start = max(0, i - window)
        vol = np.std(returns[start:i+1]) * np.sqrt(252)
        vix[i] = base_level + sensitivity * vol

    # Add mean reversion dynamics
    mean_vix = base_level + sensitivity * np.std(returns) * np.sqrt(252)
    for i in range(1, n):
        vix[i] = 0.95 * vix[i] + 0.05 * mean_vix

    # Spike on large negative returns
    large_negatives = returns < -0.02
    vix[large_negatives] *= 1.5

    return np.clip(vix, 10, 80)


def load_sample_data() -> Dict[str, np.ndarray]:
    """
    Load sample data for demonstration.

    Returns dictionary with multiple synthetic series.
    """
    np.random.seed(42)

    data = {}

    # Normal market conditions
    data['normal'] = generate_synthetic_returns(1000, 'garch', seed=42)

    # With crisis
    crisis_returns, crisis_meta = generate_crisis_scenario(seed=43)
    data['crisis'] = crisis_returns
    data['crisis_metadata'] = crisis_meta

    # Very fat tails (Lévy)
    data['fat_tails'] = generate_synthetic_returns(1000, 'levy', seed=44)

    # Regime switching
    data['regime_switching'] = generate_synthetic_returns(1000, 'regime_switching', seed=45)

    # Generate VIX
    data['vix_normal'] = generate_vix_from_returns(data['normal'])
    data['vix_crisis'] = generate_vix_from_returns(data['crisis'])

    return data


class SyntheticMarketGenerator:
    """
    Generate realistic multi-asset synthetic market data.
    """

    def __init__(self, n_assets: int = 5, seed: Optional[int] = None,
                 base_correlation: float = 0.0):
        """
        Initialize generator.

        Args:
            n_assets: Number of assets to generate
            seed: Random seed
            base_correlation: Common market correlation in [0, 1). The random
                correlation matrix is blended with a constant one,
                ``rho + (1 - rho) * C``, so every pair gets at least roughly
                ``rho`` (real equities: about 0.3-0.5). 0 (default) keeps the
                original matrix, whose pair correlations centre on zero.
        """
        if not 0.0 <= base_correlation < 1.0:
            raise ValueError(f"base_correlation must be in [0, 1), got {base_correlation}")
        self.n_assets = n_assets
        self.seed = seed
        self.base_correlation = float(base_correlation)

        if seed is not None:
            np.random.seed(seed)

    def generate(self, n_days: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate multi-asset returns with realistic correlations.

        Returns dictionary with:
        - 'returns': (n_days, n_assets) array
        - 'correlations': correlation matrix
        - 'vix': VIX-like index
        """
        # Generate correlation matrix
        # Start with random matrix and make it positive definite
        A = np.random.randn(self.n_assets, self.n_assets) * 0.3
        cov = np.eye(self.n_assets) * 0.5 + A @ A.T

        # Convert to correlation
        D = np.diag(1 / np.sqrt(np.diag(cov)))
        corr = D @ cov @ D
        if self.base_correlation > 0:
            # convex blend with a constant-correlation matrix: still a valid
            # (positive definite) correlation matrix, no extra random draws
            rho = self.base_correlation
            corr = rho * np.ones_like(corr) + (1 - rho) * corr
            np.fill_diagonal(corr, 1.0)

        # Generate correlated returns using Cholesky
        L = np.linalg.cholesky(corr)
        independent = np.random.randn(n_days, self.n_assets)

        # Base returns
        correlated = independent @ L.T

        # Add individual characteristics
        vols = np.random.uniform(0.01, 0.02, self.n_assets)
        drifts = np.random.uniform(0.0001, 0.0005, self.n_assets)

        returns = correlated * vols + drifts

        # Add fat tails through t-distribution scaling
        from scipy import stats
        t_multiplier = stats.t.rvs(df=5, size=(n_days, self.n_assets))
        returns = returns * (1 + 0.3 * (np.abs(t_multiplier) - 1))

        # Market index (weighted average)
        weights = np.ones(self.n_assets) / self.n_assets
        market = returns @ weights

        # VIX from market
        vix = generate_vix_from_returns(market)

        return {
            'returns': returns,
            'market': market,
            'correlations': corr,
            'vix': vix,
            'volatilities': vols,
            'drifts': drifts
        }
