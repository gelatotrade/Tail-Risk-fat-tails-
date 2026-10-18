"""
Statistical Utilities for Tail Risk Analysis
=============================================
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


def compute_moments(data: np.ndarray) -> Dict[str, float]:
    """Compute all moments of a distribution."""
    return {
        'mean': np.mean(data),
        'variance': np.var(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }


def jarque_bera_test(data: np.ndarray) -> Dict[str, float]:
    """Test for normality using Jarque-Bera test."""
    stat, p_value = stats.jarque_bera(data)
    return {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }


def kolmogorov_smirnov_test(data: np.ndarray, distribution: str = 'norm',
                            params: Optional[tuple] = None) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov test against a distribution family.

    By default the parameters are estimated from the data, so the test asks
    "does the data look like *some* member of this family?":
    - 'norm': loc = mean, scale = std
    - any other scipy.stats continuous distribution name ('t', 'laplace',
      'cauchy', 'logistic', ...): maximum-likelihood fit (``dist.fit(data)``)

    Pass ``params`` (the distribution's shape/loc/scale arguments, e.g.
    ``params=()`` for the standard form) to test against fixed parameters.
    Note: with estimated parameters the classical KS p-value is conservative
    (too large); treat it as an indication, not an exact test.
    """
    data = np.asarray(data, dtype=float)
    dist = getattr(stats, distribution, None)
    if not isinstance(dist, stats.rv_continuous):
        raise ValueError(f"Unknown continuous distribution: {distribution!r}")

    if params is None:
        if distribution == 'norm':
            params = (np.mean(data), np.std(data))
        else:
            params = dist.fit(data)

    # A frozen distribution keeps loc/scale working across SciPy versions
    # (SciPy >= 1.18 maps the string 'norm' to a fast path without loc/scale).
    fitted = dist(*params)
    stat, p_value = stats.kstest(data, fitted.cdf)

    return {
        'statistic': stat,
        'p_value': p_value,
        'matches_distribution': p_value > 0.05
    }


def bootstrap_confidence_interval(data: np.ndarray, statistic_func,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return lower, upper
