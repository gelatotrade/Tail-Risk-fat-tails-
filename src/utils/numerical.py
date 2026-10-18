"""
Numerical Utilities for Tail Risk Calculations
===============================================
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Safe division handling zeros (works for arrays and scalars)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.asarray(np.divide(a, b), dtype=float)
        result = np.where(np.isfinite(result), result, fill)
    # Scalars in -> NumPy scalar out (np.where always returns an array)
    return result[()] if result.ndim == 0 else result


def rolling_window(data: np.ndarray, window: int) -> np.ndarray:
    """
    Create rolling window view of data (along the last axis).

    The view is read-only: with ``as_strided`` writes into one window would
    silently alias the source and the neighbouring windows.
    """
    data = np.asarray(data)
    if window < 1 or window > data.shape[-1]:
        raise ValueError(
            f"window must be between 1 and {data.shape[-1]} (the data length), got {window}"
        )
    return np.lib.stride_tricks.sliding_window_view(data, window, axis=-1)


def exponential_weighted_mean(data: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    """
    Compute exponentially weighted moving average.

    result[t] = sum_{k<=t} (1-alpha)^(t-k) x_k / sum_{k<=t} (1-alpha)^(t-k)

    i.e. the weights are normalised over the observations available at time t
    (identical to ``pandas.Series.ewm(alpha=alpha, adjust=True).mean()``).
    Normalising over the full length instead biases every value but the last
    one towards zero.
    """
    x = np.asarray(data, dtype=float)
    decay = 1.0 - alpha
    result = np.empty(len(x))
    num = 0.0
    den = 0.0
    for i in range(len(x)):  # O(n) recursion
        num = x[i] + decay * num
        den = 1.0 + decay * den
        result[i] = num / den

    return result


def numerical_gradient(f, x: float, h: float = 1e-6) -> float:
    """Compute numerical gradient using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_hessian(f, x: float, h: Optional[float] = None) -> float:
    """
    Compute numerical second derivative (central second difference).

    Args:
        f: Scalar function
        x: Point of evaluation
        h: Step size. Default (None) = eps**(1/4) * max(1, |x|) ~ 1.2e-4 * max(1, |x|),
           the step that balances truncation error O(h^2) against round-off
           O(eps * |f| / h^2). A fixed tiny step such as 1e-6 loses all accuracy
           for functions of magnitude 1e3-1e4 (e.g. log-likelihoods).
    """
    if h is None:
        h = np.finfo(float).eps ** 0.25 * max(1.0, abs(float(x)))
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
