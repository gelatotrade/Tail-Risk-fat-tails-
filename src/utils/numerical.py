"""
Numerical Utilities for Tail Risk Calculations
===============================================
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def safe_divide(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Safe division handling zeros."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill
    return result


def rolling_window(data: np.ndarray, window: int) -> np.ndarray:
    """Create rolling window view of data."""
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def exponential_weighted_mean(data: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    """Compute exponentially weighted moving average."""
    n = len(data)
    weights = (1 - alpha) ** np.arange(n)[::-1]
    weights /= weights.sum()

    result = np.zeros(n)
    for i in range(n):
        result[i] = np.sum(data[:i+1] * weights[-(i+1):])

    return result


def numerical_gradient(f, x: float, h: float = 1e-6) -> float:
    """Compute numerical gradient using central differences."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_hessian(f, x: float, h: float = 1e-6) -> float:
    """Compute numerical second derivative."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
