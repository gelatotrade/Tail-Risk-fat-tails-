"""compute_phase_space_coordinates was vectorised for the browser; it must equal the original loop."""

from typing import Tuple

import numpy as np
import pytest

from src.physics.phase_transitions import compute_phase_space_coordinates
from src.utils.data_loader import generate_crisis_scenario, generate_synthetic_returns


def _reference(returns: np.ndarray, window: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The original per-window implementation (commit 7159919)."""
    n = len(returns)
    X, Y, Z = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(2 * window, n):
        w = returns[i - window:i]
        X[i] = np.var(np.cumsum(w)) / (np.var(w) + 1e-10)
        if np.std(w) > 0:
            Y[i] = np.abs(np.corrcoef(w[:-1], w[1:])[0, 1])
        Z[i] = 1 / (X[i] + 0.1)
    return X / (np.max(X) + 1e-10), Y / (np.max(Y) + 1e-10), Z / (np.max(Z) + 1e-10)


@pytest.mark.parametrize("window", [3, 10, 50, 120])
@pytest.mark.parametrize("kind", ["levy_jump", "garch", "crisis", "flat_patch"])
def test_vectorised_equals_reference(kind, window):
    if kind == "crisis":
        r = np.asarray(generate_crisis_scenario(seed=5)[0], dtype=float)
    else:
        r = np.asarray(generate_synthetic_returns(900, "garch" if kind == "garch" else "levy_jump", seed=11), dtype=float)
    if kind == "flat_patch":
        r[300:420] = 0.0  # constant windows: std == 0 branch
    with np.errstate(all="ignore"):
        got = compute_phase_space_coordinates(r, window)
        ref = _reference(r, window)
    for g, e in zip(got, ref):
        assert np.allclose(g, e, atol=1e-12, equal_nan=True)
