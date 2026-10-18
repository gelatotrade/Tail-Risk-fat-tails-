"""Regression tests for TailRiskMetrics.sortino_ratio (src/models/risk_metrics.py)."""

import numpy as np
import pytest

from src.models.risk_metrics import TailRiskMetrics


def test_sortino_uses_target_semideviation_over_all_days():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0])
    target = 0.0
    shortfall = np.minimum(r - target, 0.0)
    expected = (r.mean() - target) / np.sqrt(np.mean(shortfall**2)) * np.sqrt(252)
    assert TailRiskMetrics(r).sortino_ratio(target) == pytest.approx(expected, rel=1e-12)


def test_sortino_of_normal_returns_matches_closed_form_and_exceeds_sharpe():
    # X ~ N(mu, sigma²): E[min(X, 0)²] = (mu² + sigma²) Φ(-mu/sigma) - mu sigma φ(mu/sigma),
    # about sigma²/2, so Sortino ≈ sqrt(2) · Sharpe. Averaging only over the losing days gave
    # about sigma² and a ratio *below* Sharpe.
    from scipy import stats

    mu, sigma = 0.0004, 0.01
    r = mu + sigma * np.random.default_rng(3).standard_normal(400_000)
    m = TailRiskMetrics(r)
    semivar = (mu**2 + sigma**2) * stats.norm.cdf(-mu / sigma) - mu * sigma * stats.norm.pdf(mu / sigma)
    expected = mu / np.sqrt(semivar) * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252)
    assert m.sortino_ratio(0.0) == pytest.approx(expected, rel=0.05)
    assert m.sortino_ratio(0.0) > 1.3 * sharpe


def test_sortino_target_shifts_shortfall():
    r = np.array([0.01, 0.02, 0.03, 0.04])
    m = TailRiskMetrics(r)
    assert m.sortino_ratio(0.0) == np.inf  # no day below target
    target = 0.025
    shortfall = np.minimum(r - target, 0.0)
    expected = (r.mean() - target) / np.sqrt(np.mean(shortfall**2)) * np.sqrt(252)
    assert m.sortino_ratio(target) == pytest.approx(expected, rel=1e-12)
