"""IsingMarketModel.susceptibility and phase_diagram must use the same (per-agent) normalisation."""

import numpy as np
import pytest

from src.physics.phase_transitions import IsingMarketModel


def test_snapshot_susceptibility_is_per_agent_curie_law():
    for n in (12, 80, 400):
        model = IsingMarketModel(n_agents=n, J=1.0, T=1.6)
        model.spins = np.where(np.arange(n) < 3 * n // 4, 1, -1)  # M = 0.5
        assert model.susceptibility() == pytest.approx((1 - 0.5 ** 2) / 1.6)


def test_snapshot_matches_phase_diagram_for_independent_agents():
    """J = 0: agents are independent, so both estimators must agree (χ = 1/T at h = 0), whatever N is."""
    T = 2.0
    for n in (40, 200):
        np.random.seed(1)
        model = IsingMarketModel(n_agents=n, J=0.0, h=0.0, T=T)
        model.spins = np.where(np.arange(n) % 2 == 0, 1, -1)  # M = 0
        snapshot = model.susceptibility()
        # phase_diagram uses Var(|M|); for M ~ N(0, 1/N), Var(|M|) = (1 - 2/π)/N
        chi = model.phase_diagram([T], n_samples=400)["susceptibility"][0]
        assert snapshot == pytest.approx(1 / T)
        assert chi == pytest.approx((1 - 2 / np.pi) / T, rel=0.25)
