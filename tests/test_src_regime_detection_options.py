"""Options of RegimeAwareTailRisk used by the 'regimes' web tool (vol_percentile, markov_states)."""

import numpy as np
import pytest

from src.models.regime_detection import RegimeAwareTailRisk, VolatilityClusteringDetector


def _two_regime_returns(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.98 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    sd = np.where(state == 0, 0.006, 0.025)
    return rng.normal(0.0, sd), state


def test_vol_percentile_is_passed_to_the_detector():
    y, _ = _two_regime_returns()
    for pct in (60, 75, 90):
        ra = RegimeAwareTailRisk(y, vol_percentile=pct, markov_states=2)
        expected = VolatilityClusteringDetector(y).detect_vol_regime(pct)
        np.testing.assert_array_equal(ra.regimes, expected)
    # a higher percentile flags fewer days as HIGH_VOL
    hi90 = np.mean(RegimeAwareTailRisk(y, vol_percentile=90, markov_states=2).regimes == 2)
    hi60 = np.mean(RegimeAwareTailRisk(y, vol_percentile=60, markov_states=2).regimes == 2)
    assert hi90 < hi60


def test_two_state_markov_labels_and_probabilities():
    y, state = _two_regime_returns()
    ra = RegimeAwareTailRisk(y, markov_states=2)
    assert ra.markov.n_states == 2
    cur = ra.current_regime()
    assert cur["markov_regime"] in ("LOW_VOL", "HIGH_VOL")
    probs = cur["regime_probabilities"]
    assert probs["medium_vol"] == 0.0
    assert probs["low_vol"] + probs["high_vol"] == pytest.approx(1.0, abs=1e-9)
    # the fitted states recover the simulated volatilities (sorted: state 0 = low vol)
    np.testing.assert_allclose(ra.markov.state_stds, [0.006, 0.025], rtol=0.15)
    filtered_state = np.argmax(ra.markov.filtered_probs, axis=1)
    assert np.mean(filtered_state == state) > 0.9


def test_three_states_is_still_the_default():
    y, _ = _two_regime_returns(n=600)
    ra = RegimeAwareTailRisk(y)
    assert ra.markov.n_states == 3 and ra.markov_states == 3 and ra.vol_percentile == 75
    assert ra.current_regime()["markov_regime"] in ("LOW_VOL", "MEDIUM_VOL", "HIGH_VOL")


def test_invalid_markov_states():
    with pytest.raises(ValueError):
        RegimeAwareTailRisk(np.random.default_rng(0).normal(0, 0.01, 300), markov_states=4)
