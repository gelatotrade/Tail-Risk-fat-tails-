"""MarketPhaseClassifier was vectorised for the browser; it must equal the original per-window loop."""

import numpy as np
import pytest
from scipy import stats

from src.physics.phase_transitions import MarketPhaseClassifier
from src.utils.data_loader import generate_crisis_scenario, generate_synthetic_returns


@pytest.mark.parametrize("window", [20, 50])
def test_volatility_percentile_equals_reference_loop(window):
    r, _ = generate_crisis_scenario(seed=11)
    for end in (max(2 * window + 20, 60), 300, len(r)):
        x = r[:end]
        ref = stats.percentileofscore([np.std(x[i:i + window]) for i in range(len(x) - window)],
                                      np.std(x[-window:]))
        out = MarketPhaseClassifier().classify(x, window=window)
        assert out["volatility_percentile"] == pytest.approx(ref, abs=1e-9)


def test_classify_accepts_lists():
    r = generate_synthetic_returns(400, "garch", seed=2)
    a = MarketPhaseClassifier().classify(list(r))
    b = MarketPhaseClassifier().classify(np.asarray(r))
    assert a["phase"] == b["phase"] and a["volatility_percentile"] == b["volatility_percentile"]
