"""The early-warning indicators can be computed without look-ahead (needed for honest backtests)."""

import numpy as np

from src.analysis.early_warning import CompositeEarlyWarningSystem, DarkIndex, analyze_crash_risk
from src.utils.data_loader import generate_crisis_scenario, generate_vix_from_returns


def _data():
    r, _ = generate_crisis_scenario(seed=7)
    r = np.asarray(r, dtype=float)
    return r, np.asarray(generate_vix_from_returns(r), dtype=float)


def test_causal_dix_ignores_the_future():
    r, _ = _data()
    t = 600
    future_changed = r.copy()
    future_changed[t:] = -0.05  # a crash that has not happened yet at day t-1
    np.random.seed(1)
    a = DarkIndex().simulate_dix(r, lookahead=0)
    np.random.seed(1)
    b = DarkIndex().simulate_dix(future_changed, lookahead=0)
    assert np.allclose(a[:t], b[:t])


def test_original_dix_model_peeks_ahead_and_stays_the_default():
    r, _ = _data()
    t = 600
    future_changed = r.copy()
    future_changed[t:] = -0.05
    np.random.seed(1)
    a = DarkIndex().simulate_dix(r)
    np.random.seed(1)
    b = DarkIndex().simulate_dix(future_changed)
    assert not np.allclose(a[:t], b[:t])  # documented illustrative model (README figures)


def _suite(r, vix, causal):
    np.random.seed(3)  # same length -> identical random stream; only the data after t differs
    return CompositeEarlyWarningSystem().simulate_all_indicators(r, 100 * np.exp(np.cumsum(r)), vix, causal=causal)


def test_causal_indicators_up_to_t_depend_only_on_data_up_to_t():
    r, vix = _data()
    t = 650
    r2, v2 = r.copy(), vix.copy()
    r2[t:], v2[t:] = -0.05, 60.0  # a crash that has not happened yet at day t-1
    a, b = _suite(r, vix, True), _suite(r2, v2, True)
    for key, series in a.items():
        series = np.asarray(series)
        if series.ndim == 1 and len(series) >= t:
            assert np.allclose(series[:t], np.asarray(b[key])[:t], equal_nan=True), key


def test_default_model_leaks_only_through_the_documented_dix_lookahead():
    r, vix = _data()
    t = 650
    r2, v2 = r.copy(), vix.copy()
    r2[t:], v2[t:] = -0.05, 60.0
    a, b = _suite(r, vix, False), _suite(r2, v2, False)
    leaky = {k for k, s in a.items() if np.asarray(s).ndim == 1 and len(s) >= t
             and not np.allclose(np.asarray(s)[:t], np.asarray(b[k])[:t], equal_nan=True)}
    assert leaky <= {"dix", "dix_score", "composite_score"}


def test_analyze_crash_risk_accepts_causal():
    r, vix = _data()
    np.random.seed(0)
    out = analyze_crash_risk(r, vix, verbose=False, causal=True)
    assert 0 <= out["composite_score"] <= 100 if "composite_score" in out else True
