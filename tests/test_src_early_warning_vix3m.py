"""Early-warning time series on the OBSERVED VIX/VIX3M curve, and a monotone term-structure score."""

import numpy as np
import pytest

from src.analysis.early_warning import (
    CompositeEarlyWarningSystem,
    VIXTermStructure,
    analyze_crash_risk,
    compute_early_warning_coordinates,
)
from src.utils.data_loader import generate_crisis_scenario, generate_vix_from_returns


def _data(seed=7):
    r, _ = generate_crisis_scenario(seed=seed)
    r = np.asarray(r, dtype=float)
    vix = np.asarray(generate_vix_from_returns(r), dtype=float)
    # a plausible VIX3M: smoother, above spot in calm markets, below it in panics
    vix3m = 0.6 * vix + 0.4 * 22.0
    return r, vix, vix3m


def _run(r, vix, vix3m=None, causal=True, seed=3):
    np.random.seed(seed)
    return CompositeEarlyWarningSystem().simulate_all_indicators(
        r, 100 * np.exp(np.cumsum(r)), vix, causal=causal, vix3m=vix3m)


def test_vix3m_is_used_as_the_observed_curve():
    r, vix, vix3m = _data()
    ind = _run(r, vix, vix3m)
    np.testing.assert_array_equal(ind["vix_m1"], vix)
    np.testing.assert_array_equal(ind["vix_m2"], vix3m)
    expected = [VIXTermStructure()._compute_warning_score(a / b, a) for a, b in zip(vix, vix3m)]
    np.testing.assert_allclose(ind["vix_term_score"], expected)


def test_vix3m_changes_only_the_term_structure_component():
    r, vix, vix3m = _data()
    real, proxy = _run(r, vix, vix3m), _run(r, vix, None)
    for k in ("gex", "tdex", "dix", "smfi", "gex_score", "tdex_score", "dix_score", "smfi_score"):
        np.testing.assert_array_equal(real[k], proxy[k], err_msg=k)
    assert not np.allclose(real["vix_term_score"], proxy["vix_term_score"])
    w = {"gex": 0.35, "tdex": 0.25, "vix_term": 0.20, "dix": 0.10, "smfi": 0.10}
    np.testing.assert_allclose(real["composite_score"], sum(w[k] * real[k + "_score"] for k in w))


def test_vix3m_length_mismatch_is_rejected():
    r, vix, vix3m = _data()
    with pytest.raises(ValueError):
        _run(r, vix, vix3m[:-1])


def test_term_structure_score_is_non_decreasing_in_m1_m2():
    # used to be V-shaped: a deep, calm contango (M1/M2 = 0.8) scored 20, a flat curve 0
    n = 400
    r = np.zeros(n)
    vix = np.full(n, 18.0)
    ratio = np.linspace(0.75, 1.25, n)
    ind = _run(r, vix, vix / ratio)
    s = ind["vix_term_score"]
    assert np.all(np.diff(s) >= -1e-12)
    assert s[0] == 0.0 and s[-1] == pytest.approx(60.0)  # ratio part only (VIX 18 adds nothing)
    assert s.min() >= 0 and s.max() <= 100


def test_simulated_curve_score_matches_snapshot_rule():
    r, vix, _ = _data(seed=2)
    ind = _run(r, vix, None, causal=False)
    ts = VIXTermStructure()
    for i in range(0, len(r), 37):
        snap = ts.calculate_term_structure(vix[i], ind["vix_m1"][i], ind["vix_m2"][i])
        assert ind["vix_term_score"][i] == pytest.approx(snap["warning_score"])


def test_causal_real_curve_path_uses_no_future_data():
    r, vix, vix3m = _data()
    t = 650
    r2, v2, v32 = r.copy(), vix.copy(), vix3m.copy()
    r2[t:], v2[t:], v32[t:] = -0.05, 70.0, 45.0  # a crash that has not happened yet at day t-1
    a, b = _run(r, vix, vix3m), _run(r2, v2, v32)
    for key in a:
        np.testing.assert_allclose(a[key][:t], b[key][:t], err_msg=key)


def test_wrappers_pass_vix3m_through():
    r, vix, vix3m = _data()
    np.random.seed(5)
    res = analyze_crash_risk(r, vix, verbose=False, causal=True, vix3m=vix3m)
    assert res["latest"]["vix_term_ratio"] == pytest.approx(vix[-1] / vix3m[-1])
    np.random.seed(5)
    X, Y, Z = compute_early_warning_coordinates(r, vix, causal=True, vix3m=vix3m)
    np.testing.assert_allclose(Z, res["time_series"]["composite_score"] / 100)
    np.testing.assert_allclose(X, res["time_series"]["gex_score"] / 100)


def test_snapshot_term_score_is_non_decreasing_with_a_step_only_at_inversion():
    # the 0.95-1.0 ramp used to fall from 30 to 0 as the curve flattened toward inversion
    ts = VIXTermStructure()
    ratios = np.linspace(0.85, 1.2, 701)
    s = np.array([ts._compute_warning_score(x, 18.0) for x in ratios])
    d = np.diff(s)
    assert np.all(d >= -1e-9)
    jumps = ratios[1:][d > 1.0]  # steps on a 0.0005 grid
    assert len(jumps) == 1 and 1.0 < jumps[0] <= 1.0006  # only where the curve inverts
    assert ts._compute_warning_score(0.95, 18.0) == 0
    assert ts._compute_warning_score(0.975, 18.0) == pytest.approx(12.5)
    assert ts._compute_warning_score(1.0, 18.0) == pytest.approx(25.0)
    assert ts._compute_warning_score(1.0 + 1e-9, 18.0) == pytest.approx(30.0)
