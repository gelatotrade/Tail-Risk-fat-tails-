"""Regression tests for src/visualization/phase_space.py."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from src.visualization.phase_space import LyapunovExponentEstimator, PhaseSpaceAnalyzer, RecurrencePlotAnalyzer


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _two_regime_returns(seed=0, n_regime=400, low=0.004, high=0.03):
    """Calm Gaussian regime followed by a 7.5x more volatile one."""
    rng = np.random.default_rng(seed)
    return np.concatenate([low * rng.standard_normal(n_regime), high * rng.standard_normal(n_regime)])


# --------------------------------------------------------------------------- coordinates


def test_rolling_skewness_matches_per_window_scipy():
    r = 0.01 * np.random.default_rng(1).standard_t(4, 300)
    ps = PhaseSpaceAnalyzer(r, window=20)
    expected = np.zeros(300)
    for i in range(20, 300):
        expected[i] = stats.skew(r[i - 20:i])
    np.testing.assert_allclose(ps.Z, expected, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- attractors


def test_attractors_separate_the_volatility_regimes():
    r = _two_regime_returns()
    att = PhaseSpaceAnalyzer(r, window=20).detect_attractor_regions(n_clusters=2)
    high_cluster = att["regime_order"][-1] + 1
    t = np.arange(len(r))
    keep = (t >= 40) & ~((t >= 400) & (t <= 420))  # skip warm-up and the mixed windows
    agreement = np.mean((att["cluster_labels"] == high_cluster)[keep] == (t >= 400)[keep])
    # Unscaled Ward split on skewness (order 1) and ignored volatility (order 1e-2): ~50 %.
    assert agreement > 0.95
    assert att["cluster_regimes"][int(high_cluster)] == "HIGH_VOL"


def test_attractor_clusters_do_not_depend_on_return_units():
    r = _two_regime_returns(seed=3)
    decimal = PhaseSpaceAnalyzer(r).detect_attractor_regions(3)
    percent = PhaseSpaceAnalyzer(100 * r).detect_attractor_regions(3)
    np.testing.assert_array_equal(decimal["cluster_labels"], percent["cluster_labels"])


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_attractor_regime_labels_cover_every_cluster(k):
    att = PhaseSpaceAnalyzer(_two_regime_returns(seed=4, n_regime=150)).detect_attractor_regions(k)
    assert len(att["regime_labels"]) == k  # was always 3 labels for k != 2
    assert sorted(att["cluster_regimes"]) == list(range(1, k + 1))
    vol_sorted = att["centers"][att["regime_order"], 1]
    assert np.all(np.diff(vol_sorted) >= 0)
    lowest = att["cluster_regimes"][int(att["regime_order"][0]) + 1]
    assert lowest == ("NORMAL" if k == 1 else "LOW_VOL")


def test_unstandardized_clustering_is_still_available():
    r = _two_regime_returns(seed=5, n_regime=150)
    ps = PhaseSpaceAnalyzer(r)
    from scipy.cluster.hierarchy import fcluster, linkage

    coords = np.column_stack([ps.X, ps.Y, ps.Z])
    expected = fcluster(linkage(coords, method="ward"), 3, criterion="maxclust")
    np.testing.assert_array_equal(ps.detect_attractor_regions(3, standardize=False)["cluster_labels"], expected)


def test_standardized_phase_space_distance():
    ps = PhaseSpaceAnalyzer(_two_regime_returns(seed=6, n_regime=150))
    y_ref = float(np.mean(ps.Y))
    # default unchanged: raw Euclidean distance to (0, mean vol, 0)
    np.testing.assert_allclose(ps.phase_space_distance(), np.sqrt(ps.X**2 + (ps.Y - y_ref) ** 2 + ps.Z**2))
    s = np.std(np.column_stack([ps.X, ps.Y, ps.Z]), axis=0)
    expected = np.sqrt((ps.X / s[0]) ** 2 + ((ps.Y - y_ref) / s[1]) ** 2 + (ps.Z / s[2]) ** 2)
    np.testing.assert_allclose(ps.phase_space_distance(standardize=True), expected, rtol=1e-12)


# --------------------------------------------------------------------------- Lyapunov


def test_lyapunov_with_ten_embedding_vectors_returns_unknown_instead_of_crashing():
    est = LyapunovExponentEstimator(np.random.default_rng(0).standard_normal(12))  # 10 vectors (m=3)
    assert np.isnan(est.estimate_largest_lyapunov())  # was ValueError: high <= 0 (randint(0, 0))
    out = est.classify_dynamics()
    assert out["classification"] == "UNKNOWN" and "risk_implication" in out


def test_lyapunov_embedding_longer_than_series():
    est = LyapunovExponentEstimator(np.ones(3), embedding_dim=5)  # was ValueError: negative dimensions
    assert est.embedded.shape == (0, 5)
    assert est.classify_dynamics()["classification"] == "UNKNOWN"


def test_lyapunov_seed_is_reproducible_and_leaves_the_global_rng_alone():
    est = LyapunovExponentEstimator(np.random.default_rng(6).standard_normal(500))
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    a = est.estimate_largest_lyapunov(seed=7)
    assert np.isfinite(a)
    assert est.estimate_largest_lyapunov(seed=7) == a
    np.testing.assert_array_equal(np.random.get_state()[1], state)
    assert est.classify_dynamics(seed=7)["lyapunov"] == a


def test_lyapunov_without_seed_still_follows_the_global_rng():
    est = LyapunovExponentEstimator(np.random.default_rng(9).standard_normal(400))
    np.random.seed(5)
    a = est.estimate_largest_lyapunov()
    np.random.seed(5)
    assert est.estimate_largest_lyapunov() == a


# --------------------------------------------------------------------------- recurrence analysis


def _reference_det_lam(R, l_min=3):
    """Textbook DET / LAM (Marwan et al. 2007) without the line of identity, plain loops."""
    R = np.asarray(R, dtype=bool).copy()
    n = len(R)
    np.fill_diagonal(R, False)

    def points(lines):
        total = 0
        for line in lines:
            run = 0
            for v in list(line) + [False]:
                if v:
                    run += 1
                else:
                    total += run if run >= l_min else 0
                    run = 0
        return total

    diag = points(np.diagonal(R, k) for k in range(-(n - 1), n) if k != 0)
    vert = points(R[:, j] for j in range(n))
    return diag / R.sum(), vert / R.sum()


def test_determinism_and_laminarity_match_the_textbook_definition():
    ana = RecurrencePlotAnalyzer(np.cumsum(np.random.default_rng(8).standard_normal(200)))
    R = ana.compute_recurrence_matrix()
    det, lam = _reference_det_lam(R)
    out = ana.recurrence_quantification()
    assert out["recurrence_rate"] == pytest.approx(R.mean())
    assert out["determinism"] == pytest.approx(det, abs=1e-12)  # main diagonal was counted in DET's numerator
    assert out["laminarity"] == pytest.approx(lam, abs=1e-12)  # documented but missing
    assert 0 < lam < 1


def test_exactly_periodic_signal_has_determinism_one():
    x = np.sin(2 * np.pi * np.arange(300) / 20)
    out = RecurrencePlotAnalyzer(x).recurrence_quantification(epsilon=1e-9)
    assert out["recurrence_rate"] == pytest.approx(1 / 20, rel=0.01)
    assert out["determinism"] == pytest.approx(1.0)  # old formula: 1.07 > 1
    assert out["laminarity"] == 0.0


def test_epsilon_and_min_line_are_passed_through():
    ana = RecurrencePlotAnalyzer(np.cumsum(np.random.default_rng(10).standard_normal(150)))
    eps = 0.8
    R = ana.compute_recurrence_matrix(eps)
    out = ana.recurrence_quantification(epsilon=eps, min_line=2)
    assert out["recurrence_rate"] == pytest.approx(R.mean())
    assert (out["determinism"], out["laminarity"]) == pytest.approx(_reference_det_lam(R, l_min=2))

    fig = ana.plot_recurrence(epsilon=1e-12)  # only the line of identity recurs
    assert "DET: 0.000" in fig.axes[0].texts[0].get_text()
