"""Dashboard levels follow the scoring system; daily series are drawn as collections (fast in Pyodide)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.analysis.early_warning import CompositeEarlyWarningSystem  # noqa: E402
from src.visualization.early_warning_dashboard import (  # noqa: E402
    EarlyWarningDashboard,
    create_indicator_summary_table,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _indicators(n=300, composite_last=10.0, seed=3):
    rng = np.random.default_rng(seed)
    keys = ("gex_score", "tdex_score", "vix_term_score", "dix_score", "smfi_score")
    ind = {k: rng.uniform(0, 100, n) for k in keys}
    ind.update(gex=rng.normal(0, 5, n), tdex=rng.uniform(5, 25, n), vix_m1=rng.uniform(14, 20, n),
               vix_m2=rng.uniform(15, 21, n), dix=rng.uniform(38, 50, n), smfi=rng.normal(0, 5, n),
               composite_score=np.r_[rng.uniform(0, 100, n - 1), composite_last])
    prices = 4500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return ind, prices


@pytest.mark.parametrize("score", [10.0, 49.9, 50.0, 60.0, 74.9, 75.0, 89.9, 90.0, 99.0])
def test_gauge_and_summary_name_the_level_of_compute_composite(score):
    ews = CompositeEarlyWarningSystem()
    expected = ews.compute_composite(score, score, score, score, score)["level"]
    fig, ax = plt.subplots()
    EarlyWarningDashboard()._plot_risk_gauge(ax, score)
    assert ax.texts[-1].get_text() == expected
    ind, _ = _indicators(composite_last=score)
    row = [ln for ln in create_indicator_summary_table(ind).splitlines() if "COMPOSITE" in ln][0]
    signal = row.split("│")[3].strip("║ ").strip()
    assert signal == {"EXTREME": "EXTREME RISK", "HIGH": "HIGH RISK"}.get(expected, expected)


def test_weights_come_from_the_scoring_system():
    w = CompositeEarlyWarningSystem().weights
    d = EarlyWarningDashboard()
    assert d.weights == w
    ind, _ = _indicators()
    fig, ax = plt.subplots()
    d._plot_indicator_weights(ax, ind)
    got = [p.get_width() for p in ax.patches]
    want = [ind[f"{k}_score"][-1] * w[k] for k in ("gex", "tdex", "vix_term", "dix", "smfi")]
    np.testing.assert_allclose(got, want)


def test_daily_series_are_single_collections():
    n = 300
    ind, prices = _indicators(n)
    dates = np.arange("2024-01-01", n, dtype="datetime64[D]")
    fig = EarlyWarningDashboard(figsize=(16, 13)).create_dashboard(ind, prices, dates=dates)
    # one artist per series instead of one Rectangle / Line2D per day
    assert max(len(ax.patches) for ax in fig.axes) < 20
    assert max(len(ax.lines) for ax in fig.axes) < 10
    fig.canvas.draw()  # renders with dates


def test_bars_match_the_data_and_keep_zero_baseline():
    n = 50
    ind, prices = _indicators(n)
    x = np.arange(n)
    fig, ax = plt.subplots()
    EarlyWarningDashboard()._plot_gex(ax, x, ind["gex"], ind["gex_score"])
    (bars,) = [c for c in ax.collections if len(c.get_paths()) == n]
    tops = [p.vertices[1:3, 1] for p in bars.get_paths()]
    np.testing.assert_allclose([t[0] for t in tops], ind["gex"])
    lefts = [p.vertices[0, 0] for p in bars.get_paths()]
    np.testing.assert_allclose(lefts, x - 0.5)
    lo, hi = ax.get_ylim()
    assert lo <= ind["gex"].min() and hi >= ind["gex"].max()


def test_price_background_spans_the_whole_period_in_axes_height():
    n = 40
    ind, prices = _indicators(n)
    fig, ax = plt.subplots()
    EarlyWarningDashboard()._plot_price_with_composite(ax, np.arange(n), prices, ind["composite_score"])
    (bg,) = [c for c in ax.collections if len(c.get_paths()) == n - 1]
    v = np.array([p.vertices[:4] for p in bg.get_paths()])
    np.testing.assert_allclose(v[:, 0, 0], np.arange(n - 1))
    np.testing.assert_allclose(v[:, 2, 0], np.arange(1, n))
    assert v[:, :, 1].min() == 0 and v[:, :, 1].max() == 1  # full height (axes coordinates)
    assert ax.get_ylim()[0] > 0.9 * prices.min()


def test_composite_history_has_few_date_ticks():
    n = 400
    ind, _ = _indicators(n)
    fig, ax = plt.subplots(figsize=(4, 3))
    EarlyWarningDashboard()._plot_composite_history(ax, np.arange("2023-01-01", n, dtype="datetime64[D]"),
                                                   ind["composite_score"])
    fig.canvas.draw()
    lo, hi = ax.get_xlim()
    assert len([t for t in ax.get_xticks() if lo <= t <= hi]) <= 5
