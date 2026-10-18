"""Regression tests for src/visualization/dashboard.py."""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from src.visualization.dashboard import TailRiskDashboard, create_summary_report


def _t_returns(n, seed, nu=3, scale=0.01, loc=0.0003):
    return loc + scale * np.random.default_rng(seed).standard_t(nu, n)


@pytest.fixture(scope="module")
def dashboard():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tight_layout with 3D axes
        dash = TailRiskDashboard(_t_returns(300, 42))
        fig = dash.create_full_dashboard()
    yield dash, fig
    plt.close(fig)


def _axis(fig, title_prefix):
    return next(ax for ax in fig.axes if ax.get_title().startswith(title_prefix))


# --------------------------------------------------------------------------- full dashboard


def test_qq_plot_uses_plotting_positions_over_the_whole_sample(dashboard):
    dash, fig = dashboard
    offsets = _axis(fig, "Q-Q").collections[0].get_offsets()
    n = dash.n
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    # linspace(0.01, 0.99, n) clipped the theoretical quantiles at +/-2.33
    np.testing.assert_allclose(offsets[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(offsets[:, 1], np.sort(dash.returns))
    assert offsets[:, 0].max() > 2.9


def test_log_histogram_floor_follows_the_data(dashboard):
    dash, fig = dashboard
    density, _ = np.histogram(dash.returns, bins=50, density=True)
    # without a floor the Gaussian far tail pushed the axis down to ~1e-15
    assert _axis(fig, "Fat Tails").get_ylim()[0] == pytest.approx(density[density > 0].min() / 10)


def test_failing_regime_analysis_falls_back_but_does_not_swallow_interrupts(monkeypatch):
    import src.models.regime_detection as rd

    class Broken:
        def __init__(self, *args, **kwargs):
            raise ValueError("boom")

    class Interrupted(BaseException):
        pass

    class Interrupting:
        def __init__(self, *args, **kwargs):
            raise Interrupted()

    r = _t_returns(250, 1)
    monkeypatch.setattr(rd, "RegimeAwareTailRisk", Broken)
    with pytest.warns(RuntimeWarning, match="Regime analysis failed"):
        dash = TailRiskDashboard(r)
    assert dash.current_regime == {"simple_regime": "UNKNOWN"} and dash.regime_var == {}

    monkeypatch.setattr(rd, "RegimeAwareTailRisk", Interrupting)
    with pytest.raises(Interrupted):  # the bare except swallowed KeyboardInterrupt / SystemExit too
        TailRiskDashboard(r)


# --------------------------------------------------------------------------- summary report


def _value(report, label):
    line = next(line for line in report.splitlines() if label in line)
    return float(re.search(r"-?\d+\.\d+", line.split(label, 1)[1]).group())


def test_summary_report_compares_two_sided_tail_probabilities():
    r = _t_returns(3000, 7)
    report = create_summary_report(r)
    sd = np.std(r)
    empirical = np.mean(r < -3 * sd) + np.mean(r > 3 * sd)
    gaussian = 2 * stats.norm.sf(3)  # P(|Z| > 3) = 0.2700 %; was the one-sided 0.1350 %
    assert _value(report, "Gaussian Probability:") == pytest.approx(100 * gaussian, abs=5e-5)
    assert _value(report, "Empirical Probability:") == pytest.approx(100 * empirical, abs=5e-5)
    # the ratio is the ratio of the two numbers shown (was the left-tail ratio only)
    assert _value(report, "Fat Tail Ratio:") == pytest.approx(empirical / gaussian, abs=5e-3)


def test_summary_report_box_is_aligned():
    for seed in (0, 11):
        lines = create_summary_report(_t_returns(1000, seed)).strip("\n").splitlines()
        assert lines[0].startswith("╔") and lines[-1].startswith("╚")
        assert {len(line) for line in lines} == {80}
        assert all(line.endswith(("║", "╗", "╣", "╝")) for line in lines)
