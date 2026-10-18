"""Regression tests for src/visualization/risk_surface_3d.py."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.models.risk_metrics import TailRiskMetrics
from src.visualization.risk_surface_3d import CriticalPointVisualizer, TailRisk3DSurface, TailRiskDensityPlot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _returns(n=1500, seed=0):
    return 0.0003 + 0.01 * np.random.default_rng(seed).standard_t(4, n)


def _z_range(ax):
    return tuple(ax.zz_dataLim.intervalx)


# --------------------------------------------------------------------------- surfaces


def test_var_surface_with_integer_horizons_is_not_truncated():
    r = _returns()
    fig = TailRiskDensityPlot().create_tail_probability_surface(r, np.arange(1, 21), np.linspace(0.9, 0.99, 5))
    m = TailRiskMetrics(r)
    zmin, zmax = _z_range(fig.axes[0])  # was (0, 0): Z inherited the int64 dtype of the horizon grid
    assert zmin == pytest.approx(m.var_historical(0.9), rel=1e-12)
    assert zmax == pytest.approx(m.var_historical(0.99) * np.sqrt(20), rel=1e-12)


def test_risk_surface_with_integer_grids_keeps_fractional_values():
    fig = TailRisk3DSurface().create_risk_surface(np.arange(1, 5), np.arange(2, 6), lambda v, a: v / a)
    assert _z_range(fig.axes[0]) == pytest.approx((1 / 5, 4 / 2))  # was (0, 2): v/a truncated to int


def test_joint_density_accepts_lists_and_is_drawn_at_bin_centres():
    r1, r2 = _returns(800, 1), _returns(800, 2)
    fig = TailRiskDensityPlot().create_joint_tail_density(list(r1), list(r2))  # lists raised AttributeError
    ax = fig.axes[0]
    x_edges = np.linspace(r1.min(), r1.max(), 50)
    y_edges = np.linspace(r2.min(), r2.max(), 50)
    # the surface ends at the centre of the last bin, not at its left edge
    assert ax.xy_dataLim.x1 == pytest.approx(0.5 * (x_edges[-2] + x_edges[-1]), rel=1e-12)
    assert ax.xy_dataLim.y1 == pytest.approx(0.5 * (y_edges[-2] + y_edges[-1]), rel=1e-12)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Left tail boundary"]


# --------------------------------------------------------------------------- trajectory


def test_trajectory_is_one_collection_with_custom_labels_and_nan_safe_peak():
    rng = np.random.default_rng(3)
    X, Y, Z = rng.standard_normal((3, 2000)).cumsum(axis=1)
    Z[0] = np.nan
    fig = TailRisk3DSurface().create_phase_space_trajectory(X, Y, list(Z), axis_labels=("σ", "α", "Risiko"))
    ax = fig.axes[0]
    assert len(ax.lines) == 0  # was 1999 Line3D artists (seconds in Pyodide)
    segments = ax.collections[0]
    assert len(segments.get_edgecolor()) == len(X) - 1
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("σ", "α", "Risiko")
    np.testing.assert_allclose(ax.xy_dataLim.intervalx, (X.min(), X.max()))
    np.testing.assert_allclose(ax.xy_dataLim.intervaly, (Y.min(), Y.max()))
    peak = next(c for c in ax.collections if c.get_label() == "Peak Risk")
    assert peak.get_offsets()[0][0] == pytest.approx(X[np.nanargmax(Z)])  # argmax picked the NaN


def test_trajectory_default_labels_unchanged():
    fig = TailRisk3DSurface().create_phase_space_trajectory(np.arange(5.0), np.arange(5.0), np.arange(5.0))
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == (
        "X: Volatility", "Y: Tail Heaviness", "Z: Risk Level")


# --------------------------------------------------------------------------- critical point


def _ews(n=400, warmup=50, ac_end=0.9):
    """Rolling early-warning indicators with the usual NaN warm-up."""
    nan = np.full(warmup, np.nan)
    ac = np.r_[nan, np.linspace(0.0, ac_end, n - warmup)]
    var = np.r_[nan, np.linspace(1e-4, 5e-4, n - warmup)]
    skew = np.r_[nan, np.linspace(0.0, -1.0, n - warmup)]
    return ac, var, skew, np.arange(n, dtype=float)


def _near_critical(ax):
    return [c for c in ax.collections if c.get_label() == "Near Critical"]


def test_near_critical_points_are_marked_despite_nan_warmup():
    ac, var, skew, t = _ews()
    fig = CriticalPointVisualizer().create_critical_surface(ac, var, skew, t)
    ax = fig.axes[0]
    marked = _near_critical(ax)  # np.percentile(var with NaN) = NaN disabled the marker
    expected = int(np.sum((ac > 0.7) & (var > np.nanpercentile(var, 80))))
    assert expected > 0 and len(marked) == 1
    assert len(marked[0].get_offsets()) == expected
    assert [x.get_text() for x in ax.get_legend().get_texts()] == ["Near Critical"]


def test_no_empty_legend_when_nothing_is_critical():
    ac, var, skew, t = _ews(ac_end=0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "No artists with labels found to put in legend"
        fig = CriticalPointVisualizer().create_critical_surface(ac, var, skew, t)
    ax = fig.axes[0]
    assert ax.get_legend() is None and not _near_critical(ax)


def test_critical_thresholds_are_configurable():
    ac, var, skew, t = _ews(ac_end=0.4)
    fig = CriticalPointVisualizer().create_critical_surface(ac, var, skew, t, ac_threshold=0.3,
                                                            variance_percentile=50)
    expected = int(np.sum((ac > 0.3) & (var > np.nanpercentile(var, 50))))
    assert len(_near_critical(fig.axes[0])[0].get_offsets()) == expected > 0
