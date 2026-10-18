"""Regression tests for src/visualization/early_warning_dashboard.py."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.visualization.early_warning_dashboard import (  # noqa: E402
    CrashComparisonChart,
    EarlyWarningDashboard,
    create_indicator_summary_table,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _indicators(n=40, tdex_last=12.0, dix_last=45.0, seed=0):
    rng = np.random.default_rng(seed)
    scores = {k: rng.uniform(0, 100, n) for k in
              ("gex_score", "tdex_score", "vix_term_score", "dix_score", "smfi_score")}
    comp = (0.35 * scores["gex_score"] + 0.25 * scores["tdex_score"] + 0.20 * scores["vix_term_score"]
            + 0.10 * scores["dix_score"] + 0.10 * scores["smfi_score"])
    ind = {
        "gex": rng.normal(0, 5, n),
        "tdex": np.r_[rng.uniform(5, 15, n - 1), tdex_last],
        "vix_m1": rng.uniform(14, 20, n),
        "vix_m2": rng.uniform(15, 21, n),
        "dix": np.r_[rng.uniform(40, 50, n - 1), dix_last],
        "smfi": rng.normal(0, 5, n),
        "composite_score": comp,
        **scores,
    }
    prices = 4500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return ind, prices


def _panel_label(plot, *args):
    fig, ax = plt.subplots()
    plot(ax, *args)
    return ax.texts[0].get_text().split(" - ")[-1]


# ------------------------------------------------------------- level labels
@pytest.mark.parametrize("tdex_last, level", [(25.0, "EXTREME"), (20.5, "EXTREME"),
                                              (17.0, "ELEVATED"), (12.0, "NORMAL")])
def test_tdex_panel_reaches_extreme(tdex_last, level):
    # the old `'ELEVATED' if > 15 else ('EXTREME' if > 20 ...)` could never say EXTREME
    tdex = np.array([5, 10, 15, 18, tdex_last])
    label = _panel_label(EarlyWarningDashboard()._plot_tdex, np.arange(5), tdex, None)
    assert label == level


@pytest.mark.parametrize("tdex_last, dix_last", [(25.0, 39.0), (17.0, 41.5), (12.0, 45.0)])
def test_summary_table_uses_the_panel_thresholds(tdex_last, dix_last):
    ind, prices = _indicators(tdex_last=tdex_last, dix_last=dix_last)
    d = EarlyWarningDashboard()
    x = np.arange(len(prices))
    tdex_level = _panel_label(d._plot_tdex, x, ind["tdex"], ind["tdex_score"])
    dix_level = _panel_label(d._plot_dix, x, ind["dix"], prices, ind["dix_score"])
    rows = {line.split("│")[0].strip("║ ").strip(): line.split("│")[3].strip("║ ").strip()
            for line in create_indicator_summary_table(ind).splitlines() if line.count("│") == 3}
    assert rows["2. TDEX"] == tdex_level
    assert rows["4. DIX (%)"] == dix_level


# ------------------------------------------------------------- price panel
def test_price_panel_is_not_squashed_to_zero():
    ind, prices = _indicators(n=30)
    fig, ax = plt.subplots()
    EarlyWarningDashboard()._plot_price_with_composite(ax, np.arange(30), prices, ind["composite_score"])
    lo, hi = ax.get_ylim()
    # used to fill down to 0, forcing the y-axis to start at (about) 0
    assert lo > 0.9 * prices.min()
    assert hi >= prices.max()


# ------------------------------------------------------------- layout
def _overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def test_dashboard_layout_twin_axes_do_not_overlap_neighbours():
    ind, prices = _indicators(n=30)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = EarlyWarningDashboard(figsize=(16, 13)).create_dashboard(ind, prices)
    assert not [w for w in caught if "tight_layout" in str(w.message)]
    r = fig.canvas.get_renderer()
    by_spec = {}
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        by_spec.setdefault((ss.rowspan.start, ss.colspan.start), []).append(ax)
    smfi = by_spec[(3, 0)]          # SMFI + its twin price axis
    composite = by_spec[(3, 2)]
    for a in smfi:
        for b in composite:
            assert not _overlap(a.get_tightbbox(r), b.get_tightbbox(r))


def test_crash_chart_third_axis_does_not_overlap_right_column():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = CrashComparisonChart().create_comparison_chart(figsize=(12, 9))
    assert not [w for w in caught if "tight_layout" in str(w.message)]
    r = fig.canvas.get_renderer()
    left = [a for a in fig.axes if a.get_subplotspec().colspan.start == 0]
    right = [a for a in fig.axes if a.get_subplotspec().colspan.start == 1]
    for row in (0, 1):
        lb = [a.get_tightbbox(r) for a in left if a.get_subplotspec().rowspan.start == row]
        rb = [a.get_tightbbox(r) for a in right if a.get_subplotspec().rowspan.start == row]
        assert max(b.x1 for b in lb) < min(b.x0 for b in rb)


# ------------------------------------------------------------- crash chart RNG
def test_crash_chart_does_not_touch_global_rng():
    np.random.seed(123)
    before = np.random.get_state()
    fig, ax = plt.subplots()
    chart = CrashComparisonChart()
    chart._plot_crash_indicators(ax, "2020 COVID Crash", chart.crash_events["2020 COVID Crash"])
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2] == after[2]


_SNIPPET = """
import importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
# load the module file directly (skips the heavy src/__init__ imports)
spec = importlib.util.spec_from_file_location("ewd", "src/visualization/early_warning_dashboard.py")
ewd = importlib.util.module_from_spec(spec); spec.loader.exec_module(ewd)
c = ewd.CrashComparisonChart()
fig, ax = plt.subplots()
c._plot_crash_indicators(ax, "2018 Volmageddon", c.crash_events["2018 Volmageddon"])
print(np.round(np.concatenate([l.get_ydata() for a in fig.axes for l in a.lines[:1]]), 10).tobytes().hex())
"""


def test_crash_chart_is_reproducible_across_processes():
    # str hash() is salted per process (and per Pyodide boot); the noise must not depend on it
    procs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed, MPLBACKEND="Agg")
        procs.append(subprocess.Popen([sys.executable, "-c", _SNIPPET], cwd=ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        outs.append(out.strip())
    assert outs[0] and outs[0] == outs[1]
