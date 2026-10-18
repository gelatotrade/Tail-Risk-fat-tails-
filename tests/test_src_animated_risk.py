"""Regression tests for src/visualization/animated_risk.py."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import src.visualization as viz  # noqa: E402
from src.visualization.animated_risk import (  # noqa: E402
    DistributionMorphAnimator,
    PhaseSpaceAnimator,
    RiskGaugeAnimator,
    create_comprehensive_animation,
)

pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _returns(n=200, seed=7):
    return stats.t.rvs(3, size=n, random_state=np.random.default_rng(seed)) * 0.01


def _draw(anim, frame):
    anim._func(frame)  # the frame function FuncAnimation would call
    anim._draw_was_started = True  # rendered by hand: no "deleted without rendering" warning


# ------------------------------------------------------------- distribution morph
def test_gaussian_overlay_is_the_fitted_normal_density():
    r = _returns()
    anim = DistributionMorphAnimator(r, window=60).create_animation()
    fig = plt.gcf()
    frame = list(anim.new_frame_seq())[-1]
    _draw(anim, frame)
    w = r[frame - 60:frame + 1]
    mean, std = np.mean(w), np.std(w)
    for ax in fig.axes:
        line = ax.lines[0]
        xd, yd = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        np.testing.assert_allclose(yd, stats.norm.pdf(xd, mean, std), rtol=1e-10, atol=0)
        # covers mean +- 5 sigma, so it integrates to ~1 and peaks at 1/(sigma*sqrt(2 pi))
        assert _trapezoid(yd, xd) == pytest.approx(1.0, abs=1e-3)
        assert yd.max() == pytest.approx(1 / (std * np.sqrt(2 * np.pi)), rel=5e-3)
        assert (yd > 1e-3 * yd.max()).sum() > 100


def test_distribution_morph_rejects_too_short_series():
    with pytest.raises(ValueError, match="window"):
        DistributionMorphAnimator(_returns(60), window=60).create_animation()
    anim = DistributionMorphAnimator(_returns(61), window=60).create_animation()
    assert list(anim.new_frame_seq()) == [60]
    _draw(anim, 60)


# ------------------------------------------------------------- risk gauge
def _needle(level):
    anim = RiskGaugeAnimator(np.array([level])).create_animation()
    fig = plt.gcf()
    _draw(anim, 0)
    fig.canvas.draw()
    ax = fig.axes[0]
    theta = ax.lines[0].get_xdata()[0]
    centre, tip = ax.transData.transform([[theta, 0.0], [theta, 0.9]])
    return tip - centre


def test_gauge_is_an_upper_half_dial_from_left_to_right():
    left, up, right = _needle(0.0), _needle(50.0), _needle(100.0)
    assert left[0] < 0 and abs(left[1]) < 1e-6 * abs(left[0]) + 1e-6   # 0 % points left
    assert up[1] > 0 and abs(up[0]) < 1e-6 * up[1] + 1e-6              # 50 % points up
    assert right[0] > 0 and abs(right[1]) < 1e-6 * right[0] + 1e-6     # 100 % points right
    assert _needle(25.0)[1] > 0 and _needle(25.0)[0] < 0               # low risk: upper left


# ------------------------------------------------------------- phase space
def test_phase_space_history_cloud_is_drawn():
    rng = np.random.default_rng(0)
    X, Y, Z = rng.random((3, 120))
    anim = PhaseSpaceAnimator(X, Y, Z).create_animation(trail_length=10)
    fig = plt.gcf()
    _draw(anim, 80)
    fig.canvas.draw()
    sizes = sorted(len(c.get_offsets()) for c in fig.axes[0].collections)
    assert sizes == [1, 81]  # current point + every state visited so far


# ------------------------------------------------------------- comprehensive
def _coords(n, seed=1):
    return tuple(np.random.default_rng(seed).random((3, n)))


def test_comprehensive_title_goes_to_its_own_figure():
    r = _returns(100)
    anim = create_comprehensive_animation(r, _coords(100))
    fig = plt.gcf()
    other = plt.figure()  # becomes pyplot's current figure
    _draw(anim, 80)
    assert fig.get_suptitle() == "Tail Risk Dashboard - Time: 80"
    assert other.get_suptitle() == ""


def test_comprehensive_rolling_var_is_the_trailing_99pct_quantile():
    r = _returns(150)
    anim = create_comprehensive_animation(r, _coords(150))
    fig = plt.gcf()
    _draw(anim, 140)
    ax4 = fig.axes[3]
    xd, yd = ax4.lines[0].get_data()
    expected = [-np.percentile(r[i - 60:i], 1) for i in range(60, 140)]
    np.testing.assert_array_equal(np.asarray(xd), np.arange(60, 140))
    np.testing.assert_allclose(yd, expected, rtol=1e-12, atol=0)


def test_comprehensive_rejects_too_short_series():
    r = _returns(70)
    with pytest.raises(ValueError, match="70"):
        create_comprehensive_animation(r, _coords(70))
    with pytest.raises(ValueError, match="levy_coords"):
        create_comprehensive_animation(_returns(100), _coords(50))


# ------------------------------------------------------------- namespace
def test_star_import_exports_only_the_animation_api():
    from src.visualization import animated_risk

    assert set(animated_risk.__all__) == {"PhaseSpaceAnimator", "DistributionMorphAnimator",
                                          "RiskGaugeAnimator", "create_comprehensive_animation"}
    assert not hasattr(viz, "FuncAnimation") and not hasattr(viz, "PillowWriter")
