"""compute_ou_coordinates: windows without a regular OU fit get the physically right reversion speed."""

import numpy as np

from src.physics.ornstein_uhlenbeck import JumpDiffusionRiskAnalyzer, OrnsteinUhlenbeckProcess, compute_ou_coordinates


def _raw_z(r, window):
    """Un-normalised θ per window, refitted independently."""
    a = np.abs(r)
    out = []
    for i in range(2 * window, len(r)):
        b = np.polyfit(a[i - window:i][:-1], a[i - window:i][1:], 1)[0]
        out.append(b)
    return np.array(out)


def test_negative_slope_windows_rank_as_fastest_reversion():
    """A window whose AR(1) slope is <= 0 reverts faster than any regular window: Z must be at the top."""
    r = np.random.default_rng(3).normal(0, 0.01, 1500)
    window = 30
    _, _, Z = compute_ou_coordinates(r, window)
    slopes = _raw_z(r, window)
    z = Z[2 * window:]
    neg, pos = z[slopes <= 0], z[(slopes > 0.05) & (slopes < 1)]
    assert len(neg) > 20 and len(pos) > 20
    assert neg.min() >= pos.max() - 1e-12  # capped θ = ln(1000) is the maximum
    np.testing.assert_allclose(neg, 1.0, atol=1e-9)


def test_cap_equals_jump_diffusion_cap():
    r = np.random.default_rng(5).normal(0, 0.01, 400)
    window = 30
    a = np.abs(r)
    # recover the raw (un-normalised) θ of one capped window
    for i in range(2 * window, len(r)):
        x = a[i - window:i]
        if np.polyfit(x[:-1], x[1:], 1)[0] <= 0:
            fit = OrnsteinUhlenbeckProcess.fit(x, dt=1, max_theta=JumpDiffusionRiskAnalyzer.MAX_THETA_DT)
            assert fit.theta == np.log(1000.0)
            break
    else:  # pragma: no cover
        raise AssertionError("no window with a negative slope")


def test_trending_window_gets_zero_reversion_speed():
    """|r| growing steadily (b >= 1) has no mean reversion: θ = 0, the bottom of the Z scale."""
    window = 20
    calm = np.random.default_rng(0).normal(0, 0.01, 3 * window)
    ramp = np.linspace(0.01, 0.2, 2 * window)
    r = np.r_[calm, ramp]
    _, _, Z = compute_ou_coordinates(r, window)
    assert Z[-1] == 0.0
