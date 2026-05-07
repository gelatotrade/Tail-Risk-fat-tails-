#!/usr/bin/env python3
"""
3D Tail Risk Surface for Early Crash Detection
===============================================

Generates a 3D surface visualization of crash probability as a function of
volatility and tail index. Historical crises and the current market position
are projected onto the surface so the user can visually detect when the
market is approaching a "cliff" of escalating tail risk.

Axes:
    X = Annualized volatility (sigma)
    Y = Tail index (alpha) - lower = fatter tails
    Z = Crash probability over a 30-day horizon
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ---------------------------------------------------------------------------
# Risk model: maps (volatility, tail index) -> crash probability
# ---------------------------------------------------------------------------

def crash_probability(vol: np.ndarray, alpha: np.ndarray,
                      threshold: float = 0.05,
                      horizon_days: int = 30) -> np.ndarray:
    """
    Probability of at least one return below -threshold in the next horizon
    under a Lévy-stable approximation.

    Tail probability for a stable law: P(|X| > x) ~ C * (x/scale)^(-alpha).
    The daily tail probability is exponentiated to a 30-day horizon as
    1 - (1 - p_day)^horizon.
    """
    daily_scale = vol / np.sqrt(252.0)
    standardised = threshold / np.maximum(daily_scale, 1e-6)

    tail_constant = np.sin(np.pi * alpha / 2.0) * \
        np.exp(np.where(alpha > 0, np.log(np.maximum(alpha, 1e-6)), 0))
    p_day = tail_constant * np.power(standardised, -alpha) / np.pi
    p_day = np.clip(p_day, 0.0, 0.45)

    p_horizon = 1.0 - np.power(1.0 - p_day, horizon_days)
    return np.clip(p_horizon, 0.0, 1.0)


def build_risk_colormap() -> LinearSegmentedColormap:
    """Green -> yellow -> orange -> red -> purple gradient."""
    colors = [
        (0.00, "#1a9641"),
        (0.20, "#a6d96a"),
        (0.40, "#ffffbf"),
        (0.60, "#fdae61"),
        (0.80, "#d7191c"),
        (1.00, "#7b3294"),
    ]
    return LinearSegmentedColormap.from_list("tail_risk", colors)


# ---------------------------------------------------------------------------
# Crisis trajectories (representative paths through the (vol, alpha) plane)
# ---------------------------------------------------------------------------

CRISES = {
    "COVID-19 (Feb-Mar 2020)": {
        "color": "#d62728",
        "vol": np.array([0.12, 0.15, 0.22, 0.40, 0.65, 0.80, 0.55, 0.35]),
        "alpha": np.array([1.95, 1.90, 1.80, 1.55, 1.30, 1.20, 1.40, 1.65]),
        "label_idx": 5,
    },
    "2022 Bear Market": {
        "color": "#1f77b4",
        "vol": np.array([0.13, 0.18, 0.22, 0.25, 0.28, 0.30, 0.27, 0.22]),
        "alpha": np.array([1.92, 1.88, 1.85, 1.82, 1.80, 1.78, 1.82, 1.88]),
        "label_idx": 5,
    },
    "2025 Tariff Crash": {
        "color": "#ff7f0e",
        "vol": np.array([0.10, 0.12, 0.16, 0.25, 0.40, 0.55, 0.45, 0.25]),
        "alpha": np.array([1.95, 1.92, 1.85, 1.70, 1.50, 1.35, 1.55, 1.80]),
        "label_idx": 5,
    },
}

CURRENT_MARKET = {"vol": 0.104, "alpha": 1.92, "label": "Jan 2026"}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _surface_grid(resolution: int = 80):
    vol_axis = np.linspace(0.05, 0.85, resolution)
    alpha_axis = np.linspace(1.05, 2.00, resolution)
    vol_grid, alpha_grid = np.meshgrid(vol_axis, alpha_axis)
    risk_grid = crash_probability(vol_grid, alpha_grid)
    return vol_axis, alpha_axis, vol_grid, alpha_grid, risk_grid


def _draw_main_surface(ax, vol_grid, alpha_grid, risk_grid, cmap):
    surf = ax.plot_surface(
        vol_grid, alpha_grid, risk_grid,
        cmap=cmap, linewidth=0, antialiased=True,
        alpha=0.92, rcount=80, ccount=80,
    )

    ax.contourf(
        vol_grid, alpha_grid, risk_grid,
        zdir="z", offset=-0.05, cmap=cmap,
        levels=np.linspace(0, 1, 11), alpha=0.55,
    )

    warning_levels = [0.05, 0.15, 0.30, 0.50]
    contour = ax.contour(
        vol_grid, alpha_grid, risk_grid,
        zdir="z", offset=-0.05,
        levels=warning_levels, colors="black",
        linewidths=0.9, linestyles="--",
    )
    ax.clabel(contour, fmt={lvl: f"{int(lvl*100)}%" for lvl in warning_levels},
              fontsize=8)
    return surf


def _draw_trajectories(ax):
    for name, data in CRISES.items():
        vol = data["vol"]
        alpha = data["alpha"]
        risk = crash_probability(vol, alpha)
        ax.plot(vol, alpha, risk + 0.015, color=data["color"], linewidth=3.0,
                marker="o", markersize=6, label=name, alpha=0.97,
                markeredgecolor="white", markeredgewidth=0.6)
        peak = int(np.argmax(risk))
        ax.scatter([vol[peak]], [alpha[peak]], [risk[peak] + 0.04],
                   color=data["color"], edgecolor="black",
                   s=180, marker="*", zorder=10, linewidth=1.0)


def _draw_current_market(ax):
    vol = CURRENT_MARKET["vol"]
    alpha = CURRENT_MARKET["alpha"]
    risk = float(crash_probability(np.array([vol]), np.array([alpha]))[0])
    ax.scatter([vol], [alpha], [risk + 0.02], color="#2ca02c",
               edgecolor="black", s=180, marker="D", zorder=11,
               label=f"Current market ({CURRENT_MARKET['label']})")
    ax.plot([vol, vol], [alpha, alpha], [-0.05, risk + 0.02],
            color="#2ca02c", linewidth=1.2, linestyle=":", alpha=0.8)


def _annotate_zones(ax):
    zone_text = {
        (0.10, 1.95, 0.02): "Calm regime",
        (0.30, 1.65, 0.20): "Stress build-up",
        (0.55, 1.35, 0.55): "Crash cliff",
        (0.75, 1.15, 0.85): "Black-Swan zone",
    }
    for (x, y, z), text in zone_text.items():
        ax.text(x, y, z, text, fontsize=8.5, color="black",
                ha="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="grey", alpha=0.85))


def _projection_panel(ax, vol_axis, alpha_axis, risk_grid, cmap):
    pcm = ax.pcolormesh(vol_axis, alpha_axis, risk_grid, cmap=cmap,
                        shading="auto", vmin=0, vmax=1)
    cs = ax.contour(vol_axis, alpha_axis, risk_grid,
                    levels=[0.05, 0.15, 0.30, 0.50], colors="black",
                    linewidths=1.0, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"{int(v*100)}%", fontsize=8)

    for name, data in CRISES.items():
        ax.plot(data["vol"], data["alpha"], color=data["color"], linewidth=2,
                marker="o", markersize=4, label=name)

    ax.scatter([CURRENT_MARKET["vol"]], [CURRENT_MARKET["alpha"]],
               color="#2ca02c", edgecolor="black", s=140, marker="D",
               zorder=10, label=f"Current ({CURRENT_MARKET['label']})")

    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Tail index $\\alpha$")
    ax.set_title("Top-down view: warning zones",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    plt.colorbar(pcm, ax=ax, fraction=0.04, pad=0.03,
                 label="30-day crash probability")


def _crisis_timeline_panel(ax):
    for name, data in CRISES.items():
        risk = crash_probability(data["vol"], data["alpha"])
        steps = np.arange(len(risk))
        ax.plot(steps, risk * 100, color=data["color"], linewidth=2.2,
                marker="o", markersize=4, label=name)

    ax.axhline(5, color="grey", linestyle=":", linewidth=1)
    ax.axhline(15, color="orange", linestyle=":", linewidth=1)
    ax.axhline(30, color="red", linestyle=":", linewidth=1)
    ax.text(0.02, 6, "Watch", fontsize=8, color="grey",
            transform=ax.get_yaxis_transform())
    ax.text(0.02, 17, "Elevated", fontsize=8, color="orange",
            transform=ax.get_yaxis_transform())
    ax.text(0.02, 32, "Critical", fontsize=8, color="red",
            transform=ax.get_yaxis_transform())

    ax.set_xlabel("Time step (relative)")
    ax.set_ylabel("Crash probability (%)")
    ax.set_title("Crisis trajectories along the surface",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _legend_box(ax):
    ax.axis("off")
    text = (
        "How to read the surface\n"
        "----------------------------------------\n"
        "X  Annualised volatility (5% to 85%)\n"
        "Y  Tail index alpha (1.0 fat - 2.0 Gauss)\n"
        "Z  Probability of a >5% one-day loss\n"
        "      within the next 30 trading days\n\n"
        "Reading guide\n"
        "----------------------------------------\n"
        "Green plateau   Calm regime\n"
        "Yellow / orange Stress build-up\n"
        "Red ridge       Crash cliff\n"
        "Purple peak     Black-Swan territory\n\n"
        "Early warning rule of thumb\n"
        "----------------------------------------\n"
        "Move from green to yellow within a\n"
        "few weeks --> raise hedges\n"
        "Cross the orange contour --> reduce\n"
        "exposure, monitor GEX and TDEX\n"
        "Cross the red contour --> defensive\n"
        "positioning, tail hedges\n"
    )
    ax.text(0.02, 0.98, text, fontsize=9, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7",
                      ec="#444", alpha=0.95))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def generate_surface(output_path: Path) -> Path:
    cmap = build_risk_colormap()
    vol_axis, alpha_axis, vol_grid, alpha_grid, risk_grid = _surface_grid()

    fig = plt.figure(figsize=(22, 13), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.6, 1.05, 1.05],
        height_ratios=[1.0, 0.85],
        left=0.03, right=0.97,
        top=0.92, bottom=0.06,
        wspace=0.30, hspace=0.32,
    )

    ax_main = fig.add_subplot(gs[:, 0], projection="3d")
    surf = _draw_main_surface(ax_main, vol_grid, alpha_grid, risk_grid, cmap)
    _draw_trajectories(ax_main)
    _draw_current_market(ax_main)
    _annotate_zones(ax_main)

    ax_main.set_xlabel("\nVolatility (annualised)", fontsize=12)
    ax_main.set_ylabel("\nTail index $\\alpha$", fontsize=12)
    ax_main.set_zlabel("\n30-day crash probability", fontsize=12)
    ax_main.set_title(
        "3D Tail-Risk Surface  -  early detection of crash regimes",
        fontsize=14, fontweight="bold", pad=18,
    )
    ax_main.view_init(elev=26, azim=-62)
    ax_main.set_zlim(-0.05, 1.0)
    ax_main.invert_yaxis()
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.92)

    cbar = fig.colorbar(surf, ax=ax_main, shrink=0.55, pad=0.10, aspect=14)
    cbar.set_label("Crash probability", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax_proj = fig.add_subplot(gs[0, 1:])
    _projection_panel(ax_proj, vol_axis, alpha_axis, risk_grid, cmap)

    ax_time = fig.add_subplot(gs[1, 1])
    _crisis_timeline_panel(ax_time)

    ax_legend = fig.add_subplot(gs[1, 2])
    _legend_box(ax_legend)

    fig.suptitle(
        "Tail-Risk Surface  -  visualising the cliff between calm markets "
        "and Black-Swan events",
        fontsize=16, fontweight="bold", y=0.985,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return output_path


def main():
    output = Path("outputs/tail_risk_surface_3d.png")
    saved = generate_surface(output)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
