#!/usr/bin/env python3
"""
Animated 3D Tail Risk Surface
==============================

Renders an animated GIF showing how the tail-risk surface changes when
its driving variables change.

Three things vary across frames:
    1. The crash threshold sweeps from 3% to 10% one-day loss, so the
       "crash cliff" moves visibly across the (volatility, alpha) plane.
    2. A synthetic market trajectory walks across the surface, simulating
       a build-up from a calm regime into a crisis and a recovery.
    3. The camera rotates slowly so the 3D shape is unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from generate_tail_risk_surface import (
    crash_probability,
    build_risk_colormap,
)


# ---------------------------------------------------------------------------
# Synthetic market trajectory: calm -> stress build-up -> crash -> recovery
# ---------------------------------------------------------------------------

def market_trajectory(num_frames: int):
    t = np.linspace(0.0, 1.0, num_frames)
    vol = 0.10 + 0.55 * np.exp(-((t - 0.62) / 0.16) ** 2) + 0.05 * t
    alpha = 1.95 - 0.70 * np.exp(-((t - 0.62) / 0.18) ** 2) - 0.05 * t
    alpha = np.clip(alpha, 1.10, 1.98)
    return vol, alpha


def _setup_grid(resolution: int = 60):
    vol_axis = np.linspace(0.05, 0.85, resolution)
    alpha_axis = np.linspace(1.05, 2.00, resolution)
    vol_grid, alpha_grid = np.meshgrid(vol_axis, alpha_axis)
    return vol_axis, alpha_axis, vol_grid, alpha_grid


def _threshold_for_frame(t: float) -> float:
    """Sweep threshold between 3% and 10% in a smooth triangle wave."""
    phase = 2 * abs(t - 0.5)  # 1 -> 0 -> 1
    return 0.03 + 0.07 * (1.0 - phase)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def render_animation(output_path: Path,
                     num_frames: int = 72,
                     fps: int = 14) -> Path:
    cmap = build_risk_colormap()
    vol_axis, alpha_axis, vol_grid, alpha_grid = _setup_grid()
    traj_vol, traj_alpha = market_trajectory(num_frames)

    fig = plt.figure(figsize=(11.5, 7.0), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.6, 1.0],
        height_ratios=[1.0, 0.55],
        left=0.02, right=0.97, top=0.92, bottom=0.08,
        wspace=0.22, hspace=0.40,
    )
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, 1])

    trajectory_history_vol = []
    trajectory_history_alpha = []
    trajectory_history_risk = []
    trajectory_history_time = []

    def init():
        return []

    def update(frame: int):
        t = frame / max(num_frames - 1, 1)
        threshold = _threshold_for_frame(t)

        risk_grid = crash_probability(vol_grid, alpha_grid, threshold=threshold)

        cur_vol = traj_vol[frame]
        cur_alpha = traj_alpha[frame]
        cur_risk = float(crash_probability(
            np.array([cur_vol]), np.array([cur_alpha]), threshold=threshold,
        )[0])

        trajectory_history_vol.append(cur_vol)
        trajectory_history_alpha.append(cur_alpha)
        trajectory_history_risk.append(cur_risk)
        trajectory_history_time.append(t)

        # ---- Main 3D surface ---------------------------------------------
        ax_3d.clear()
        ax_3d.plot_surface(
            vol_grid, alpha_grid, risk_grid,
            cmap=cmap, linewidth=0, antialiased=True,
            alpha=0.92, rcount=60, ccount=60,
        )
        ax_3d.contourf(
            vol_grid, alpha_grid, risk_grid,
            zdir="z", offset=-0.05, cmap=cmap,
            levels=np.linspace(0, 1, 11), alpha=0.55,
        )

        warn_levels = [0.05, 0.15, 0.30, 0.50]
        cs = ax_3d.contour(
            vol_grid, alpha_grid, risk_grid,
            zdir="z", offset=-0.05, levels=warn_levels,
            colors="black", linewidths=0.8, linestyles="--",
        )
        ax_3d.clabel(cs, fmt={lvl: f"{int(lvl*100)}%" for lvl in warn_levels},
                     fontsize=7)

        if len(trajectory_history_vol) > 1:
            ax_3d.plot(
                trajectory_history_vol,
                trajectory_history_alpha,
                np.array(trajectory_history_risk) + 0.015,
                color="white", linewidth=2.2, alpha=0.85,
            )
            ax_3d.plot(
                trajectory_history_vol,
                trajectory_history_alpha,
                np.array(trajectory_history_risk) + 0.015,
                color="black", linewidth=0.8, alpha=0.7,
            )

        ax_3d.scatter(
            [cur_vol], [cur_alpha], [cur_risk + 0.04],
            color="#2ca02c", edgecolor="black", s=170,
            marker="D", zorder=11, linewidth=1.0,
        )

        ax_3d.set_xlabel("\nVolatility (annualised)", fontsize=10)
        ax_3d.set_ylabel("\nTail index $\\alpha$", fontsize=10)
        ax_3d.set_zlabel("\nCrash probability", fontsize=10)
        ax_3d.set_xlim(0.05, 0.85)
        ax_3d.set_ylim(1.05, 2.00)
        ax_3d.set_zlim(-0.05, 1.0)
        ax_3d.invert_yaxis()
        azim = -65 + 28 * np.sin(2 * np.pi * t)
        elev = 24 + 4 * np.sin(2 * np.pi * t * 0.5)
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.set_title(
            f"Tail-Risk Surface  -  threshold = {threshold*100:.1f}%   "
            f"horizon = 30 trading days",
            fontsize=11, fontweight="bold", pad=12,
        )

        # ---- Top-down view -----------------------------------------------
        ax_top.clear()
        pcm = ax_top.pcolormesh(
            vol_axis, alpha_axis, risk_grid,
            cmap=cmap, shading="auto", vmin=0, vmax=1,
        )
        ax_top.contour(
            vol_axis, alpha_axis, risk_grid,
            levels=warn_levels, colors="black",
            linewidths=0.9, linestyles="--",
        )
        if len(trajectory_history_vol) > 1:
            ax_top.plot(
                trajectory_history_vol, trajectory_history_alpha,
                color="white", linewidth=2.2, alpha=0.9,
            )
        ax_top.scatter(
            [cur_vol], [cur_alpha],
            color="#2ca02c", edgecolor="black",
            s=110, marker="D", zorder=10,
        )
        ax_top.set_xlabel("Volatility")
        ax_top.set_ylabel("Tail index $\\alpha$")
        ax_top.invert_yaxis()
        ax_top.set_title("Top-down view", fontsize=10, fontweight="bold")

        # ---- Probability time-series -------------------------------------
        ax_prob.clear()
        ax_prob.plot(
            trajectory_history_time,
            np.array(trajectory_history_risk) * 100,
            color="#d62728", linewidth=2.2,
        )
        ax_prob.fill_between(
            trajectory_history_time,
            0, np.array(trajectory_history_risk) * 100,
            color="#d62728", alpha=0.18,
        )
        ax_prob.axhline(5, color="grey", linestyle=":", linewidth=0.9)
        ax_prob.axhline(15, color="orange", linestyle=":", linewidth=0.9)
        ax_prob.axhline(30, color="red", linestyle=":", linewidth=0.9)
        ax_prob.set_xlim(0, 1)
        ax_prob.set_ylim(0, 100)
        ax_prob.set_xlabel("Scenario progress")
        ax_prob.set_ylabel("Crash prob. (%)")
        ax_prob.set_title("Live readout", fontsize=10, fontweight="bold")
        ax_prob.grid(alpha=0.3)

        progress_pct = 100 * t
        fig.suptitle(
            "How the tail-risk surface deforms when threshold and market "
            f"state change   -   scenario progress {progress_pct:5.1f}%",
            fontsize=12, fontweight="bold", y=0.985,
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=num_frames, interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=85)
    plt.close(fig)
    return output_path


def main():
    out = Path("outputs/tail_risk_surface_animation.gif")
    saved = render_animation(out)
    size_mb = saved.stat().st_size / (1024 * 1024)
    print(f"Saved: {saved} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
