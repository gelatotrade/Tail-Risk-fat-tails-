"""
Animated Tail Risk Visualizations
=================================

Animations reveal the temporal evolution of tail risk,
showing how market states change over time.

Key Animations:
1. Phase space trajectory evolution
2. Distribution morphing (fat tails emerging)
3. Risk surface dynamics
4. Real-time risk gauge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Dict, List, Callable
import warnings


class PhaseSpaceAnimator:
    """
    Animate trajectory through 3D phase space.

    Shows how the market state evolves over time,
    revealing patterns and regime transitions.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                 figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize animator.

        Args:
            X, Y, Z: Coordinate arrays
            figsize: Figure size
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        self.n = len(X)
        self.figsize = figsize

    def create_animation(self, trail_length: int = 50,
                        interval: int = 50,
                        save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animated phase space trajectory.

        Args:
            trail_length: Number of points in the trailing path
            interval: Milliseconds between frames
            save_path: Optional path to save animation

        Returns:
            FuncAnimation object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Set axis limits
        ax.set_xlim(np.nanmin(self.X), np.nanmax(self.X))
        ax.set_ylim(np.nanmin(self.Y), np.nanmax(self.Y))
        ax.set_zlim(np.nanmin(self.Z), np.nanmax(self.Z))

        ax.set_xlabel('Volatility')
        ax.set_ylabel('Tail Heaviness')
        ax.set_zlabel('Risk Level')
        ax.set_title('Phase Space Evolution')

        # Initialize plot elements
        trail, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1)
        current_point = ax.scatter([], [], [], c='red', s=100, marker='o')
        history = ax.scatter([], [], [], c=[], cmap='viridis', s=10, alpha=0.3)

        def init():
            return trail, current_point, history

        def update(frame):
            start = max(0, frame - trail_length)

            # Update trail
            trail.set_data(self.X[start:frame+1], self.Y[start:frame+1])
            trail.set_3d_properties(self.Z[start:frame+1])

            # Update current point
            current_point._offsets3d = ([self.X[frame]], [self.Y[frame]], [self.Z[frame]])

            # Update title with current state
            ax.set_title(f'Phase Space Evolution (t={frame})\n'
                        f'Volatility={self.X[frame]:.3f}, Risk={self.Z[frame]:.3f}')

            return trail, current_point, history

        anim = FuncAnimation(fig, update, frames=range(0, self.n, 2),
                            init_func=init, blit=False, interval=interval)

        if save_path:
            writer = PillowWriter(fps=20)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        return anim


class DistributionMorphAnimator:
    """
    Animate the evolution of return distributions.

    Shows how distributions develop fat tails over time
    (e.g., during crisis buildup).
    """

    def __init__(self, returns: np.ndarray, window: int = 60):
        """
        Initialize animator.

        Args:
            returns: Time series of returns
            window: Rolling window for distribution estimation
        """
        self.returns = returns
        self.window = window
        self.n = len(returns)

    def create_animation(self, interval: int = 100,
                        save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animated distribution evolution.

        Shows both the histogram and fitted distribution changing.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # x-axis for distributions
        x = np.linspace(-5, 5, 200)

        def update(frame):
            start = max(0, frame - self.window)
            window_returns = self.returns[start:frame+1]

            if len(window_returns) < 10:
                return

            # Clear axes
            for ax in axes:
                ax.clear()

            # Left: Histogram with Gaussian overlay
            std = np.std(window_returns)
            mean = np.mean(window_returns)

            axes[0].hist(window_returns, bins=30, density=True, alpha=0.7,
                        color='blue', label='Empirical')

            # Gaussian
            from scipy.stats import norm
            gaussian = norm.pdf(x, mean, std)
            axes[0].plot(x * std + mean, gaussian / std, 'r-', linewidth=2,
                        label='Gaussian')

            axes[0].set_xlabel('Returns')
            axes[0].set_ylabel('Density')
            axes[0].set_title(f'Distribution at t={frame}')
            axes[0].legend()
            axes[0].set_xlim(-0.1, 0.1)

            # Right: Log scale to show tails
            axes[1].hist(window_returns, bins=30, density=True, alpha=0.7,
                        color='blue', log=True)
            axes[1].plot(x * std + mean, gaussian / std, 'r-', linewidth=2)

            # Compute tail metric
            kurtosis = np.mean((window_returns - mean)**4) / std**4 - 3

            axes[1].set_xlabel('Returns')
            axes[1].set_ylabel('Log Density')
            axes[1].set_title(f'Log Scale (Excess Kurtosis: {kurtosis:.2f})')
            axes[1].set_xlim(-0.1, 0.1)

        frames = range(self.window, self.n, 5)
        anim = FuncAnimation(fig, update, frames=frames, interval=interval)

        if save_path:
            writer = PillowWriter(fps=10)
            anim.save(save_path, writer=writer)

        return anim


class RiskGaugeAnimator:
    """
    Animated risk gauge showing current tail risk level.

    Like a speedometer but for market risk.
    """

    def __init__(self, risk_series: np.ndarray):
        """
        Initialize gauge animator.

        Args:
            risk_series: Time series of risk values (0-100 scale)
        """
        self.risk = np.clip(risk_series, 0, 100)
        self.n = len(risk_series)

    def create_animation(self, interval: int = 100,
                        save_path: Optional[str] = None) -> FuncAnimation:
        """Create animated risk gauge."""
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

        # Gauge setup
        theta_min, theta_max = np.pi, 0  # Left to right
        theta = np.linspace(theta_min, theta_max, 100)

        # Color zones
        zones = [
            (0, 25, 'green', 'Low'),
            (25, 50, 'yellow', 'Moderate'),
            (50, 75, 'orange', 'Elevated'),
            (75, 100, 'red', 'Extreme')
        ]

        def update(frame):
            ax.clear()

            # Draw gauge background
            for zone_min, zone_max, color, label in zones:
                theta_zone = np.linspace(
                    theta_min - (theta_min - theta_max) * zone_min / 100,
                    theta_min - (theta_min - theta_max) * zone_max / 100,
                    50
                )
                ax.fill_between(theta_zone, 0.5, 1, color=color, alpha=0.5)

            # Current risk level
            risk_level = self.risk[frame]
            needle_theta = theta_min - (theta_min - theta_max) * risk_level / 100

            # Draw needle
            ax.plot([needle_theta, needle_theta], [0, 0.9], 'k-', linewidth=3)
            ax.plot([needle_theta], [0.9], 'ko', markersize=10)

            # Settings
            ax.set_ylim(0, 1.2)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetalim(0, np.pi)

            # Remove radial ticks
            ax.set_rticks([])
            ax.set_xticks([])

            # Title with current value
            ax.set_title(f'Tail Risk Gauge\nCurrent Level: {risk_level:.1f}%',
                        fontsize=14, fontweight='bold', pad=20)

        anim = FuncAnimation(fig, update, frames=range(self.n),
                            interval=interval)

        if save_path:
            writer = PillowWriter(fps=10)
            anim.save(save_path, writer=writer)

        return anim


def create_comprehensive_animation(returns: np.ndarray,
                                  levy_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  save_path: Optional[str] = None) -> FuncAnimation:
    """
    Create comprehensive animated dashboard.

    Shows multiple synchronized views of tail risk evolution.
    """
    from scipy import stats

    n = len(returns)

    fig = plt.figure(figsize=(16, 12))

    # 3D Phase space
    ax1 = fig.add_subplot(221, projection='3d')

    # Return distribution
    ax2 = fig.add_subplot(222)

    # Risk time series
    ax3 = fig.add_subplot(223)

    # VaR evolution
    ax4 = fig.add_subplot(224)

    window = 60

    def update(frame):
        if frame < window:
            return

        # Clear all axes
        for ax in [ax2, ax3, ax4]:
            ax.clear()

        # 1. Phase space trajectory
        ax1.clear()
        start = max(0, frame - 100)
        ax1.scatter(levy_coords[0][start:frame], levy_coords[1][start:frame],
                   levy_coords[2][start:frame], c=np.arange(frame - start),
                   cmap='viridis', s=10, alpha=0.5)
        ax1.scatter([levy_coords[0][frame]], [levy_coords[1][frame]],
                   [levy_coords[2][frame]], c='red', s=100, marker='o')
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Tail Index')
        ax1.set_zlabel('Jump Intensity')
        ax1.set_title(f'Phase Space (t={frame})')

        # 2. Distribution
        window_returns = returns[frame-window:frame]
        ax2.hist(window_returns, bins=30, density=True, alpha=0.7, color='blue')

        x = np.linspace(-0.1, 0.1, 100)
        gaussian = stats.norm.pdf(x, np.mean(window_returns), np.std(window_returns))
        ax2.plot(x, gaussian, 'r-', linewidth=2, label='Gaussian')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Density')
        ax2.set_title('Return Distribution')
        ax2.legend()
        ax2.set_xlim(-0.1, 0.1)

        # 3. Risk time series
        risk = levy_coords[2][:frame]
        ax3.plot(risk, 'b-', linewidth=1)
        ax3.axhline(y=np.percentile(levy_coords[2], 90), color='r', linestyle='--',
                   label='90th percentile')
        ax3.scatter([frame-1], [risk[-1]], c='red', s=50, zorder=5)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Risk Level')
        ax3.set_title('Tail Risk Evolution')
        ax3.legend()

        # 4. Rolling VaR
        var_series = []
        for i in range(window, frame):
            var_series.append(-np.percentile(returns[i-window:i], 1))
        ax4.plot(range(window, frame), var_series, 'b-', linewidth=1)
        if len(var_series) > 0:
            ax4.scatter([frame-1], [var_series[-1]], c='red', s=50, zorder=5)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('VaR (99%)')
        ax4.set_title('Rolling Value at Risk')

        plt.suptitle(f'Tail Risk Dashboard - Time: {frame}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    frames = range(window + 10, n, 5)
    anim = FuncAnimation(fig, update, frames=frames, interval=100)

    if save_path:
        writer = PillowWriter(fps=15)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    return anim
