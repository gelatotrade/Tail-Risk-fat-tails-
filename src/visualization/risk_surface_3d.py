"""
3D Tail Risk Surface Visualization
===================================

This module creates sophisticated 3D visualizations of tail risk
using physics-inspired coordinate systems.

The 3D Risk Phase Space:
========================
We map financial market states into a 3-dimensional phase space
inspired by statistical physics:

Axis Interpretations (Multiple Coordinate Systems):

1. Lévy Flight Coordinates:
   - X: Volatility regime σ (normalized)
   - Y: Tail index α (fatness of tails)
   - Z: Jump intensity λ (frequency of large moves)

2. Thermodynamic Coordinates (Tsallis):
   - X: Entropic index q (non-extensivity)
   - Y: Tsallis entropy S_q (uncertainty)
   - Z: Temperature β^(-1) (volatility energy)

3. Phase Transition Coordinates:
   - X: Susceptibility χ (sensitivity to shocks)
   - Y: Order parameter M (trend strength)
   - Z: Distance from criticality |T - T_c|

Each coordinate system reveals different aspects of tail risk.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Tuple, Optional, Dict, List
import warnings


class TailRisk3DSurface:
    """
    Generate 3D tail risk surface visualizations.

    The surface represents tail risk probability as a function
    of two market state variables.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize 3D visualizer.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self._setup_colormaps()

    def _setup_colormaps(self):
        """Create custom colormaps for risk visualization."""
        # Risk colormap: green (low) -> yellow -> red (high) -> purple (extreme)
        colors = ['#00ff00', '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#ff00ff']
        self.risk_cmap = LinearSegmentedColormap.from_list('risk', colors)

        # Phase colormap: blue (stable) -> white (critical) -> red (unstable)
        phase_colors = ['#0000ff', '#4040ff', '#8080ff', '#ffffff',
                       '#ff8080', '#ff4040', '#ff0000']
        self.phase_cmap = LinearSegmentedColormap.from_list('phase', phase_colors)

    def create_risk_surface(self, vol_range: np.ndarray,
                           tail_index_range: np.ndarray,
                           compute_risk_func,
                           title: str = "3D Tail Risk Surface") -> plt.Figure:
        """
        Create 3D surface plot of tail risk.

        Args:
            vol_range: Array of volatility values (x-axis)
            tail_index_range: Array of tail index values (y-axis)
            compute_risk_func: Function(vol, tail_index) -> risk
            title: Plot title

        Returns:
            Matplotlib figure
        """
        X, Y = np.meshgrid(vol_range, tail_index_range)
        Z = np.zeros_like(X)

        for i in range(len(tail_index_range)):
            for j in range(len(vol_range)):
                Z[i, j] = compute_risk_func(vol_range[j], tail_index_range[i])

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=self.risk_cmap,
                               linewidth=0, antialiased=True, alpha=0.8)

        # Add contours on the base
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=self.risk_cmap, alpha=0.5)

        # Labels
        ax.set_xlabel('Volatility (σ)', fontsize=12)
        ax.set_ylabel('Tail Index (α)', fontsize=12)
        ax.set_zlabel('Tail Risk Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Risk Level')

        return fig

    def create_phase_space_trajectory(self, X: np.ndarray, Y: np.ndarray,
                                      Z: np.ndarray, time: Optional[np.ndarray] = None,
                                      title: str = "Risk Phase Space Trajectory") -> plt.Figure:
        """
        Visualize the trajectory through 3D phase space.

        Args:
            X, Y, Z: Coordinate arrays (same length)
            time: Optional time array for coloring
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        if time is None:
            time = np.arange(len(X))

        # Normalize time for colormap
        norm = Normalize(vmin=time.min(), vmax=time.max())

        # Plot trajectory with time-based coloring
        for i in range(len(X) - 1):
            ax.plot([X[i], X[i+1]], [Y[i], Y[i+1]], [Z[i], Z[i+1]],
                   color=self.risk_cmap(norm(time[i])), linewidth=1.5, alpha=0.7)

        # Mark start and end points
        ax.scatter([X[0]], [Y[0]], [Z[0]], c='green', s=100, marker='o', label='Start')
        ax.scatter([X[-1]], [Y[-1]], [Z[-1]], c='red', s=100, marker='s', label='Current')

        # Mark extreme points
        max_z_idx = np.argmax(Z)
        ax.scatter([X[max_z_idx]], [Y[max_z_idx]], [Z[max_z_idx]],
                  c='purple', s=150, marker='*', label='Peak Risk')

        ax.set_xlabel('X: Volatility', fontsize=12)
        ax.set_ylabel('Y: Tail Heaviness', fontsize=12)
        ax.set_zlabel('Z: Risk Level', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

        return fig

    def create_multi_coordinate_view(self, levy_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                     tsallis_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                     phase_coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> plt.Figure:
        """
        Create side-by-side 3D views of different coordinate systems.

        Shows the same market state in three different "physics spaces".
        """
        fig = plt.figure(figsize=(18, 6))

        # Lévy coordinates
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(levy_coords[0], levy_coords[1], levy_coords[2],
                   c=np.arange(len(levy_coords[0])), cmap='viridis', s=20, alpha=0.6)
        ax1.set_xlabel('σ (Volatility)')
        ax1.set_ylabel('α (Tail Index)')
        ax1.set_zlabel('λ (Jump Intensity)')
        ax1.set_title('Lévy Flight Space', fontweight='bold')

        # Tsallis coordinates
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(tsallis_coords[0], tsallis_coords[1], tsallis_coords[2],
                   c=np.arange(len(tsallis_coords[0])), cmap='plasma', s=20, alpha=0.6)
        ax2.set_xlabel('q (Entropic Index)')
        ax2.set_ylabel('S_q (Entropy)')
        ax2.set_zlabel('T (Temperature)')
        ax2.set_title('Thermodynamic Space', fontweight='bold')

        # Phase transition coordinates
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(phase_coords[0], phase_coords[1], phase_coords[2],
                   c=np.arange(len(phase_coords[0])), cmap='coolwarm', s=20, alpha=0.6)
        ax3.set_xlabel('χ (Susceptibility)')
        ax3.set_ylabel('M (Order Parameter)')
        ax3.set_zlabel('|T-Tc| (Criticality)')
        ax3.set_title('Phase Transition Space', fontweight='bold')

        plt.tight_layout()
        return fig


class TailRiskDensityPlot:
    """
    Create 3D density plots showing tail probability distributions.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize

    def create_joint_tail_density(self, returns1: np.ndarray, returns2: np.ndarray,
                                  asset_names: Tuple[str, str] = ('Asset 1', 'Asset 2'),
                                  title: str = "Joint Tail Density") -> plt.Figure:
        """
        Create 3D visualization of joint tail density between two assets.

        Shows how tail events are correlated between assets.
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Create 2D histogram
        x_edges = np.linspace(np.min(returns1), np.max(returns1), 50)
        y_edges = np.linspace(np.min(returns2), np.max(returns2), 50)

        H, xedges, yedges = np.histogram2d(returns1, returns2, bins=[x_edges, y_edges], density=True)

        # Create meshgrid for plotting
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        # Plot surface
        surf = ax.plot_surface(X, Y, H.T, cmap='viridis',
                               linewidth=0, antialiased=True, alpha=0.8)

        # Mark tail regions
        std1 = np.std(returns1)
        std2 = np.std(returns2)

        # Left tail region
        ax.plot([returns1.min(), -2*std1], [-2*std2, -2*std2], [0, 0],
                'r--', linewidth=2, label='Left tail boundary')

        ax.set_xlabel(f'{asset_names[0]} Returns', fontsize=12)
        ax.set_ylabel(f'{asset_names[1]} Returns', fontsize=12)
        ax.set_zlabel('Joint Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        return fig

    def create_tail_probability_surface(self, returns: np.ndarray,
                                        horizon_range: np.ndarray,
                                        confidence_range: np.ndarray) -> plt.Figure:
        """
        3D surface of tail probabilities across horizons and confidence levels.

        X: Time horizon
        Y: Confidence level
        Z: VaR or tail probability
        """
        from ..models.risk_metrics import TailRiskMetrics

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(horizon_range, confidence_range)
        Z = np.zeros_like(X)

        metrics = TailRiskMetrics(returns)

        for i, conf in enumerate(confidence_range):
            for j, horizon in enumerate(horizon_range):
                # Scale VaR by sqrt(horizon) for simplicity
                Z[i, j] = metrics.var_historical(conf) * np.sqrt(horizon)

        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn_r',
                               linewidth=0, antialiased=True, alpha=0.8)

        ax.set_xlabel('Time Horizon (days)', fontsize=12)
        ax.set_ylabel('Confidence Level', fontsize=12)
        ax.set_zlabel('VaR', fontsize=12)
        ax.set_title('VaR Surface: Horizon × Confidence', fontsize=14, fontweight='bold')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        return fig


class CriticalPointVisualizer:
    """
    Visualize the approach to critical points (phase transitions).

    Near financial crashes, markets exhibit "critical slowing down"
    similar to physical phase transitions.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        self.figsize = figsize

    def create_critical_surface(self, autocorr: np.ndarray, variance: np.ndarray,
                               skewness: np.ndarray, time: np.ndarray) -> plt.Figure:
        """
        3D visualization of Early Warning Signals approaching criticality.

        As the system approaches critical point:
        - Autocorrelation increases (X axis)
        - Variance increases (Y axis)
        - Skewness becomes more negative (Z axis)
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Create segments colored by time
        norm = Normalize(vmin=time.min(), vmax=time.max())

        # Plot trajectory
        scatter = ax.scatter(autocorr, variance, skewness,
                            c=time, cmap='hot', s=30, alpha=0.7)

        # Draw path
        ax.plot(autocorr, variance, skewness, 'b-', alpha=0.3, linewidth=0.5)

        # Mark critical point region
        crit_mask = (autocorr > 0.7) & (variance > np.percentile(variance, 80))
        if np.any(crit_mask):
            ax.scatter(autocorr[crit_mask], variance[crit_mask], skewness[crit_mask],
                      c='red', s=100, marker='x', label='Near Critical')

        ax.set_xlabel('Autocorrelation (AC-1)', fontsize=12)
        ax.set_ylabel('Variance', fontsize=12)
        ax.set_zlabel('Skewness', fontsize=12)
        ax.set_title('Approach to Critical Point (Phase Transition)', fontsize=14, fontweight='bold')

        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Time')
        ax.legend()

        return fig


def create_comprehensive_3d_dashboard(returns: np.ndarray,
                                     vix: Optional[np.ndarray] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive 3D tail risk dashboard.

    Includes:
    1. Risk surface plot
    2. Phase space trajectory
    3. Critical point approach
    4. Distribution comparison
    """
    from ..physics.levy_flight import levy_flight_3d_coordinates
    from ..physics.tsallis_statistics import compute_tsallis_coordinates
    from ..physics.phase_transitions import compute_phase_space_coordinates

    # Compute all coordinate systems
    levy_coords = levy_flight_3d_coordinates(returns)
    tsallis_coords = compute_tsallis_coordinates(returns)
    phase_coords = compute_phase_space_coordinates(returns)

    # Create figure with multiple 3D subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Lévy Phase Space
    ax1 = fig.add_subplot(221, projection='3d')
    time_color = np.arange(len(returns))
    scatter1 = ax1.scatter(levy_coords[0], levy_coords[1], levy_coords[2],
                          c=time_color, cmap='viridis', s=15, alpha=0.6)
    ax1.set_xlabel('Volatility (σ)')
    ax1.set_ylabel('Tail Index (α⁻¹)')
    ax1.set_zlabel('Jump Intensity')
    ax1.set_title('Lévy Flight Phase Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Time', shrink=0.6)

    # 2. Tsallis Thermodynamic Space
    ax2 = fig.add_subplot(222, projection='3d')
    scatter2 = ax2.scatter(tsallis_coords[0], tsallis_coords[1], tsallis_coords[2],
                          c=time_color, cmap='plasma', s=15, alpha=0.6)
    ax2.set_xlabel('q (Entropic Index)')
    ax2.set_ylabel('Entropy')
    ax2.set_zlabel('Temperature')
    ax2.set_title('Thermodynamic Phase Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Time', shrink=0.6)

    # 3. Phase Transition Space
    ax3 = fig.add_subplot(223, projection='3d')
    scatter3 = ax3.scatter(phase_coords[0], phase_coords[1], phase_coords[2],
                          c=time_color, cmap='coolwarm', s=15, alpha=0.6)
    ax3.set_xlabel('Susceptibility')
    ax3.set_ylabel('Order Parameter')
    ax3.set_zlabel('Criticality Distance')
    ax3.set_title('Critical Phenomena Space', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Time', shrink=0.6)

    # 4. Composite Risk Surface
    ax4 = fig.add_subplot(224, projection='3d')

    # Create risk surface from coordinates
    risk_level = (levy_coords[2] + 0.5 * tsallis_coords[0] + phase_coords[0]) / 2.5
    scatter4 = ax4.scatter(levy_coords[0], tsallis_coords[1], risk_level,
                          c=risk_level, cmap='RdYlGn_r', s=20, alpha=0.7)
    ax4.set_xlabel('Volatility')
    ax4.set_ylabel('Entropy')
    ax4.set_zlabel('Composite Risk')
    ax4.set_title('Composite Tail Risk Surface', fontsize=12, fontweight='bold')
    plt.colorbar(scatter4, ax=ax4, label='Risk Level', shrink=0.6)

    plt.suptitle('Comprehensive 3D Tail Risk Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")

    return fig
