"""
Early Warning Dashboard Visualization
======================================

Comprehensive visualization of the 5 early warning indicators:
1. Net Gamma Exposure (GEX) - Market mechanics
2. TailDex (TDEX) - Tail risk pricing
3. VIX Term Structure - Fear curve
4. Dark Index (DIX) - Institutional flows
5. Smart Money Flow (SMFI) - Professional positioning

This module provides:
- Individual indicator panels
- Composite score dashboard
- 3D phase space visualization
- Historical comparison with crash events
"""

import zlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.dates as mdates
from typing import Dict, Optional, Tuple, List
import warnings


def _scoring_system() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Indicator weights and composite level thresholds of the scoring system.

    Read from CompositeEarlyWarningSystem (single source of truth), so the
    dashboard and the summary table name the same level as compute_composite
    and get_risk_assessment (NORMAL < 50 <= ELEVATED < 75 <= HIGH < 90 <= EXTREME).
    """
    from ..analysis.early_warning import CompositeEarlyWarningSystem

    ews = CompositeEarlyWarningSystem()
    return dict(ews.weights), dict(ews.thresholds)


def _composite_level(score: float, thresholds: Dict[str, float]) -> str:
    if score >= thresholds['extreme']:
        return 'EXTREME'
    if score >= thresholds['high']:
        return 'HIGH'
    if score >= thresholds['elevated']:
        return 'ELEVATED'
    return 'NORMAL'


def _x_numeric(ax, x) -> np.ndarray:
    """x in the axis' float units (datetime64 -> Matplotlib date numbers); registers the units."""
    ax.xaxis.update_units(x)
    return np.asarray(ax.xaxis.convert_units(x), dtype=float)


def _bars(ax, x, heights, colors, alpha, width=1.0):
    """
    The picture of ``ax.bar(x, heights, width=width, color=colors, alpha=alpha)``
    as ONE collection instead of one Rectangle per day (fast for daily series).
    """
    xn = _x_numeric(ax, x)
    h = np.nan_to_num(np.asarray(heights, dtype=float), nan=0.0)
    left, right, zero = xn - width / 2, xn + width / 2, np.zeros_like(h)
    verts = np.stack([np.column_stack([left, zero]), np.column_stack([left, h]),
                      np.column_stack([right, h]), np.column_stack([right, zero])], axis=1)
    bars = PolyCollection(verts, facecolors=colors, edgecolors='none', alpha=alpha)
    bars.sticky_edges.y.append(0)  # like bar(): no margin below a zero baseline
    ax.add_collection(bars, autolim=True)
    ax.autoscale_view()
    return bars


class EarlyWarningDashboard:
    """
    Comprehensive early warning dashboard visualization.

    Creates a multi-panel dashboard showing all 5 indicators
    plus composite risk score with historical context.
    """

    def __init__(self, figsize: Tuple[int, int] = (20, 16)):
        """
        Initialize dashboard.

        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize

        # Weights and composite levels of the scoring system being displayed
        self.weights, self.thresholds = _scoring_system()

        # Color schemes
        self.risk_colors = {
            'safe': '#00ff00',      # Green
            'elevated': '#ffff00',   # Yellow
            'high': '#ff8000',       # Orange
            'extreme': '#ff0000',    # Red
            'critical': '#ff00ff'    # Magenta
        }

        # Create custom colormap for risk gradient
        colors = ['#00ff00', '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#ff00ff']
        self.risk_cmap = LinearSegmentedColormap.from_list('risk', colors, N=256)

        # Indicator colors
        self.indicator_colors = {
            'gex': '#e74c3c',      # Red
            'tdex': '#9b59b6',     # Purple
            'vix_term': '#3498db', # Blue
            'dix': '#2ecc71',      # Green
            'smfi': '#f39c12'      # Orange
        }

    def create_dashboard(self,
                         indicators: Dict[str, np.ndarray],
                         prices: np.ndarray,
                         dates: Optional[np.ndarray] = None,
                         title: str = "Early Warning System Dashboard") -> plt.Figure:
        """
        Create comprehensive early warning dashboard.

        Args:
            indicators: Dict with all indicator time series
            prices: Price time series
            dates: Optional date array
            title: Dashboard title

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor('#1a1a2e')

        # Create grid: 4 rows, 4 columns
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.25)

        # Title
        fig.suptitle(title, fontsize=18, fontweight='bold', color='white', y=0.98)

        n = len(prices)
        x = dates if dates is not None else np.arange(n)

        # Row 1: Main price chart with composite score overlay
        ax_main = fig.add_subplot(gs[0, :3])
        self._plot_price_with_composite(ax_main, x, prices, indicators['composite_score'])

        # Row 1, Col 4: Current Risk Gauge
        ax_gauge = fig.add_subplot(gs[0, 3])
        self._plot_risk_gauge(ax_gauge, indicators['composite_score'][-1])

        # Row 2: Individual indicators (GEX, TDEX)
        ax_gex = fig.add_subplot(gs[1, :2])
        self._plot_gex(ax_gex, x, indicators['gex'], indicators['gex_score'])

        ax_tdex = fig.add_subplot(gs[1, 2:])
        self._plot_tdex(ax_tdex, x, indicators['tdex'], indicators['tdex_score'])

        # Row 3: VIX Term Structure, DIX
        ax_vix = fig.add_subplot(gs[2, :2])
        self._plot_vix_term_structure(ax_vix, x, indicators['vix_m1'],
                                       indicators['vix_m2'], indicators['vix_term_score'])

        ax_dix = fig.add_subplot(gs[2, 2:])
        self._plot_dix(ax_dix, x, indicators['dix'], prices, indicators['dix_score'])

        # Row 4: SMFI, Composite Score History, Indicator Weights
        ax_smfi = fig.add_subplot(gs[3, :2])
        self._plot_smfi(ax_smfi, x, indicators['smfi'], prices, indicators['smfi_score'])

        ax_composite = fig.add_subplot(gs[3, 2])
        self._plot_composite_history(ax_composite, x, indicators['composite_score'])

        ax_weights = fig.add_subplot(gs[3, 3])
        self._plot_indicator_weights(ax_weights, indicators)

        # plt.tight_layout skips a GridSpec with explicit hspace/wspace (it only
        # warned and left twin-axis labels overlapping) and acts on the *current*
        # figure; lay out this figure's grid directly.
        gs.tight_layout(fig, rect=[0, 0.02, 1, 0.96])
        return fig

    def _plot_price_with_composite(self, ax, x, prices, composite):
        """Plot price with composite score background."""
        ax.set_facecolor('#16213e')

        # Background colour by composite score: the span x[i-1]..x[i] gets the
        # colour of day i (one collection instead of one axvspan per day)
        n = len(prices)
        if n > 1:
            xn = _x_numeric(ax, x)
            lo, hi = np.zeros(n - 1), np.ones(n - 1)
            verts = np.stack([np.column_stack([xn[:-1], lo]), np.column_stack([xn[:-1], hi]),
                              np.column_stack([xn[1:], hi]), np.column_stack([xn[1:], lo])], axis=1)
            colors = self.risk_cmap(np.asarray(composite, dtype=float)[1:n] / 100)
            ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors=colors, alpha=0.3,
                                             transform=ax.get_xaxis_transform()),
                              autolim=False)  # x range is set by the price line below

        # Plot price
        ax.plot(x, prices, color='white', linewidth=1.5, label='Price')
        # fill down to the lowest price, not to 0 (which squashed the price line)
        ax.fill_between(x, prices, np.nanmin(prices), alpha=0.1, color='white')

        ax.set_title('Price with Composite Risk Overlay', fontsize=12,
                     fontweight='bold', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add legend for risk levels
        from matplotlib.patches import Patch
        el, hi, ex = (self.thresholds[k] for k in ('elevated', 'high', 'extreme'))
        legend_elements = [
            Patch(facecolor='#00ff00', alpha=0.5, label=f'Normal (<{el:g})'),
            Patch(facecolor='#ffff00', alpha=0.5, label=f'Elevated ({el:g}-{hi:g})'),
            Patch(facecolor='#ff8000', alpha=0.5, label=f'High ({hi:g}-{ex:g})'),
            Patch(facecolor='#ff0000', alpha=0.5, label=f'Extreme (\u2265{ex:g})')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                  facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    def _plot_risk_gauge(self, ax, current_score):
        """Plot circular risk gauge."""
        ax.set_facecolor('#16213e')
        ax.set_aspect('equal')

        # Draw gauge background
        theta = np.linspace(0.75 * np.pi, 0.25 * np.pi, 100)
        r = 1.0

        # Background arc segments: the composite levels of the scoring system
        el, hi, ex = (self.thresholds[k] for k in ('elevated', 'high', 'extreme'))
        segments = [
            (0, el, '#00ff00'),
            (el, hi, '#ffff00'),
            (hi, ex, '#ff8000'),
            (ex, 100, '#ff0000')
        ]

        for start, end, color in segments:
            t_start = 0.75 * np.pi - (start / 100) * 0.5 * np.pi
            t_end = 0.75 * np.pi - (end / 100) * 0.5 * np.pi
            t = np.linspace(t_start, t_end, 20)
            x_outer = 1.0 * np.cos(t)
            y_outer = 1.0 * np.sin(t)
            x_inner = 0.7 * np.cos(t)
            y_inner = 0.7 * np.sin(t)
            ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
                    np.concatenate([y_outer, y_inner[::-1]]),
                    color=color, alpha=0.7)

        # Draw needle
        needle_angle = 0.75 * np.pi - (current_score / 100) * 0.5 * np.pi
        ax.arrow(0, 0, 0.6 * np.cos(needle_angle), 0.6 * np.sin(needle_angle),
                 head_width=0.1, head_length=0.05, fc='white', ec='white')

        # Center circle
        circle = plt.Circle((0, 0), 0.15, color='#1a1a2e', ec='white', linewidth=2)
        ax.add_patch(circle)

        # Score text
        ax.text(0, -0.5, f'{current_score:.0f}', fontsize=24, fontweight='bold',
                color='white', ha='center', va='center')
        ax.text(0, -0.75, 'COMPOSITE SCORE', fontsize=10, color='white',
                ha='center', va='center')

        # Risk level (same rule as CompositeEarlyWarningSystem.compute_composite)
        level = _composite_level(current_score, self.thresholds)
        color = {'EXTREME': '#ff0000', 'HIGH': '#ff8000',
                 'ELEVATED': '#ffff00', 'NORMAL': '#00ff00'}[level]

        ax.text(0, -0.95, level, fontsize=14, fontweight='bold',
                color=color, ha='center', va='center')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_title('Current Risk Level', fontsize=12, fontweight='bold',
                     color='white', pad=10)

    def _plot_gex(self, ax, x, gex, gex_score):
        """Plot Net Gamma Exposure."""
        ax.set_facecolor('#16213e')

        # Color based on positive/negative
        colors = ['#ff4444' if g < 0 else '#44ff44' for g in gex]

        _bars(ax, x, gex, colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)

        # Add danger zone
        ax.axhspan(-15, 0, alpha=0.1, color='red')
        ax.axhspan(0, 15, alpha=0.1, color='green')

        ax.set_title(f'1. Net Gamma Exposure (GEX) - {self.weights["gex"]:.0%} Weight', fontsize=11,
                     fontweight='bold', color='#e74c3c')
        ax.set_ylabel('GEX (Billions $)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add annotation
        current = gex[-1]
        regime = 'NEGATIVE (Accelerating)' if current < 0 else 'POSITIVE (Stabilizing)'
        ax.text(0.02, 0.98, f'Current: {current:.1f}B$ - {regime}',
                transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    def _plot_tdex(self, ax, x, tdex, tdex_score):
        """Plot TailDex."""
        ax.set_facecolor('#16213e')

        ax.plot(x, tdex, color='#9b59b6', linewidth=1.5)
        ax.fill_between(x, tdex, alpha=0.3, color='#9b59b6')

        # Add threshold lines
        ax.axhline(y=15, color='#ff8000', linestyle='--', linewidth=1, alpha=0.7, label='Elevated')
        ax.axhline(y=20, color='#ff0000', linestyle='--', linewidth=1, alpha=0.7, label='Extreme')

        ax.set_title(f'2. TailDex (TDEX) - {self.weights["tdex"]:.0%} Weight', fontsize=11,
                     fontweight='bold', color='#9b59b6')
        ax.set_ylabel('TDEX Level', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add annotation
        current = tdex[-1]
        level = 'EXTREME' if current > 20 else ('ELEVATED' if current > 15 else 'NORMAL')
        ax.text(0.02, 0.98, f'Current: {current:.1f} - {level}',
                transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    def _plot_vix_term_structure(self, ax, x, vix_m1, vix_m2, vix_term_score):
        """Plot VIX Term Structure."""
        ax.set_facecolor('#16213e')

        ratio = vix_m1 / vix_m2

        # Color based on contango/backwardation
        colors = ['#ff4444' if r > 1 else '#3498db' for r in ratio]

        ax.fill_between(x, ratio, 1.0, where=(ratio > 1), alpha=0.3, color='red',
                        label='Backwardation')
        ax.fill_between(x, ratio, 1.0, where=(ratio <= 1), alpha=0.3, color='blue',
                        label='Contango')
        ax.plot(x, ratio, color='white', linewidth=1.5)

        ax.axhline(y=1.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_title(f'3. VIX Term Structure (M1/M2) - {self.weights["vix_term"]:.0%} Weight', fontsize=11,
                     fontweight='bold', color='#3498db')
        ax.set_ylabel('M1/M2 Ratio', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                  edgecolor='white', labelcolor='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add annotation
        current = ratio[-1]
        structure = 'INVERTED (Panic)' if current > 1 else 'CONTANGO (Normal)'
        ax.text(0.02, 0.98, f'Current: {current:.3f} - {structure}',
                transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    def _plot_dix(self, ax, x, dix, prices, dix_score):
        """Plot Dark Index with price overlay."""
        ax.set_facecolor('#16213e')

        ax2 = ax.twinx()

        # DIX
        ax.fill_between(x, dix, 45, where=(dix < 43), alpha=0.3, color='red')
        ax.fill_between(x, dix, 45, where=(dix >= 43), alpha=0.3, color='green')
        ax.plot(x, dix, color='#2ecc71', linewidth=1.5, label='DIX')

        # Price on secondary axis
        ax2.plot(x, prices, color='white', linewidth=1, alpha=0.5, label='Price')

        ax.axhline(y=43, color='#ff8000', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=40, color='#ff0000', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_title(f'4. Dark Index (DIX) - {self.weights["dix"]:.0%} Weight', fontsize=11,
                     fontweight='bold', color='#2ecc71')
        ax.set_ylabel('DIX %', color='#2ecc71')
        ax2.set_ylabel('Price', color='white', alpha=0.5)
        ax.tick_params(colors='white')
        ax2.tick_params(colors='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        for spine in ax2.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add annotation
        current = dix[-1]
        level = 'DISTRIBUTION' if current < 40 else ('WEAK' if current < 43 else 'NORMAL')
        ax.text(0.02, 0.98, f'Current: {current:.1f}% - {level}',
                transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    def _plot_smfi(self, ax, x, smfi, prices, smfi_score):
        """Plot Smart Money Flow Index."""
        ax.set_facecolor('#16213e')

        ax2 = ax.twinx()

        # SMFI
        colors = ['#ff4444' if s < 0 else '#f39c12' for s in smfi]
        _bars(ax, x, smfi, colors, alpha=0.5, width=1.0)
        ax.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)

        # Price on secondary axis
        ax2.plot(x, prices, color='white', linewidth=1, alpha=0.5)

        ax.set_title(f'5. Smart Money Flow Index (SMFI) - {self.weights["smfi"]:.0%} Weight', fontsize=11,
                     fontweight='bold', color='#f39c12')
        ax.set_ylabel('SMFI', color='#f39c12')
        ax2.set_ylabel('Price', color='white', alpha=0.5)
        ax.tick_params(colors='white')
        ax2.tick_params(colors='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        for spine in ax2.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add annotation
        current = smfi[-1]
        signal = 'SELLING' if current < 0 else 'BUYING'
        ax.text(0.02, 0.98, f'Current: {current:.1f} - Smart Money {signal}',
                transform=ax.transAxes, fontsize=9, color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    def _plot_composite_history(self, ax, x, composite):
        """Plot composite score history."""
        ax.set_facecolor('#16213e')

        # Colour gradient: segment i-1 -> i gets the colour of day i
        # (one LineCollection instead of one Line2D per day)
        c = np.asarray(composite, dtype=float)
        pts = np.column_stack([_x_numeric(ax, x), c])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(segments, colors=self.risk_cmap(c[1:] / 100),
                                         linewidths=2, capstyle='projecting'), autolim=True)
        ax.autoscale_view()

        # Add threshold lines (composite levels of the scoring system)
        ax.axhline(y=self.thresholds['elevated'], color='#ffff00', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=self.thresholds['high'], color='#ff8000', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=self.thresholds['extreme'], color='#ff0000', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_title('Composite Score History', fontsize=11,
                     fontweight='bold', color='white')
        ax.set_ylabel('Score', color='white')
        ax.set_ylim(0, 100)

        # Narrow panel: at most 4 ticks, so date labels do not run into each other
        if np.issubdtype(np.asarray(x).dtype, np.datetime64):
            locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

    def _plot_indicator_weights(self, ax, indicators):
        """Plot indicator contribution breakdown."""
        ax.set_facecolor('#16213e')

        # Get latest scores
        w = self.weights
        scores = {
            'GEX': indicators['gex_score'][-1] * w['gex'],
            'TDEX': indicators['tdex_score'][-1] * w['tdex'],
            'VIX Term': indicators['vix_term_score'][-1] * w['vix_term'],
            'DIX': indicators['dix_score'][-1] * w['dix'],
            'SMFI': indicators['smfi_score'][-1] * w['smfi']
        }

        colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']
        names = list(scores.keys())
        values = list(scores.values())

        bars = ax.barh(names, values, color=colors, alpha=0.8)

        ax.set_title('Current Risk Contribution', fontsize=11,
                     fontweight='bold', color='white')
        ax.set_xlabel('Contribution to Score', color='white')
        ax.tick_params(colors='white')
        ax.set_xlim(0, max(40.0, 115 * max(w.values())))

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', va='center', color='white', fontsize=9)


class CrashComparisonChart:
    """
    Create comparison charts showing indicator behavior during historical crashes.
    """

    def __init__(self):
        """Initialize crash comparison chart."""
        self.crash_events = {
            '2018 Volmageddon': {
                'date': 'Feb 5, 2018',
                'drop': -4.1,  # Single day
                'vix_spike': 115.6,  # % change
                'trigger': 'Inverse VIX ETP Implosion'
            },
            '2020 COVID Crash': {
                'date': 'Mar 16, 2020',
                'drop': -34.0,  # Peak to trough
                'vix_spike': 82.7,
                'trigger': 'Pandemic Liquidity Crisis'
            },
            '2022 Bear Market': {
                'date': 'Oct 12, 2022',
                'drop': -25.4,
                'vix_spike': 35.0,  # Max VIX
                'trigger': 'Inflation/Rate Shock'
            },
            '2025 Tariff Crash': {
                'date': 'Apr 7, 2025',
                'drop': -23.0,
                'vix_spike': 60.0,
                'trigger': 'Liberation Day Tariffs'
            }
        }

    def create_comparison_chart(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create crash comparison visualization.

        Shows how indicators behaved before each crash.
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')

        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Simulated indicator patterns for each crash
        # These represent the typical patterns observed
        crashes = list(self.crash_events.keys())

        for idx, crash_name in enumerate(crashes):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            self._plot_crash_indicators(ax, crash_name, self.crash_events[crash_name])

        fig.suptitle('Early Warning Indicators: Historical Crash Comparison',
                     fontsize=16, fontweight='bold', color='white', y=0.98)

        # see create_dashboard; also keeps the outward third y-axis of the left
        # panels from overlapping the right panels
        gs.tight_layout(fig, rect=[0, 0.02, 1, 0.95])
        return fig

    def _plot_crash_indicators(self, ax, crash_name, crash_info):
        """Plot indicator pattern for a specific crash."""
        ax.set_facecolor('#16213e')

        # Generate representative patterns
        days = np.arange(-60, 31)  # 60 days before to 30 days after

        # Different patterns for different crashes
        if '2018' in crash_name:
            # VIX term structure was key
            gex = np.where(days < 0, 2 + 0.02 * days, -5 + 0.1 * days)
            tdex = np.where(days < 0, 6 + 0.05 * np.abs(days), 25 - 0.3 * days)
            vix_ratio = np.where(days < 0, 0.92 + 0.001 * days, 1.3 - 0.01 * days)
        elif '2020' in crash_name:
            # Gamma flip was dominant
            gex = np.where(days < 0, 3 - 0.08 * np.abs(days), -10 + 0.2 * days)
            tdex = np.where(days < 0, 8 + 0.1 * np.abs(days), 30 - 0.4 * days)
            vix_ratio = np.where(days < 0, 0.95 - 0.002 * np.abs(days), 1.4 - 0.015 * days)
        elif '2022' in crash_name:
            # Slow bleed - less dramatic signals
            gex = np.where(days < 0, 1 - 0.02 * np.abs(days), -2 + 0.05 * days)
            tdex = np.where(days < 0, 10 + 0.03 * np.abs(days), 18 - 0.15 * days)
            vix_ratio = np.where(days < 0, 0.97 - 0.0005 * np.abs(days), 1.05 - 0.002 * days)
        else:  # 2025
            # Sharp gamma flip + TDEX divergence
            gex = np.where(days < 0, 4 - 0.1 * np.abs(days), -8 + 0.15 * days)
            tdex = np.where(days < -20, 6 + 0.2 * np.abs(days + 20),
                           np.where(days < 0, 14 + 0.3 * np.abs(days), 28 - 0.35 * days))
            vix_ratio = np.where(days < 0, 0.94 - 0.001 * np.abs(days), 1.25 - 0.01 * days)

        # Add noise. The seed must be stable across processes (str hash() is salted
        # per interpreter) and must not reset the caller's global NumPy RNG.
        rng = np.random.default_rng(zlib.crc32(crash_name.encode('utf-8')))
        gex += rng.normal(0, 0.5, len(days))
        tdex += rng.normal(0, 1, len(days))
        vix_ratio += rng.normal(0, 0.02, len(days))

        # Normalize for plotting
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))

        l1, = ax.plot(days, gex, color='#e74c3c', linewidth=2, label='GEX')
        l2, = ax2.plot(days, tdex, color='#9b59b6', linewidth=2, label='TDEX')
        l3, = ax3.plot(days, vix_ratio, color='#3498db', linewidth=2, label='VIX Ratio')

        # Crash day marker
        ax.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvspan(0, 10, alpha=0.2, color='red')

        ax.set_title(f'{crash_name}\n{crash_info["date"]} | Drop: {crash_info["drop"]}%',
                     fontsize=11, fontweight='bold', color='white')
        ax.set_xlabel('Days from Crash', color='white')
        ax.set_ylabel('GEX (B$)', color='#e74c3c')
        ax2.set_ylabel('TDEX', color='#9b59b6')
        ax3.set_ylabel('VIX M1/M2', color='#3498db')

        ax.tick_params(colors='white')
        ax2.tick_params(colors='white')
        ax3.tick_params(colors='white')

        ax.legend([l1, l2, l3], ['GEX', 'TDEX', 'VIX Ratio'],
                  loc='upper left', fontsize=8, facecolor='#1a1a2e',
                  edgecolor='white', labelcolor='white')

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)


class EarlyWarning3DVisualization:
    """
    3D visualization of early warning phase space.
    """

    def __init__(self):
        """Initialize 3D visualization."""
        pass

    def create_3d_phase_space(self,
                               gex_scores: np.ndarray,
                               tdex_scores: np.ndarray,
                               composite_scores: np.ndarray,
                               figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create 3D phase space visualization.

        Coordinates:
        - X: GEX-based risk (market mechanics)
        - Y: TDEX-based risk (tail pricing)
        - Z: Composite score (overall warning level)
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#16213e')

        # Normalize scores
        X = gex_scores / 100
        Y = tdex_scores / 100
        Z = composite_scores / 100

        # Color by composite score
        colors = plt.cm.RdYlGn_r(Z)

        # Plot trajectory
        ax.plot(X, Y, Z, color='white', alpha=0.3, linewidth=0.5)

        # Scatter with color gradient
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='RdYlGn_r', s=20, alpha=0.7)

        # Mark current position
        ax.scatter([X[-1]], [Y[-1]], [Z[-1]], color='white', s=200,
                   marker='*', edgecolors='black', linewidths=2)

        # Add danger zone plane
        xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        zz = np.ones_like(xx) * 0.75
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='red')

        ax.set_xlabel('GEX Risk', color='white', fontsize=10)
        ax.set_ylabel('TDEX Risk', color='white', fontsize=10)
        ax.set_zlabel('Composite Score', color='white', fontsize=10)

        ax.set_title('Early Warning 3D Phase Space', fontsize=14,
                     fontweight='bold', color='white', pad=20)

        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Risk Level', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        return fig


def create_indicator_summary_table(indicators: Dict[str, np.ndarray]) -> str:
    """
    Create text summary table of current indicator values.

    Returns formatted string for terminal/report output.
    """
    w, th = _scoring_system()
    latest = {
        'GEX': indicators['gex'][-1],
        'TDEX': indicators['tdex'][-1],
        'VIX Ratio': indicators['vix_m1'][-1] / indicators['vix_m2'][-1],
        'DIX': indicators['dix'][-1],
        'SMFI': indicators['smfi'][-1],
        'Composite': indicators['composite_score'][-1]
    }

    table = """
╔══════════════════════════════════════════════════════════════════╗
║           EARLY WARNING SYSTEM - CURRENT STATUS                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Indicator          │ Value      │ Weight │ Signal               ║
╠═════════════════════╪════════════╪════════╪══════════════════════╣
"""

    # GEX
    gex_signal = 'NEGATIVE' if latest['GEX'] < 0 else 'POSITIVE'
    table += f"║  1. GEX (B$)        │ {latest['GEX']:>10.2f} │ {w['gex']:>4.0%}   │ {gex_signal:<20} ║\n"

    # TDEX
    # same thresholds as the dashboard panels (TDEX 15/20, DIX 43/40)
    tdex_signal = ('EXTREME' if latest['TDEX'] > 20 else
                   'ELEVATED' if latest['TDEX'] > 15 else 'NORMAL')
    table += f"║  2. TDEX            │ {latest['TDEX']:>10.1f} │ {w['tdex']:>4.0%}   │ {tdex_signal:<20} ║\n"

    # VIX Ratio
    vix_signal = 'INVERTED' if latest['VIX Ratio'] > 1 else 'CONTANGO'
    table += f"║  3. VIX M1/M2       │ {latest['VIX Ratio']:>10.3f} │ {w['vix_term']:>4.0%}   │ {vix_signal:<20} ║\n"

    # DIX
    dix_signal = ('DISTRIBUTION' if latest['DIX'] < 40 else
                  'WEAK' if latest['DIX'] < 43 else 'NORMAL')
    table += f"║  4. DIX (%)         │ {latest['DIX']:>10.1f} │ {w['dix']:>4.0%}   │ {dix_signal:<20} ║\n"

    # SMFI
    smfi_signal = 'SELLING' if latest['SMFI'] < 0 else 'BUYING'
    table += f"║  5. SMFI            │ {latest['SMFI']:>10.1f} │ {w['smfi']:>4.0%}   │ {smfi_signal:<20} ║\n"

    table += "╠═════════════════════╪════════════╪════════╪══════════════════════╣\n"

    # Composite (levels of CompositeEarlyWarningSystem.compute_composite)
    comp_signal = {'EXTREME': 'EXTREME RISK', 'HIGH': 'HIGH RISK', 'ELEVATED': 'ELEVATED',
                   'NORMAL': 'NORMAL'}[_composite_level(latest['Composite'], th)]

    table += f"║  COMPOSITE SCORE    │ {latest['Composite']:>10.1f} │  100%  │ {comp_signal:<20} ║\n"
    table += "╚══════════════════════════════════════════════════════════════════╝\n"

    return table
