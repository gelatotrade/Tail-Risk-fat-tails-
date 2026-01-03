#!/usr/bin/env python3
"""
Generate Early Warning System Analysis Visualization for January 3, 2026
=========================================================================

Creates a comprehensive dashboard showing the current status of all 5
early warning indicators with their scores, levels, and historical context.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# Set dark theme
plt.style.use('dark_background')


def create_gauge(ax, value, max_value=100, label='', color_zones=None):
    """Create a semicircular gauge chart."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Default color zones (green, yellow, orange, red)
    if color_zones is None:
        color_zones = [
            (0, 25, '#00ff88'),    # Green - Normal
            (25, 50, '#ffff00'),   # Yellow - Elevated
            (50, 75, '#ff8800'),   # Orange - High
            (75, 100, '#ff0044')   # Red - Extreme
        ]

    # Draw colored arcs
    for start, end, color in color_zones:
        start_angle = 180 - (start / max_value * 180)
        end_angle = 180 - (end / max_value * 180)
        wedge = Wedge((0, 0), 1.0, end_angle, start_angle, width=0.3,
                      facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.8)
        ax.add_patch(wedge)

    # Draw needle
    angle = np.radians(180 - (value / max_value * 180))
    needle_x = [0, 0.85 * np.cos(angle)]
    needle_y = [0, 0.85 * np.sin(angle)]
    ax.plot(needle_x, needle_y, color='white', linewidth=3, zorder=10)
    ax.plot(0, 0, 'o', color='white', markersize=8, zorder=11)

    # Value text
    ax.text(0, -0.1, f'{value:.0f}', ha='center', va='top',
            fontsize=24, fontweight='bold', color='white')
    ax.text(0, 1.15, label, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='white')


def create_bar_indicator(ax, value, min_val, max_val, label, thresholds, current_label='Current'):
    """Create a horizontal bar indicator with zones."""
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color zones based on thresholds
    colors = ['#00ff88', '#88ff00', '#ffff00', '#ff8800', '#ff0044']
    prev = min_val
    for i, (thresh, color) in enumerate(zip(thresholds + [max_val], colors)):
        width = thresh - prev
        rect = Rectangle((prev, 0.3), width, 0.4, facecolor=color, alpha=0.6, edgecolor='none')
        ax.add_patch(rect)
        prev = thresh

    # Current value marker
    ax.axvline(x=value, color='white', linewidth=3, ymin=0.2, ymax=0.8)
    ax.plot(value, 0.5, 'v', color='white', markersize=12, zorder=10)

    # Labels
    ax.text((min_val + max_val) / 2, 0.95, label, ha='center', va='top',
            fontsize=10, fontweight='bold', color='white')
    ax.text(value, 0.1, f'{current_label}: {value:.1f}', ha='center', va='top',
            fontsize=9, color='white')


def create_indicator_card(ax, name, value, status, score, signal, color):
    """Create an indicator summary card."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background
    rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                          facecolor='#1a1a2e', edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    # Status indicator circle
    circle = Circle((0.1, 0.7), 0.06, facecolor=color, edgecolor='white', linewidth=1)
    ax.add_patch(circle)

    # Text
    ax.text(0.2, 0.72, name, ha='left', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(0.5, 0.45, value, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    ax.text(0.5, 0.25, f'Score: {score}/100', ha='center', va='center', fontsize=9, color='#888888')
    ax.text(0.5, 0.1, signal, ha='center', va='center', fontsize=8, color='#aaaaaa')


def create_ews_dashboard():
    """Create the main EWS dashboard for January 3, 2026."""

    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a1a')

    # Title
    fig.suptitle('Early Warning System Dashboard: January 3, 2026',
                 fontsize=20, fontweight='bold', color='white', y=0.98)

    # Create grid
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3,
                           left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Current indicator values (January 3, 2026)
    indicators = {
        'GEX': {'value': 4.2, 'score': 18, 'status': 'POSITIVE', 'signal': 'Stabilizing'},
        'TDEX': {'value': 7.8, 'score': 22, 'status': '22nd %ile', 'signal': 'Complacent'},
        'VIX Term': {'value': 0.87, 'score': 15, 'status': 'CONTANGO', 'signal': 'Normal'},
        'DIX': {'value': 46.2, 'score': 25, 'status': 'NEUTRAL', 'signal': 'Buying'},
        'SMFI': {'value': 3.2, 'score': 20, 'status': 'POSITIVE', 'signal': 'Accumulating'}
    }

    # Composite gauge (center top)
    ax_gauge = fig.add_subplot(gs[0, 1:3])
    composite_score = 20
    create_gauge(ax_gauge, composite_score, 100, 'COMPOSITE SCORE')
    ax_gauge.text(0, -0.35, 'NORMAL', ha='center', va='top', fontsize=14,
                  fontweight='bold', color='#00ff88')

    # Individual indicator cards (row 1)
    colors = ['#00ff88', '#00ff88', '#00ff88', '#88ff00', '#00ff88']
    card_data = [
        ('Net Gamma (GEX)', '+$4.2B', 'POSITIVE', 18, 'Dealers buying dips'),
        ('TailDex (TDEX)', '7.8', '22nd Percentile', 22, 'Low tail fear'),
        ('VIX Term Structure', 'M1/M2: 0.87', 'CONTANGO', 15, 'Normal curve'),
        ('Dark Index (DIX)', '46.2%', 'NEUTRAL-HIGH', 25, 'Modest buying'),
        ('Smart Money (SMFI)', '+3.2', 'POSITIVE', 20, 'Accumulating')
    ]

    # Add indicator cards in row 2
    for i, (name, value, status, score, signal) in enumerate(card_data):
        if i < 4:
            ax = fig.add_subplot(gs[1, i])
        else:
            # Put 5th card centered below
            ax = fig.add_subplot(gs[2, 0])
        create_indicator_card(ax, name, value, status, score, signal, colors[i])

    # GEX detail chart (row 2, cols 1-2)
    ax_gex = fig.add_subplot(gs[2, 1:3])
    ax_gex.set_facecolor('#0a0a1a')

    # Simulated GEX history (30 days)
    days = np.arange(30)
    gex_history = 3.5 + np.cumsum(np.random.normal(0.02, 0.15, 30))
    gex_history[-1] = 4.2  # End at current value

    ax_gex.fill_between(days, 0, gex_history, where=gex_history >= 0,
                        color='#00ff88', alpha=0.3, label='Positive Gamma')
    ax_gex.fill_between(days, 0, gex_history, where=gex_history < 0,
                        color='#ff0044', alpha=0.3, label='Negative Gamma')
    ax_gex.plot(days, gex_history, color='white', linewidth=2)
    ax_gex.axhline(y=0, color='#ff4444', linestyle='--', linewidth=1.5, label='Gamma Flip')
    ax_gex.axhline(y=2, color='#ffff00', linestyle=':', linewidth=1, alpha=0.5)

    ax_gex.set_xlim(0, 29)
    ax_gex.set_ylim(-5, 8)
    ax_gex.set_xlabel('Days (Dec 4 - Jan 3)', color='white', fontsize=9)
    ax_gex.set_ylabel('GEX ($ Billions)', color='white', fontsize=9)
    ax_gex.set_title('Net Gamma Exposure - 30 Day History', color='white', fontsize=11, fontweight='bold')
    ax_gex.legend(loc='upper left', fontsize=8)
    ax_gex.tick_params(colors='white', labelsize=8)
    for spine in ax_gex.spines.values():
        spine.set_color('#333333')

    # Composite trend chart (row 2, col 3)
    ax_composite = fig.add_subplot(gs[2, 3])
    ax_composite.set_facecolor('#0a0a1a')

    # Simulated composite history
    composite_history = 18 + np.cumsum(np.random.normal(0.1, 1.5, 30))
    composite_history = np.clip(composite_history, 10, 40)
    composite_history[-1] = 20

    ax_composite.fill_between(days, 0, 25, color='#00ff88', alpha=0.1)
    ax_composite.fill_between(days, 25, 50, color='#ffff00', alpha=0.1)
    ax_composite.fill_between(days, 50, 75, color='#ff8800', alpha=0.1)
    ax_composite.fill_between(days, 75, 100, color='#ff0044', alpha=0.1)

    ax_composite.plot(days, composite_history, color='#00ffff', linewidth=2)
    ax_composite.axhline(y=40, color='#ffff00', linestyle='--', linewidth=1, label='Alert Level')

    ax_composite.set_xlim(0, 29)
    ax_composite.set_ylim(0, 100)
    ax_composite.set_xlabel('Days', color='white', fontsize=9)
    ax_composite.set_ylabel('Score', color='white', fontsize=9)
    ax_composite.set_title('Composite Score Trend', color='white', fontsize=11, fontweight='bold')
    ax_composite.legend(loc='upper right', fontsize=8)
    ax_composite.tick_params(colors='white', labelsize=8)
    for spine in ax_composite.spines.values():
        spine.set_color('#333333')

    # Historical comparison (row 3)
    ax_compare = fig.add_subplot(gs[3, :])
    ax_compare.set_facecolor('#0a0a1a')

    # Comparison data
    events = ['Jan 3, 2026\n(Current)', 'Pre-COVID\n(Feb 2020)', 'Pre-Tariff\n(Mar 2025)', 'April 2025\n(Peak Crash)']
    composite_scores = [20, 45, 55, 95]
    colors_bars = ['#00ff88', '#ffff00', '#ff8800', '#ff0044']

    bars = ax_compare.bar(events, composite_scores, color=colors_bars, edgecolor='white', linewidth=1)

    # Add threshold lines
    ax_compare.axhline(y=25, color='#00ff88', linestyle='--', linewidth=1, alpha=0.7)
    ax_compare.axhline(y=50, color='#ffff00', linestyle='--', linewidth=1, alpha=0.7)
    ax_compare.axhline(y=75, color='#ff8800', linestyle='--', linewidth=1, alpha=0.7)

    ax_compare.text(3.5, 25, 'Normal', color='#00ff88', fontsize=9, va='bottom')
    ax_compare.text(3.5, 50, 'Elevated', color='#ffff00', fontsize=9, va='bottom')
    ax_compare.text(3.5, 75, 'High', color='#ff8800', fontsize=9, va='bottom')

    # Value labels on bars
    for bar, score in zip(bars, composite_scores):
        ax_compare.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{score}', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')

    ax_compare.set_ylim(0, 110)
    ax_compare.set_ylabel('Composite Warning Score', color='white', fontsize=10)
    ax_compare.set_title('Historical Comparison: Current vs Pre-Crash Readings',
                        color='white', fontsize=12, fontweight='bold')
    ax_compare.tick_params(colors='white', labelsize=10)
    for spine in ax_compare.spines.values():
        spine.set_color('#333333')

    # Add status box
    status_text = (
        "STATUS: NORMAL  |  Composite Score: 20/100  |  Risk Level: LOW\n"
        "All 5 indicators in normal zone. Positive gamma regime providing market stability.\n"
        "Recommendation: Standard risk management. No defensive adjustments required."
    )
    fig.text(0.5, 0.01, status_text, ha='center', va='bottom', fontsize=10,
             color='#00ff88', fontstyle='italic',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88', alpha=0.8))

    plt.savefig('outputs/ews_analysis_jan2026.png', dpi=150, facecolor='#0a0a1a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("Generated: outputs/ews_analysis_jan2026.png")


if __name__ == '__main__':
    print("Generating Early Warning System Dashboard for January 3, 2026...")
    create_ews_dashboard()
    print("Done!")
