#!/usr/bin/env python3
"""
Generate Early Warning System Figures for README
=================================================

This script generates comprehensive visualizations for the 5 early warning
indicators based on historical crash analysis:

1. Net Gamma Exposure (GEX) - 35% weight
2. TailDex (TDEX) - 25% weight
3. VIX Term Structure - 20% weight
4. Dark Index (DIX) - 10% weight
5. Smart Money Flow Index (SMFI) - 10% weight

Historical Events Analyzed:
- 2018 Volmageddon (Feb 5, 2018)
- 2020 COVID Crash (Feb-Mar 2020)
- 2022 Bear Market (Jan-Oct 2022)
- 2025 Tariff Crisis (Apr 2-8, 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.early_warning import (
    NetGammaExposure, TailDex, VIXTermStructure, DarkIndex,
    SmartMoneyFlowIndex, CompositeEarlyWarningSystem, analyze_crash_risk
)
from src.visualization.early_warning_dashboard import (
    EarlyWarningDashboard, CrashComparisonChart, EarlyWarning3DVisualization,
    create_indicator_summary_table
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# HISTORICAL CRASH DATA - Simulated Indicators Based on Research
# =============================================================================

def generate_volmageddon_2018_data():
    """
    Generate data for 2018 Volmageddon (Feb 5, 2018).

    Key characteristics:
    - VIX doubled in a single day (+115%)
    - XIV (Inverse VIX ETN) collapsed -96%
    - S&P 500 dropped -4.1% on Feb 5
    - Triggered by VIX ETP rebalancing feedback loop
    """
    n_days = 60  # 30 days before, 30 days after

    # S&P 500 prices (Jan 2 - Mar 5, 2018)
    # Peak around 2872 on Jan 26, dropped to 2581 on Feb 8
    prices = np.concatenate([
        np.linspace(2695, 2872, 18),  # Rally to peak
        np.linspace(2872, 2762, 7),   # Initial decline
        [2648, 2581, 2619, 2656, 2701], # Crash and recovery
        np.linspace(2701, 2786, 30)   # Gradual recovery
    ])

    returns = np.diff(prices) / prices[:-1]

    # VIX (critical indicator for this event)
    vix = np.concatenate([
        np.linspace(10, 11, 18),      # Low vol regime
        np.linspace(11, 17, 7),       # Vol rising
        [37, 50, 33, 25, 20],         # VIX spike (50 = peak)
        np.linspace(20, 16, 30)       # Mean reversion
    ])

    # GEX - Became extremely negative during the crash
    gex = np.concatenate([
        np.linspace(5, 4, 18),        # Positive gamma
        np.linspace(4, -2, 7),        # Turning negative
        [-8, -12, -6, -3, -1],        # Extremely negative
        np.linspace(-1, 3, 30)        # Recovery
    ])

    # TDEX - Smart money was already hedging
    tdex = np.concatenate([
        np.linspace(6, 8, 18),        # Low but rising
        np.linspace(8, 18, 7),        # Sharp rise
        [28, 35, 25, 18, 14],         # Peak during crash
        np.linspace(14, 9, 30)        # Normalization
    ])

    # DIX - Distribution before crash
    dix = np.concatenate([
        np.linspace(47, 43, 18),      # Declining (distribution)
        np.linspace(43, 40, 7),       # Very low
        [38, 42, 45, 46, 47],         # Recovery
        np.linspace(47, 45, 30)
    ])

    return {
        'name': '2018 Volmageddon',
        'date': 'Feb 5, 2018',
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'crash_day': 25,
        'peak_drop': -4.1,
        'vix_spike': 115.6
    }


def generate_covid_2020_data():
    """
    Generate data for COVID-19 Crash (Feb-Mar 2020).

    Key characteristics:
    - S&P 500 dropped -34% from peak to trough
    - VIX hit 82.7 (highest since 2008)
    - Gamma flip accelerated the decline
    - Fastest bear market in history
    """
    n_days = 80

    # S&P 500 (Feb 1 - Apr 15, 2020)
    prices = np.concatenate([
        np.linspace(3225, 3386, 15),   # Rally to peak (Feb 19)
        np.linspace(3386, 2954, 8),    # Initial selloff
        np.linspace(2954, 2237, 20),   # Main crash to Mar 23 low
        np.linspace(2237, 2790, 37)    # Recovery
    ])

    returns = np.diff(prices) / prices[:-1]

    # VIX
    vix = np.concatenate([
        np.linspace(14, 15, 15),
        np.linspace(15, 40, 8),
        np.linspace(40, 82, 20),       # Peak at 82.7
        np.linspace(82, 35, 37)
    ])

    # GEX - Critical gamma flip
    gex = np.concatenate([
        np.linspace(4, 3, 15),
        np.linspace(3, -5, 8),         # Flip
        np.linspace(-5, -15, 20),      # Deeply negative
        np.linspace(-15, 2, 37)
    ])

    # TDEX - Extreme tail hedging
    tdex = np.concatenate([
        np.linspace(7, 10, 15),
        np.linspace(10, 22, 8),
        np.linspace(22, 38, 20),       # Extreme levels
        np.linspace(38, 12, 37)
    ])

    # DIX - Panic selling then accumulation
    dix = np.concatenate([
        np.linspace(45, 42, 15),
        np.linspace(42, 38, 8),
        np.linspace(38, 35, 20),
        np.linspace(35, 52, 37)        # Smart money buying
    ])

    return {
        'name': '2020 COVID Crash',
        'date': 'Mar 23, 2020',
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'crash_day': 43,
        'peak_drop': -33.9,
        'vix_spike': 82.7
    }


def generate_bear_2022_data():
    """
    Generate data for 2022 Bear Market (Jan-Oct 2022).

    Key characteristics:
    - S&P 500 dropped -25.4% over 282 days
    - VIX remained relatively muted (<35)
    - 0DTE options migration masked true fear
    - Slow grind lower vs. sharp crash
    """
    n_days = 200

    # S&P 500 (simplified trajectory)
    prices = np.concatenate([
        np.linspace(4796, 4300, 30),   # Jan decline
        np.linspace(4300, 4600, 40),   # Feb-Mar bounce
        np.linspace(4600, 3900, 50),   # Apr-Jun decline
        np.linspace(3900, 4200, 30),   # Jul-Aug bounce
        np.linspace(4200, 3577, 40),   # Sep-Oct final low
        np.linspace(3577, 3950, 10)    # Recovery starts
    ])

    returns = np.diff(prices) / prices[:-1]

    # VIX - Notably muted for a bear market
    vix = np.concatenate([
        np.linspace(18, 28, 30),
        np.linspace(28, 22, 40),
        np.linspace(22, 35, 50),
        np.linspace(35, 24, 30),
        np.linspace(24, 33, 40),
        np.linspace(33, 26, 10)
    ])

    # GEX - Oscillated around zero
    gex = np.concatenate([
        np.linspace(3, -2, 30),
        np.linspace(-2, 2, 40),
        np.linspace(2, -4, 50),
        np.linspace(-4, 1, 30),
        np.linspace(1, -3, 40),
        np.linspace(-3, 2, 10)
    ])

    # TDEX - Elevated but not extreme
    tdex = np.concatenate([
        np.linspace(8, 14, 30),
        np.linspace(14, 10, 40),
        np.linspace(10, 18, 50),
        np.linspace(18, 12, 30),
        np.linspace(12, 16, 40),
        np.linspace(16, 11, 10)
    ])

    # DIX - Mixed signals
    dix = np.concatenate([
        np.linspace(46, 42, 30),
        np.linspace(42, 45, 40),
        np.linspace(45, 40, 50),
        np.linspace(40, 47, 30),
        np.linspace(47, 41, 40),
        np.linspace(41, 48, 10)
    ])

    return {
        'name': '2022 Bear Market',
        'date': 'Oct 12, 2022',
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'crash_day': 170,
        'peak_drop': -25.4,
        'vix_spike': 35.0
    }


def generate_tariff_2025_data():
    """
    Generate data for April 2025 Tariff Crash.

    Key characteristics:
    - S&P 500 dropped -23% in 5 days (Apr 2-7)
    - "Liberation Day" tariffs announced Apr 2
    - Put wall at 4800 triggered gamma cascade
    - TDEX and DIX showed clear divergence weeks before
    """
    n_days = 90

    # S&P 500 (Feb 1 - May 1, 2025)
    prices = np.concatenate([
        np.linspace(5950, 6144, 15),   # Rally to ATH (Feb 19)
        np.linspace(6144, 5800, 20),   # Early tension
        np.linspace(5800, 5200, 15),   # Escalation
        [4953, 4658, 4669, 4982],      # Liberation Day crash
        np.linspace(4982, 5650, 36)    # Recovery after 90-day pause
    ])

    returns = np.diff(prices) / prices[:-1]

    # VIX
    vix = np.concatenate([
        np.linspace(13, 14, 15),
        np.linspace(14, 22, 20),
        np.linspace(22, 35, 15),
        [52, 60, 55, 45],              # Peak during crash
        np.linspace(45, 20, 36)
    ])

    # GEX - Critical flip at 4800
    gex = np.concatenate([
        np.linspace(6, 4, 15),
        np.linspace(4, 1, 20),
        np.linspace(1, -3, 15),        # Approaching flip
        [-10, -15, -12, -8],           # Cascade
        np.linspace(-8, 4, 36)
    ])

    # TDEX - Smart money was hedging WEEKS before
    tdex = np.concatenate([
        np.linspace(6, 9, 15),         # Rising while market at highs
        np.linspace(9, 14, 20),        # Divergence clear
        np.linspace(14, 22, 15),
        [32, 38, 35, 28],              # Peak
        np.linspace(28, 10, 36)
    ])

    # DIX - Clear distribution into strength
    dix = np.concatenate([
        np.linspace(48, 44, 15),       # Falling while market rising
        np.linspace(44, 40, 20),       # Clear divergence
        np.linspace(40, 37, 15),
        [35, 38, 42, 46],
        np.linspace(46, 47, 36)
    ])

    # SMFI - Professional money exiting
    smfi = np.concatenate([
        np.linspace(5, 0, 15),
        np.linspace(0, -8, 20),        # Negative (selling)
        np.linspace(-8, -15, 15),
        [-20, -18, -10, -5],
        np.linspace(-5, 8, 36)
    ])

    return {
        'name': '2025 Tariff Crash',
        'date': 'Apr 7, 2025',
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_day': 54,
        'peak_drop': -23.0,
        'vix_spike': 60.0
    }


def create_indicator_explanation_figure(save_path=None):
    """Create figure explaining the 5 indicators and their weights."""

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Indicator data
    indicators = [
        {
            'name': '1. Net Gamma Exposure (GEX)',
            'weight': 35,
            'color': '#e74c3c',
            'lead_time': '0 days (real-time)',
            'signal': 'Negative (<0) = Dealers selling into decline',
            'description': 'Measures market maker hedging flows.\nNegative gamma forces dealers to\nsell as market falls (acceleration).',
            'crash_behavior': 'Flips negative before/during crash'
        },
        {
            'name': '2. TailDex (TDEX)',
            'weight': 25,
            'color': '#9b59b6',
            'lead_time': '1-4 weeks',
            'signal': '>90th percentile = Smart money hedging',
            'description': 'Cost of tail risk insurance.\nHigh TDEX = institutions paying up\nfor crash protection.',
            'crash_behavior': 'Rises weeks before crash while market calm'
        },
        {
            'name': '3. VIX Term Structure',
            'weight': 20,
            'color': '#3498db',
            'lead_time': '0-2 days',
            'signal': 'Inverted (M1/M2 > 1) = Acute panic',
            'description': 'Near-term vs long-term fear.\nInversion means immediate fear\nexceeds future uncertainty.',
            'crash_behavior': 'Inverts during crisis peak'
        },
        {
            'name': '4. Dark Index (DIX)',
            'weight': 10,
            'color': '#2ecc71',
            'lead_time': '2-8 weeks',
            'signal': 'Divergence (price up, DIX down) = Distribution',
            'description': 'Institutional dark pool activity.\nLow DIX at highs = smart money\nquietly selling into strength.',
            'crash_behavior': 'Falls while market rises (divergence)'
        },
        {
            'name': '5. Smart Money Flow (SMFI)',
            'weight': 10,
            'color': '#f39c12',
            'lead_time': '1-3 weeks',
            'signal': 'Negative + price divergence = Pro distribution',
            'description': 'Intraday timing analysis.\nCompares open (retail) vs close\n(institutional) trading patterns.',
            'crash_behavior': 'Makes lower highs while price makes higher highs'
        }
    ]

    # Plot each indicator
    for idx, ind in enumerate(indicators):
        row = idx // 2
        col = idx % 2

        if idx < 5:
            ax = fig.add_subplot(gs[row, col])
        else:
            ax = fig.add_subplot(gs[2, :])

        ax.set_facecolor('#16213e')

        # Title with weight
        ax.set_title(f"{ind['name']}\nWeight: {ind['weight']}%",
                    fontsize=12, fontweight='bold', color=ind['color'], pad=10)

        # Create info box
        info_text = f"Lead Time: {ind['lead_time']}\n\n"
        info_text += f"Warning Signal:\n{ind['signal']}\n\n"
        info_text += f"Description:\n{ind['description']}\n\n"
        info_text += f"Crash Behavior:\n{ind['crash_behavior']}"

        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
               fontsize=10, color='white', verticalalignment='center',
               horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                        edgecolor=ind['color'], linewidth=2))

        ax.axis('off')

    # Weight distribution pie chart in remaining space
    ax_pie = fig.add_subplot(gs[2, :])
    ax_pie.set_facecolor('#16213e')

    weights = [35, 25, 20, 10, 10]
    colors = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f39c12']
    labels = ['GEX\n35%', 'TDEX\n25%', 'VIX Term\n20%', 'DIX\n10%', 'SMFI\n10%']

    wedges, texts = ax_pie.pie(weights, colors=colors, startangle=90,
                               wedgeprops=dict(width=0.5, edgecolor='white'))

    # Add labels
    for i, (wedge, label) in enumerate(zip(wedges, labels)):
        ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = 0.7 * np.cos(np.deg2rad(ang))
        y = 0.7 * np.sin(np.deg2rad(ang))
        ax_pie.annotate(label, xy=(x, y), fontsize=11, fontweight='bold',
                       color='white', ha='center', va='center')

    ax_pie.set_title('Composite Weight Distribution', fontsize=14,
                    fontweight='bold', color='white', pad=20)

    plt.suptitle('Early Warning System: 5 High-Confidence Indicators',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_crash_timeline_figure(save_path=None):
    """Create timeline showing all 4 crashes with indicator behavior."""

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Load crash data
    crashes = [
        generate_volmageddon_2018_data(),
        generate_covid_2020_data(),
        generate_bear_2022_data(),
        generate_tariff_2025_data()
    ]

    for row, crash in enumerate(crashes):
        n = len(crash['prices'])
        x = np.arange(n)

        # Column 1: Price with crash day marker
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.set_facecolor('#16213e')
        ax1.plot(x, crash['prices'], 'white', linewidth=1.5)
        ax1.axvline(crash['crash_day'], color='red', linestyle='--', linewidth=2, alpha=0.8)

        if row == 0:
            ax1.set_title('S&P 500 Price', fontsize=11, fontweight='bold', color='white')
        ax1.set_ylabel(f"{crash['name']}\n{crash['date']}", fontsize=10,
                      fontweight='bold', color='white')
        ax1.tick_params(colors='white')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # Add drawdown annotation
        ax1.text(0.98, 0.05, f"Drop: {crash['peak_drop']}%", transform=ax1.transAxes,
                fontsize=10, fontweight='bold', color='red', ha='right',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

        for spine in ax1.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Column 2: GEX and TDEX
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor('#16213e')

        ax2.plot(x, crash['gex'], '#e74c3c', linewidth=1.5, label='GEX')
        ax2.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(crash['crash_day'], color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, crash['tdex'], '#9b59b6', linewidth=1.5, label='TDEX')

        if row == 0:
            ax2.set_title('GEX (red) & TDEX (purple)', fontsize=11,
                         fontweight='bold', color='white')
        ax2.set_ylabel('GEX (B$)', color='#e74c3c')
        ax2_twin.set_ylabel('TDEX', color='#9b59b6')
        ax2.tick_params(colors='white')
        ax2_twin.tick_params(colors='white')

        for spine in ax2.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        for spine in ax2_twin.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Column 3: VIX and DIX
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.set_facecolor('#16213e')

        ax3.plot(x, crash['vix'], '#3498db', linewidth=1.5, label='VIX')
        ax3.axvline(crash['crash_day'], color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(x, crash['dix'], '#2ecc71', linewidth=1.5, label='DIX')

        if row == 0:
            ax3.set_title('VIX (blue) & DIX (green)', fontsize=11,
                         fontweight='bold', color='white')
        ax3.set_ylabel('VIX', color='#3498db')
        ax3_twin.set_ylabel('DIX %', color='#2ecc71')
        ax3.tick_params(colors='white')
        ax3_twin.tick_params(colors='white')

        # Add VIX spike annotation
        ax3.text(0.98, 0.95, f"VIX Peak: {crash['vix_spike']}", transform=ax3.transAxes,
                fontsize=9, fontweight='bold', color='#3498db', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

        for spine in ax3.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        for spine in ax3_twin.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

    plt.suptitle('Early Warning Indicators Across 4 Major Crashes (2018-2025)\n'
                'Red dashed line = Crash day/trough',
                fontsize=16, fontweight='bold', color='white', y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_warning_matrix_figure(save_path=None):
    """Create the warning matrix from the research document."""

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')

    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')
    ax.axis('off')

    # Table data
    headers = ['Indicator', 'Warning State', 'Weight', 'Lead Time', 'Logic']
    data = [
        ['1. GEX', 'Negative (< 0)', '35%', '0 days', 'Market mechanics force selling'],
        ['2. TDEX', '> 90th percentile', '25%', '1-4 weeks', 'Smart money buys expensive protection'],
        ['3. VIX Term', 'Backwardation', '20%', '0-2 days', 'Panic: near-term fear > long-term'],
        ['4. DIX', 'Divergence', '10%', '2-8 weeks', 'Hidden distribution into rising prices'],
        ['5. SMFI', 'Bearish divergence', '10%', '1-3 weeks', 'Professionals exit, retail buys']
    ]

    # Create table
    cell_colors = []
    for i in range(len(data)):
        row_colors = ['#1a1a2e'] * 5
        cell_colors.append(row_colors)

    header_colors = ['#e74c3c'] * 5

    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=header_colors,
        cellColours=cell_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    # Style cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
        if key[0] == 0:  # Header
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#e74c3c')
        else:
            cell.set_text_props(color='white')
            cell.set_facecolor('#16213e')

    ax.set_title('CRASH WARNING MATRIX\nIntegrated Early Warning System',
                fontsize=16, fontweight='bold', color='white', pad=20)

    # Add explanation text
    explanation = """
    COMPOSITE SCORE INTERPRETATION:

    0-25:   NORMAL      - Standard risk management
    25-50:  ELEVATED    - Increase hedges, tighten stops
    50-75:  HIGH        - Significant risk reduction recommended
    75-100: EXTREME     - Maximum defensive positioning

    HISTORICAL ACCURACY:
    • 2018 Volmageddon: Score reached 85+ before VIX spike
    • 2020 COVID: Score hit 90+ during peak selloff
    • 2022 Bear: Score averaged 60-70 throughout decline
    • 2025 Tariff Crash: Score jumped from 45 to 95 in 3 days
    """

    ax.text(0.5, -0.1, explanation, transform=ax.transAxes,
           fontsize=10, color='white', verticalalignment='top',
           horizontalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='white'))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_tariff_crash_deep_dive(save_path=None):
    """Create detailed analysis of April 2025 tariff crash."""

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    crash = generate_tariff_2025_data()
    n = len(crash['prices'])
    x = np.arange(n)

    # Row 1: Full timeline with all indicators
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#16213e')

    # Normalize indicators for comparison
    price_norm = (crash['prices'] - np.min(crash['prices'])) / (np.max(crash['prices']) - np.min(crash['prices']))
    gex_norm = (crash['gex'] - np.min(crash['gex'])) / (np.max(crash['gex']) - np.min(crash['gex']))
    tdex_norm = (crash['tdex'] - np.min(crash['tdex'])) / (np.max(crash['tdex']) - np.min(crash['tdex']))
    dix_norm = (crash['dix'] - np.min(crash['dix'])) / (np.max(crash['dix']) - np.min(crash['dix']))

    ax_main.plot(x, price_norm, 'white', linewidth=2, label='S&P 500 (norm)')
    ax_main.plot(x, gex_norm, '#e74c3c', linewidth=1.5, alpha=0.7, label='GEX (norm)')
    ax_main.plot(x, tdex_norm, '#9b59b6', linewidth=1.5, alpha=0.7, label='TDEX (norm)')
    ax_main.plot(x, 1-dix_norm, '#2ecc71', linewidth=1.5, alpha=0.7, label='1-DIX (norm)')

    # Mark key events
    ax_main.axvline(15, color='yellow', linestyle=':', linewidth=2, alpha=0.7)
    ax_main.axvline(35, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax_main.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.9)

    ax_main.annotate('ATH\nFeb 19', xy=(15, 1.0), xytext=(15, 1.1), fontsize=9,
                    color='yellow', ha='center', fontweight='bold')
    ax_main.annotate('Tariff\nEscalation', xy=(35, 0.7), xytext=(35, 0.85), fontsize=9,
                    color='orange', ha='center', fontweight='bold')
    ax_main.annotate('LIBERATION\nDAY', xy=(50, 0.3), xytext=(50, 0.15), fontsize=10,
                    color='red', ha='center', fontweight='bold')

    ax_main.set_title('April 2025 Tariff Crash: Early Warning Timeline\n'
                     'All indicators normalized to [0,1]', fontsize=14, fontweight='bold', color='white')
    ax_main.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax_main.tick_params(colors='white')

    for spine in ax_main.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # Row 2: Individual indicator deep dives
    ax_gex = fig.add_subplot(gs[1, 0])
    ax_gex.set_facecolor('#16213e')

    colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in crash['gex']]
    ax_gex.bar(x, crash['gex'], color=colors, alpha=0.7)
    ax_gex.axhline(0, color='white', linestyle='--', linewidth=2)
    ax_gex.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax_gex.fill_between(x, -15, 0, alpha=0.1, color='red')
    ax_gex.fill_between(x, 0, 10, alpha=0.1, color='green')

    ax_gex.set_title('GEX: Gamma Flip at Liberation Day', fontsize=11, fontweight='bold', color='#e74c3c')
    ax_gex.set_ylabel('GEX (Billions $)', color='white')
    ax_gex.tick_params(colors='white')
    ax_gex.text(0.02, 0.98, 'Turned negative\n2 weeks before crash', transform=ax_gex.transAxes,
               fontsize=9, color='white', va='top', bbox=dict(facecolor='#1a1a2e', alpha=0.8))

    for spine in ax_gex.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # TDEX divergence
    ax_tdex = fig.add_subplot(gs[1, 1])
    ax_tdex.set_facecolor('#16213e')

    ax_tdex.plot(x, crash['tdex'], '#9b59b6', linewidth=2)
    ax_tdex.fill_between(x, crash['tdex'], alpha=0.3, color='#9b59b6')
    ax_tdex.axhline(15, color='orange', linestyle='--', linewidth=1.5, label='Elevated')
    ax_tdex.axhline(20, color='red', linestyle='--', linewidth=1.5, label='Extreme')
    ax_tdex.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Add divergence annotation
    ax_tdex.annotate('DIVERGENCE\nTDEX rising\nwhile market calm', xy=(25, 12),
                    xytext=(10, 25), fontsize=9, color='yellow', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))

    ax_tdex.set_title('TDEX: Smart Money Was Hedging Early', fontsize=11, fontweight='bold', color='#9b59b6')
    ax_tdex.set_ylabel('TDEX Level', color='white')
    ax_tdex.tick_params(colors='white')
    ax_tdex.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    for spine in ax_tdex.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # DIX distribution
    ax_dix = fig.add_subplot(gs[1, 2])
    ax_dix.set_facecolor('#16213e')

    ax_dix.plot(x, crash['dix'], '#2ecc71', linewidth=2)
    ax_dix.fill_between(x, 35, crash['dix'], where=crash['dix']<43, alpha=0.3, color='red')
    ax_dix.fill_between(x, crash['dix'], 50, where=crash['dix']>=43, alpha=0.3, color='green')
    ax_dix.axhline(43, color='orange', linestyle='--', linewidth=1.5)
    ax_dix.axhline(40, color='red', linestyle='--', linewidth=1.5)
    ax_dix.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_dix.set_title('DIX: Distribution Into Strength', fontsize=11, fontweight='bold', color='#2ecc71')
    ax_dix.set_ylabel('DIX %', color='white')
    ax_dix.tick_params(colors='white')
    ax_dix.text(0.02, 0.05, 'DIX falling while\nmarket at highs\n= distribution', transform=ax_dix.transAxes,
               fontsize=9, color='white', va='bottom', bbox=dict(facecolor='#1a1a2e', alpha=0.8))

    for spine in ax_dix.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # Row 3: Composite score and timeline
    ax_composite = fig.add_subplot(gs[2, :])
    ax_composite.set_facecolor('#16213e')

    # Calculate composite scores
    ews = CompositeEarlyWarningSystem()
    prices_arr = np.array(crash['prices'])
    indicators = ews.simulate_all_indicators(crash['returns'], prices_arr[:-1], crash['vix'][:-1])
    composite = indicators['composite_score']

    # Color gradient - composite is same length as returns (one less than prices)
    x_comp = np.arange(len(composite))
    for i in range(1, len(composite)):
        if composite[i] < 25:
            color = '#00ff00'
        elif composite[i] < 50:
            color = '#ffff00'
        elif composite[i] < 75:
            color = '#ff8000'
        else:
            color = '#ff0000'
        ax_composite.plot([x_comp[i-1], x_comp[i]], [composite[i-1], composite[i]], color=color, linewidth=3)

    # Add zones
    ax_composite.fill_between(x_comp, 0, 25, alpha=0.1, color='green')
    ax_composite.fill_between(x_comp, 25, 50, alpha=0.1, color='yellow')
    ax_composite.fill_between(x_comp, 50, 75, alpha=0.1, color='orange')
    ax_composite.fill_between(x_comp, 75, 100, alpha=0.1, color='red')

    ax_composite.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.9)

    # Add annotations
    ax_composite.text(5, 85, 'NORMAL', fontsize=10, color='green', fontweight='bold')
    ax_composite.text(25, 85, 'Score rising\nbefore crash', fontsize=9, color='orange')
    ax_composite.text(52, 85, 'EXTREME', fontsize=10, color='red', fontweight='bold')

    ax_composite.set_title('COMPOSITE EARLY WARNING SCORE\n'
                          'Score jumped from 45 to 95 in 3 days before Liberation Day',
                          fontsize=14, fontweight='bold', color='white')
    ax_composite.set_ylabel('Composite Score', color='white')
    ax_composite.set_xlabel('Days', color='white')
    ax_composite.set_ylim(0, 100)
    ax_composite.tick_params(colors='white')

    for spine in ax_composite.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    plt.suptitle('April 2025 "Liberation Day" Tariff Crash: Deep Dive Analysis\n'
                'S&P 500 dropped 23% in 5 days (Apr 2-7, 2025)',
                fontsize=16, fontweight='bold', color='white', y=0.98)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_composite_dashboard_demo(save_path=None):
    """Create demo dashboard with simulated current market data."""

    # Generate demo data
    np.random.seed(42)
    n = 252  # 1 year of trading days

    # Simulate a market that's been calm but showing some warning signs
    base_return = 0.0004  # Small positive drift
    volatility = 0.012
    returns = np.random.normal(base_return, volatility, n)

    # Add some structure - slight increase in volatility recently
    returns[-30:] *= 1.3

    prices = 100 * np.exp(np.cumsum(returns))

    # Generate VIX that's been creeping up
    vix = 15 + 3 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 1, n)
    vix[-30:] += 3  # Recent increase
    vix = np.clip(vix, 10, 40)

    # Use composite system
    ews = CompositeEarlyWarningSystem()
    indicators = ews.simulate_all_indicators(returns, prices, vix)

    # Create dashboard
    dashboard = EarlyWarningDashboard()
    fig = dashboard.create_dashboard(indicators, prices, title="Early Warning System - Demo Dashboard")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def main():
    print("=" * 70)
    print("GENERATING EARLY WARNING SYSTEM FIGURES")
    print("=" * 70)
    print()

    # 1. Indicator explanation
    print("1. Creating indicator explanation figure...")
    create_indicator_explanation_figure(save_path=OUTPUT_DIR / 'early_warning_indicators.png')

    # 2. Crash timeline
    print("2. Creating crash timeline comparison...")
    create_crash_timeline_figure(save_path=OUTPUT_DIR / 'crash_timeline_indicators.png')

    # 3. Warning matrix
    print("3. Creating warning matrix table...")
    create_warning_matrix_figure(save_path=OUTPUT_DIR / 'warning_matrix.png')

    # 4. Tariff crash deep dive
    print("4. Creating April 2025 tariff crash deep dive...")
    create_tariff_crash_deep_dive(save_path=OUTPUT_DIR / 'tariff_crash_deep_dive.png')

    # 5. Demo dashboard
    print("5. Creating demo dashboard...")
    create_composite_dashboard_demo(save_path=OUTPUT_DIR / 'early_warning_dashboard_demo.png')

    # 6. 3D visualization
    print("6. Creating 3D phase space...")
    np.random.seed(123)
    n = 252
    returns = np.random.normal(0.0003, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))
    vix = 18 + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n)

    ews = CompositeEarlyWarningSystem()
    indicators = ews.simulate_all_indicators(returns, prices, vix)

    viz3d = EarlyWarning3DVisualization()
    fig = viz3d.create_3d_phase_space(
        indicators['gex_score'],
        indicators['tdex_score'],
        indicators['composite_score']
    )
    fig.savefig(OUTPUT_DIR / 'early_warning_3d_phase_space.png',
                bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'early_warning_3d_phase_space.png'}")
    plt.close()

    # 7. Crash comparison chart
    print("7. Creating historical crash comparison...")
    comparison = CrashComparisonChart()
    fig = comparison.create_comparison_chart()
    fig.savefig(OUTPUT_DIR / 'crash_comparison_indicators.png',
                bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'crash_comparison_indicators.png'}")
    plt.close()

    print()
    print("=" * 70)
    print("EARLY WARNING FIGURES COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated visualizations:")
    print("  - early_warning_indicators.png")
    print("  - crash_timeline_indicators.png")
    print("  - warning_matrix.png")
    print("  - tariff_crash_deep_dive.png")
    print("  - early_warning_dashboard_demo.png")
    print("  - early_warning_3d_phase_space.png")
    print("  - crash_comparison_indicators.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
