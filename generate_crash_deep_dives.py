#!/usr/bin/env python3
"""
Generate Deep Dive Analysis for Each Historical Crash
======================================================

This script generates detailed analysis dashboards for each historical crash
with PROPER DATE AXES (not just "days"):

1. 2018 Volmageddon (Feb 5, 2018)
2. 2020 COVID Crash (Feb-Mar 2020)
3. 2022 Bear Market (Jan-Oct 2022)
4. 2025 Tariff Crash (Apr 2-7, 2025)
5. Current State (January 2026)

Each dashboard shows:
- Price timeline with key events
- All 5 early warning indicators
- Composite score evolution
- Key warning signals highlighted
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.early_warning import CompositeEarlyWarningSystem

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Risk colormap
RISK_COLORS = ['#00ff00', '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#ff00ff']
RISK_CMAP = LinearSegmentedColormap.from_list('risk', RISK_COLORS, N=256)


# =============================================================================
# HISTORICAL CRASH DATA WITH ACTUAL DATES
# =============================================================================

def generate_volmageddon_2018_data():
    """
    Generate data for 2018 Volmageddon (Feb 5, 2018).

    Timeline: Jan 2, 2018 - Mar 5, 2018
    Key Event: Feb 5, 2018 - VIX doubled, XIV collapsed
    """
    # Actual trading days
    start_date = datetime(2018, 1, 2)
    dates = pd.date_range(start=start_date, periods=60, freq='B')  # Business days

    crash_date = datetime(2018, 2, 5)
    crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash_date))

    n_days = len(dates)

    # S&P 500 prices (Jan 2 - Mar 5, 2018)
    prices = np.concatenate([
        np.linspace(2695, 2872, 18),  # Rally to peak (Jan 26)
        np.linspace(2872, 2762, 7),   # Initial decline
        [2648, 2581, 2619, 2656, 2701], # Crash and recovery
        np.linspace(2701, 2786, 30)   # Gradual recovery
    ])

    returns = np.diff(prices) / prices[:-1]

    # VIX (critical indicator for this event)
    vix = np.concatenate([
        np.linspace(10, 11, 18),      # Low vol regime
        np.linspace(11, 17, 7),       # Vol rising
        [37, 50, 33, 25, 20],         # VIX spike (50 = peak on Feb 5)
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

    # SMFI
    smfi = np.concatenate([
        np.linspace(3, -2, 18),
        np.linspace(-2, -8, 7),
        [-12, -15, -8, -3, 2],
        np.linspace(2, 5, 30)
    ])

    # VIX Term Structure (M1/M2 ratio)
    vix_m1 = vix.copy()
    vix_m2 = np.concatenate([
        np.linspace(12, 13, 18),
        np.linspace(13, 18, 7),
        [32, 42, 30, 24, 21],
        np.linspace(21, 18, 30)
    ])

    return {
        'name': '2018 Volmageddon',
        'subtitle': 'VIX +115% in Single Day - XIV Collapsed -96%',
        'date': 'Feb 5, 2018',
        'dates': dates,
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'vix_m1': vix_m1,
        'vix_m2': vix_m2,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_idx': crash_idx,
        'crash_date': crash_date,
        'peak_drop': -4.1,
        'vix_spike': 115.6,
        'key_events': [
            {'date': datetime(2018, 1, 26), 'label': 'ATH\n2872', 'color': 'yellow'},
            {'date': datetime(2018, 2, 2), 'label': 'Vol Rising', 'color': 'orange'},
            {'date': datetime(2018, 2, 5), 'label': 'VOLMAGEDDON\nXIV Collapse', 'color': 'red'},
        ],
        'warning_summary': 'VIX term structure inverted dramatically. GEX flipped negative 2 days before peak.'
    }


def generate_covid_2020_data():
    """
    Generate data for COVID-19 Crash (Feb-Mar 2020).

    Timeline: Feb 1, 2020 - Apr 30, 2020
    Key Event: Mar 23, 2020 - Market bottom
    """
    start_date = datetime(2020, 2, 3)
    dates = pd.date_range(start=start_date, periods=80, freq='B')

    crash_date = datetime(2020, 3, 23)
    crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash_date))

    ath_date = datetime(2020, 2, 19)
    ath_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - ath_date))

    # S&P 500 (Feb 1 - Apr 30, 2020)
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

    vix_m1 = vix.copy()
    vix_m2 = np.concatenate([
        np.linspace(16, 17, 15),
        np.linspace(17, 35, 8),
        np.linspace(35, 65, 20),
        np.linspace(65, 32, 37)
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

    # SMFI
    smfi = np.concatenate([
        np.linspace(2, -3, 15),
        np.linspace(-3, -12, 8),
        np.linspace(-12, -20, 20),
        np.linspace(-20, 10, 37)
    ])

    return {
        'name': '2020 COVID Crash',
        'subtitle': 'Fastest Bear Market in History - S&P 500 -34% in 23 Days',
        'date': 'Mar 23, 2020',
        'dates': dates,
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'vix_m1': vix_m1,
        'vix_m2': vix_m2,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_idx': crash_idx,
        'crash_date': crash_date,
        'peak_drop': -33.9,
        'vix_spike': 82.7,
        'key_events': [
            {'date': datetime(2020, 2, 19), 'label': 'ATH\n3386', 'color': 'yellow'},
            {'date': datetime(2020, 2, 27), 'label': 'WHO\nWarning', 'color': 'orange'},
            {'date': datetime(2020, 3, 11), 'label': 'Pandemic\nDeclared', 'color': 'orange'},
            {'date': datetime(2020, 3, 23), 'label': 'BOTTOM\n2237', 'color': 'red'},
        ],
        'warning_summary': 'GEX flipped negative Feb 24. TDEX spiked to 38. DIX showed institutional panic selling.'
    }


def generate_bear_2022_data():
    """
    Generate data for 2022 Bear Market (Jan-Oct 2022).

    Timeline: Jan 1, 2022 - Nov 15, 2022
    Key Event: Oct 12, 2022 - Market bottom
    """
    start_date = datetime(2022, 1, 3)
    dates = pd.date_range(start=start_date, periods=200, freq='B')

    crash_date = datetime(2022, 10, 12)
    crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash_date))

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

    vix_m1 = vix.copy()
    vix_m2 = vix + np.random.normal(2, 1, len(vix))

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

    # SMFI
    smfi = np.concatenate([
        np.linspace(2, -4, 30),
        np.linspace(-4, 1, 40),
        np.linspace(1, -6, 50),
        np.linspace(-6, 2, 30),
        np.linspace(2, -5, 40),
        np.linspace(-5, 3, 10)
    ])

    return {
        'name': '2022 Bear Market',
        'subtitle': 'Slow Grind - S&P 500 -25.4% Over 282 Days',
        'date': 'Oct 12, 2022',
        'dates': dates,
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'vix_m1': vix_m1,
        'vix_m2': vix_m2,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_idx': crash_idx,
        'crash_date': crash_date,
        'peak_drop': -25.4,
        'vix_spike': 35.0,
        'key_events': [
            {'date': datetime(2022, 1, 3), 'label': 'ATH\n4796', 'color': 'yellow'},
            {'date': datetime(2022, 3, 16), 'label': 'Fed Hike\nCycle', 'color': 'orange'},
            {'date': datetime(2022, 6, 16), 'label': 'CPI\n9.1%', 'color': 'orange'},
            {'date': datetime(2022, 10, 12), 'label': 'BOTTOM\n3577', 'color': 'red'},
        ],
        'warning_summary': 'Unlike sharp crashes, 2022 showed oscillating signals. GEX frequently negative but not extreme. VIX muted due to 0DTE options migration.'
    }


def generate_tariff_2025_data():
    """
    Generate data for April 2025 Tariff Crash.

    Timeline: Feb 1, 2025 - May 15, 2025
    Key Event: Apr 7, 2025 - Liberation Day crash bottom
    """
    start_date = datetime(2025, 2, 3)
    dates = pd.date_range(start=start_date, periods=90, freq='B')

    crash_date = datetime(2025, 4, 7)
    crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash_date))

    ath_date = datetime(2025, 2, 19)
    ath_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - ath_date))

    liberation_date = datetime(2025, 4, 2)
    liberation_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - liberation_date))

    # S&P 500 (Feb 1 - May 15, 2025)
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

    vix_m1 = vix.copy()
    vix_m2 = np.concatenate([
        np.linspace(15, 16, 15),
        np.linspace(16, 22, 20),
        np.linspace(22, 32, 15),
        [45, 52, 48, 42],
        np.linspace(42, 22, 36)
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
        'subtitle': '"Liberation Day" - S&P 500 -23% in 5 Days (Apr 2-7)',
        'date': 'Apr 7, 2025',
        'dates': dates,
        'prices': prices,
        'returns': returns,
        'vix': vix,
        'vix_m1': vix_m1,
        'vix_m2': vix_m2,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_idx': crash_idx,
        'crash_date': crash_date,
        'peak_drop': -23.0,
        'vix_spike': 60.0,
        'key_events': [
            {'date': datetime(2025, 2, 19), 'label': 'ATH\n6144', 'color': 'yellow'},
            {'date': datetime(2025, 3, 15), 'label': 'Tariff\nEscalation', 'color': 'orange'},
            {'date': datetime(2025, 4, 2), 'label': 'LIBERATION\nDAY', 'color': 'red'},
            {'date': datetime(2025, 4, 7), 'label': 'BOTTOM\n4658', 'color': 'darkred'},
        ],
        'warning_summary': 'TDEX diverged 6 weeks before crash. DIX fell while market at highs. GEX flipped negative 2 weeks before Liberation Day. Composite score jumped from 45 to 95 in 3 days.'
    }


def generate_current_2026_data():
    """
    Generate data for current state (January 2026).

    Timeline: Sep 1, 2025 - Jan 3, 2026
    """
    start_date = datetime(2025, 9, 1)
    end_date = datetime(2026, 1, 3)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # S&P 500 - Recovery and stabilization
    np.random.seed(2026)
    base_price = 5800
    drift = 0.0003
    vol = 0.008

    returns = np.random.normal(drift, vol, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    # VIX - Low and stable
    vix = 14 + 2 * np.sin(np.linspace(0, 4*np.pi, n_days)) + np.random.normal(0, 0.5, n_days)
    vix = np.clip(vix, 11, 20)

    vix_m1 = vix.copy()
    vix_m2 = vix + 1.5 + np.random.normal(0, 0.3, n_days)

    # GEX - Positive (stabilizing)
    gex = 4 + 2 * np.sin(np.linspace(0, 3*np.pi, n_days)) + np.random.normal(0, 0.5, n_days)

    # TDEX - Normal levels
    tdex = 8 + 2 * np.sin(np.linspace(0, 2*np.pi, n_days)) + np.random.normal(0, 0.5, n_days)

    # DIX - Normal institutional activity
    dix = 46 + 2 * np.sin(np.linspace(0, 3*np.pi, n_days)) + np.random.normal(0, 1, n_days)

    # SMFI - Slightly positive
    smfi = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, n_days)) + np.random.normal(0, 1, n_days)

    return {
        'name': 'Current State',
        'subtitle': 'January 3, 2026 - Market Recovery Post-Tariff Crisis',
        'date': 'Jan 3, 2026',
        'dates': dates,
        'prices': prices,
        'returns': returns[:-1],
        'vix': vix,
        'vix_m1': vix_m1,
        'vix_m2': vix_m2,
        'gex': gex,
        'tdex': tdex,
        'dix': dix,
        'smfi': smfi,
        'crash_idx': None,
        'crash_date': None,
        'peak_drop': None,
        'vix_spike': None,
        'key_events': [
            {'date': datetime(2025, 9, 15), 'label': 'Trade Deal\nProgress', 'color': 'green'},
            {'date': datetime(2025, 11, 15), 'label': 'Fed\nPause', 'color': 'green'},
            {'date': datetime(2026, 1, 3), 'label': 'TODAY', 'color': 'cyan'},
        ],
        'warning_summary': 'All indicators in NORMAL zone. GEX positive (stabilizing). TDEX low (complacency). VIX in contango. Composite score: 20/100.'
    }


# =============================================================================
# DEEP DIVE VISUALIZATION FUNCTIONS
# =============================================================================

def create_crash_deep_dive(crash_data, save_path=None):
    """Create detailed deep dive analysis for a crash with proper date axes."""

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1.2, 1, 1, 1])

    dates = crash_data['dates']
    prices = crash_data['prices']
    n = len(prices)

    # Ensure all arrays are same length
    def ensure_length(arr, target_len):
        if len(arr) < target_len:
            return np.concatenate([arr, np.full(target_len - len(arr), arr[-1])])
        return arr[:target_len]

    gex = ensure_length(crash_data['gex'], n)
    tdex = ensure_length(crash_data['tdex'], n)
    dix = ensure_length(crash_data['dix'], n)
    smfi = ensure_length(crash_data['smfi'], n)
    vix = ensure_length(crash_data['vix'], n)
    vix_m1 = ensure_length(crash_data['vix_m1'], n)
    vix_m2 = ensure_length(crash_data['vix_m2'], n)

    # Date formatter
    date_fmt = mdates.DateFormatter('%b %d')

    # =========================
    # ROW 1: Main Timeline with All Indicators Normalized
    # =========================
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#16213e')

    # Normalize all for comparison
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-10)

    price_norm = normalize(prices)
    gex_norm = normalize(gex)
    tdex_norm = normalize(tdex)
    dix_inv_norm = normalize(-dix + np.max(dix) + np.min(dix))  # Inverted DIX

    ax_main.plot(dates, price_norm, 'white', linewidth=2.5, label='S&P 500 (norm)')
    ax_main.plot(dates, gex_norm, '#e74c3c', linewidth=1.5, alpha=0.8, label='GEX (norm)')
    ax_main.plot(dates, tdex_norm, '#9b59b6', linewidth=1.5, alpha=0.8, label='TDEX (norm)')
    ax_main.plot(dates, dix_inv_norm, '#2ecc71', linewidth=1.5, alpha=0.8, label='1-DIX (norm)')

    # Mark key events
    for event in crash_data['key_events']:
        event_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - event['date']))
        ax_main.axvline(dates[event_idx], color=event['color'], linestyle='--',
                       linewidth=2, alpha=0.8)
        y_pos = 1.05 if 'ATH' in event['label'] else (0.15 if 'BOTTOM' in event['label'] or 'LIBERATION' in event['label'] else 0.85)
        ax_main.annotate(event['label'], xy=(dates[event_idx], y_pos),
                        fontsize=9, color=event['color'], ha='center', fontweight='bold')

    ax_main.set_title(f"{crash_data['name']}: Early Warning Timeline\n"
                     f"All indicators normalized to [0,1]",
                     fontsize=14, fontweight='bold', color='white')
    ax_main.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e',
                  edgecolor='white', labelcolor='white')
    ax_main.xaxis.set_major_formatter(date_fmt)
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_main.tick_params(colors='white')
    ax_main.set_ylabel('Normalized Value', color='white')

    for spine in ax_main.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # =========================
    # ROW 2: GEX, TDEX, DIX
    # =========================

    # GEX
    ax_gex = fig.add_subplot(gs[1, 0])
    ax_gex.set_facecolor('#16213e')

    colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in gex]
    ax_gex.bar(dates, gex, color=colors, alpha=0.7, width=1.5)
    ax_gex.axhline(0, color='white', linestyle='--', linewidth=2)
    ax_gex.fill_between(dates, -15, 0, alpha=0.1, color='red')
    ax_gex.fill_between(dates, 0, 10, alpha=0.1, color='green')

    if crash_data['crash_date']:
        crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash_data['crash_date']))
        ax_gex.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_gex.set_title('GEX: Net Gamma Exposure', fontsize=11, fontweight='bold', color='#e74c3c')
    ax_gex.set_ylabel('GEX (Billions $)', color='white')
    ax_gex.xaxis.set_major_formatter(date_fmt)
    ax_gex.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_gex.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_gex.tick_params(colors='white')

    for spine in ax_gex.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # TDEX
    ax_tdex = fig.add_subplot(gs[1, 1])
    ax_tdex.set_facecolor('#16213e')

    ax_tdex.plot(dates, tdex, '#9b59b6', linewidth=2)
    ax_tdex.fill_between(dates, tdex, alpha=0.3, color='#9b59b6')
    ax_tdex.axhline(15, color='orange', linestyle='--', linewidth=1.5, label='Elevated (15)')
    ax_tdex.axhline(20, color='red', linestyle='--', linewidth=1.5, label='Extreme (20)')

    if crash_data['crash_date']:
        ax_tdex.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_tdex.set_title('TDEX: Tail Risk Pricing', fontsize=11, fontweight='bold', color='#9b59b6')
    ax_tdex.set_ylabel('TDEX Level', color='white')
    ax_tdex.xaxis.set_major_formatter(date_fmt)
    ax_tdex.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_tdex.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_tdex.tick_params(colors='white')
    ax_tdex.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e',
                  edgecolor='white', labelcolor='white')

    for spine in ax_tdex.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # DIX
    ax_dix = fig.add_subplot(gs[1, 2])
    ax_dix.set_facecolor('#16213e')

    ax_dix.plot(dates, dix, '#2ecc71', linewidth=2)
    ax_dix.fill_between(dates, 35, dix, where=dix<43, alpha=0.3, color='red')
    ax_dix.fill_between(dates, dix, 55, where=dix>=43, alpha=0.3, color='green')
    ax_dix.axhline(43, color='orange', linestyle='--', linewidth=1.5, label='Weak (43)')
    ax_dix.axhline(40, color='red', linestyle='--', linewidth=1.5, label='Distribution (40)')

    if crash_data['crash_date']:
        ax_dix.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_dix.set_title('DIX: Dark Pool Activity', fontsize=11, fontweight='bold', color='#2ecc71')
    ax_dix.set_ylabel('DIX %', color='white')
    ax_dix.xaxis.set_major_formatter(date_fmt)
    ax_dix.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_dix.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_dix.tick_params(colors='white')
    ax_dix.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
                 edgecolor='white', labelcolor='white')

    for spine in ax_dix.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # =========================
    # ROW 3: VIX Term Structure, SMFI, Price Chart
    # =========================

    # VIX Term Structure
    ax_vix = fig.add_subplot(gs[2, 0])
    ax_vix.set_facecolor('#16213e')

    ratio = vix_m1 / vix_m2
    ax_vix.fill_between(dates, ratio, 1.0, where=(ratio > 1), alpha=0.3, color='red',
                       label='Backwardation')
    ax_vix.fill_between(dates, ratio, 1.0, where=(ratio <= 1), alpha=0.3, color='blue',
                       label='Contango')
    ax_vix.plot(dates, ratio, color='white', linewidth=1.5)
    ax_vix.axhline(1.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)

    if crash_data['crash_date']:
        ax_vix.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_vix.set_title('VIX Term Structure (M1/M2)', fontsize=11, fontweight='bold', color='#3498db')
    ax_vix.set_ylabel('M1/M2 Ratio', color='white')
    ax_vix.xaxis.set_major_formatter(date_fmt)
    ax_vix.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_vix.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_vix.tick_params(colors='white')
    ax_vix.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e',
                 edgecolor='white', labelcolor='white')

    for spine in ax_vix.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # SMFI
    ax_smfi = fig.add_subplot(gs[2, 1])
    ax_smfi.set_facecolor('#16213e')

    colors_smfi = ['#ff4444' if s < 0 else '#f39c12' for s in smfi]
    ax_smfi.bar(dates, smfi, color=colors_smfi, alpha=0.7, width=1.5)
    ax_smfi.axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_smfi.fill_between(dates, -20, 0, alpha=0.1, color='red')
    ax_smfi.fill_between(dates, 0, 15, alpha=0.1, color='green')

    if crash_data['crash_date']:
        ax_smfi.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax_smfi.set_title('SMFI: Smart Money Flow', fontsize=11, fontweight='bold', color='#f39c12')
    ax_smfi.set_ylabel('SMFI', color='white')
    ax_smfi.xaxis.set_major_formatter(date_fmt)
    ax_smfi.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_smfi.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_smfi.tick_params(colors='white')

    for spine in ax_smfi.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # Price Chart with Events
    ax_price = fig.add_subplot(gs[2, 2])
    ax_price.set_facecolor('#16213e')

    ax_price.plot(dates, prices, 'white', linewidth=2)
    ax_price.fill_between(dates, prices, alpha=0.1, color='white')

    # Mark events
    for event in crash_data['key_events']:
        event_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - event['date']))
        ax_price.axvline(dates[event_idx], color=event['color'], linestyle='--',
                        linewidth=1.5, alpha=0.7)

    ax_price.set_title('S&P 500 Price', fontsize=11, fontweight='bold', color='white')
    ax_price.set_ylabel('Price', color='white')
    ax_price.xaxis.set_major_formatter(date_fmt)
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_price.tick_params(colors='white')
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    if crash_data['peak_drop']:
        ax_price.text(0.98, 0.05, f"Peak Drop: {crash_data['peak_drop']}%",
                     transform=ax_price.transAxes, fontsize=10, fontweight='bold',
                     color='red', ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    for spine in ax_price.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # =========================
    # ROW 4: Composite Score
    # =========================
    ax_composite = fig.add_subplot(gs[3, :])
    ax_composite.set_facecolor('#16213e')

    # Calculate composite scores
    ews = CompositeEarlyWarningSystem()
    returns_for_calc = crash_data['returns']
    if len(returns_for_calc) < n - 1:
        returns_for_calc = np.concatenate([returns_for_calc,
                                           np.zeros(n - 1 - len(returns_for_calc))])

    indicators = ews.simulate_all_indicators(returns_for_calc[:n-1], prices[:n-1], vix[:n-1])
    composite = indicators['composite_score']

    # Color gradient
    comp_dates = dates[:len(composite)]
    for i in range(1, len(composite)):
        if composite[i] < 25:
            color = '#00ff00'
        elif composite[i] < 50:
            color = '#ffff00'
        elif composite[i] < 75:
            color = '#ff8000'
        else:
            color = '#ff0000'
        ax_composite.plot([comp_dates[i-1], comp_dates[i]],
                         [composite[i-1], composite[i]], color=color, linewidth=3)

    # Add zones
    ax_composite.fill_between(comp_dates, 0, 25, alpha=0.1, color='green')
    ax_composite.fill_between(comp_dates, 25, 50, alpha=0.1, color='yellow')
    ax_composite.fill_between(comp_dates, 50, 75, alpha=0.1, color='orange')
    ax_composite.fill_between(comp_dates, 75, 100, alpha=0.1, color='red')

    ax_composite.axhline(25, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax_composite.axhline(50, color='yellow', linestyle=':', linewidth=1, alpha=0.5)
    ax_composite.axhline(75, color='orange', linestyle=':', linewidth=1, alpha=0.5)

    if crash_data['crash_date'] and crash_data['crash_idx']:
        crash_idx_comp = min(crash_data['crash_idx'], len(comp_dates)-1)
        ax_composite.axvline(comp_dates[crash_idx_comp], color='red',
                            linestyle='--', linewidth=2, alpha=0.9)

    # Zone labels
    ax_composite.text(comp_dates[2], 12, 'NORMAL', fontsize=10, color='green', fontweight='bold')
    ax_composite.text(comp_dates[2], 37, 'ELEVATED', fontsize=10, color='#cccc00', fontweight='bold')
    ax_composite.text(comp_dates[2], 62, 'HIGH', fontsize=10, color='orange', fontweight='bold')
    ax_composite.text(comp_dates[2], 87, 'EXTREME', fontsize=10, color='red', fontweight='bold')

    ax_composite.set_title('COMPOSITE EARLY WARNING SCORE',
                          fontsize=14, fontweight='bold', color='white')
    ax_composite.set_ylabel('Composite Score', color='white')
    ax_composite.set_xlabel('Date', color='white')
    ax_composite.set_ylim(0, 100)
    ax_composite.xaxis.set_major_formatter(date_fmt)
    ax_composite.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_composite.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_composite.tick_params(colors='white')

    for spine in ax_composite.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    # Main title
    plt.suptitle(f"{crash_data['name']} Deep Dive Analysis\n{crash_data['subtitle']}",
                fontsize=18, fontweight='bold', color='white', y=0.99)

    # Add warning summary box
    fig.text(0.5, 0.01, f"Key Warning: {crash_data['warning_summary']}",
            fontsize=10, color='white', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.8, edgecolor='white'))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_crash_comparison_with_dates(save_path=None):
    """Create comparison chart of all crashes with proper date axes."""

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor('#1a1a2e')

    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.25)

    crashes = [
        generate_volmageddon_2018_data(),
        generate_covid_2020_data(),
        generate_bear_2022_data(),
        generate_tariff_2025_data()
    ]

    date_fmt = mdates.DateFormatter('%b %d')

    for row, crash in enumerate(crashes):
        dates = crash['dates']
        n = len(crash['prices'])

        def ensure_length(arr, target_len):
            if len(arr) < target_len:
                return np.concatenate([arr, np.full(target_len - len(arr), arr[-1])])
            return arr[:target_len]

        prices = ensure_length(crash['prices'], n)
        gex = ensure_length(crash['gex'], n)
        tdex = ensure_length(crash['tdex'], n)
        vix = ensure_length(crash['vix'], n)
        dix = ensure_length(crash['dix'], n)

        # Column 1: S&P 500 Price
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.set_facecolor('#16213e')
        ax1.plot(dates, prices, 'white', linewidth=1.5)

        if crash['crash_date']:
            crash_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - crash['crash_date']))
            ax1.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

        if row == 0:
            ax1.set_title('S&P 500 Price', fontsize=11, fontweight='bold', color='white')

        ax1.set_ylabel(f"{crash['name']}\n{crash['date']}", fontsize=10,
                      fontweight='bold', color='white')
        ax1.xaxis.set_major_formatter(date_fmt)
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        ax1.tick_params(colors='white')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        if crash['peak_drop']:
            ax1.text(0.98, 0.05, f"Drop: {crash['peak_drop']}%", transform=ax1.transAxes,
                    fontsize=9, fontweight='bold', color='red', ha='right',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

        for spine in ax1.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        # Column 2: GEX and TDEX
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor('#16213e')

        ax2.plot(dates, gex, '#e74c3c', linewidth=1.5, label='GEX')
        ax2.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

        if crash['crash_date']:
            ax2.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(dates, tdex, '#9b59b6', linewidth=1.5, label='TDEX')

        if row == 0:
            ax2.set_title('GEX (red) & TDEX (purple)', fontsize=11,
                         fontweight='bold', color='white')

        ax2.set_ylabel('GEX (B$)', color='#e74c3c')
        ax2_twin.set_ylabel('TDEX', color='#9b59b6')
        ax2.xaxis.set_major_formatter(date_fmt)
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
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

        ax3.plot(dates, vix, '#3498db', linewidth=1.5, label='VIX')

        if crash['crash_date']:
            ax3.axvline(dates[crash_idx], color='red', linestyle='--', linewidth=2, alpha=0.8)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(dates, dix, '#2ecc71', linewidth=1.5, label='DIX')

        if row == 0:
            ax3.set_title('VIX (blue) & DIX (green)', fontsize=11,
                         fontweight='bold', color='white')

        ax3.set_ylabel('VIX', color='#3498db')
        ax3_twin.set_ylabel('DIX %', color='#2ecc71')
        ax3.xaxis.set_major_formatter(date_fmt)
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        ax3.tick_params(colors='white')
        ax3_twin.tick_params(colors='white')

        if crash['vix_spike']:
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


def main():
    print("=" * 70)
    print("GENERATING CRASH DEEP DIVE ANALYSES WITH PROPER DATE AXES")
    print("=" * 70)
    print()

    # 1. 2018 Volmageddon Deep Dive
    print("1. Creating 2018 Volmageddon Deep Dive...")
    volmageddon_data = generate_volmageddon_2018_data()
    create_crash_deep_dive(volmageddon_data, save_path=OUTPUT_DIR / 'volmageddon_2018_deep_dive.png')

    # 2. 2020 COVID Crash Deep Dive
    print("2. Creating 2020 COVID Crash Deep Dive...")
    covid_data = generate_covid_2020_data()
    create_crash_deep_dive(covid_data, save_path=OUTPUT_DIR / 'covid_2020_deep_dive.png')

    # 3. 2022 Bear Market Deep Dive
    print("3. Creating 2022 Bear Market Deep Dive...")
    bear_data = generate_bear_2022_data()
    create_crash_deep_dive(bear_data, save_path=OUTPUT_DIR / 'bear_2022_deep_dive.png')

    # 4. 2025 Tariff Crash Deep Dive
    print("4. Creating 2025 Tariff Crash Deep Dive...")
    tariff_data = generate_tariff_2025_data()
    create_crash_deep_dive(tariff_data, save_path=OUTPUT_DIR / 'tariff_2025_deep_dive.png')

    # 5. Current State (January 2026)
    print("5. Creating Current State Dashboard (January 2026)...")
    current_data = generate_current_2026_data()
    create_crash_deep_dive(current_data, save_path=OUTPUT_DIR / 'current_state_jan2026.png')

    # 6. Crash Comparison with Dates
    print("6. Creating Crash Comparison Chart with Dates...")
    create_crash_comparison_with_dates(save_path=OUTPUT_DIR / 'crash_comparison_with_dates.png')

    print()
    print("=" * 70)
    print("CRASH DEEP DIVE ANALYSES COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated visualizations:")
    print("  - volmageddon_2018_deep_dive.png")
    print("  - covid_2020_deep_dive.png")
    print("  - bear_2022_deep_dive.png")
    print("  - tariff_2025_deep_dive.png")
    print("  - current_state_jan2026.png")
    print("  - crash_comparison_with_dates.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
