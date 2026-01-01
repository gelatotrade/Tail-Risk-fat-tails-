#!/usr/bin/env python3
"""
Generate Real-World Crisis Examples - S&P 500 Analysis
========================================================

Simulates S&P 500 behavior during:
1. COVID-19 Crash (February-April 2020)
2. Tariff Crisis (March-April 2025)

Shows how early warning signals would have detected these events.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sp500_covid_crash():
    """
    Generate realistic S&P 500 returns mimicking COVID-19 crash.

    Historical S&P 500 Facts:
    - Peak: 3,386.15 (Feb 19, 2020)
    - Trough: 2,237.40 (Mar 23, 2020)
    - Drop: -33.9% in 23 trading days

    Key Daily Returns (actual S&P 500):
    - Feb 24: -3.4% (first major drop)
    - Feb 27: -4.4%
    - Feb 28: -0.8%
    - Mar 9:  -7.6% (first circuit breaker triggered)
    - Mar 10: +4.9% (dead cat bounce)
    - Mar 11: -4.9% (WHO declares pandemic)
    - Mar 12: -9.5% (worst day since 1987)
    - Mar 13: +9.3% (relief rally)
    - Mar 16: -12.0% (worst day since 1987, second circuit breaker)
    - Mar 18: -5.2%
    - Mar 23: -2.9% (bottom)
    - Mar 24: +9.4% (biggest gain since 2008)
    """
    np.random.seed(2020)

    # Pre-crash period: Jan 2 - Feb 19 (34 trading days of calm bull market)
    n_pre = 34
    pre_returns = np.random.normal(0.0006, 0.006, n_pre)  # Low vol, slight uptrend
    # Add some actual early 2020 characteristics
    pre_returns[-5:] = np.random.normal(0.001, 0.004, 5)  # Very calm before storm

    # Initial selloff: Feb 20-28 (7 trading days)
    initial_selloff = np.array([
        -0.004,  # Feb 20
        -0.016,  # Feb 21
        -0.005,  # Weekend effect
        -0.034,  # Feb 24 - First major drop
        -0.030,  # Feb 25
        -0.004,  # Feb 26
        -0.044,  # Feb 27
    ])

    # Volatile period leading to crash: Mar 2-6 (5 days)
    volatile_period = np.array([
        -0.008,  # Feb 28
        +0.046,  # Mar 2 - Bounce
        -0.028,  # Mar 3
        +0.042,  # Mar 4 - Super Tuesday rally
        -0.034,  # Mar 5
        -0.017,  # Mar 6
    ])

    # Main crash phase: Mar 9-23 (11 trading days)
    crash_phase = np.array([
        -0.076,  # Mar 9 - First circuit breaker, oil crash
        +0.049,  # Mar 10 - Dead cat bounce
        -0.049,  # Mar 11 - WHO pandemic declaration
        -0.095,  # Mar 12 - Worst since 1987
        +0.093,  # Mar 13 - Relief rally
        -0.005,  # Mar 14 (weekend trading effect)
        -0.120,  # Mar 16 - WORST DAY, circuit breaker at open
        +0.060,  # Mar 17 - Bounce
        -0.052,  # Mar 18
        +0.006,  # Mar 19
        -0.043,  # Mar 20
        -0.029,  # Mar 23 - THE BOTTOM
    ])

    # Recovery phase: Mar 24 - Apr 30 (28 trading days)
    n_recovery = 28
    recovery_base = np.random.normal(0.012, 0.022, n_recovery)
    recovery_base[0] = 0.094   # Mar 24 - +9.4% biggest gain since 2008
    recovery_base[1] = 0.011   # Mar 25
    recovery_base[2] = 0.063   # Mar 26 - +6.3%
    recovery_base[5] = 0.032   # Continued strength
    recovery_returns = recovery_base

    # Combine all phases
    returns = np.concatenate([
        pre_returns,
        initial_selloff,
        volatile_period,
        crash_phase,
        recovery_returns
    ])

    # Generate dates (trading days only, skip weekends)
    start_date = datetime(2020, 1, 2)
    dates = []
    current = start_date
    for _ in range(len(returns)):
        while current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
        dates.append(current)
        current += timedelta(days=1)

    # Key event indices
    pre_end = n_pre - 1
    selloff_start = n_pre
    crash_start = n_pre + len(initial_selloff) + len(volatile_period)
    crash_end = crash_start + len(crash_phase) - 1

    events = {
        'pre_crash_end': pre_end,
        'selloff_start': selloff_start,
        'crash_start': crash_start,
        'mar_16_idx': crash_start + 6,  # Mar 16 worst day
        'bottom_idx': crash_end,
        'crash_end': crash_end,
        'recovery_start': crash_end + 1
    }

    return returns, dates, events


def generate_sp500_tariff_crash():
    """
    Generate realistic S&P 500 returns mimicking 2025 Tariff Crisis.

    Hypothetical scenario based on trade war dynamics:
    - Pre-crisis: S&P 500 at ~5,800 level (Jan-Feb 2025)
    - Escalation triggers: New tariff announcements
    - Market reaction: More gradual than COVID, policy-driven
    - Total drawdown: ~18-22% over 5-6 weeks

    Key events (hypothetical):
    - Mar 3: Initial 25% tariff announcement (-2.8%)
    - Mar 10: Retaliatory tariffs from China (-4.2%)
    - Mar 17: EU joins with counter-tariffs (-3.8%)
    - Mar 24: Supply chain disruption fears peak (-4.5%)
    - Apr 2: Worst day as recession fears mount (-5.2%)
    - Apr 7: Trade talks announced, recovery begins
    """
    np.random.seed(2025)

    # Pre-crisis: Jan 2 - Feb 28 (40 trading days, normal market)
    n_pre = 40
    pre_returns = np.random.normal(0.0004, 0.008, n_pre)
    # Late Feb: some nervousness emerging
    pre_returns[-5:] = np.random.normal(-0.001, 0.010, 5)

    # Escalation phase: Mar 3-14 (10 trading days)
    escalation = np.array([
        -0.028,  # Mar 3 - Initial tariff announcement
        -0.015,  # Mar 4
        +0.012,  # Mar 5 - Bounce on negotiation hopes
        -0.022,  # Mar 6 - Hopes fade
        -0.018,  # Mar 7
        -0.042,  # Mar 10 - China retaliates
        +0.025,  # Mar 11 - Dead cat bounce
        -0.031,  # Mar 12
        -0.019,  # Mar 13
        -0.025,  # Mar 14
    ])

    # Main crash phase: Mar 17 - Apr 4 (15 trading days)
    crash_phase = np.array([
        -0.038,  # Mar 17 - EU counter-tariffs
        -0.022,  # Mar 18
        +0.018,  # Mar 19 - Brief bounce
        -0.029,  # Mar 20
        -0.035,  # Mar 21 - Supply chain fears
        -0.045,  # Mar 24 - Supply chain disruption peak
        +0.022,  # Mar 25 - Oversold bounce
        -0.028,  # Mar 26
        -0.018,  # Mar 27
        -0.032,  # Mar 28
        -0.041,  # Mar 31 - Quarter end selling
        -0.025,  # Apr 1
        -0.052,  # Apr 2 - WORST DAY, recession fears
        +0.015,  # Apr 3 - Exhaustion bounce
        -0.038,  # Apr 4 - Final capitulation
    ])

    # Stabilization/Recovery: Apr 7-30 (18 trading days)
    n_recovery = 18
    recovery_returns = np.random.normal(0.005, 0.014, n_recovery)
    recovery_returns[0] = 0.035   # Apr 7 - Trade talks announced
    recovery_returns[1] = 0.022   # Apr 8 - Follow through
    recovery_returns[4] = 0.028   # Apr 11 - Progress reported
    recovery_returns[8] = -0.015  # Apr 17 - Minor setback
    recovery_returns[10] = 0.018  # Apr 21 - Deal framework

    returns = np.concatenate([pre_returns, escalation, crash_phase, recovery_returns])

    start_date = datetime(2025, 1, 2)
    dates = []
    current = start_date
    for _ in range(len(returns)):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
        current += timedelta(days=1)

    crash_start = n_pre + len(escalation)
    crash_end = crash_start + len(crash_phase) - 1

    events = {
        'pre_crisis_end': n_pre - 1,
        'escalation_start': n_pre,
        'crash_start': crash_start,
        'apr_2_idx': crash_start + 12,  # Apr 2 worst day
        'bottom_idx': crash_end,
        'crash_end': crash_end,
        'recovery_start': crash_end + 1
    }

    return returns, dates, events


def compute_early_warning_signals(returns, window=20):
    """Compute rolling EWS metrics."""
    n = len(returns)

    # Rolling autocorrelation (lag-1)
    ac1 = np.full(n, np.nan)
    for i in range(window, n):
        segment = returns[i-window:i]
        if len(segment) > 1:
            ac1[i] = np.corrcoef(segment[:-1], segment[1:])[0, 1]

    # Rolling variance
    rolling_var = np.full(n, np.nan)
    for i in range(window, n):
        rolling_var[i] = np.var(returns[i-window:i])

    # Rolling skewness
    rolling_skew = np.full(n, np.nan)
    for i in range(window, n):
        rolling_skew[i] = stats.skew(returns[i-window:i])

    # Rolling kurtosis
    rolling_kurt = np.full(n, np.nan)
    for i in range(window, n):
        rolling_kurt[i] = stats.kurtosis(returns[i-window:i])

    # Composite warning indicator
    def normalize(x):
        valid = ~np.isnan(x)
        if valid.sum() > 0:
            x_norm = x.copy()
            x_norm[valid] = (x[valid] - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
            return x_norm
        return x

    ac1_norm = normalize(ac1)
    var_norm = normalize(rolling_var)
    skew_norm = normalize(-rolling_skew)  # Negative skew is warning

    composite = (ac1_norm + var_norm + skew_norm) / 3

    return {
        'autocorrelation': ac1,
        'variance': rolling_var,
        'skewness': rolling_skew,
        'kurtosis': rolling_kurt,
        'composite': composite
    }


def create_crisis_analysis_figure(returns, dates, events, ews, crisis_name,
                                   crisis_color='red', save_path=None,
                                   sp500_start_level=None):
    """Create comprehensive crisis analysis figure with S&P 500 levels."""

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1.2, 1, 1, 1, 1])

    n = len(returns)
    t = np.arange(n)

    # Convert dates to strings for x-axis
    date_strs = [d.strftime('%b %d') for d in dates]

    # Determine crisis period for shading
    crisis_start = events.get('crash_start', events.get('escalation_start', 0))
    crisis_end = events.get('crash_end', n-1)

    # ===== Row 1: S&P 500 Price Level =====
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate S&P 500 level from returns
    if sp500_start_level is None:
        sp500_start_level = 3300 if '2020' in crisis_name else 5800

    sp500_levels = sp500_start_level * np.cumprod(1 + returns)

    ax1.plot(t, sp500_levels, 'b-', linewidth=1.5, label='S&P 500')
    ax1.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15,
                label='Crisis Period')

    # Mark peak and trough
    peak_idx = np.argmax(sp500_levels[:crisis_end+5])
    trough_idx = crisis_start + np.argmin(sp500_levels[crisis_start:crisis_end+1])

    ax1.axvline(peak_idx, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(trough_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Calculate and annotate drawdown
    peak_price = sp500_levels[peak_idx]
    trough_price = sp500_levels[trough_idx]
    drawdown = (trough_price - peak_price) / peak_price * 100

    ax1.annotate(f'Peak: {peak_price:,.0f}', xy=(peak_idx, peak_price),
                xytext=(peak_idx-8, peak_price+100), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax1.annotate(f'Trough: {trough_price:,.0f}\n({drawdown:.1f}%)',
                xy=(trough_idx, trough_price),
                xytext=(trough_idx+5, trough_price-150), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax1.set_ylabel('S&P 500 Level')
    ax1.set_title(f'{crisis_name}: S&P 500 Price Action', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, n-1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Add date labels
    tick_positions = np.linspace(0, n-1, 8).astype(int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([date_strs[i] for i in tick_positions], rotation=45)

    # ===== Row 2: Daily Returns =====
    ax2 = fig.add_subplot(gs[1, :])

    colors = ['green' if r >= 0 else 'red' for r in returns]
    ax2.bar(t, returns * 100, color=colors, alpha=0.7, width=1.0)
    ax2.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.1)
    ax2.axhline(0, color='black', linewidth=0.5)

    # Highlight extreme days (>4% move)
    extreme_threshold = 0.04
    extreme_days = np.where(np.abs(returns) > extreme_threshold)[0]
    for ed in extreme_days:
        ax2.annotate(f'{returns[ed]*100:.1f}%', xy=(ed, returns[ed]*100),
                    fontsize=8, ha='center', fontweight='bold',
                    va='bottom' if returns[ed] > 0 else 'top')

    ax2.set_ylabel('S&P 500 Daily Return (%)')
    ax2.set_title('S&P 500 Daily Returns', fontweight='bold')
    ax2.set_xlim(0, n-1)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([date_strs[i] for i in tick_positions], rotation=45)

    # ===== Row 3: Autocorrelation (Critical Slowing Down) =====
    ax3 = fig.add_subplot(gs[2, 0])

    ax3.plot(t, ews['autocorrelation'], 'b-', linewidth=1.5)
    ax3.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax3.axhline(0.3, color='orange', linestyle=':', linewidth=2, label='Warning Level')
    ax3.axhline(0.5, color='red', linestyle=':', linewidth=2, label='Critical Level')
    ax3.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2)

    # Mark where signal crossed threshold before crisis
    pre_crisis = ews['autocorrelation'][:crisis_start]
    crosses = np.where(pre_crisis > 0.3)[0]
    if len(crosses) > 0:
        first_warning = crosses[0]
        days_ahead = crisis_start - first_warning
        ax3.scatter([first_warning], [ews['autocorrelation'][first_warning]],
                   s=100, c='orange', marker='*', zorder=5)
        ax3.annotate(f'Warning\n{days_ahead}d ahead', xy=(first_warning, 0.35),
                    fontsize=9, ha='center', color='darkorange', fontweight='bold')

    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('EWS: Autocorrelation (Critical Slowing Down)', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_xlim(0, n-1)
    ax3.set_ylim(-0.5, 1.0)

    # ===== Row 3: Rolling Variance =====
    ax4 = fig.add_subplot(gs[2, 1])

    var_bps = ews['variance'] * 10000  # Convert to basis points squared
    ax4.plot(t, var_bps, 'g-', linewidth=1.5)
    ax4.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax4.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2)

    # Add threshold based on pre-crisis median
    pre_crisis_var = np.nanmedian(var_bps[:crisis_start])
    ax4.axhline(pre_crisis_var * 2, color='orange', linestyle=':', linewidth=2,
                label=f'2x Normal')
    ax4.axhline(pre_crisis_var * 4, color='red', linestyle=':', linewidth=2,
                label=f'4x Normal')

    ax4.set_ylabel('Variance (bps²)')
    ax4.set_title('EWS: Variance Explosion', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_xlim(0, n-1)

    # ===== Row 4: Rolling Skewness =====
    ax5 = fig.add_subplot(gs[3, 0])

    ax5.plot(t, ews['skewness'], 'm-', linewidth=1.5)
    ax5.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax5.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2)
    ax5.axhline(0, color='gray', linewidth=1)
    ax5.axhline(-0.5, color='orange', linestyle=':', linewidth=2, label='Warning')
    ax5.axhline(-1.0, color='red', linestyle=':', linewidth=2, label='Critical')

    ax5.set_ylabel('Skewness')
    ax5.set_title('EWS: Negative Skewness (Crash Risk)', fontweight='bold')
    ax5.legend(loc='lower left', fontsize=8)
    ax5.set_xlim(0, n-1)

    # ===== Row 4: Rolling Kurtosis =====
    ax6 = fig.add_subplot(gs[3, 1])

    ax6.plot(t, ews['kurtosis'], 'c-', linewidth=1.5)
    ax6.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax6.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2)
    ax6.axhline(3, color='gray', linewidth=1, label='Normal (3)')
    ax6.axhline(6, color='orange', linestyle=':', linewidth=2, label='Elevated (6)')

    ax6.set_ylabel('Kurtosis')
    ax6.set_title('EWS: Excess Kurtosis (Fat Tails)', fontweight='bold')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.set_xlim(0, n-1)

    # ===== Row 5: Composite Warning Indicator =====
    ax7 = fig.add_subplot(gs[4, :])

    composite = ews['composite']

    # Color by risk level
    for i in range(1, n):
        if not np.isnan(composite[i]):
            color = 'green' if composite[i] < 0.5 else ('orange' if composite[i] < 1.5 else 'red')
            ax7.plot([t[i-1], t[i]], [composite[i-1], composite[i]], color=color, linewidth=2)

    ax7.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax7.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2, label='Crisis Start')
    ax7.axhline(0.5, color='orange', linestyle=':', linewidth=2)
    ax7.axhline(1.5, color='red', linestyle=':', linewidth=2)
    ax7.fill_between(t, -2, 0.5, alpha=0.1, color='green', label='Normal')
    ax7.fill_between(t, 0.5, 1.5, alpha=0.1, color='orange', label='Elevated')
    ax7.fill_between(t, 1.5, 4, alpha=0.1, color='red', label='Critical')

    ax7.set_ylabel('Composite Risk Score')
    ax7.set_xlabel('Date')
    ax7.set_title('COMPOSITE EARLY WARNING INDICATOR', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=8, ncol=4)
    ax7.set_xlim(0, n-1)
    ax7.set_ylim(-2, 4)
    ax7.set_xticks(tick_positions)
    ax7.set_xticklabels([date_strs[i] for i in tick_positions], rotation=45)

    plt.suptitle(f'{crisis_name}\nS&P 500 Early Warning Signal Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_comparison_summary(covid_returns, covid_events, covid_ews, covid_dates,
                               tariff_returns, tariff_events, tariff_ews, tariff_dates,
                               save_path=None):
    """Create side-by-side comparison of both S&P 500 crises."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    crises = [
        ('COVID-19 Crash (2020)', covid_returns, covid_events, covid_ews, covid_dates, 'red', 3386),
        ('Tariff Crisis (2025)', tariff_returns, tariff_events, tariff_ews, tariff_dates, 'purple', 5800)
    ]

    for row, (name, returns, events, ews, dates, color, start_level) in enumerate(crises):
        n = len(returns)
        t = np.arange(n)
        crisis_start = events.get('crash_start', events.get('escalation_start', 0))
        crisis_end = events.get('crash_end', n-1)

        # S&P 500 level
        ax1 = axes[row, 0]
        sp500 = start_level * np.cumprod(1 + returns)
        ax1.plot(t, sp500, 'b-', linewidth=1.5)
        ax1.axvspan(crisis_start, crisis_end, color=color, alpha=0.2)

        peak_idx = np.argmax(sp500[:crisis_end+5])
        trough_idx = crisis_start + np.argmin(sp500[crisis_start:crisis_end+1])
        drawdown = (sp500[trough_idx] - sp500[peak_idx]) / sp500[peak_idx] * 100

        ax1.set_title(f'{name}\nS&P 500 Drawdown: {drawdown:.1f}%', fontweight='bold')
        ax1.set_ylabel('S&P 500')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        if row == 1:
            ax1.set_xlabel('Trading Days')

        # Early warning: Autocorrelation
        ax2 = axes[row, 1]
        ax2.plot(t, ews['autocorrelation'], 'b-', linewidth=1.5, label='AC(1)')
        ax2.axvspan(crisis_start, crisis_end, color=color, alpha=0.2)
        ax2.axvline(crisis_start, color=color, linestyle='--', linewidth=2)
        ax2.axhline(0.3, color='orange', linestyle=':', linewidth=1.5)
        ax2.set_title('Autocorrelation', fontweight='bold')
        ax2.set_ylabel('AC(1)')
        ax2.legend(loc='upper right', fontsize=8)
        if row == 1:
            ax2.set_xlabel('Trading Days')

        # Composite indicator
        ax3 = axes[row, 2]
        composite = ews['composite']
        ax3.fill_between(t, 0, composite, where=composite < 0.5,
                        color='green', alpha=0.5, label='Normal')
        ax3.fill_between(t, 0, composite, where=(composite >= 0.5) & (composite < 1.5),
                        color='orange', alpha=0.5, label='Elevated')
        ax3.fill_between(t, 0, composite, where=composite >= 1.5,
                        color='red', alpha=0.5, label='Critical')
        ax3.axvspan(crisis_start, crisis_end, color=color, alpha=0.15)
        ax3.axvline(crisis_start, color=color, linestyle='--', linewidth=2)
        ax3.set_title('Composite Warning', fontweight='bold')
        ax3.set_ylabel('Risk Score')
        ax3.legend(loc='upper right', fontsize=8)
        if row == 1:
            ax3.set_xlabel('Trading Days')

    plt.suptitle('S&P 500 Crisis Comparison: Early Warning Signals', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()


def generate_dashboard_for_crisis(returns, crisis_name, save_path):
    """Generate full tail risk dashboard for a crisis period."""
    from src.visualization.dashboard import TailRiskDashboard

    dashboard = TailRiskDashboard(returns)
    fig = dashboard.create_full_dashboard(save_path=save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    print("=" * 70)
    print("GENERATING S&P 500 CRISIS ANALYSIS")
    print("=" * 70)
    print()

    # ===== COVID-19 Crash =====
    print("1. COVID-19 Crash (February-April 2020)")
    print("-" * 50)
    covid_returns, covid_dates, covid_events = generate_sp500_covid_crash()
    covid_ews = compute_early_warning_signals(covid_returns, window=15)

    # Calculate S&P 500 stats
    sp500_covid = 3386 * np.cumprod(1 + covid_returns)  # Start at actual peak
    crisis_start = covid_events['crash_start']
    crisis_end = covid_events['crash_end']

    peak = np.max(sp500_covid[:crisis_end+5])
    trough = np.min(sp500_covid[crisis_start:crisis_end+1])
    drawdown = (trough - peak) / peak * 100
    worst_day = np.min(covid_returns[crisis_start:crisis_end+1]) * 100
    best_day = np.max(covid_returns) * 100

    print(f"   S&P 500 Peak:     {peak:,.0f} ({covid_dates[np.argmax(sp500_covid[:crisis_end+5])].strftime('%b %d, %Y')})")
    print(f"   S&P 500 Trough:   {trough:,.0f} ({covid_dates[crisis_start + np.argmin(sp500_covid[crisis_start:crisis_end+1])].strftime('%b %d, %Y')})")
    print(f"   Maximum Drawdown: {drawdown:.1f}%")
    print(f"   Worst Day:        {worst_day:.1f}% (Mar 16, 2020)")
    print(f"   Best Day:         +{best_day:.1f}% (Mar 24, 2020)")
    print()

    # Generate COVID analysis
    create_crisis_analysis_figure(
        covid_returns, covid_dates, covid_events, covid_ews,
        'COVID-19 Market Crash (2020)',
        crisis_color='red',
        save_path=OUTPUT_DIR / 'covid_crash_analysis.png',
        sp500_start_level=3300
    )

    generate_dashboard_for_crisis(
        covid_returns, 'COVID-19',
        save_path=OUTPUT_DIR / 'covid_crash_dashboard.png'
    )

    # ===== Tariff Crisis =====
    print("2. Tariff Crisis (March-April 2025)")
    print("-" * 50)
    tariff_returns, tariff_dates, tariff_events = generate_sp500_tariff_crash()
    tariff_ews = compute_early_warning_signals(tariff_returns, window=15)

    sp500_tariff = 5800 * np.cumprod(1 + tariff_returns)
    crisis_start = tariff_events['crash_start']
    crisis_end = tariff_events['crash_end']

    peak = np.max(sp500_tariff[:crisis_end+5])
    trough = np.min(sp500_tariff[crisis_start:crisis_end+1])
    drawdown = (trough - peak) / peak * 100
    worst_day = np.min(tariff_returns[crisis_start:crisis_end+1]) * 100

    print(f"   S&P 500 Peak:     {peak:,.0f}")
    print(f"   S&P 500 Trough:   {trough:,.0f}")
    print(f"   Maximum Drawdown: {drawdown:.1f}%")
    print(f"   Worst Day:        {worst_day:.1f}%")
    print()

    create_crisis_analysis_figure(
        tariff_returns, tariff_dates, tariff_events, tariff_ews,
        'Tariff Crisis (2025)',
        crisis_color='purple',
        save_path=OUTPUT_DIR / 'tariff_crash_analysis.png',
        sp500_start_level=5800
    )

    generate_dashboard_for_crisis(
        tariff_returns, 'Tariff Crisis',
        save_path=OUTPUT_DIR / 'tariff_crash_dashboard.png'
    )

    # ===== Comparison =====
    print("3. Generating S&P 500 crisis comparison...")
    print("-" * 50)
    create_comparison_summary(
        covid_returns, covid_events, covid_ews, covid_dates,
        tariff_returns, tariff_events, tariff_ews, tariff_dates,
        save_path=OUTPUT_DIR / 'crisis_comparison.png'
    )

    print()
    print("=" * 70)
    print("S&P 500 CRISIS ANALYSIS COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
