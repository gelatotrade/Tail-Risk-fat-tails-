#!/usr/bin/env python3
"""
Generate Real-World Crisis Examples
====================================

Simulates market behavior during:
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


def generate_covid_crash_data():
    """
    Generate realistic market data mimicking COVID-19 crash.

    Key characteristics:
    - Pre-crash: Low volatility bull market (Jan-Feb 2020)
    - Crash: -34% in ~23 trading days (Feb 19 - Mar 23, 2020)
    - Recovery: V-shaped recovery (Mar 23 - Apr 2020)

    S&P 500 facts:
    - Peak: 3,386 (Feb 19, 2020)
    - Trough: 2,237 (Mar 23, 2020)
    - Drop: -33.9%
    """
    np.random.seed(2020)

    # Pre-crash period: ~50 days of low vol bull market (Jan 2 - Feb 19)
    n_pre = 50
    pre_returns = np.random.normal(0.0008, 0.008, n_pre)  # Calm market

    # Building tension: ~8 days of increasing uncertainty (Feb 20-28)
    n_tension = 8
    tension_returns = np.random.normal(-0.002, 0.015, n_tension)  # Rising vol

    # Crash phase: ~18 days of extreme moves (Mar 2 - Mar 23)
    n_crash = 18
    # Large negative returns with some dead cat bounces
    crash_base = np.random.normal(-0.025, 0.04, n_crash)
    # Add specific extreme days (Mar 9: -7.6%, Mar 12: -9.5%, Mar 16: -12%)
    crash_base[5] = -0.076   # Mar 9 - First circuit breaker
    crash_base[8] = -0.095   # Mar 12 - Worst day since 1987
    crash_base[10] = -0.12   # Mar 16 - Second worst day ever
    crash_base[7] = 0.049    # Mar 10 - Dead cat bounce +4.9%
    crash_base[13] = 0.06    # Mar 17 - Relief rally
    crash_returns = crash_base

    # Recovery phase: ~30 days (Mar 24 - Apr 30)
    n_recovery = 30
    recovery_returns = np.random.normal(0.015, 0.025, n_recovery)  # V-shaped
    recovery_returns[0] = 0.094  # Mar 24: +9.4% (biggest gain since 2008)

    # Combine all phases
    returns = np.concatenate([pre_returns, tension_returns, crash_returns, recovery_returns])

    # Generate dates (trading days only)
    start_date = datetime(2020, 1, 2)
    dates = []
    current = start_date
    for _ in range(len(returns)):
        while current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
        dates.append(current)
        current += timedelta(days=1)

    # Key event indices
    events = {
        'pre_crash_end': n_pre - 1,
        'tension_start': n_pre,
        'crash_start': n_pre + n_tension,
        'crash_peak': n_pre + n_tension + 10,  # Mar 16
        'crash_end': n_pre + n_tension + n_crash - 1,
        'recovery_start': n_pre + n_tension + n_crash
    }

    return returns, dates, events


def generate_tariff_crash_data():
    """
    Generate realistic market data mimicking 2025 Tariff Crisis.

    Hypothetical scenario based on trade war dynamics:
    - Pre-crisis: Normal market conditions (Jan-Feb 2025)
    - Escalation: Tariff announcements cause uncertainty (early Mar)
    - Crash: Retaliatory tariffs trigger selloff (mid Mar - early Apr)
    - Stabilization: Policy negotiations bring calm (Apr)

    Characteristics:
    - More gradual than COVID (policy-driven, not pandemic shock)
    - Multiple waves as different tariffs announced
    - ~20% drawdown over 4-5 weeks
    """
    np.random.seed(2025)

    # Pre-crisis: ~40 days normal market (Jan-Feb 2025)
    n_pre = 40
    pre_returns = np.random.normal(0.0005, 0.009, n_pre)

    # Escalation phase: ~15 days of rising uncertainty (early Mar)
    n_escalation = 15
    escalation_returns = np.random.normal(-0.003, 0.018, n_escalation)
    escalation_returns[3] = -0.028   # First major tariff announcement
    escalation_returns[8] = -0.035   # Retaliatory tariffs announced
    escalation_returns[12] = 0.022   # Brief hope of negotiations

    # Crash phase: ~20 days of sustained selling (mid Mar - early Apr)
    n_crash = 20
    crash_returns = np.random.normal(-0.012, 0.028, n_crash)
    crash_returns[2] = -0.045   # Major escalation
    crash_returns[5] = -0.052   # China retaliates
    crash_returns[8] = 0.032    # Dead cat bounce
    crash_returns[11] = -0.048  # EU joins tariffs
    crash_returns[15] = -0.038  # Supply chain fears
    crash_returns[18] = -0.042  # Peak panic

    # Stabilization: ~25 days (negotiations begin)
    n_recovery = 25
    recovery_returns = np.random.normal(0.006, 0.015, n_recovery)
    recovery_returns[2] = 0.035   # Talks announced
    recovery_returns[8] = 0.028   # Progress reported

    returns = np.concatenate([pre_returns, escalation_returns, crash_returns, recovery_returns])

    start_date = datetime(2025, 1, 2)
    dates = []
    current = start_date
    for _ in range(len(returns)):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
        current += timedelta(days=1)

    events = {
        'pre_crisis_end': n_pre - 1,
        'escalation_start': n_pre,
        'crash_start': n_pre + n_escalation,
        'crash_peak': n_pre + n_escalation + 18,
        'crash_end': n_pre + n_escalation + n_crash - 1,
        'recovery_start': n_pre + n_escalation + n_crash
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
    # Normalize each signal and combine
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
                                   crisis_color='red', save_path=None):
    """Create comprehensive crisis analysis figure."""

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

    # ===== Row 1: Price trajectory and cumulative returns =====
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate cumulative returns (price path)
    cum_returns = np.cumprod(1 + returns) * 100  # Starting at 100

    ax1.plot(t, cum_returns, 'b-', linewidth=1.5, label='Price Index')
    ax1.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15,
                label='Crisis Period')

    # Mark key events
    peak_idx = np.argmax(cum_returns[:crisis_end+5])
    trough_idx = crisis_start + np.argmin(cum_returns[crisis_start:crisis_end+1])

    ax1.axvline(peak_idx, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(trough_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Calculate drawdown
    peak_price = cum_returns[peak_idx]
    trough_price = cum_returns[trough_idx]
    drawdown = (trough_price - peak_price) / peak_price * 100

    ax1.annotate(f'Peak: {peak_price:.1f}', xy=(peak_idx, peak_price),
                xytext=(peak_idx-5, peak_price+3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green', lw=1))
    ax1.annotate(f'Trough: {trough_price:.1f}\n({drawdown:.1f}%)',
                xy=(trough_idx, trough_price),
                xytext=(trough_idx+3, trough_price-5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax1.set_ylabel('Price Index')
    ax1.set_title(f'{crisis_name}: Market Price Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, n-1)

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

    # Highlight extreme days
    extreme_threshold = 0.04
    extreme_days = np.where(np.abs(returns) > extreme_threshold)[0]
    for ed in extreme_days:
        ax2.annotate(f'{returns[ed]*100:.1f}%', xy=(ed, returns[ed]*100),
                    fontsize=7, ha='center',
                    va='bottom' if returns[ed] > 0 else 'top')

    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title('Daily Returns', fontweight='bold')
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
        ax3.annotate(f'Warning\n{days_ahead}d before', xy=(first_warning, 0.35),
                    fontsize=8, ha='center', color='darkorange', fontweight='bold')

    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Early Warning: Autocorrelation (Critical Slowing Down)', fontweight='bold')
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
                label=f'2x Normal ({pre_crisis_var*2:.0f})')
    ax4.axhline(pre_crisis_var * 4, color='red', linestyle=':', linewidth=2,
                label=f'4x Normal ({pre_crisis_var*4:.0f})')

    ax4.set_ylabel('Variance (bps²)')
    ax4.set_title('Early Warning: Variance Explosion', fontweight='bold')
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
    ax5.set_title('Early Warning: Negative Skewness', fontweight='bold')
    ax5.legend(loc='lower left', fontsize=8)
    ax5.set_xlim(0, n-1)

    # ===== Row 4: Rolling Kurtosis =====
    ax6 = fig.add_subplot(gs[3, 1])

    ax6.plot(t, ews['kurtosis'], 'c-', linewidth=1.5)
    ax6.axvspan(crisis_start, crisis_end, color=crisis_color, alpha=0.15)
    ax6.axvline(crisis_start, color=crisis_color, linestyle='--', linewidth=2)
    ax6.axhline(3, color='gray', linewidth=1, label='Normal (3)')
    ax6.axhline(6, color='orange', linestyle=':', linewidth=2, label='Fat Tails (6)')

    ax6.set_ylabel('Kurtosis')
    ax6.set_title('Early Warning: Excess Kurtosis (Fat Tails)', fontweight='bold')
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

    plt.suptitle(f'{crisis_name}\nEarly Warning Signal Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def create_comparison_summary(covid_returns, covid_events, covid_ews,
                               tariff_returns, tariff_events, tariff_ews,
                               save_path=None):
    """Create side-by-side comparison of both crises."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    crises = [
        ('COVID-19 Crash (2020)', covid_returns, covid_events, covid_ews, 'red'),
        ('Tariff Crisis (2025)', tariff_returns, tariff_events, tariff_ews, 'purple')
    ]

    for row, (name, returns, events, ews, color) in enumerate(crises):
        n = len(returns)
        t = np.arange(n)
        crisis_start = events.get('crash_start', events.get('escalation_start', 0))
        crisis_end = events.get('crash_end', n-1)

        # Price trajectory
        ax1 = axes[row, 0]
        cum_returns = np.cumprod(1 + returns) * 100
        ax1.plot(t, cum_returns, 'b-', linewidth=1.5)
        ax1.axvspan(crisis_start, crisis_end, color=color, alpha=0.2)

        peak_idx = np.argmax(cum_returns[:crisis_end+5])
        trough_idx = crisis_start + np.argmin(cum_returns[crisis_start:crisis_end+1])
        drawdown = (cum_returns[trough_idx] - cum_returns[peak_idx]) / cum_returns[peak_idx] * 100

        ax1.set_title(f'{name}\nMax Drawdown: {drawdown:.1f}%', fontweight='bold')
        ax1.set_ylabel('Price Index')
        if row == 1:
            ax1.set_xlabel('Trading Days')

        # Early warning signals
        ax2 = axes[row, 1]
        ax2.plot(t, ews['autocorrelation'], 'b-', linewidth=1.5, label='AC')
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

    plt.suptitle('Crisis Comparison: Early Warning Signals', fontsize=14, fontweight='bold')
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
    print("GENERATING REAL-WORLD CRISIS EXAMPLES")
    print("=" * 70)
    print()

    # Generate COVID-19 crash data
    print("1. COVID-19 Crash (February-April 2020)")
    print("-" * 50)
    covid_returns, covid_dates, covid_events = generate_covid_crash_data()
    covid_ews = compute_early_warning_signals(covid_returns, window=15)

    # Calculate key stats
    crisis_start = covid_events['crash_start']
    crisis_end = covid_events['crash_end']
    crash_returns = covid_returns[crisis_start:crisis_end+1]
    total_crash = (np.prod(1 + crash_returns) - 1) * 100
    max_daily = min(crash_returns) * 100

    print(f"   Period: {covid_dates[0].strftime('%Y-%m-%d')} to {covid_dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Crisis: {covid_dates[crisis_start].strftime('%Y-%m-%d')} to {covid_dates[crisis_end].strftime('%Y-%m-%d')}")
    print(f"   Total crash: {total_crash:.1f}%")
    print(f"   Worst day: {max_daily:.1f}%")
    print()

    # Generate COVID analysis figure
    create_crisis_analysis_figure(
        covid_returns, covid_dates, covid_events, covid_ews,
        'COVID-19 Market Crash (2020)',
        crisis_color='red',
        save_path=OUTPUT_DIR / 'covid_crash_analysis.png'
    )

    # Generate dashboard
    generate_dashboard_for_crisis(
        covid_returns, 'COVID-19',
        save_path=OUTPUT_DIR / 'covid_crash_dashboard.png'
    )

    # Generate Tariff crisis data
    print("2. Tariff Crisis (March-April 2025)")
    print("-" * 50)
    tariff_returns, tariff_dates, tariff_events = generate_tariff_crash_data()
    tariff_ews = compute_early_warning_signals(tariff_returns, window=15)

    crisis_start = tariff_events['crash_start']
    crisis_end = tariff_events['crash_end']
    crash_returns = tariff_returns[crisis_start:crisis_end+1]
    total_crash = (np.prod(1 + crash_returns) - 1) * 100
    max_daily = min(crash_returns) * 100

    print(f"   Period: {tariff_dates[0].strftime('%Y-%m-%d')} to {tariff_dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Crisis: {tariff_dates[crisis_start].strftime('%Y-%m-%d')} to {tariff_dates[crisis_end].strftime('%Y-%m-%d')}")
    print(f"   Total crash: {total_crash:.1f}%")
    print(f"   Worst day: {max_daily:.1f}%")
    print()

    # Generate Tariff analysis figure
    create_crisis_analysis_figure(
        tariff_returns, tariff_dates, tariff_events, tariff_ews,
        'Tariff Crisis (2025)',
        crisis_color='purple',
        save_path=OUTPUT_DIR / 'tariff_crash_analysis.png'
    )

    # Generate dashboard
    generate_dashboard_for_crisis(
        tariff_returns, 'Tariff Crisis',
        save_path=OUTPUT_DIR / 'tariff_crash_dashboard.png'
    )

    # Generate comparison
    print("3. Generating comparison summary...")
    print("-" * 50)
    create_comparison_summary(
        covid_returns, covid_events, covid_ews,
        tariff_returns, tariff_events, tariff_ews,
        save_path=OUTPUT_DIR / 'crisis_comparison.png'
    )

    print()
    print("=" * 70)
    print("CRISIS EXAMPLES GENERATED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
