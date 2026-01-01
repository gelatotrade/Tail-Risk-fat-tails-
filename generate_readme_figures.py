#!/usr/bin/env python3
"""
Generate Educational Figures for README
========================================

Creates clear, educational visualizations that explain tail risk concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_fat_tails_comparison():
    """
    Generate comparison of Gaussian vs Fat-Tailed distributions.
    Shows why normal distribution fails for market returns.
    """
    from src.utils.data_loader import generate_synthetic_returns
    from src.physics.levy_flight import LevyStableDistribution

    # Generate fat-tailed returns
    np.random.seed(42)
    returns = generate_synthetic_returns(5000, 'levy_jump', seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ===== Plot 1: Linear Scale Histogram =====
    ax1 = axes[0]

    # Histogram of returns
    counts, bins, _ = ax1.hist(returns, bins=80, density=True, alpha=0.7,
                                color='steelblue', edgecolor='white', label='Market Returns')

    # Fit Gaussian
    mu, sigma = np.mean(returns), np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 200)
    gaussian_pdf = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, gaussian_pdf, 'r-', linewidth=2.5, label=f'Gaussian (σ={sigma:.3f})')

    # Add sigma markers
    for n_sigma in [2, 3]:
        ax1.axvline(mu - n_sigma*sigma, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(mu + n_sigma*sigma, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Return')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Distribution Comparison (Linear Scale)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.08, 0.08)

    # ===== Plot 2: Log Scale (Fat Tails Visible) =====
    ax2 = axes[1]

    # Log-scale histogram
    ax2.hist(returns, bins=80, density=True, alpha=0.7,
             color='steelblue', edgecolor='white', label='Market Returns')
    ax2.plot(x, gaussian_pdf, 'r-', linewidth=2.5, label='Gaussian')

    ax2.set_yscale('log')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Probability Density (log scale)')
    ax2.set_title('Fat Tails Revealed (Log Scale)')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-0.08, 0.08)
    ax2.set_ylim(1e-3, 100)

    # Annotate the fat tail region
    ax2.annotate('Fat Tails:\nMore extreme\nevents than\nGaussian predicts',
                 xy=(-0.055, 0.1), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ===== Plot 3: Tail Probability Comparison =====
    ax3 = axes[2]

    sigma_levels = [2, 3, 4, 5]
    gaussian_probs = []
    empirical_probs = []

    for n in sigma_levels:
        # Gaussian probability of exceeding n sigma
        gauss_prob = 2 * (1 - stats.norm.cdf(n)) * 100
        gaussian_probs.append(gauss_prob)

        # Empirical probability
        threshold = n * sigma
        emp_prob = (np.sum(np.abs(returns - mu) > threshold) / len(returns)) * 100
        empirical_probs.append(emp_prob)

    x_pos = np.arange(len(sigma_levels))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, gaussian_probs, width, label='Gaussian Prediction',
                    color='lightcoral', edgecolor='darkred', linewidth=1.5)
    bars2 = ax3.bar(x_pos + width/2, empirical_probs, width, label='Actual (Empirical)',
                    color='steelblue', edgecolor='navy', linewidth=1.5)

    ax3.set_yscale('log')
    ax3.set_xlabel('Event Size (σ)')
    ax3.set_ylabel('Probability (%)')
    ax3.set_title('Tail Event Probability: Gaussian vs Reality')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{n}σ' for n in sigma_levels])
    ax3.legend()

    # Add ratio annotations
    for i, (g, e) in enumerate(zip(gaussian_probs, empirical_probs)):
        if e > 0 and g > 0:
            ratio = e / g
            ax3.annotate(f'{ratio:.0f}x', xy=(x_pos[i], max(g, e) * 1.5),
                        ha='center', fontsize=10, fontweight='bold', color='darkgreen')

    ax3.set_ylim(1e-4, 20)

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'fat_tails_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_early_warning_signals():
    """
    Generate visualization showing early warning signals before a crash.
    Demonstrates critical slowing down concept.
    """
    from src.utils.data_loader import generate_crisis_scenario
    from src.physics.phase_transitions import CriticalSlowingDownDetector

    # Generate crisis scenario
    returns, meta = generate_crisis_scenario(seed=123)
    crisis_start = meta['crisis_start']
    crisis_end = meta['crisis_end']

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    t = np.arange(len(returns))

    # ===== Plot 1: Returns with Crisis Period =====
    ax1 = axes[0]
    ax1.plot(t, returns * 100, 'k-', linewidth=0.8, alpha=0.8)
    ax1.axvspan(crisis_start, crisis_end, color='red', alpha=0.2, label='Crisis Period')
    ax1.axvline(crisis_start, color='red', linestyle='--', linewidth=2, label='Crisis Start')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Market Returns with Crisis Period', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-15, 10)

    # ===== Compute EWS =====
    detector = CriticalSlowingDownDetector(window=50)

    # Rolling autocorrelation (lag-1)
    window = 50
    ac1 = np.array([np.corrcoef(returns[max(0,i-window):i],
                                returns[max(1,i-window+1):i+1])[0,1]
                   if i >= window else np.nan
                   for i in range(len(returns))])

    # Rolling variance
    rolling_var = np.array([np.var(returns[max(0,i-window):i+1])
                           if i >= window else np.nan
                           for i in range(len(returns))])

    # Rolling skewness
    rolling_skew = np.array([stats.skew(returns[max(0,i-window):i+1])
                            if i >= window else np.nan
                            for i in range(len(returns))])

    # ===== Plot 2: Autocorrelation (Critical Slowing Down) =====
    ax2 = axes[1]
    ax2.plot(t, ac1, 'b-', linewidth=1.5, label='Lag-1 Autocorrelation')
    ax2.axvspan(crisis_start, crisis_end, color='red', alpha=0.2)
    ax2.axvline(crisis_start, color='red', linestyle='--', linewidth=2)
    ax2.axhline(0.5, color='orange', linestyle=':', linewidth=2, label='Warning Threshold')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Early Warning: Autocorrelation (Critical Slowing Down)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.5, 1.0)

    # ===== Plot 3: Rolling Variance =====
    ax3 = axes[2]
    ax3.plot(t, rolling_var * 10000, 'g-', linewidth=1.5, label='Rolling Variance')
    ax3.axvspan(crisis_start, crisis_end, color='red', alpha=0.2)
    ax3.axvline(crisis_start, color='red', linestyle='--', linewidth=2)
    ax3.set_ylabel('Variance (bps²)')
    ax3.set_title('Early Warning: Variance Increase', fontweight='bold')
    ax3.legend(loc='upper right')

    # ===== Plot 4: Rolling Skewness =====
    ax4 = axes[3]
    ax4.plot(t, rolling_skew, 'm-', linewidth=1.5, label='Rolling Skewness')
    ax4.axvspan(crisis_start, crisis_end, color='red', alpha=0.2)
    ax4.axvline(crisis_start, color='red', linestyle='--', linewidth=2)
    ax4.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax4.set_ylabel('Skewness')
    ax4.set_xlabel('Time (days)')
    ax4.set_title('Early Warning: Negative Skewness', fontweight='bold')
    ax4.legend(loc='upper right')

    # Add annotation box
    textstr = 'Before Crisis:\n• Autocorrelation ↑\n• Variance ↑\n• Skewness ↓'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    ax1.text(crisis_start - 120, 7, textstr, fontsize=11, verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'early_warning_signals.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_var_comparison():
    """
    Generate VaR comparison showing underestimation by Gaussian methods.
    """
    from src.utils.data_loader import generate_synthetic_returns
    from src.models.risk_metrics import TailRiskMetrics
    from src.models.extreme_value import EVTTailRiskAnalyzer

    # Generate data
    returns = generate_synthetic_returns(2000, 'levy_jump', seed=42)

    metrics = TailRiskMetrics(returns)
    evt = EVTTailRiskAnalyzer(returns)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Plot 1: VaR at Different Confidence Levels =====
    ax1 = axes[0]

    confidence_levels = [0.90, 0.95, 0.99, 0.995]

    var_gaussian = [abs(metrics.var_parametric_gaussian(cl)) * 100 for cl in confidence_levels]
    var_historical = [abs(metrics.var_historical(cl)) * 100 for cl in confidence_levels]
    var_cf = [abs(metrics.var_cornish_fisher(cl)) * 100 for cl in confidence_levels]

    x_pos = np.arange(len(confidence_levels))
    width = 0.25

    bars1 = ax1.bar(x_pos - width, var_gaussian, width, label='Gaussian VaR',
                    color='lightcoral', edgecolor='darkred')
    bars2 = ax1.bar(x_pos, var_historical, width, label='Historical VaR',
                    color='steelblue', edgecolor='navy')
    bars3 = ax1.bar(x_pos + width, var_cf, width, label='Cornish-Fisher VaR',
                    color='lightgreen', edgecolor='darkgreen')

    ax1.set_xlabel('Confidence Level')
    ax1.set_ylabel('VaR (% loss)')
    ax1.set_title('Value at Risk: Gaussian vs Fat-Tail Methods', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
    ax1.legend()

    # Annotate underestimation
    for i, (g, h) in enumerate(zip(var_gaussian, var_historical)):
        if h > g:
            pct = ((h - g) / g) * 100
            ax1.annotate(f'+{pct:.0f}%', xy=(x_pos[i], h + 0.1),
                        ha='center', fontsize=9, fontweight='bold', color='darkred')

    # ===== Plot 2: Expected Shortfall vs VaR =====
    ax2 = axes[1]

    cl = 0.99
    var_99 = abs(metrics.var_historical(cl)) * 100
    es_99 = abs(metrics.expected_shortfall(cl)) * 100

    # Also get return levels from EVT
    rl_10y = abs(evt.return_level(252*10)) * 100
    rl_100y = abs(evt.return_level(252*100)) * 100

    categories = ['VaR 99%', 'ES 99%', '10-Year\nReturn Level', '100-Year\nReturn Level']
    values = [var_99, es_99, rl_10y, rl_100y]
    colors = ['steelblue', 'coral', 'purple', 'darkred']

    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Expected Loss (%)')
    ax2.set_title('Risk Metrics Comparison', fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Add annotation
    ax2.annotate('ES captures\naverage loss\nbeyond VaR',
                 xy=(1, es_99), xytext=(1.5, es_99 + 2),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'var_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_tail_index_interpretation():
    """
    Generate visualization explaining tail index (alpha) interpretation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Plot 1: Different Alpha Values =====
    ax1 = axes[0]

    x = np.linspace(-5, 5, 1000)

    # Different distributions with varying tail heaviness
    # Simulate different alpha values using t-distribution
    for df, alpha_equiv, color, label in [
        (30, '~∞', 'green', 'α ≈ ∞ (Gaussian)'),
        (5, '~5', 'blue', 'α ≈ 5 (Moderate tails)'),
        (3, '~3', 'orange', 'α ≈ 3 (Fat tails)'),
        (1.5, '~1.5', 'red', 'α ≈ 1.5 (Very fat tails)')
    ]:
        pdf = stats.t.pdf(x, df)
        ax1.plot(x, pdf, color=color, linewidth=2, label=label)

    ax1.set_xlabel('Standardized Return')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Tail Index (α) Effect on Distribution Shape', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 0.45)

    # ===== Plot 2: Alpha Interpretation Scale =====
    ax2 = axes[1]

    # Create a horizontal bar showing alpha ranges
    alpha_ranges = [
        (0, 1, 'Infinite Mean & Variance', 'darkred'),
        (1, 2, 'Infinite Variance', 'red'),
        (2, 3, 'Fat Tails (Finite Variance)', 'orange'),
        (3, 4, 'Moderate Fat Tails', 'gold'),
        (4, 5, 'Near-Gaussian', 'lightgreen'),
        (5, 6, 'Gaussian-like', 'green')
    ]

    for i, (start, end, label, color) in enumerate(alpha_ranges):
        ax2.barh(0, end - start, left=start, height=0.5, color=color,
                edgecolor='black', linewidth=1)
        ax2.text((start + end) / 2, 0.35, label, ha='center', va='bottom',
                fontsize=9, rotation=0)

    # Market typical range
    ax2.axvline(1.7, color='blue', linestyle='--', linewidth=2, label='Typical stocks (α ≈ 1.7)')
    ax2.axvline(2.0, color='purple', linestyle=':', linewidth=2, label='Gaussian (α = 2)')

    ax2.set_xlim(0, 6)
    ax2.set_ylim(-0.5, 1)
    ax2.set_xlabel('Tail Index (α)')
    ax2.set_title('Tail Index Interpretation Guide', fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right')

    # Add key insight box
    textstr = 'Key Insight:\nStock returns typically have α ≈ 1.5-2.0\nThis means infinite variance and\nextreme events occur frequently!'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    ax2.text(3.5, -0.35, textstr, fontsize=10, bbox=props, ha='center')

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'tail_index_interpretation.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_physics_mapping_diagram():
    """
    Generate a visual diagram showing physics to finance mapping.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Physics → Finance: Conceptual Mapping',
            fontsize=18, fontweight='bold', ha='center')

    # Physics side (left)
    physics_items = [
        ('Lévy Flights', 'Particle diffusion\nwith large jumps'),
        ('Fokker-Planck', 'Probability density\nevolution'),
        ('Tsallis Entropy', 'Non-extensive\nthermodynamics'),
        ('Phase Transitions', 'Critical phenomena\n& symmetry breaking')
    ]

    # Finance side (right)
    finance_items = [
        ('Price Jumps', 'Black Swan events\nmarket crashes'),
        ('Risk Evolution', 'VaR/ES dynamics\nover time'),
        ('Fat Tails', 'Extreme events\nheavy tails'),
        ('Market Crashes', 'Regime shifts\nherding behavior')
    ]

    # Draw boxes and arrows
    left_x, right_x = 1.5, 8.5
    box_width, box_height = 2.5, 1.2

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, ((p_title, p_desc), (f_title, f_desc), color) in enumerate(
            zip(physics_items, finance_items, colors)):
        y = 5.5 - i * 1.5

        # Physics box
        rect1 = plt.Rectangle((left_x - box_width/2, y - box_height/2),
                              box_width, box_height,
                              facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect1)
        ax.text(left_x, y + 0.2, p_title, ha='center', fontsize=11, fontweight='bold')
        ax.text(left_x, y - 0.25, p_desc, ha='center', fontsize=9, color='gray')

        # Finance box
        rect2 = plt.Rectangle((right_x - box_width/2, y - box_height/2),
                              box_width, box_height,
                              facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect2)
        ax.text(right_x, y + 0.2, f_title, ha='center', fontsize=11, fontweight='bold')
        ax.text(right_x, y - 0.25, f_desc, ha='center', fontsize=9, color='gray')

        # Arrow
        ax.annotate('', xy=(right_x - box_width/2 - 0.1, y),
                   xytext=(left_x + box_width/2 + 0.1, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Labels
    ax.text(left_x, 6.8, 'PHYSICS', fontsize=14, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(right_x, 6.8, 'FINANCE', fontsize=14, fontweight='bold', ha='center', color='#2c3e50')

    # Central insight
    ax.text(5, 0.5, '"Markets are complex systems—physics provides the mathematical tools\nto model their non-Gaussian, extreme behavior."',
            fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_path = OUTPUT_DIR / 'physics_finance_mapping.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all README figures."""
    print("=" * 60)
    print("Generating Educational Figures for README")
    print("=" * 60)
    print()

    print("1. Fat Tails vs Gaussian Comparison...")
    generate_fat_tails_comparison()

    print("2. Early Warning Signals...")
    generate_early_warning_signals()

    print("3. VaR Comparison Chart...")
    generate_var_comparison()

    print("4. Tail Index Interpretation...")
    generate_tail_index_interpretation()

    print("5. Physics-Finance Mapping Diagram...")
    generate_physics_mapping_diagram()

    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
