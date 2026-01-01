"""
Comprehensive Tail Risk Dashboard
==================================

Integrated dashboard combining all visualization components
for complete tail risk analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional, Dict, List
import warnings


class TailRiskDashboard:
    """
    Comprehensive tail risk visualization dashboard.

    Combines:
    - 3D phase space visualization
    - Distribution analysis
    - Risk metrics over time
    - Regime classification
    - Early warning signals
    """

    def __init__(self, returns: np.ndarray, vix: Optional[np.ndarray] = None):
        """
        Initialize dashboard.

        Args:
            returns: Time series of returns
            vix: Optional VIX time series
        """
        self.returns = np.asarray(returns)
        self.vix = vix if vix is not None else self._generate_synthetic_vix()
        self.n = len(returns)

        # Compute all required metrics
        self._compute_metrics()

    def _generate_synthetic_vix(self) -> np.ndarray:
        """Generate synthetic VIX from returns."""
        window = 20
        vol = np.array([np.std(self.returns[max(0, i-window):i+1]) * np.sqrt(252) * 100
                       for i in range(self.n)])
        return vol

    def _compute_metrics(self):
        """Compute all tail risk metrics."""
        from ..physics.levy_flight import levy_flight_3d_coordinates, estimate_tail_index
        from ..physics.tsallis_statistics import compute_tsallis_coordinates
        from ..physics.phase_transitions import compute_phase_space_coordinates
        from ..models.risk_metrics import RollingRiskMetrics
        from ..models.regime_detection import RegimeAwareTailRisk

        # Coordinate systems
        self.levy_coords = levy_flight_3d_coordinates(self.returns)
        self.tsallis_coords = compute_tsallis_coordinates(self.returns)
        self.phase_coords = compute_phase_space_coordinates(self.returns)

        # Rolling risk metrics
        rolling = RollingRiskMetrics(self.returns, window=60)
        self.rolling_metrics = rolling.compute_all()

        # Regime analysis
        try:
            regime_analyzer = RegimeAwareTailRisk(self.returns)
            self.current_regime = regime_analyzer.current_regime()
            self.regime_var = regime_analyzer.regime_conditional_var()
        except:
            self.current_regime = {'simple_regime': 'UNKNOWN'}
            self.regime_var = {}

    def create_full_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive dashboard figure.

        Returns:
            Matplotlib figure with all components
        """
        from scipy import stats

        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # ===== Row 1: 3D Phase Spaces =====

        # 1.1 Lévy Phase Space
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        time_color = np.arange(self.n)
        scatter1 = ax1.scatter(self.levy_coords[0], self.levy_coords[1],
                              self.levy_coords[2], c=time_color, cmap='viridis',
                              s=10, alpha=0.5)
        ax1.set_xlabel('σ', fontsize=9)
        ax1.set_ylabel('α⁻¹', fontsize=9)
        ax1.set_zlabel('λ', fontsize=9)
        ax1.set_title('Lévy Flight Space', fontsize=10, fontweight='bold')

        # 1.2 Tsallis Space
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax2.scatter(self.tsallis_coords[0], self.tsallis_coords[1],
                   self.tsallis_coords[2], c=time_color, cmap='plasma',
                   s=10, alpha=0.5)
        ax2.set_xlabel('q', fontsize=9)
        ax2.set_ylabel('S', fontsize=9)
        ax2.set_zlabel('T', fontsize=9)
        ax2.set_title('Thermodynamic Space', fontsize=10, fontweight='bold')

        # 1.3 Phase Transition Space
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')
        ax3.scatter(self.phase_coords[0], self.phase_coords[1],
                   self.phase_coords[2], c=time_color, cmap='coolwarm',
                   s=10, alpha=0.5)
        ax3.set_xlabel('χ', fontsize=9)
        ax3.set_ylabel('M', fontsize=9)
        ax3.set_zlabel('|T-Tc|', fontsize=9)
        ax3.set_title('Critical Phenomena', fontsize=10, fontweight='bold')

        # 1.4 Composite Risk Surface
        ax4 = fig.add_subplot(gs[0, 3], projection='3d')
        risk_level = (self.levy_coords[2] + 0.5 * self.tsallis_coords[0] +
                     self.phase_coords[0]) / 2.5
        scatter4 = ax4.scatter(self.levy_coords[0], self.tsallis_coords[1],
                              risk_level, c=risk_level, cmap='RdYlGn_r',
                              s=15, alpha=0.6)
        ax4.set_xlabel('Vol', fontsize=9)
        ax4.set_ylabel('Entropy', fontsize=9)
        ax4.set_zlabel('Risk', fontsize=9)
        ax4.set_title('Composite Risk', fontsize=10, fontweight='bold')

        # ===== Row 2: Distributions and Metrics =====

        # 2.1 Return Distribution with Fat Tails
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.hist(self.returns, bins=50, density=True, alpha=0.7,
                color='blue', label='Empirical')
        x = np.linspace(self.returns.min(), self.returns.max(), 200)
        gaussian = stats.norm.pdf(x, np.mean(self.returns), np.std(self.returns))
        ax5.plot(x, gaussian, 'r-', linewidth=2, label='Gaussian')
        ax5.set_xlabel('Returns', fontsize=9)
        ax5.set_ylabel('Density', fontsize=9)
        ax5.set_title('Return Distribution', fontsize=10, fontweight='bold')
        ax5.legend(fontsize=8)

        # 2.2 Log-Scale Tails
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.hist(self.returns, bins=50, density=True, alpha=0.7,
                color='blue', log=True)
        ax6.plot(x, gaussian, 'r-', linewidth=2)
        ax6.set_xlabel('Returns', fontsize=9)
        ax6.set_ylabel('Log Density', fontsize=9)
        ax6.set_title('Fat Tails (Log Scale)', fontsize=10, fontweight='bold')

        # 2.3 QQ Plot
        ax7 = fig.add_subplot(gs[1, 2])
        sorted_returns = np.sort(self.returns)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(self.returns)))
        ax7.scatter(theoretical_quantiles, sorted_returns, s=5, alpha=0.5)
        ax7.plot([-4, 4], [np.mean(self.returns) - 4*np.std(self.returns),
                          np.mean(self.returns) + 4*np.std(self.returns)],
                'r-', linewidth=2, label='Normal')
        ax7.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax7.set_ylabel('Sample Quantiles', fontsize=9)
        ax7.set_title('Q-Q Plot (vs Normal)', fontsize=10, fontweight='bold')
        ax7.legend(fontsize=8)

        # 2.4 Tail Index Evolution
        ax8 = fig.add_subplot(gs[1, 3])
        tail_index = self.rolling_metrics.get('tail_index', np.ones(self.n) * 2)
        ax8.plot(tail_index, 'b-', linewidth=1)
        ax8.axhline(y=2, color='r', linestyle='--', label='Gaussian (α=2)')
        ax8.axhline(y=3, color='orange', linestyle='--', label='Cubic law (α=3)')
        ax8.set_xlabel('Time', fontsize=9)
        ax8.set_ylabel('Tail Index (α)', fontsize=9)
        ax8.set_title('Tail Index Evolution', fontsize=10, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.set_ylim(0, 5)

        # ===== Row 3: Time Series =====

        # 3.1 Returns and Volatility
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.plot(self.returns, 'b-', linewidth=0.5, alpha=0.7, label='Returns')
        ax9_twin = ax9.twinx()
        vol = np.array([np.std(self.returns[max(0, i-20):i+1])
                       for i in range(self.n)])
        ax9_twin.plot(vol, 'r-', linewidth=1, alpha=0.7, label='Volatility')
        ax9.set_xlabel('Time', fontsize=9)
        ax9.set_ylabel('Returns', fontsize=9, color='blue')
        ax9_twin.set_ylabel('Volatility', fontsize=9, color='red')
        ax9.set_title('Returns and Volatility', fontsize=10, fontweight='bold')

        # 3.2 VaR and ES
        ax10 = fig.add_subplot(gs[2, 2:])
        var_99 = self.rolling_metrics.get('var_99', np.zeros(self.n))
        es_99 = self.rolling_metrics.get('es_99', np.zeros(self.n))
        ax10.fill_between(range(self.n), var_99, es_99, alpha=0.3, color='red',
                         label='ES - VaR Gap')
        ax10.plot(var_99, 'b-', linewidth=1, label='VaR 99%')
        ax10.plot(es_99, 'r-', linewidth=1, label='ES 99%')
        ax10.set_xlabel('Time', fontsize=9)
        ax10.set_ylabel('Risk', fontsize=9)
        ax10.set_title('VaR and Expected Shortfall', fontsize=10, fontweight='bold')
        ax10.legend(fontsize=8)

        # ===== Row 4: Risk Indicators =====

        # 4.1 Kurtosis and Skewness
        ax11 = fig.add_subplot(gs[3, 0])
        kurt = self.rolling_metrics.get('kurtosis', np.zeros(self.n))
        skew = self.rolling_metrics.get('skewness', np.zeros(self.n))
        ax11.plot(kurt, 'b-', linewidth=1, label='Kurtosis')
        ax11.plot(skew, 'g-', linewidth=1, label='Skewness')
        ax11.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax11.set_xlabel('Time', fontsize=9)
        ax11.set_ylabel('Value', fontsize=9)
        ax11.set_title('Higher Moments', fontsize=10, fontweight='bold')
        ax11.legend(fontsize=8)

        # 4.2 Composite Risk Indicator
        ax12 = fig.add_subplot(gs[3, 1])
        composite = (risk_level - np.nanmin(risk_level)) / (
            np.nanmax(risk_level) - np.nanmin(risk_level) + 1e-10) * 100
        ax12.fill_between(range(self.n), composite, alpha=0.5, color='red')
        ax12.plot(composite, 'r-', linewidth=1)
        ax12.axhline(y=75, color='darkred', linestyle='--', label='Extreme')
        ax12.axhline(y=50, color='orange', linestyle='--', label='Elevated')
        ax12.set_xlabel('Time', fontsize=9)
        ax12.set_ylabel('Risk Level (%)', fontsize=9)
        ax12.set_title('Composite Risk Index', fontsize=10, fontweight='bold')
        ax12.legend(fontsize=8)
        ax12.set_ylim(0, 100)

        # 4.3 Current State Summary
        ax13 = fig.add_subplot(gs[3, 2])
        ax13.axis('off')

        # Compute current metrics
        current_var = var_99[-1] if len(var_99) > 0 and not np.isnan(var_99[-1]) else 0
        current_es = es_99[-1] if len(es_99) > 0 and not np.isnan(es_99[-1]) else 0
        current_kurt = kurt[-1] if len(kurt) > 0 and not np.isnan(kurt[-1]) else 0
        current_alpha = tail_index[-1] if len(tail_index) > 0 and not np.isnan(tail_index[-1]) else 2
        current_risk = composite[-1] if len(composite) > 0 else 0

        # Risk level classification
        if current_risk > 75:
            risk_class = 'EXTREME'
            risk_color = 'darkred'
        elif current_risk > 50:
            risk_class = 'ELEVATED'
            risk_color = 'orange'
        elif current_risk > 25:
            risk_class = 'MODERATE'
            risk_color = 'gold'
        else:
            risk_class = 'LOW'
            risk_color = 'green'

        summary_text = f"""
CURRENT TAIL RISK SUMMARY
═══════════════════════════

Risk Level: {risk_class}
Risk Score: {current_risk:.1f}%

Key Metrics:
• VaR (99%): {current_var*100:.2f}%
• ES (99%): {current_es*100:.2f}%
• Kurtosis: {current_kurt:.2f}
• Tail Index: {current_alpha:.2f}

Regime: {self.current_regime.get('simple_regime', 'UNKNOWN')}

Interpretation:
{'⚠️ Fat tails detected!' if current_alpha < 3 else '✓ Tails within normal range'}
{'⚠️ High volatility regime!' if current_var > 0.02 else '✓ Normal volatility'}
        """

        ax13.text(0.1, 0.9, summary_text, transform=ax13.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4.4 Tail Probability Comparison
        ax14 = fig.add_subplot(gs[3, 3])
        sigmas = [2, 3, 4, 5]
        gaussian_probs = [2 * (1 - stats.norm.cdf(s)) for s in sigmas]
        empirical_probs = [np.mean(np.abs(self.returns) > s * np.std(self.returns))
                          for s in sigmas]

        x = np.arange(len(sigmas))
        width = 0.35

        bars1 = ax14.bar(x - width/2, gaussian_probs, width, label='Gaussian',
                        color='blue', alpha=0.7)
        bars2 = ax14.bar(x + width/2, empirical_probs, width, label='Empirical',
                        color='red', alpha=0.7)

        ax14.set_xlabel('σ Threshold', fontsize=9)
        ax14.set_ylabel('Probability', fontsize=9)
        ax14.set_title('Tail Probability: Gaussian vs Empirical', fontsize=10, fontweight='bold')
        ax14.set_xticks(x)
        ax14.set_xticklabels([f'{s}σ' for s in sigmas])
        ax14.legend(fontsize=8)
        ax14.set_yscale('log')

        # Main title
        fig.suptitle('COMPREHENSIVE TAIL RISK DASHBOARD\n'
                    'Physics-Based Fat Tail Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Dashboard saved to {save_path}")

        return fig


def create_summary_report(returns: np.ndarray) -> str:
    """
    Generate text summary of tail risk analysis.

    Returns:
        Formatted string with analysis results
    """
    from scipy import stats
    from ..physics.levy_flight import estimate_tail_index
    from ..models.risk_metrics import TailRiskMetrics

    metrics = TailRiskMetrics(returns)

    # Basic statistics
    mean = np.mean(returns)
    std = np.std(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    # Tail analysis
    try:
        tail_index = estimate_tail_index(returns)
    except:
        tail_index = 2.0

    # VaR and ES
    var_99 = metrics.var_historical(0.99)
    es_99 = metrics.expected_shortfall(0.99)

    # Tail ratios
    tail_ratio = metrics.tail_ratio(3)

    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           TAIL RISK ANALYSIS REPORT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BASIC STATISTICS                                                            ║
║  ───────────────                                                             ║
║  Mean Return:        {mean*100:>10.4f}%                                      ║
║  Standard Deviation: {std*100:>10.4f}%                                       ║
║  Skewness:           {skewness:>10.4f}                                       ║
║  Excess Kurtosis:    {kurtosis:>10.4f}                                       ║
║                                                                              ║
║  TAIL RISK METRICS                                                           ║
║  ─────────────────                                                           ║
║  Tail Index (α):     {tail_index:>10.4f}  {'(Fat tails!)' if tail_index < 3 else '(Normal)':>15} ║
║  VaR (99%):          {var_99*100:>10.4f}%                                    ║
║  ES (99%):           {es_99*100:>10.4f}%                                     ║
║  ES/VaR Ratio:       {es_99/var_99:>10.4f}  {'(High tail risk)' if es_99/var_99 > 1.5 else '':>15} ║
║                                                                              ║
║  TAIL PROBABILITY COMPARISON (3σ)                                            ║
║  ────────────────────────────────                                            ║
║  Gaussian Probability:   {tail_ratio['gaussian_tail_prob']*100:>8.4f}%                           ║
║  Empirical Probability:  {(tail_ratio['left_tail_prob'] + tail_ratio['right_tail_prob'])*100:>8.4f}%                           ║
║  Fat Tail Ratio:         {tail_ratio['left_ratio']:>8.2f}x                                    ║
║                                                                              ║
║  INTERPRETATION                                                              ║
║  ──────────────                                                              ║
║  {'⚠️  Distribution has significant fat tails' if kurtosis > 3 else '✓  Distribution close to normal':<74} ║
║  {'⚠️  Negative skewness indicates crash risk' if skewness < -0.5 else '✓  Symmetric distribution':<74} ║
║  {'⚠️  Tail events are significantly more likely than Gaussian predicts' if tail_ratio['left_ratio'] > 2 else '✓  Tail probabilities close to Gaussian':<74} ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

    return report
