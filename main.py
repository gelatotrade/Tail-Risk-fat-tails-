#!/usr/bin/env python3
"""
Tail Risk Modeling Framework - Main Entry Point
=================================================

This script demonstrates the complete tail risk analysis pipeline
using physics-based models for fat-tail distributions.

Usage:
    python main.py [--mode MODE] [--output OUTPUT_DIR]

Modes:
    - demo: Run full demonstration with synthetic data
    - analyze: Analyze provided data file
    - visualize: Generate all visualizations
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.utils.data_loader import (
    generate_synthetic_returns,
    generate_crisis_scenario,
    load_sample_data
)
from src.physics.levy_flight import (
    LevyStableDistribution,
    LevyFlightProcess,
    levy_flight_3d_coordinates,
    estimate_tail_index
)
from src.physics.tsallis_statistics import (
    TsallisDistribution,
    TsallisTailRiskModel,
    compute_tsallis_coordinates
)
from src.physics.fokker_planck import (
    FokkerPlanckSolver,
    compute_fokker_planck_coordinates
)
from src.physics.phase_transitions import (
    CriticalSlowingDownDetector,
    MarketPhaseClassifier,
    compute_phase_space_coordinates
)
from src.models.risk_metrics import TailRiskMetrics, RollingRiskMetrics
from src.models.extreme_value import EVTTailRiskAnalyzer
from src.visualization.dashboard import TailRiskDashboard, create_summary_report
from src.visualization.risk_surface_3d import create_comprehensive_3d_dashboard


def run_demo(output_dir: Path):
    """Run full demonstration with synthetic data."""
    print("=" * 70)
    print("TAIL RISK MODELING FRAMEWORK - DEMONSTRATION")
    print("Physics-Based Fat Tail Analysis")
    print("=" * 70)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("1. Generating synthetic market data...")
    print("-" * 40)

    # Normal market
    returns_normal = generate_synthetic_returns(1000, 'levy_jump', seed=42)
    print(f"   Generated 1000 days of normal market data")

    # Crisis scenario
    returns_crisis, crisis_meta = generate_crisis_scenario(seed=43)
    print(f"   Generated crisis scenario data ({len(returns_crisis)} days)")
    print(f"   Crisis period: day {crisis_meta['crisis_start']} to {crisis_meta['crisis_end']}")
    print()

    # 2. Fit Physics-Based Distributions
    print("2. Fitting physics-based distributions...")
    print("-" * 40)

    # Lévy Stable
    levy = LevyStableDistribution.fit(returns_normal)
    print(f"   Lévy Stable: α={levy.alpha:.3f}, β={levy.beta:.3f}")
    print(f"   (α < 2 indicates fat tails; α ≈ 1.7 typical for stocks)")

    # Tsallis
    tsallis = TsallisDistribution.fit(returns_normal)
    print(f"   Tsallis q-Gaussian: q={tsallis.q:.3f}")
    print(f"   (q > 1 indicates non-extensive system with fat tails)")

    # Tail index
    alpha = estimate_tail_index(returns_normal)
    print(f"   Hill tail index: α={alpha:.3f}")
    print(f"   (α < 3 means variance dominates by extremes)")
    print()

    # 3. Compute Risk Metrics
    print("3. Computing tail risk metrics...")
    print("-" * 40)

    metrics = TailRiskMetrics(returns_normal)

    print(f"   VaR (99%, Historical):      {metrics.var_historical(0.99)*100:.4f}%")
    print(f"   VaR (99%, Gaussian):        {metrics.var_parametric_gaussian(0.99)*100:.4f}%")
    print(f"   VaR (99%, Cornish-Fisher):  {metrics.var_cornish_fisher(0.99)*100:.4f}%")
    print(f"   Expected Shortfall (99%):   {metrics.expected_shortfall(0.99)*100:.4f}%")

    tail_ratio = metrics.tail_ratio(3)
    print(f"\n   3-sigma tail probability:")
    print(f"   - Gaussian predicts: {tail_ratio['gaussian_tail_prob']*100:.6f}%")
    print(f"   - Empirical:         {(tail_ratio['left_tail_prob']+tail_ratio['right_tail_prob'])*100:.6f}%")
    print(f"   - Fat tail ratio:    {tail_ratio['left_ratio']:.2f}x more frequent!")
    print()

    # 4. Extreme Value Theory Analysis
    print("4. Extreme Value Theory analysis...")
    print("-" * 40)

    evt = EVTTailRiskAnalyzer(returns_normal)
    print(f"   Left tail index (ξ):  {evt.tail_index('left'):.3f}")
    print(f"   Right tail index (ξ): {evt.tail_index('right'):.3f}")
    print(f"   (ξ > 0 confirms fat tails / Fréchet domain)")

    print(f"\n   Return levels (expected extreme losses):")
    print(f"   - 10-year event:  {evt.return_level(252*10)*100:.2f}%")
    print(f"   - 100-year event: {evt.return_level(252*100)*100:.2f}%")
    print()

    # 5. Phase Transition Analysis
    print("5. Phase transition analysis (crisis detection)...")
    print("-" * 40)

    ews_detector = CriticalSlowingDownDetector()
    ews = ews_detector.compute_ews(returns_crisis)

    print(f"   Early Warning Signals for crisis data:")
    print(f"   - Autocorrelation trend (τ): {ews['ac1_trend']['tau']:.3f}")
    print(f"   - Variance trend (τ):        {ews['var_trend']['tau']:.3f}")
    print(f"   - Warning level:             {ews['warning_level']}")

    classifier = MarketPhaseClassifier()
    phase = classifier.classify(returns_crisis)
    print(f"\n   Current market phase: {phase['phase']}")
    print(f"   Confidence: {phase['confidence']:.2f}")
    print()

    # 6. Generate 3D Visualizations
    print("6. Generating 3D phase space visualizations...")
    print("-" * 40)

    # Compute all coordinate systems
    levy_coords = levy_flight_3d_coordinates(returns_normal)
    tsallis_coords = compute_tsallis_coordinates(returns_normal)
    phase_coords = compute_phase_space_coordinates(returns_normal)

    print("   Computed phase space coordinates:")
    print("   - Lévy flight space (volatility, tail index, jump intensity)")
    print("   - Tsallis thermodynamic space (q, entropy, temperature)")
    print("   - Critical phenomena space (susceptibility, order, criticality)")

    # Create comprehensive dashboard
    print("\n   Creating comprehensive 3D dashboard...")
    fig = create_comprehensive_3d_dashboard(returns_normal,
                                           save_path=output_dir / 'tail_risk_3d_dashboard.png')
    plt.close(fig)
    print(f"   Saved: {output_dir / 'tail_risk_3d_dashboard.png'}")

    # 7. Full Dashboard
    print("\n7. Generating full analysis dashboard...")
    print("-" * 40)

    dashboard = TailRiskDashboard(returns_normal)
    fig = dashboard.create_full_dashboard(save_path=output_dir / 'full_dashboard.png')
    plt.close(fig)
    print(f"   Saved: {output_dir / 'full_dashboard.png'}")

    # Crisis dashboard
    dashboard_crisis = TailRiskDashboard(returns_crisis)
    fig = dashboard_crisis.create_full_dashboard(save_path=output_dir / 'crisis_dashboard.png')
    plt.close(fig)
    print(f"   Saved: {output_dir / 'crisis_dashboard.png'}")

    # 8. Print Summary Report
    print("\n8. Summary Report")
    print("-" * 40)
    report = create_summary_report(returns_normal)
    print(report)

    # Save report
    with open(output_dir / 'tail_risk_report.txt', 'w') as f:
        f.write(report)
    print(f"\n   Report saved: {output_dir / 'tail_risk_report.txt'}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)


def analyze_data(data_path: str, output_dir: Path):
    """Analyze provided data file."""
    print(f"Loading data from {data_path}...")

    # Load data (assuming CSV with 'returns' column or single column)
    import pandas as pd
    try:
        df = pd.read_csv(data_path)
        if 'returns' in df.columns:
            returns = df['returns'].values
        elif 'close' in df.columns.str.lower():
            prices = df[df.columns[df.columns.str.lower() == 'close'][0]].values
            returns = np.diff(np.log(prices))
        else:
            returns = df.iloc[:, 0].values
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded {len(returns)} observations")

    # Run analysis
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard = TailRiskDashboard(returns)
    fig = dashboard.create_full_dashboard(save_path=output_dir / 'analysis_dashboard.png')
    plt.close(fig)

    report = create_summary_report(returns)
    print(report)

    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Tail Risk Modeling Framework')
    parser.add_argument('--mode', choices=['demo', 'analyze', 'visualize'],
                       default='demo', help='Operation mode')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--data', type=str, help='Data file path (for analyze mode)')

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.mode == 'demo':
        run_demo(output_dir)
    elif args.mode == 'analyze':
        if not args.data:
            print("Error: --data required for analyze mode")
            return
        analyze_data(args.data, output_dir)
    elif args.mode == 'visualize':
        run_demo(output_dir)


if __name__ == '__main__':
    main()
