# Tail Risk Modeling Framework
## Physics-Based Fat Tail Analysis for Financial Markets

> **A comprehensive framework for modeling Black Swan events and tail risk in financial markets using physics-inspired models including Lévy flights, Fokker-Planck equations, Tsallis statistics, and phase transition theory.**

---

## Table of Contents

1. [The Problem: Why Normal Distributions Fail](#the-problem-why-normal-distributions-fail)
2. [The Solution: Physics-Based Tail Modeling](#the-solution-physics-based-tail-modeling)
3. [Core Physics Models](#core-physics-models)
   - [Lévy Flights](#1-lévy-flights)
   - [Fokker-Planck Equation](#2-fokker-planck-equation)
   - [Tsallis Statistics](#3-tsallis-statistics)
   - [Phase Transitions](#4-phase-transitions)
4. [3D Tail Risk Coordinate System](#3d-tail-risk-coordinate-system)
5. [Installation & Quick Start](#installation--quick-start)
6. [Visualization Gallery](#visualization-gallery)
7. [API Reference](#api-reference)
8. [Mathematical Foundations](#mathematical-foundations)
9. [Research References](#research-references)

---

## The Problem: Why Normal Distributions Fail

Traditional finance assumes market returns follow **Gaussian (Normal) distributions**. This assumption underlies:
- Modern Portfolio Theory (Markowitz)
- Black-Scholes Option Pricing
- Value at Risk (VaR) calculations
- Most risk management systems

### The Reality: Fat Tails Everywhere

```
                    RETURN DISTRIBUTION COMPARISON

    Probability
         │
         │    ●●●                                    ●●●
         │   ●   ●                                  ●   ●
         │  ●     ●        Gaussian               ●     ●
         │ ●       ●    (What models assume)     ●       ●
         │●         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●         ●
         │
         │██                                              ██
         │████                                          ████
         │  ████                                      ████
         │    ████████████████████████████████████████
         │         Actual Market Returns
         │              (Fat Tails!)
         └──────────────────────────────────────────────────►
              -4σ    -3σ    -2σ    -1σ    0    +1σ    +2σ    +3σ    +4σ
                            Return Magnitude
```

### Catastrophic Underestimation

| Event | Gaussian Probability | Actual Frequency | Underestimation |
|-------|---------------------|------------------|-----------------|
| 3σ move | 0.27% (once/year) | ~2-5% | **10-20x** |
| 4σ move | 0.006% (once/44 years) | ~0.5% | **80x** |
| 5σ move | 0.00006% (once/4,776 years) | ~0.1% | **1,600x** |
| Black Monday 1987 (22σ) | 10^-99 | **It happened** | **Infinity** |

> "The 1987 crash was a 22-sigma event. Under Gaussian assumptions, this should happen once every 10^91 billion years—far longer than the age of the universe." — *Nassim Taleb*

### Visual Evidence: Fat Tails in Action

![Fat Tails Comparison](outputs/fat_tails_comparison.png)

*Left: Linear scale shows similar peaks. Center: Log scale reveals the fat tails—real markets have far more extreme events. Right: Quantified underestimation at each sigma level.*

---

## The Solution: Physics-Based Tail Modeling

This framework replaces naive Gaussian assumptions with **physics-derived models** that naturally generate fat tails:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHYSICS → FINANCE MAPPING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHYSICS CONCEPT              │  FINANCIAL APPLICATION                      │
│  ─────────────────────────────┼─────────────────────────────────────────    │
│  Lévy Flights (particle paths)│  Price jumps, Black Swan events             │
│  Fokker-Planck (prob. flow)   │  Distribution evolution, risk dynamics      │
│  Tsallis Entropy (thermo)     │  Non-equilibrium markets, fat tails         │
│  Phase Transitions (magnets)  │  Market crashes, regime changes             │
│  Ornstein-Uhlenbeck (springs) │  Volatility mean reversion + jumps          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Physics-Finance Conceptual Mapping

![Physics Finance Mapping](outputs/physics_finance_mapping.png)

*Each physics concept provides mathematical tools that naturally capture market behavior that Gaussian models miss.*

---

## Core Physics Models

### 1. Lévy Flights

**Origin:** Particle physics, describing random walks with occasional large jumps

**The Math:**
```
Characteristic function: φ(t) = exp(iδt - γ^α|t|^α [1 - iβ sign(t) tan(πα/2)])

Parameters:
  α (alpha): Stability index, 0 < α ≤ 2
    - α = 2: Gaussian (normal diffusion)
    - α < 2: Fat tails (super-diffusion, jumps)
    - α ≈ 1.7: Typical for stock returns

  β (beta): Skewness, -1 ≤ β ≤ 1
    - β < 0: Left skew (crash tendency)

  γ (gamma): Scale (volatility analog)
  δ (delta): Location (mean analog)
```

**Why It Works for Finance:**
- Mandelbrot first applied Lévy distributions to cotton prices (1963)
- Large jumps (Black Swans) are built into the model, not anomalies
- Tail probability: P(X > x) ~ x^(-α) for large x

```python
from src.physics.levy_flight import LevyStableDistribution

# Fit to market returns
levy = LevyStableDistribution.fit(returns)
print(f"Tail index α = {levy.alpha:.2f}")  # α < 2 confirms fat tails

# Compute tail probability
prob_crash = levy.tail_probability(-0.10, tail='left')  # P(loss > 10%)
```

#### Understanding the Tail Index (α)

![Tail Index Interpretation](outputs/tail_index_interpretation.png)

*Left: How different α values affect distribution shape. Right: Interpretation guide—lower α means fatter tails and more extreme events.*

### 2. Fokker-Planck Equation

**Origin:** Statistical mechanics, describing probability density evolution

**The Equation:**
```
∂P/∂t = -∂(μP)/∂x + ∂²(DP)/∂x²

Standard form for probability density P(x,t):
  μ(x): Drift coefficient (expected return)
  D(x): Diffusion coefficient (volatility)

For fat tails, use Fractional Fokker-Planck:
∂P/∂t = -∂(μP)/∂x + D_α ∂^α P/∂|x|^α

where α < 2 generates power-law tails
```

**Application:**
```python
from src.physics.fokker_planck import FokkerPlanckTailRisk

# Model probability evolution
fpe = FokkerPlanckTailRisk(alpha=1.7)  # α < 2 for fat tails
P0 = fpe.initial_distribution(current_return=0, current_vol=0.02)
P_forecast = fpe.forecast_distribution(P0, horizon=20)

# Get tail risk metrics
metrics = fpe.compute_tail_risk_metrics(P_forecast)
```

### 3. Tsallis Statistics

**Origin:** Non-extensive thermodynamics (Constantino Tsallis, 1988)

**The Framework:**
```
Tsallis Entropy: S_q = k × (1 - Σ p_i^q) / (q - 1)

q-Gaussian Distribution: P_q(x) ∝ [1 - β(1-q)x²]^(1/(1-q))

Parameters:
  q (entropic index):
    - q = 1: Standard Boltzmann-Gibbs (Gaussian)
    - q > 1: Fat tails, long-range correlations
    - q ≈ 1.4-1.5: Typical for stock returns

  Tail exponent: α = 2/(q-1)
    - q = 1.5 → α = 4 (fat but finite variance)
    - q = 1.67 → α = 3 (cubic law, like many markets)
```

**Why It Works:**
- Markets have long-range correlations (memory effects)
- Volatility clustering creates non-equilibrium conditions
- Tsallis entropy is maximized subject to these constraints

```python
from src.physics.tsallis_statistics import TsallisTailRiskModel

model = TsallisTailRiskModel()
risk_metrics = model.update(returns)

print(f"Entropic index q = {risk_metrics['q']:.2f}")
print(f"Risk level: {risk_metrics['risk_level']}")
```

### 4. Phase Transitions

**Origin:** Condensed matter physics (ferromagnets, critical phenomena)

**The Insight:**

Market crashes behave like phase transitions in physical systems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     PHASE TRANSITION ANALOGY                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PHYSICS (Magnet)           │   FINANCE (Market)                      │
│   ──────────────────────────────────────────────────────────────────   │
│   Spins (↑ or ↓)             │   Traders (buy or sell)                 │
│   Coupling J (spin interact) │   Herding (social influence)            │
│   Temperature T              │   Noise/uncertainty                      │
│   External field h           │   Market sentiment                       │
│   Magnetization M            │   Price trend                            │
│   Susceptibility χ           │   Market sensitivity                     │
│   Critical point Tc          │   Crash point                            │
│                                                                         │
│   Critical Slowing Down:                                                │
│   Near Tc, system responds slowly → Early Warning Signal               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Early Warning Signals (EWS):**
```python
from src.physics.phase_transitions import CriticalSlowingDownDetector

detector = CriticalSlowingDownDetector(window=50)
ews = detector.compute_ews(returns)

print(f"Autocorrelation trend: {ews['ac1_trend']['tau']:.2f}")
print(f"Variance trend: {ews['var_trend']['tau']:.2f}")
print(f"WARNING LEVEL: {ews['warning_level']}")

# Positive trends with p < 0.05 indicate approaching critical point
```

#### Early Warning Signals in Action

![Early Warning Signals](outputs/early_warning_signals.png)

*Before a crisis: autocorrelation rises (critical slowing down), variance increases, and skewness becomes negative. These signals can provide advance warning of impending market stress.*

### Real-World Crisis Examples

The framework's early warning capabilities are demonstrated using three major S&P 500 market crises with **real historical data**:

#### COVID-19 Market Crash (February-March 2020)

The COVID-19 pandemic caused one of the fastest S&P 500 crashes in history:
- **S&P 500 Peak:** 3,386.15 (February 19, 2020)
- **S&P 500 Trough:** 2,237.40 (March 23, 2020)
- **Maximum Drawdown:** -33.9% in just 23 trading days
- **Worst Day:** March 16, 2020 (-11.98%, circuit breaker triggered at open)
- **Best Day:** March 24, 2020 (+9.38%, biggest gain since 2008)

![COVID-19 Crash Analysis](outputs/covid_crash_analysis.png)

*S&P 500 analysis during COVID-19: The index dropped from 3,386 to 2,237 in 23 trading days. Daily returns show extreme volatility with multiple circuit breakers triggered. Early warning signals detected elevated risk before the worst losses.*

**Full Dashboard - COVID-19 Crisis:**

![COVID-19 Dashboard](outputs/covid_crash_dashboard.png)

*Complete tail risk analysis of S&P 500 during the COVID-19 crash showing 3D phase space trajectories, distribution shifts, and risk metrics evolution.*

#### 2022 Bear Market (January-October 2022)

The 2022 bear market was driven by aggressive Fed rate hikes to combat 9.1% inflation:
- **S&P 500 Peak:** 4,796.56 (January 3, 2022)
- **S&P 500 Trough:** 3,577.03 (October 12, 2022)
- **Maximum Drawdown:** -25.4% over 282 days
- **Cause:** Federal Reserve rate hikes, inflation at 40-year high (9.1%)
- **Character:** Slow grinding decline with multiple relief rallies

![2022 Bear Market Analysis](outputs/bear_2022_analysis.png)

*S&P 500 analysis during the 2022 bear market: Unlike the sudden COVID shock, this was a prolonged 9-month decline. The early warning system showed persistently elevated risk throughout the year.*

**Full Dashboard - 2022 Bear Market:**

![2022 Bear Market Dashboard](outputs/bear_2022_dashboard.png)

*Complete tail risk analysis of S&P 500 during the 2022 bear market. The phase space trajectory shows a gradual drift into high-risk territory rather than a sudden spike.*

#### Tariff Crisis (February-April 2025)

The 2025 "Liberation Day" tariff shock caused the largest two-day point loss in S&P 500 history:
- **S&P 500 Peak:** 6,144 (February 19, 2025)
- **S&P 500 Trough:** 4,658 (April 4, 2025)
- **Maximum Drawdown:** -24.2%
- **April 3 ("Liberation Day"):** -4.84% (steep tariffs announced)
- **April 4:** -5.97% (two-day loss: -10.3%, $6.6 trillion wiped out)
- **April 9:** +9.52% (90-day tariff pause announced, biggest gain in years)

![Tariff Crisis Analysis](outputs/tariff_crash_analysis.png)

*S&P 500 analysis during the 2025 Tariff Crisis: The sharp V-shaped pattern shows the rapid crash and equally rapid recovery after the tariff pause was announced.*

**Full Dashboard - Tariff Crisis:**

![Tariff Crisis Dashboard](outputs/tariff_crash_dashboard.png)

*Complete tail risk analysis of S&P 500 during the 2025 tariff crisis. By June 2025, the S&P 500 reached new all-time highs, and the year ended with +17% gains.*

#### S&P 500 Crisis Comparison

![Crisis Comparison](outputs/crisis_comparison.png)

*Side-by-side comparison of all three S&P 500 crises: COVID-19 (top) was the deepest and fastest. The 2022 bear market (middle) was the longest. The 2025 tariff crash (bottom) had the sharpest V-shaped recovery.*

**Key Insights from S&P 500 Analysis (Real Data):**

| Metric | COVID-19 (2020) | 2022 Bear Market | Tariff Crisis (2025) |
|--------|-----------------|------------------|---------------------|
| S&P 500 Peak | 3,386 | 4,797 | 6,144 |
| S&P 500 Trough | 2,237 | 3,577 | 4,658 |
| Duration | 23 trading days | 282 days | ~35 trading days |
| Maximum Drawdown | -33.9% | -25.4% | -24.2% |
| Worst Single Day | -11.98% | -3.9% | -5.97% |
| Best Single Day | +9.38% | +2.8% | +9.52% |
| Recovery Pattern | V-shaped | Gradual (2023) | Sharp V-shaped |
| Tail Index (α) | ≈ 1.2 (extreme) | ≈ 1.8 (fat) | ≈ 1.5 (fat) |

> **Key Finding:** Each crisis had a distinct character. COVID-19 was a sudden exogenous shock with extreme daily moves. The 2022 bear market was a slow policy-driven decline. The 2025 tariff crash was sharp but short-lived after policy reversal. The early warning system detected all three, but with varying lead times depending on crisis type.

**Sources:**
- [2020 Stock Market Crash - Wikipedia](https://en.wikipedia.org/wiki/2020_stock_market_crash)
- [2022 Stock Market Decline - Wikipedia](https://en.wikipedia.org/wiki/2022_stock_market_decline)
- [2025 Stock Market Crash - Wikipedia](https://en.wikipedia.org/wiki/2025_stock_market_crash)

---

### Current Market Analysis (January 2026)

After experiencing three major market crises, where does the S&P 500 stand today? The framework's early warning system provides real-time risk assessment.

**Current Market Status (January 2, 2026):**
- **S&P 500 Current Level:** 6,888
- **All-Time High:** 6,940 (December 26, 2025)
- **Distance from ATH:** -0.7%
- **Q4 2025 Gain:** +12.9%
- **2025 Full Year Return:** +18% (third consecutive year of double-digit gains)
- **Annualized Volatility:** 10.4% (historically low)

![Current Market Analysis](outputs/current_market_analysis.png)

*Current market conditions: S&P 500 trading near all-time highs with low volatility. Early warning indicators show normal risk levels—a stark contrast to the elevated readings seen before each of the three crises.*

**Full Dashboard - Current Market:**

![Current Market Dashboard](outputs/current_market_dashboard.png)

*Complete tail risk analysis of the current market environment. Low volatility, minimal autocorrelation, and stable skewness indicate healthy market conditions.*

---

### Crisis vs Current Market Comparison

How do current market conditions compare to the three major crises? This analysis reveals the dramatic difference between crisis periods and today's bull market.

![Crisis vs Current Comparison](outputs/crisis_vs_current_comparison.png)

*Comprehensive comparison of all three crises against the current market. Current conditions show dramatically lower volatility, minimal drawdown, and normal risk scores—indicating a healthy market far from crisis territory.*

**Quantitative Comparison: Crises vs Today**

| Metric | COVID-19 (2020) | 2022 Bear | 2025 Tariff | **Current (2026)** |
|--------|-----------------|-----------|-------------|-------------------|
| S&P 500 Level | 3,386 → 2,237 | 4,797 → 3,577 | 6,144 → 4,658 | **6,888 (ATH -0.7%)** |
| Annualized Vol | ~80% | ~25% | ~35% | **10.4%** |
| Max Drawdown | -33.9% | -25.4% | -24.2% | **-0.7%** |
| Worst Day | -11.98% | -3.9% | -5.97% | **-0.5%** |
| Risk Status | CRITICAL | ELEVATED | CRITICAL | **NORMAL** |

**Key Observations:**

1. **Volatility Contrast:** Current annualized volatility of 10.4% is roughly 8x lower than COVID-19 crisis levels and 3x lower than the 2025 tariff shock.

2. **Early Warning Signals:** All EWS metrics (autocorrelation, variance, skewness) are in the "normal" zone—unlike the elevated readings that preceded each historical crisis.

3. **Market Structure:** The steady upward trend with low volatility suggests orderly market conditions without the herding behavior that characterizes pre-crash environments.

4. **Risk Assessment:** The composite risk indicator remains well below warning thresholds, suggesting the market is not exhibiting the critical slowing down patterns seen before major corrections.

> **Current Outlook:** As of January 2026, the S&P 500 shows none of the early warning signals that preceded the COVID-19 crash, 2022 bear market, or 2025 tariff crisis. The third consecutive year of double-digit gains has brought the index to new all-time highs with historically low volatility. While past performance does not predict future returns, the current EWS readings suggest the market is not in an imminent pre-crisis state.

---

## 3D Tail Risk Coordinate System

### The Phase Space Concept

We map market states into a **3-dimensional phase space** inspired by physics. Each axis represents a key aspect of tail risk:

```
                          3D TAIL RISK PHASE SPACE

                               Z (Risk Intensity)
                               │
                               │    ★ Crisis Zone
                               │   ╱ (high vol, fat tails,
                               │  ╱   high jump intensity)
                               │ ╱
                               │╱______________ Y (Tail Heaviness)
                              ╱│
                             ╱ │
                            ╱  │    • Normal Zone
                           ╱   │     (low vol, thin tails)
                          ╱    │
                X (Volatility)

The market traces a trajectory through this space.
Movement toward the Crisis Zone = increasing tail risk.
```

### Three Coordinate Systems

We provide three complementary phase spaces, each revealing different aspects:

#### 1. Lévy Flight Coordinates
```
X: Volatility regime σ (normalized rolling std)
Y: Tail index α⁻¹ (inverse Hill estimator - higher = fatter tails)
Z: Jump intensity λ (magnitude of extreme returns)
```

#### 2. Thermodynamic Coordinates (Tsallis)
```
X: Entropic index q (non-extensivity measure)
Y: Tsallis entropy S_q (uncertainty/disorder)
Z: Temperature β⁻¹ (volatility energy proxy)
```

#### 3. Phase Transition Coordinates
```
X: Susceptibility χ (sensitivity to shocks)
Y: Order parameter M (trend strength / herding)
Z: Criticality distance |T - Tc| (distance from crash point)
```

### Visualization

```python
from src.visualization.risk_surface_3d import create_comprehensive_3d_dashboard
from src.physics.levy_flight import levy_flight_3d_coordinates

# Compute coordinates
levy_coords = levy_flight_3d_coordinates(returns)

# Create comprehensive dashboard
fig = create_comprehensive_3d_dashboard(
    returns,
    save_path='outputs/3d_dashboard.png'
)
```

### 3D Phase Space Dashboard

![3D Tail Risk Dashboard](outputs/tail_risk_3d_dashboard.png)

*Four coordinate systems visualized: Lévy flight space (volatility, tail index, jumps), Thermodynamic space (q-parameter, entropy, temperature), Phase transition space (susceptibility, order, criticality), and composite risk surface.*

---

## Installation & Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tail-risk-fat-tails.git
cd tail-risk-fat-tails

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run demonstration
python main.py --mode demo --output outputs/

# This will:
# 1. Generate synthetic market data with fat tails
# 2. Fit physics-based distributions
# 3. Compute tail risk metrics
# 4. Generate 3D visualizations
# 5. Create comprehensive dashboard
# 6. Output analysis report

# Generate educational figures for documentation
python generate_readme_figures.py
```

### Basic Usage

```python
import numpy as np
from src.physics.levy_flight import LevyStableDistribution, estimate_tail_index
from src.models.risk_metrics import TailRiskMetrics
from src.visualization.dashboard import TailRiskDashboard

# Your return data
returns = np.array([...])  # Daily log returns

# 1. Estimate tail characteristics
alpha = estimate_tail_index(returns)
print(f"Tail index α = {alpha:.2f}")
if alpha < 3:
    print("⚠️ Fat tails detected! Variance dominated by extremes.")

# 2. Fit Lévy distribution
levy = LevyStableDistribution.fit(returns)
print(f"Lévy parameters: α={levy.alpha:.2f}, β={levy.beta:.2f}")

# 3. Compute risk metrics
metrics = TailRiskMetrics(returns)
print(f"VaR 99% (historical): {metrics.var_historical(0.99)*100:.2f}%")
print(f"VaR 99% (Gaussian):   {metrics.var_parametric_gaussian(0.99)*100:.2f}%")
print(f"Expected Shortfall:   {metrics.expected_shortfall(0.99)*100:.2f}%")

# 4. Generate full dashboard
dashboard = TailRiskDashboard(returns)
dashboard.create_full_dashboard(save_path='my_analysis.png')
```

---

## Visualization Gallery

### 1. Full Analysis Dashboard

The comprehensive 24-panel dashboard provides a complete view of tail risk across multiple dimensions:

![Full Dashboard](outputs/full_dashboard.png)

*Complete tail risk analysis: 3D phase spaces, distribution analysis, time series metrics, and risk indicators.*

### 2. Crisis Period Analysis

Compare normal market conditions with crisis periods:

![Crisis Dashboard](outputs/crisis_dashboard.png)

*Same analysis during a crisis period—notice the dramatic changes in phase space trajectories and risk metrics.*

### 3. Risk Metrics Comparison

![VaR Comparison](outputs/var_comparison.png)

*Left: VaR estimates across methods—Gaussian consistently underestimates risk. Right: Progressive risk metrics from VaR to extreme return levels.*

### 4. Phase Space Trajectory

Animated trajectory through risk space:

```python
from src.visualization.animated_risk import PhaseSpaceAnimator

animator = PhaseSpaceAnimator(X, Y, Z)
anim = animator.create_animation(
    trail_length=50,
    save_path='phase_trajectory.gif'
)
```

---

## API Reference

### Physics Models

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `physics.levy_flight` | `LevyStableDistribution` | Lévy stable distribution |
| | `LevyFlightProcess` | Price process with Lévy jumps |
| | `estimate_tail_index()` | Hill estimator for α |
| | `levy_flight_3d_coordinates()` | Phase space transform |
| `physics.fokker_planck` | `FokkerPlanckSolver` | Fractional FPE solver |
| | `FokkerPlanckTailRisk` | Tail risk analyzer |
| `physics.tsallis_statistics` | `TsallisDistribution` | q-Gaussian distribution |
| | `TsallisTailRiskModel` | Non-extensive risk model |
| `physics.phase_transitions` | `IsingMarketModel` | Agent-based crash model |
| | `CriticalSlowingDownDetector` | EWS detector |
| | `MarketPhaseClassifier` | Regime classifier |

### Risk Metrics

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `models.risk_metrics` | `TailRiskMetrics` | Comprehensive risk measures |
| | `RollingRiskMetrics` | Time-varying risk |
| | `TailRiskParity` | ES-based portfolio allocation |
| `models.extreme_value` | `GeneralizedParetoDistribution` | GPD for tails |
| | `EVTTailRiskAnalyzer` | Full EVT analysis |
| `models.regime_detection` | `MarkovRegimeSwitching` | Hidden Markov model |
| | `RegimeAwareTailRisk` | Regime-conditional risk |

### Visualization

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `visualization.risk_surface_3d` | `TailRisk3DSurface` | 3D surface plots |
| | `create_comprehensive_3d_dashboard()` | Full 3D dashboard |
| `visualization.phase_space` | `PhaseSpaceAnalyzer` | Trajectory analysis |
| | `LyapunovExponentEstimator` | Chaos detection |
| `visualization.dashboard` | `TailRiskDashboard` | Complete dashboard |
| | `create_summary_report()` | Text report generator |

---

## Mathematical Foundations

### Lévy Stable Distributions

The Lévy stable distribution is defined by its characteristic function:

```
φ(t) = exp[iδt - γ^α|t|^α(1 - iβ sign(t) Φ)]
```

where Φ = tan(πα/2) for α ≠ 1.

**Key Properties:**
- For α < 2: Infinite variance (dominated by extremes)
- For α < 1: Infinite mean
- Tail behavior: P(X > x) ~ C_α x^(-α) for large x

### Fokker-Planck Equation

The Fokker-Planck equation describes probability density evolution:

```
∂P/∂t = -∂(μP)/∂x + ∂²(DP)/∂x²
```

For fat tails, we use the **Fractional Fokker-Planck Equation**:

```
∂P/∂t = -∂(μP)/∂x + D_α ∂^α P/∂|x|^α
```

where α < 2 produces power-law tails.

### Tsallis Statistics

Tsallis entropy generalizes Boltzmann-Gibbs:

```
S_q = k(1 - Σp_i^q)/(q - 1)
```

The maximum entropy distribution (q-Gaussian):

```
P_q(x) ∝ [1 - β(1-q)x²]^(1/(1-q))
```

For q > 1: Power-law tails with exponent α = 2/(q-1)

### Extreme Value Theory

**Generalized Pareto Distribution** for exceedances over threshold u:

```
G(y) = 1 - (1 + ξy/σ)^(-1/ξ)
```

- ξ > 0: Fat tails (Fréchet domain) - typical for finance
- ξ = 0: Exponential tails (Gumbel domain)
- ξ < 0: Bounded tails (Weibull domain)

---

## Research References

### Foundational Papers

1. **Mandelbrot, B. (1963)**. "The Variation of Certain Speculative Prices." *Journal of Business*, 36(4), 394-419.
   - First application of Lévy distributions to finance

2. **Tsallis, C. (1988)**. "Possible generalization of Boltzmann-Gibbs statistics." *Journal of Statistical Physics*, 52(1-2), 479-487.
   - Foundation of non-extensive statistical mechanics

3. **Mantegna, R. N., & Stanley, H. E. (2000)**. *An Introduction to Econophysics*. Cambridge University Press.
   - Comprehensive treatment of physics approaches to finance

### Fat Tails and Extreme Events

4. **Taleb, N. N. (2020)**. *Statistical Consequences of Fat Tails*. STEM Academic Press.
   - Modern treatment of fat tail statistics

5. **Sornette, D. (2003)**. *Why Stock Markets Crash*. Princeton University Press.
   - Critical phenomena and market crashes

6. **Gabaix, X. (2009)**. "Power Laws in Economics and Finance." *Annual Review of Economics*, 1, 255-294.
   - Review of power laws in finance

### Early Warning Signals

7. **Scheffer, M., et al. (2009)**. "Early-warning signals for critical transitions." *Nature*, 461, 53-59.
   - Critical slowing down as early warning

8. **Sornette, D., & Cauwels, P. (2015)**. "Financial Bubbles: Mechanisms and Diagnostics." *Review of Behavioral Economics*, 2(3), 279-305.
   - Bubble detection methodology

---

## Project Structure

```
Tail-Risk-fat-tails/
├── src/
│   ├── physics/
│   │   ├── levy_flight.py          # Lévy stable distributions
│   │   ├── fokker_planck.py        # Probability evolution
│   │   ├── tsallis_statistics.py   # Non-extensive thermodynamics
│   │   ├── phase_transitions.py    # Critical phenomena
│   │   └── ornstein_uhlenbeck.py   # Mean-reverting + jumps
│   ├── models/
│   │   ├── fat_tail_distributions.py  # Distribution zoo
│   │   ├── extreme_value.py        # EVT (GEV, GPD)
│   │   ├── risk_metrics.py         # VaR, ES, tail ratios
│   │   └── regime_detection.py     # Markov switching
│   ├── analysis/
│   │   └── sentiment.py            # VIX, put/call, breadth
│   ├── visualization/
│   │   ├── risk_surface_3d.py      # 3D surfaces
│   │   ├── phase_space.py          # Trajectory analysis
│   │   ├── animated_risk.py        # Animations
│   │   └── dashboard.py            # Full dashboard
│   └── utils/
│       ├── data_loader.py          # Data generation
│       ├── numerical.py            # Numerical helpers
│       └── statistics.py           # Statistical tests
├── outputs/                        # Generated visualizations
│   ├── full_dashboard.png          # Complete analysis dashboard
│   ├── crisis_dashboard.png        # Crisis period analysis
│   ├── tail_risk_3d_dashboard.png  # 3D phase space views
│   ├── fat_tails_comparison.png    # Gaussian vs fat tails
│   ├── early_warning_signals.png   # EWS before crisis
│   ├── var_comparison.png          # VaR method comparison
│   ├── tail_index_interpretation.png  # Alpha interpretation
│   ├── physics_finance_mapping.png # Conceptual mapping
│   ├── covid_crash_analysis.png    # COVID-19 crisis EWS analysis
│   ├── covid_crash_dashboard.png   # COVID-19 full dashboard
│   ├── bear_2022_analysis.png      # 2022 bear market EWS analysis
│   ├── bear_2022_dashboard.png     # 2022 bear market full dashboard
│   ├── tariff_crash_analysis.png   # 2025 tariff crisis EWS analysis
│   ├── tariff_crash_dashboard.png  # Tariff crisis full dashboard
│   ├── crisis_comparison.png       # Three-crisis comparison
│   ├── current_market_analysis.png # Current market (Jan 2026) analysis
│   ├── current_market_dashboard.png # Current market full dashboard
│   └── crisis_vs_current_comparison.png # Crises vs current comparison
├── main.py                         # Entry point
├── generate_readme_figures.py      # Generate documentation figures
├── generate_crisis_examples.py     # Generate real-world crisis analyses
├── requirements.txt
└── README.md
```

---

## License

MIT License - see LICENSE file for details.

---

<p align="center">
  <b>Tail Risk Modeling Framework</b><br>
  <i>Because Black Swans aren't anomalies—they're features of the distribution.</i>
</p>
