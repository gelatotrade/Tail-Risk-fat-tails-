"""
Market Sentiment Analysis for Tail Risk Prediction
===================================================

Market sentiment indicators often precede major market moves.
This module provides sentiment metrics that correlate with
increased tail risk.

Key Indicators:
1. VIX and volatility term structure
2. Put/Call ratios (fear gauge)
3. Market breadth (participation)
4. Credit spreads (risk appetite)
5. Safe haven flows (flight to quality)
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, List
import warnings


class VIXAnalyzer:
    """
    Analyze VIX (Volatility Index) for tail risk signals.

    VIX measures 30-day implied volatility from S&P 500 options.
    - VIX < 15: Complacency (low fear, but risk may be building)
    - VIX 15-25: Normal market conditions
    - VIX 25-35: Elevated fear
    - VIX > 35: Panic / Crisis mode

    Key signals:
    - VIX spikes: Sudden fear increase
    - Low VIX persistence: Risk building
    - Term structure inversion: Near-term fear > long-term
    """

    def __init__(self, vix_data: Optional[np.ndarray] = None):
        """
        Initialize VIX analyzer.

        Args:
            vix_data: Historical VIX values (optional, can add later)
        """
        self.vix = vix_data if vix_data is not None else np.array([])
        self.thresholds = {
            'complacent': 15,
            'normal': 25,
            'elevated': 35,
            'panic': 50
        }

    def add_data(self, vix_values: np.ndarray):
        """Add VIX data."""
        self.vix = np.concatenate([self.vix, np.asarray(vix_values)])

    def current_regime(self) -> str:
        """Classify current VIX regime."""
        if len(self.vix) == 0:
            return 'UNKNOWN'

        current = self.vix[-1]

        if current < self.thresholds['complacent']:
            return 'COMPLACENT'
        elif current < self.thresholds['normal']:
            return 'NORMAL'
        elif current < self.thresholds['elevated']:
            return 'ELEVATED'
        elif current < self.thresholds['panic']:
            return 'HIGH_FEAR'
        else:
            return 'PANIC'

    def vix_percentile(self, window: int = 252) -> float:
        """Current VIX percentile over lookback window."""
        if len(self.vix) < window:
            window = len(self.vix)

        if window == 0:
            return 50.0

        return stats.percentileofscore(self.vix[-window:], self.vix[-1])

    def vix_spike_detection(self, threshold: float = 2.0) -> Dict:
        """
        Detect VIX spikes (sudden fear increases).

        A spike is defined as VIX move > threshold * rolling std.
        """
        if len(self.vix) < 20:
            return {'spike_detected': False}

        # Rolling statistics
        window = 20
        rolling_mean = np.mean(self.vix[-window:-1])
        rolling_std = np.std(self.vix[-window:-1])

        current_change = self.vix[-1] - self.vix[-2]
        z_score = current_change / (rolling_std + 1e-10)

        return {
            'spike_detected': z_score > threshold,
            'z_score': z_score,
            'current_change': current_change,
            'current_vix': self.vix[-1],
            'rolling_mean': rolling_mean
        }

    def mean_reversion_signal(self, window: int = 20) -> Dict:
        """
        VIX tends to mean-revert. Extreme readings often reverse.

        High VIX = potential market bottom (fear max)
        Very low VIX = potential market top (complacency max)
        """
        if len(self.vix) < window:
            return {'signal': 'NEUTRAL', 'strength': 0}

        current = self.vix[-1]
        mean = np.mean(self.vix[-252:]) if len(self.vix) > 252 else np.mean(self.vix)
        std = np.std(self.vix[-252:]) if len(self.vix) > 252 else np.std(self.vix)

        z_score = (current - mean) / (std + 1e-10)

        if z_score > 2:
            signal = 'HIGH_FEAR_REVERSAL_LIKELY'
            strength = min(z_score / 3, 1)
        elif z_score < -1.5:
            signal = 'COMPLACENCY_WARNING'
            strength = min(-z_score / 2, 1)
        else:
            signal = 'NEUTRAL'
            strength = 0

        return {
            'signal': signal,
            'strength': strength,
            'z_score': z_score,
            'days_to_mean': abs(current - mean) / (std / np.sqrt(window)) if std > 0 else 0
        }


class PutCallRatioAnalyzer:
    """
    Analyze Put/Call ratios as fear gauge.

    Put/Call ratio = Put volume / Call volume

    - High ratio (> 1.0): More puts, bearish sentiment, fear
    - Low ratio (< 0.7): More calls, bullish sentiment, greed
    - Extreme readings often precede reversals (contrarian)
    """

    def __init__(self):
        self.equity_pc = np.array([])  # Equity put/call
        self.index_pc = np.array([])   # Index put/call
        self.total_pc = np.array([])   # Total put/call

    def add_data(self, equity: Optional[float] = None,
                 index: Optional[float] = None,
                 total: Optional[float] = None):
        """Add put/call ratio data."""
        if equity is not None:
            self.equity_pc = np.append(self.equity_pc, equity)
        if index is not None:
            self.index_pc = np.append(self.index_pc, index)
        if total is not None:
            self.total_pc = np.append(self.total_pc, total)

    def fear_greed_indicator(self, pc_ratio: float) -> Dict:
        """
        Convert put/call ratio to fear/greed indicator.

        Uses historical context to normalize.
        """
        if len(self.equity_pc) < 20:
            # Default thresholds
            if pc_ratio > 1.0:
                return {'level': 'EXTREME_FEAR', 'score': -100}
            elif pc_ratio > 0.85:
                return {'level': 'FEAR', 'score': -50}
            elif pc_ratio < 0.6:
                return {'level': 'EXTREME_GREED', 'score': 100}
            elif pc_ratio < 0.7:
                return {'level': 'GREED', 'score': 50}
            else:
                return {'level': 'NEUTRAL', 'score': 0}

        # Use historical percentile
        percentile = stats.percentileofscore(self.equity_pc, pc_ratio)

        if percentile > 90:
            return {'level': 'EXTREME_FEAR', 'score': -(percentile - 50) * 2}
        elif percentile > 70:
            return {'level': 'FEAR', 'score': -(percentile - 50)}
        elif percentile < 10:
            return {'level': 'EXTREME_GREED', 'score': (50 - percentile) * 2}
        elif percentile < 30:
            return {'level': 'GREED', 'score': (50 - percentile)}
        else:
            return {'level': 'NEUTRAL', 'score': 50 - percentile}

    def contrarian_signal(self) -> Dict:
        """
        Generate contrarian signal from P/C ratio.

        Extreme fear often marks bottoms.
        Extreme greed often marks tops.
        """
        if len(self.equity_pc) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 0}

        current = self.equity_pc[-1]
        percentile = stats.percentileofscore(self.equity_pc, current)

        if percentile > 90:
            return {
                'signal': 'CONTRARIAN_BULLISH',
                'confidence': (percentile - 90) / 10,
                'reason': 'Extreme fear often marks bottoms'
            }
        elif percentile < 10:
            return {
                'signal': 'CONTRARIAN_BEARISH',
                'confidence': (10 - percentile) / 10,
                'reason': 'Extreme greed often marks tops'
            }
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0}


class MarketBreadthAnalyzer:
    """
    Analyze market breadth for tail risk signals.

    Market breadth = how many stocks participate in a move.

    Warning signs:
    - Narrow leadership: Few stocks driving index
    - Divergence: Index up but breadth deteriorating
    - Extreme readings: Very high or very low advance/decline
    """

    def __init__(self):
        self.advance_decline = np.array([])  # A/D line
        self.percent_above_50ma = np.array([])  # % above 50-day MA
        self.percent_above_200ma = np.array([])  # % above 200-day MA
        self.new_highs_lows = np.array([])  # New highs - new lows

    def add_data(self, ad_line: Optional[float] = None,
                 pct_50ma: Optional[float] = None,
                 pct_200ma: Optional[float] = None,
                 nh_nl: Optional[float] = None):
        """Add breadth data."""
        if ad_line is not None:
            self.advance_decline = np.append(self.advance_decline, ad_line)
        if pct_50ma is not None:
            self.percent_above_50ma = np.append(self.percent_above_50ma, pct_50ma)
        if pct_200ma is not None:
            self.percent_above_200ma = np.append(self.percent_above_200ma, pct_200ma)
        if nh_nl is not None:
            self.new_highs_lows = np.append(self.new_highs_lows, nh_nl)

    def breadth_thrust_signal(self) -> Dict:
        """
        Detect breadth thrust (powerful advance signal).

        When breadth moves from oversold to overbought quickly,
        it often signals start of a new bull market.
        """
        if len(self.percent_above_50ma) < 20:
            return {'thrust_detected': False}

        current = self.percent_above_50ma[-1]
        prior = self.percent_above_50ma[-10]

        thrust = current - prior

        return {
            'thrust_detected': thrust > 40,  # 40% improvement in 10 days
            'thrust_magnitude': thrust,
            'current_breadth': current,
            'prior_breadth': prior
        }

    def divergence_warning(self, price_index: np.ndarray) -> Dict:
        """
        Detect breadth/price divergence (warning signal).

        If prices make new highs but breadth doesn't, it's bearish.
        """
        if len(self.advance_decline) < 50 or len(price_index) < 50:
            return {'divergence': False}

        # Check if price at 20-day high
        price_at_high = price_index[-1] >= np.max(price_index[-20:])

        # Check if breadth NOT at high
        breadth_lagging = self.advance_decline[-1] < np.max(self.advance_decline[-20:])

        return {
            'divergence': price_at_high and breadth_lagging,
            'price_at_high': price_at_high,
            'breadth_lagging': breadth_lagging,
            'warning_level': 'HIGH' if (price_at_high and breadth_lagging) else 'LOW'
        }


class CompositeSentimentIndicator:
    """
    Combine multiple sentiment indicators into unified score.

    The composite indicator ranges from -100 (extreme fear) to +100 (extreme greed).
    Extreme readings often precede reversals.
    """

    def __init__(self):
        self.vix_analyzer = VIXAnalyzer()
        self.pc_analyzer = PutCallRatioAnalyzer()
        self.breadth_analyzer = MarketBreadthAnalyzer()

        self.weights = {
            'vix': 0.3,
            'put_call': 0.25,
            'breadth': 0.25,
            'momentum': 0.2
        }

    def compute_composite(self, vix: Optional[float] = None,
                         put_call: Optional[float] = None,
                         breadth: Optional[float] = None,
                         momentum: Optional[float] = None) -> Dict:
        """
        Compute composite sentiment score.

        All inputs should be normalized to [-100, 100] scale.
        """
        scores = {}
        total_weight = 0

        if vix is not None:
            # VIX: high = fear (-), low = greed (+)
            scores['vix'] = 50 - vix  # Invert
            total_weight += self.weights['vix']

        if put_call is not None:
            # P/C ratio: high = fear (-), low = greed (+)
            fg = self.pc_analyzer.fear_greed_indicator(put_call)
            scores['put_call'] = fg['score']
            total_weight += self.weights['put_call']

        if breadth is not None:
            # Breadth: high = bullish (+), low = bearish (-)
            scores['breadth'] = (breadth - 50) * 2
            total_weight += self.weights['breadth']

        if momentum is not None:
            # Momentum: positive = bullish, negative = bearish
            scores['momentum'] = np.clip(momentum * 10, -100, 100)
            total_weight += self.weights['momentum']

        if total_weight == 0:
            return {'composite': 0, 'level': 'UNKNOWN', 'components': {}}

        # Weighted average
        composite = sum(
            scores.get(k, 0) * self.weights[k]
            for k in scores
        ) / total_weight

        # Classify
        if composite > 75:
            level = 'EXTREME_GREED'
        elif composite > 25:
            level = 'GREED'
        elif composite < -75:
            level = 'EXTREME_FEAR'
        elif composite < -25:
            level = 'FEAR'
        else:
            level = 'NEUTRAL'

        return {
            'composite': composite,
            'level': level,
            'components': scores,
            'tail_risk_signal': self._tail_risk_from_sentiment(composite)
        }

    def _tail_risk_from_sentiment(self, composite: float) -> Dict:
        """
        Convert sentiment to tail risk signal.

        Extreme greed (complacency) = elevated tail risk
        Extreme fear = tail risk materializing, may be near peak
        """
        if composite > 75:
            return {
                'level': 'ELEVATED',
                'direction': 'Complacency high - crash risk elevated',
                'action': 'Reduce risk exposure'
            }
        elif composite > 50:
            return {
                'level': 'MODERATE',
                'direction': 'Bullish sentiment - monitor for extremes',
                'action': 'Maintain hedges'
            }
        elif composite < -75:
            return {
                'level': 'EXTREME',
                'direction': 'Panic selling - may be near capitulation',
                'action': 'Prepare for reversal'
            }
        elif composite < -50:
            return {
                'level': 'HIGH',
                'direction': 'Fear rising - volatility expected',
                'action': 'Increase hedges'
            }
        else:
            return {
                'level': 'NORMAL',
                'direction': 'Neutral sentiment',
                'action': 'Normal positioning'
            }


def compute_sentiment_coordinates(vix_series: np.ndarray,
                                  returns: np.ndarray,
                                  window: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform sentiment data into 3D phase space.

    Coordinates:
    - X: VIX level (fear intensity)
    - Y: VIX momentum (fear acceleration)
    - Z: Return-VIX correlation (fear efficiency)

    High X, positive Y, negative Z = peak fear, tail risk materializing
    Low X, negative Y, positive Z = complacency, tail risk building
    """
    n = len(returns)
    vix = vix_series if len(vix_series) == n else np.interp(
        np.arange(n), np.linspace(0, n, len(vix_series)), vix_series
    )

    X = np.zeros(n)  # VIX level
    Y = np.zeros(n)  # VIX momentum
    Z = np.zeros(n)  # Correlation

    for i in range(window, n):
        X[i] = vix[i]
        Y[i] = vix[i] - np.mean(vix[i-window:i])

        # Correlation between returns and VIX changes
        vix_changes = np.diff(vix[i-window:i+1])
        ret_window = returns[i-window+1:i+1]
        if len(vix_changes) == len(ret_window) and len(vix_changes) > 2:
            Z[i] = np.corrcoef(ret_window, vix_changes)[0, 1]

    # Normalize
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-10)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-10)
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-10)

    return X, Y, Z
