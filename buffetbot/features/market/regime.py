"""
Market Regime Detection Module

Professional market regime classification using multiple indicators
to identify trending, ranging, and volatile market conditions.
These regime features significantly improve ML model performance
by providing context about market state.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Professional market regime detection and classification.

    This class identifies different market regimes using multiple
    technical indicators and statistical measures.
    """

    @staticmethod
    def adx_regime(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> dict[str, pd.Series]:
        """
        Classify market regime using Average Directional Index (ADX).

        ADX measures trend strength:
        - ADX > 25: Strong trending market
        - ADX 20-25: Moderate trend
        - ADX < 20: Ranging/sideways market

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for ADX calculation (default: 14)

        Returns:
            Dictionary containing:
            - adx: ADX values
            - plus_di: Positive Directional Indicator
            - minus_di: Negative Directional Indicator
            - regime: Market regime classification
            - trend_direction: Trend direction (up/down/sideways)

        Example:
            >>> regime = MarketRegimeDetector.adx_regime(high, low, close)
            >>> current_regime = regime['regime'].iloc[-1]
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if len(close) < period * 2:
                logger.warning(
                    f"Insufficient data for ADX calculation: {len(close)} < {period * 2}"
                )
                empty_series = pd.Series(index=close.index, dtype=float)
                empty_regime = pd.Series(index=close.index, dtype=str)
                return {
                    "adx": empty_series,
                    "plus_di": empty_series,
                    "minus_di": empty_series,
                    "regime": empty_regime,
                    "trend_direction": empty_regime,
                }

            # Calculate True Range (TR)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate Directional Movements
            plus_dm = pd.Series(index=close.index, dtype=float)
            minus_dm = pd.Series(index=close.index, dtype=float)

            high_diff = high.diff()
            low_diff = -low.diff()

            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            plus_dm = pd.Series(plus_dm, index=close.index)
            minus_dm = pd.Series(minus_dm, index=close.index)

            # Smooth with Wilder's moving average
            tr_smooth = true_range.ewm(alpha=1 / period, adjust=False).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1 / period, adjust=False).mean()

            # Classify regime based on ADX
            regime = pd.Series(index=close.index, dtype=str)
            regime = np.where(
                adx >= 25,
                "trending_strong",
                np.where(adx >= 20, "trending_moderate", "ranging"),
            )
            regime = pd.Series(regime, index=close.index)

            # Determine trend direction
            trend_direction = pd.Series(index=close.index, dtype=str)
            trend_direction = np.where(
                plus_di > minus_di,
                "up",
                np.where(minus_di > plus_di, "down", "sideways"),
            )
            trend_direction = pd.Series(trend_direction, index=close.index)

            return {
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "regime": regime,
                "trend_direction": trend_direction,
            }

        except Exception as e:
            logger.error(f"Error calculating ADX regime: {str(e)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            empty_regime = pd.Series(index=close.index, dtype=str)
            return {
                "adx": empty_series,
                "plus_di": empty_series,
                "minus_di": empty_series,
                "regime": empty_regime,
                "trend_direction": empty_regime,
            }

    @staticmethod
    def volatility_regime(
        close: pd.Series, period: int = 20, threshold_factor: float = 1.5
    ) -> dict[str, pd.Series]:
        """
        Classify market regime based on volatility levels.

        Uses rolling standard deviation to identify:
        - High volatility periods (volatile regime)
        - Normal volatility periods (normal regime)
        - Low volatility periods (calm regime)

        Args:
            close: Series of closing prices
            period: Period for volatility calculation (default: 20)
            threshold_factor: Factor for volatility thresholds (default: 1.5)

        Returns:
            Dictionary containing:
            - volatility: Rolling volatility values
            - volatility_percentile: Volatility percentile ranking
            - regime: Volatility-based regime classification
            - vol_z_score: Z-score of current volatility

        Example:
            >>> regime = MarketRegimeDetector.volatility_regime(close)
            >>> high_vol_periods = regime['regime'] == 'high_volatility'
        """
        try:
            # Input validation
            if not isinstance(close, pd.Series):
                raise ValueError("close must be a pandas Series")

            if len(close) < period * 2:
                logger.warning(
                    f"Insufficient data for volatility regime: {len(close)} < {period * 2}"
                )
                empty_series = pd.Series(index=close.index, dtype=float)
                empty_regime = pd.Series(index=close.index, dtype=str)
                return {
                    "volatility": empty_series,
                    "volatility_percentile": empty_series,
                    "regime": empty_regime,
                    "vol_z_score": empty_series,
                }

            # Calculate returns
            returns = close.pct_change().dropna()

            # Calculate rolling volatility (annualized)
            volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100

            # Calculate volatility percentiles over longer period
            lookback_period = min(252, len(volatility))  # 1 year or available data
            volatility_percentile = (
                volatility.rolling(window=lookback_period).rank(pct=True) * 100
            )

            # Calculate z-score of volatility
            vol_mean = volatility.rolling(window=lookback_period).mean()
            vol_std = volatility.rolling(window=lookback_period).std()
            vol_z_score = (volatility - vol_mean) / vol_std

            # Classify volatility regime
            regime = pd.Series(index=close.index, dtype=str)

            # Use percentile-based classification
            high_vol_threshold = 75  # 75th percentile
            low_vol_threshold = 25  # 25th percentile

            regime = np.where(
                volatility_percentile >= high_vol_threshold,
                "high_volatility",
                np.where(
                    volatility_percentile <= low_vol_threshold,
                    "low_volatility",
                    "normal_volatility",
                ),
            )
            regime = pd.Series(regime, index=close.index)

            # Alternative z-score based classification for more dynamic thresholds
            regime_zscore = np.where(
                vol_z_score >= threshold_factor,
                "high_volatility",
                np.where(
                    vol_z_score <= -threshold_factor,
                    "low_volatility",
                    "normal_volatility",
                ),
            )

            return {
                "volatility": volatility,
                "volatility_percentile": volatility_percentile,
                "regime": regime,
                "vol_z_score": vol_z_score,
            }

        except Exception as e:
            logger.error(f"Error calculating volatility regime: {str(e)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            empty_regime = pd.Series(index=close.index, dtype=str)
            return {
                "volatility": empty_series,
                "volatility_percentile": empty_series,
                "regime": empty_regime,
                "vol_z_score": empty_series,
            }

    @staticmethod
    def price_action_regime(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> dict[str, pd.Series]:
        """
        Classify market regime based on price action patterns.

        Analyzes price movement patterns to identify:
        - Trending markets (consistent directional movement)
        - Range-bound markets (price oscillating in range)
        - Breakout markets (breaking out of ranges)

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for analysis (default: 20)

        Returns:
            Dictionary containing:
            - price_range: Rolling price range percentage
            - trend_consistency: Measure of trend consistency
            - regime: Price action regime classification
            - range_position: Position within recent range (0-100%)

        Example:
            >>> regime = MarketRegimeDetector.price_action_regime(high, low, close)
            >>> trending_periods = regime['regime'] == 'trending'
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if len(close) < period:
                logger.warning(
                    f"Insufficient data for price action regime: {len(close)} < {period}"
                )
                empty_series = pd.Series(index=close.index, dtype=float)
                empty_regime = pd.Series(index=close.index, dtype=str)
                return {
                    "price_range": empty_series,
                    "trend_consistency": empty_series,
                    "regime": empty_regime,
                    "range_position": empty_series,
                }

            # Calculate rolling price range
            rolling_high = high.rolling(window=period).max()
            rolling_low = low.rolling(window=period).min()
            price_range = ((rolling_high - rolling_low) / rolling_low) * 100

            # Calculate trend consistency
            returns = close.pct_change()
            rolling_returns = returns.rolling(window=period)

            # Measure how consistent the direction is
            positive_returns = (returns > 0).rolling(window=period).sum()
            trend_consistency = abs(positive_returns - period / 2) / (period / 2) * 100

            # Calculate range position (where current price sits in recent range)
            range_position = (
                (close - rolling_low) / (rolling_high - rolling_low)
            ) * 100

            # Calculate price momentum
            price_momentum = ((close - close.shift(period)) / close.shift(period)) * 100

            # Classify regime
            regime = pd.Series(index=close.index, dtype=str)

            # Trending: high trend consistency and significant momentum
            trending_condition = (trend_consistency >= 40) & (abs(price_momentum) >= 5)

            # Range-bound: low trend consistency and small momentum
            ranging_condition = (trend_consistency <= 20) & (abs(price_momentum) <= 3)

            # Breakout: high momentum with expanding range
            breakout_condition = (abs(price_momentum) >= 8) & (
                price_range >= price_range.shift(5)
            )

            regime = np.where(
                breakout_condition,
                "breakout",
                np.where(
                    trending_condition,
                    "trending",
                    np.where(ranging_condition, "ranging", "transitional"),
                ),
            )
            regime = pd.Series(regime, index=close.index)

            return {
                "price_range": price_range,
                "trend_consistency": trend_consistency,
                "regime": regime,
                "range_position": range_position,
            }

        except Exception as e:
            logger.error(f"Error calculating price action regime: {str(e)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            empty_regime = pd.Series(index=close.index, dtype=str)
            return {
                "price_range": empty_series,
                "trend_consistency": empty_series,
                "regime": empty_regime,
                "range_position": empty_series,
            }

    @staticmethod
    def composite_regime(
        high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict[str, pd.Series]:
        """
        Create composite market regime using multiple methods.

        Combines ADX, volatility, and price action regimes to create
        a more robust overall market regime classification.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices

        Returns:
            Dictionary containing:
            - adx_regime: ADX-based regime
            - volatility_regime: Volatility-based regime
            - price_action_regime: Price action regime
            - composite_regime: Combined regime classification
            - regime_confidence: Confidence score for regime classification

        Example:
            >>> regime = MarketRegimeDetector.composite_regime(high, low, close)
            >>> current_regime = regime['composite_regime'].iloc[-1]
            >>> confidence = regime['regime_confidence'].iloc[-1]
        """
        try:
            # Get individual regimes
            adx_result = MarketRegimeDetector.adx_regime(high, low, close)
            vol_result = MarketRegimeDetector.volatility_regime(close)
            price_result = MarketRegimeDetector.price_action_regime(high, low, close)

            # Extract regime classifications
            adx_regime = adx_result["regime"]
            vol_regime = vol_result["regime"]
            price_regime = price_result["regime"]

            # Create composite regime
            composite_regime = pd.Series(index=close.index, dtype=str)
            regime_confidence = pd.Series(index=close.index, dtype=float)

            for i in range(len(close.index)):
                try:
                    adx_reg = adx_regime.iloc[i] if i < len(adx_regime) else "unknown"
                    vol_reg = vol_regime.iloc[i] if i < len(vol_regime) else "unknown"
                    price_reg = (
                        price_regime.iloc[i] if i < len(price_regime) else "unknown"
                    )

                    # Regime determination logic
                    regime_votes = {
                        "trending": 0,
                        "ranging": 0,
                        "volatile": 0,
                        "breakout": 0,
                    }

                    # ADX vote
                    if "trending" in str(adx_reg):
                        regime_votes["trending"] += 2
                    elif "ranging" in str(adx_reg):
                        regime_votes["ranging"] += 2

                    # Volatility vote
                    if "high_volatility" in str(vol_reg):
                        regime_votes["volatile"] += 2
                    elif "low_volatility" in str(vol_reg):
                        regime_votes["ranging"] += 1

                    # Price action vote
                    if "trending" in str(price_reg):
                        regime_votes["trending"] += 2
                    elif "ranging" in str(price_reg):
                        regime_votes["ranging"] += 2
                    elif "breakout" in str(price_reg):
                        regime_votes["breakout"] += 3

                    # Determine winning regime
                    if regime_votes["breakout"] >= 3:
                        final_regime = "breakout"
                        confidence = min(100, regime_votes["breakout"] * 20)
                    elif regime_votes["volatile"] >= 2:
                        final_regime = "volatile"
                        confidence = min(100, regime_votes["volatile"] * 25)
                    elif regime_votes["trending"] >= regime_votes["ranging"]:
                        final_regime = "trending"
                        confidence = min(100, regime_votes["trending"] * 20)
                    else:
                        final_regime = "ranging"
                        confidence = min(100, regime_votes["ranging"] * 20)

                    composite_regime.iloc[i] = final_regime
                    regime_confidence.iloc[i] = confidence

                except Exception as e:
                    composite_regime.iloc[i] = "unknown"
                    regime_confidence.iloc[i] = 0.0

            return {
                "adx_regime": adx_regime,
                "volatility_regime": vol_regime,
                "price_action_regime": price_regime,
                "composite_regime": composite_regime,
                "regime_confidence": regime_confidence,
            }

        except Exception as e:
            logger.error(f"Error calculating composite regime: {str(e)}")
            empty_regime = pd.Series(index=close.index, dtype=str)
            empty_confidence = pd.Series(index=close.index, dtype=float)
            return {
                "adx_regime": empty_regime,
                "volatility_regime": empty_regime,
                "price_action_regime": empty_regime,
                "composite_regime": empty_regime,
                "regime_confidence": empty_confidence,
            }

    @staticmethod
    def regime_transitions(
        regime_series: pd.Series,
    ) -> dict[str, Union[pd.Series, int, float]]:
        """
        Analyze regime transitions and stability.

        Provides insights into how often regimes change and
        how stable the current regime is.

        Args:
            regime_series: Series of regime classifications

        Returns:
            Dictionary containing:
            - regime_changes: Boolean series indicating regime changes
            - regime_duration: Duration of current regime
            - transition_frequency: Average regime duration
            - stability_score: Regime stability score (0-100)

        Example:
            >>> transitions = MarketRegimeDetector.regime_transitions(regime_series)
            >>> current_stability = transitions['stability_score']
        """
        try:
            if regime_series.empty:
                return {
                    "regime_changes": pd.Series(dtype=bool),
                    "regime_duration": 0,
                    "transition_frequency": 0.0,
                    "stability_score": 0.0,
                }

            # Identify regime changes
            regime_changes = regime_series != regime_series.shift(1)
            regime_changes.iloc[0] = False  # First observation is not a change

            # Calculate current regime duration
            current_regime = regime_series.iloc[-1]
            regime_duration = 0

            for i in range(len(regime_series) - 1, -1, -1):
                if regime_series.iloc[i] == current_regime:
                    regime_duration += 1
                else:
                    break

            # Calculate transition frequency
            total_changes = regime_changes.sum()
            transition_frequency = (
                len(regime_series) / (total_changes + 1)
                if total_changes > 0
                else len(regime_series)
            )

            # Calculate stability score
            # Higher scores for longer average regime durations
            max_possible_duration = len(regime_series)
            stability_score = min(
                100, (transition_frequency / max_possible_duration) * 100
            )

            return {
                "regime_changes": regime_changes,
                "regime_duration": int(regime_duration),
                "transition_frequency": float(transition_frequency),
                "stability_score": float(stability_score),
            }

        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {str(e)}")
            return {
                "regime_changes": pd.Series(dtype=bool),
                "regime_duration": 0,
                "transition_frequency": 0.0,
                "stability_score": 0.0,
            }


def detect_market_regime(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, Union[str, float, dict]]:
    """
    Comprehensive market regime detection for ML features.

    Combines all regime detection methods to provide a complete
    picture of current market conditions.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices

    Returns:
        Dictionary containing:
        - current_regime: Current market regime
        - regime_confidence: Confidence in regime classification
        - regime_features: Dictionary of regime-based ML features
        - regime_history: Recent regime classifications

    Example:
        >>> regime_info = detect_market_regime(high, low, close)
        >>> if regime_info['current_regime'] == 'trending':
        >>>     print(f"Trending market (confidence: {regime_info['regime_confidence']:.1f}%)")
    """
    try:
        # Get composite regime analysis
        regime_analysis = MarketRegimeDetector.composite_regime(high, low, close)

        # Current regime and confidence
        current_regime = (
            regime_analysis["composite_regime"].iloc[-1]
            if not regime_analysis["composite_regime"].empty
            else "unknown"
        )
        regime_confidence = (
            regime_analysis["regime_confidence"].iloc[-1]
            if not regime_analysis["regime_confidence"].empty
            else 0.0
        )

        # Analyze regime transitions
        transitions = MarketRegimeDetector.regime_transitions(
            regime_analysis["composite_regime"]
        )

        # Create ML features
        regime_features = {
            "is_trending": 1.0 if current_regime == "trending" else 0.0,
            "is_ranging": 1.0 if current_regime == "ranging" else 0.0,
            "is_volatile": 1.0 if current_regime == "volatile" else 0.0,
            "is_breakout": 1.0 if current_regime == "breakout" else 0.0,
            "regime_confidence": regime_confidence,
            "regime_stability": transitions["stability_score"],
            "regime_duration": transitions["regime_duration"],
            "days_since_regime_change": transitions["regime_duration"],
        }

        # Recent regime history (last 10 periods)
        regime_history = regime_analysis["composite_regime"].tail(10).tolist()

        return {
            "current_regime": current_regime,
            "regime_confidence": regime_confidence,
            "regime_features": regime_features,
            "regime_history": regime_history,
            "full_analysis": regime_analysis,
        }

    except Exception as e:
        logger.error(f"Error in comprehensive regime detection: {str(e)}")
        return {
            "current_regime": "unknown",
            "regime_confidence": 0.0,
            "regime_features": {},
            "regime_history": [],
            "full_analysis": {},
        }
