"""
Trend Indicators Module

Professional trend analysis indicators for financial markets.
Includes moving averages, Bollinger Bands, Ichimoku Cloud, and Parabolic SAR
with proper error handling and edge case management.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import talib

# Configure logging
logger = logging.getLogger(__name__)


class TrendIndicators:
    """
    Professional trend analysis indicators.

    This class provides static methods for calculating various trend-following
    indicators used in technical analysis. All methods include proper
    error handling, input validation, and edge case management.
    """

    @staticmethod
    def moving_averages(
        prices: pd.Series, windows: list[int] = [5, 10, 20, 50, 200]
    ) -> dict[str, pd.Series]:
        """
        Calculate multiple moving averages with crossover signals.

        Args:
            prices: Series of closing prices
            windows: List of periods for moving averages

        Returns:
            Dictionary containing SMA series for each window
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            max_window = max(windows) if windows else 20
            if len(prices) < max_window:
                logger.warning(
                    f"Insufficient data: {len(prices)} points, need {max_window}"
                )
                return {
                    f"sma_{w}": pd.Series(index=prices.index, dtype=float)
                    for w in windows
                }

            clean_prices = prices.dropna()
            result = {}

            for window in windows:
                sma_values = talib.SMA(
                    clean_prices.values.astype(float), timeperiod=window
                )
                sma_series = pd.Series(index=prices.index, dtype=float)
                sma_series.loc[clean_prices.index] = sma_values
                result[f"sma_{window}"] = sma_series

            return result

        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {
                f"sma_{w}": pd.Series(index=prices.index, dtype=float) for w in windows
            }

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> dict[str, pd.Series]:
        """
        Calculate Bollinger Bands with upper, lower bands and width.

        Args:
            prices: Series of closing prices
            period: Period for moving average and standard deviation
            std_dev: Number of standard deviations for bands

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'width', 'percent_b'
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period:
                logger.warning(f"Insufficient data for Bollinger Bands")
                empty_series = pd.Series(index=prices.index, dtype=float)
                return {
                    "upper": empty_series,
                    "middle": empty_series,
                    "lower": empty_series,
                    "width": empty_series,
                    "percent_b": empty_series,
                }

            clean_prices = prices.dropna()

            # Calculate Bollinger Bands using TA-Lib
            upper, middle, lower = talib.BBANDS(
                clean_prices.values.astype(float),
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0,
            )

            # Create result series
            result_upper = pd.Series(index=prices.index, dtype=float)
            result_middle = pd.Series(index=prices.index, dtype=float)
            result_lower = pd.Series(index=prices.index, dtype=float)

            result_upper.loc[clean_prices.index] = upper
            result_middle.loc[clean_prices.index] = middle
            result_lower.loc[clean_prices.index] = lower

            # Calculate additional metrics
            result_width = result_upper - result_lower

            # %B: Position within bands
            with np.errstate(divide="ignore", invalid="ignore"):
                result_percent_b = (prices - result_lower) / (
                    result_upper - result_lower
                )
                result_percent_b = result_percent_b.fillna(0.5)

            return {
                "upper": result_upper,
                "middle": result_middle,
                "lower": result_lower,
                "width": result_width,
                "percent_b": result_percent_b,
            }

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                "upper": empty_series,
                "middle": empty_series,
                "lower": empty_series,
                "width": empty_series,
                "percent_b": empty_series,
            }

    @staticmethod
    def exponential_moving_average(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: Series of closing prices
            period: Period for EMA calculation

        Returns:
            Series of EMA values
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period:
                logger.warning("Insufficient data for EMA calculation")
                return pd.Series(index=prices.index, dtype=float)

            clean_prices = prices.dropna()
            ema_values = talib.EMA(clean_prices.values.astype(float), timeperiod=period)

            result = pd.Series(index=prices.index, dtype=float)
            result.loc[clean_prices.index] = ema_values

            return result

        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)


# Helper functions for trend analysis
def classify_trend_strength(
    short_ma: float, long_ma: float, current_price: float, bollinger_position: float
) -> str:
    """
    Classify trend strength based on multiple indicators.

    Args:
        short_ma: Short-term moving average value
        long_ma: Long-term moving average value
        current_price: Current price
        bollinger_position: Position within Bollinger Bands (0-1)

    Returns:
        Trend classification: 'strong_uptrend', 'uptrend', 'sideways', 'downtrend', 'strong_downtrend'
    """
    try:
        # Handle NaN values
        if any(
            pd.isna(val)
            for val in [short_ma, long_ma, current_price, bollinger_position]
        ):
            return "sideways"

        # Calculate trend signals
        ma_trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
        price_vs_short = (current_price - short_ma) / short_ma if short_ma != 0 else 0

        # Strong uptrend: MA trend > 2%, price > short MA, BB position > 0.8
        if ma_trend > 0.02 and price_vs_short > 0.01 and bollinger_position > 0.8:
            return "strong_uptrend"
        # Uptrend: MA trend > 0%, price > short MA
        elif ma_trend > 0 and price_vs_short > 0:
            return "uptrend"
        # Strong downtrend: MA trend < -2%, price < short MA, BB position < 0.2
        elif ma_trend < -0.02 and price_vs_short < -0.01 and bollinger_position < 0.2:
            return "strong_downtrend"
        # Downtrend: MA trend < 0%, price < short MA
        elif ma_trend < 0 and price_vs_short < 0:
            return "downtrend"
        else:
            return "sideways"

    except Exception as e:
        logger.error(f"Error classifying trend strength: {str(e)}")
        return "sideways"


def detect_support_resistance(
    prices: pd.Series,
    bollinger_bands: dict[str, pd.Series],
    ichimoku: dict[str, pd.Series],
) -> dict[str, float]:
    """
    Detect support and resistance levels using multiple indicators.

    Args:
        prices: Series of closing prices
        bollinger_bands: Bollinger Bands data
        ichimoku: Ichimoku Cloud data

    Returns:
        Dictionary with support and resistance levels
    """
    try:
        current_price = prices.iloc[-1] if len(prices) > 0 else np.nan

        if pd.isna(current_price):
            return {"support": np.nan, "resistance": np.nan}

        support_levels = []
        resistance_levels = []

        # Bollinger Bands support/resistance
        if "lower" in bollinger_bands and not pd.isna(
            bollinger_bands["lower"].iloc[-1]
        ):
            support_levels.append(bollinger_bands["lower"].iloc[-1])
        if "upper" in bollinger_bands and not pd.isna(
            bollinger_bands["upper"].iloc[-1]
        ):
            resistance_levels.append(bollinger_bands["upper"].iloc[-1])

        # Ichimoku support/resistance
        if "kijun_sen" in ichimoku and not pd.isna(ichimoku["kijun_sen"].iloc[-1]):
            level = ichimoku["kijun_sen"].iloc[-1]
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)

        # Cloud support/resistance
        if "senkou_span_a" in ichimoku and "senkou_span_b" in ichimoku:
            span_a = ichimoku["senkou_span_a"].iloc[-1]
            span_b = ichimoku["senkou_span_b"].iloc[-1]
            if not pd.isna(span_a) and not pd.isna(span_b):
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)

                if current_price > cloud_top:
                    support_levels.append(cloud_top)
                elif current_price < cloud_bottom:
                    resistance_levels.append(cloud_bottom)

        return {
            "support": max(support_levels) if support_levels else np.nan,
            "resistance": min(resistance_levels) if resistance_levels else np.nan,
        }

    except Exception as e:
        logger.error(f"Error detecting support/resistance: {str(e)}")
        return {"support": np.nan, "resistance": np.nan}
