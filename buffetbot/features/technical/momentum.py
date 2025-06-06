"""
Momentum Indicators Module

Professional momentum indicator calculations for financial analysis.
Includes RSI, MACD, Stochastic Oscillator, and Williams %R with proper
error handling and edge case management.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import talib

# Configure logging
logger = logging.getLogger(__name__)


class MomentumIndicators:
    """
    Professional momentum indicator calculations.

    This class provides static methods for calculating various momentum
    indicators used in technical analysis. All methods include proper
    error handling, input validation, and edge case management.
    """

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes. Values above 70 typically indicate overbought
        conditions, while values below 30 indicate oversold conditions.

        Args:
            prices: Series of closing prices
            period: Number of periods for RSI calculation (default: 14)

        Returns:
            Series of RSI values (0-100 scale)

        Raises:
            ValueError: If invalid parameters provided

        Example:
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
            >>> rsi = MomentumIndicators.rsi(prices, period=6)
        """
        try:
            # Input validation
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period + 1:
                logger.warning(
                    f"Insufficient data: {len(prices)} points, need {period + 1}"
                )
                return pd.Series(index=prices.index, dtype=float)

            if period <= 0:
                raise ValueError("period must be positive")

            # Remove any NaN values
            clean_prices = prices.dropna()

            if len(clean_prices) < period + 1:
                logger.warning("Insufficient clean data after removing NaN values")
                return pd.Series(index=prices.index, dtype=float)

            # Calculate RSI using TA-Lib for accuracy
            rsi_values = talib.RSI(clean_prices.values.astype(float), timeperiod=period)

            # Create result series with original index
            result = pd.Series(index=prices.index, dtype=float)
            result.loc[clean_prices.index] = rsi_values

            return result

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.

        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)

        Returns:
            Dictionary containing:
            - 'macd': MACD line (fast EMA - slow EMA)
            - 'signal': Signal line (EMA of MACD)
            - 'histogram': MACD histogram (MACD - Signal)

        Example:
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
            >>> macd_data = MomentumIndicators.macd(prices)
            >>> macd_line = macd_data['macd']
        """
        try:
            # Input validation
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if fast >= slow:
                raise ValueError("fast period must be less than slow period")

            if len(prices) < slow + signal:
                logger.warning(f"Insufficient data for MACD calculation")
                empty_series = pd.Series(index=prices.index, dtype=float)
                return {
                    "macd": empty_series,
                    "signal": empty_series,
                    "histogram": empty_series,
                }

            # Remove NaN values
            clean_prices = prices.dropna()

            if len(clean_prices) < slow + signal:
                logger.warning("Insufficient clean data for MACD calculation")
                empty_series = pd.Series(index=prices.index, dtype=float)
                return {
                    "macd": empty_series,
                    "signal": empty_series,
                    "histogram": empty_series,
                }

            # Calculate MACD using TA-Lib
            macd_line, signal_line, histogram = talib.MACD(
                clean_prices.values.astype(float),
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )

            # Create result series with original index
            result_macd = pd.Series(index=prices.index, dtype=float)
            result_signal = pd.Series(index=prices.index, dtype=float)
            result_histogram = pd.Series(index=prices.index, dtype=float)

            result_macd.loc[clean_prices.index] = macd_line
            result_signal.loc[clean_prices.index] = signal_line
            result_histogram.loc[clean_prices.index] = histogram

            return {
                "macd": result_macd,
                "signal": result_signal,
                "histogram": result_histogram,
            }

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                "macd": empty_series,
                "signal": empty_series,
                "histogram": empty_series,
            }

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).

        The stochastic oscillator compares a particular closing price of a security
        to a range of its prices over a certain period of time.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D (moving average of %K) (default: 3)

        Returns:
            Dictionary containing:
            - 'k': %K values (fast stochastic)
            - 'd': %D values (slow stochastic, SMA of %K)

        Example:
            >>> high = pd.Series([105, 107, 106, 108, 110, 109, 111])
            >>> low = pd.Series([98, 100, 99, 101, 103, 102, 104])
            >>> close = pd.Series([102, 104, 103, 105, 107, 106, 108])
            >>> stoch = MomentumIndicators.stochastic(high, low, close)
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if not (len(high) == len(low) == len(close)):
                raise ValueError("high, low, and close must have the same length")

            if len(close) < k_period + d_period:
                logger.warning("Insufficient data for Stochastic calculation")
                empty_series = pd.Series(index=close.index, dtype=float)
                return {"k": empty_series, "d": empty_series}

            # Align indices and remove NaN values
            df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()

            if len(df) < k_period + d_period:
                logger.warning("Insufficient clean data for Stochastic calculation")
                empty_series = pd.Series(index=close.index, dtype=float)
                return {"k": empty_series, "d": empty_series}

            # Calculate Stochastic using TA-Lib
            slowk, slowd = talib.STOCH(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                fastk_period=k_period,
                slowk_period=3,  # Standard smoothing
                slowk_matype=0,  # Simple moving average
                slowd_period=d_period,
                slowd_matype=0,  # Simple moving average
            )

            # Create result series with original index
            result_k = pd.Series(index=close.index, dtype=float)
            result_d = pd.Series(index=close.index, dtype=float)

            result_k.loc[df.index] = slowk
            result_d.loc[df.index] = slowd

            return {"k": result_k, "d": result_d}

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            return {"k": empty_series, "d": empty_series}

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R.

        Williams %R is a momentum indicator that measures overbought and oversold levels.
        Values typically range from -100 to 0, with readings above -20 considered
        overbought and readings below -80 considered oversold.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Number of periods for calculation (default: 14)

        Returns:
            Series of Williams %R values (-100 to 0 scale)

        Example:
            >>> high = pd.Series([105, 107, 106, 108, 110, 109, 111])
            >>> low = pd.Series([98, 100, 99, 101, 103, 102, 104])
            >>> close = pd.Series([102, 104, 103, 105, 107, 106, 108])
            >>> wr = MomentumIndicators.williams_r(high, low, close)
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if not (len(high) == len(low) == len(close)):
                raise ValueError("high, low, and close must have the same length")

            if len(close) < period:
                logger.warning("Insufficient data for Williams %R calculation")
                return pd.Series(index=close.index, dtype=float)

            if period <= 0:
                raise ValueError("period must be positive")

            # Align indices and remove NaN values
            df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()

            if len(df) < period:
                logger.warning("Insufficient clean data for Williams %R calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate Williams %R using TA-Lib
            wr_values = talib.WILLR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=period,
            )

            # Create result series with original index
            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = wr_values

            return result

        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Price Momentum.

        Momentum measures the rate of change in price over a specified period.
        Positive values indicate upward momentum, negative values indicate downward momentum.

        Args:
            prices: Series of closing prices
            period: Number of periods for momentum calculation (default: 10)

        Returns:
            Series of momentum values

        Example:
            >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
            >>> mom = MomentumIndicators.momentum(prices, period=5)
        """
        try:
            # Input validation
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period + 1:
                logger.warning("Insufficient data for Momentum calculation")
                return pd.Series(index=prices.index, dtype=float)

            if period <= 0:
                raise ValueError("period must be positive")

            # Remove NaN values
            clean_prices = prices.dropna()

            if len(clean_prices) < period + 1:
                logger.warning("Insufficient clean data for Momentum calculation")
                return pd.Series(index=prices.index, dtype=float)

            # Calculate Momentum using TA-Lib
            momentum_values = talib.MOM(
                clean_prices.values.astype(float), timeperiod=period
            )

            # Create result series with original index
            result = pd.Series(index=prices.index, dtype=float)
            result.loc[clean_prices.index] = momentum_values

            return result

        except Exception as e:
            logger.error(f"Error calculating Momentum: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)


# Helper functions for momentum analysis
def classify_momentum_signal(
    rsi: float, macd_histogram: float, williams_r: float
) -> str:
    """
    Classify momentum signals based on multiple indicators.

    Args:
        rsi: RSI value (0-100)
        macd_histogram: MACD histogram value
        williams_r: Williams %R value (-100 to 0)

    Returns:
        Signal classification: 'strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish'
    """
    try:
        # Handle NaN values
        if pd.isna(rsi) or pd.isna(macd_histogram) or pd.isna(williams_r):
            return "neutral"

        bullish_signals = 0
        bearish_signals = 0

        # RSI analysis
        if rsi > 70:
            bearish_signals += 1
        elif rsi < 30:
            bullish_signals += 1

        # MACD histogram analysis
        if macd_histogram > 0:
            bullish_signals += 1
        elif macd_histogram < 0:
            bearish_signals += 1

        # Williams %R analysis
        if williams_r > -20:
            bearish_signals += 1
        elif williams_r < -80:
            bullish_signals += 1

        # Classify signal strength
        if bullish_signals >= 2 and bearish_signals == 0:
            return "strong_bullish"
        elif bullish_signals > bearish_signals:
            return "bullish"
        elif bearish_signals >= 2 and bullish_signals == 0:
            return "strong_bearish"
        elif bearish_signals > bullish_signals:
            return "bearish"
        else:
            return "neutral"

    except Exception as e:
        logger.error(f"Error classifying momentum signal: {str(e)}")
        return "neutral"
