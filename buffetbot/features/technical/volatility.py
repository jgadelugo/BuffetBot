"""
Volatility Indicators Module

Professional volatility and risk-based indicators for financial markets.
Includes Average True Range, Historical Volatility, and Volatility Ratio
with proper error handling and edge case management.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import talib

# Configure logging
logger = logging.getLogger(__name__)


class VolatilityIndicators:
    """
    Volatility and risk-based indicators.

    This class provides static methods for calculating various volatility
    indicators used in technical analysis. All methods include proper
    error handling, input validation, and edge case management.
    """

    @staticmethod
    def average_true_range(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of an asset
        price for that period. Higher ATR values indicate higher volatility.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for ATR calculation

        Returns:
            Series of ATR values
        """
        try:
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if not (len(high) == len(low) == len(close)):
                raise ValueError("high, low, and close must have the same length")

            if len(close) < period + 1:
                logger.warning("Insufficient data for ATR calculation")
                return pd.Series(index=close.index, dtype=float)

            # Align indices and remove NaN values
            df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()

            if len(df) < period + 1:
                logger.warning("Insufficient clean data for ATR calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate ATR using TA-Lib
            atr_values = talib.ATR(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                timeperiod=period,
            )

            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = atr_values

            return result

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def historical_volatility(
        prices: pd.Series, period: int = 20, annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Historical Volatility.

        Historical volatility measures the standard deviation of price returns
        over a specified period, providing a measure of price dispersion.

        Args:
            prices: Series of closing prices
            period: Period for volatility calculation
            annualize: Whether to annualize the volatility (default: True)

        Returns:
            Series of historical volatility values
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period + 1:
                logger.warning(
                    "Insufficient data for historical volatility calculation"
                )
                return pd.Series(index=prices.index, dtype=float)

            clean_prices = prices.dropna()

            if len(clean_prices) < period + 1:
                logger.warning(
                    "Insufficient clean data for historical volatility calculation"
                )
                return pd.Series(index=prices.index, dtype=float)

            # Calculate returns
            returns = clean_prices.pct_change().dropna()

            # Calculate rolling standard deviation
            volatility = returns.rolling(window=period).std()

            # Annualize if requested (multiply by sqrt(252) for daily data)
            if annualize:
                volatility = volatility * np.sqrt(252)

            # Align with original index
            result = pd.Series(index=prices.index, dtype=float)
            result.loc[volatility.index] = volatility

            return result

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def volatility_ratio(
        prices: pd.Series, short_period: int = 10, long_period: int = 30
    ) -> pd.Series:
        """
        Calculate Volatility Ratio for regime detection.

        Volatility ratio compares short-term to long-term volatility to identify
        changes in market regimes. Values > 1 indicate increasing volatility.

        Args:
            prices: Series of closing prices
            short_period: Short-term period for volatility calculation
            long_period: Long-term period for volatility calculation

        Returns:
            Series of volatility ratio values
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if short_period >= long_period:
                raise ValueError("short_period must be less than long_period")

            if len(prices) < long_period + 1:
                logger.warning("Insufficient data for volatility ratio calculation")
                return pd.Series(index=prices.index, dtype=float)

            # Calculate short and long-term volatility
            short_vol = VolatilityIndicators.historical_volatility(
                prices, period=short_period, annualize=False
            )
            long_vol = VolatilityIndicators.historical_volatility(
                prices, period=long_period, annualize=False
            )

            # Calculate ratio
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = short_vol / long_vol
                ratio = ratio.replace([np.inf, -np.inf], np.nan)

            return ratio

        except Exception as e:
            logger.error(f"Error calculating volatility ratio: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate True Range.

        True Range is the greatest of:
        - Current High less Current Low
        - Current High less Previous Close (absolute value)
        - Current Low less Previous Close (absolute value)

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices

        Returns:
            Series of True Range values
        """
        try:
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if not (len(high) == len(low) == len(close)):
                raise ValueError("high, low, and close must have the same length")

            if len(close) < 2:
                logger.warning("Insufficient data for True Range calculation")
                return pd.Series(index=close.index, dtype=float)

            # Align indices and remove NaN values
            df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()

            if len(df) < 2:
                logger.warning("Insufficient clean data for True Range calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate True Range using TA-Lib
            tr_values = talib.TRANGE(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
            )

            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = tr_values

            return result

        except Exception as e:
            logger.error(f"Error calculating True Range: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def normalized_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Normalized Average True Range (ATR%).

        Normalized ATR expresses ATR as a percentage of the closing price,
        making it easier to compare volatility across different price levels.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for ATR calculation

        Returns:
            Series of normalized ATR values (as percentages)
        """
        try:
            # Calculate regular ATR
            atr = VolatilityIndicators.average_true_range(high, low, close, period)

            # Normalize by closing price
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_atr = (atr / close) * 100
                normalized_atr = normalized_atr.replace([np.inf, -np.inf], np.nan)

            return normalized_atr

        except Exception as e:
            logger.error(f"Error calculating normalized ATR: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def volatility_bands(
        prices: pd.Series, period: int = 20, multiplier: float = 2.0
    ) -> dict[str, pd.Series]:
        """
        Calculate Volatility Bands based on ATR.

        Similar to Bollinger Bands but uses ATR instead of standard deviation
        to create dynamic support and resistance levels.

        Args:
            prices: Series of closing prices
            period: Period for calculations
            multiplier: Multiplier for ATR bands

        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        try:
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < period + 1:
                logger.warning("Insufficient data for volatility bands calculation")
                empty_series = pd.Series(index=prices.index, dtype=float)
                return {
                    "upper": empty_series,
                    "middle": empty_series,
                    "lower": empty_series,
                }

            # Calculate middle line (SMA)
            middle = talib.SMA(prices.dropna().values.astype(float), timeperiod=period)
            middle_series = pd.Series(index=prices.index, dtype=float)
            middle_series.loc[prices.dropna().index] = middle

            # For volatility bands, we need high, low, close data
            # If only prices provided, approximate with prices as all values
            atr_series = VolatilityIndicators.average_true_range(
                prices, prices, prices, period
            )

            # Calculate bands
            upper_band = middle_series + (multiplier * atr_series)
            lower_band = middle_series - (multiplier * atr_series)

            return {"upper": upper_band, "middle": middle_series, "lower": lower_band}

        except Exception as e:
            logger.error(f"Error calculating volatility bands: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                "upper": empty_series,
                "middle": empty_series,
                "lower": empty_series,
            }
