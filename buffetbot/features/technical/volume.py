"""
Volume Indicators Module

Professional volume-based technical analysis indicators for financial markets.
Includes On-Balance Volume, Accumulation/Distribution Line, Volume Rate of Change,
and other volume-based indicators with proper error handling.

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


class VolumeIndicators:
    """
    Volume-based technical indicators.

    This class provides static methods for calculating various volume-based
    indicators used in technical analysis. All methods include proper
    error handling, input validation, and edge case management.
    """

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average.

        Args:
            volume: Series of volume data
            period: Period for moving average calculation

        Returns:
            Series of volume SMA values
        """
        try:
            if not isinstance(volume, pd.Series):
                raise ValueError("volume must be a pandas Series")

            if len(volume) < period:
                logger.warning("Insufficient data for Volume SMA calculation")
                return pd.Series(index=volume.index, dtype=float)

            clean_volume = volume.dropna()
            volume_sma = talib.SMA(clean_volume.values.astype(float), timeperiod=period)

            result = pd.Series(index=volume.index, dtype=float)
            result.loc[clean_volume.index] = volume_sma

            return result

        except Exception as e:
            logger.error(f"Error calculating Volume SMA: {str(e)}")
            return pd.Series(index=volume.index, dtype=float)

    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a momentum indicator that uses volume flow to predict
        changes in stock price.

        Args:
            close: Series of closing prices
            volume: Series of volume data

        Returns:
            Series of OBV values
        """
        try:
            if not all(isinstance(series, pd.Series) for series in [close, volume]):
                raise ValueError("close and volume must be pandas Series")

            if len(close) != len(volume):
                raise ValueError("close and volume must have the same length")

            if len(close) < 2:
                logger.warning("Insufficient data for OBV calculation")
                return pd.Series(index=close.index, dtype=float)

            # Align indices and remove NaN values
            df = pd.DataFrame({"close": close, "volume": volume}).dropna()

            if len(df) < 2:
                logger.warning("Insufficient clean data for OBV calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate OBV using TA-Lib
            obv_values = talib.OBV(
                df["close"].values.astype(float), df["volume"].values.astype(float)
            )

            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = obv_values

            return result

        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def volume_rate_of_change(volume: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Volume Rate of Change.

        Volume ROC measures the percentage change in volume over a specified period.

        Args:
            volume: Series of volume data
            period: Period for ROC calculation

        Returns:
            Series of Volume ROC values
        """
        try:
            if not isinstance(volume, pd.Series):
                raise ValueError("volume must be a pandas Series")

            if len(volume) < period + 1:
                logger.warning("Insufficient data for Volume ROC calculation")
                return pd.Series(index=volume.index, dtype=float)

            clean_volume = volume.dropna()

            if len(clean_volume) < period + 1:
                logger.warning("Insufficient clean data for Volume ROC calculation")
                return pd.Series(index=volume.index, dtype=float)

            # Calculate Volume ROC using TA-Lib
            vol_roc = talib.ROC(clean_volume.values.astype(float), timeperiod=period)

            result = pd.Series(index=volume.index, dtype=float)
            result.loc[clean_volume.index] = vol_roc

            return result

        except Exception as e:
            logger.error(f"Error calculating Volume ROC: {str(e)}")
            return pd.Series(index=volume.index, dtype=float)

    @staticmethod
    def accumulation_distribution(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        The A/D Line is a volume-based indicator designed to measure
        the cumulative flow of money into and out of a security.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data

        Returns:
            Series of A/D Line values
        """
        try:
            if not all(
                isinstance(series, pd.Series) for series in [high, low, close, volume]
            ):
                raise ValueError("high, low, close, and volume must be pandas Series")

            if not (len(high) == len(low) == len(close) == len(volume)):
                raise ValueError("All series must have the same length")

            if len(close) < 1:
                logger.warning("Insufficient data for A/D calculation")
                return pd.Series(index=close.index, dtype=float)

            # Align indices and remove NaN values
            df = pd.DataFrame(
                {"high": high, "low": low, "close": close, "volume": volume}
            ).dropna()

            if len(df) < 1:
                logger.warning("Insufficient clean data for A/D calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate A/D Line using TA-Lib
            ad_values = talib.AD(
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
            )

            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = ad_values

            return result

        except Exception as e:
            logger.error(f"Error calculating A/D Line: {str(e)}")
            return pd.Series(index=close.index, dtype=float)

    @staticmethod
    def volume_weighted_average_price(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        VWAP is the average price a security has traded at throughout the day,
        based on both volume and price.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            period: Rolling window period

        Returns:
            Series of VWAP values
        """
        try:
            if not all(
                isinstance(series, pd.Series) for series in [high, low, close, volume]
            ):
                raise ValueError("All inputs must be pandas Series")

            if not (len(high) == len(low) == len(close) == len(volume)):
                raise ValueError("All series must have the same length")

            # Align indices and remove NaN values
            df = pd.DataFrame(
                {"high": high, "low": low, "close": close, "volume": volume}
            ).dropna()

            if len(df) < period:
                logger.warning("Insufficient data for VWAP calculation")
                return pd.Series(index=close.index, dtype=float)

            # Calculate typical price
            typical_price = (df["high"] + df["low"] + df["close"]) / 3

            # Calculate VWAP using rolling windows
            price_volume = typical_price * df["volume"]
            vwap = (
                price_volume.rolling(window=period).sum()
                / df["volume"].rolling(window=period).sum()
            )

            result = pd.Series(index=close.index, dtype=float)
            result.loc[df.index] = vwap

            return result

        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(index=close.index, dtype=float)
