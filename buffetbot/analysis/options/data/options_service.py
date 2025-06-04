"""
Options data service implementation.

This module provides concrete implementation for options data access
using the existing options fetcher.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from buffetbot.data.options_fetcher import (
    OptionsDataError,
    OptionsResult,
    fetch_long_dated_calls,
)
from buffetbot.utils.logger import setup_logger

from ..core.domain_models import OptionsData
from ..core.exceptions import DataSourceError, ErrorContext
from .repositories import OptionsRepository

logger = setup_logger(__name__, "logs/options_service.log")


class DefaultOptionsService(OptionsRepository):
    """Options service implementation using existing options fetcher."""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache = {}

    def fetch_options_data(self, ticker: str, min_days: int = 180) -> OptionsData:
        """
        Fetch options data using the legacy options fetcher.

        Args:
            ticker: Stock ticker symbol
            min_days: Minimum days to expiration

        Returns:
            OptionsData: Structured options data

        Raises:
            DataSourceError: If data fetching fails
        """
        logger.info(f"Fetching options data for {ticker} with min_days={min_days}")

        try:
            # Use legacy fetcher - it returns a dictionary with structure:
            # {'data': DataFrame, 'data_available': bool, 'error_message': str, ...}
            from buffetbot.data.options_fetcher import fetch_long_dated_calls

            result = fetch_long_dated_calls(ticker, min_days)

            # Check if data is available and extract DataFrame
            if not result.get("data_available", False):
                error_msg = result.get("error_message", "Unknown error")
                raise DataSourceError(
                    f"Options fetcher returned no data: {error_msg}",
                    ErrorContext(ticker=ticker, additional_data={"min_days": min_days}),
                )

            options_df = result.get("data")
            if options_df is None or options_df.empty:
                raise DataSourceError(
                    f"Empty options data returned for {ticker}",
                    ErrorContext(ticker=ticker, additional_data={"min_days": min_days}),
                )

            # Calculate metadata
            total_volume = (
                float(options_df["volume"].sum())
                if "volume" in options_df.columns
                else 0.0
            )
            avg_iv = self._calculate_average_iv(options_df)

            return OptionsData(
                options_df=options_df,
                total_volume=total_volume,
                avg_iv=avg_iv,
                source=result.get("source_used", "unknown"),
                fetch_time=datetime.now(),
            )

        except DataSourceError:
            raise  # Re-raise our custom errors
        except Exception as e:
            context = ErrorContext(
                ticker=ticker,
                additional_data={"min_days": min_days, "original_error": str(e)},
            )
            raise DataSourceError(
                f"Failed to fetch options data for {ticker}: {str(e)}", context=context
            )

    def _calculate_average_iv(self, options_df: pd.DataFrame) -> float:
        """
        Calculate volume-weighted average implied volatility.

        Args:
            options_df: Options DataFrame

        Returns:
            float: Average implied volatility
        """
        if "impliedVolatility" not in options_df.columns:
            return 0.25  # Default IV if not available

        iv_data = options_df["impliedVolatility"].dropna()
        if iv_data.empty:
            return 0.25

        # Use volume weighting if available
        if "volume" in options_df.columns:
            volume_data = options_df.loc[iv_data.index, "volume"].fillna(1.0)
            if volume_data.sum() > 0:
                return float((iv_data * volume_data).sum() / volume_data.sum())

        # Simple average if no volume data
        return float(iv_data.mean())

    def fetch_put_options(self, ticker: str, min_days: int) -> OptionsResult:
        """
        Fetch put options data.

        Args:
            ticker: Stock ticker symbol
            min_days: Minimum days to expiry

        Returns:
            OptionsResult: Put options data result

        Raises:
            DataSourceError: If put options data cannot be fetched
        """
        logger.info(f"Fetching put options data for {ticker} with min_days={min_days}")

        # Check cache first
        cache_key = f"{ticker}:{min_days}:puts"
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Returning cached put options data for {ticker}")
            return self._cache[cache_key]

        try:
            # Import the put options fetcher from the original module
            from buffetbot.analysis.options_advisor import fetch_put_options

            result = fetch_put_options(ticker, min_days_to_expiry=min_days)

            # Validate result
            if result.options_df.empty:
                context = ErrorContext(ticker=ticker, strategy="put_options_fetch")
                raise DataSourceError(
                    f"No put options data available for {ticker}",
                    context=context,
                    source_name="options_fetcher",
                )

            # Cache the result
            if self.cache_enabled:
                self._cache[cache_key] = result

            logger.info(
                f"Successfully fetched {len(result.options_df)} put options for {ticker}"
            )
            return result

        except OptionsDataError as e:
            context = ErrorContext(ticker=ticker, strategy="put_options_fetch")
            raise DataSourceError(
                f"Put options data error for {ticker}: {str(e)}",
                context=context,
                source_name="options_fetcher",
            )
        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="put_options_fetch")
            raise DataSourceError(
                f"Failed to fetch put options data for {ticker}: {str(e)}",
                context=context,
                source_name="options_fetcher",
            )

    def clear_cache(self) -> None:
        """Clear the options data cache."""
        self._cache.clear()
        logger.debug("Options data cache cleared")
