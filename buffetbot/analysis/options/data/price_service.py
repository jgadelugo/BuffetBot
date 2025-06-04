"""
Price data service implementation.

This module provides concrete implementation for price data access
using existing data sources like yfinance.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from buffetbot.utils.errors import DataError, ErrorSeverity, handle_data_error
from buffetbot.utils.logger import setup_logger

from ..core.exceptions import DataSourceError, ErrorContext, InsufficientDataError
from .repositories import PriceRepository

logger = setup_logger(__name__, "logs/price_service.log")


class YFinancePriceService(PriceRepository):
    """Price service implementation using yfinance."""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache = {}

    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.Series:
        """
        Fetch historical price data using yfinance.

        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data

        Returns:
            pd.Series: Historical closing prices

        Raises:
            DataSourceError: If price data cannot be fetched
            InsufficientDataError: If insufficient data points
        """
        logger.info(f"Fetching {period} price history for {ticker}")

        # Check cache first
        cache_key = f"{ticker}:{period}"
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Returning cached price data for {ticker}")
            return self._cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                context = ErrorContext(ticker=ticker, strategy="price_fetch")
                raise DataSourceError(
                    f"No price data available for {ticker}",
                    context=context,
                    source_name="yfinance",
                )

            if len(hist) < 30:  # Minimum data points for meaningful analysis
                context = ErrorContext(
                    ticker=ticker,
                    strategy="price_fetch",
                    additional_data={
                        "days_available": len(hist),
                        "minimum_required": 30,
                    },
                )
                raise InsufficientDataError(
                    f"Insufficient price data for {ticker}: {len(hist)} days (minimum 30 required)",
                    context=context,
                    data_points=len(hist),
                    required_points=30,
                )

            prices = hist["Close"]

            # Cache the result
            if self.cache_enabled:
                self._cache[cache_key] = prices

            logger.info(
                f"Successfully fetched {len(hist)} days of price data for {ticker}"
            )
            return prices

        except Exception as e:
            if isinstance(e, (DataSourceError, InsufficientDataError)):
                raise

            context = ErrorContext(ticker=ticker, strategy="price_fetch")
            raise DataSourceError(
                f"Failed to fetch price data for {ticker}: {str(e)}",
                context=context,
                source_name="yfinance",
            )

    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            float: Current price

        Raises:
            DataSourceError: If current price cannot be fetched
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Try different price fields
            current_price = None
            for price_field in ["currentPrice", "regularMarketPrice", "previousClose"]:
                if price_field in info and info[price_field] is not None:
                    current_price = float(info[price_field])
                    break

            if current_price is None:
                # Fallback to latest price from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])

            if current_price is None or current_price <= 0:
                context = ErrorContext(ticker=ticker, strategy="current_price_fetch")
                raise DataSourceError(
                    f"Could not fetch valid current price for {ticker}",
                    context=context,
                    source_name="yfinance",
                )

            logger.debug(f"Current price for {ticker}: ${current_price:.2f}")
            return current_price

        except Exception as e:
            if isinstance(e, DataSourceError):
                raise

            context = ErrorContext(ticker=ticker, strategy="current_price_fetch")
            raise DataSourceError(
                f"Failed to fetch current price for {ticker}: {str(e)}",
                context=context,
                source_name="yfinance",
            )

    def clear_cache(self) -> None:
        """Clear the price data cache."""
        self._cache.clear()
        logger.debug("Price data cache cleared")


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute percentage returns from price series.

    Args:
        prices: Series of price data

    Returns:
        pd.Series: Percentage returns

    Raises:
        DataSourceError: If return calculation fails
    """
    try:
        if len(prices) < 2:
            raise DataSourceError(
                "Need at least 2 data points to calculate returns",
                source_name="returns_calculation",
            )

        returns = prices.pct_change().dropna()

        if returns.empty:
            raise DataSourceError(
                "No valid returns could be calculated",
                source_name="returns_calculation",
            )

        logger.debug(f"Computed {len(returns)} return values from {len(prices)} prices")
        return returns

    except Exception as e:
        if isinstance(e, DataSourceError):
            raise

        raise DataSourceError(
            f"Failed to compute returns: {str(e)}", source_name="returns_calculation"
        )
