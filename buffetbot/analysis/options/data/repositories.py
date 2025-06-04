"""
Repository pattern implementation for data access.

This module provides abstract interfaces and concrete implementations
for accessing market data, following the repository pattern for clean
architecture and testability.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import pandas as pd

from buffetbot.data.options_fetcher import OptionsResult

from ..core.domain_models import MarketData
from ..core.exceptions import DataSourceError, ErrorContext


class DataRepository(Protocol):
    """Protocol defining the interface for data access."""

    def get_options_data(self, ticker: str, min_days: int) -> OptionsResult:
        """Get options data for a ticker."""
        ...

    def get_price_history(self, ticker: str, period: str) -> pd.Series:
        """Get historical price data for a ticker."""
        ...

    def get_spy_prices(self, period: str) -> pd.Series:
        """Get SPY price data for beta calculation."""
        ...

    def get_forecast_data(self, ticker: str) -> float:
        """Get analyst forecast confidence for a ticker."""
        ...


class OptionsRepository(ABC):
    """Abstract repository for options data access."""

    @abstractmethod
    def fetch_options_data(self, ticker: str, min_days: int) -> OptionsResult:
        """Fetch options data from data source."""
        pass

    @abstractmethod
    def fetch_put_options(self, ticker: str, min_days: int) -> OptionsResult:
        """Fetch put options data from data source."""
        pass


class PriceRepository(ABC):
    """Abstract repository for price data access."""

    @abstractmethod
    def fetch_price_history(self, ticker: str, period: str) -> pd.Series:
        """Fetch historical price data."""
        pass

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker."""
        pass


class ForecastRepository(ABC):
    """Abstract repository for forecast data access."""

    @abstractmethod
    def fetch_forecast_data(self, ticker: str) -> float:
        """Fetch analyst forecast confidence."""
        pass


class DefaultDataRepository:
    """Default implementation of DataRepository using existing fetchers."""

    def __init__(
        self,
        options_repo: OptionsRepository,
        price_repo: PriceRepository,
        forecast_repo: ForecastRepository,
    ):
        self.options_repo = options_repo
        self.price_repo = price_repo
        self.forecast_repo = forecast_repo

    def get_options_data(self, ticker: str, min_days: int) -> OptionsResult:
        """Get options data for a ticker."""
        try:
            return self.options_repo.fetch_options_data(ticker, min_days)
        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="data_fetch")
            raise DataSourceError(
                f"Failed to fetch options data for {ticker}: {str(e)}",
                context=context,
                source_name="options",
            )

    def get_price_history(self, ticker: str, period: str = "1y") -> pd.Series:
        """Get historical price data for a ticker."""
        try:
            return self.price_repo.fetch_price_history(ticker, period)
        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="data_fetch")
            raise DataSourceError(
                f"Failed to fetch price history for {ticker}: {str(e)}",
                context=context,
                source_name="price",
            )

    def get_spy_prices(self, period: str = "1y") -> pd.Series:
        """Get SPY price data for beta calculation."""
        try:
            return self.price_repo.fetch_price_history("SPY", period)
        except Exception as e:
            context = ErrorContext(ticker="SPY", strategy="data_fetch")
            raise DataSourceError(
                f"Failed to fetch SPY price data: {str(e)}",
                context=context,
                source_name="price",
            )

    def get_forecast_data(self, ticker: str) -> float:
        """Get analyst forecast confidence for a ticker."""
        try:
            return self.forecast_repo.fetch_forecast_data(ticker)
        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="data_fetch")
            raise DataSourceError(
                f"Failed to fetch forecast data for {ticker}: {str(e)}",
                context=context,
                source_name="forecast",
            )

    def get_market_data(self, ticker: str, min_days: int) -> MarketData:
        """Get all market data needed for analysis."""
        try:
            # Fetch all required data
            options_result = self.get_options_data(ticker, min_days)
            stock_prices = self.get_price_history(ticker)
            spy_prices = self.get_spy_prices()

            # Get current price
            current_price = self.price_repo.get_current_price(ticker)

            return MarketData(
                ticker=ticker,
                stock_prices=stock_prices,
                spy_prices=spy_prices,
                options_data=options_result.options_df,
                current_price=current_price,
            )
        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="market_data_fetch")
            raise DataSourceError(
                f"Failed to fetch market data for {ticker}: {str(e)}",
                context=context,
                source_name="market_data",
            )
