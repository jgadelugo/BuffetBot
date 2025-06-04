"""
Forecast data service implementation.

This module provides concrete implementation for analyst forecast data access
using the existing forecast fetcher.
"""

import logging
from typing import Optional

from buffetbot.data.forecast_fetcher import ForecastFetchError, get_analyst_forecast
from buffetbot.utils.logger import setup_logger

from ..core.exceptions import DataSourceError, ErrorContext
from .repositories import ForecastRepository

logger = setup_logger(__name__, "logs/forecast_service.log")


class DefaultForecastService(ForecastRepository):
    """Forecast service implementation using existing forecast fetcher."""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache = {}

    def fetch_forecast_data(self, ticker: str) -> float:
        """
        Fetch analyst forecast confidence using existing forecast fetcher.

        Args:
            ticker: Stock ticker symbol

        Returns:
            float: Forecast confidence score (0.0 to 1.0)

        Raises:
            DataSourceError: If forecast data cannot be fetched
        """
        logger.info(f"Fetching forecast data for {ticker}")

        # Check cache first
        if self.cache_enabled and ticker in self._cache:
            logger.debug(f"Returning cached forecast data for {ticker}")
            return self._cache[ticker]

        try:
            # Get analyst forecast data
            forecast_data = get_analyst_forecast(ticker)

            # Extract confidence score from forecast data
            # The original function returns various forecast metrics
            # We need to convert this to a confidence score
            confidence = self._calculate_confidence_score(forecast_data)

            # Cache the result
            if self.cache_enabled:
                self._cache[ticker] = confidence

            logger.info(
                f"Successfully fetched forecast confidence for {ticker}: {confidence:.3f}"
            )
            return confidence

        except ForecastFetchError as e:
            # Handle gracefully - forecast data is not critical
            logger.warning(f"Forecast data not available for {ticker}: {str(e)}")
            default_confidence = 0.5  # Neutral confidence when no data available

            if self.cache_enabled:
                self._cache[ticker] = default_confidence

            return default_confidence

        except Exception as e:
            context = ErrorContext(ticker=ticker, strategy="forecast_fetch")
            raise DataSourceError(
                f"Failed to fetch forecast data for {ticker}: {str(e)}",
                context=context,
                source_name="forecast_fetcher",
            )

    def _calculate_confidence_score(self, forecast_data) -> float:
        """
        Calculate confidence score from forecast data.

        Args:
            forecast_data: Raw forecast data from fetcher

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            # Handle different forecast data formats
            if isinstance(forecast_data, dict):
                # Look for common confidence indicators
                if "confidence" in forecast_data:
                    return max(0.0, min(1.0, float(forecast_data["confidence"])))

                # Calculate from recommendation metrics
                if "recommendations" in forecast_data:
                    recs = forecast_data["recommendations"]
                    total_recs = sum(recs.values()) if isinstance(recs, dict) else 0

                    if total_recs > 0:
                        # Weight by recommendation strength
                        strong_buy = recs.get("strongBuy", 0)
                        buy = recs.get("buy", 0)
                        hold = recs.get("hold", 0)
                        sell = recs.get("sell", 0)
                        strong_sell = recs.get("strongSell", 0)

                        # Calculate weighted score
                        weighted_score = (
                            strong_buy * 1.0
                            + buy * 0.75
                            + hold * 0.5
                            + sell * 0.25
                            + strong_sell * 0.0
                        ) / total_recs

                        return weighted_score

                # Calculate from target price vs current price
                if "targetPrice" in forecast_data and "currentPrice" in forecast_data:
                    target = forecast_data["targetPrice"]
                    current = forecast_data["currentPrice"]

                    if target and current and current > 0:
                        upside = (target - current) / current
                        # Convert upside to confidence (0-30% upside -> 0.5-1.0 confidence)
                        confidence = 0.5 + max(0, min(0.5, upside / 0.3 * 0.5))
                        return confidence

            elif isinstance(forecast_data, (int, float)):
                # Direct confidence score
                return max(0.0, min(1.0, float(forecast_data)))

            # Default neutral confidence
            return 0.5

        except Exception as e:
            logger.warning(f"Error calculating confidence score: {str(e)}")
            return 0.5  # Neutral confidence on error

    def clear_cache(self) -> None:
        """Clear the forecast data cache."""
        self._cache.clear()
        logger.debug("Forecast data cache cleared")
