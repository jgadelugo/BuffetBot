"""
Technical indicators calculation for options analysis.

This module provides a service for calculating technical indicators
using the existing utilities while maintaining clean separation of concerns.
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from buffetbot.utils.logger import setup_logger
from buffetbot.utils.options_math import (
    OptionsMathError,
    calculate_average_iv,
    calculate_beta,
    calculate_momentum,
    calculate_rsi,
)

from ..core.domain_models import MarketData, TechnicalIndicators
from ..core.exceptions import CalculationError, ErrorContext

logger = setup_logger(__name__, "logs/technical_indicators.log")


class TechnicalIndicatorsCalculator:
    """Calculator for technical indicators used in options analysis."""

    def __init__(self):
        pass

    def calculate_all_indicators(
        self, market_data: MarketData, forecast_confidence: float = 0.5
    ) -> TechnicalIndicators:
        """
        Calculate all technical indicators for market data.

        Args:
            market_data: Market data for calculation
            forecast_confidence: Analyst forecast confidence (0.0 to 1.0)

        Returns:
            TechnicalIndicators: Calculated indicators with availability tracking

        Raises:
            CalculationError: If critical calculations fail
        """
        logger.info(f"Calculating technical indicators for {market_data.ticker}")

        data_availability = {}

        try:
            # Calculate RSI
            rsi, rsi_available = self._calculate_rsi_safe(market_data)
            data_availability["rsi"] = rsi_available

            # Calculate Beta
            beta, beta_available = self._calculate_beta_safe(market_data)
            data_availability["beta"] = beta_available

            # Calculate Momentum
            momentum, momentum_available = self._calculate_momentum_safe(market_data)
            data_availability["momentum"] = momentum_available

            # Calculate Average IV
            avg_iv, iv_available = self._calculate_iv_safe(market_data)
            data_availability["iv"] = iv_available

            # Forecast confidence is provided
            data_availability["forecast"] = forecast_confidence is not None
            if forecast_confidence is None:
                forecast_confidence = 0.5  # Default neutral

            indicators = TechnicalIndicators(
                rsi=rsi,
                beta=beta,
                momentum=momentum,
                avg_iv=avg_iv,
                forecast_confidence=forecast_confidence,
                data_availability=data_availability,
            )

            # Log availability summary
            available_count = sum(data_availability.values())
            total_count = len(data_availability)
            logger.info(
                f"Technical indicators calculated for {market_data.ticker}: "
                f"{available_count}/{total_count} indicators available"
            )

            return indicators

        except Exception as e:
            context = ErrorContext(
                ticker=market_data.ticker,
                strategy="technical_indicators",
                additional_data=data_availability,
            )
            raise CalculationError(
                f"Failed to calculate technical indicators for {market_data.ticker}: {str(e)}",
                context=context,
                calculation_type="technical_indicators",
            )

    def _calculate_rsi_safe(self, market_data: MarketData) -> tuple[float, bool]:
        """Safely calculate RSI with error handling."""
        try:
            rsi = calculate_rsi(market_data.stock_prices)
            logger.debug(f"RSI calculated for {market_data.ticker}: {rsi:.2f}")
            return rsi, True
        except OptionsMathError as e:
            logger.warning(f"RSI calculation failed for {market_data.ticker}: {str(e)}")
            return 50.0, False  # Default neutral RSI
        except Exception as e:
            logger.warning(
                f"Unexpected error in RSI calculation for {market_data.ticker}: {str(e)}"
            )
            return 50.0, False

    def _calculate_beta_safe(self, market_data: MarketData) -> tuple[float, bool]:
        """Safely calculate Beta with error handling."""
        try:
            beta = calculate_beta(market_data.stock_prices, market_data.spy_prices)
            logger.debug(f"Beta calculated for {market_data.ticker}: {beta:.3f}")
            return beta, True
        except OptionsMathError as e:
            logger.warning(
                f"Beta calculation failed for {market_data.ticker}: {str(e)}"
            )
            return 1.0, False  # Default market beta
        except Exception as e:
            logger.warning(
                f"Unexpected error in Beta calculation for {market_data.ticker}: {str(e)}"
            )
            return 1.0, False

    def _calculate_momentum_safe(self, market_data: MarketData) -> tuple[float, bool]:
        """Safely calculate Momentum with error handling."""
        try:
            momentum = calculate_momentum(market_data.stock_prices)
            logger.debug(
                f"Momentum calculated for {market_data.ticker}: {momentum:.4f}"
            )
            return momentum, True
        except OptionsMathError as e:
            logger.warning(
                f"Momentum calculation failed for {market_data.ticker}: {str(e)}"
            )
            return 0.0, False  # Default neutral momentum
        except Exception as e:
            logger.warning(
                f"Unexpected error in Momentum calculation for {market_data.ticker}: {str(e)}"
            )
            return 0.0, False

    def _calculate_iv_safe(self, market_data: MarketData) -> tuple[float, bool]:
        """Safely calculate Average IV with error handling."""
        try:
            avg_iv = calculate_average_iv(market_data.options_data)
            logger.debug(
                f"Average IV calculated for {market_data.ticker}: {avg_iv:.4f}"
            )
            return avg_iv, True
        except OptionsMathError as e:
            logger.warning(f"IV calculation failed for {market_data.ticker}: {str(e)}")
            return 0.25, False  # Default reasonable IV
        except Exception as e:
            logger.warning(
                f"Unexpected error in IV calculation for {market_data.ticker}: {str(e)}"
            )
            return 0.25, False


def compute_scores(
    ticker: str,
    stock_prices: pd.Series,
    spy_prices: pd.Series,
    options_df: pd.DataFrame,
) -> tuple[float, float, float, float, float, dict[str, bool]]:
    """
    Legacy function wrapper for backward compatibility.

    This function maintains compatibility with the original options_advisor
    while using the new modular architecture internally.
    """
    # Create market data object
    market_data = MarketData(
        ticker=ticker,
        stock_prices=stock_prices,
        spy_prices=spy_prices,
        options_data=options_df,
        current_price=float(stock_prices.iloc[-1]),
    )

    # Calculate indicators
    calculator = TechnicalIndicatorsCalculator()

    # Default forecast confidence for legacy compatibility
    forecast_confidence = 0.5

    indicators = calculator.calculate_all_indicators(market_data, forecast_confidence)

    return (
        indicators.rsi,
        indicators.beta,
        indicators.momentum,
        indicators.avg_iv,
        indicators.forecast_confidence,
        indicators.data_availability,
    )
