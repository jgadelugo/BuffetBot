"""
Main strategy dispatcher for options analysis.

This module provides the primary entry point for options strategy analysis,
orchestrating data fetching, technical analysis, scoring, and strategy execution.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from buffetbot.utils.logger import setup_logger

from ..config.scoring_weights import get_scoring_weights
from ..core.domain_models import (
    AnalysisRequest,
    AnalysisResult,
    RiskTolerance,
    StrategyType,
    TimeHorizon,
)
from ..core.exceptions import ErrorContext, OptionsAdvisorError, StrategyValidationError
from ..data.forecast_service import DefaultForecastService
from ..data.options_service import DefaultOptionsService
from ..data.price_service import YFinancePriceService
from ..data.repositories import DefaultDataRepository
from ..scoring.technical_indicators import TechnicalIndicatorsCalculator
from .strategy_registry import get_strategy_registry

logger = setup_logger(__name__, "logs/strategy_dispatcher.log")


def analyze_options_strategy(
    strategy_type: str,
    ticker: str,
    min_days: int = 180,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
    time_horizon: str = "Medium-term (3-6 months)",
) -> pd.DataFrame:
    """
    Main entry point for options strategy analysis.

    This function maintains backward compatibility with the original API
    while using the new modular architecture internally.

    Args:
        strategy_type: Options strategy to analyze
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry
        top_n: Number of recommendations
        risk_tolerance: Risk tolerance level
        time_horizon: Investment time horizon

    Returns:
        pd.DataFrame: Strategy recommendations

    Raises:
        OptionsAdvisorError: If analysis fails
    """
    start_time = datetime.now()
    logger.info(f"Starting options analysis: {strategy_type} for {ticker}")

    try:
        # Create analysis request
        request = AnalysisRequest(
            ticker=ticker,
            strategy_type=StrategyType(strategy_type),
            min_days=min_days,
            top_n=top_n,
            risk_tolerance=RiskTolerance(risk_tolerance),
            time_horizon=TimeHorizon(time_horizon),
        )

        # Execute analysis using new architecture
        result = execute_strategy_analysis(request)

        # Add legacy metadata for backward compatibility
        recommendations = result.recommendations.copy()
        if not recommendations.empty:
            recommendations["strategy_type"] = strategy_type
            recommendations["risk_tolerance"] = risk_tolerance
            recommendations["time_horizon"] = time_horizon
            recommendations["analysis_date"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Analysis completed for {strategy_type} on {ticker} in {execution_time:.2f}s"
        )

        return recommendations

    except Exception as e:
        logger.error(f"Strategy analysis failed: {str(e)}", exc_info=True)
        if isinstance(e, OptionsAdvisorError):
            raise
        raise OptionsAdvisorError(
            f"Strategy analysis failed for {strategy_type} on {ticker}: {str(e)}"
        )


def execute_strategy_analysis(request: AnalysisRequest) -> AnalysisResult:
    """
    Execute strategy analysis using the new modular architecture.

    Args:
        request: Analysis request object

    Returns:
        AnalysisResult: Complete analysis result

    Raises:
        OptionsAdvisorError: If analysis fails
    """
    start_time = datetime.now()
    logger.info(
        f"Executing strategy analysis: {request.strategy_type.value} for {request.ticker}"
    )

    try:
        # Initialize data repositories
        price_service = YFinancePriceService()
        options_service = DefaultOptionsService()
        forecast_service = DefaultForecastService()

        data_repo = DefaultDataRepository(
            options_repo=options_service,
            price_repo=price_service,
            forecast_repo=forecast_service,
        )

        # Fetch market data
        logger.debug("Fetching market data...")
        market_data = data_repo.get_market_data(request.ticker, request.min_days)

        # Get forecast data
        logger.debug("Fetching forecast data...")
        forecast_confidence = data_repo.get_forecast_data(request.ticker)

        # Calculate technical indicators
        logger.debug("Calculating technical indicators...")
        indicators_calculator = TechnicalIndicatorsCalculator()
        technical_indicators = indicators_calculator.calculate_all_indicators(
            market_data, forecast_confidence
        )

        # Get strategy-specific scoring weights
        scoring_weights = get_scoring_weights(request.strategy_type)

        # For now, use the legacy strategy implementations
        # TODO: Replace with new strategy pattern implementations
        recommendations = _execute_legacy_strategy(
            request, market_data, technical_indicators, scoring_weights
        )

        # Create analysis result
        execution_time = (datetime.now() - start_time).total_seconds()
        result = AnalysisResult(
            request=request,
            recommendations=recommendations,
            technical_indicators=technical_indicators,
            execution_time_seconds=execution_time,
            metadata={
                "data_sources_used": list(
                    technical_indicators.data_availability.keys()
                ),
                "scoring_weights": scoring_weights.to_dict(),
                "market_data_timestamp": market_data.data_timestamp.isoformat(),
            },
        )

        logger.info(
            f"Strategy analysis completed successfully in {execution_time:.2f}s"
        )
        return result

    except Exception as e:
        context = ErrorContext(
            ticker=request.ticker,
            strategy=request.strategy_type.value,
            correlation_id=request.correlation_id,
        )

        if isinstance(e, OptionsAdvisorError):
            e.context = context
            raise

        raise OptionsAdvisorError(
            f"Strategy analysis failed: {str(e)}", context=context
        )


def _execute_legacy_strategy(
    request: AnalysisRequest, market_data, technical_indicators, scoring_weights
) -> pd.DataFrame:
    """
    Execute strategy using legacy implementations.

    This is a temporary bridge function that will be replaced
    with the new strategy pattern implementations.
    """
    # Import legacy functions
    from buffetbot.analysis.options_advisor import (
        recommend_bull_call_spread,
        recommend_cash_secured_put,
        recommend_covered_call,
        recommend_long_calls,
        update_scoring_weights,
    )

    # Map strategy types to legacy functions
    strategy_map = {
        StrategyType.LONG_CALLS: recommend_long_calls,
        StrategyType.BULL_CALL_SPREAD: recommend_bull_call_spread,
        StrategyType.COVERED_CALL: recommend_covered_call,
        StrategyType.CASH_SECURED_PUT: recommend_cash_secured_put,
    }

    if request.strategy_type not in strategy_map:
        raise StrategyValidationError(
            f"Strategy {request.strategy_type.value} not implemented",
            validation_errors={"strategy_type": "Not implemented"},
        )

    # Update legacy system with strategy-specific weights
    logger.info(
        f"Updating legacy system with strategy-specific weights: {scoring_weights.to_dict()}"
    )
    update_scoring_weights(scoring_weights.to_dict())

    # Execute legacy strategy
    strategy_func = strategy_map[request.strategy_type]

    # Adjust min_days for income strategies
    min_days = request.min_days
    if request.strategy_type in [
        StrategyType.COVERED_CALL,
        StrategyType.CASH_SECURED_PUT,
    ]:
        if min_days > 90:
            min_days = min(90, min_days)
            logger.info(
                f"Adjusted min_days to {min_days} for {request.strategy_type.value}"
            )

    # Risk tolerance adjustments - initialize top_n with default value
    top_n = request.top_n

    if request.risk_tolerance == RiskTolerance.CONSERVATIVE:
        min_days = max(min_days, 60)
    elif request.risk_tolerance == RiskTolerance.AGGRESSIVE:
        top_n = min(request.top_n * 2, 10)
    # MODERATE uses the default top_n

    # Execute strategy
    recommendations = strategy_func(
        ticker=request.ticker,
        min_days=min_days,
        top_n=top_n,
        risk_tolerance=request.risk_tolerance.value,
    )

    return recommendations


def validate_analysis_request(request: AnalysisRequest) -> None:
    """
    Validate analysis request parameters.

    Args:
        request: Analysis request to validate

    Raises:
        StrategyValidationError: If validation fails
    """
    errors = {}

    # Validate ticker
    if not request.ticker or len(request.ticker.strip()) == 0:
        errors["ticker"] = "Ticker cannot be empty"

    # Validate min_days
    if request.min_days <= 0:
        errors["min_days"] = "min_days must be positive"
    elif request.min_days > 365:
        errors["min_days"] = "min_days cannot exceed 365"

    # Validate top_n
    if request.top_n <= 0:
        errors["top_n"] = "top_n must be positive"
    elif request.top_n > 50:
        errors["top_n"] = "top_n cannot exceed 50"

    if errors:
        context = ErrorContext(
            ticker=request.ticker, strategy=request.strategy_type.value
        )
        raise StrategyValidationError(
            "Request validation failed", context=context, validation_errors=errors
        )
