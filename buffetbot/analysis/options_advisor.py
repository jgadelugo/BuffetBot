"""
Options Advisor Module

This module analyzes different options strategies for a given stock and recommends
the best ones using a comprehensive technical scoring framework. It combines
multiple technical indicators (RSI, Beta, Momentum, Implied Volatility) with
forward-looking analyst forecast data to provide a composite score for each
option contract.

The module supports multiple strategies:
- Long Calls: Bullish strategy with unlimited upside potential
- Bull Call Spread: Limited risk/reward bullish strategy
- Covered Call: Income generation strategy for existing positions
- Cash-Secured Put: Income generation with potential stock acquisition

The module is designed to be extensible and integrates seamlessly with
existing data fetching and mathematical utilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from buffetbot.analysis.ecosystem import EcosystemAnalyzer
from buffetbot.data.forecast_fetcher import ForecastFetchError, get_analyst_forecast
from buffetbot.data.options_fetcher import (
    OptionsDataError,
    OptionsResult,
    fetch_long_dated_calls,
)
from buffetbot.data.peer_fetcher import PeerFetchError, get_peers
from buffetbot.utils.errors import (
    DataError,
    DataFetcherError,
    ErrorSeverity,
    handle_data_error,
)
from buffetbot.utils.logger import setup_logger
from buffetbot.utils.options_math import (
    OptionsMathError,
    calculate_average_iv,
    calculate_beta,
    calculate_momentum,
    calculate_rsi,
)
from buffetbot.utils.validators import validate_ticker

# Initialize logger
logger = setup_logger(__name__, "logs/options_advisor.log")

# Global scoring weights for technical indicators including forecast
SCORING_WEIGHTS: dict[str, float] = {
    "rsi": 0.20,  # RSI contribution to composite score
    "beta": 0.20,  # Beta contribution to composite score
    "momentum": 0.20,  # Momentum contribution to composite score
    "iv": 0.20,  # Implied Volatility contribution to composite score
    "forecast": 0.20,  # Analyst forecast contribution to composite score
}

# Ecosystem scoring weights and multipliers
ECOSYSTEM_SCORING_WEIGHTS: dict[str, float] = {
    "confirm": 1.1,  # Boost score by 10% for ecosystem confirmation
    "neutral": 1.0,  # No adjustment for neutral ecosystem signal
    "veto": 0.9,  # Reduce score by 10% for ecosystem veto
}


class OptionsAdvisorError(Exception):
    """Custom exception for options advisor module errors."""

    pass


class InsufficientDataError(OptionsAdvisorError):
    """Raised when there's insufficient data for analysis."""

    pass


class CalculationError(OptionsAdvisorError):
    """Raised when technical indicator calculations fail."""

    pass


def _validate_inputs(ticker: str, min_days: int, top_n: int) -> None:
    """
    Validate input parameters for the recommendation function.

    Args:
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry
        top_n: Number of top recommendations to return

    Raises:
        OptionsAdvisorError: If any input validation fails
    """
    if not ticker or not isinstance(ticker, str):
        raise OptionsAdvisorError("Ticker must be a non-empty string")

    if min_days <= 0:
        raise OptionsAdvisorError("min_days must be positive")

    if top_n <= 0:
        raise OptionsAdvisorError("top_n must be positive")

    try:
        validate_ticker(ticker)
    except Exception as e:
        raise OptionsAdvisorError(f"Invalid ticker format '{ticker}': {str(e)}")


def fetch_price_history(ticker: str, period: str = "1y") -> pd.Series:
    """
    Fetch historical price data for a given ticker.

    Args:
        ticker: Stock ticker symbol
        period: Time period for historical data (default: "1y")

    Returns:
        pd.Series: Historical closing prices

    Raises:
        OptionsAdvisorError: If price data cannot be fetched or is insufficient
    """
    logger.info(f"Fetching {period} price history for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            error = DataError(
                code="PRICE_DATA_EMPTY",
                message=f"No price data available for {ticker}",
                severity=ErrorSeverity.HIGH,
            )
            handle_data_error(error, logger)
            raise OptionsAdvisorError(f"No price data available for {ticker}")

        if len(hist) < 30:  # Minimum data points for meaningful analysis
            error = DataError(
                code="INSUFFICIENT_PRICE_DATA",
                message=f"Insufficient price data for {ticker}: {len(hist)} days",
                severity=ErrorSeverity.MEDIUM,
                details={"days_available": len(hist), "minimum_required": 30},
            )
            handle_data_error(error, logger)
            raise InsufficientDataError(
                f"Insufficient price data for {ticker}: {len(hist)} days (minimum 30 required)"
            )

        logger.info(f"Successfully fetched {len(hist)} days of price data for {ticker}")
        return hist["Close"]

    except Exception as e:
        if isinstance(e, (OptionsAdvisorError, InsufficientDataError)):
            raise

        error = DataError(
            code="PRICE_FETCH_ERROR",
            message=f"Failed to fetch price data for {ticker}: {str(e)}",
            severity=ErrorSeverity.HIGH,
        )
        handle_data_error(error, logger)
        raise OptionsAdvisorError(f"Failed to fetch price data for {ticker}: {str(e)}")


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute percentage returns from price series.

    Args:
        prices: Series of price data

    Returns:
        pd.Series: Percentage returns

    Raises:
        CalculationError: If return calculation fails
    """
    try:
        if len(prices) < 2:
            raise CalculationError("Need at least 2 data points to calculate returns")

        returns = prices.pct_change().dropna()

        if returns.empty:
            raise CalculationError("No valid returns could be calculated")

        logger.debug(f"Computed {len(returns)} return values from {len(prices)} prices")
        return returns

    except Exception as e:
        if isinstance(e, CalculationError):
            raise
        raise CalculationError(f"Failed to compute returns: {str(e)}")


def normalize_scoring_weights(
    input_weights: dict[str, float], available_sources: list[str]
) -> dict[str, float]:
    """
    Normalize scoring weights dynamically based on available data sources.

    When one or more data sources are unavailable, this function redistributes
    the weights proportionally among the remaining indicators, ensuring the
    total weight always sums to 1.0.

    Args:
        input_weights: Original scoring weights for all indicators
        available_sources: List of data source keys that have valid data

    Returns:
        dict: Normalized weights that sum to 1.0, containing only available sources

    Examples:
        >>> original = {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2}
        >>> available = ["rsi", "beta", "momentum", "iv"]  # forecast missing
        >>> normalized = normalize_scoring_weights(original, available)
        >>> # Result: {"rsi": 0.25, "beta": 0.25, "momentum": 0.25, "iv": 0.25}
    """
    if not available_sources:
        logger.error("No available data sources for weight normalization")
        return {}

    if len(available_sources) == 1:
        # Only one source available, give it 100% weight
        return {available_sources[0]: 1.0}

    # Calculate total weight of available sources
    available_weight = sum(
        input_weights[source] for source in available_sources if source in input_weights
    )

    if available_weight == 0:
        # If no original weights exist for available sources, distribute equally
        equal_weight = 1.0 / len(available_sources)
        return {source: equal_weight for source in available_sources}

    # Redistribute weights proportionally
    normalized_weights = {}
    for source in available_sources:
        original_weight = input_weights.get(source, 0)
        normalized_weights[source] = original_weight / available_weight

    # Verify weights sum to 1.0 (within floating point tolerance)
    total_weight = sum(normalized_weights.values())
    if abs(total_weight - 1.0) > 0.001:
        logger.warning(
            f"Normalized weights sum to {total_weight:.6f}, not 1.0. Adjusting..."
        )
        # Minor adjustment to ensure exact sum of 1.0
        adjustment_factor = 1.0 / total_weight
        normalized_weights = {
            k: v * adjustment_factor for k, v in normalized_weights.items()
        }

    logger.info(
        f"Normalized weights for {len(available_sources)} sources: {normalized_weights}"
    )
    return normalized_weights


def compute_scores(
    ticker: str,
    stock_prices: pd.Series,
    spy_prices: pd.Series,
    options_df: pd.DataFrame,
) -> tuple[float, float, float, float, float, dict[str, bool]]:
    """
    Compute technical indicator scores and analyst forecast for options analysis.

    Args:
        ticker: Stock ticker symbol
        stock_prices: Historical stock prices
        spy_prices: Historical SPY (benchmark) prices
        options_df: Options data DataFrame

    Returns:
        Tuple containing:
            - RSI value (float)
            - Beta value (float)
            - Momentum value (float)
            - Average IV value (float)
            - Forecast confidence (float)
            - Data availability status (dict[str, bool])

    Raises:
        CalculationError: If any technical indicator calculation fails
    """
    logger.info(f"Computing technical scores and forecast for {ticker}")

    # Track which data sources are available
    data_availability = {
        "rsi": False,
        "beta": False,
        "momentum": False,
        "iv": False,
        "forecast": False,
    }

    try:
        # Calculate RSI
        try:
            rsi = calculate_rsi(stock_prices, period=14)
            data_availability["rsi"] = True
            logger.debug(f"RSI calculated: {rsi:.2f}")
        except OptionsMathError as e:
            logger.warning(f"RSI calculation failed for {ticker}: {str(e)}")
            rsi = 50.0  # Neutral RSI as fallback

        # Calculate Beta
        try:
            stock_returns = compute_returns(stock_prices)
            spy_returns = compute_returns(spy_prices)

            # Align the series by index (in case of different date ranges)
            aligned_stock, aligned_spy = stock_returns.align(spy_returns, join="inner")

            if len(aligned_stock) < 20:  # Need sufficient data for meaningful beta
                logger.warning(
                    f"Insufficient aligned data for beta calculation: {len(aligned_stock)} points"
                )
                beta = 1.0  # Market beta as fallback
            else:
                beta = calculate_beta(aligned_stock, aligned_spy)
                data_availability["beta"] = True
                logger.debug(f"Beta calculated: {beta:.3f}")
        except (OptionsMathError, CalculationError) as e:
            logger.warning(f"Beta calculation failed for {ticker}: {str(e)}")
            beta = 1.0  # Market beta as fallback

        # Calculate Momentum
        try:
            momentum = calculate_momentum(stock_prices, window=20)
            data_availability["momentum"] = True
            logger.debug(f"Momentum calculated: {momentum:.4f}")
        except OptionsMathError as e:
            logger.warning(f"Momentum calculation failed for {ticker}: {str(e)}")
            momentum = 0.0  # Neutral momentum as fallback

        # Calculate Average Implied Volatility
        try:
            avg_iv = calculate_average_iv(options_df)
            data_availability["iv"] = True
            logger.debug(f"Average IV calculated: {avg_iv:.4f}")
        except OptionsMathError as e:
            logger.warning(f"IV calculation failed for {ticker}: {str(e)}")
            avg_iv = 0.2  # Default 20% IV as fallback

        # Get Analyst Forecast
        try:
            forecast_data = get_analyst_forecast(ticker)
            forecast_confidence = forecast_data["confidence"]
            data_availability["forecast"] = True
            logger.info(
                f"Analyst forecast fetched for {ticker}: "
                f"mean_target=${forecast_data['mean_target']:.2f}, "
                f"confidence={forecast_confidence:.3f}"
            )
        except ForecastFetchError as e:
            logger.warning(f"Forecast fetch failed for {ticker}: {str(e)}")
            forecast_confidence = 0.5  # Neutral confidence as fallback

        available_sources = [k for k, v in data_availability.items() if v]
        logger.info(
            f"Data availability for {ticker}: {sum(data_availability.values())}/{len(data_availability)} sources available: {available_sources}"
        )
        logger.info(
            f"All scores computed for {ticker}: RSI={rsi:.2f}, Beta={beta:.3f}, "
            f"Momentum={momentum:.4f}, IV={avg_iv:.4f}, Forecast={forecast_confidence:.3f}"
        )
        return rsi, beta, momentum, avg_iv, forecast_confidence, data_availability

    except Exception as e:
        error_msg = f"Failed to compute scores for {ticker}: {str(e)}"
        logger.error(error_msg)
        raise CalculationError(error_msg)


def _normalize_score(
    value: float, min_val: float, max_val: float, invert: bool = False
) -> float:
    """
    Normalize a score to 0-1 range.

    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
        invert: Whether to invert the score (higher raw value = lower score)

    Returns:
        float: Normalized score between 0 and 1
    """
    if max_val == min_val:
        return 0.5  # Neutral score if no variation

    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

    return 1.0 - normalized if invert else normalized


def _calculate_composite_scores(
    options_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
) -> pd.DataFrame:
    """
    Calculate composite scores for each option contract with dynamic weight normalization.

    Args:
        options_df: DataFrame containing options data
        rsi: RSI value for the underlying stock
        beta: Beta value for the underlying stock
        momentum: Momentum value for the underlying stock
        avg_iv: Average implied volatility
        forecast_confidence: Analyst forecast confidence score (0-1)
        data_availability: Dict indicating which data sources are available

    Returns:
        pd.DataFrame: Options DataFrame with composite scores and metadata added
    """
    logger.info(f"Calculating composite scores for {len(options_df)} option contracts")

    # Create a copy to avoid modifying the original DataFrame
    scored_df = options_df.copy()

    # Determine available data sources
    available_sources = [k for k, v in data_availability.items() if v]

    # Log warning if fewer than 4 sources are available
    if len(available_sources) < 4:
        logger.warning(
            f"Scoring with partial data: only {len(available_sources)}/5 sources available: {available_sources}. "
            f"Missing: {[k for k, v in data_availability.items() if not v]}"
        )

    # Normalize weights based on available sources
    normalized_weights = normalize_scoring_weights(SCORING_WEIGHTS, available_sources)

    # Calculate individual normalized scores
    score_components = {}

    if data_availability["rsi"]:
        # RSI: Lower RSI (oversold) is better for calls, so we invert
        rsi_score = _normalize_score(rsi, 0, 100, invert=True)
        score_components["rsi"] = rsi_score

    if data_availability["beta"]:
        # Beta: Moderate beta (around 1.0) is often preferred, but higher beta can mean more upside
        # We'll favor slightly higher beta but penalize extreme values
        if beta < 0.5:
            beta_score = 0.2  # Very low beta gets low score
        elif beta > 2.0:
            beta_score = 0.3  # Very high beta gets penalized
        else:
            beta_score = _normalize_score(beta, 0.5, 1.5, invert=False)
        score_components["beta"] = beta_score

    if data_availability["momentum"]:
        # Momentum: Higher momentum is better for calls
        momentum_score = _normalize_score(momentum, -0.1, 0.1, invert=False)
        score_components["momentum"] = momentum_score

    if data_availability["forecast"]:
        # Forecast: Higher confidence is better for calls (already normalized 0-1)
        forecast_score = forecast_confidence
        score_components["forecast"] = forecast_score

    # Handle IV scoring (per-option basis)
    if data_availability["iv"]:
        # For individual options, we'll use their specific IV vs the average
        # Lower IV relative to average is better (cheaper options)
        scored_df["iv_score"] = scored_df["impliedVolatility"].apply(
            lambda iv: _normalize_score(iv, avg_iv * 0.5, avg_iv * 1.5, invert=True)
        )
    else:
        # If IV data is not available, assign neutral score
        scored_df["iv_score"] = 0.5

    # Calculate composite score for each option using normalized weights
    scored_df["CompositeScore"] = 0.0

    for source, weight in normalized_weights.items():
        if source == "iv":
            # IV is calculated per option
            scored_df["CompositeScore"] += weight * scored_df["iv_score"]
        elif source in score_components:
            # Other indicators are the same for all options of this ticker
            scored_df["CompositeScore"] += weight * score_components[source]

    # Add score details metadata to each row
    score_details = {source: weight for source, weight in normalized_weights.items()}
    scored_df["score_details"] = [score_details] * len(scored_df)

    # Add technical indicator values
    scored_df["RSI"] = rsi
    scored_df["Beta"] = beta
    scored_df["Momentum"] = momentum
    scored_df["ForecastConfidence"] = forecast_confidence

    # Drop the temporary iv_score column
    scored_df = scored_df.drop("iv_score", axis=1)

    logger.info(
        f"Composite scores calculated using {len(available_sources)}/5 data sources. "
        f"Score range: {scored_df['CompositeScore'].min():.3f} - {scored_df['CompositeScore'].max():.3f}"
    )

    return scored_df


def recommend_long_calls(
    ticker: str,
    min_days: int = 180,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
) -> pd.DataFrame:
    """
    Analyze long-dated call options and recommend the best ones using comprehensive scoring.

    This function fetches long-dated call options for the specified ticker and applies
    a comprehensive analysis framework to score each option. The scoring combines
    technical indicators (RSI, Beta, Momentum, Implied Volatility), analyst forecast
    confidence, and ecosystem analysis to identify the most attractive option contracts.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        min_days: Minimum days to expiry for options to consider (default: 180)
        top_n: Number of top recommendations to return (default: 5)
        risk_tolerance: Risk tolerance level ('Conservative', 'Moderate', 'Aggressive')

    Returns:
        pd.DataFrame: Top-ranked option recommendations with risk-adjusted filtering

    Raises:
        OptionsAdvisorError: If input validation fails or analysis cannot be completed
        InsufficientDataError: If there's insufficient data for meaningful analysis
        CalculationError: If technical indicator calculations fail
    """
    start_time = datetime.now()
    logger.info(
        f"Starting options analysis for {ticker} (min_days={min_days}, top_n={top_n}, risk_tolerance={risk_tolerance})"
    )

    # Input validation
    _validate_inputs(ticker, min_days, top_n)
    ticker = ticker.upper().strip()

    try:
        # Step 1: Fetch options data
        logger.info("Step 1: Fetching long-dated call options")
        options_result = fetch_long_dated_calls(ticker, min_days_to_expiry=min_days)

        # Check if data was fetched successfully
        if not options_result["data_available"]:
            error_msg = options_result.get(
                "error_message", "Unknown error fetching options data"
            )
            error = DataError(
                code="NO_OPTIONS_DATA",
                message=f"No long-dated call options found for {ticker} with min {min_days} days: {error_msg}",
                severity=ErrorSeverity.HIGH,
                details={"ticker": ticker, "min_days": min_days},
            )
            handle_data_error(error, logger)
            raise InsufficientDataError(
                f"No long-dated call options found for {ticker} with minimum {min_days} days to expiry: {error_msg}"
            )

        # Extract the DataFrame from the result
        options_df = options_result["data"]

        if options_df.empty:
            error = DataError(
                code="EMPTY_OPTIONS_DATA",
                message=f"Empty options DataFrame for {ticker} with min {min_days} days",
                severity=ErrorSeverity.HIGH,
                details={"ticker": ticker, "min_days": min_days},
            )
            handle_data_error(error, logger)
            raise InsufficientDataError(
                f"No long-dated call options found for {ticker} with minimum {min_days} days to expiry"
            )

        logger.info(f"Found {len(options_df)} long-dated call options for {ticker}")

        # Step 2: Apply risk tolerance filtering
        logger.info(f"Step 2: Applying {risk_tolerance} risk tolerance filtering")
        options_df = _apply_long_calls_risk_filtering(
            options_df, ticker, risk_tolerance
        )

        if options_df.empty:
            logger.warning(
                f"No options remain after {risk_tolerance} risk filtering for {ticker}"
            )
            # Return empty dataframe with proper structure
            return pd.DataFrame()

        logger.info(f"Retained {len(options_df)} options after risk filtering")

        # Step 3: Fetch historical price data
        logger.info("Step 3: Fetching historical price data")
        stock_prices = fetch_price_history(ticker, period="1y")
        spy_prices = fetch_price_history("SPY", period="1y")

        # Step 4: Compute technical indicators
        logger.info("Step 4: Computing technical indicators")
        (
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        ) = compute_scores(ticker, stock_prices, spy_prices, options_df)

        # Step 5: Fetch peers and perform ecosystem analysis
        logger.info("Step 5: Performing ecosystem analysis")
        ecosystem_score = 0.5  # Default neutral score
        ecosystem_signal = "neutral"
        ecosystem_confidence = 0.5

        try:
            # Get peer tickers
            peer_result = get_peers(ticker)

            # Check if peer data is available and extract peer list
            if peer_result.get("data_available", False):
                peer_tickers = peer_result["peers"]
                logger.info(
                    f"Found {len(peer_tickers)} peers for {ticker}: {peer_tickers}"
                )

                # Initialize ecosystem analyzer
                ecosystem_analyzer = EcosystemAnalyzer()

                # Perform ecosystem analysis with custom peer list
                ecosystem_analysis = ecosystem_analyzer.analyze_ecosystem(
                    ticker, custom_peers=peer_tickers
                )

                # Extract ecosystem metrics
                ecosystem_score = ecosystem_analysis.ecosystem_score.normalized_score
                ecosystem_signal = ecosystem_analysis.recommendation
                ecosystem_confidence = ecosystem_analysis.confidence

                logger.info(
                    f"Ecosystem analysis complete - Score: {ecosystem_score:.3f}, "
                    f"Signal: {ecosystem_signal}, Confidence: {ecosystem_confidence:.3f}"
                )
            else:
                error_msg = peer_result.get("error_message", "Unknown error")
                logger.warning(f"No peer data available for {ticker}: {error_msg}")
                logger.info("Proceeding with technical analysis only")

        except (PeerFetchError, DataFetcherError) as e:
            logger.warning(f"Ecosystem analysis failed for {ticker}: {str(e)}")
            logger.info("Proceeding with technical analysis only")
        except Exception as e:
            logger.error(
                f"Unexpected error in ecosystem analysis for {ticker}: {str(e)}"
            )
            logger.info("Proceeding with technical analysis only")

        # Step 6: Calculate composite scores with risk tolerance adjustments
        logger.info("Step 6: Calculating risk-adjusted composite scores")
        scored_df = _calculate_composite_scores_with_risk_tolerance(
            options_df,
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
            risk_tolerance,
        )

        # Step 7: Apply ecosystem adjustment to composite scores
        logger.info("Step 7: Applying ecosystem signal adjustments")
        ecosystem_multiplier = ECOSYSTEM_SCORING_WEIGHTS.get(ecosystem_signal, 1.0)
        scored_df["AdjustedCompositeScore"] = (
            scored_df["CompositeScore"] * ecosystem_multiplier
        )

        # Add ecosystem columns
        scored_df["ecosystem_score"] = ecosystem_score
        scored_df["signal"] = ecosystem_signal
        scored_df["confidence"] = ecosystem_confidence

        logger.info(
            f"Applied ecosystem adjustment: {ecosystem_signal} "
            f"(multiplier: {ecosystem_multiplier:.1f})"
        )

        # Step 8: Rank and select top recommendations with risk tolerance considerations
        logger.info("Step 8: Ranking and selecting top recommendations")

        # Apply risk tolerance to final selection
        final_df = _apply_final_risk_tolerance_selection(
            scored_df, risk_tolerance, top_n
        )

        # Prepare final output DataFrame with required columns
        result_df = final_df[
            [
                "strike",
                "expiry",
                "lastPrice",
                "impliedVolatility",
                "RSI",
                "Beta",
                "Momentum",
                "ForecastConfidence",
                "CompositeScore",
                "AdjustedCompositeScore",
                "ecosystem_score",
                "signal",
                "confidence",
                "score_details",  # Add score details for transparency
            ]
        ].copy()

        # Add ticker column and rename columns
        result_df.insert(0, "ticker", ticker)
        result_df = result_df.rename(
            columns={"impliedVolatility": "IV", "AdjustedCompositeScore": "FinalScore"}
        )

        # Add risk tolerance metadata
        result_df["risk_tolerance_applied"] = risk_tolerance

        # Reset index for clean output
        result_df = result_df.reset_index(drop=True)

        # Log detailed information about data sources used
        available_sources = [k for k, v in data_availability.items() if v]
        if len(available_sources) < 5:
            logger.info(
                f"Final recommendations based on {len(available_sources)}/5 data sources: {available_sources}. "
                f"Score weights were dynamically normalized."
            )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Options analysis completed for {ticker} with {risk_tolerance} risk tolerance. "
            f"Returned {len(result_df)} recommendations in {execution_time:.2f} seconds. "
            f"Ecosystem signal: {ecosystem_signal} ({'✅' if ecosystem_signal == 'confirm' else '⚠️' if ecosystem_signal == 'neutral' else '❌'})"
        )

        return result_df

    except (OptionsAdvisorError, InsufficientDataError, CalculationError):
        # Re-raise our custom exceptions without wrapping
        raise

    except OptionsDataError as e:
        # Handle options fetcher errors
        error = DataError(
            code="OPTIONS_FETCH_ERROR",
            message=f"Failed to fetch options data for {ticker}: {str(e)}",
            severity=ErrorSeverity.HIGH,
        )
        handle_data_error(error, logger)
        raise OptionsAdvisorError(
            f"Failed to fetch options data for {ticker}: {str(e)}"
        )

    except Exception as e:
        # Handle any unexpected errors
        error = DataError(
            code="UNEXPECTED_ERROR",
            message=f"Unexpected error in options analysis for {ticker}: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
            details={"ticker": ticker, "min_days": min_days, "top_n": top_n},
        )
        handle_data_error(error, logger)
        raise OptionsAdvisorError(
            f"Unexpected error in options analysis for {ticker}: {str(e)}"
        )


def get_scoring_weights() -> dict[str, float]:
    """
    Get the current scoring weights configuration.

    Returns:
        dict: Current scoring weights for each technical indicator
    """
    return SCORING_WEIGHTS.copy()


def update_scoring_weights(new_weights: dict[str, float]) -> None:
    """
    Update the scoring weights configuration.

    Args:
        new_weights: Dictionary of new weights. Must contain all required keys
                    and weights must sum to 1.0

    Raises:
        OptionsAdvisorError: If weights are invalid
    """
    global SCORING_WEIGHTS

    required_keys = {"rsi", "beta", "momentum", "iv", "forecast"}
    if set(new_weights.keys()) != required_keys:
        raise OptionsAdvisorError(
            f"Weights must contain exactly these keys: {required_keys}. "
            f"Got: {set(new_weights.keys())}"
        )

    total_weight = sum(new_weights.values())
    if abs(total_weight - 1.0) > 0.001:  # Allow small floating point tolerance
        raise OptionsAdvisorError(f"Weights must sum to 1.0, got {total_weight:.6f}")

    SCORING_WEIGHTS.update(new_weights)
    logger.info(f"Updated scoring weights: {SCORING_WEIGHTS}")


def get_ecosystem_scoring_weights() -> dict[str, float]:
    """
    Get the current ecosystem scoring weights configuration.

    Returns:
        dict: Current ecosystem scoring multipliers for each signal type
    """
    return ECOSYSTEM_SCORING_WEIGHTS.copy()


def update_ecosystem_scoring_weights(new_weights: dict[str, float]) -> None:
    """
    Update the ecosystem scoring weights configuration.

    Args:
        new_weights: Dictionary of new multipliers. Must contain keys:
                    'confirm', 'neutral', 'veto'

    Raises:
        OptionsAdvisorError: If weights are invalid
    """
    global ECOSYSTEM_SCORING_WEIGHTS

    required_keys = {"confirm", "neutral", "veto"}
    if set(new_weights.keys()) != required_keys:
        raise OptionsAdvisorError(
            f"Ecosystem weights must contain exactly these keys: {required_keys}. "
            f"Got: {set(new_weights.keys())}"
        )

    # Validate multiplier ranges (should be reasonable values)
    for signal, multiplier in new_weights.items():
        if not (0.1 <= multiplier <= 2.0):
            raise OptionsAdvisorError(
                f"Ecosystem multiplier for '{signal}' must be between 0.1 and 2.0, "
                f"got {multiplier:.3f}"
            )

    ECOSYSTEM_SCORING_WEIGHTS.update(new_weights)
    logger.info(f"Updated ecosystem scoring weights: {ECOSYSTEM_SCORING_WEIGHTS}")


def fetch_put_options(ticker: str, min_days_to_expiry: int = 180) -> OptionsResult:
    """
    Fetch long-dated put options for cash-secured put strategies.

    Args:
        ticker: Stock ticker symbol
        min_days_to_expiry: Minimum days to expiry for options filtering

    Returns:
        OptionsResult: Similar to fetch_long_dated_calls but for put options
    """
    # Initialize default response structure
    result: OptionsResult = {
        "data": pd.DataFrame(),
        "data_available": False,
        "error_message": None,
        "ticker": ticker,
        "min_days_to_expiry": min_days_to_expiry,
        "total_expiry_dates": None,
        "valid_chains_processed": None,
        "source_used": None,
    }

    try:
        # Input validation
        if not ticker or not isinstance(ticker, str):
            error_msg = "Ticker must be a non-empty string"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            result["source_used"] = "none"
            return result

        ticker = ticker.upper().strip()
        result["ticker"] = ticker

        logger.info(
            f"Fetching put options for {ticker} with min {min_days_to_expiry} days to expiry"
        )

        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)

        # Get available expiry dates
        try:
            expiry_dates = stock.options
        except Exception as e:
            error_msg = (
                f"Failed to retrieve options expiry dates for {ticker}: {str(e)}"
            )
            logger.error(error_msg)
            result["error_message"] = error_msg
            result["source_used"] = "none"
            return result

        if not expiry_dates:
            error_msg = f"No options data available for ticker {ticker}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            result["source_used"] = "none"
            return result

        result["total_expiry_dates"] = len(expiry_dates)

        # Process each expiry date for puts
        all_puts_data = []
        valid_chains_count = 0

        for expiry_date in expiry_dates:
            try:
                # Calculate days to expiry
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
                days_to_expiry = (expiry_dt - datetime.now()).days

                # Filter by minimum days to expiry
                if days_to_expiry < min_days_to_expiry:
                    continue

                # Get options chain for this expiry
                options_chain = stock.option_chain(expiry_date)

                # Process puts instead of calls
                if not hasattr(options_chain, "puts") or options_chain.puts.empty:
                    logger.warning(
                        f"No put options data available for expiry {expiry_date}"
                    )
                    continue

                puts_df = options_chain.puts.copy()
                puts_df["expiry"] = expiry_date
                puts_df["daysToExpiry"] = days_to_expiry

                # Ensure required columns exist
                if "delta" not in puts_df.columns:
                    puts_df["delta"] = np.nan
                if "ask" not in puts_df.columns:
                    puts_df["ask"] = np.nan
                if "bid" not in puts_df.columns:
                    puts_df["bid"] = np.nan

                # Select columns
                columns_to_keep = [
                    "expiry",
                    "strike",
                    "lastPrice",
                    "impliedVolatility",
                    "volume",
                    "openInterest",
                    "delta",
                    "ask",
                    "bid",
                    "daysToExpiry",
                ]
                available_columns = [
                    col for col in columns_to_keep if col in puts_df.columns
                ]
                puts_df = puts_df[available_columns]

                if not puts_df.empty:
                    all_puts_data.append(puts_df)
                    valid_chains_count += 1

            except Exception as e:
                logger.warning(
                    f"Error processing put expiry {expiry_date} for {ticker}: {str(e)}"
                )
                continue

        result["valid_chains_processed"] = valid_chains_count

        # Combine all put data
        if not all_puts_data:
            error_msg = f"No put options found for {ticker} with min {min_days_to_expiry} days to expiry"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            result["source_used"] = "none"
            return result

        combined_df = pd.concat(all_puts_data, ignore_index=True)

        # Sort by expiry and strike
        combined_df = combined_df.sort_values(["expiry", "strike"]).reset_index(
            drop=True
        )

        # Clean up data types
        numeric_columns = [
            "strike",
            "lastPrice",
            "impliedVolatility",
            "volume",
            "openInterest",
            "delta",
            "ask",
            "bid",
        ]
        for col in numeric_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        # Set successful result
        result["data"] = combined_df
        result["data_available"] = True
        result["error_message"] = None
        result["source_used"] = "yahoo"

        logger.info(
            f"✅ Successfully fetched {len(combined_df)} put options for {ticker}"
        )
        return result

    except Exception as e:
        error_msg = f"Unexpected error fetching put options for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
        result["source_used"] = "none"
        return result


def _apply_long_calls_risk_filtering(
    options_df: pd.DataFrame, ticker: str, risk_tolerance: str
) -> pd.DataFrame:
    """
    Apply risk tolerance filtering to long call options.

    Args:
        options_df: DataFrame containing options data
        ticker: Stock ticker symbol
        risk_tolerance: Risk tolerance level

    Returns:
        pd.DataFrame: Filtered options based on risk tolerance
    """
    filtered_df = options_df.copy()

    try:
        # Get current stock price for moneyness calculations
        stock = yf.Ticker(ticker)
        current_price = stock.info.get(
            "currentPrice", stock.info.get("regularMarketPrice", 100)
        )

        # Calculate moneyness (how close strike is to current price)
        filtered_df["moneyness"] = filtered_df["strike"] / current_price

        if risk_tolerance == "Conservative":
            # Conservative: Prefer ITM to ATM calls (higher probability of success)
            # Moneyness 0.85 to 1.05 (slightly ITM to slightly OTM)
            filtered_df = filtered_df[
                (filtered_df["moneyness"] >= 0.85) & (filtered_df["moneyness"] <= 1.05)
            ]

            # Filter out extremely high IV options (too expensive)
            if "impliedVolatility" in filtered_df.columns:
                iv_95th = filtered_df["impliedVolatility"].quantile(0.95)
                filtered_df = filtered_df[filtered_df["impliedVolatility"] <= iv_95th]

            # Prefer longer time to expiry for conservative approach
            if "daysToExpiry" in filtered_df.columns:
                min_days_conservative = max(filtered_df["daysToExpiry"].min(), 120)
                filtered_df = filtered_df[
                    filtered_df["daysToExpiry"] >= min_days_conservative
                ]

            logger.info(
                f"Conservative filtering: ITM/ATM options, lower IV, longer expiry"
            )

        elif risk_tolerance == "Aggressive":
            # Aggressive: Prefer OTM calls (higher leverage, higher risk/reward)
            # Moneyness 1.0 to 1.25 (ATM to significantly OTM)
            filtered_df = filtered_df[
                (filtered_df["moneyness"] >= 1.0) & (filtered_df["moneyness"] <= 1.25)
            ]

            # Allow shorter time to expiry for more aggressive plays
            if "daysToExpiry" in filtered_df.columns:
                min_days_aggressive = max(filtered_df["daysToExpiry"].min(), 30)
                filtered_df = filtered_df[
                    filtered_df["daysToExpiry"] >= min_days_aggressive
                ]

            logger.info(f"Aggressive filtering: OTM options, higher leverage potential")

        else:  # Moderate
            # Moderate: Prefer ATM to moderately OTM
            # Moneyness 0.95 to 1.15 (slightly ITM to moderately OTM)
            filtered_df = filtered_df[
                (filtered_df["moneyness"] >= 0.95) & (filtered_df["moneyness"] <= 1.15)
            ]

            logger.info(f"Moderate filtering: ATM to moderately OTM options")

        # Remove the temporary moneyness column
        filtered_df = filtered_df.drop("moneyness", axis=1)

        logger.info(
            f"Risk filtering ({risk_tolerance}): {len(options_df)} -> {len(filtered_df)} options"
        )

        return filtered_df

    except Exception as e:
        logger.warning(
            f"Risk filtering failed for {ticker}: {str(e)}. Using unfiltered data."
        )
        return options_df


def _calculate_composite_scores_with_risk_tolerance(
    options_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
    risk_tolerance: str,
) -> pd.DataFrame:
    """
    Calculate composite scores with risk tolerance adjustments.

    This function applies the base composite score calculation and then
    adjusts scoring based on risk tolerance preferences.
    """
    # First calculate base composite scores
    scored_df = _calculate_composite_scores(
        options_df, rsi, beta, momentum, avg_iv, forecast_confidence, data_availability
    )

    # Apply risk tolerance adjustments
    if risk_tolerance == "Conservative":
        # Conservative: Boost lower volatility, penalize high volatility
        if "impliedVolatility" in scored_df.columns:
            iv_penalty = scored_df["impliedVolatility"].apply(
                lambda iv: max(0, (iv - avg_iv) / avg_iv * 0.2)  # Penalty for high IV
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 - iv_penalty)

        # Boost scores for longer time to expiry
        if "daysToExpiry" in scored_df.columns:
            time_boost = scored_df["daysToExpiry"].apply(
                lambda days: min(0.15, days / 365 * 0.1)  # Boost for longer time
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 + time_boost)

        logger.info("Applied conservative scoring adjustments: -high IV, +longer time")

    elif risk_tolerance == "Aggressive":
        # Aggressive: Boost momentum and growth indicators
        if momentum > 0:
            momentum_boost = min(0.2, momentum * 10)  # Boost for positive momentum
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + momentum_boost
            )

        # Boost scores for higher beta (more volatile stocks)
        if beta > 1.0:
            beta_boost = min(0.15, (beta - 1.0) * 0.2)  # Boost for higher beta
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 + beta_boost)

        logger.info("Applied aggressive scoring adjustments: +momentum, +high beta")

    # Moderate tolerance uses base scores without adjustments

    # Ensure scores remain in valid range
    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    # Add risk tolerance metadata to score details
    scored_df["score_details"] = scored_df["score_details"].apply(
        lambda details: {**details, "risk_tolerance": risk_tolerance}
    )

    return scored_df


def _apply_final_risk_tolerance_selection(
    scored_df: pd.DataFrame, risk_tolerance: str, top_n: int
) -> pd.DataFrame:
    """
    Apply final risk tolerance considerations to option selection and ranking.

    Args:
        scored_df: DataFrame with calculated composite scores
        risk_tolerance: Risk tolerance level
        top_n: Number of recommendations to return

    Returns:
        pd.DataFrame: Final ranked and filtered recommendations
    """
    if risk_tolerance == "Conservative":
        # Conservative: Prefer higher probability of success
        # Sort by: Composite Score (primary), then lower IV (secondary), then longer time (tertiary)
        sort_columns = ["AdjustedCompositeScore"]
        sort_ascending = [False]

        if "impliedVolatility" in scored_df.columns:
            sort_columns.append("impliedVolatility")
            sort_ascending.append(True)  # Lower IV first

        if "daysToExpiry" in scored_df.columns:
            sort_columns.append("daysToExpiry")
            sort_ascending.append(False)  # Longer time first

        ranked_df = scored_df.sort_values(sort_columns, ascending=sort_ascending)

        # Conservative gets fewer but higher quality recommendations
        conservative_top_n = max(3, min(top_n, 5))
        final_recommendations = ranked_df.head(conservative_top_n)

        logger.info(
            f"Conservative selection: {conservative_top_n} high-probability recommendations"
        )

    elif risk_tolerance == "Aggressive":
        # Aggressive: Prefer higher potential returns (accepting higher risk)
        # Sort by: Composite Score (primary), then higher potential leverage
        sort_columns = ["AdjustedCompositeScore"]
        sort_ascending = [False]

        # For aggressive, we might want to include more OTM options with higher potential
        ranked_df = scored_df.sort_values(sort_columns, ascending=sort_ascending)

        # Aggressive gets more recommendations to choose from
        aggressive_top_n = min(top_n * 2, 15)  # More options
        final_recommendations = ranked_df.head(aggressive_top_n)

        logger.info(
            f"Aggressive selection: {aggressive_top_n} high-leverage recommendations"
        )

    else:  # Moderate
        # Moderate: Balanced approach
        ranked_df = scored_df.sort_values(
            ["AdjustedCompositeScore", "daysToExpiry"], ascending=[False, True]
        )
        final_recommendations = ranked_df.head(top_n)

        logger.info(f"Moderate selection: {top_n} balanced recommendations")

    return final_recommendations


def recommend_bull_call_spread(
    ticker: str,
    min_days: int = 180,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
) -> pd.DataFrame:
    """
    Analyze bull call spread opportunities using comprehensive scoring.

    Bull call spreads involve buying a lower strike call and selling a higher strike call
    with the same expiration date. This strategy has limited risk and limited reward.

    Args:
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry
        top_n: Number of top recommendations
        risk_tolerance: Risk tolerance level ('Conservative', 'Moderate', 'Aggressive')

    Returns:
        pd.DataFrame: Top bull call spread recommendations with spread analysis
    """
    start_time = datetime.now()
    logger.info(
        f"Starting bull call spread analysis for {ticker} with {risk_tolerance} risk tolerance"
    )

    try:
        # Get call options data
        options_result = fetch_long_dated_calls(ticker, min_days_to_expiry=min_days)

        if not options_result["data_available"]:
            raise InsufficientDataError(f"No call options data available for {ticker}")

        options_df = options_result["data"]
        if options_df.empty:
            raise InsufficientDataError(f"No call options found for {ticker}")

        # Get current stock price for spread analysis
        stock = yf.Ticker(ticker)
        current_price = stock.info.get(
            "currentPrice", stock.info.get("regularMarketPrice", 100)
        )

        # Generate bull call spreads with risk tolerance considerations
        spreads_data = []

        # Risk tolerance affects spread width and selection
        if risk_tolerance == "Conservative":
            max_spread_width = current_price * 0.10  # Narrower spreads
            min_profit_ratio = 1.5  # Higher profit ratio required
        elif risk_tolerance == "Aggressive":
            max_spread_width = current_price * 0.20  # Wider spreads
            min_profit_ratio = 1.0  # Lower profit ratio acceptable
        else:  # Moderate
            max_spread_width = current_price * 0.15  # Moderate spreads
            min_profit_ratio = 1.2  # Moderate profit ratio

        # Group by expiry date
        for expiry_date in options_df["expiry"].unique():
            expiry_options = options_df[
                options_df["expiry"] == expiry_date
            ].sort_values("strike")

            # Create spreads (buy lower strike, sell higher strike)
            for i in range(len(expiry_options) - 1):
                long_call = expiry_options.iloc[i]  # Lower strike (buy)
                for j in range(
                    i + 1, min(i + 8, len(expiry_options))
                ):  # Extended search
                    short_call = expiry_options.iloc[j]  # Higher strike (sell)

                    # Check spread width against risk tolerance
                    spread_width = short_call["strike"] - long_call["strike"]
                    if spread_width > max_spread_width:
                        continue

                    # Calculate spread metrics
                    net_premium = long_call["lastPrice"] - short_call["lastPrice"]
                    max_profit = spread_width - net_premium
                    max_loss = net_premium
                    break_even = long_call["strike"] + net_premium

                    # Skip if net premium is negative (would be a credit spread)
                    if net_premium <= 0:
                        continue

                    # Skip if max profit is negative
                    if max_profit <= 0:
                        continue

                    # Calculate profit ratio and filter by risk tolerance
                    profit_ratio = max_profit / net_premium if net_premium > 0 else 0
                    if profit_ratio < min_profit_ratio:
                        continue

                    spreads_data.append(
                        {
                            "ticker": ticker,
                            "expiry": expiry_date,
                            "long_strike": long_call["strike"],
                            "short_strike": short_call["strike"],
                            "long_price": long_call["lastPrice"],
                            "short_price": short_call["lastPrice"],
                            "net_premium": net_premium,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "break_even": break_even,
                            "profit_ratio": profit_ratio,
                            "spread_width": spread_width,
                            "daysToExpiry": long_call["daysToExpiry"],
                            "long_iv": long_call["impliedVolatility"],
                            "short_iv": short_call["impliedVolatility"],
                            "risk_tolerance_applied": risk_tolerance,
                        }
                    )

        if not spreads_data:
            raise InsufficientDataError(
                f"No viable bull call spreads found for {ticker} with {risk_tolerance} criteria"
            )

        spreads_df = pd.DataFrame(spreads_data)

        # Get technical scores (same as long calls)
        stock_prices = fetch_price_history(ticker, period="1y")
        spy_prices = fetch_price_history("SPY", period="1y")
        (
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        ) = compute_scores(ticker, stock_prices, spy_prices, options_df)

        # Add technical indicators to spreads
        spreads_df["RSI"] = rsi
        spreads_df["Beta"] = beta
        spreads_df["Momentum"] = momentum
        spreads_df["IV"] = (spreads_df["long_iv"] + spreads_df["short_iv"]) / 2
        spreads_df["ForecastConfidence"] = forecast_confidence

        # Calculate composite score with spread-specific weighting and risk tolerance
        spreads_df = _calculate_spread_composite_scores_with_risk_tolerance(
            spreads_df,
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
            risk_tolerance,
        )

        # Apply risk tolerance to final selection
        if risk_tolerance == "Conservative":
            # Conservative: Prefer higher profit ratios and longer time
            sort_cols = ["CompositeScore", "profit_ratio", "daysToExpiry"]
            sort_asc = [False, False, False]
            final_top_n = min(top_n, 5)
        elif risk_tolerance == "Aggressive":
            # Aggressive: Prefer higher potential returns
            sort_cols = ["CompositeScore", "max_profit", "profit_ratio"]
            sort_asc = [False, False, False]
            final_top_n = min(top_n * 2, 10)
        else:  # Moderate
            sort_cols = ["CompositeScore", "profit_ratio"]
            sort_asc = [False, False]
            final_top_n = top_n

        spreads_df = spreads_df.sort_values(sort_cols, ascending=sort_asc)

        # Return top recommendations
        result_df = spreads_df.head(final_top_n).reset_index(drop=True)

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Bull call spread analysis completed for {ticker} with {risk_tolerance} risk tolerance in {execution_time:.2f}s"
        )

        return result_df

    except Exception as e:
        logger.error(f"Error in bull call spread analysis for {ticker}: {str(e)}")
        raise OptionsAdvisorError(
            f"Bull call spread analysis failed for {ticker}: {str(e)}"
        )


def _calculate_spread_composite_scores_with_risk_tolerance(
    spreads_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
    risk_tolerance: str,
) -> pd.DataFrame:
    """Calculate composite scores for bull call spreads with risk tolerance adjustments."""
    # Start with base spread scoring
    scored_df = _calculate_spread_composite_scores(
        spreads_df, rsi, beta, momentum, avg_iv, forecast_confidence, data_availability
    )

    # Apply risk tolerance specific adjustments
    if risk_tolerance == "Conservative":
        # Conservative: Favor higher profit ratios and lower risk
        profit_ratio_boost = (
            scored_df["profit_ratio"]
            .apply(lambda x: min((x - 1.0) * 0.2, 0.3))  # Boost for profit ratio > 1.0
            .clip(0, 0.3)
        )

        # Penalize very wide spreads (higher risk)
        if "spread_width" in scored_df.columns:
            width_penalty = scored_df["spread_width"].apply(
                lambda x: min(x / (scored_df["spread_width"].mean() * 2) * 0.1, 0.2)
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + profit_ratio_boost - width_penalty
            )
        else:
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + profit_ratio_boost
            )

    elif risk_tolerance == "Aggressive":
        # Aggressive: Favor higher absolute profit potential
        max_profit_boost = (
            scored_df["max_profit"]
            .apply(lambda x: min(x / scored_df["max_profit"].mean() * 0.15, 0.25))
            .clip(0, 0.25)
        )
        scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
            1 + max_profit_boost
        )

    # Ensure scores remain valid
    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    return scored_df


def recommend_covered_call(
    ticker: str,
    min_days: int = 30,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
) -> pd.DataFrame:
    """
    Analyze covered call opportunities for income generation.

    Covered calls involve owning 100 shares of stock and selling call options against them.
    This strategy generates income but caps upside potential.

    Args:
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry (shorter for covered calls)
        top_n: Number of top recommendations
        risk_tolerance: Risk tolerance level ('Conservative', 'Moderate', 'Aggressive')

    Returns:
        pd.DataFrame: Top covered call recommendations
    """
    start_time = datetime.now()
    logger.info(
        f"Starting covered call analysis for {ticker} with {risk_tolerance} risk tolerance"
    )

    try:
        # Get call options data
        options_result = fetch_long_dated_calls(ticker, min_days_to_expiry=min_days)

        if not options_result["data_available"]:
            raise InsufficientDataError(f"No call options data available for {ticker}")

        options_df = options_result["data"]
        if options_df.empty:
            raise InsufficientDataError(f"No call options found for {ticker}")

        # Get current stock price
        stock = yf.Ticker(ticker)
        current_price = stock.info.get(
            "currentPrice", stock.info.get("regularMarketPrice", 100)
        )

        # Apply risk tolerance filtering for covered calls
        if risk_tolerance == "Conservative":
            # Conservative: Further out-of-the-money to reduce assignment risk
            min_otm_pct = 0.05  # At least 5% OTM
            max_otm_pct = 0.15  # No more than 15% OTM
            min_yield_threshold = 0.005  # At least 0.5% premium yield
        elif risk_tolerance == "Aggressive":
            # Aggressive: Closer to the money for higher premiums
            min_otm_pct = 0.0  # Can be ATM
            max_otm_pct = 0.08  # No more than 8% OTM
            min_yield_threshold = 0.015  # At least 1.5% premium yield
        else:  # Moderate
            min_otm_pct = 0.02  # At least 2% OTM
            max_otm_pct = 0.12  # No more than 12% OTM
            min_yield_threshold = 0.01  # At least 1% premium yield

        # Filter for appropriate covered call candidates based on risk tolerance
        covered_calls = options_df[
            (options_df["strike"] >= current_price * (1 + min_otm_pct))
            & (options_df["strike"] <= current_price * (1 + max_otm_pct))
        ].copy()

        if covered_calls.empty:
            raise InsufficientDataError(
                f"No suitable covered call options found for {ticker} with {risk_tolerance} criteria"
            )

        # Calculate covered call metrics
        covered_calls["current_price"] = current_price
        covered_calls["premium_yield"] = (
            covered_calls["lastPrice"] / current_price
        ) * 100
        covered_calls["annualized_yield"] = (
            covered_calls["premium_yield"] * 365
        ) / covered_calls["daysToExpiry"]
        covered_calls["upside_capture"] = (
            (covered_calls["strike"] - current_price) / current_price
        ) * 100
        covered_calls["total_return"] = (
            covered_calls["premium_yield"] + covered_calls["upside_capture"]
        )
        covered_calls["assignment_probability"] = covered_calls["strike"].apply(
            lambda x: max(
                0, min(1, (current_price - x) / current_price + 0.5)
            )  # Rough estimate
        )

        # Filter by minimum yield threshold
        covered_calls = covered_calls[
            covered_calls["premium_yield"] >= min_yield_threshold * 100
        ]

        if covered_calls.empty:
            raise InsufficientDataError(
                f"No covered call options meet minimum yield requirements for {risk_tolerance}"
            )

        # Get technical scores
        stock_prices = fetch_price_history(ticker, period="1y")
        spy_prices = fetch_price_history("SPY", period="1y")
        (
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        ) = compute_scores(ticker, stock_prices, spy_prices, options_df)

        # Add technical indicators
        covered_calls["RSI"] = rsi
        covered_calls["Beta"] = beta
        covered_calls["Momentum"] = momentum
        covered_calls["IV"] = covered_calls["impliedVolatility"]
        covered_calls["ForecastConfidence"] = forecast_confidence
        covered_calls["risk_tolerance_applied"] = risk_tolerance

        # Calculate composite score for covered calls (emphasize income and lower volatility)
        covered_calls = _calculate_covered_call_composite_scores_with_risk_tolerance(
            covered_calls,
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
            risk_tolerance,
        )

        # Apply risk tolerance to final selection
        if risk_tolerance == "Conservative":
            # Conservative: Prefer lower assignment risk and consistent income
            sort_cols = ["CompositeScore", "assignment_probability", "annualized_yield"]
            sort_asc = [False, True, False]  # Lower assignment probability first
            final_top_n = min(top_n, 5)
        elif risk_tolerance == "Aggressive":
            # Aggressive: Prefer higher income potential
            sort_cols = ["CompositeScore", "annualized_yield", "total_return"]
            sort_asc = [False, False, False]
            final_top_n = min(top_n * 2, 8)
        else:  # Moderate
            sort_cols = ["CompositeScore", "annualized_yield"]
            sort_asc = [False, False]
            final_top_n = top_n

        covered_calls = covered_calls.sort_values(sort_cols, ascending=sort_asc)

        # Select relevant columns for display
        result_columns = [
            "ticker",
            "strike",
            "expiry",
            "lastPrice",
            "IV",
            "RSI",
            "Beta",
            "Momentum",
            "ForecastConfidence",
            "CompositeScore",
            "premium_yield",
            "annualized_yield",
            "upside_capture",
            "total_return",
            "assignment_probability",
            "daysToExpiry",
            "risk_tolerance_applied",
        ]

        available_columns = [
            col for col in result_columns if col in covered_calls.columns
        ]
        result_df = (
            covered_calls[available_columns].head(final_top_n).reset_index(drop=True)
        )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Covered call analysis completed for {ticker} with {risk_tolerance} risk tolerance in {execution_time:.2f}s"
        )

        return result_df

    except Exception as e:
        logger.error(f"Error in covered call analysis for {ticker}: {str(e)}")
        raise OptionsAdvisorError(
            f"Covered call analysis failed for {ticker}: {str(e)}"
        )


def recommend_cash_secured_put(
    ticker: str,
    min_days: int = 30,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
) -> pd.DataFrame:
    """
    Analyze cash-secured put opportunities for income generation and potential stock acquisition.

    Cash-secured puts involve selling put options while holding enough cash to buy
    100 shares if assigned. This generates income and can result in stock acquisition at
    a discount.

    Args:
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry
        top_n: Number of top recommendations
        risk_tolerance: Risk tolerance level ('Conservative', 'Moderate', 'Aggressive')

    Returns:
        pd.DataFrame: Top cash-secured put recommendations
    """
    start_time = datetime.now()
    logger.info(
        f"Starting cash-secured put analysis for {ticker} with {risk_tolerance} risk tolerance"
    )

    try:
        # Get put options data
        puts_result = fetch_put_options(ticker, min_days_to_expiry=min_days)

        if not puts_result["data_available"]:
            raise InsufficientDataError(f"No put options data available for {ticker}")

        puts_df = puts_result["data"]
        if puts_df.empty:
            raise InsufficientDataError(f"No put options found for {ticker}")

        # Get current stock price
        stock = yf.Ticker(ticker)
        current_price = stock.info.get(
            "currentPrice", stock.info.get("regularMarketPrice", 100)
        )

        # Apply risk tolerance filtering for cash-secured puts
        if risk_tolerance == "Conservative":
            # Conservative: Further out-of-the-money to reduce assignment risk
            min_otm_pct = 0.08  # At least 8% below current price
            max_otm_pct = 0.25  # No more than 25% below
            min_yield_threshold = 0.005  # At least 0.5% premium yield
        elif risk_tolerance == "Aggressive":
            # Aggressive: Closer to the money for higher premiums and potential stock acquisition
            min_otm_pct = 0.02  # At least 2% below current price
            max_otm_pct = 0.15  # No more than 15% below
            min_yield_threshold = 0.015  # At least 1.5% premium yield
        else:  # Moderate
            min_otm_pct = 0.05  # At least 5% below current price
            max_otm_pct = 0.20  # No more than 20% below
            min_yield_threshold = 0.01  # At least 1% premium yield

        # Filter for out-of-the-money puts (cash-secured put candidates)
        cash_secured_puts = puts_df[
            (puts_df["strike"] <= current_price * (1 - min_otm_pct))
            & (puts_df["strike"] >= current_price * (1 - max_otm_pct))
        ].copy()

        if cash_secured_puts.empty:
            raise InsufficientDataError(
                f"No suitable cash-secured put options found for {ticker} with {risk_tolerance} criteria"
            )

        # Calculate cash-secured put metrics
        cash_secured_puts["current_price"] = current_price
        cash_secured_puts["premium_yield"] = (
            cash_secured_puts["lastPrice"] / cash_secured_puts["strike"]
        ) * 100
        cash_secured_puts["annualized_yield"] = (
            cash_secured_puts["premium_yield"] * 365
        ) / cash_secured_puts["daysToExpiry"]
        cash_secured_puts["assignment_discount"] = (
            (current_price - cash_secured_puts["strike"]) / current_price
        ) * 100
        cash_secured_puts["effective_cost"] = (
            cash_secured_puts["strike"] - cash_secured_puts["lastPrice"]
        )
        cash_secured_puts["discount_to_current"] = (
            (current_price - cash_secured_puts["effective_cost"]) / current_price
        ) * 100
        cash_secured_puts["assignment_probability"] = cash_secured_puts["strike"].apply(
            lambda x: max(
                0, min(1, (x - current_price) / current_price + 0.5)
            )  # Rough estimate
        )

        # Filter by minimum yield threshold
        cash_secured_puts = cash_secured_puts[
            cash_secured_puts["premium_yield"] >= min_yield_threshold * 100
        ]

        if cash_secured_puts.empty:
            raise InsufficientDataError(
                f"No cash-secured put options meet minimum yield requirements for {risk_tolerance}"
            )

        # Get technical scores (using call options for technical analysis)
        calls_result = fetch_long_dated_calls(ticker, min_days_to_expiry=min_days)
        if calls_result["data_available"]:
            calls_df = calls_result["data"]
        else:
            calls_df = pd.DataFrame()  # Fallback for technical analysis

        stock_prices = fetch_price_history(ticker, period="1y")
        spy_prices = fetch_price_history("SPY", period="1y")
        (
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        ) = compute_scores(ticker, stock_prices, spy_prices, calls_df)

        # Add technical indicators
        cash_secured_puts["RSI"] = rsi
        cash_secured_puts["Beta"] = beta
        cash_secured_puts["Momentum"] = momentum
        cash_secured_puts["IV"] = cash_secured_puts["impliedVolatility"]
        cash_secured_puts["ForecastConfidence"] = forecast_confidence
        cash_secured_puts["risk_tolerance_applied"] = risk_tolerance

        # Calculate composite score for cash-secured puts
        cash_secured_puts = _calculate_csp_composite_scores_with_risk_tolerance(
            cash_secured_puts,
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
            risk_tolerance,
        )

        # Apply risk tolerance to final selection
        if risk_tolerance == "Conservative":
            # Conservative: Prefer lower assignment risk and good entry points
            sort_cols = [
                "CompositeScore",
                "assignment_probability",
                "discount_to_current",
            ]
            sort_asc = [
                False,
                True,
                False,
            ]  # Lower assignment probability, higher discount
            final_top_n = min(top_n, 5)
        elif risk_tolerance == "Aggressive":
            # Aggressive: Prefer higher income and willing to accept assignment
            sort_cols = ["CompositeScore", "annualized_yield", "assignment_probability"]
            sort_asc = [
                False,
                False,
                False,
            ]  # Higher yield, higher assignment probability OK
            final_top_n = min(top_n * 2, 8)
        else:  # Moderate
            sort_cols = ["CompositeScore", "annualized_yield"]
            sort_asc = [False, False]
            final_top_n = top_n

        cash_secured_puts = cash_secured_puts.sort_values(sort_cols, ascending=sort_asc)

        # Select relevant columns for display
        result_columns = [
            "ticker",
            "strike",
            "expiry",
            "lastPrice",
            "IV",
            "RSI",
            "Beta",
            "Momentum",
            "ForecastConfidence",
            "CompositeScore",
            "premium_yield",
            "annualized_yield",
            "assignment_discount",
            "effective_cost",
            "discount_to_current",
            "assignment_probability",
            "daysToExpiry",
            "risk_tolerance_applied",
        ]

        available_columns = [
            col for col in result_columns if col in cash_secured_puts.columns
        ]
        result_df = (
            cash_secured_puts[available_columns]
            .head(final_top_n)
            .reset_index(drop=True)
        )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Cash-secured put analysis completed for {ticker} with {risk_tolerance} risk tolerance in {execution_time:.2f}s"
        )

        return result_df

    except Exception as e:
        logger.error(f"Error in cash-secured put analysis for {ticker}: {str(e)}")
        raise OptionsAdvisorError(
            f"Cash-secured put analysis failed for {ticker}: {str(e)}"
        )


def _calculate_covered_call_composite_scores_with_risk_tolerance(
    covered_calls_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
    risk_tolerance: str,
) -> pd.DataFrame:
    """Calculate composite scores for covered calls with risk tolerance adjustments."""
    # Start with base covered call scoring
    scored_df = _calculate_covered_call_composite_scores(
        covered_calls_df,
        rsi,
        beta,
        momentum,
        avg_iv,
        forecast_confidence,
        data_availability,
    )

    # Apply risk tolerance specific adjustments
    if risk_tolerance == "Conservative":
        # Conservative: Heavily weight assignment probability (lower is better)
        if "assignment_probability" in scored_df.columns:
            assignment_penalty = scored_df["assignment_probability"].apply(
                lambda x: x * 0.3  # High penalty for high assignment probability
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 - assignment_penalty
            )

        # Boost for longer time to expiry
        if "daysToExpiry" in scored_df.columns:
            time_boost = scored_df["daysToExpiry"].apply(
                lambda days: min(0.2, days / 90 * 0.1)  # Boost for longer time
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 + time_boost)

    elif risk_tolerance == "Aggressive":
        # Aggressive: Boost higher yields even with higher assignment risk
        if "annualized_yield" in scored_df.columns:
            yield_boost = scored_df["annualized_yield"].apply(
                lambda x: min(x / 50.0, 0.3)  # Boost for higher yields
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + yield_boost
            )

        # Less penalty for assignment probability
        if "assignment_probability" in scored_df.columns:
            assignment_bonus = scored_df["assignment_probability"].apply(
                lambda x: x
                * 0.1  # Small bonus for higher assignment probability (willing to sell)
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + assignment_bonus
            )

    # Ensure scores remain valid
    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    return scored_df


def _calculate_csp_composite_scores_with_risk_tolerance(
    csp_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
    risk_tolerance: str,
) -> pd.DataFrame:
    """Calculate composite scores for cash-secured puts with risk tolerance adjustments."""
    # Start with base CSP scoring
    scored_df = _calculate_csp_composite_scores(
        csp_df, rsi, beta, momentum, avg_iv, forecast_confidence, data_availability
    )

    # Apply risk tolerance specific adjustments
    if risk_tolerance == "Conservative":
        # Conservative: Favor lower assignment probability and better entry discounts
        if "assignment_probability" in scored_df.columns:
            assignment_penalty = scored_df["assignment_probability"].apply(
                lambda x: x * 0.25  # Penalty for high assignment probability
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 - assignment_penalty
            )

        # Boost for better entry discounts
        if "discount_to_current" in scored_df.columns:
            discount_boost = scored_df["discount_to_current"].apply(
                lambda x: min(x / 20.0, 0.2)  # Boost for higher discount
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + discount_boost
            )

    elif risk_tolerance == "Aggressive":
        # Aggressive: Favor higher yields and willing to accept assignment
        if "annualized_yield" in scored_df.columns:
            yield_boost = scored_df["annualized_yield"].apply(
                lambda x: min(x / 40.0, 0.3)  # Strong boost for higher yields
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + yield_boost
            )

        # Assignment probability is less of a concern (willing to own stock)
        if "assignment_probability" in scored_df.columns:
            assignment_bonus = scored_df["assignment_probability"].apply(
                lambda x: x * 0.05  # Small bonus for assignment (want to own stock)
            )
            scored_df["CompositeScore"] = scored_df["CompositeScore"] * (
                1 + assignment_bonus
            )

    # Ensure scores remain valid
    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    return scored_df


def analyze_options_strategy(
    strategy_type: str,
    ticker: str,
    min_days: int = 180,
    top_n: int = 5,
    risk_tolerance: str = "Conservative",
    time_horizon: str = "Medium-term (3-6 months)",
) -> pd.DataFrame:
    """
    Main strategy dispatcher that routes to the appropriate options analysis function.

    Args:
        strategy_type: Options strategy to analyze
        ticker: Stock ticker symbol
        min_days: Minimum days to expiry
        top_n: Number of recommendations
        risk_tolerance: Risk tolerance level
        time_horizon: Investment time horizon

    Returns:
        pd.DataFrame: Strategy-specific recommendations

    Raises:
        OptionsAdvisorError: If strategy is not supported or analysis fails
    """
    start_time = datetime.now()
    logger.info(f"Analyzing {strategy_type} strategy for {ticker}")

    # Validate strategy type
    supported_strategies = [
        "Long Calls",
        "Bull Call Spread",
        "Covered Call",
        "Cash-Secured Put",
    ]

    if strategy_type not in supported_strategies:
        raise OptionsAdvisorError(
            f"Unsupported strategy: {strategy_type}. Supported: {supported_strategies}"
        )

    # Adjust parameters based on strategy and risk tolerance
    if strategy_type in ["Covered Call", "Cash-Secured Put"]:
        # Shorter timeframes for income strategies
        if min_days > 90:
            min_days = min(90, min_days)
            logger.info(f"Adjusted min_days to {min_days} for {strategy_type} strategy")

    # Risk tolerance adjustments
    if risk_tolerance == "Conservative":
        min_days = max(min_days, 60)  # Longer timeframes for conservative
    elif risk_tolerance == "Aggressive":
        top_n = min(top_n * 2, 10)  # More options for aggressive traders

    try:
        # Route to appropriate strategy function
        if strategy_type == "Long Calls":
            recommendations = recommend_long_calls(
                ticker, min_days, top_n, risk_tolerance
            )
        elif strategy_type == "Bull Call Spread":
            recommendations = recommend_bull_call_spread(
                ticker, min_days, top_n, risk_tolerance
            )
        elif strategy_type == "Covered Call":
            recommendations = recommend_covered_call(
                ticker, min_days, top_n, risk_tolerance
            )
        elif strategy_type == "Cash-Secured Put":
            recommendations = recommend_cash_secured_put(
                ticker, min_days, top_n, risk_tolerance
            )
        else:
            raise OptionsAdvisorError(f"Strategy dispatcher error: {strategy_type}")

        # Add strategy metadata
        if not recommendations.empty:
            recommendations["strategy_type"] = strategy_type
            recommendations["risk_tolerance"] = risk_tolerance
            recommendations["time_horizon"] = time_horizon
            recommendations["analysis_date"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Strategy analysis completed for {strategy_type} on {ticker} in {execution_time:.2f}s"
        )

        return recommendations

    except (OptionsAdvisorError, InsufficientDataError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in strategy analysis: {str(e)}", exc_info=True)
        raise OptionsAdvisorError(
            f"Strategy analysis failed for {strategy_type} on {ticker}: {str(e)}"
        )


def _calculate_covered_call_composite_scores(
    covered_calls_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
) -> pd.DataFrame:
    """Calculate base composite scores for covered calls (emphasize income and stability)."""
    scored_df = covered_calls_df.copy()

    # Normalize weights - for covered calls, we want lower volatility and higher income
    available_sources = [k for k, v in data_availability.items() if v]
    normalized_weights = normalize_scoring_weights(SCORING_WEIGHTS, available_sources)

    # Calculate individual scores (adjusted for covered call strategy)
    score_components = {}

    if data_availability["rsi"]:
        # For covered calls, moderate RSI is preferred (not oversold)
        rsi_score = _normalize_score(rsi, 30, 70, invert=False) if rsi < 70 else 0.3
        score_components["rsi"] = rsi_score

    if data_availability["beta"]:
        # Lower beta preferred for covered calls (less volatility)
        beta_score = _normalize_score(beta, 0.5, 1.2, invert=True)
        score_components["beta"] = beta_score

    if data_availability["momentum"]:
        # Slight positive momentum is good, but not too aggressive
        momentum_score = _normalize_score(momentum, -0.05, 0.05, invert=False)
        score_components["momentum"] = momentum_score

    if data_availability["forecast"]:
        # Moderate forecast confidence (don't want too bullish for covered calls)
        forecast_score = min(forecast_confidence, 0.7)  # Cap at 0.7
        score_components["forecast"] = forecast_score

    # Calculate IV score
    if data_availability["iv"]:
        # Higher IV is better for covered calls (more premium income)
        scored_df["iv_score"] = scored_df["IV"].apply(
            lambda iv: _normalize_score(iv, avg_iv * 0.8, avg_iv * 1.5, invert=False)
        )
    else:
        scored_df["iv_score"] = 0.5

    # Calculate composite score
    scored_df["CompositeScore"] = 0.0

    for source, weight in normalized_weights.items():
        if source == "iv":
            scored_df["CompositeScore"] += weight * scored_df["iv_score"]
        elif source in score_components:
            scored_df["CompositeScore"] += weight * score_components[source]

    # Boost score based on annualized yield
    if "annualized_yield" in scored_df.columns:
        yield_boost = scored_df["annualized_yield"].apply(
            lambda x: min(x / 50.0, 0.3)
        )  # Cap boost at 0.3
        scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 + yield_boost)

    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    # Add score details
    scored_df["score_details"] = scored_df.apply(
        lambda row: {
            k: v for k, v in normalized_weights.items() if k in available_sources
        },
        axis=1,
    )

    return scored_df


def _calculate_csp_composite_scores(
    csp_df: pd.DataFrame,
    rsi: float,
    beta: float,
    momentum: float,
    avg_iv: float,
    forecast_confidence: float,
    data_availability: dict[str, bool],
) -> pd.DataFrame:
    """Calculate base composite scores for cash-secured puts."""
    scored_df = csp_df.copy()

    # Normalize weights
    available_sources = [k for k, v in data_availability.items() if v]
    normalized_weights = normalize_scoring_weights(SCORING_WEIGHTS, available_sources)

    # Calculate individual scores (adjusted for CSP strategy)
    score_components = {}

    if data_availability["rsi"]:
        # For CSPs, slightly oversold is good (better entry point)
        rsi_score = _normalize_score(rsi, 20, 60, invert=True)
        score_components["rsi"] = rsi_score

    if data_availability["beta"]:
        # Moderate beta for CSPs
        beta_score = _normalize_score(beta, 0.7, 1.3, invert=False)
        score_components["beta"] = beta_score

    if data_availability["momentum"]:
        # Slight negative momentum can be good for CSPs (potential bounce)
        momentum_score = _normalize_score(momentum, -0.1, 0.05, invert=True)
        score_components["momentum"] = momentum_score

    if data_availability["forecast"]:
        # Good forecast confidence for eventual stock ownership
        score_components["forecast"] = forecast_confidence

    # Calculate IV score
    if data_availability["iv"]:
        # Higher IV is better for CSPs (more premium income)
        scored_df["iv_score"] = scored_df["IV"].apply(
            lambda iv: _normalize_score(iv, avg_iv * 0.8, avg_iv * 1.5, invert=False)
        )
    else:
        scored_df["iv_score"] = 0.5

    # Calculate composite score
    scored_df["CompositeScore"] = 0.0

    for source, weight in normalized_weights.items():
        if source == "iv":
            scored_df["CompositeScore"] += weight * scored_df["iv_score"]
        elif source in score_components:
            scored_df["CompositeScore"] += weight * score_components[source]

    # Boost score based on annualized yield and discount
    if "annualized_yield" in scored_df.columns:
        yield_boost = scored_df["annualized_yield"].apply(lambda x: min(x / 40.0, 0.2))
    else:
        yield_boost = 0

    if "discount_to_current" in scored_df.columns:
        discount_boost = scored_df["discount_to_current"].apply(
            lambda x: min(x / 20.0, 0.2)
        )
    else:
        discount_boost = 0

    total_boost = yield_boost + discount_boost

    scored_df["CompositeScore"] = scored_df["CompositeScore"] * (1 + total_boost)
    scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

    # Add score details
    scored_df["score_details"] = scored_df.apply(
        lambda row: {
            k: v for k, v in normalized_weights.items() if k in available_sources
        },
        axis=1,
    )

    return scored_df
