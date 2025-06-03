"""
Options Advisor Module

This module analyzes long-dated call options for a given stock and recommends
the best ones using a comprehensive technical scoring framework. It combines
multiple technical indicators (RSI, Beta, Momentum, Implied Volatility) with
forward-looking analyst forecast data to provide a composite score for each
option contract.

The module is designed to be extensible and integrates seamlessly with
existing data fetching and mathematical utilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from analysis.ecosystem import EcosystemAnalyzer
from data.forecast_fetcher import ForecastFetchError, get_analyst_forecast
from data.options_fetcher import OptionsDataError, OptionsResult, fetch_long_dated_calls
from data.peer_fetcher import PeerFetchError, get_peers
from utils.errors import DataError, DataFetcherError, ErrorSeverity, handle_data_error
from utils.logger import setup_logger
from utils.options_math import (
    OptionsMathError,
    calculate_average_iv,
    calculate_beta,
    calculate_momentum,
    calculate_rsi,
)
from utils.validators import validate_ticker

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
    ticker: str, min_days: int = 180, top_n: int = 5
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

    Returns:
        pd.DataFrame: Top-ranked option recommendations with columns:
            - ticker: Stock ticker symbol
            - strike: Option strike price
            - expiry: Option expiry date
            - lastPrice: Last traded option price
            - IV: Implied volatility (renamed from impliedVolatility)
            - RSI: RSI value for the underlying stock
            - Beta: Beta coefficient vs SPY
            - Momentum: Price momentum indicator
            - ForecastConfidence: Analyst forecast confidence (0-1)
            - CompositeScore: Weighted composite score
            - ecosystem_score: Ecosystem analysis score (0-1)
            - signal: Ecosystem signal (confirm/neutral/veto)
            - confidence: Ecosystem confidence level (0-1)

    Raises:
        OptionsAdvisorError: If input validation fails or analysis cannot be completed
        InsufficientDataError: If there's insufficient data for meaningful analysis
        CalculationError: If technical indicator calculations fail

    Examples:
        >>> # Get top 5 long-term call recommendations for Apple
        >>> recommendations = recommend_long_calls('AAPL')
        >>> print(f"Top recommendation: Strike ${recommendations.iloc[0]['strike']:.2f}")

        >>> # Get top 3 recommendations with minimum 1 year to expiry
        >>> long_term_calls = recommend_long_calls('MSFT', min_days=365, top_n=3)
        >>> print(f"Found {len(long_term_calls)} recommendations")
    """
    start_time = datetime.now()
    logger.info(
        f"Starting options analysis for {ticker} (min_days={min_days}, top_n={top_n})"
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

        # Step 2: Fetch historical price data
        logger.info("Step 2: Fetching historical price data")
        stock_prices = fetch_price_history(ticker, period="1y")
        spy_prices = fetch_price_history("SPY", period="1y")

        # Step 3: Compute technical indicators
        logger.info("Step 3: Computing technical indicators")
        (
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        ) = compute_scores(ticker, stock_prices, spy_prices, options_df)

        # Step 4: Fetch peers and perform ecosystem analysis
        logger.info("Step 4: Performing ecosystem analysis")
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

        # Step 5: Calculate composite scores
        logger.info("Step 5: Calculating composite scores")
        scored_df = _calculate_composite_scores(
            options_df,
            rsi,
            beta,
            momentum,
            avg_iv,
            forecast_confidence,
            data_availability,
        )

        # Step 6: Apply ecosystem adjustment to composite scores
        logger.info("Step 6: Applying ecosystem signal adjustments")
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

        # Step 7: Rank and select top recommendations
        logger.info("Step 7: Ranking and selecting top recommendations")

        # Sort by adjusted composite score (descending) and then by days to expiry (ascending for ties)
        ranked_df = scored_df.sort_values(
            ["AdjustedCompositeScore", "daysToExpiry"], ascending=[False, True]
        )

        # Select top N recommendations
        top_recommendations = ranked_df.head(top_n)

        # Prepare final output DataFrame with required columns
        result_df = top_recommendations[
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
            f"Options analysis completed for {ticker}. "
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
