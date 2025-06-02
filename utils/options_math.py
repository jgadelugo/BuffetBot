"""
Options Math Module

This module provides reusable, well-documented functions to calculate technical indicators
for use in options analysis and screening. All functions are designed to be robust,
testable, and extensible.

Author: BuffetBot Financial Analysis System
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from .errors import DataError, DataValidationError, ErrorSeverity
from .logger import get_logger

# Module logger
logger = get_logger(__name__)


class OptionsMathError(Exception):
    """Custom exception for options math calculation errors."""

    pass


def _validate_series(
    data: pd.Series, name: str, min_length: int = 1, allow_nan: bool = False
) -> None:
    """
    Validate pandas Series input data.

    Args:
        data: The pandas Series to validate
        name: Name of the parameter for error messages
        min_length: Minimum required length of the series
        allow_nan: Whether to allow NaN values in the series

    Raises:
        OptionsMathError: If validation fails
    """
    if not isinstance(data, pd.Series):
        raise OptionsMathError(f"{name} must be a pandas Series, got {type(data)}")

    if len(data) < min_length:
        raise OptionsMathError(
            f"{name} must have at least {min_length} data points, got {len(data)}"
        )

    if not allow_nan and data.isna().any():
        raise OptionsMathError(f"{name} contains NaN values")

    if len(data.dropna()) == 0:
        raise OptionsMathError(f"{name} contains no valid (non-NaN) data")


def _validate_dataframe(
    data: pd.DataFrame,
    name: str,
    required_columns: list | None = None,
    min_rows: int = 1,
) -> None:
    """
    Validate pandas DataFrame input data.

    Args:
        data: The pandas DataFrame to validate
        name: Name of the parameter for error messages
        required_columns: List of required column names
        min_rows: Minimum required number of rows

    Raises:
        OptionsMathError: If validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise OptionsMathError(f"{name} must be a pandas DataFrame, got {type(data)}")

    if len(data) < min_rows:
        raise OptionsMathError(
            f"{name} must have at least {min_rows} rows, got {len(data)}"
        )

    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise OptionsMathError(
                f"{name} missing required columns: {missing_cols}. "
                f"Available columns: {list(data.columns)}"
            )


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate the Relative Strength Index (RSI) for a price series.

    The RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    RSI values range from 0 to 100, with values above 70 typically considered overbought
    and values below 30 considered oversold.

    Args:
        prices: Series of price data (typically closing prices)
        period: Number of periods to use for RSI calculation (default: 14)

    Returns:
        float: Current RSI value (0-100)

    Raises:
        OptionsMathError: If input validation fails or calculation cannot be performed

    Examples:
        >>> import pandas as pd
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        >>> rsi = calculate_rsi(prices, period=5)
        >>> print(f"RSI: {rsi:.2f}")
        RSI: 66.67

        >>> # With longer period
        >>> rsi_14 = calculate_rsi(prices, period=14)  # Uses default period
    """
    logger.info(f"Calculating RSI with period={period} for {len(prices)} price points")

    # Input validation
    _validate_series(prices, "prices", min_length=period + 1, allow_nan=True)

    if period <= 0:
        raise OptionsMathError(f"Period must be positive, got {period}")

    try:
        # Remove NaN values and ensure we have enough data
        clean_prices = prices.dropna()
        if len(clean_prices) < period + 1:
            raise OptionsMathError(
                f"Not enough valid data points for RSI calculation. "
                f"Need at least {period + 1}, got {len(clean_prices)}"
            )

        # Calculate price changes
        price_changes = clean_prices.diff().dropna()

        # Separate gains and losses
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)

        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=period).mean()
        avg_losses = losses.ewm(span=period).mean()

        # Calculate relative strength and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Get the most recent RSI value
        current_rsi = float(rsi.iloc[-1])

        logger.info(f"RSI calculation completed: {current_rsi:.2f}")
        return current_rsi

    except Exception as e:
        error_msg = f"Failed to calculate RSI: {str(e)}"
        logger.error(error_msg)
        raise OptionsMathError(error_msg) from e


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate the beta coefficient of a stock relative to the market.

    Beta measures the volatility of a stock relative to the overall market.
    A beta of 1.0 indicates the stock moves with the market, >1.0 indicates
    higher volatility, and <1.0 indicates lower volatility.

    Args:
        stock_returns: Series of stock return percentages
        market_returns: Series of market/benchmark return percentages

    Returns:
        float: Beta coefficient

    Raises:
        OptionsMathError: If input validation fails or calculation cannot be performed

    Examples:
        >>> import pandas as pd
        >>> stock_rets = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.01])
        >>> market_rets = pd.Series([0.015, -0.005, 0.02, -0.015, 0.005, 0.03, -0.005])
        >>> beta = calculate_beta(stock_rets, market_rets)
        >>> print(f"Beta: {beta:.3f}")
        Beta: 1.234
    """
    logger.info(
        f"Calculating beta for {len(stock_returns)} stock returns vs "
        f"{len(market_returns)} market returns"
    )

    # Input validation
    _validate_series(stock_returns, "stock_returns", min_length=2, allow_nan=True)
    _validate_series(market_returns, "market_returns", min_length=2, allow_nan=True)

    if len(stock_returns) != len(market_returns):
        raise OptionsMathError(
            f"Stock returns and market returns must have same length. "
            f"Got {len(stock_returns)} and {len(market_returns)}"
        )

    try:
        # Create aligned DataFrame and remove NaN values
        combined = pd.DataFrame(
            {"stock": stock_returns, "market": market_returns}
        ).dropna()

        if len(combined) < 2:
            raise OptionsMathError(
                "Not enough valid data points for beta calculation after removing NaNs"
            )

        # Calculate covariance and variance
        covariance = combined["stock"].cov(combined["market"])
        market_variance = combined["market"].var()

        if market_variance == 0:
            raise OptionsMathError(
                "Market returns have zero variance - cannot calculate beta"
            )

        # Calculate beta
        beta = covariance / market_variance

        logger.info(
            f"Beta calculation completed: {beta:.4f} "
            f"(covariance: {covariance:.6f}, market_var: {market_variance:.6f})"
        )
        return float(beta)

    except Exception as e:
        error_msg = f"Failed to calculate beta: {str(e)}"
        logger.error(error_msg)
        raise OptionsMathError(error_msg) from e


def calculate_momentum(prices: pd.Series, window: int = 20) -> float:
    """
    Calculate price momentum as the percentage change over a specified window.

    Momentum measures the rate of change in price over a given period.
    Positive momentum indicates upward price movement, while negative momentum
    indicates downward movement.

    Args:
        prices: Series of price data (typically closing prices)
        window: Number of periods to look back for momentum calculation (default: 20)

    Returns:
        float: Momentum as percentage change (e.g., 0.15 for 15% gain)

    Raises:
        OptionsMathError: If input validation fails or calculation cannot be performed

    Examples:
        >>> import pandas as pd
        >>> prices = pd.Series([100, 102, 105, 103, 108, 110, 112, 115, 118, 120])
        >>> momentum = calculate_momentum(prices, window=5)
        >>> print(f"5-period momentum: {momentum:.2%}")
        5-period momentum: 11.11%

        >>> # Using default 20-period window
        >>> long_prices = pd.Series(range(100, 125))  # 25 data points
        >>> momentum_20 = calculate_momentum(long_prices)
        >>> print(f"20-period momentum: {momentum_20:.2%}")
        20-period momentum: 24.00%
    """
    logger.info(
        f"Calculating momentum with window={window} for {len(prices)} price points"
    )

    # Input validation
    _validate_series(prices, "prices", min_length=window + 1, allow_nan=True)

    if window <= 0:
        raise OptionsMathError(f"Window must be positive, got {window}")

    try:
        # Remove NaN values and ensure we have enough data
        clean_prices = prices.dropna()
        if len(clean_prices) < window + 1:
            raise OptionsMathError(
                f"Not enough valid data points for momentum calculation. "
                f"Need at least {window + 1}, got {len(clean_prices)}"
            )

        # Calculate momentum as percentage change
        current_price = clean_prices.iloc[-1]
        past_price = clean_prices.iloc[-(window + 1)]

        if past_price == 0:
            raise OptionsMathError("Cannot calculate momentum: past price is zero")

        momentum = (current_price - past_price) / past_price

        logger.info(
            f"Momentum calculation completed: {momentum:.4f} "
            f"(current: {current_price}, past: {past_price})"
        )
        return float(momentum)

    except Exception as e:
        error_msg = f"Failed to calculate momentum: {str(e)}"
        logger.error(error_msg)
        raise OptionsMathError(error_msg) from e


def calculate_average_iv(option_data: pd.DataFrame) -> float:
    """
    Calculate the average implied volatility from options data.

    This function computes a volume-weighted average of implied volatility
    across all options in the dataset, giving more weight to options with
    higher trading volumes.

    Args:
        option_data: DataFrame containing options data with columns:
                    - 'impliedVolatility': Implied volatility values
                    - 'volume': Trading volume (optional, for weighting)

    Returns:
        float: Average implied volatility as a decimal (e.g., 0.25 for 25%)

    Raises:
        OptionsMathError: If input validation fails or calculation cannot be performed

    Examples:
        >>> import pandas as pd
        >>> options_df = pd.DataFrame({
        ...     'impliedVolatility': [0.20, 0.25, 0.30, 0.22, 0.28],
        ...     'volume': [100, 200, 150, 300, 80]
        ... })
        >>> avg_iv = calculate_average_iv(options_df)
        >>> print(f"Average IV: {avg_iv:.2%}")
        Average IV: 24.21%

        >>> # Without volume data (simple average)
        >>> simple_df = pd.DataFrame({'impliedVolatility': [0.20, 0.25, 0.30]})
        >>> simple_avg = calculate_average_iv(simple_df)
        >>> print(f"Simple average IV: {simple_avg:.2%}")
        Simple average IV: 25.00%
    """
    logger.info(f"Calculating average IV for {len(option_data)} options")

    # Input validation
    _validate_dataframe(
        option_data, "option_data", required_columns=["impliedVolatility"], min_rows=1
    )

    try:
        # Extract implied volatility data
        iv_data = option_data["impliedVolatility"].dropna()

        if len(iv_data) == 0:
            raise OptionsMathError("No valid implied volatility data found")

        # Check for valid IV values (should be positive)
        if (iv_data < 0).any():
            logger.warning(
                "Found negative implied volatility values, filtering them out"
            )
            iv_data = iv_data[iv_data >= 0]

        if len(iv_data) == 0:
            raise OptionsMathError("No positive implied volatility values found")

        # Calculate weighted average if volume data is available
        if "volume" in option_data.columns:
            volume_data = option_data.loc[iv_data.index, "volume"].fillna(0)

            # Filter out zero volume entries for weighting
            valid_volume = volume_data > 0
            if valid_volume.any():
                iv_weighted = iv_data[valid_volume]
                weights = volume_data[valid_volume]
                avg_iv = float(np.average(iv_weighted, weights=weights))
                logger.info(f"Volume-weighted average IV calculated: {avg_iv:.4f}")
            else:
                # Fall back to simple average if no volume data
                avg_iv = float(iv_data.mean())
                logger.info(
                    f"Simple average IV calculated (no volume data): {avg_iv:.4f}"
                )
        else:
            # Simple average when no volume column
            avg_iv = float(iv_data.mean())
            logger.info(f"Simple average IV calculated: {avg_iv:.4f}")

        return avg_iv

    except Exception as e:
        error_msg = f"Failed to calculate average IV: {str(e)}"
        logger.error(error_msg)
        raise OptionsMathError(error_msg) from e


def _calculate_rolling_statistic(
    data: pd.Series, window: int, statistic: str = "mean"
) -> pd.Series:
    """
    Helper function to calculate rolling statistics with proper error handling.

    Args:
        data: Input data series
        window: Rolling window size
        statistic: Type of statistic ('mean', 'std', 'var', 'min', 'max')

    Returns:
        pd.Series: Rolling statistic series

    Raises:
        OptionsMathError: If invalid statistic type or calculation fails
    """
    valid_statistics = ["mean", "std", "var", "min", "max"]
    if statistic not in valid_statistics:
        raise OptionsMathError(
            f"Invalid statistic '{statistic}'. Must be one of: {valid_statistics}"
        )

    try:
        rolling_obj = data.rolling(window=window, min_periods=window)
        return getattr(rolling_obj, statistic)()
    except Exception as e:
        raise OptionsMathError(
            f"Failed to calculate rolling {statistic}: {str(e)}"
        ) from e


def validate_options_math_inputs(
    prices: pd.Series | None = None,
    returns: pd.Series | None = None,
    option_data: pd.DataFrame | None = None,
    periods: list | None = None,
) -> dict:
    """
    Comprehensive validation function for options math inputs.

    This helper function provides centralized validation for common input types
    used across multiple functions in this module.

    Args:
        prices: Price series to validate
        returns: Returns series to validate
        option_data: Options DataFrame to validate
        periods: List of period values to validate

    Returns:
        dict: Validation results and cleaned data

    Raises:
        OptionsMathError: If any validation fails
    """
    results = {}

    if prices is not None:
        _validate_series(prices, "prices", allow_nan=True)
        results["prices_valid_count"] = len(prices.dropna())

    if returns is not None:
        _validate_series(returns, "returns", allow_nan=True)
        results["returns_valid_count"] = len(returns.dropna())

    if option_data is not None:
        _validate_dataframe(option_data, "option_data")
        results["option_data_rows"] = len(option_data)

    if periods is not None:
        if not isinstance(periods, (list, tuple)):
            raise OptionsMathError("Periods must be a list or tuple")
        if any(p <= 0 for p in periods):
            raise OptionsMathError("All periods must be positive")
        results["periods"] = periods

    return results
