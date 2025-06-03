"""
Options data fetcher module for long-dated call options.

This module provides functionality to fetch and process options data,
specifically focusing on long-dated call options with customizable
minimum days to expiry filtering.

All functions implement robust fault handling - if real data is unavailable, the
functions return empty structures with clear metadata flags, ensuring
the advisory pipeline continues without breaking.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict, Union

import numpy as np
import pandas as pd
import yfinance as yf

from utils.errors import DataError, DataFetcherError, ErrorSeverity, handle_data_error
from utils.logger import setup_logger
from utils.validators import validate_ticker

# Initialize logger
logger = setup_logger(__name__, "logs/options_fetcher.log")


class OptionsDataError(Exception):
    """Custom exception for options data fetching errors."""

    def __init__(self, message: str, error_code: str = "OPTIONS_ERROR"):
        self.error_code = error_code
        super().__init__(message)


class OptionsResult(TypedDict):
    """Type definition for options fetcher result structure."""

    data: pd.DataFrame
    data_available: bool
    error_message: str | None
    ticker: str
    min_days_to_expiry: int
    total_expiry_dates: int | None
    valid_chains_processed: int | None


def _calculate_days_to_expiry(expiry_date: str) -> int:
    """
    Calculate the number of days from today to expiry date.

    Args:
        expiry_date: Expiry date string in YYYY-MM-DD format

    Returns:
        int: Number of days to expiry

    Raises:
        ValueError: If date format is invalid

    Examples:
        >>> _calculate_days_to_expiry('2024-12-20')
        45
        >>> _calculate_days_to_expiry('invalid-date')
        Traceback (most recent call last):
        ...
        ValueError: Invalid expiry date format: invalid-date
    """
    try:
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        today = datetime.now()
        return (expiry - today).days
    except ValueError as e:
        logger.error(f"Invalid date format {expiry_date}: {str(e)}")
        raise ValueError(f"Invalid expiry date format: {expiry_date}")


def _validate_options_data(df: pd.DataFrame) -> list[str]:
    """
    Validate options DataFrame for required columns and data quality.

    Args:
        df: Options DataFrame to validate

    Returns:
        List[str]: List of validation errors (empty if valid)

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'strike': [100, 110], 'lastPrice': [5.0, 3.0],
        ...                   'impliedVolatility': [0.2, 0.25], 'volume': [100, 50],
        ...                   'openInterest': [500, 300]})
        >>> _validate_options_data(df)
        []
        >>> empty_df = pd.DataFrame()
        >>> _validate_options_data(empty_df)
        ['Options data is empty']
    """
    errors = []

    required_columns = [
        "strike",
        "lastPrice",
        "impliedVolatility",
        "volume",
        "openInterest",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if df.empty:
        errors.append("Options data is empty")

    # Check for valid numeric data
    if not df.empty:
        numeric_columns = ["strike", "lastPrice", "impliedVolatility"]
        for col in numeric_columns:
            if col in df.columns and df[col].isna().all():
                errors.append(f"All values in column '{col}' are NaN")

    return errors


def _process_options_chain(
    options_chain, expiry_date: str, min_days_to_expiry: int
) -> pd.DataFrame | None:
    """
    Process a single options chain for a specific expiry date with robust error handling.

    Args:
        options_chain: Options chain object from yfinance
        expiry_date: Expiry date string
        min_days_to_expiry: Minimum days to expiry filter

    Returns:
        Optional[pd.DataFrame]: Processed options data or None if filtered out or error occurred

    Examples:
        >>> # This function is typically called internally by fetch_long_dated_calls
        >>> # It processes raw yfinance options chain data into structured DataFrame
        >>> # Returns None if chain is invalid or doesn't meet minimum days criteria
    """
    try:
        days_to_expiry = _calculate_days_to_expiry(expiry_date)

        # Filter by minimum days to expiry
        if days_to_expiry < min_days_to_expiry:
            logger.debug(
                f"Skipping expiry {expiry_date} ({days_to_expiry} days < {min_days_to_expiry} minimum)"
            )
            return None

        # Get calls data
        if not hasattr(options_chain, "calls") or options_chain.calls.empty:
            logger.warning(f"No call options data available for expiry {expiry_date}")
            return None

        calls_df = options_chain.calls.copy()
        calls_df["expiry"] = expiry_date
        calls_df["daysToExpiry"] = days_to_expiry

        # Ensure required columns exist, fill missing with appropriate defaults
        if "delta" not in calls_df.columns:
            calls_df["delta"] = np.nan
        if "ask" not in calls_df.columns:
            calls_df["ask"] = np.nan
        if "bid" not in calls_df.columns:
            calls_df["bid"] = np.nan

        # Select and reorder columns
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

        # Only keep columns that exist in the DataFrame
        available_columns = [col for col in columns_to_keep if col in calls_df.columns]
        calls_df = calls_df[available_columns]

        logger.debug(f"Processed {len(calls_df)} call options for expiry {expiry_date}")
        return calls_df

    except Exception as e:
        logger.error(f"Error processing options chain for {expiry_date}: {str(e)}")
        return None


def fetch_long_dated_calls(ticker: str, min_days_to_expiry: int = 180) -> OptionsResult:
    """
    Fetch long-dated call options for a given ticker symbol with robust error handling.

    This function retrieves call options data for the specified ticker,
    filtering for options with at least the minimum days to expiry.
    The results are sorted by expiry date and strike price. If real data
    is unavailable, it returns an empty DataFrame with clear error information.
    The pipeline will continue without breaking.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        min_days_to_expiry: Minimum number of days to expiry to include
                           in results (default: 180 days)

    Returns:
        OptionsResult: Dictionary containing:
            - data (pd.DataFrame): DataFrame with long-dated call options or empty DataFrame
            - data_available (bool): True if real data was fetched successfully
            - error_message (Optional[str]): Error description if data_available=False
            - ticker (str): The requested ticker symbol
            - min_days_to_expiry (int): The applied minimum days filter
            - total_expiry_dates (Optional[int]): Number of expiry dates found
            - valid_chains_processed (Optional[int]): Number of valid chains processed

        DataFrame columns (when data_available=True):
            - expiry: Option expiry date (YYYY-MM-DD)
            - strike: Strike price
            - lastPrice: Last traded price
            - impliedVolatility: Implied volatility
            - volume: Trading volume
            - openInterest: Open interest
            - delta: Option delta (if available)
            - ask: Ask price (if available)
            - bid: Bid price (if available)
            - daysToExpiry: Number of days to expiry

    Examples:
        >>> # Fetch call options for AAPL with at least 6 months to expiry
        >>> result = fetch_long_dated_calls('AAPL', min_days_to_expiry=180)
        >>> if result['data_available']:
        ...     calls_df = result['data']
        ...     print(f"Found {len(calls_df)} long-dated call options")
        ... else:
        ...     print(f"Options unavailable: {result['error_message']}")

        >>> # Fetch call options with at least 1 year to expiry
        >>> result = fetch_long_dated_calls('MSFT', min_days_to_expiry=365)
        >>> if result['data_available'] and not result['data'].empty:
        ...     print(f"Processing {len(result['data'])} long-term options")

    Note:
        This function never raises exceptions - it always returns a valid
        OptionsResult structure. Check the 'data_available' flag to determine
        if real data was successfully fetched.
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
    }

    try:
        # Input validation
        if not ticker or not isinstance(ticker, str):
            error_msg = "Ticker must be a non-empty string"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

        if min_days_to_expiry < 0:
            error_msg = "min_days_to_expiry must be non-negative"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

        # Validate ticker format
        try:
            validate_ticker(ticker)
        except Exception as e:
            error_msg = f"Invalid ticker format '{ticker}': {str(e)}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

        ticker = ticker.upper().strip()
        result["ticker"] = ticker  # Update with normalized ticker

        logger.info(
            f"Fetching long-dated call options for {ticker} with min {min_days_to_expiry} days to expiry"
        )

        try:
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
                return result

            if not expiry_dates:
                error_msg = f"No options data available for ticker {ticker}"
                logger.warning(error_msg)
                result["error_message"] = error_msg
                return result

            result["total_expiry_dates"] = len(expiry_dates)
            logger.info(f"Found {len(expiry_dates)} expiry dates for {ticker}")

            # Process each expiry date
            all_calls_data = []
            valid_chains_count = 0

            for expiry_date in expiry_dates:
                try:
                    # Get options chain for this expiry
                    options_chain = stock.option_chain(expiry_date)

                    # Process the chain
                    processed_chain = _process_options_chain(
                        options_chain, expiry_date, min_days_to_expiry
                    )

                    if processed_chain is not None and not processed_chain.empty:
                        all_calls_data.append(processed_chain)
                        valid_chains_count += 1

                except Exception as e:
                    logger.warning(
                        f"Error processing expiry {expiry_date} for {ticker}: {str(e)}"
                    )
                    continue

            result["valid_chains_processed"] = valid_chains_count

            # Combine all data
            if not all_calls_data:
                error_msg = f"No long-dated call options found for {ticker} with min {min_days_to_expiry} days to expiry"
                logger.warning(error_msg)
                result["error_message"] = error_msg
                return result  # Return empty DataFrame with error

            combined_df = pd.concat(all_calls_data, ignore_index=True)

            # Validate the combined data
            validation_errors = _validate_options_data(combined_df)
            if validation_errors:
                logger.warning(
                    f"Data validation warnings for {ticker}: {validation_errors}"
                )

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

            logger.info(
                f"Successfully fetched {len(combined_df)} long-dated call options for {ticker} "
                f"from {valid_chains_count} expiry dates"
            )

            return result

        except Exception as e:
            error_msg = f"Unexpected error fetching options data for {ticker}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            return result

    except Exception as e:
        # Catch-all for any unexpected errors in validation or setup
        error_msg = f"Critical error in options fetcher for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
        return result


def get_options_summary(options_result: OptionsResult) -> dict | None:
    """
    Generate a summary of options data with robust error handling.

    Args:
        options_result: OptionsResult structure from fetch_long_dated_calls

    Returns:
        Optional[dict]: Summary statistics about the options data, or None if unavailable

    Examples:
        >>> result = fetch_long_dated_calls('AAPL')
        >>> summary = get_options_summary(result)
        >>> if summary:
        ...     print(f"Found {summary['total_options']} options")
        ... else:
        ...     print("Options summary unavailable")

        >>> # Example summary structure:
        >>> {
        ...     'total_options': 245,
        ...     'expiry_dates': ['2024-12-20', '2025-01-17', '2025-02-21'],
        ...     'strike_range': {'min': 150.0, 'max': 250.0},
        ...     'avg_implied_vol': 0.285,
        ...     'total_volume': 15432,
        ...     'total_open_interest': 89765
        ... }
    """
    try:
        if not options_result["data_available"]:
            logger.warning(
                f"Cannot generate options summary: {options_result['error_message']}"
            )
            return None

        options_df = options_result["data"]

        if options_df.empty:
            return {
                "total_options": 0,
                "expiry_dates": [],
                "strike_range": None,
                "avg_implied_vol": None,
                "total_volume": 0,
                "total_open_interest": 0,
            }

        return {
            "total_options": len(options_df),
            "expiry_dates": sorted(options_df["expiry"].unique().tolist()),
            "strike_range": {
                "min": float(options_df["strike"].min()),
                "max": float(options_df["strike"].max()),
            },
            "avg_implied_vol": float(options_df["impliedVolatility"].mean())
            if "impliedVolatility" in options_df.columns
            and not options_df["impliedVolatility"].isna().all()
            else None,
            "total_volume": int(options_df["volume"].sum())
            if "volume" in options_df.columns and not options_df["volume"].isna().all()
            else 0,
            "total_open_interest": int(options_df["openInterest"].sum())
            if "openInterest" in options_df.columns
            and not options_df["openInterest"].isna().all()
            else 0,
        }

    except Exception as e:
        logger.error(
            f"Unexpected error generating options summary: {str(e)}", exc_info=True
        )
        return None
