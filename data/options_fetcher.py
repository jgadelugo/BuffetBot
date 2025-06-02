"""
Options data fetcher module for long-dated call options.

This module provides functionality to fetch and process options data,
specifically focusing on long-dated call options with customizable
minimum days to expiry filtering.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union

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

    pass


def _calculate_days_to_expiry(expiry_date: str) -> int:
    """
    Calculate the number of days from today to expiry date.

    Args:
        expiry_date: Expiry date string in YYYY-MM-DD format

    Returns:
        int: Number of days to expiry

    Raises:
        ValueError: If date format is invalid
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
    Process a single options chain for a specific expiry date.

    Args:
        options_chain: Options chain object from yfinance
        expiry_date: Expiry date string
        min_days_to_expiry: Minimum days to expiry filter

    Returns:
        Optional[pd.DataFrame]: Processed options data or None if filtered out
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


def fetch_long_dated_calls(ticker: str, min_days_to_expiry: int = 180) -> pd.DataFrame:
    """
    Fetch long-dated call options for a given ticker symbol.

    This function retrieves call options data for the specified ticker,
    filtering for options with at least the minimum days to expiry.
    The results are sorted by expiry date and strike price.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        min_days_to_expiry: Minimum number of days to expiry to include
                           in results (default: 180 days)

    Returns:
        pd.DataFrame: DataFrame containing long-dated call options with columns:
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

    Raises:
        OptionsDataError: If ticker is invalid, no options data is available,
                         or there's an error fetching the data
        ValueError: If min_days_to_expiry is negative

    Examples:
        >>> # Fetch call options for AAPL with at least 6 months to expiry
        >>> calls_df = fetch_long_dated_calls('AAPL', min_days_to_expiry=180)
        >>> print(f"Found {len(calls_df)} long-dated call options")

        >>> # Fetch call options with at least 1 year to expiry
        >>> long_calls = fetch_long_dated_calls('MSFT', min_days_to_expiry=365)
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        raise OptionsDataError("Ticker must be a non-empty string")

    if min_days_to_expiry < 0:
        raise ValueError("min_days_to_expiry must be non-negative")

    # Validate ticker format
    try:
        validate_ticker(ticker)
    except Exception as e:
        raise OptionsDataError(f"Invalid ticker format '{ticker}': {str(e)}")

    ticker = ticker.upper().strip()
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
            raise OptionsDataError(error_msg)

        if not expiry_dates:
            error_msg = f"No options data available for ticker {ticker}"
            logger.warning(error_msg)
            raise OptionsDataError(error_msg)

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

        # Combine all data
        if not all_calls_data:
            error_msg = f"No long-dated call options found for {ticker} with min {min_days_to_expiry} days to expiry"
            logger.warning(error_msg)
            return pd.DataFrame()  # Return empty DataFrame instead of raising error

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

        logger.info(
            f"Successfully fetched {len(combined_df)} long-dated call options for {ticker} "
            f"from {valid_chains_count} expiry dates"
        )

        return combined_df

    except OptionsDataError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        error_msg = f"Unexpected error fetching options data for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise OptionsDataError(error_msg)


def get_options_summary(options_df: pd.DataFrame) -> dict:
    """
    Generate a summary of options data.

    Args:
        options_df: DataFrame containing options data

    Returns:
        dict: Summary statistics about the options data
    """
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
        else None,
        "total_volume": int(options_df["volume"].sum())
        if "volume" in options_df.columns
        else 0,
        "total_open_interest": int(options_df["openInterest"].sum())
        if "openInterest" in options_df.columns
        else 0,
    }
