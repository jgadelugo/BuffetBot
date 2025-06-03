"""
Data validation utilities for the data fetcher.
"""

from typing import Any, Dict, List, Union

import pandas as pd

from .errors import DataError, ErrorSeverity


def validate_ticker(ticker: str) -> list[DataError]:
    """
    Validate a stock ticker.

    Args:
        ticker: Stock ticker to validate

    Returns:
        List of validation errors
    """
    errors = []

    if not ticker:
        errors.append(
            DataError(
                code="INVALID_TICKER",
                message="Ticker cannot be empty",
                severity=ErrorSeverity.HIGH,
            )
        )
    elif not isinstance(ticker, str):
        errors.append(
            DataError(
                code="INVALID_TICKER_TYPE",
                message="Ticker must be a string",
                severity=ErrorSeverity.HIGH,
            )
        )
    elif len(ticker) > 10:  # Most tickers are 1-5 characters
        errors.append(
            DataError(
                code="INVALID_TICKER_LENGTH",
                message="Ticker length exceeds maximum allowed",
                severity=ErrorSeverity.MEDIUM,
            )
        )

    return errors


def validate_date_range(start_date: str, end_date: str) -> list[DataError]:
    """
    Validate date range parameters.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        List of validation errors
    """
    errors = []

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if start > end:
            errors.append(
                DataError(
                    code="INVALID_DATE_RANGE",
                    message="Start date must be before end date",
                    severity=ErrorSeverity.HIGH,
                )
            )
    except Exception as e:
        errors.append(
            DataError(
                code="INVALID_DATE_FORMAT",
                message=f"Invalid date format: {str(e)}",
                severity=ErrorSeverity.HIGH,
            )
        )

    return errors


def validate_price_data(data: pd.DataFrame) -> list[DataError]:
    """
    Validate price data DataFrame.

    Args:
        data: Price data DataFrame to validate

    Returns:
        List of validation errors
    """
    errors = []

    if data is None or data.empty:
        errors.append(
            DataError(
                code="EMPTY_PRICE_DATA",
                message="Price data is empty",
                severity=ErrorSeverity.HIGH,
            )
        )
        return errors

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        errors.append(
            DataError(
                code="MISSING_PRICE_COLUMNS",
                message=f"Missing required columns: {', '.join(missing_columns)}",
                severity=ErrorSeverity.HIGH,
            )
        )

    # Check for negative values
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_columns:
        if col in data.columns and (data[col] < 0).any():
            errors.append(
                DataError(
                    code="NEGATIVE_PRICE_VALUES",
                    message=f"Found negative values in {col} column",
                    severity=ErrorSeverity.HIGH,
                )
            )

    return errors


def validate_financial_data(data: dict[str, Any]) -> list[DataError]:
    """
    Validate financial data dictionary.

    Args:
        data: Financial data dictionary to validate

    Returns:
        List of validation errors
    """
    errors = []

    if not data:
        errors.append(
            DataError(
                code="EMPTY_FINANCIAL_DATA",
                message="Financial data is empty",
                severity=ErrorSeverity.HIGH,
            )
        )
        return errors

    # Check for required fields
    required_fields = ["market_cap", "pe_ratio", "eps"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        errors.append(
            DataError(
                code="MISSING_FINANCIAL_FIELDS",
                message=f"Missing required fields: {', '.join(missing_fields)}",
                severity=ErrorSeverity.HIGH,
            )
        )

    # Validate numeric fields
    numeric_fields = ["market_cap", "pe_ratio", "eps", "dividend_yield", "beta"]
    for field in numeric_fields:
        if field in data and data[field] is not None:
            try:
                float(data[field])
            except (ValueError, TypeError):
                errors.append(
                    DataError(
                        code="INVALID_NUMERIC_FIELD",
                        message=f"Invalid numeric value for {field}",
                        severity=ErrorSeverity.HIGH,
                    )
                )

    return errors
