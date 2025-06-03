from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from buffetbot.utils.errors import (
    DataError,
    DataValidationError,
    ErrorSeverity,
    handle_data_error,
)
from buffetbot.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, "logs/validators.log")


def validate_ticker(ticker: str) -> bool:
    """
    Validate stock ticker symbol.

    Args:
        ticker: Stock ticker symbol

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not isinstance(ticker, str):
            logger.error(f"Invalid ticker type: {type(ticker)}")
            return False

        if not ticker:
            logger.error("Empty ticker symbol")
            return False

        if not ticker.isalnum():
            logger.error(f"Invalid ticker format: {ticker}")
            return False

        if len(ticker) > 10:  # Most tickers are 1-5 characters
            logger.error(f"Ticker too long: {ticker}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating ticker: {str(e)}")
        return False


def validate_date_range(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    max_years: int = 10,
) -> bool:
    """
    Validate date range for data fetching.

    Args:
        start_date: Start date
        end_date: End date
        max_years: Maximum number of years allowed

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=365 * max_years)

        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            logger.error("Invalid date type")
            return False

        if start_date >= end_date:
            logger.error("Start date must be before end date")
            return False

        if (end_date - start_date).days > 365 * max_years:
            logger.error(f"Date range exceeds maximum of {max_years} years")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating date range: {str(e)}")
        return False


def validate_financial_data(
    data: dict[str, pd.DataFrame | dict],
    required_statements: list[str] = ["income_stmt", "balance_sheet", "cash_flow"],
) -> bool:
    """
    Validate financial data structure and content.

    Args:
        data: Dictionary containing financial statements
        required_statements: List of required statement types

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not isinstance(data, dict):
            logger.error("Invalid data type")
            return False

        # Check required statements
        for statement in required_statements:
            if statement not in data:
                logger.error(f"Missing required statement: {statement}")
                return False

            if not isinstance(data[statement], pd.DataFrame):
                logger.error(f"Invalid statement type for {statement}")
                return False

            if data[statement].empty:
                logger.error(f"Empty statement: {statement}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating financial data: {str(e)}")
        return False


def validate_price_data(data: pd.DataFrame) -> list[DataError]:
    """
    Validate price history data for common issues.

    Args:
        data: Price history DataFrame

    Returns:
        List[DataError]: List of validation errors found
    """
    errors = []

    try:
        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            error = DataError(
                code="MISSING_COLUMNS",
                message=f"Missing required columns: {missing_cols}",
                severity=ErrorSeverity.HIGH,
                details={"missing_columns": missing_cols},
            )
            errors.append(error)
            handle_data_error(error, logger)

        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            error = DataError(
                code="MISSING_VALUES",
                message=f"Found missing values in columns: {missing_values[missing_values > 0].to_dict()}",
                severity=ErrorSeverity.MEDIUM,
                details={
                    "missing_counts": missing_values[missing_values > 0].to_dict()
                },
            )
            errors.append(error)
            handle_data_error(error, logger)

        # Check for negative values in price columns
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in data.columns and (data[col] < 0).any():
                error = DataError(
                    code="NEGATIVE_PRICES",
                    message=f"Found negative values in {col}",
                    severity=ErrorSeverity.HIGH,
                    details={"column": col},
                )
                errors.append(error)
                handle_data_error(error, logger)

        # Check for price consistency
        if all(col in data.columns for col in ["High", "Low", "Open", "Close"]):
            invalid_high_low = (data["High"] < data["Low"]).any()
            invalid_open_close = (
                (data["Open"] > data["High"])
                | (data["Open"] < data["Low"])
                | (data["Close"] > data["High"])
                | (data["Close"] < data["Low"])
            ).any()

            if invalid_high_low:
                error = DataError(
                    code="INVALID_HIGH_LOW",
                    message="Found cases where High < Low",
                    severity=ErrorSeverity.HIGH,
                )
                errors.append(error)
                handle_data_error(error, logger)

            if invalid_open_close:
                error = DataError(
                    code="INVALID_OPEN_CLOSE",
                    message="Found cases where Open/Close outside High/Low range",
                    severity=ErrorSeverity.HIGH,
                )
                errors.append(error)
                handle_data_error(error, logger)

        # Check for volume consistency
        if "Volume" in data.columns:
            if (data["Volume"] < 0).any():
                error = DataError(
                    code="NEGATIVE_VOLUME",
                    message="Found negative volume values",
                    severity=ErrorSeverity.HIGH,
                )
                errors.append(error)
                handle_data_error(error, logger)

        return errors

    except Exception as e:
        error = DataError(
            code="VALIDATION_ERROR",
            message=f"Error during price data validation: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
        )
        handle_data_error(error, logger)
        raise DataValidationError(error)


def validate_analysis_parameters(
    params: dict[str, Any],
    required_params: list[str] = ["growth_rate", "discount_rate", "forecast_years"],
) -> bool:
    """
    Validate analysis parameters.

    Args:
        params: Dictionary containing analysis parameters
        required_params: List of required parameters

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not isinstance(params, dict):
            logger.error("Invalid parameters type")
            return False

        # Check required parameters
        for param in required_params:
            if param not in params:
                logger.error(f"Missing required parameter: {param}")
                return False

        # Validate growth rate
        if "growth_rate" in params:
            growth_rate = params["growth_rate"]
            if not isinstance(growth_rate, (int, float)):
                logger.error("Invalid growth rate type")
                return False
            if growth_rate < -1 or growth_rate > 1:
                logger.error("Growth rate must be between -1 and 1")
                return False

        # Validate discount rate
        if "discount_rate" in params:
            discount_rate = params["discount_rate"]
            if not isinstance(discount_rate, (int, float)):
                logger.error("Invalid discount rate type")
                return False
            if discount_rate <= 0 or discount_rate > 1:
                logger.error("Discount rate must be between 0 and 1")
                return False

        # Validate forecast years
        if "forecast_years" in params:
            forecast_years = params["forecast_years"]
            if not isinstance(forecast_years, int):
                logger.error("Invalid forecast years type")
                return False
            if forecast_years < 1 or forecast_years > 20:
                logger.error("Forecast years must be between 1 and 20")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating analysis parameters: {str(e)}")
        return False


def validate_recommendation(
    recommendation: dict[str, Any],
    required_fields: list[str] = ["score", "level", "explanation"],
) -> bool:
    """
    Validate recommendation structure and content.

    Args:
        recommendation: Dictionary containing recommendation data
        required_fields: List of required fields

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not isinstance(recommendation, dict):
            logger.error("Invalid recommendation type")
            return False

        # Check required fields
        for field in required_fields:
            if field not in recommendation:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate score
        if "score" in recommendation:
            score = recommendation["score"]
            if not isinstance(score, (int, float)):
                logger.error("Invalid score type")
                return False
            if score < 0 or score > 100:
                logger.error("Score must be between 0 and 100")
                return False

        # Validate level
        if "level" in recommendation:
            level = recommendation["level"]
            if not isinstance(level, str):
                logger.error("Invalid level type")
                return False
            valid_levels = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
            if level not in valid_levels:
                logger.error(f"Invalid level: {level}")
                return False

        # Validate explanation
        if "explanation" in recommendation:
            explanation = recommendation["explanation"]
            if not isinstance(explanation, str):
                logger.error("Invalid explanation type")
                return False
            if not explanation:
                logger.error("Empty explanation")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating recommendation: {str(e)}")
        return False
