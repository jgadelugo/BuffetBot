import re
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from buffetbot.utils.errors import (
    DataCleaningError,
    DataError,
    ErrorSeverity,
    handle_data_error,
)
from buffetbot.utils.logger import setup_logger
from buffetbot.utils.validators import validate_financial_data, validate_price_data

# Initialize logger
logger = setup_logger(__name__, "logs/data_cleaner.log")


def _parse_market_cap(value: str | float | None) -> float | None:
    """
    Parse market cap string (e.g., '1.2B', '500M') to float value in millions.

    Args:
        value: Market cap value as string or float

    Returns:
        float: Market cap in millions, or None if parsing fails
    """
    if pd.isna(value) or value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value) / 1e6  # Convert to millions

    try:
        # Remove any currency symbols and whitespace
        value = str(value).strip().replace("$", "").replace(",", "")

        # Extract number and multiplier
        match = re.match(r"([\d.]+)([KMBT]?)", value.upper())
        if not match:
            error = DataError(
                code="INVALID_MARKET_CAP",
                message=f"Invalid market cap format: {value}",
                severity=ErrorSeverity.LOW,
            )
            handle_data_error(error, logger)
            return None

        number, multiplier = match.groups()
        number = float(number)

        # Convert to millions based on multiplier
        multipliers = {"K": 1e-3, "M": 1, "B": 1e3, "T": 1e6}
        return number * multipliers.get(multiplier, 1)

    except (ValueError, TypeError) as e:
        error = DataError(
            code="MARKET_CAP_PARSE_ERROR",
            message=f"Failed to parse market cap value: {value}",
            severity=ErrorSeverity.MEDIUM,
            details={"error": str(e)},
        )
        handle_data_error(error, logger)
        return None


def clean_price_history(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare historical price data for analysis.

    Args:
        data: Raw price history DataFrame from fetcher

    Returns:
        pd.DataFrame: Cleaned price history with:
            - Datetime index
            - Sorted by date
            - Forward-filled missing values
            - Validated numeric columns

    Raises:
        DataCleaningError: If there's an error cleaning the data
    """
    try:
        logger.info("Starting price history cleaning")

        # Create a copy to avoid modifying original
        df = data.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Converting index to datetime")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                error = DataError(
                    code="DATETIME_CONVERSION_ERROR",
                    message="Failed to convert index to datetime",
                    severity=ErrorSeverity.HIGH,
                    details={"error": str(e)},
                )
                handle_data_error(error, logger)
                raise DataCleaningError(error)

        # Sort by date
        df = df.sort_index()

        # Check for missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            error = DataError(
                code="MISSING_VALUES",
                message=f"Found {missing_before} missing values in price history",
                severity=ErrorSeverity.MEDIUM,
                details={"missing_count": missing_before},
            )
            handle_data_error(error, logger)

            # Forward fill missing values for OHLC
            price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
            df[price_cols] = df[price_cols].fillna(method="ffill")

            # Fill remaining missing values with 0 for Volume
            df["Volume"] = df["Volume"].fillna(0)

            missing_after = df.isnull().sum().sum()
            logger.info(f"Filled {missing_before - missing_after} missing values")

        # Validate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                error = DataError(
                    code="NEGATIVE_VALUES",
                    message=f"Found negative values in {col}",
                    severity=ErrorSeverity.MEDIUM,
                    details={"column": col},
                )
                handle_data_error(error, logger)
                df[col] = df[col].abs()

        logger.info("Successfully cleaned price history")
        return df

    except Exception as e:
        error = DataError(
            code="PRICE_CLEANING_ERROR",
            message=f"Error cleaning price history: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
        )
        handle_data_error(error, logger)
        raise DataCleaningError(error)


def clean_fundamentals(
    data: dict[str, float | str | None]
) -> dict[str, float | str | None]:
    """
    Clean and validate fundamental financial metrics.

    Args:
        data: Raw fundamentals dictionary from fetcher

    Returns:
        Dict: Cleaned fundamentals with:
            - Consistent numeric types
            - Parsed market cap values
            - Validated metrics
            - Missing values as np.nan
    """
    try:
        logger.info("Starting fundamentals cleaning")

        # Create a copy to avoid modifying original
        cleaned = data.copy()

        # Define metric categories
        numeric_metrics = [
            "pe_ratio",
            "pb_ratio",
            "eps",
            "roe",
            "dividend_yield",
            "beta",
            "total_debt",
            "total_equity",
            "current_assets",
            "current_liabilities",
            "interest_expense",
            "ebit",
            "gross_profit",
            "operating_income",
            "net_income",
            "revenue",
        ]

        string_metrics = ["sector", "industry"]

        # Check for missing required metrics
        required_metrics = [
            "pe_ratio",
            "pb_ratio",
            "eps",
            "roe",
            "market_cap",
            "total_debt",
            "total_equity",
            "current_assets",
            "current_liabilities",
            "net_income",
            "revenue",
        ]
        missing_metrics = [
            metric for metric in required_metrics if metric not in cleaned
        ]
        if missing_metrics:
            logger.warning(f"Missing required metrics: {missing_metrics}")

        # Clean numeric values
        for key in numeric_metrics:
            if key in cleaned:
                value = cleaned[key]
                if pd.isna(value) or value is None:
                    cleaned[key] = np.nan
                else:
                    try:
                        cleaned[key] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid numeric value for {key}: {value}")
                        cleaned[key] = np.nan

        # Parse market cap
        if "market_cap" in cleaned:
            cleaned["market_cap"] = _parse_market_cap(cleaned["market_cap"])

        # Clean string values
        for key in string_metrics:
            if key in cleaned:
                value = cleaned[key]
                if pd.isna(value) or value is None:
                    cleaned[key] = ""
                else:
                    cleaned[key] = str(value).strip()

        # Calculate derived metrics if possible
        if all(
            k in cleaned and not pd.isna(cleaned[k])
            for k in ["current_assets", "current_liabilities"]
        ):
            cleaned["current_ratio"] = (
                cleaned["current_assets"] / cleaned["current_liabilities"]
            )

        if all(
            k in cleaned and not pd.isna(cleaned[k])
            for k in ["total_debt", "total_equity"]
        ):
            cleaned["debt_to_equity"] = cleaned["total_debt"] / cleaned["total_equity"]

        if all(
            k in cleaned and not pd.isna(cleaned[k]) for k in ["net_income", "revenue"]
        ):
            cleaned["net_margin"] = cleaned["net_income"] / cleaned["revenue"]

        logger.info("Successfully cleaned fundamentals")
        return cleaned

    except Exception as e:
        logger.error(f"Error cleaning fundamentals: {str(e)}")
        return {}


def clean_analyst_predictions(data: dict[str, float | str | None]) -> pd.DataFrame:
    """
    Clean and prepare analyst predictions for analysis.

    Args:
        data: Raw analyst predictions dictionary from fetcher

    Returns:
        pd.DataFrame: Cleaned predictions with:
            - Normalized column names
            - Consistent data types
            - Dropped unnecessary columns
    """
    try:
        logger.info("Starting analyst predictions cleaning")

        # Convert dict to DataFrame
        df = pd.DataFrame([data])

        # Normalize column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Drop unnecessary columns if present
        cols_to_drop = ["currency", "exchange", "quote_type"]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Clean numeric columns
        numeric_cols = [
            "target_price",
            "num_analysts",
            "strong_buy",
            "buy",
            "hold",
            "sell",
            "strong_sell",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean recommendation
        if "recommendation" in df.columns:
            df["recommendation"] = df["recommendation"].fillna("unknown")
            df["recommendation"] = df["recommendation"].str.lower()

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(
                f"Missing values in predictions: {missing[missing > 0].to_dict()}"
            )

        logger.info("Successfully cleaned analyst predictions")
        return df

    except Exception as e:
        logger.error(f"Error cleaning analyst predictions: {str(e)}")
        return pd.DataFrame()


def clean_financial_statement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean financial statement data by handling missing values and formatting issues.

    Args:
        df: DataFrame containing financial statement data

    Returns:
        Cleaned DataFrame

    Note:
        - Handles timestamp objects properly
        - Converts string values to numeric where possible
        - Fills missing values with 0
        - Removes any non-numeric characters from column names
    """
    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return pd.DataFrame()

        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        # Clean column names
        df_clean.columns = [str(col).strip() for col in df_clean.columns]

        # Handle timestamp objects in index
        if isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = df_clean.index.strftime("%Y-%m-%d")
        else:
            df_clean.index = [
                str(idx).strip() if isinstance(idx, str) else idx
                for idx in df_clean.index
            ]

        # Convert string values to numeric where possible
        for col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {str(e)}")

        # Fill missing values with 0
        df_clean = df_clean.fillna(0)

        return df_clean

    except Exception as e:
        logger.error(f"Error cleaning financial statement: {str(e)}")
        return pd.DataFrame()


def clean_financial_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Clean and standardize financial data.

    Args:
        data: Dictionary containing financial statements DataFrames

    Returns:
        Dictionary containing cleaned DataFrames
    """
    try:
        logger.info("Starting financial data cleaning")
        cleaned_data = {}

        # Set pandas option to suppress downcasting warnings
        pd.set_option("future.no_silent_downcasting", True)

        for statement_name, df in data.items():
            if df is not None and not df.empty:
                try:
                    # Create a copy to avoid modifying the original
                    df_clean = df.copy()

                    # Handle index conversion
                    if not isinstance(df_clean.index, pd.DatetimeIndex):
                        try:
                            # First try to convert directly
                            df_clean.index = pd.to_datetime(df_clean.index)
                        except Exception as e:
                            # If direct conversion fails, try to extract dates from strings
                            try:
                                # Look for date patterns in the index
                                date_patterns = [
                                    r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
                                    r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
                                    r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY
                                    r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
                                ]

                                # Try to find dates in the index strings
                                dates = []
                                for idx in df_clean.index:
                                    date_found = False
                                    for pattern in date_patterns:
                                        match = re.search(pattern, str(idx))
                                        if match:
                                            try:
                                                dates.append(
                                                    pd.to_datetime(match.group())
                                                )
                                                date_found = True
                                                break
                                            except:
                                                continue
                                    if not date_found:
                                        # If no date found, use a default date
                                        dates.append(pd.Timestamp("1970-01-01"))

                                df_clean.index = pd.DatetimeIndex(dates)
                                logger.info(
                                    f"Successfully converted {statement_name} index to datetime using pattern matching"
                                )
                            except Exception as e2:
                                # If all conversion attempts fail, keep original index
                                logger.warning(
                                    f"Could not convert {statement_name} index to datetime: {str(e2)}"
                                )
                                # Add a warning column to indicate potential issues
                                df_clean["_index_conversion_warning"] = True

                    # Fill missing values with 0
                    df_clean = df_clean.fillna(0)

                    # Convert numeric columns to float
                    numeric_cols = df_clean.select_dtypes(
                        include=["int64", "float64"]
                    ).columns
                    df_clean[numeric_cols] = df_clean[numeric_cols].astype(float)

                    # Ensure all required columns exist
                    required_columns = {
                        "income_stmt": [
                            "Total Revenue",
                            "Gross Profit",
                            "Operating Income",
                            "Net Income",
                        ],
                        "balance_sheet": [
                            "Total Assets",
                            "Total Current Assets",
                            "Total Liabilities",
                            "Total Current Liabilities",
                            "Total Stockholder Equity",
                        ],
                        "cash_flow": [
                            "Operating Cash Flow",
                            "Capital Expenditure",
                            "Free Cash Flow",
                        ],
                    }

                    if statement_name in required_columns:
                        for col in required_columns[statement_name]:
                            if col not in df_clean.columns:
                                logger.warning(
                                    f"Missing required column {col} in {statement_name}"
                                )
                                df_clean[col] = 0

                    cleaned_data[statement_name] = df_clean
                    logger.info(f"Successfully cleaned {statement_name}")
                except Exception as e:
                    error = DataError(
                        code="STATEMENT_CLEANING_ERROR",
                        message=f"Error cleaning {statement_name}: {str(e)}",
                        severity=ErrorSeverity.HIGH,
                        details={"statement": statement_name, "error": str(e)},
                    )
                    handle_data_error(error, logger)
                    cleaned_data[statement_name] = df
            else:
                logger.warning(f"{statement_name} is empty or None")
                cleaned_data[statement_name] = df

        logger.info("Completed financial data cleaning")
        return cleaned_data

    except Exception as e:
        error = DataError(
            code="FINANCIAL_CLEANING_ERROR",
            message=f"Error in financial data cleaning: {str(e)}",
            severity=ErrorSeverity.CRITICAL,
        )
        handle_data_error(error, logger)
        return data


def _calculate_metrics(
    price_data: pd.DataFrame,
    fundamentals: dict,
    income_stmt: pd.DataFrame | None,
    balance_sheet: pd.DataFrame | None,
    cash_flow: pd.DataFrame | None,
) -> dict:
    """
    Calculate additional financial metrics.

    Args:
        price_data: Cleaned price history
        fundamentals: Cleaned fundamental metrics
        income_stmt: Cleaned income statement
        balance_sheet: Cleaned balance sheet
        cash_flow: Cleaned cash flow statement

    Returns:
        Dict containing calculated metrics:
            - Price metrics (latest, change, volatility)
            - Technical indicators (RSI, momentum)
            - Financial ratios (if data available)
    """
    try:
        metrics = {
            "latest_price": price_data["Close"].iloc[-1],
            "price_change": (price_data["Close"].iloc[-1] / price_data["Close"].iloc[0])
            - 1,
            "volatility": price_data["Close"].pct_change().std() * np.sqrt(252),
            "rsi": calculate_rsi(price_data["Close"]),
            "momentum": (price_data["Close"].iloc[-1] / price_data["Close"].iloc[-20])
            - 1,
        }

        # Calculate additional financial ratios if data is available
        if not income_stmt.empty and not balance_sheet.empty:
            # Get latest values
            latest_income = income_stmt.iloc[0]
            latest_balance = balance_sheet.iloc[0]

            # Calculate ratios
            if "total_revenue" in latest_income and "net_income" in latest_income:
                metrics["net_margin"] = (
                    latest_income["net_income"] / latest_income["total_revenue"]
                )

            if "total_assets" in latest_balance and "net_income" in latest_income:
                metrics["roa"] = (
                    latest_income["net_income"] / latest_balance["total_assets"]
                )

            if "total_equity" in latest_balance and "net_income" in latest_income:
                metrics["roe"] = (
                    latest_income["net_income"] / latest_balance["total_equity"]
                )

            if (
                "total_current_assets" in latest_balance
                and "total_current_liabilities" in latest_balance
            ):
                metrics["current_ratio"] = (
                    latest_balance["total_current_assets"]
                    / latest_balance["total_current_liabilities"]
                )

            if "total_debt" in latest_balance and "total_equity" in latest_balance:
                metrics["debt_to_equity"] = (
                    latest_balance["total_debt"] / latest_balance["total_equity"]
                )

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        float: RSI value
    """
    try:
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return 50.0  # Return neutral RSI on error
