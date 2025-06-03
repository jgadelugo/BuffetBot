"""
Main data fetcher module.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd
import yfinance as yf

from data.fetcher.utils.financial_calculations import calculate_rsi
from data.fetcher.utils.standardization import standardize_financial_data
from utils.errors import DataError, DataFetcherError, ErrorSeverity, handle_data_error
from utils.logger import setup_logger
from utils.validators import (
    validate_date_range,
    validate_financial_data,
    validate_price_data,
    validate_ticker,
)

# Initialize logger
logger = setup_logger(__name__, "logs/data_fetcher.log")


class DataFetcher:
    """
    A class to fetch financial data from various sources.
    Currently implements yfinance as the data source.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the data fetcher.

        Args:
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Base delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = None

    def _handle_rate_limit(self, attempt: int) -> None:
        """
        Handle rate limiting with exponential backoff.

        Args:
            attempt: Current attempt number
        """
        delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
        logger.warning(f"Rate limit hit, waiting {delay:.2f} seconds")
        time.sleep(delay)

    def fetch_price_history(
        self, ticker: str, period: str = "5y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (e.g., "5y", "1y", "6mo")
            interval: Data interval (e.g., "1d", "1h", "15m")

        Returns:
            pd.DataFrame: Historical price data

        Raises:
            DataFetcherError: If there's an error fetching the data
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching price history for {ticker} (attempt {attempt + 1}/{self.max_retries})"
                )

                # Fetch data
                stock = yf.Ticker(ticker)
                data = stock.history(period=period, interval=interval)

                if data.empty:
                    error = DataError(
                        code="NO_PRICE_DATA",
                        message=f"No price data found for {ticker}",
                        severity=ErrorSeverity.HIGH,
                    )
                    handle_data_error(error, logger)
                    raise DataFetcherError(error)

                # Validate data
                validation_errors = validate_price_data(data)
                if validation_errors:
                    error = DataError(
                        code="PRICE_VALIDATION_ERROR",
                        message=f"Price data validation failed for {ticker}",
                        severity=ErrorSeverity.HIGH,
                        details={
                            "validation_errors": [
                                e.to_dict() for e in validation_errors
                            ]
                        },
                    )
                    handle_data_error(error, logger)
                    raise DataFetcherError(error)

                logger.info(
                    f"Successfully fetched {len(data)} price records for {ticker}"
                )
                return data

            except Exception as e:
                if "rate limit" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        self._handle_rate_limit(attempt)
                        continue

                error = DataError(
                    code="PRICE_FETCH_ERROR",
                    message=f"Error fetching price data for {ticker}: {str(e)}",
                    severity=ErrorSeverity.CRITICAL,
                )
                handle_data_error(error, logger)
                raise DataFetcherError(error)

    def fetch_fundamentals(self, ticker: str) -> dict[str, float | str | None]:
        """
        Fetch fundamental data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict[str, Union[float, str, None]]: Dictionary of fundamental metrics

        Raises:
            DataFetcherError: If there's an error fetching the data
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching fundamentals for {ticker} (attempt {attempt + 1}/{self.max_retries})"
                )

                # Fetch data
                stock = yf.Ticker(ticker)
                info = stock.info

                if not info:
                    error = DataError(
                        code="NO_FUNDAMENTAL_DATA",
                        message=f"No fundamental data found for {ticker}",
                        severity=ErrorSeverity.HIGH,
                    )
                    handle_data_error(error, logger)
                    raise DataFetcherError(error)

                # Extract key metrics
                fundamentals = {
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "eps": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                }

                # Validate data
                validation_errors = validate_financial_data(fundamentals)
                if validation_errors:
                    error = DataError(
                        code="FUNDAMENTAL_VALIDATION_ERROR",
                        message=f"Fundamental data validation failed for {ticker}",
                        severity=ErrorSeverity.HIGH,
                        details={
                            "validation_errors": [
                                e.to_dict() for e in validation_errors
                            ]
                        },
                    )
                    handle_data_error(error, logger)
                    raise DataFetcherError(error)

                logger.info(f"Successfully fetched fundamental data for {ticker}")

                if hasattr(info, "sharesOutstanding"):
                    fundamentals["shares_outstanding"] = getattr(
                        info, "sharesOutstanding", None
                    )
                elif hasattr(info, "impliedSharesOutstanding"):
                    fundamentals["shares_outstanding"] = getattr(
                        info, "impliedSharesOutstanding", None
                    )

                fundamentals["latest_price"] = (
                    getattr(info, "regularMarketPrice", None)
                    or getattr(info, "currentPrice", None)
                    or getattr(info, "previousClose", None)
                )

                return fundamentals

            except Exception as e:
                if "rate limit" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        self._handle_rate_limit(attempt)
                        continue

                error = DataError(
                    code="FUNDAMENTAL_FETCH_ERROR",
                    message=f"Error fetching fundamental data for {ticker}: {str(e)}",
                    severity=ErrorSeverity.CRITICAL,
                )
                handle_data_error(error, logger)
                raise DataFetcherError(error)

    def fetch_financial_data(self, ticker: str) -> dict[str, pd.DataFrame | dict]:
        """
        Fetch financial data for a given ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing:
                - income: Income statement DataFrame
                - balance: Balance sheet DataFrame
                - cash_flow: Cash flow statement DataFrame
                - fundamentals: Dictionary of fundamental metrics
        """
        try:
            logger.info(f"Fetching financial data for {ticker}")

            # Fetch raw data
            stock = yf.Ticker(ticker)
            logger.info(f"Successfully created yfinance Ticker object for {ticker}")

            # Fetch financial statements with validation
            logger.info("Fetching income statement...")
            income_stmt = stock.income_stmt
            logger.info(
                f"Income statement type: {type(income_stmt)}, empty: {income_stmt is None or (hasattr(income_stmt, 'empty') and income_stmt.empty)}"
            )
            if income_stmt is not None and not income_stmt.empty:
                logger.info(f"Income statement columns: {income_stmt.columns.tolist()}")
                # Ensure dates are in index
                if isinstance(income_stmt.columns[0], pd.Timestamp):
                    logger.info("Transposing income statement to have dates as index")
                    income_stmt = income_stmt.T
            else:
                logger.warning(f"No income statement data found for {ticker}")
                income_stmt = pd.DataFrame()

            logger.info("Fetching balance sheet...")
            balance_sheet = stock.balance_sheet
            logger.info(
                f"Balance sheet type: {type(balance_sheet)}, empty: {balance_sheet is None or (hasattr(balance_sheet, 'empty') and balance_sheet.empty)}"
            )
            if balance_sheet is not None and not balance_sheet.empty:
                logger.info(f"Balance sheet columns: {balance_sheet.columns.tolist()}")
                # Ensure dates are in index
                if isinstance(balance_sheet.columns[0], pd.Timestamp):
                    logger.info("Transposing balance sheet to have dates as index")
                    balance_sheet = balance_sheet.T
            else:
                logger.warning(f"No balance sheet data found for {ticker}")
                balance_sheet = pd.DataFrame()

            logger.info("Fetching cash flow statement...")
            cash_flow = stock.cashflow
            logger.info(
                f"Cash flow type: {type(cash_flow)}, empty: {cash_flow is None or (hasattr(cash_flow, 'empty') and cash_flow.empty)}"
            )
            if cash_flow is not None and not cash_flow.empty:
                logger.info(f"Cash flow columns: {cash_flow.columns.tolist()}")
                # Ensure dates are in index
                if isinstance(cash_flow.columns[0], pd.Timestamp):
                    logger.info(
                        "Transposing cash flow statement to have dates as index"
                    )
                    cash_flow = cash_flow.T
            else:
                logger.warning(f"No cash flow data found for {ticker}")
                cash_flow = pd.DataFrame()

            # Try to fetch fundamentals
            try:
                logger.info("Fetching fundamentals...")
                fundamentals = self.fetch_fundamentals(ticker)
                logger.info(
                    f"Successfully fetched fundamentals: {list(fundamentals.keys())}"
                )
            except Exception as e:
                logger.warning(f"Error fetching fundamentals for {ticker}: {str(e)}")
                fundamentals = {}

            # Prepare data for standardization
            data_to_standardize = {
                "income": income_stmt,
                "balance": balance_sheet,
                "cash_flow": cash_flow,
            }
            logger.info("Preparing to standardize financial data...")
            for stmt_type, df in data_to_standardize.items():
                if df is not None and not df.empty:
                    logger.info(f"{stmt_type} statement shape: {df.shape}")
                    logger.info(f"{stmt_type} statement index type: {type(df.index)}")
                    logger.info(
                        f"{stmt_type} statement columns type: {[type(col) for col in df.columns]}"
                    )

            # Standardize the data
            standardized_data = standardize_financial_data(data_to_standardize)
            logger.info("Financial data standardization completed")

            # Add fundamentals to the result
            standardized_data["fundamentals"] = fundamentals

            return standardized_data

        except Exception as e:
            error = DataError(
                code="FINANCIAL_FETCH_ERROR",
                message=f"Error fetching financial data for {ticker}: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
            )
            handle_data_error(error, logger)
            logger.error(f"Stack trace:", exc_info=True)

            # Return empty DataFrames with the correct structure
            return {
                "income": pd.DataFrame(),
                "balance": pd.DataFrame(),
                "cash_flow": pd.DataFrame(),
                "fundamentals": {},
            }
