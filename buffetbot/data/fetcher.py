import difflib
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

from buffetbot.utils.errors import (
    DataError,
    DataFetcherError,
    ErrorSeverity,
    handle_data_error,
)
from buffetbot.utils.logger import setup_logger
from buffetbot.utils.validators import (
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
                metrics = {
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "eps": info.get("trailingEps"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                }

                # Validate data
                validation_errors = validate_financial_data(metrics)
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
                return metrics

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

    def fetch_analyst_predictions(self, symbol: str) -> dict[str, float | str]:
        """
        Fetch analyst predictions and recommendations for a given stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dict containing analyst predictions:
                - Target Price
                - Recommendation
                - Number of Analysts
                - Strong Buy/Buy/Hold/Sell/Strong Sell counts
        """
        try:
            self.logger.info(f"Fetching analyst predictions for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            predictions = {
                "target_price": info.get("targetMeanPrice", None),
                "recommendation": info.get("recommendationKey", None),
                "num_analysts": info.get("numberOfAnalystOpinions", None),
                "strong_buy": info.get("strongBuy", None),
                "buy": info.get("buy", None),
                "hold": info.get("hold", None),
                "sell": info.get("sell", None),
                "strong_sell": info.get("strongSell", None),
            }

            self.logger.info(f"Successfully fetched analyst predictions for {symbol}")
            return predictions

        except Exception as e:
            self.logger.error(
                f"Error fetching analyst predictions for {symbol}: {str(e)}"
            )
            return {}


def fetch_stock_data(ticker: str, years: int = 5) -> dict[str, pd.DataFrame | dict]:
    """
    Fetch historical stock data and financial statements.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch

    Returns:
        Dict containing:
            - price_data: DataFrame with historical price data
            - income_stmt: DataFrame with income statement data
            - balance_sheet: DataFrame with balance sheet data
            - cash_flow: DataFrame with cash flow statement data
            - fundamentals: Dict with key financial metrics
            - metrics: Dict with calculated metrics

    Raises:
        ValueError: If ticker is invalid or no data is found
        Exception: For other errors during data fetching
    """
    try:
        # Validate inputs
        if not validate_ticker(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        logger.info(f"Fetching data for {ticker}")

        # Initialize yfinance Ticker object
        stock = yf.Ticker(ticker)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        if not validate_date_range(start_date, end_date, years):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")

        # Fetch historical price data
        logger.info(f"Fetching {years} years of price data for {ticker}")
        price_data = stock.history(start=start_date, end=end_date)

        if price_data.empty:
            raise ValueError(f"No price data found for {ticker}")

        # Handle missing values
        price_data = price_data.ffill()  # Forward fill missing values
        price_data = price_data.bfill()  # Backward fill any remaining missing values

        logger.info(f"Successfully fetched {len(price_data)} days of data for {ticker}")

        # Fetch financial statements
        logger.info(f"Fetching financial statements for {ticker}")

        # Income Statement
        income_stmt = stock.income_stmt
        if income_stmt is not None:
            income_stmt = _standardize_column_names(income_stmt, "income")
            income_stmt = income_stmt.fillna(0)  # Fill missing values with 0

        # Balance Sheet
        balance_sheet = stock.balance_sheet
        if balance_sheet is not None:
            balance_sheet = _standardize_column_names(balance_sheet, "balance")
            balance_sheet = balance_sheet.fillna(0)  # Fill missing values with 0

        # Cash Flow Statement
        cash_flow = stock.cashflow
        if cash_flow is not None:
            cash_flow = _standardize_column_names(cash_flow, "cash_flow")
            cash_flow = cash_flow.fillna(0)  # Fill missing values with 0

        # Fetch fundamental data
        logger.info(f"Fetching fundamental data for {ticker}")
        stock_info = stock.info  # Get stock info once

        if not stock_info:
            logger.warning(f"No fundamental data found for {ticker}")
            stock_info = {}

        fundamentals = {
            "pe_ratio": stock_info.get("trailingPE", None),
            "pb_ratio": stock_info.get("priceToBook", None),
            "eps": stock_info.get("trailingEps", None),
            "roe": stock_info.get("returnOnEquity", None),
            "market_cap": stock_info.get("marketCap", None),
            "dividend_yield": stock_info.get("dividendYield", None),
            "beta": stock_info.get("beta", None),
            "sector": stock_info.get("sector", None),
            "industry": stock_info.get("industry", None),
            "total_debt": stock_info.get("totalDebt", None),
            "total_equity": stock_info.get("totalStockholderEquity", None),
            "current_assets": stock_info.get("totalCurrentAssets", None),
            "current_liabilities": stock_info.get("totalCurrentLiabilities", None),
            "interest_expense": stock_info.get("interestExpense", None),
            "ebit": stock_info.get("ebit", None),
            "gross_profit": stock_info.get("grossProfit", None),
            "operating_income": stock_info.get("operatingIncome", None),
            "net_income": stock_info.get("netIncome", None),
            "revenue": stock_info.get("totalRevenue", None),
        }

        # Calculate additional metrics
        metrics = {
            "latest_price": price_data["Close"].iloc[-1],
            "price_change": (price_data["Close"].iloc[-1] / price_data["Close"].iloc[0])
            - 1,
            "volatility": price_data["Close"].pct_change().std() * np.sqrt(252),
            "rsi": calculate_rsi(price_data["Close"]),
            "momentum": (price_data["Close"].iloc[-1] / price_data["Close"].iloc[-20])
            - 1,
        }

        return {
            "price_data": price_data,
            "income_stmt": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "fundamentals": fundamentals,
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        return None


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


def fetch_fundamentals(ticker: str) -> dict[str, float | str | int]:
    """
    Fetch fundamental data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Dict[str, Union[float, str, int]]: Dictionary containing fundamental data:
            - Market Cap: Company's market capitalization
            - P/E Ratio: Price to Earnings ratio
            - P/B Ratio: Price to Book ratio
            - Dividend Yield: Annual dividend yield
            - Beta: Stock's beta coefficient
            - 52 Week High: 52-week high price
            - 52 Week Low: 52-week low price
            - Sector: Company's sector
            - Industry: Company's industry

    Raises:
        ValueError: If ticker is invalid
        Exception: For other errors during data fetching
    """
    try:
        logger.info(f"Fetching fundamental data for {ticker}")

        # Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker symbol")

        # Fetch data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract relevant metrics
        fundamentals = {
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "pb_ratio": info.get("priceToBook", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", 0),
            "week_high_52": info.get("fiftyTwoWeekHigh", 0),
            "week_low_52": info.get("fiftyTwoWeekLow", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }

        logger.info(f"Successfully fetched fundamental data for {ticker}")
        return fundamentals

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
        raise


def fetch_income_statement(ticker: str) -> pd.DataFrame:
    """
    Fetch income statement data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
        pd.DataFrame: DataFrame containing income statement data with columns:
            - Revenue: Total revenue
            - Gross Profit: Gross profit
            - Operating Income: Operating income
            - Net Income: Net income
            - EPS: Earnings per share

    Raises:
        ValueError: If ticker is invalid
        Exception: For other errors during data fetching
    """
    try:
        logger.info(f"Fetching income statement for {ticker}")

        # Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker symbol")

        # Fetch data
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt

        if income_stmt.empty:
            logger.warning(f"No income statement data found for {ticker}")
            return pd.DataFrame()

        # Clean and format the data
        income_stmt = income_stmt.fillna(0)
        income_stmt = income_stmt.round(2)

        logger.info(f"Successfully fetched income statement for {ticker}")
        return income_stmt

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching income statement for {ticker}: {str(e)}")
        raise


def fetch_balance_sheet(ticker: str) -> pd.DataFrame:
    """
    Fetch balance sheet data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
        pd.DataFrame: DataFrame containing balance sheet data with columns:
            - Total Assets: Total assets
            - Total Liabilities: Total liabilities
            - Total Equity: Total equity
            - Current Assets: Current assets
            - Current Liabilities: Current liabilities
            - Long Term Debt: Long term debt

    Raises:
        ValueError: If ticker is invalid
        Exception: For other errors during data fetching
    """
    try:
        logger.info(f"Fetching balance sheet for {ticker}")

        # Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker symbol")

        # Fetch data
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet

        if balance_sheet.empty:
            logger.warning(f"No balance sheet data found for {ticker}")
            return pd.DataFrame()

        # Clean and format the data
        balance_sheet = balance_sheet.fillna(0)
        balance_sheet = balance_sheet.round(2)

        logger.info(f"Successfully fetched balance sheet for {ticker}")
        return balance_sheet

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching balance sheet for {ticker}: {str(e)}")
        raise


def fetch_cash_flow(ticker: str) -> pd.DataFrame:
    """
    Fetch cash flow statement data for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
        pd.DataFrame: DataFrame containing cash flow data with columns:
            - Operating Cash Flow: Cash from operations
            - Investing Cash Flow: Cash from investing
            - Financing Cash Flow: Cash from financing
            - Free Cash Flow: Free cash flow
            - Capital Expenditure: Capital expenditure

    Raises:
        ValueError: If ticker is invalid
        Exception: For other errors during data fetching
    """
    try:
        logger.info(f"Fetching cash flow statement for {ticker}")

        # Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker symbol")

        # Fetch data
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow

        if cash_flow.empty:
            logger.warning(f"No cash flow data found for {ticker}")
            return pd.DataFrame()

        # Clean and format the data
        cash_flow = cash_flow.fillna(0)
        cash_flow = cash_flow.round(2)

        logger.info(f"Successfully fetched cash flow statement for {ticker}")
        return cash_flow

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching cash flow statement for {ticker}: {str(e)}")
        raise


def _standardize_column_names(df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """
    Standardize column names across different financial statements.

    Args:
        df: DataFrame to standardize
        statement_type: Type of financial statement ('income', 'balance', 'cash_flow')

    Returns:
        DataFrame with standardized column names
    """
    if df is None or df.empty:
        logger.warning(f"Empty DataFrame provided for {statement_type} statement")
        return pd.DataFrame()

    # Create a copy to avoid modifying the original
    df_std = df.copy()

    # Log original columns for debugging
    logger.info(
        f"Original columns in {statement_type} statement: {list(df_std.columns)}"
    )
    logger.info(
        f"Sample data from {statement_type} statement (first 3 rows):\n{df_std.head(3).to_string()}"
    )

    # Define column name mappings for each statement type
    column_mappings = {
        "income": {
            "Total Revenue": [
                "Revenue",
                "Total Revenue",
                "Sales",
                "Net Sales",
                "Total Sales",
                "Net Revenue",
                "Operating Revenue",
                "Total Operating Revenue",
                "Revenue From Contract With Customer",
                "Revenue From Contract With Customer Excluding Assessed Tax",
                "Operating Revenue",
                "Total Revenue",
                "Net Revenue",
                "Operating Revenue",
                "Revenue From Contract With Customer",
                "Revenue From Contract With Customer Excluding Assessed Tax",
                "Total Revenue",
                "Sales",
                "Net Sales",
                "Total Sales",
                "Net Revenue",
                "Operating Revenue",
                "Total Operating Revenue",
                "Net Revenue",
                "Total Revenue",
                # New/expanded
                "Operating Revenue",
                "Operating Revenues",
                "Total Revenues",
                "Total Net Revenue",
                "Total Net Revenues",
                "Total Sales Revenue",
                "Total Net Sales",
                "Total Net Sales Revenue",
                "Total Net Sales Revenues",
                "Operating Income Revenue",
                "Operating Income Revenues",
                "Operating Revenue Net",
                "Operating Revenue Gross",
                "Operating Revenue Total",
                "Operating Revenue (Net)",
                "Operating Revenue (Gross)",
                "Operating Revenue (Total)",
            ],
            "Gross Profit": [
                "Gross Profit",
                "Gross Income",
                "Gross Margin",
                "Gross Earnings",
                "Gross Profit Loss",
                "Gross Profit Margin",
                "Gross Profit From Revenue",
                "Gross Profit",
                "Gross Income",
                "Gross Margin",
                "Gross Earnings",
                "Gross Profit",
                "Gross Income",
                "Gross Margin",
                "Gross Earnings",
                "Gross Profit",
                "Gross Income",
                "Gross Margin",
                "Gross Earnings",
                # New/expanded
                "Gross Profit (Loss)",
                "Gross Profit (Income)",
                "Gross Profit (Earnings)",
                "Gross Profit (Margin)",
                "Gross Profit (Gross)",
                "Gross Profit (Net)",
                "Gross Profit (Total)",
                "Gross Profit (Revenue)",
                "Gross Profit (Sales)",
                "Gross Profit (Operating)",
                "Gross Profit (Operating Income)",
                "Gross Profit (Operating Earnings)",
                "Gross Profit (Operating Margin)",
            ],
            "Operating Income": [
                "Operating Income",
                "Operating Profit",
                "EBIT",
                "Operating Earnings",
                "Income from Operations",
                "Total Operating Income As Reported",
                "Operating Income Loss",
                "Operating Income Before Depreciation",
                "Operating Income Before Depreciation And Amortization",
                "Total Operating Income As Reported",
                "Operating Income Loss",
                "Operating Income",
                "Operating Profit",
                "EBIT",
                "Operating Earnings",
                "Operating Income",
                "Operating Profit",
                "EBIT",
                "Operating Earnings",
                "Income from Operations",
                "Total Operating Income As Reported",
                # New/expanded
                "EBITDA",
                "Operating Income (Loss)",
                "Operating Income (Profit)",
                "Operating Income (Earnings)",
                "Operating Income (Margin)",
                "Operating Income (Net)",
                "Operating Income (Total)",
                "Operating Income (Revenue)",
                "Operating Income (Sales)",
                "Operating Income (Operating)",
                "Operating Income (Operating Income)",
                "Operating Income (Operating Earnings)",
                "Operating Income (Operating Margin)",
            ],
            "Net Income": [
                "Net Income",
                "Net Earnings",
                "Net Profit",
                "Profit",
                "Net Income Common Stockholders",
                "Net Income to Common",
                "Net Income From Continuing Operation Net Minority Interest",
                "Net Income Continuous Operations",
                "Net Income Including Noncontrolling Interests",
                "Net Income From Continuing And Discontinued Operation",
                "Net Income From Continuing Operations",
                "Net Income From Continuing And Discontinued Operation",
                "Net Income",
                "Net Earnings",
                "Net Profit",
                "Profit",
                "Net Income",
                "Net Earnings",
                "Net Profit",
                "Profit",
                "Net Income Common Stockholders",
                "Net Income to Common",
                # New/expanded
                "Net Income (Loss)",
                "Net Income (Profit)",
                "Net Income (Earnings)",
                "Net Income (Margin)",
                "Net Income (Net)",
                "Net Income (Total)",
                "Net Income (Revenue)",
                "Net Income (Sales)",
                "Net Income (Operating)",
                "Net Income (Operating Income)",
                "Net Income (Operating Earnings)",
                "Net Income (Operating Margin)",
                "Net Income Attributable To Shareholders",
                "Net Income Attributable To Common Shareholders",
                "Net Income Attributable To Parent",
                "Net Income Attributable To Owners Of Parent",
                "Net Income Attributable To Noncontrolling Interest",
                "Net Income Attributable To Noncontrolling Interests",
                "Net Income Attributable To Minority Interest",
                "Net Income Attributable To Minority Interests",
            ],
            "Cost of Revenue": [
                "Cost of Revenue",
                "Cost Of Revenue",
                "Cost of Sales",
                "Cost Of Sales",
                "Reconciled Cost Of Revenue",
                "Cost of Goods Sold",
                "Cost Of Goods Sold",
                "Cost of Revenue",
                "Cost Of Revenue",
                "Cost of Sales",
                "Cost Of Sales",
                "Cost of Revenue",
                "Cost Of Revenue",
                "Cost of Sales",
                "Cost Of Sales",
                "Reconciled Cost Of Revenue",
                "Cost of Goods Sold",
                # New/expanded
                "Cost Of Revenue (Net)",
                "Cost Of Revenue (Gross)",
                "Cost Of Revenue (Total)",
                "Cost Of Revenue (Sales)",
                "Cost Of Revenue (Operating)",
                "Cost Of Revenue (Operating Income)",
                "Cost Of Revenue (Operating Earnings)",
                "Cost Of Revenue (Operating Margin)",
                "Cost Of Revenue (COGS)",
                "COGS",
                "COGS (Cost Of Goods Sold)",
                "COGS (Net)",
                "COGS (Gross)",
                "COGS (Total)",
                "COGS (Sales)",
                "COGS (Operating)",
                "COGS (Operating Income)",
                "COGS (Operating Earnings)",
                "COGS (Operating Margin)",
            ],
            "Operating Expenses": [
                "Operating Expenses",
                "Operating Expense",
                "Total Operating Expenses",
                "Selling General And Administration",
                "Research And Development",
                "Operating Expenses",
                "Operating Expense",
                "Total Operating Expenses",
                "Operating Expenses",
                "Operating Expense",
                "Total Operating Expenses",
                "Selling General And Administration",
                "Research And Development",
                # New/expanded
                "Operating Expenses (Net)",
                "Operating Expenses (Gross)",
                "Operating Expenses (Total)",
                "Operating Expenses (Sales)",
                "Operating Expenses (Operating)",
                "Operating Expenses (Operating Income)",
                "Operating Expenses (Operating Earnings)",
                "Operating Expenses (Operating Margin)",
                "SG&A",
                "SGA",
                "SG and A",
                "SG & A",
                "R&D",
                "RandD",
                "R and D",
                "R & D",
            ],
            "Interest Expense": [
                "Interest Expense",
                "Interest Expenses",
                "Interest Paid",
                "Interest Cost",
                "Interest Charges",
                "Interest Expense (Net)",
                "Interest Expense (Gross)",
                "Interest Expense (Total)",
                "Interest Expense (Operating)",
                "Interest Expense (Operating Income)",
                "Interest Expense (Operating Earnings)",
                "Interest Expense (Operating Margin)",
            ],
            "Tax Expense": [
                "Tax Expense",
                "Income Tax Expense",
                "Provision for Income Taxes",
                "Income Taxes",
                "Tax Expense (Net)",
                "Tax Expense (Gross)",
                "Tax Expense (Total)",
                "Tax Expense (Operating)",
                "Tax Expense (Operating Income)",
                "Tax Expense (Operating Earnings)",
                "Tax Expense (Operating Margin)",
            ],
            "EBITDA": [
                "EBITDA",
                "Earnings Before Interest Taxes Depreciation And Amortization",
                "EBITDA (Net)",
                "EBITDA (Gross)",
                "EBITDA (Total)",
                "EBITDA (Operating)",
                "EBITDA (Operating Income)",
                "EBITDA (Operating Earnings)",
                "EBITDA (Operating Margin)",
            ],
            "EBIT": [
                "EBIT",
                "Earnings Before Interest And Taxes",
                "Operating Income",
                "EBIT (Net)",
                "EBIT (Gross)",
                "EBIT (Total)",
                "EBIT (Operating)",
                "EBIT (Operating Income)",
                "EBIT (Operating Earnings)",
                "EBIT (Operating Margin)",
            ],
            "Research and Development": [
                "Research and Development",
                "R&D",
                "Research And Development",
                "Research & Development",
                "Research and Development Expense",
                "Research And Development Expense",
                "R&D Expense",
                "Research and Development (Net)",
                "Research and Development (Gross)",
                "Research and Development (Total)",
                "Research and Development (Operating)",
            ],
            "SG&A": [
                "SG&A",
                "Selling General and Administrative",
                "Selling, General and Administrative",
                "Selling General And Administrative",
                "Selling, General & Administrative",
                "SG&A Expense",
                "Selling General and Administrative Expense",
                "SG&A (Net)",
                "SG&A (Gross)",
                "SG&A (Total)",
                "SG&A (Operating)",
            ],
            "Non-Operating Income": [
                "Non-Operating Income",
                "Non Operating Income",
                "Other Income",
                "Non-Operating Income (Net)",
                "Non-Operating Income (Gross)",
                "Non-Operating Income (Total)",
                "Non-Operating Income (Operating)",
            ],
            "Extraordinary Items": [
                "Extraordinary Items",
                "Extraordinary Income",
                "Extraordinary Expenses",
                "Extraordinary Items (Net)",
                "Extraordinary Items (Gross)",
                "Extraordinary Items (Total)",
                "Extraordinary Items (Operating)",
            ],
        },
        "balance": {
            "Total Assets": [
                "Total Assets",
                "Assets",
                "Total Assets And Liabilities",
                "Total Assets & Liabilities",
                "Total Assets Net Minority Interest",
                "Total Assets",
                "Assets",
                "Total Assets And Liabilities",
                "Total Assets",
                "Assets",
                "Total Assets And Liabilities",
                "Total Assets",
                "Assets",
                "Total Assets And Liabilities",
                # New/expanded
                "Total Asset",
                "Total Asset Net Minority Interest",
                "Total Asset (Net)",
                "Total Asset (Gross)",
                "Total Asset (Total)",
                "Total Asset (Current)",
                "Total Asset (Noncurrent)",
                "Total Asset (Short Term)",
                "Total Asset (Long Term)",
                "Total Asset (Operating)",
                "Total Asset (Operating Income)",
                "Total Asset (Operating Earnings)",
                "Total Asset (Operating Margin)",
            ],
            "Total Current Assets": [
                "Current Assets",
                "Total Current Assets",
                "Current Assets Total",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Accounts Receivable",
                "Inventory",
                "Other Current Assets",
                "Total Current Assets Net Minority Interest",
                "Cash And Cash Equivalents",
                "Accounts Receivable",
                "Inventory",
                "Other Current Assets",
                "Current Assets",
                "Total Current Assets",
                "Current Assets Total",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                # New/expanded
                "Current Asset",
                "Current Asset Total",
                "Current Asset (Net)",
                "Current Asset (Gross)",
                "Current Asset (Total)",
                "Current Asset (Short Term)",
                "Current Asset (Long Term)",
                "Current Asset (Operating)",
                "Current Asset (Operating Income)",
                "Current Asset (Operating Earnings)",
                "Current Asset (Operating Margin)",
            ],
            "Total Liabilities": [
                "Total Liabilities",
                "Liabilities",
                "Total Liabilities And Equity",
                "Total Liabilities & Equity",
                "Total Liabilities Net Minority Interest",
                "Total Liabilities And Stockholders Equity",
                "Total Liabilities",
                "Liabilities",
                "Total Liabilities And Equity",
                "Total Liabilities",
                "Liabilities",
                "Total Liabilities And Equity",
                "Total Liabilities",
                "Liabilities",
                "Total Liabilities And Equity",
                # New/expanded
                "Total Liability",
                "Total Liability Net Minority Interest",
                "Total Liability (Net)",
                "Total Liability (Gross)",
                "Total Liability (Total)",
                "Total Liability (Current)",
                "Total Liability (Noncurrent)",
                "Total Liability (Short Term)",
                "Total Liability (Long Term)",
                "Total Liability (Operating)",
                "Total Liability (Operating Income)",
                "Total Liability (Operating Earnings)",
                "Total Liability (Operating Margin)",
            ],
            "Total Current Liabilities": [
                "Current Liabilities",
                "Total Current Liabilities",
                "Current Liabilities Total",
                "Accounts Payable",
                "Other Current Liabilities",
                "Current Debt",
                "Total Current Liabilities Net Minority Interest",
                "Accounts Payable",
                "Other Current Liabilities",
                "Current Debt",
                "Current Liabilities",
                "Total Current Liabilities",
                "Current Liabilities Total",
                "Accounts Payable",
                "Other Current Liabilities",
                "Current Debt",
                # New/expanded
                "Current Liability",
                "Current Liability Total",
                "Current Liability (Net)",
                "Current Liability (Gross)",
                "Current Liability (Total)",
                "Current Liability (Short Term)",
                "Current Liability (Long Term)",
                "Current Liability (Operating)",
                "Current Liability (Operating Income)",
                "Current Liability (Operating Earnings)",
                "Current Liability (Operating Margin)",
            ],
            "Total Stockholder Equity": [
                "Total Equity",
                "Stockholders Equity",
                "Shareholders Equity",
                "Total Stockholders Equity",
                "Total Shareholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity",
                "Total Stockholders Equity Net Minority Interest",
                "Total Equity",
                "Stockholders Equity",
                "Shareholders Equity",
                "Total Equity",
                "Stockholders Equity",
                "Shareholders Equity",
                "Total Stockholders Equity",
                "Total Shareholders Equity",
                # New/expanded
                "Total Stockholder Equity (Net)",
                "Total Stockholder Equity (Gross)",
                "Total Stockholder Equity (Total)",
                "Total Stockholder Equity (Current)",
                "Total Stockholder Equity (Noncurrent)",
                "Total Stockholder Equity (Short Term)",
                "Total Stockholder Equity (Long Term)",
                "Total Stockholder Equity (Operating)",
                "Total Stockholder Equity (Operating Income)",
                "Total Stockholder Equity (Operating Earnings)",
                "Total Stockholder Equity (Operating Margin)",
                "Common Equity",
                "Common Stock",
                "Common Stockholder Equity",
                "Common Stockholders Equity",
                "Common Shareholder Equity",
                "Common Shareholders Equity",
                "Shareholder Equity",
                "Shareholders Equity",
                "Stockholder Equity",
                "Stockholders Equity",
            ],
            "Cash and Cash Equivalents": [
                "Cash and Cash Equivalents",
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash Financial",
                "Cash Equivalents",
                "Cash and Cash Equivalents",
                "Cash And Cash Equivalents",
                "Cash and Cash Equivalents",
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash Financial",
                # New/expanded
                "Cash",
                "Cash Equivalents",
                "Short Term Investments",
                "Cash & Cash Equivalents",
                "Cash & Equivalents",
                "Cash/Equivalents",
                "Cash-Equivalents",
                "Cash Equivalents And Short Term Investments",
                "Cash And Short Term Investments",
                "Cash And Short-Term Investments",
                "Cash And Cash Equivalents And Short Term Investments",
            ],
            "Accounts Receivable": [
                "Accounts Receivable",
                "Receivables",
                "Other Receivables",
                "Accounts Receivable",
                "Receivables",
                "Other Receivables",
                "Accounts Receivable",
                "Receivables",
                "Other Receivables",
                "Accounts Receivable",
                "Receivables",
                "Other Receivables",
                # New/expanded
                "Receivable",
                "Account Receivable",
                "Account Receivables",
                "Trade Receivables",
                "Trade Receivable",
                "Trade And Other Receivables",
                "Trade And Other Receivable",
                "Other Receivable",
                "Other Receivables",
                "Receivables Net",
                "Receivables Gross",
                "Receivables Total",
                "Receivables (Net)",
                "Receivables (Gross)",
                "Receivables (Total)",
            ],
            "Total Debt": [
                "Total Debt",
                "Long Term Debt",
                "Long-Term Debt",
                "Total Long Term Debt",
                "Total Long-Term Debt",
                "Debt",
                "Total Debt (Net)",
                "Total Debt (Gross)",
                "Total Debt (Total)",
                "Total Debt (Current)",
                "Total Debt (Noncurrent)",
                "Total Debt (Short Term)",
                "Total Debt (Long Term)",
                "Total Debt (Operating)",
                "Total Debt (Operating Income)",
                "Total Debt (Operating Earnings)",
                "Total Debt (Operating Margin)",
            ],
            "Short Term Debt": [
                "Short Term Debt",
                "Short-Term Debt",
                "Current Debt",
                "Current Portion of Long Term Debt",
                "Current Portion of Long-Term Debt",
                "Short Term Debt (Net)",
                "Short Term Debt (Gross)",
                "Short Term Debt (Total)",
                "Short Term Debt (Operating)",
                "Short Term Debt (Operating Income)",
                "Short Term Debt (Operating Earnings)",
                "Short Term Debt (Operating Margin)",
            ],
            "Long Term Debt": [
                "Long Term Debt",
                "Long-Term Debt",
                "Noncurrent Debt",
                "Long Term Debt (Net)",
                "Long Term Debt (Gross)",
                "Long Term Debt (Total)",
                "Long Term Debt (Operating)",
                "Long Term Debt (Operating Income)",
                "Long Term Debt (Operating Earnings)",
                "Long Term Debt (Operating Margin)",
            ],
            "Working Capital": [
                "Working Capital",
                "Net Working Capital",
                "Working Capital (Net)",
                "Working Capital (Gross)",
                "Working Capital (Total)",
                "Working Capital (Operating)",
                "Working Capital (Operating Income)",
                "Working Capital (Operating Earnings)",
                "Working Capital (Operating Margin)",
            ],
            "Retained Earnings": [
                "Retained Earnings",
                "Retained Earnings (Net)",
                "Retained Earnings (Gross)",
                "Retained Earnings (Total)",
                "Retained Earnings (Operating)",
                "Retained Earnings (Operating Income)",
                "Retained Earnings (Operating Earnings)",
                "Retained Earnings (Operating Margin)",
            ],
            "Goodwill": [
                "Goodwill",
                "Goodwill (Net)",
                "Goodwill (Gross)",
                "Goodwill (Total)",
                "Goodwill (Operating)",
                "Goodwill (Operating Income)",
                "Goodwill (Operating Earnings)",
                "Goodwill (Operating Margin)",
            ],
            "Intangible Assets": [
                "Intangible Assets",
                "Intangible Assets (Net)",
                "Intangible Assets (Gross)",
                "Intangible Assets (Total)",
                "Intangible Assets (Operating)",
                "Intangible Assets (Operating Income)",
                "Intangible Assets (Operating Earnings)",
                "Intangible Assets (Operating Margin)",
            ],
            "Property Plant and Equipment": [
                "Property Plant and Equipment",
                "PP&E",
                "Property, Plant and Equipment",
                "Property Plant & Equipment",
                "Fixed Assets",
                "Tangible Assets",
                "PP&E (Net)",
                "PP&E (Gross)",
                "PP&E (Total)",
                "PP&E (Operating)",
            ],
            "Accounts Payable": [
                "Accounts Payable",
                "Trade Payables",
                "Payables",
                "Accounts Payable (Net)",
                "Accounts Payable (Gross)",
                "Accounts Payable (Total)",
                "Accounts Payable (Operating)",
            ],
            "Inventory": [
                "Inventory",
                "Inventories",
                "Stock",
                "Inventory (Net)",
                "Inventory (Gross)",
                "Inventory (Total)",
                "Inventory (Operating)",
            ],
            "Prepaid Expenses": [
                "Prepaid Expenses",
                "Prepaid Assets",
                "Prepaid Items",
                "Prepaid Expenses (Net)",
                "Prepaid Expenses (Gross)",
                "Prepaid Expenses (Total)",
                "Prepaid Expenses (Operating)",
            ],
            "Deferred Revenue": [
                "Deferred Revenue",
                "Unearned Revenue",
                "Deferred Income",
                "Deferred Revenue (Net)",
                "Deferred Revenue (Gross)",
                "Deferred Revenue (Total)",
                "Deferred Revenue (Operating)",
            ],
            "Accumulated Depreciation": [
                "Accumulated Depreciation",
                "Accumulated Depreciation and Amortization",
                "Accumulated Depreciation (Net)",
                "Accumulated Depreciation (Gross)",
                "Accumulated Depreciation (Total)",
                "Accumulated Depreciation (Operating)",
            ],
            "Deferred Tax Assets": [
                "Deferred Tax Assets",
                "Deferred Income Tax Assets",
                "Deferred Tax Assets (Net)",
                "Deferred Tax Assets (Gross)",
                "Deferred Tax Assets (Total)",
                "Deferred Tax Assets (Operating)",
            ],
            "Deferred Tax Liabilities": [
                "Deferred Tax Liabilities",
                "Deferred Income Tax Liabilities",
                "Deferred Tax Liabilities (Net)",
                "Deferred Tax Liabilities (Gross)",
                "Deferred Tax Liabilities (Total)",
                "Deferred Tax Liabilities (Operating)",
            ],
            "Minority Interest": [
                "Minority Interest",
                "Noncontrolling Interest",
                "Minority Interests",
                "Minority Interest (Net)",
                "Minority Interest (Gross)",
                "Minority Interest (Total)",
                "Minority Interest (Operating)",
            ],
        },
        "cash_flow": {
            "Operating Cash Flow": [
                "Operating Cash Flow",
                "Cash Flow From Operations",
                "Net Cash Provided by Operating Activities",
                "Cash from Operations",
                "Cash Flow From Continuing Operating Activities",
                "Net Cash Provided By Operating Activities",
                "Operating Cash Flow",
                "Cash Flow From Operations",
                "Net Cash Provided by Operating Activities",
                "Operating Cash Flow",
                "Cash Flow From Operations",
                "Net Cash Provided by Operating Activities",
                "Cash from Operations",
                # New/expanded
                "Net Cash Provided By Operations",
                "Net Cash Provided By Operating Activity",
                "Net Cash Provided By Operating Activities",
                "Net Cash Provided By Operating Activities (Continuing)",
                "Net Cash Provided By Operating Activities (Discontinued)",
                "Net Cash Provided By Operating Activities (Total)",
                "Net Cash Provided By Operating Activities (Net)",
                "Net Cash Provided By Operating Activities (Gross)",
                "Net Cash Provided By Operating Activities (Operating)",
                "Net Cash Provided By Operating Activities (Operating Income)",
                "Net Cash Provided By Operating Activities (Operating Earnings)",
                "Net Cash Provided By Operating Activities (Operating Margin)",
            ],
            "Capital Expenditure": [
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase of PPE",
                "Purchase of Property Plant and Equipment",
                "Capital Spending",
                "Net PPE Purchase And Sale",
                "Purchase Of Property Plant And Equipment",
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase of PPE",
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase of PPE",
                "Purchase of Property Plant and Equipment",
                # New/expanded
                "Purchase Of PPE",
                "Purchase Of Property, Plant And Equipment",
                "Purchase Of Property, Plant, And Equipment",
                "Purchase Of Property Plant And Equipment",
                "Purchase Of Property Plant & Equipment",
                "Purchase Of Property, Plant & Equipment",
                "Purchase Of Property, Plant, And Equipment",
                "Purchase Of Property, Plant, And Equipment (Net)",
                "Purchase Of Property, Plant, And Equipment (Gross)",
                "Purchase Of Property, Plant, And Equipment (Total)",
                "Purchase Of Property, Plant, And Equipment (Operating)",
                "Purchase Of Property, Plant, And Equipment (Operating Income)",
                "Purchase Of Property, Plant, And Equipment (Operating Earnings)",
                "Purchase Of Property, Plant, And Equipment (Operating Margin)",
            ],
            "Free Cash Flow": [
                "Free Cash Flow",
                "FCF",
                "Free Cash Flow to Firm",
                "FCFF",
                "Free Cash Flow to Equity",
                "FCFE",
                "Free Cash Flow Per Share",
                "Free Cash Flow",
                "FCF",
                "Free Cash Flow",
                "FCF",
                "Free Cash Flow to Firm",
                "FCFF",
                "Free Cash Flow to Equity",
                "FCFE",
                # New/expanded
                "Free Cash Flow (Firm)",
                "Free Cash Flow (Equity)",
                "Free Cash Flow (Per Share)",
                "Free Cash Flow (Net)",
                "Free Cash Flow (Gross)",
                "Free Cash Flow (Total)",
                "Free Cash Flow (Operating)",
                "Free Cash Flow (Operating Income)",
                "Free Cash Flow (Operating Earnings)",
                "Free Cash Flow (Operating Margin)",
            ],
            "Net Income": [
                "Net Income",
                "Net Income From Continuing Operations",
                "Net Income From Continuing And Discontinued Operation",
                "Net Income",
                "Net Income From Continuing Operations",
                "Net Income",
                "Net Income From Continuing Operations",
                "Net Income",
                "Net Income From Continuing Operations",
                # New/expanded
                "Net Income (Loss)",
                "Net Income (Profit)",
                "Net Income (Earnings)",
                "Net Income (Margin)",
                "Net Income (Net)",
                "Net Income (Total)",
                "Net Income (Revenue)",
                "Net Income (Sales)",
                "Net Income (Operating)",
                "Net Income (Operating Income)",
                "Net Income (Operating Earnings)",
                "Net Income (Operating Margin)",
            ],
            "Depreciation and Amortization": [
                "Depreciation and Amortization",
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation and Amortization",
                "Depreciation And Amortization",
                "Depreciation and Amortization",
                "Depreciation And Amortization",
                "Depreciation Amortization Depletion",
                # New/expanded
                "Depreciation",
                "Amortization",
                "Depreciation Expense",
                "Amortization Expense",
                "Depreciation And Amortization Expense",
                "Depreciation & Amortization",
                "Depreciation/Amortization",
                "Depreciation-Amortization",
                "Depreciation And Amortization (Net)",
                "Depreciation And Amortization (Gross)",
                "Depreciation And Amortization (Total)",
            ],
            "Dividends Paid": [
                "Dividends Paid",
                "Cash Dividends Paid",
                "Dividend Payments",
                "Dividends Paid (Net)",
                "Dividends Paid (Gross)",
                "Dividends Paid (Total)",
                "Dividends Paid (Operating)",
                "Dividends Paid (Operating Income)",
                "Dividends Paid (Operating Earnings)",
                "Dividends Paid (Operating Margin)",
            ],
            "Stock Repurchases": [
                "Stock Repurchases",
                "Treasury Stock Purchases",
                "Share Repurchases",
                "Stock Repurchases (Net)",
                "Stock Repurchases (Gross)",
                "Stock Repurchases (Total)",
                "Stock Repurchases (Operating)",
                "Stock Repurchases (Operating Income)",
                "Stock Repurchases (Operating Earnings)",
                "Stock Repurchases (Operating Margin)",
            ],
            "Net Debt Issuance": [
                "Net Debt Issuance",
                "Net Debt Issuance (Net)",
                "Net Debt Issuance (Gross)",
                "Net Debt Issuance (Total)",
                "Net Debt Issuance (Operating)",
                "Net Debt Issuance (Operating Income)",
                "Net Debt Issuance (Operating Earnings)",
                "Net Debt Issuance (Operating Margin)",
            ],
            "Net Equity Issuance": [
                "Net Equity Issuance",
                "Net Equity Issuance (Net)",
                "Net Equity Issuance (Gross)",
                "Net Equity Issuance (Total)",
                "Net Equity Issuance (Operating)",
                "Net Equity Issuance (Operating Income)",
                "Net Equity Issuance (Operating Earnings)",
                "Net Equity Issuance (Operating Margin)",
            ],
            "Change in Working Capital": [
                "Change in Working Capital",
                "Changes in Working Capital",
                "Working Capital Changes",
                "Net Change in Working Capital",
                "Change in Working Capital (Net)",
                "Change in Working Capital (Gross)",
                "Change in Working Capital (Total)",
                "Change in Working Capital (Operating)",
            ],
            "Change in Cash": [
                "Change in Cash",
                "Net Change in Cash",
                "Cash Change",
                "Change in Cash (Net)",
                "Change in Cash (Gross)",
                "Change in Cash (Total)",
                "Change in Cash (Operating)",
            ],
            "Foreign Exchange Effects": [
                "Foreign Exchange Effects",
                "Foreign Currency Effects",
                "Foreign Exchange Impact",
                "Currency Translation Effects",
                "Foreign Exchange Effects (Net)",
                "Foreign Exchange Effects (Gross)",
                "Foreign Exchange Effects (Total)",
                "Foreign Exchange Effects (Operating)",
            ],
            "Acquisitions": [
                "Acquisitions",
                "Business Acquisitions",
                "Acquisition of Businesses",
                "Acquisitions (Net)",
                "Acquisitions (Gross)",
                "Acquisitions (Total)",
                "Acquisitions (Operating)",
            ],
            "Disposals": [
                "Disposals",
                "Business Disposals",
                "Disposal of Businesses",
                "Disposals (Net)",
                "Disposals (Gross)",
                "Disposals (Total)",
                "Disposals (Operating)",
            ],
            "Investments": [
                "Investments",
                "Investment Activities",
                "Investment in Securities",
                "Investments (Net)",
                "Investments (Gross)",
                "Investments (Total)",
                "Investments (Operating)",
            ],
        },
    }

    # Get the mapping for this statement type
    mapping = column_mappings.get(statement_type, {})

    # Create a mapping of all possible variations to standard names
    name_mapping = {}
    for std_name, variations in mapping.items():
        for var in variations:
            name_mapping[var] = std_name
            name_mapping[var.lower()] = std_name
            name_mapping[var.upper()] = std_name
            for sep in [" ", "_", "-", "."]:
                name_mapping[var.replace(" ", sep)] = std_name
                name_mapping[var.replace(" ", sep).lower()] = std_name
                name_mapping[var.replace(" ", sep).upper()] = std_name

    # Check if we need to transpose (dates as columns)
    needs_transpose = False
    if isinstance(df_std.columns[0], pd.Timestamp):
        needs_transpose = True
        df_std = df_std.T
        logger.info(f"Transposed {statement_type} statement to have dates as index")

    # Log column mapping process
    logger.info(f"Mapping columns for {statement_type} statement:")
    fuzzy_suggestions = {}
    # Lower fuzzy matching cutoff to 0.7
    fuzzy_cutoff = 0.7
    for col in df_std.columns:
        std_name = name_mapping.get(col, None)
        if std_name:
            logger.info(f"  Mapping '{col}' to '{std_name}'")
        else:
            close_matches = difflib.get_close_matches(
                col, name_mapping.keys(), n=1, cutoff=fuzzy_cutoff
            )
            if close_matches:
                suggested = name_mapping[close_matches[0]]
                fuzzy_suggestions[col] = suggested
                logger.warning(
                    f"  Fuzzy mapping: '{col}' -> '{suggested}' (via '{close_matches[0]}')"
                )
            else:
                logger.warning(f"  No mapping found for '{col}'")

    # Create a new DataFrame with standardized columns
    standardized_data = {}
    for col in df_std.columns:
        std_name = name_mapping.get(col, None)
        if not std_name:
            # Fuzzy fallback
            close_matches = difflib.get_close_matches(
                col, name_mapping.keys(), n=1, cutoff=fuzzy_cutoff
            )
            if close_matches:
                std_name = name_mapping[close_matches[0]]
            else:
                std_name = col  # preserve as-is
        if std_name in standardized_data:
            standardized_data[std_name] = standardized_data[std_name].add(
                df_std[col], fill_value=0
            )
            logger.info(f"  Combined values for '{std_name}' from '{col}'")
        else:
            standardized_data[std_name] = df_std[col]
            logger.info(f"  Added new column '{std_name}' from '{col}'")

    # Create new DataFrame with standardized columns
    df_std = pd.DataFrame(standardized_data)

    # Log any unmapped columns and fuzzy suggestions
    unmapped_cols = [
        col
        for col in df_std.columns
        if col not in name_mapping.values() and col not in mapping.keys()
    ]
    if unmapped_cols:
        logger.warning(
            f"Unmapped columns in {statement_type} statement: {unmapped_cols}"
        )
        logger.warning(f"These columns will be preserved as-is")
    if fuzzy_suggestions:
        logger.warning(
            f"Fuzzy mapping suggestions for {statement_type} statement: {fuzzy_suggestions}"
        )

    # Add missing standard columns with default values
    missing_cols = []
    for std_name in mapping.keys():
        if std_name not in df_std.columns:
            missing_cols.append(std_name)
            df_std[std_name] = 0
            logger.warning(f"Added missing column '{std_name}' with default value 0")

    if missing_cols:
        logger.warning(
            f"Missing required columns in {statement_type} statement: {missing_cols}"
        )

    # Transpose back if needed
    if needs_transpose:
        df_std = df_std.T
        logger.info(
            f"Transposed {statement_type} statement back to original orientation"
        )

    # Log final column list and sample data after mapping
    logger.info(f"Final columns in {statement_type} statement: {list(df_std.columns)}")
    logger.info(
        f"Sample data from {statement_type} statement after mapping (first 3 rows):\n{df_std.head(3).to_string()}"
    )

    return df_std


def fetch_financial_data(ticker: str) -> dict[str, pd.DataFrame | dict]:
    """
    Fetch financial data for a given ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing:
            - income_stmt: Income statement DataFrame
            - balance_sheet: Balance sheet DataFrame
            - cash_flow: Cash flow statement DataFrame
            - fundamentals: Dictionary of fundamental metrics
    """
    try:
        logger.info(f"Fetching financial data for {ticker}")

        # Fetch raw data
        income_stmt = fetch_income_statement(ticker)
        balance_sheet = fetch_balance_sheet(ticker)
        cash_flow = fetch_cash_flow(ticker)
        fundamentals = fetch_fundamentals(ticker)

        # Standardize column names
        income_stmt = _standardize_column_names(income_stmt, "income")
        balance_sheet = _standardize_column_names(balance_sheet, "balance")
        cash_flow = _standardize_column_names(cash_flow, "cash_flow")

        # Fill missing values with 0
        income_stmt = income_stmt.fillna(0)
        balance_sheet = balance_sheet.fillna(0)
        cash_flow = cash_flow.fillna(0)

        return {
            "income_stmt": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "fundamentals": fundamentals,
        }

    except Exception as e:
        logger.error(f"Error fetching financial data: {str(e)}")
        return {
            "income_stmt": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
            "fundamentals": {},
        }
