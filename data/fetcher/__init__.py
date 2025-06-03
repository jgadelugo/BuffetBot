"""
Data fetcher package for financial data.
"""

import numpy as np
import pandas as pd

from .fetcher import DataFetcher
from .utils.financial_calculations import (
    calculate_current_ratio,
    calculate_debt_to_equity,
    calculate_gross_margin,
    calculate_net_margin,
    calculate_operating_margin,
    calculate_quick_ratio,
    calculate_roa,
    calculate_roe,
    calculate_rsi,
    calculate_working_capital,
)

# Create a singleton instance
_fetcher = DataFetcher()


def fetch_stock_data(ticker: str, years: int = 5) -> dict:
    """
    Convenience function to fetch stock data.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch

    Returns:
        Dictionary containing:
            - price_data: Historical price data
            - income_stmt: Income statement
            - balance_sheet: Balance sheet
            - cash_flow: Cash flow statement
            - fundamentals: Fundamental metrics
            - metrics: Calculated financial metrics
    """
    # Fetch price history
    price_data = _fetcher.fetch_price_history(ticker, period=f"{years}y")

    # Fetch financial data
    financial_data = _fetcher.fetch_financial_data(ticker)

    # Get metrics
    latest_price = price_data["Close"].iloc[-1] if not price_data.empty else None

    # Calculate price change (percentage change from previous close)
    price_change = 0.0
    if not price_data.empty and len(price_data) > 1:
        prev_close = price_data["Close"].iloc[-2]
        if prev_close > 0:
            price_change = ((latest_price - prev_close) / prev_close) * 100

    # Calculate additional metrics
    metrics = {
        "latest_price": latest_price,
        "price_change": price_change,
        "volatility": price_data["Return"].std() * np.sqrt(252)
        if "Return" in price_data
        else 0,
        "momentum": price_data["Close"].pct_change(20).iloc[-1]
        if len(price_data) > 20
        else 0,
        "rsi": calculate_rsi(price_data["Close"]) if not price_data.empty else 50,
        "working_capital": calculate_working_capital(
            financial_data.get("balance", pd.DataFrame())
        ),
        "debt_to_equity": calculate_debt_to_equity(
            financial_data.get("balance", pd.DataFrame())
        ),
        "current_ratio": calculate_current_ratio(
            financial_data.get("balance", pd.DataFrame())
        ),
        "roa": calculate_roa(
            financial_data.get("income", pd.DataFrame()),
            financial_data.get("balance", pd.DataFrame()),
        ),
        "roe": calculate_roe(
            financial_data.get("income", pd.DataFrame()),
            financial_data.get("balance", pd.DataFrame()),
        ),
    }

    return {
        "price_data": price_data,
        "income_stmt": financial_data["income"],
        "balance_sheet": financial_data["balance"],
        "cash_flow": financial_data["cash_flow"],
        "fundamentals": financial_data["fundamentals"],
        "metrics": metrics,
    }


__all__ = [
    "DataFetcher",
    "fetch_stock_data",
    "calculate_rsi",
    "calculate_working_capital",
    "calculate_debt_to_equity",
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_roa",
    "calculate_roe",
    "calculate_gross_margin",
    "calculate_operating_margin",
    "calculate_net_margin",
]
