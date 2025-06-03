"""
Financial calculation utilities.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd


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
        return 50.0  # Return neutral RSI on error


def calculate_working_capital(balance_sheet: pd.DataFrame) -> float:
    """Calculate working capital from balance sheet."""
    if balance_sheet.empty:
        return 0.0

    current_assets = balance_sheet.get("Total Current Assets", pd.Series([0])).iloc[0]
    current_liabilities = balance_sheet.get(
        "Total Current Liabilities", pd.Series([0])
    ).iloc[0]

    return float(current_assets - current_liabilities)


def calculate_debt_to_equity(balance_sheet: pd.DataFrame) -> float:
    """Calculate debt-to-equity ratio from balance sheet."""
    if balance_sheet.empty:
        return 0.0

    total_debt = balance_sheet.get("Long Term Debt", pd.Series([0])).iloc[0]
    if "Short Term Debt" in balance_sheet.columns:
        total_debt += balance_sheet.get("Short Term Debt", pd.Series([0])).iloc[0]

    total_equity = balance_sheet.get("Total Stockholder Equity", pd.Series([0])).iloc[0]

    if total_equity == 0:
        return 0.0

    return float(total_debt / total_equity)


def calculate_current_ratio(balance_sheet: pd.DataFrame) -> float:
    """Calculate current ratio from balance sheet."""
    if balance_sheet.empty:
        return 0.0

    current_assets = balance_sheet.get("Total Current Assets", pd.Series([0])).iloc[0]
    current_liabilities = balance_sheet.get(
        "Total Current Liabilities", pd.Series([0])
    ).iloc[0]

    if current_liabilities == 0:
        return 0.0

    return float(current_assets / current_liabilities)


def calculate_quick_ratio(balance_sheet: pd.DataFrame) -> float:
    """Calculate quick ratio from balance sheet."""
    if balance_sheet.empty:
        return 0.0

    current_assets = balance_sheet.get("Total Current Assets", pd.Series([0])).iloc[0]
    inventory = balance_sheet.get("Inventory", pd.Series([0])).iloc[0]
    current_liabilities = balance_sheet.get(
        "Total Current Liabilities", pd.Series([0])
    ).iloc[0]

    if current_liabilities == 0:
        return 0.0

    return float((current_assets - inventory) / current_liabilities)


def calculate_roa(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
    """Calculate return on assets."""
    if income_stmt.empty or balance_sheet.empty:
        return 0.0

    net_income = income_stmt.get("Net Income", pd.Series([0])).iloc[0]
    total_assets = balance_sheet.get("Total Assets", pd.Series([0])).iloc[0]

    if total_assets == 0:
        return 0.0

    return float(net_income / total_assets)


def calculate_roe(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> float:
    """Calculate return on equity."""
    if income_stmt.empty or balance_sheet.empty:
        return 0.0

    net_income = income_stmt.get("Net Income", pd.Series([0])).iloc[0]
    total_equity = balance_sheet.get("Total Stockholder Equity", pd.Series([0])).iloc[0]

    if total_equity == 0:
        return 0.0

    return float(net_income / total_equity)


def calculate_gross_margin(gross_profit: float, revenue: float) -> float:
    """
    Calculate gross margin.

    Args:
        gross_profit: Gross profit
        revenue: Total revenue

    Returns:
        float: Gross margin
    """
    if revenue == 0:
        return 0.0
    return gross_profit / revenue


def calculate_operating_margin(operating_income: float, revenue: float) -> float:
    """
    Calculate operating margin.

    Args:
        operating_income: Operating income
        revenue: Total revenue

    Returns:
        float: Operating margin
    """
    if revenue == 0:
        return 0.0
    return operating_income / revenue


def calculate_net_margin(net_income: float, revenue: float) -> float:
    """
    Calculate net margin.

    Args:
        net_income: Net income
        revenue: Total revenue

    Returns:
        float: Net margin
    """
    if revenue == 0:
        return 0.0
    return net_income / revenue
