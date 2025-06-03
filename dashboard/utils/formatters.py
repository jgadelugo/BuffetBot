"""Safe formatting utilities for displaying financial data."""

from typing import Union

import numpy as np
import pandas as pd


def safe_format_currency(value: float | None, decimal_places: int = 2) -> str:
    """Safely format a value as currency, handling None/NaN/invalid values.

    Args:
        value: Value to format
        decimal_places: Number of decimal places to show

    Returns:
        str: Formatted currency string or "N/A" for invalid values
    """
    try:
        # Handle None and NaN values
        if value is None:
            return "N/A"

        # Handle pandas NaN and numpy values
        if pd.isna(value):
            return "N/A"

        # Handle infinity values
        if np.isinf(value):
            return "N/A"

        # Convert to float and format
        numeric_value = float(value)
        return f"${numeric_value:,.{decimal_places}f}"

    except (ValueError, TypeError, AttributeError):
        return "N/A"


def safe_format_percentage(value: float | None, decimal_places: int = 1) -> str:
    """Safely format a value as percentage, handling None/NaN/invalid values.

    Args:
        value: Value to format (as decimal, e.g., 0.15 for 15%)
        decimal_places: Number of decimal places to show

    Returns:
        str: Formatted percentage string or "N/A" for invalid values
    """
    try:
        # Handle None and NaN values
        if value is None:
            return "N/A"

        # Handle pandas NaN and numpy values
        if pd.isna(value):
            return "N/A"

        # Handle infinity values
        if np.isinf(value):
            return "N/A"

        # Convert to float and format
        numeric_value = float(value)
        return f"{numeric_value:.{decimal_places}%}"

    except (ValueError, TypeError, AttributeError):
        return "N/A"


def safe_format_number(value: float | None, decimal_places: int = 2) -> str:
    """Safely format a number, handling None/NaN/invalid values.

    Args:
        value: Value to format
        decimal_places: Number of decimal places to show

    Returns:
        str: Formatted number string or "N/A" for invalid values
    """
    try:
        # Handle None and NaN values
        if value is None:
            return "N/A"

        # Handle pandas NaN and numpy values
        if pd.isna(value):
            return "N/A"

        # Handle infinity values
        if np.isinf(value):
            return "N/A"

        # Convert to float and format
        numeric_value = float(value)
        return f"{numeric_value:.{decimal_places}f}"

    except (ValueError, TypeError, AttributeError):
        return "N/A"
