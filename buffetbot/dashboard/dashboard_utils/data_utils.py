"""Data utilities for safe data extraction and manipulation."""

from typing import Any, Union

import numpy as np
import pandas as pd


def safe_get_nested_value(data: dict[str, Any], *keys) -> Any:
    """Safely get a nested dictionary value, returning None if any key is missing.

    Args:
        data: The dictionary to extract from
        *keys: The sequence of keys to traverse

    Returns:
        The value at the nested location, or None if any key is missing
    """
    try:
        result = data
        for key in keys:
            if result is None or not isinstance(result, dict) or key not in result:
                return None
            result = result[key]
        return result
    except (KeyError, TypeError, AttributeError):
        return None


def safe_get_last_price(price_data: pd.DataFrame | None) -> float | None:
    """Safely get the last closing price from price data.

    Args:
        price_data: DataFrame containing price data with 'Close' column

    Returns:
        The last closing price, or None if data is invalid/missing
    """
    try:
        if (
            price_data is None
            or not isinstance(price_data, pd.DataFrame)
            or price_data.empty
            or "Close" not in price_data.columns
        ):
            return None

        last_price = price_data["Close"].iloc[-1]

        # Check if the last price is NaN
        if pd.isna(last_price):
            return None

        return float(last_price)
    except (IndexError, KeyError, ValueError, TypeError, AttributeError):
        return None
