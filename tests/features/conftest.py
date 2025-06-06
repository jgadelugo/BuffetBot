"""
Pytest configuration and fixtures for feature engineering tests.

Provides common test data, fixtures, and utilities used across all
feature engineering test modules.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_market_data() -> dict[str, pd.Series]:
    """
    Generate realistic sample market data for testing.

    Returns:
        Dictionary containing OHLCV data with 100 data points
    """
    np.random.seed(42)  # For reproducible tests
    n_points = 100
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Generate realistic price walk
    base_price = 100.0
    returns = np.random.normal(
        0.001, 0.02, n_points
    )  # Daily returns ~1% drift, 2% volatility

    # Calculate prices using cumulative returns
    price_multipliers = np.cumprod(1 + returns)
    close_prices = base_price * price_multipliers

    # Generate OHLC data with realistic relationships
    close = pd.Series(close_prices, index=dates, name="close")

    # High and low should be reasonable relative to close
    daily_range = np.abs(np.random.normal(0, 0.015, n_points))  # ~1.5% daily range
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)

    # Open should be close to previous close (with some gap)
    open_gaps = np.random.normal(0, 0.005, n_points)  # ~0.5% overnight gaps
    open_prices = close.shift(1) * (1 + open_gaps)
    open_prices.iloc[0] = close.iloc[0]  # First day open = close

    # Volume should be realistic (millions of shares)
    base_volume = 5_000_000
    volume_multiplier = np.random.lognormal(0, 0.3, n_points)
    volume = pd.Series(
        (base_volume * volume_multiplier).astype(int), index=dates, name="volume"
    )

    return {
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dates": dates,
    }


@pytest.fixture
def small_market_data() -> dict[str, pd.Series]:
    """
    Generate small sample data for edge case testing.

    Returns:
        Dictionary containing OHLCV data with only 10 data points
    """
    n_points = 10
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    close = pd.Series(prices, index=dates, name="close")
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([1_000_000] * n_points, index=dates, name="volume")

    return {
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dates": dates,
    }


@pytest.fixture
def insufficient_data() -> dict[str, pd.Series]:
    """
    Generate data with insufficient points for most indicators.

    Returns:
        Dictionary containing OHLCV data with only 3 data points
    """
    n_points = 3
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    prices = [100, 101, 102]
    close = pd.Series(prices, index=dates, name="close")
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([1_000_000] * n_points, index=dates, name="volume")

    return {
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dates": dates,
    }


@pytest.fixture
def data_with_nans() -> dict[str, pd.Series]:
    """
    Generate data with NaN values to test robustness.

    Returns:
        Dictionary containing OHLCV data with NaN values
    """
    n_points = 50
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Create data with some NaN values
    prices = np.random.normal(100, 5, n_points)
    prices[10:15] = np.nan  # Block of NaN values
    prices[25] = np.nan  # Individual NaN
    prices[40] = np.nan  # Another individual NaN

    close = pd.Series(prices, index=dates, name="close")
    high = close * 1.02
    low = close * 0.98
    volume = pd.Series(
        np.random.randint(1_000_000, 10_000_000, n_points), index=dates, name="volume"
    )

    # Add some NaN values to volume too
    volume.iloc[12:14] = np.nan

    return {
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dates": dates,
    }


@pytest.fixture
def performance_data() -> dict[str, pd.Series]:
    """
    Generate larger dataset for performance testing.

    Returns:
        Dictionary containing OHLCV data with 1000 data points
    """
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range("2023-01-01", periods=n_points, freq="h")  # Hourly data

    # Generate realistic price walk
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.005, n_points)  # Hourly returns

    price_multipliers = np.cumprod(1 + returns)
    close_prices = base_price * price_multipliers

    close = pd.Series(close_prices, index=dates, name="close")

    daily_range = np.abs(np.random.normal(0, 0.003, n_points))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)

    volume = pd.Series(
        np.random.randint(100_000, 5_000_000, n_points), index=dates, name="volume"
    )

    return {
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "dates": dates,
    }


@pytest.fixture
def indicator_expected_ranges() -> dict[str, tuple[float, float]]:
    """
    Expected ranges for various technical indicators for validation.

    Returns:
        Dictionary mapping indicator names to (min, max) expected ranges
    """
    return {
        "rsi": (0, 100),
        "williams_r": (-100, 0),
        "stochastic_k": (0, 100),
        "stochastic_d": (0, 100),
        "bollinger_percent_b": (-0.5, 1.5),  # Can go outside bands
        "atr": (0, float("inf")),
        "historical_volatility": (0, float("inf")),
        "volume_roc": (-100, float("inf")),
    }


class TestDataValidator:
    """Utility class for validating test results."""

    @staticmethod
    def is_series_valid(series: pd.Series, allow_nan: bool = True) -> bool:
        """Check if a pandas Series is valid."""
        if not isinstance(series, pd.Series):
            return False

        if len(series) == 0:
            return False

        if not allow_nan and series.isna().any():
            return False

        return True

    @staticmethod
    def values_in_range(
        series: pd.Series, min_val: float, max_val: float, tolerance: float = 0.01
    ) -> bool:
        """Check if series values are within expected range (excluding NaN)."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return True  # No values to check

        # Allow small tolerance for floating point precision
        return (clean_series >= (min_val - tolerance)).all() and (
            clean_series <= (max_val + tolerance)
        ).all()

    @staticmethod
    def has_reasonable_non_nan_ratio(series: pd.Series, min_ratio: float = 0.5) -> bool:
        """Check if series has reasonable number of non-NaN values."""
        if len(series) == 0:
            return False

        non_nan_ratio = series.notna().sum() / len(series)
        return non_nan_ratio >= min_ratio


def assert_indicator_output_valid(
    result, expected_type=pd.Series, expected_range=None, min_non_nan_ratio=0.5
):
    """
    Common assertion helper for validating indicator outputs.

    Args:
        result: The indicator calculation result
        expected_type: Expected type of the result
        expected_range: Tuple of (min, max) expected values
        min_non_nan_ratio: Minimum ratio of non-NaN values required
    """
    # Check type
    assert isinstance(
        result, expected_type
    ), f"Expected {expected_type}, got {type(result)}"

    if isinstance(result, pd.Series):
        # Check that we have some data
        assert len(result) > 0, "Result series is empty"

        # Check non-NaN ratio
        validator = TestDataValidator()
        assert validator.has_reasonable_non_nan_ratio(
            result, min_non_nan_ratio
        ), f"Too many NaN values: {result.isna().sum()}/{len(result)}"

        # Check value ranges if specified
        if expected_range:
            min_val, max_val = expected_range
            assert validator.values_in_range(
                result, min_val, max_val
            ), f"Values outside expected range {expected_range}: min={result.min()}, max={result.max()}"

    elif isinstance(result, dict):
        # For dictionary results (like MACD), validate each component
        assert len(result) > 0, "Result dictionary is empty"

        for key, value in result.items():
            assert isinstance(
                value, pd.Series
            ), f"Dictionary value '{key}' is not a Series"
            assert len(value) > 0, f"Series '{key}' is empty"
