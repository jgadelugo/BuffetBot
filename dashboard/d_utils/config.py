"""Dashboard configuration and utility functions."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardConfig:
    """Configuration constants for the dashboard."""

    # Colors for different statuses
    STATUS_COLORS = {
        "good": {"bg": "#d4edda", "border": "#28a745", "text": "#155724"},
        "warning": {"bg": "#fff3cd", "border": "#ffc107", "text": "#856404"},
        "bad": {"bg": "#f8d7da", "border": "#dc3545", "text": "#721c24"},
        "neutral": {"bg": "#e9ecef", "border": "#6c757d", "text": "#495057"},
    }

    # Thresholds for various metrics
    THRESHOLDS = {
        "margin_of_safety": {
            "excellent": 0.25,
            "good": 0.15,
            "fair": 0.05,
            "poor": -0.10,
        },
        "pe_ratio": {"low": 15, "moderate": 25, "high": 35},
        "volatility": {"low": 0.15, "moderate": 0.25, "high": 0.40},
        "rsi": {
            "oversold": 30,
            "neutral_low": 40,
            "neutral_high": 60,
            "overbought": 70,
        },
        "debt_to_equity": {"low": 0.5, "moderate": 1.0, "high": 2.0},
    }

    # Chart configurations
    CHART_CONFIG = {
        "height": 400,
        "margin": {"l": 20, "r": 20, "t": 60, "b": 20},
        "font_family": "Arial, sans-serif",
        "color_scheme": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    }

    # Technical indicator periods
    TECHNICAL_PERIODS = {
        "sma_short": 20,
        "sma_medium": 50,
        "sma_long": 200,
        "ema_short": 12,
        "ema_long": 26,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_std": 2,
    }


def get_status_color(value: float, metric_type: str) -> str:
    """Determine status color based on metric value and type.

    Args:
        value: The metric value
        metric_type: Type of metric (e.g., 'margin_of_safety', 'pe_ratio')

    Returns:
        Status string ('good', 'warning', 'bad', 'neutral')
    """
    try:
        thresholds = DashboardConfig.THRESHOLDS.get(metric_type, {})

        if metric_type == "margin_of_safety":
            if value >= thresholds["excellent"]:
                return "good"
            elif value >= thresholds["good"]:
                return "good"
            elif value >= thresholds["fair"]:
                return "warning"
            elif value >= thresholds["poor"]:
                return "warning"
            else:
                return "bad"

        elif metric_type == "pe_ratio":
            if value < thresholds["low"]:
                return "good"
            elif value < thresholds["moderate"]:
                return "neutral"
            elif value < thresholds["high"]:
                return "warning"
            else:
                return "bad"

        elif metric_type == "volatility":
            if value < thresholds["low"]:
                return "good"
            elif value < thresholds["moderate"]:
                return "neutral"
            elif value < thresholds["high"]:
                return "warning"
            else:
                return "bad"

        elif metric_type == "rsi":
            if value < thresholds["oversold"]:
                return "good"  # Buying opportunity
            elif value > thresholds["overbought"]:
                return "warning"  # Selling opportunity
            else:
                return "neutral"

        elif metric_type == "debt_to_equity":
            if value < thresholds["low"]:
                return "good"
            elif value < thresholds["moderate"]:
                return "neutral"
            elif value < thresholds["high"]:
                return "warning"
            else:
                return "bad"

        else:
            return "neutral"

    except Exception as e:
        logger.error(f"Error determining status color: {str(e)}")
        return "neutral"


def format_currency(value: float | int | None, decimals: int = 2) -> str:
    """Format value as currency string.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"

        if abs(value) >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        else:
            return f"${value:.{decimals}f}"

    except Exception as e:
        logger.error(f"Error formatting currency: {str(e)}")
        return "N/A"


def format_percentage(value: float | None, decimals: int = 1) -> str:
    """Format value as percentage string.

    Args:
        value: Numeric value to format (0.1 = 10%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"

        return f"{value * 100:.{decimals}f}%"

    except Exception as e:
        logger.error(f"Error formatting percentage: {str(e)}")
        return "N/A"


def calculate_status(
    current: float, benchmark: float, metric_type: str = "higher_better"
) -> tuple[str, str]:
    """Calculate status and delta based on current vs benchmark.

    Args:
        current: Current value
        benchmark: Benchmark value
        metric_type: 'higher_better' or 'lower_better'

    Returns:
        Tuple of (status, delta_string)
    """
    try:
        if pd.isna(current) or pd.isna(benchmark) or benchmark == 0:
            return "neutral", "N/A"

        delta = (current - benchmark) / abs(benchmark)
        delta_str = f"{delta:+.1%}"

        if metric_type == "higher_better":
            if delta > 0.1:
                status = "good"
            elif delta > -0.1:
                status = "neutral"
            else:
                status = "bad"
        else:  # lower_better
            if delta < -0.1:
                status = "good"
            elif delta < 0.1:
                status = "neutral"
            else:
                status = "bad"

        return status, delta_str

    except Exception as e:
        logger.error(f"Error calculating status: {str(e)}")
        return "neutral", "N/A"


def validate_data(data: dict, required_fields: list) -> tuple[bool, list]:
    """Validate that required data fields are present and valid.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    missing_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif data[field] is None:
            missing_fields.append(field)
        elif isinstance(data[field], pd.DataFrame) and data[field].empty:
            missing_fields.append(field)
        elif isinstance(data[field], dict) and not data[field]:
            missing_fields.append(field)

    is_valid = len(missing_fields) == 0

    if not is_valid:
        logger.warning(f"Missing required fields: {missing_fields}")

    return is_valid, missing_fields


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or invalid.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if division fails

    Returns:
        Division result or default value
    """
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return default
        return numerator / denominator
    except Exception as e:
        logger.error(f"Error in safe division: {str(e)}")
        return default


def calculate_moving_average(
    data: pd.Series, window: int, method: str = "simple"
) -> pd.Series:
    """Calculate moving average with error handling.

    Args:
        data: Price series
        window: Window size
        method: 'simple' or 'exponential'

    Returns:
        Moving average series
    """
    try:
        if len(data) < window:
            logger.warning(f"Insufficient data for {window}-period moving average")
            return pd.Series(index=data.index)

        if method == "simple":
            return data.rolling(window=window).mean()
        elif method == "exponential":
            return data.ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        logger.error(f"Error calculating moving average: {str(e)}")
        return pd.Series(index=data.index)


def get_trend_direction(
    current: float, previous: float, threshold: float = 0.01
) -> str:
    """Determine trend direction based on change threshold.

    Args:
        current: Current value
        previous: Previous value
        threshold: Minimum change to be considered significant

    Returns:
        'up', 'down', or 'flat'
    """
    try:
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return "flat"

        change = (current - previous) / abs(previous)

        if change > threshold:
            return "up"
        elif change < -threshold:
            return "down"
        else:
            return "flat"

    except Exception as e:
        logger.error(f"Error determining trend direction: {str(e)}")
        return "flat"
