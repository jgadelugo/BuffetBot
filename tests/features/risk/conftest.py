"""
Pytest configuration and fixtures specific to risk feature testing.

Provides specialized test data and fixtures for risk metrics,
drawdown analysis, correlation metrics, and risk-adjusted returns.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def risk_returns_data() -> dict[str, pd.Series]:
    """
    Generate realistic return series for risk analysis testing.

    Creates returns with realistic characteristics including:
    - Volatility clustering
    - Negative skewness
    - Excess kurtosis
    - Occasional extreme events

    Returns:
        Dictionary containing various return series
    """
    np.random.seed(42)
    n_points = 252  # One year of daily data
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Create realistic return patterns
    stock_drift = 0.0008  # ~20% annual return
    stock_vol = 0.02  # ~32% annual volatility

    # Add volatility clustering (GARCH-like effects)
    vol_persistence = 0.9
    vol_shock = 0.1
    current_vol = stock_vol

    stock_returns = []
    market_returns = []

    for i in range(n_points):
        # Volatility clustering
        vol_innovation = np.random.normal(0, vol_shock)
        current_vol = (
            vol_persistence * current_vol
            + (1 - vol_persistence) * stock_vol
            + vol_innovation
        )
        current_vol = max(0.005, min(0.05, current_vol))  # Bound volatility

        # Generate correlated returns
        market_return = np.random.normal(stock_drift * 0.8, current_vol * 0.7)

        # Stock return correlated with market (correlation ~0.7)
        stock_return = (
            np.random.normal(stock_drift, current_vol) * 0.7 + market_return * 0.3
        )

        # Add occasional extreme events
        if np.random.random() < 0.02:  # 2% chance of extreme event
            extreme_shock = np.random.normal(-0.05, 0.02)  # Average -5% shock
            stock_return += extreme_shock
            market_return += extreme_shock * 0.8

        stock_returns.append(stock_return)
        market_returns.append(market_return)

    # Convert to pandas Series
    stock_returns = pd.Series(stock_returns, index=dates, name="stock_returns")
    market_returns = pd.Series(market_returns, index=dates, name="market_returns")

    return {
        "stock_returns": stock_returns,
        "market_returns": market_returns,
        "benchmark_returns": market_returns * 0.8,  # Lower volatility benchmark
        "dates": dates,
    }


@pytest.fixture
def price_data_with_drawdowns() -> pd.Series:
    """
    Generate price series with realistic drawdown patterns.

    Returns:
        Price series designed to test drawdown calculations
    """
    np.random.seed(42)
    n_points = 252
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Create price series with notable drawdowns
    returns = np.random.normal(0.001, 0.015, n_points)

    # Add some specific drawdown periods
    # Major drawdown in the middle
    returns[100:130] = np.random.normal(-0.02, 0.01, 30)
    # Recovery period
    returns[130:150] = np.random.normal(0.015, 0.01, 20)
    # Another smaller drawdown
    returns[200:210] = np.random.normal(-0.015, 0.008, 10)

    # Convert to prices
    prices = (1 + pd.Series(returns, index=dates)).cumprod() * 100

    return prices


@pytest.fixture
def multi_asset_returns() -> dict[str, pd.Series]:
    """
    Generate multiple correlated asset return series for correlation testing.

    Returns:
        Dictionary of return series for different asset classes
    """
    np.random.seed(42)
    n_points = 252
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Create correlation structure
    correlation_matrix = np.array(
        [
            [1.00, 0.70, 0.50, 0.30, -0.20],  # Stock
            [0.70, 1.00, 0.40, 0.25, -0.15],  # Market
            [0.50, 0.40, 1.00, 0.60, -0.10],  # Tech
            [0.30, 0.25, 0.60, 1.00, -0.05],  # Growth
            [-0.20, -0.15, -0.10, -0.05, 1.00],  # Bonds
        ]
    )

    # Generate correlated random variables
    independent_vars = np.random.multivariate_normal(
        mean=[0.001, 0.0008, 0.0012, 0.0015, 0.0003],
        cov=correlation_matrix * 0.02**2,  # Scale by volatility
        size=n_points,
    )

    asset_names = ["stock", "market", "tech", "growth", "bonds"]
    return {
        name: pd.Series(returns, index=dates, name=name)
        for name, returns in zip(asset_names, independent_vars.T)
    }


@pytest.fixture
def extreme_returns() -> pd.Series:
    """
    Generate return series with extreme values for stress testing.

    Returns:
        Return series with fat tails and extreme events
    """
    np.random.seed(42)
    n_points = 100
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Mix of normal and extreme returns
    normal_returns = np.random.normal(0, 0.01, n_points)

    # Add some extreme negative returns (crashes)
    extreme_indices = [20, 50, 80]
    normal_returns[extreme_indices] = [-0.15, -0.12, -0.08]  # Extreme losses

    # Add some extreme positive returns
    extreme_indices = [25, 55, 85]
    normal_returns[extreme_indices] = [0.10, 0.08, 0.06]  # Extreme gains

    return pd.Series(normal_returns, index=dates, name="extreme_returns")


@pytest.fixture
def insufficient_risk_data() -> pd.Series:
    """
    Generate insufficient data for testing edge cases.

    Returns:
        Very short return series for testing insufficient data handling
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008], index=dates)
    return returns


@pytest.fixture
def returns_with_nans() -> pd.Series:
    """
    Generate return series with NaN values for robustness testing.

    Returns:
        Return series containing NaN values
    """
    np.random.seed(42)
    n_points = 50
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    returns = np.random.normal(0.001, 0.02, n_points)

    # Add NaN values
    returns[10:15] = np.nan  # Block of NaN
    returns[25] = np.nan  # Individual NaN
    returns[40] = np.nan  # Another individual NaN

    return pd.Series(returns, index=dates, name="returns_with_nans")


@pytest.fixture
def risk_free_rates() -> dict[str, float]:
    """
    Provide various risk-free rates for testing.

    Returns:
        Dictionary of risk-free rates for different scenarios
    """
    return {
        "zero": 0.0,
        "low": 0.02,  # 2% annual
        "normal": 0.05,  # 5% annual
        "high": 0.08,  # 8% annual
    }


@pytest.fixture
def expected_risk_ranges() -> dict[str, tuple[float, float]]:
    """
    Expected ranges for risk metrics for validation.

    Returns:
        Dictionary mapping risk metric names to expected ranges
    """
    return {
        "var_95": (-1.0, 0.0),  # VaR should be negative
        "var_99": (-1.0, 0.0),  # VaR should be negative
        "expected_shortfall": (-1.0, 0.0),  # ES should be negative
        "max_drawdown": (-1.0, 0.0),  # Drawdown should be negative
        "sharpe_ratio": (-5.0, 5.0),  # Reasonable Sharpe range
        "sortino_ratio": (-5.0, 10.0),  # Sortino can be higher
        "calmar_ratio": (-10.0, 10.0),  # Calmar range
        "correlation": (-1.0, 1.0),  # Correlation bounds
        "beta": (-3.0, 3.0),  # Reasonable beta range
    }


class RiskTestUtilities:
    """Utility functions for risk feature testing."""

    @staticmethod
    def validate_var_output(var_result: dict[str, pd.Series]) -> bool:
        """Validate VaR calculation output structure."""
        required_keys = ["var_95", "var_99"]
        return all(key in var_result for key in required_keys)

    @staticmethod
    def validate_drawdown_output(dd_result: dict[str, pd.Series]) -> bool:
        """Validate drawdown calculation output structure."""
        required_keys = [
            "drawdown",
            "underwater",
            "drawdown_duration",
            "cumulative_max",
        ]
        return all(key in dd_result for key in required_keys)

    @staticmethod
    def validate_var_values(
        var_series: pd.Series, expected_range: tuple[float, float]
    ) -> bool:
        """Validate VaR values are within expected range."""
        valid_values = var_series.dropna()
        if len(valid_values) == 0:
            return True  # Empty series is valid
        return (valid_values >= expected_range[0]).all() and (
            valid_values <= expected_range[1]
        ).all()

    @staticmethod
    def validate_correlation_matrix(corr_matrix: pd.DataFrame) -> bool:
        """Validate correlation matrix properties."""
        if not isinstance(corr_matrix, pd.DataFrame):
            return False

        # Check diagonal is 1
        diagonal_ones = np.allclose(np.diag(corr_matrix), 1.0, atol=1e-10)

        # Check symmetric
        is_symmetric = np.allclose(corr_matrix, corr_matrix.T, atol=1e-10)

        # Check values in [-1, 1]
        values_in_range = ((corr_matrix >= -1.0) & (corr_matrix <= 1.0)).all().all()

        return diagonal_ones and is_symmetric and values_in_range

    @staticmethod
    def check_series_properties(
        series: pd.Series,
        expected_length: int = None,
        allow_nan: bool = True,
        value_range: tuple[float, float] = None,
    ) -> bool:
        """Check basic properties of a pandas Series."""
        if not isinstance(series, pd.Series):
            return False

        if expected_length and len(series) != expected_length:
            return False

        if not allow_nan and series.isna().any():
            return False

        if value_range:
            non_nan_values = series.dropna()
            if len(non_nan_values) > 0:
                if (non_nan_values < value_range[0]).any() or (
                    non_nan_values > value_range[1]
                ).any():
                    return False

        return True


@pytest.fixture
def risk_test_utils():
    """Provide risk testing utilities."""
    return RiskTestUtilities()
