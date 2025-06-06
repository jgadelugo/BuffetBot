"""
Test suite for trend indicators.

Comprehensive tests for Simple Moving Average, Exponential Moving Average,
and Bollinger Bands indicators including edge cases, error handling,
and performance validation.
"""

import time

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.technical.trend import TrendIndicators
from tests.features.conftest import assert_indicator_output_valid


class TestTrendIndicators:
    """Test suite for trend indicators."""

    def test_sma_basic_calculation(self, sample_market_data):
        """Test basic Simple Moving Average calculation."""
        close = sample_market_data["close"]

        sma = TrendIndicators.sma(close)

        assert_indicator_output_valid(
            sma, expected_type=pd.Series, min_non_nan_ratio=0.8
        )

        # SMA should be smoother than original price series
        sma_clean = sma.dropna()
        close_clean = close.loc[sma_clean.index]

        if len(sma_clean) > 1 and len(close_clean) > 1:
            sma_volatility = sma_clean.std()
            close_volatility = close_clean.std()
            assert (
                sma_volatility < close_volatility
            ), "SMA should be less volatile than close prices"

    @pytest.mark.parametrize("period", [5, 10, 20, 50])
    def test_sma_different_periods(self, sample_market_data, period):
        """Test SMA calculation with different periods."""
        close = sample_market_data["close"]

        sma = TrendIndicators.sma(close, period=period)

        assert_indicator_output_valid(sma)

        # Check expected number of NaN values
        nan_count = sma.isna().sum()
        expected_nans = min(period - 1, len(close))
        assert nan_count >= expected_nans

    def test_sma_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of SMA."""
        close = sample_market_data["close"]
        period = 10

        sma = TrendIndicators.sma(close, period=period)

        # For periods where we have enough data, SMA should equal manual calculation
        for i in range(period - 1, min(len(close), period + 5)):  # Test a few points
            manual_sma = close.iloc[i - period + 1 : i + 1].mean()
            if not pd.isna(sma.iloc[i]):
                np.testing.assert_almost_equal(sma.iloc[i], manual_sma, decimal=10)

    def test_ema_basic_calculation(self, sample_market_data):
        """Test basic Exponential Moving Average calculation."""
        close = sample_market_data["close"]

        ema = TrendIndicators.ema(close)

        assert_indicator_output_valid(
            ema,
            expected_type=pd.Series,
            min_non_nan_ratio=0.9,  # EMA should have fewer NaN values than SMA
        )

        # EMA should respond faster to price changes than SMA
        sma = TrendIndicators.sma(close, period=20)
        ema_clean = ema.dropna()
        sma_clean = sma.dropna()

        # Compare volatility on common index
        common_index = ema_clean.index.intersection(sma_clean.index)
        if len(common_index) > 10:
            ema_subset = ema_clean.loc[common_index]
            sma_subset = sma_clean.loc[common_index]

            # EMA should be more responsive (higher volatility)
            ema_volatility = ema_subset.std()
            sma_volatility = sma_subset.std()
            assert ema_volatility >= sma_volatility * 0.9  # Allow some tolerance

    @pytest.mark.parametrize("period", [5, 12, 20, 26])
    def test_ema_different_periods(self, sample_market_data, period):
        """Test EMA calculation with different periods."""
        close = sample_market_data["close"]

        ema = TrendIndicators.ema(close, period=period)

        assert_indicator_output_valid(ema)

        # EMA should have minimal NaN values (only the first one typically)
        nan_count = ema.isna().sum()
        assert nan_count <= 1, f"EMA should have at most 1 NaN value, got {nan_count}"

    def test_ema_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of EMA."""
        close = sample_market_data["close"]
        period = 10

        ema = TrendIndicators.ema(close, period=period)

        # First EMA value should equal the first close price
        first_valid_idx = ema.first_valid_index()
        if first_valid_idx is not None:
            assert abs(ema.loc[first_valid_idx] - close.loc[first_valid_idx]) < 1e-10

        # EMA should converge to close price values in range
        ema_clean = ema.dropna()
        close_clean = close.dropna()

        if len(ema_clean) > 0 and len(close_clean) > 0:
            # EMA values should be within reasonable range of close prices
            close_min, close_max = close_clean.min(), close_clean.max()
            buffer = (close_max - close_min) * 0.1  # 10% buffer

            assert ema_clean.min() >= close_min - buffer
            assert ema_clean.max() <= close_max + buffer

    def test_bollinger_bands_basic_calculation(self, sample_market_data):
        """Test basic Bollinger Bands calculation."""
        close = sample_market_data["close"]

        bb_result = TrendIndicators.bollinger_bands(close)

        # Bollinger Bands returns a dictionary
        assert isinstance(bb_result, dict)
        expected_keys = {"upper", "middle", "lower", "percent_b", "bandwidth"}
        assert set(bb_result.keys()) == expected_keys

        # Validate each component
        for key, series in bb_result.items():
            assert_indicator_output_valid(series, min_non_nan_ratio=0.7)

        # Test mathematical relationships
        upper = bb_result["upper"].dropna()
        middle = bb_result["middle"].dropna()
        lower = bb_result["lower"].dropna()

        # Find common index for all three
        common_index = upper.index.intersection(middle.index).intersection(lower.index)

        if len(common_index) > 0:
            upper_subset = upper.loc[common_index]
            middle_subset = middle.loc[common_index]
            lower_subset = lower.loc[common_index]

            # Upper band should be >= middle >= lower band
            assert (
                upper_subset >= middle_subset
            ).all(), "Upper band should be >= middle band"
            assert (
                middle_subset >= lower_subset
            ).all(), "Middle band should be >= lower band"

            # Middle band should be SMA
            sma = TrendIndicators.sma(close, period=20)
            sma_subset = sma.loc[common_index].dropna()
            if len(sma_subset) > 0:
                common_sma_index = middle_subset.index.intersection(sma_subset.index)
                if len(common_sma_index) > 0:
                    np.testing.assert_array_almost_equal(
                        middle_subset.loc[common_sma_index].values,
                        sma_subset.loc[common_sma_index].values,
                        decimal=10,
                    )

    @pytest.mark.parametrize(
        "period,std_dev",
        [
            (20, 2),  # Default
            (10, 1.5),  # Faster, tighter
            (50, 2.5),  # Slower, wider
        ],
    )
    def test_bollinger_bands_different_parameters(
        self, sample_market_data, period, std_dev
    ):
        """Test Bollinger Bands with different parameters."""
        close = sample_market_data["close"]

        bb_result = TrendIndicators.bollinger_bands(
            close, period=period, std_dev=std_dev
        )

        assert isinstance(bb_result, dict)
        expected_keys = {"upper", "middle", "lower", "percent_b", "bandwidth"}
        assert set(bb_result.keys()) == expected_keys

        for series in bb_result.values():
            assert isinstance(series, pd.Series)
            assert len(series) == len(close)

    def test_bollinger_bands_percent_b_properties(self, sample_market_data):
        """Test Bollinger Bands %B indicator properties."""
        close = sample_market_data["close"]

        bb_result = TrendIndicators.bollinger_bands(close)
        percent_b = bb_result["percent_b"].dropna()

        if len(percent_b) > 0:
            # %B should typically be between -0.2 and 1.2 (allowing for some outliers)
            # Most values should be between 0 and 1
            within_normal_range = ((percent_b >= 0) & (percent_b <= 1)).sum()
            total_values = len(percent_b)

            # At least 70% of values should be in normal range
            normal_ratio = within_normal_range / total_values
            assert (
                normal_ratio >= 0.7
            ), f"Only {normal_ratio:.2%} of %B values in normal range"

    def test_bollinger_bands_bandwidth_properties(self, sample_market_data):
        """Test Bollinger Bands Bandwidth properties."""
        close = sample_market_data["close"]

        bb_result = TrendIndicators.bollinger_bands(close)
        bandwidth = bb_result["bandwidth"].dropna()

        if len(bandwidth) > 0:
            # Bandwidth should always be positive
            assert (bandwidth > 0).all(), "Bandwidth should always be positive"

            # Bandwidth should be reasonable (typically between 0.01 and 0.5 for most stocks)
            reasonable_values = ((bandwidth >= 0.001) & (bandwidth <= 1.0)).sum()
            total_values = len(bandwidth)

            reasonable_ratio = reasonable_values / total_values
            assert (
                reasonable_ratio >= 0.9
            ), f"Only {reasonable_ratio:.2%} of bandwidth values reasonable"

    def test_edge_cases_insufficient_data(self, insufficient_data):
        """Test trend indicators with insufficient data."""
        close = insufficient_data["close"]

        # SMA with period longer than data
        sma = TrendIndicators.sma(close, period=20)
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(close)
        # Should be mostly NaN

        # EMA should handle better
        ema = TrendIndicators.ema(close, period=20)
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(close)

        # Bollinger Bands
        bb = TrendIndicators.bollinger_bands(close, period=20)
        assert isinstance(bb, dict)

    def test_with_nan_values(self, data_with_nans):
        """Test trend indicators robustness with NaN values."""
        close = data_with_nans["close"]

        # SMA should handle NaN values
        sma = TrendIndicators.sma(close)
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(close)

        # Should have some valid values
        valid_count = sma.notna().sum()
        assert (
            valid_count > 0
        ), "SMA should produce some valid values despite NaN inputs"

        # EMA should handle NaN values
        ema = TrendIndicators.ema(close)
        assert isinstance(ema, pd.Series)
        valid_ema_count = ema.notna().sum()
        assert (
            valid_ema_count > 0
        ), "EMA should produce some valid values despite NaN inputs"

        # Bollinger Bands should handle NaN values
        bb = TrendIndicators.bollinger_bands(close)
        assert isinstance(bb, dict)

        for key, series in bb.items():
            valid_bb_count = series.notna().sum()
            # Allow for some components to have no valid values if input is too sparse
            # but at least some should work

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        # Empty series
        empty_series = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            TrendIndicators.sma(empty_series)

        # Non-numeric data
        text_series = pd.Series(["a", "b", "c"])

        with pytest.raises((ValueError, TypeError)):
            TrendIndicators.sma(text_series)

        # Invalid periods
        valid_close = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            TrendIndicators.sma(valid_close, period=0)

        with pytest.raises(ValueError):
            TrendIndicators.sma(valid_close, period=-1)

        # Invalid standard deviation for Bollinger Bands
        with pytest.raises(ValueError):
            TrendIndicators.bollinger_bands(valid_close, std_dev=0)

    def test_performance_benchmark(self, performance_data):
        """Test performance with larger dataset."""
        close = performance_data["close"]

        # Test SMA performance
        start_time = time.time()
        sma = TrendIndicators.sma(close, period=20)
        sma_time = time.time() - start_time

        assert sma_time < 0.5, f"SMA took too long: {sma_time:.3f}s"
        assert_indicator_output_valid(sma)

        # Test EMA performance
        start_time = time.time()
        ema = TrendIndicators.ema(close, period=20)
        ema_time = time.time() - start_time

        assert ema_time < 0.5, f"EMA took too long: {ema_time:.3f}s"
        assert_indicator_output_valid(ema)

        # Test Bollinger Bands performance
        start_time = time.time()
        bb = TrendIndicators.bollinger_bands(close)
        bb_time = time.time() - start_time

        assert bb_time < 1.0, f"Bollinger Bands took too long: {bb_time:.3f}s"
        assert isinstance(bb, dict)

    def test_consistency_across_runs(self, sample_market_data):
        """Test that indicators produce consistent results across runs."""
        close = sample_market_data["close"]

        # SMA consistency
        sma1 = TrendIndicators.sma(close, period=20)
        sma2 = TrendIndicators.sma(close, period=20)
        pd.testing.assert_series_equal(sma1, sma2)

        # EMA consistency
        ema1 = TrendIndicators.ema(close, period=20)
        ema2 = TrendIndicators.ema(close, period=20)
        pd.testing.assert_series_equal(ema1, ema2)

        # Bollinger Bands consistency
        bb1 = TrendIndicators.bollinger_bands(close)
        bb2 = TrendIndicators.bollinger_bands(close)

        for key in bb1.keys():
            pd.testing.assert_series_equal(bb1[key], bb2[key])

    def test_trend_comparison(self, sample_market_data):
        """Test comparison between different trend indicators."""
        close = sample_market_data["close"]

        # Compare different MA periods
        sma_fast = TrendIndicators.sma(close, period=10)
        sma_slow = TrendIndicators.sma(close, period=20)

        # Fast MA should be more responsive
        sma_fast_clean = sma_fast.dropna()
        sma_slow_clean = sma_slow.dropna()

        if len(sma_fast_clean) > 1 and len(sma_slow_clean) > 1:
            # Fast MA should have higher volatility
            fast_volatility = sma_fast_clean.std()
            slow_volatility = sma_slow_clean.std()
            assert fast_volatility >= slow_volatility * 0.95  # Allow small tolerance

        # Compare EMA vs SMA - EMA should be more responsive
        ema = TrendIndicators.ema(close, period=20)
        sma = TrendIndicators.sma(close, period=20)

        ema_clean = ema.dropna()
        sma_clean = sma.dropna()

        if len(ema_clean) > 1 and len(sma_clean) > 1:
            # EMA should generally be more volatile than SMA
            ema_volatility = ema_clean.std()
            sma_volatility = sma_clean.std()
            assert ema_volatility >= sma_volatility * 0.9  # Allow tolerance
