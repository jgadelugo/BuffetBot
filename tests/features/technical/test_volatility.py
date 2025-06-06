"""
Test suite for volatility indicators.

Comprehensive tests for Average True Range, Historical Volatility,
Volatility Ratio, True Range, Normalized ATR, and Volatility Bands.
"""

import time

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.technical.volatility import VolatilityIndicators
from tests.features.conftest import assert_indicator_output_valid


class TestVolatilityIndicators:
    """Test suite for volatility indicators."""

    def test_atr_basic_calculation(self, sample_market_data):
        """Test basic Average True Range calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        atr = VolatilityIndicators.atr(high, low, close)

        assert_indicator_output_valid(
            atr,
            expected_type=pd.Series,
            expected_range=(0, float("inf")),
            min_non_nan_ratio=0.8,
        )

        # ATR should be reasonable relative to price range
        atr_clean = atr.dropna()
        close_clean = close.dropna()

        if len(atr_clean) > 0 and len(close_clean) > 0:
            typical_price = close_clean.median()
            median_atr = atr_clean.median()

            # ATR should typically be a small percentage of price
            atr_percentage = median_atr / typical_price
            assert (
                0.001 <= atr_percentage <= 0.2
            ), f"ATR percentage {atr_percentage:.4f} seems unreasonable"

    @pytest.mark.parametrize("period", [7, 14, 21, 30])
    def test_atr_different_periods(self, sample_market_data, period):
        """Test ATR calculation with different periods."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        atr = VolatilityIndicators.atr(high, low, close, period=period)

        assert_indicator_output_valid(atr, expected_range=(0, float("inf")))

        # Check expected number of NaN values
        nan_count = atr.isna().sum()
        expected_nans = min(period, len(close))
        assert nan_count >= expected_nans - 1

    def test_historical_volatility_basic_calculation(self, sample_market_data):
        """Test basic Historical Volatility calculation."""
        close = sample_market_data["close"]

        hist_vol = VolatilityIndicators.historical_volatility(close)

        assert_indicator_output_valid(
            hist_vol,
            expected_type=pd.Series,
            expected_range=(0, float("inf")),
            min_non_nan_ratio=0.7,
        )

        # Historical volatility should be reasonable (typically 5-100% annualized)
        hist_vol_clean = hist_vol.dropna()
        if len(hist_vol_clean) > 0:
            reasonable_values = ((hist_vol_clean >= 1) & (hist_vol_clean <= 200)).sum()
            total_values = len(hist_vol_clean)

            reasonable_ratio = reasonable_values / total_values
            assert (
                reasonable_ratio >= 0.8
            ), f"Only {reasonable_ratio:.2%} of volatility values reasonable"

    @pytest.mark.parametrize("period", [10, 20, 30, 60])
    def test_historical_volatility_different_periods(self, sample_market_data, period):
        """Test Historical Volatility with different periods."""
        close = sample_market_data["close"]

        hist_vol = VolatilityIndicators.historical_volatility(close, period=period)

        assert_indicator_output_valid(hist_vol, expected_range=(0, float("inf")))

    def test_volatility_ratio_basic_calculation(self, sample_market_data):
        """Test basic Volatility Ratio calculation."""
        close = sample_market_data["close"]

        vol_ratio = VolatilityIndicators.volatility_ratio(close)

        assert_indicator_output_valid(
            vol_ratio,
            expected_type=pd.Series,
            expected_range=(0, float("inf")),
            min_non_nan_ratio=0.6,
        )

        # Volatility ratio should typically be around 1.0 (can vary significantly)
        vol_ratio_clean = vol_ratio.dropna()
        if len(vol_ratio_clean) > 0:
            # Most values should be between 0.1 and 10
            reasonable_values = (
                (vol_ratio_clean >= 0.1) & (vol_ratio_clean <= 10)
            ).sum()
            total_values = len(vol_ratio_clean)

            reasonable_ratio = reasonable_values / total_values
            assert (
                reasonable_ratio >= 0.7
            ), f"Only {reasonable_ratio:.2%} of volatility ratio values reasonable"

    def test_true_range_basic_calculation(self, sample_market_data):
        """Test basic True Range calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        tr = VolatilityIndicators.true_range(high, low, close)

        assert_indicator_output_valid(
            tr,
            expected_type=pd.Series,
            expected_range=(0, float("inf")),
            min_non_nan_ratio=0.9,
        )

        # True Range should always be >= High - Low
        tr_clean = tr.dropna()
        high_subset = high.loc[tr_clean.index]
        low_subset = low.loc[tr_clean.index]

        daily_range = high_subset - low_subset

        # TR should be at least as large as daily range
        assert (
            tr_clean >= daily_range - 1e-10
        ).all(), "True Range should be >= High - Low"

    def test_true_range_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of True Range."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        tr = VolatilityIndicators.true_range(high, low, close)

        # Test manual calculation for first few points
        for i in range(1, min(5, len(tr))):
            if not pd.isna(tr.iloc[i]):
                h, l, c_prev, c_curr = (
                    high.iloc[i],
                    low.iloc[i],
                    close.iloc[i - 1],
                    close.iloc[i],
                )

                range1 = h - l
                range2 = abs(h - c_prev)
                range3 = abs(l - c_prev)

                expected_tr = max(range1, range2, range3)
                np.testing.assert_almost_equal(tr.iloc[i], expected_tr, decimal=10)

    def test_normalized_atr_basic_calculation(self, sample_market_data):
        """Test basic Normalized ATR calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        natr = VolatilityIndicators.normalized_atr(high, low, close)

        assert_indicator_output_valid(
            natr,
            expected_type=pd.Series,
            expected_range=(0, float("inf")),
            min_non_nan_ratio=0.8,
        )

        # Normalized ATR should be in percentage terms (typically 0-20%)
        natr_clean = natr.dropna()
        if len(natr_clean) > 0:
            reasonable_values = ((natr_clean >= 0) & (natr_clean <= 50)).sum()
            total_values = len(natr_clean)

            reasonable_ratio = reasonable_values / total_values
            assert (
                reasonable_ratio >= 0.9
            ), f"Only {reasonable_ratio:.2%} of NATR values reasonable"

    def test_volatility_bands_basic_calculation(self, sample_market_data):
        """Test basic Volatility Bands calculation."""
        close = sample_market_data["close"]

        vol_bands = VolatilityIndicators.volatility_bands(close)

        # Volatility Bands returns a dictionary
        assert isinstance(vol_bands, dict)
        expected_keys = {"upper", "middle", "lower"}
        assert set(vol_bands.keys()) == expected_keys

        # Validate each component
        for key, series in vol_bands.items():
            assert_indicator_output_valid(series, min_non_nan_ratio=0.7)

        # Test mathematical relationships
        upper = vol_bands["upper"].dropna()
        middle = vol_bands["middle"].dropna()
        lower = vol_bands["lower"].dropna()

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

    def test_edge_cases_insufficient_data(self, insufficient_data):
        """Test volatility indicators with insufficient data."""
        high = insufficient_data["high"]
        low = insufficient_data["low"]
        close = insufficient_data["close"]

        # ATR with period longer than data
        atr = VolatilityIndicators.atr(high, low, close, period=20)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)

        # True Range should work with minimal data
        tr = VolatilityIndicators.true_range(high, low, close)
        assert isinstance(tr, pd.Series)
        assert len(tr) == len(close)

        # Historical Volatility
        hist_vol = VolatilityIndicators.historical_volatility(close, period=20)
        assert isinstance(hist_vol, pd.Series)

    def test_with_nan_values(self, data_with_nans):
        """Test volatility indicators robustness with NaN values."""
        high = data_with_nans["high"]
        low = data_with_nans["low"]
        close = data_with_nans["close"]

        # ATR should handle NaN values
        atr = VolatilityIndicators.atr(high, low, close)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)

        # Should have some valid values
        valid_count = atr.notna().sum()
        assert (
            valid_count > 0
        ), "ATR should produce some valid values despite NaN inputs"

        # Historical volatility should handle NaN values
        hist_vol = VolatilityIndicators.historical_volatility(close)
        assert isinstance(hist_vol, pd.Series)
        valid_hist_vol_count = hist_vol.notna().sum()
        # May have no valid values if too many NaNs, but shouldn't crash

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        # Empty series
        empty_series = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            VolatilityIndicators.atr(empty_series, empty_series, empty_series)

        # Non-numeric data
        text_series = pd.Series(["a", "b", "c"])

        with pytest.raises((ValueError, TypeError)):
            VolatilityIndicators.atr(text_series, text_series, text_series)

        # Invalid periods
        valid_data = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            VolatilityIndicators.atr(valid_data, valid_data, valid_data, period=0)

        with pytest.raises(ValueError):
            VolatilityIndicators.atr(valid_data, valid_data, valid_data, period=-1)

    def test_performance_benchmark(self, performance_data):
        """Test performance with larger dataset."""
        high = performance_data["high"]
        low = performance_data["low"]
        close = performance_data["close"]

        # Test ATR performance
        start_time = time.time()
        atr = VolatilityIndicators.atr(high, low, close)
        atr_time = time.time() - start_time

        assert atr_time < 0.5, f"ATR took too long: {atr_time:.3f}s"
        assert_indicator_output_valid(atr, expected_range=(0, float("inf")))

        # Test Historical Volatility performance
        start_time = time.time()
        hist_vol = VolatilityIndicators.historical_volatility(close)
        hist_vol_time = time.time() - start_time

        assert (
            hist_vol_time < 1.0
        ), f"Historical Volatility took too long: {hist_vol_time:.3f}s"
        assert_indicator_output_valid(hist_vol, expected_range=(0, float("inf")))

        # Test True Range performance
        start_time = time.time()
        tr = VolatilityIndicators.true_range(high, low, close)
        tr_time = time.time() - start_time

        assert tr_time < 0.5, f"True Range took too long: {tr_time:.3f}s"
        assert_indicator_output_valid(tr, expected_range=(0, float("inf")))

    def test_consistency_across_runs(self, sample_market_data):
        """Test that indicators produce consistent results across runs."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        # ATR consistency
        atr1 = VolatilityIndicators.atr(high, low, close)
        atr2 = VolatilityIndicators.atr(high, low, close)
        pd.testing.assert_series_equal(atr1, atr2)

        # Historical Volatility consistency
        hist_vol1 = VolatilityIndicators.historical_volatility(close)
        hist_vol2 = VolatilityIndicators.historical_volatility(close)
        pd.testing.assert_series_equal(hist_vol1, hist_vol2)

        # True Range consistency
        tr1 = VolatilityIndicators.true_range(high, low, close)
        tr2 = VolatilityIndicators.true_range(high, low, close)
        pd.testing.assert_series_equal(tr1, tr2)

    def test_volatility_relationships(self, sample_market_data):
        """Test relationships between different volatility indicators."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        # ATR should be smoother than True Range
        atr = VolatilityIndicators.atr(high, low, close, period=14)
        tr = VolatilityIndicators.true_range(high, low, close)

        atr_clean = atr.dropna()
        tr_subset = tr.loc[atr_clean.index]

        if len(atr_clean) > 1 and len(tr_subset) > 1:
            atr_volatility = atr_clean.std()
            tr_volatility = tr_subset.std()
            assert (
                atr_volatility < tr_volatility
            ), "ATR should be smoother than True Range"

        # Normalized ATR should be proportional to ATR
        natr = VolatilityIndicators.normalized_atr(high, low, close)
        natr_clean = natr.dropna()
        atr_subset = atr.loc[natr_clean.index]
        close_subset = close.loc[natr_clean.index]

        # NATR should approximately equal (ATR / Close) * 100
        if len(natr_clean) > 0 and len(atr_subset) > 0:
            common_index = natr_clean.index.intersection(atr_subset.index).intersection(
                close_subset.index
            )
            if len(common_index) > 0:
                expected_natr = (
                    atr_subset.loc[common_index] / close_subset.loc[common_index]
                ) * 100
                actual_natr = natr_clean.loc[common_index]

                # Allow for some numerical differences
                np.testing.assert_array_almost_equal(
                    actual_natr.values, expected_natr.values, decimal=6
                )
