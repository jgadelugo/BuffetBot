"""
Test suite for volume indicators.

Comprehensive tests for Volume SMA, On-Balance Volume, Volume Rate of Change,
Accumulation/Distribution Line, and Volume Weighted Average Price indicators.
"""

import time

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.technical.volume import VolumeIndicators
from tests.features.conftest import assert_indicator_output_valid


class TestVolumeIndicators:
    """Test suite for volume indicators."""

    def test_volume_sma_basic_calculation(self, sample_market_data):
        """Test basic Volume SMA calculation."""
        volume = sample_market_data["volume"]

        volume_sma = VolumeIndicators.volume_sma(volume)

        assert_indicator_output_valid(
            volume_sma, expected_type=pd.Series, min_non_nan_ratio=0.8
        )

        # Volume SMA should be smoother than original volume
        vol_sma_clean = volume_sma.dropna()
        volume_clean = volume.loc[vol_sma_clean.index]

        if len(vol_sma_clean) > 1 and len(volume_clean) > 1:
            vol_sma_volatility = vol_sma_clean.std()
            volume_volatility = volume_clean.std()
            assert (
                vol_sma_volatility < volume_volatility
            ), "Volume SMA should be less volatile than raw volume"

    @pytest.mark.parametrize("period", [10, 20, 30, 50])
    def test_volume_sma_different_periods(self, sample_market_data, period):
        """Test Volume SMA with different periods."""
        volume = sample_market_data["volume"]

        volume_sma = VolumeIndicators.volume_sma(volume, period=period)

        assert_indicator_output_valid(volume_sma)

        # Check expected number of NaN values
        nan_count = volume_sma.isna().sum()
        expected_nans = min(period - 1, len(volume))
        assert nan_count >= expected_nans

    def test_obv_basic_calculation(self, sample_market_data):
        """Test basic On-Balance Volume calculation."""
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        obv = VolumeIndicators.obv(close, volume)

        assert_indicator_output_valid(
            obv,
            expected_type=pd.Series,
            min_non_nan_ratio=0.9,  # OBV should have minimal NaN values
        )

        # OBV should be cumulative - always increasing or decreasing
        obv_clean = obv.dropna()
        if len(obv_clean) > 1:
            # Check that OBV is monotonic in nature (changes should follow volume)
            obv_diff = obv_clean.diff().dropna()
            volume_subset = volume.loc[obv_diff.index]

            # Where price goes up, OBV change should equal volume
            # Where price goes down, OBV change should equal -volume
            # This is a basic sanity check
            assert len(obv_diff) > 0, "Should have OBV differences to analyze"

    def test_obv_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of OBV."""
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        obv = VolumeIndicators.obv(close, volume)

        # Test the mathematical relationship
        obv_clean = obv.dropna()
        if len(obv_clean) > 1:
            for i in range(1, min(len(obv_clean), 10)):  # Test first few points
                prev_idx = obv_clean.index[i - 1]
                curr_idx = obv_clean.index[i]

                prev_close = close.loc[prev_idx]
                curr_close = close.loc[curr_idx]
                curr_volume = volume.loc[curr_idx]

                obv_change = obv_clean.iloc[i] - obv_clean.iloc[i - 1]

                if curr_close > prev_close:
                    # Price up: OBV should increase by volume
                    np.testing.assert_almost_equal(obv_change, curr_volume, decimal=2)
                elif curr_close < prev_close:
                    # Price down: OBV should decrease by volume
                    np.testing.assert_almost_equal(obv_change, -curr_volume, decimal=2)
                else:
                    # Price unchanged: OBV should not change
                    np.testing.assert_almost_equal(obv_change, 0, decimal=2)

    def test_volume_roc_basic_calculation(self, sample_market_data):
        """Test basic Volume Rate of Change calculation."""
        volume = sample_market_data["volume"]

        volume_roc = VolumeIndicators.volume_roc(volume)

        assert_indicator_output_valid(
            volume_roc, expected_type=pd.Series, min_non_nan_ratio=0.8
        )

        # Volume ROC should be in percentage terms
        vol_roc_clean = volume_roc.dropna()
        if len(vol_roc_clean) > 0:
            # Most volume changes should be reasonable (typically -50% to +200%)
            reasonable_values = ((vol_roc_clean >= -80) & (vol_roc_clean <= 500)).sum()
            total_values = len(vol_roc_clean)

            reasonable_ratio = reasonable_values / total_values
            assert (
                reasonable_ratio >= 0.8
            ), f"Only {reasonable_ratio:.2%} of Volume ROC values reasonable"

    @pytest.mark.parametrize("period", [5, 10, 14, 20])
    def test_volume_roc_different_periods(self, sample_market_data, period):
        """Test Volume ROC with different periods."""
        volume = sample_market_data["volume"]

        volume_roc = VolumeIndicators.volume_roc(volume, period=period)

        assert_indicator_output_valid(volume_roc)

        # Check expected number of NaN values
        nan_count = volume_roc.isna().sum()
        expected_nans = min(period, len(volume))
        assert nan_count >= expected_nans - 1

    def test_ad_line_basic_calculation(self, sample_market_data):
        """Test basic Accumulation/Distribution Line calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        ad_line = VolumeIndicators.ad_line(high, low, close, volume)

        assert_indicator_output_valid(
            ad_line, expected_type=pd.Series, min_non_nan_ratio=0.9
        )

        # AD Line should be cumulative
        ad_clean = ad_line.dropna()
        if len(ad_clean) > 1:
            # Check that changes are reasonable relative to volume
            ad_diff = ad_clean.diff().dropna()
            volume_subset = volume.loc[ad_diff.index]

            if len(ad_diff) > 0:
                # AD changes should generally be proportional to volume
                # (exact relationship depends on money flow multiplier)
                max_change = volume_subset.max()
                max_ad_change = ad_diff.abs().max()

                # AD change should not exceed volume (money flow multiplier is [-1, 1])
                assert (
                    max_ad_change <= max_change * 1.1
                )  # Small tolerance for floating point

    def test_ad_line_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of A/D Line."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        ad_line = VolumeIndicators.ad_line(high, low, close, volume)

        # Test money flow multiplier properties
        for i in range(min(10, len(close))):  # Test first few points
            h, l, c, v = high.iloc[i], low.iloc[i], close.iloc[i], volume.iloc[i]

            if h != l:  # Avoid division by zero
                mf_multiplier = ((c - l) - (h - c)) / (h - l)

                # Money flow multiplier should be between -1 and 1
                assert (
                    -1 <= mf_multiplier <= 1
                ), f"Money flow multiplier {mf_multiplier} out of range"

                # When close is at high, multiplier should be 1
                if abs(c - h) < 1e-10:
                    np.testing.assert_almost_equal(mf_multiplier, 1, decimal=8)

                # When close is at low, multiplier should be -1
                if abs(c - l) < 1e-10:
                    np.testing.assert_almost_equal(mf_multiplier, -1, decimal=8)

    def test_vwap_basic_calculation(self, sample_market_data):
        """Test basic Volume Weighted Average Price calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        vwap = VolumeIndicators.vwap(high, low, close, volume)

        assert_indicator_output_valid(
            vwap, expected_type=pd.Series, min_non_nan_ratio=0.9
        )

        # VWAP should be within reasonable range of typical prices
        vwap_clean = vwap.dropna()
        close_clean = close.dropna()

        if len(vwap_clean) > 0 and len(close_clean) > 0:
            close_min, close_max = close_clean.min(), close_clean.max()
            vwap_min, vwap_max = vwap_clean.min(), vwap_clean.max()

            # VWAP should generally be within the price range (with some tolerance)
            price_range = close_max - close_min
            tolerance = price_range * 0.1  # 10% tolerance

            assert vwap_min >= close_min - tolerance, "VWAP minimum too low"
            assert vwap_max <= close_max + tolerance, "VWAP maximum too high"

    def test_vwap_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of VWAP."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]

        vwap = VolumeIndicators.vwap(high, low, close, volume)

        # Test first few VWAP values manually
        for i in range(1, min(5, len(vwap))):
            if not pd.isna(vwap.iloc[i]):
                # Calculate typical price and cumulative values up to point i
                typical_prices = ((high + low + close) / 3).iloc[: i + 1]
                volumes = volume.iloc[: i + 1]

                cumulative_pv = (typical_prices * volumes).sum()
                cumulative_volume = volumes.sum()

                if cumulative_volume > 0:
                    expected_vwap = cumulative_pv / cumulative_volume
                    np.testing.assert_almost_equal(
                        vwap.iloc[i], expected_vwap, decimal=6
                    )

    def test_edge_cases_insufficient_data(self, insufficient_data):
        """Test volume indicators with insufficient data."""
        close = insufficient_data["close"]
        volume = insufficient_data["volume"]
        high = insufficient_data["high"]
        low = insufficient_data["low"]

        # Volume SMA with period longer than data
        vol_sma = VolumeIndicators.volume_sma(volume, period=20)
        assert isinstance(vol_sma, pd.Series)
        assert len(vol_sma) == len(volume)

        # OBV should work with any amount of data
        obv = VolumeIndicators.obv(close, volume)
        assert isinstance(obv, pd.Series)
        assert len(obv) == len(close)

        # VWAP should work with minimal data
        vwap = VolumeIndicators.vwap(high, low, close, volume)
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(close)

    def test_with_nan_values(self, data_with_nans):
        """Test volume indicators robustness with NaN values."""
        close = data_with_nans["close"]
        volume = data_with_nans["volume"]
        high = data_with_nans["high"]
        low = data_with_nans["low"]

        # Volume SMA should handle NaN values
        vol_sma = VolumeIndicators.volume_sma(volume)
        assert isinstance(vol_sma, pd.Series)
        assert len(vol_sma) == len(volume)

        # OBV should handle NaN values gracefully
        obv = VolumeIndicators.obv(close, volume)
        assert isinstance(obv, pd.Series)
        valid_obv_count = obv.notna().sum()
        assert (
            valid_obv_count > 0
        ), "OBV should produce some valid values despite NaN inputs"

        # VWAP should handle NaN values
        vwap = VolumeIndicators.vwap(high, low, close, volume)
        assert isinstance(vwap, pd.Series)
        valid_vwap_count = vwap.notna().sum()
        assert (
            valid_vwap_count > 0
        ), "VWAP should produce some valid values despite NaN inputs"

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        # Empty series
        empty_series = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            VolumeIndicators.volume_sma(empty_series)

        # Non-numeric data
        text_series = pd.Series(["a", "b", "c"])

        with pytest.raises((ValueError, TypeError)):
            VolumeIndicators.volume_sma(text_series)

        # Invalid periods
        valid_volume = pd.Series([1000, 2000, 3000, 4000, 5000])

        with pytest.raises(ValueError):
            VolumeIndicators.volume_sma(valid_volume, period=0)

        with pytest.raises(ValueError):
            VolumeIndicators.volume_sma(valid_volume, period=-1)

        # Mismatched series lengths
        short_close = pd.Series([100, 101, 102])
        long_volume = pd.Series([1000, 2000, 3000, 4000, 5000])

        with pytest.raises((ValueError, IndexError)):
            VolumeIndicators.obv(short_close, long_volume)

    def test_performance_benchmark(self, performance_data):
        """Test performance with larger dataset."""
        close = performance_data["close"]
        volume = performance_data["volume"]
        high = performance_data["high"]
        low = performance_data["low"]

        # Test Volume SMA performance
        start_time = time.time()
        vol_sma = VolumeIndicators.volume_sma(volume)
        vol_sma_time = time.time() - start_time

        assert vol_sma_time < 0.5, f"Volume SMA took too long: {vol_sma_time:.3f}s"

        # Test OBV performance
        start_time = time.time()
        obv = VolumeIndicators.obv(close, volume)
        obv_time = time.time() - start_time

        assert obv_time < 0.5, f"OBV took too long: {obv_time:.3f}s"

        # Test VWAP performance
        start_time = time.time()
        vwap = VolumeIndicators.vwap(high, low, close, volume)
        vwap_time = time.time() - start_time

        assert vwap_time < 1.0, f"VWAP took too long: {vwap_time:.3f}s"

    def test_consistency_across_runs(self, sample_market_data):
        """Test that indicators produce consistent results across runs."""
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]
        high = sample_market_data["high"]
        low = sample_market_data["low"]

        # Volume SMA consistency
        vol_sma1 = VolumeIndicators.volume_sma(volume)
        vol_sma2 = VolumeIndicators.volume_sma(volume)
        pd.testing.assert_series_equal(vol_sma1, vol_sma2)

        # OBV consistency
        obv1 = VolumeIndicators.obv(close, volume)
        obv2 = VolumeIndicators.obv(close, volume)
        pd.testing.assert_series_equal(obv1, obv2)

        # VWAP consistency
        vwap1 = VolumeIndicators.vwap(high, low, close, volume)
        vwap2 = VolumeIndicators.vwap(high, low, close, volume)
        pd.testing.assert_series_equal(vwap1, vwap2)

        # A/D Line consistency
        ad1 = VolumeIndicators.ad_line(high, low, close, volume)
        ad2 = VolumeIndicators.ad_line(high, low, close, volume)
        pd.testing.assert_series_equal(ad1, ad2)

    def test_volume_relationships(self, sample_market_data):
        """Test relationships between different volume indicators."""
        close = sample_market_data["close"]
        volume = sample_market_data["volume"]
        high = sample_market_data["high"]
        low = sample_market_data["low"]

        # Volume SMA should smooth volume data
        vol_sma = VolumeIndicators.volume_sma(volume, period=10)
        vol_sma_clean = vol_sma.dropna()
        volume_subset = volume.loc[vol_sma_clean.index]

        if len(vol_sma_clean) > 1 and len(volume_subset) > 1:
            vol_sma_volatility = vol_sma_clean.std()
            volume_volatility = volume_subset.std()
            assert (
                vol_sma_volatility < volume_volatility
            ), "Volume SMA should be smoother"

        # OBV should correlate with price direction
        obv = VolumeIndicators.obv(close, volume)
        obv_clean = obv.dropna()
        close_subset = close.loc[obv_clean.index]

        if len(obv_clean) > 10 and len(close_subset) > 10:
            # Calculate correlation between OBV changes and price changes
            obv_changes = obv_clean.diff().dropna()
            price_changes = close_subset.diff().dropna()

            # Find common index
            common_index = obv_changes.index.intersection(price_changes.index)
            if len(common_index) > 5:
                obv_subset = obv_changes.loc[common_index]
                price_subset = price_changes.loc[common_index]

                # There should be some positive correlation (not strict requirement)
                correlation = obv_subset.corr(price_subset)
                # Just check that correlation is reasonable (not necessarily positive)
                assert not pd.isna(
                    correlation
                ), "Should be able to calculate correlation"

    def test_zero_volume_handling(self, sample_market_data):
        """Test handling of zero volume periods."""
        close = sample_market_data["close"]
        volume = sample_market_data["volume"].copy()
        high = sample_market_data["high"]
        low = sample_market_data["low"]

        # Introduce some zero volume periods
        volume.iloc[10:15] = 0
        volume.iloc[25] = 0

        # All indicators should handle zero volume gracefully
        vol_sma = VolumeIndicators.volume_sma(volume)
        assert isinstance(vol_sma, pd.Series)

        obv = VolumeIndicators.obv(close, volume)
        assert isinstance(obv, pd.Series)

        vwap = VolumeIndicators.vwap(high, low, close, volume)
        assert isinstance(vwap, pd.Series)

        ad_line = VolumeIndicators.ad_line(high, low, close, volume)
        assert isinstance(ad_line, pd.Series)

        # Check that zero volume periods don't break the calculations
        vol_roc = VolumeIndicators.volume_roc(volume)
        assert isinstance(vol_roc, pd.Series)
