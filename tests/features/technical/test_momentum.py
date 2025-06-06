"""
Test suite for momentum indicators.

Comprehensive tests for RSI, MACD, Stochastic Oscillator, Williams %R,
and Price Momentum indicators including edge cases, error handling,
and performance validation.
"""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.technical.momentum import MomentumIndicators
from tests.features.conftest import assert_indicator_output_valid


class TestMomentumIndicators:
    """Test suite for momentum indicators."""

    def test_rsi_basic_calculation(self, sample_market_data):
        """Test basic RSI calculation with valid data."""
        close = sample_market_data["close"]

        # Test default period
        rsi = MomentumIndicators.rsi(close)

        # Validate output
        assert_indicator_output_valid(
            rsi,
            expected_type=pd.Series,
            expected_range=(0, 100),
            min_non_nan_ratio=0.8,  # RSI needs 14+ periods
        )

        # Check that RSI values are reasonable
        rsi_clean = rsi.dropna()
        assert len(rsi_clean) > 0, "No valid RSI values calculated"
        assert (
            20 <= rsi_clean.median() <= 80
        ), "RSI median should be in reasonable range"

    @pytest.mark.parametrize("period", [7, 14, 21, 30])
    def test_rsi_different_periods(self, sample_market_data, period):
        """Test RSI calculation with different periods."""
        close = sample_market_data["close"]

        rsi = MomentumIndicators.rsi(close, period=period)

        assert_indicator_output_valid(rsi, expected_range=(0, 100))

        # Check that we get fewer NaN values with shorter periods
        nan_count = rsi.isna().sum()
        expected_nans = min(period, len(close))  # At least 'period' NaNs expected
        assert nan_count >= expected_nans - 1  # Allow for slight variation

    def test_rsi_edge_cases(self, insufficient_data):
        """Test RSI behavior with insufficient data."""
        close = insufficient_data["close"]

        # Should handle gracefully but may return mostly NaN
        rsi = MomentumIndicators.rsi(close)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(close)

    def test_rsi_with_nans(self, data_with_nans):
        """Test RSI robustness with NaN values in input."""
        close = data_with_nans["close"]

        rsi = MomentumIndicators.rsi(close)

        # Should still produce valid output despite NaN inputs
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(close)

        # Should have some valid values
        valid_count = rsi.notna().sum()
        assert (
            valid_count > 0
        ), "RSI should produce some valid values despite NaN inputs"

    def test_macd_basic_calculation(self, sample_market_data):
        """Test basic MACD calculation."""
        close = sample_market_data["close"]

        macd_result = MomentumIndicators.macd(close)

        # MACD returns a dictionary
        assert isinstance(macd_result, dict)
        expected_keys = {"macd", "signal", "histogram"}
        assert set(macd_result.keys()) == expected_keys

        # Validate each component
        for key, series in macd_result.items():
            assert_indicator_output_valid(series, min_non_nan_ratio=0.6)

        # Test relationships between components
        macd_clean = macd_result["macd"].dropna()
        signal_clean = macd_result["signal"].dropna()
        histogram_clean = macd_result["histogram"].dropna()

        if len(macd_clean) > 0 and len(signal_clean) > 0 and len(histogram_clean) > 0:
            # Histogram should approximately equal MACD - Signal
            common_index = macd_clean.index.intersection(
                signal_clean.index
            ).intersection(histogram_clean.index)
            if len(common_index) > 0:
                calculated_histogram = (
                    macd_clean.loc[common_index] - signal_clean.loc[common_index]
                )
                actual_histogram = histogram_clean.loc[common_index]

                # Allow for small numerical differences
                np.testing.assert_array_almost_equal(
                    calculated_histogram.values, actual_histogram.values, decimal=6
                )

    @pytest.mark.parametrize(
        "fast_period,slow_period,signal_period",
        [
            (12, 26, 9),  # Default
            (8, 21, 5),  # Faster
            (15, 30, 12),  # Slower
        ],
    )
    def test_macd_different_parameters(
        self, sample_market_data, fast_period, slow_period, signal_period
    ):
        """Test MACD with different parameter combinations."""
        close = sample_market_data["close"]

        macd_result = MomentumIndicators.macd(
            close,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

        assert isinstance(macd_result, dict)
        assert set(macd_result.keys()) == {"macd", "signal", "histogram"}

        for series in macd_result.values():
            assert isinstance(series, pd.Series)
            assert len(series) == len(close)

    def test_stochastic_basic_calculation(self, sample_market_data):
        """Test basic Stochastic Oscillator calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        stoch_result = MomentumIndicators.stochastic(high, low, close)

        # Stochastic returns a dictionary
        assert isinstance(stoch_result, dict)
        expected_keys = {"%K", "%D"}
        assert set(stoch_result.keys()) == expected_keys

        # Validate each component
        for key, series in stoch_result.items():
            assert_indicator_output_valid(
                series, expected_range=(0, 100), min_non_nan_ratio=0.7
            )

        # %D should be smoother than %K (less volatile)
        k_values = stoch_result["%K"].dropna()
        d_values = stoch_result["%D"].dropna()

        if len(k_values) > 1 and len(d_values) > 1:
            k_volatility = k_values.std()
            d_volatility = d_values.std()
            assert d_volatility <= k_volatility * 1.1  # %D should be less volatile

    @pytest.mark.parametrize(
        "k_period,d_period",
        [
            (14, 3),  # Default
            (5, 3),  # Faster
            (21, 5),  # Slower
        ],
    )
    def test_stochastic_different_periods(self, sample_market_data, k_period, d_period):
        """Test Stochastic with different periods."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        stoch_result = MomentumIndicators.stochastic(
            high, low, close, k_period=k_period, d_period=d_period
        )

        assert isinstance(stoch_result, dict)
        for series in stoch_result.values():
            assert_indicator_output_valid(series, expected_range=(0, 100))

    def test_williams_r_basic_calculation(self, sample_market_data):
        """Test basic Williams %R calculation."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        williams_r = MomentumIndicators.williams_r(high, low, close)

        assert_indicator_output_valid(
            williams_r, expected_range=(-100, 0), min_non_nan_ratio=0.8
        )

        # Check that values are reasonable
        wr_clean = williams_r.dropna()
        assert len(wr_clean) > 0
        assert -80 <= wr_clean.median() <= -20  # Should be in middle range typically

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_williams_r_different_periods(self, sample_market_data, period):
        """Test Williams %R with different periods."""
        high = sample_market_data["high"]
        low = sample_market_data["low"]
        close = sample_market_data["close"]

        williams_r = MomentumIndicators.williams_r(high, low, close, period=period)

        assert_indicator_output_valid(williams_r, expected_range=(-100, 0))

    def test_price_momentum_basic_calculation(self, sample_market_data):
        """Test basic Price Momentum calculation."""
        close = sample_market_data["close"]

        momentum = MomentumIndicators.price_momentum(close)

        assert_indicator_output_valid(momentum, min_non_nan_ratio=0.8)

        # Price momentum can be positive or negative
        mom_clean = momentum.dropna()
        assert len(mom_clean) > 0

        # Should have both positive and negative values typically
        has_positive = (mom_clean > 0).any()
        has_negative = (mom_clean < 0).any()
        # Note: We don't assert both must exist as market data could be purely trending

    @pytest.mark.parametrize("period", [5, 10, 14, 20])
    def test_price_momentum_different_periods(self, sample_market_data, period):
        """Test Price Momentum with different periods."""
        close = sample_market_data["close"]

        momentum = MomentumIndicators.price_momentum(close, period=period)

        assert_indicator_output_valid(momentum)

        # Check expected number of NaN values
        nan_count = momentum.isna().sum()
        assert nan_count >= period - 1  # At least period-1 NaNs expected

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        # Empty series - should return empty series, not raise
        empty_series = pd.Series([], dtype=float)

        rsi = MomentumIndicators.rsi(empty_series)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 0

        # Non-numeric data - should handle gracefully and return empty series
        text_series = pd.Series(["a", "b", "c"])

        rsi_text = MomentumIndicators.rsi(text_series)
        assert isinstance(rsi_text, pd.Series)
        # Should return empty or all-NaN series due to data validation

        # Invalid periods
        valid_close = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            MomentumIndicators.rsi(valid_close, period=0)

        with pytest.raises(ValueError):
            MomentumIndicators.rsi(valid_close, period=-1)

    def test_performance_benchmark(self, performance_data):
        """Test performance with larger dataset."""
        close = performance_data["close"]
        high = performance_data["high"]
        low = performance_data["low"]

        # Test RSI performance
        start_time = time.time()
        rsi = MomentumIndicators.rsi(close)
        rsi_time = time.time() - start_time

        assert rsi_time < 1.0, f"RSI took too long: {rsi_time:.3f}s"
        assert_indicator_output_valid(rsi, expected_range=(0, 100))

        # Test MACD performance
        start_time = time.time()
        macd = MomentumIndicators.macd(close)
        macd_time = time.time() - start_time

        assert macd_time < 1.0, f"MACD took too long: {macd_time:.3f}s"
        assert isinstance(macd, dict)

        # Test Stochastic performance
        start_time = time.time()
        stoch = MomentumIndicators.stochastic(high, low, close)
        stoch_time = time.time() - start_time

        assert stoch_time < 1.0, f"Stochastic took too long: {stoch_time:.3f}s"
        assert isinstance(stoch, dict)

    def test_consistency_across_runs(self, sample_market_data):
        """Test that indicators produce consistent results across multiple runs."""
        close = sample_market_data["close"]
        high = sample_market_data["high"]
        low = sample_market_data["low"]

        # Run indicators multiple times
        rsi1 = MomentumIndicators.rsi(close)
        rsi2 = MomentumIndicators.rsi(close)

        # Results should be identical
        pd.testing.assert_series_equal(rsi1, rsi2)

        # Test with MACD
        macd1 = MomentumIndicators.macd(close)
        macd2 = MomentumIndicators.macd(close)

        for key in macd1.keys():
            pd.testing.assert_series_equal(macd1[key], macd2[key])

        # Test with Stochastic
        stoch1 = MomentumIndicators.stochastic(high, low, close)
        stoch2 = MomentumIndicators.stochastic(high, low, close)

        for key in stoch1.keys():
            pd.testing.assert_series_equal(stoch1[key], stoch2[key])

    def test_mathematical_properties(self, sample_market_data):
        """Test mathematical properties of indicators."""
        close = sample_market_data["close"]
        high = sample_market_data["high"]
        low = sample_market_data["low"]

        # RSI should be bounded between 0 and 100
        rsi = MomentumIndicators.rsi(close)
        rsi_clean = rsi.dropna()
        if len(rsi_clean) > 0:
            assert rsi_clean.min() >= 0
            assert rsi_clean.max() <= 100

        # Williams %R should be bounded between -100 and 0
        wr = MomentumIndicators.williams_r(high, low, close)
        wr_clean = wr.dropna()
        if len(wr_clean) > 0:
            assert wr_clean.min() >= -100
            assert wr_clean.max() <= 0

        # Stochastic should be bounded between 0 and 100
        stoch = MomentumIndicators.stochastic(high, low, close)
        for key in ["%K", "%D"]:
            stoch_clean = stoch[key].dropna()
            if len(stoch_clean) > 0:
                assert stoch_clean.min() >= 0
                assert stoch_clean.max() <= 100
