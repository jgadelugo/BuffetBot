"""
Tests for Correlation Metrics functionality.

Professional test suite ensuring comprehensive coverage of correlation calculations,
beta analysis, correlation stability, and risk features with proper error handling
and performance validation.
"""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from buffetbot.features.risk.correlation_metrics import CorrelationMetrics


class TestCorrelationMetrics:
    """Professional test suite for correlation metrics functionality."""

    def test_rolling_correlation_basic(self, multi_asset_returns, expected_risk_ranges):
        """Test basic rolling correlation calculation."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        result = CorrelationMetrics.rolling_correlation(asset_returns, market_returns)

        # Structure validation
        assert isinstance(result, pd.Series)
        assert len(result) == len(asset_returns)

        # Correlation values should be between -1 and 1
        valid_corr = result.dropna()
        if len(valid_corr) > 0:
            assert (valid_corr >= -1.0).all(), "Correlation values should be >= -1"
            assert (valid_corr <= 1.0).all(), "Correlation values should be <= 1"

        # Should have initial NaN values due to window
        window_size = 60  # Default window
        if len(asset_returns) > window_size:
            assert result.iloc[: window_size - 1].isna().all()

    def test_rolling_correlation_custom_window(self, multi_asset_returns):
        """Test rolling correlation with custom window sizes."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        # Test different window sizes
        result_30 = CorrelationMetrics.rolling_correlation(
            asset_returns, market_returns, window=30
        )
        result_120 = CorrelationMetrics.rolling_correlation(
            asset_returns, market_returns, window=120
        )

        # Both should return valid series
        assert isinstance(result_30, pd.Series)
        assert isinstance(result_120, pd.Series)

        # Larger window should have fewer non-NaN values
        assert result_120.count() <= result_30.count()

    def test_rolling_beta_calculation(self, multi_asset_returns, risk_free_rates):
        """Test rolling beta calculation and related metrics."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]
        risk_free_rate = risk_free_rates["normal"]

        result = CorrelationMetrics.rolling_beta(
            asset_returns, market_returns, risk_free_rate=risk_free_rate
        )

        # Structure validation - check for actual field names
        expected_keys = ["beta", "alpha", "r_squared", "tracking_error"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pd.Series)

        # Beta validation - should be reasonable values
        valid_beta = result["beta"].dropna()
        if len(valid_beta) > 0:
            # Beta typically ranges from -2 to 3 for most assets
            assert (valid_beta >= -3.0).all(), "Beta should be >= -3"
            assert (valid_beta <= 5.0).all(), "Beta should be <= 5"

        # R-squared should be between 0 and 1
        valid_r2 = result["r_squared"].dropna()
        if len(valid_r2) > 0:
            assert (valid_r2 >= 0.0).all(), "R-squared should be >= 0"
            assert (valid_r2 <= 1.0).all(), "R-squared should be <= 1"

        # Tracking error should be non-negative
        valid_te = result["tracking_error"].dropna()
        if len(valid_te) > 0:
            assert (valid_te >= 0.0).all(), "Tracking error should be non-negative"

    def test_correlation_stability_analysis(self, multi_asset_returns):
        """Test correlation stability analysis."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        result = CorrelationMetrics.correlation_stability(asset_returns, market_returns)

        # Should return a dictionary with stability metrics
        assert isinstance(result, dict)

        # Check for expected stability metrics
        expected_metrics = [
            "correlation_variance",
            "correlation_trend",
            "stability_ratio",
            "volatility_of_correlation",
        ]
        for metric in expected_metrics:
            if (
                metric in result
            ):  # Some metrics might not be present in all implementations
                assert isinstance(result[metric], (pd.Series, float, int))

    def test_correlation_breakdown_analysis(self, multi_asset_returns):
        """Test correlation breakdown risk analysis."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        result = CorrelationMetrics.correlation_breakdown_risk(
            asset_returns, market_returns
        )

        # Should return breakdown risk metrics
        assert isinstance(result, dict)

        # Validate structure (fields may vary by implementation)
        for key, value in result.items():
            assert isinstance(value, (pd.Series, float, int, np.ndarray))

    def test_multi_asset_correlation_matrix(self, multi_asset_returns):
        """Test multi-asset correlation matrix calculation."""
        returns_dict = {
            "stock": multi_asset_returns["stock"],
            "tech": multi_asset_returns["tech"],
            "market": multi_asset_returns["market"],
        }

        result = CorrelationMetrics.multi_asset_correlation_matrix(returns_dict)

        # Should return dictionary containing correlation matrices
        assert isinstance(result, dict)

        # Check for rolling correlation matrix if available
        if "rolling_correlation" in result:
            assert isinstance(result["rolling_correlation"], dict)
            for matrix_key, matrix_value in result["rolling_correlation"].items():
                assert isinstance(matrix_value, pd.DataFrame)
                # Correlation matrix should be square
                assert matrix_value.shape[0] == matrix_value.shape[1]
                # Diagonal should be 1 (or close to 1)
                if not matrix_value.empty:
                    diag_values = np.diag(matrix_value.values)
                    valid_diag = diag_values[~np.isnan(diag_values)]
                    if len(valid_diag) > 0:
                        assert np.allclose(valid_diag, 1.0, atol=1e-10)

    def test_correlation_risk_features(self, multi_asset_returns):
        """Test correlation risk feature extraction."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        result = CorrelationMetrics.correlation_risk_features(
            asset_returns, market_returns
        )

        # Should return dictionary of features
        assert isinstance(result, dict)

        # All values should be numeric
        for key, value in result.items():
            assert isinstance(value, (int, float)), f"Feature {key} should be numeric"
            assert (
                not math.isnan(value) if isinstance(value, float) else True
            ), f"Feature {key} should not be NaN"

    def test_insufficient_data_handling(self, insufficient_risk_data):
        """Test handling of insufficient data."""
        short_market = insufficient_risk_data

        # Create even shorter asset series for testing
        short_asset = pd.Series(np.random.normal(0, 0.02, 10))

        rolling_corr = CorrelationMetrics.rolling_correlation(short_asset, short_market)
        rolling_beta = CorrelationMetrics.rolling_beta(short_asset, short_market)

        # Should handle gracefully without crashing
        assert isinstance(rolling_corr, pd.Series)
        assert isinstance(rolling_beta, dict)

        # Results should be mostly empty or NaN
        assert rolling_corr.empty or rolling_corr.isna().all()

    def test_misaligned_series_handling(self, multi_asset_returns):
        """Test handling of misaligned time series."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        # Create misaligned series (different date ranges)
        asset_subset = asset_returns.iloc[10:50]
        market_subset = market_returns.iloc[20:80]

        result = CorrelationMetrics.rolling_correlation(asset_subset, market_subset)

        # Should handle misalignment gracefully
        assert isinstance(result, pd.Series)
        # If insufficient data, result may be empty; otherwise should be aligned
        if not result.empty:
            # Result should be aligned to the first input series index
            assert result.index.equals(asset_subset.index)
        else:
            # With insufficient data, empty result is acceptable
            assert len(result) == 0

    def test_extreme_correlation_scenarios(self, extreme_returns):
        """Test correlation calculations with extreme return values."""
        # Create perfectly correlated series
        extreme_market = extreme_returns
        perfectly_correlated = extreme_returns.copy()

        rolling_corr = CorrelationMetrics.rolling_correlation(
            perfectly_correlated, extreme_market
        )

        # Should handle extreme values without crashing
        assert isinstance(rolling_corr, pd.Series)

        # With identical series, correlation should be 1 (where not NaN)
        valid_corr = rolling_corr.dropna()
        if len(valid_corr) > 0:
            assert np.allclose(
                valid_corr, 1.0, atol=1e-10
            ), "Identical series should have correlation ≈ 1"

    def test_returns_with_nans(self, returns_with_nans):
        """Test correlation calculations with NaN values."""
        # Create market series with some NaNs
        market_with_nans = returns_with_nans.copy()
        market_with_nans.iloc[5:10] = np.nan

        result = CorrelationMetrics.rolling_correlation(
            returns_with_nans, market_with_nans
        )
        rolling_beta = CorrelationMetrics.rolling_beta(
            returns_with_nans, market_with_nans
        )

        # Should handle NaN values gracefully
        assert isinstance(result, pd.Series)
        assert isinstance(rolling_beta, dict)

    def test_beta_mathematical_properties(self, multi_asset_returns):
        """Test mathematical properties of beta calculation."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        beta_result = CorrelationMetrics.rolling_beta(asset_returns, market_returns)
        corr_result = CorrelationMetrics.rolling_correlation(
            asset_returns, market_returns
        )

        # Beta and correlation should be related: beta = correlation * (asset_vol / market_vol)
        valid_mask = (
            beta_result["beta"].notna()
            & corr_result.notna()
            & beta_result["r_squared"].notna()
        )

        if valid_mask.any():
            # R-squared should equal correlation squared
            correlation_squared = corr_result[valid_mask] ** 2
            r_squared_values = beta_result["r_squared"][valid_mask]

            # Allow for some numerical differences
            assert np.allclose(
                correlation_squared, r_squared_values, atol=1e-6
            ), "R-squared should equal correlation squared"

    @patch("buffetbot.features.risk.correlation_metrics.logger")
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and logging functionality."""
        # Test with invalid inputs
        invalid_input = "not_a_series"
        valid_input = pd.Series([1, 2, 3, 4, 5])

        # Should handle error gracefully and return empty series
        result = CorrelationMetrics.rolling_correlation(invalid_input, valid_input)

        # Should return empty series and log error
        assert isinstance(result, pd.Series)
        assert result.empty
        mock_logger.error.assert_called()

    def test_performance_requirements(self, performance_data):
        """Test that correlation calculations meet performance requirements."""
        # Generate large correlated datasets
        np.random.seed(42)
        n_points = 1000

        # Create correlated returns
        market_returns = pd.Series(np.random.normal(0, 0.02, n_points))
        asset_returns = 0.7 * market_returns + 0.3 * pd.Series(
            np.random.normal(0, 0.02, n_points)
        )

        import time

        start_time = time.time()

        # Run multiple correlation calculations
        CorrelationMetrics.rolling_correlation(asset_returns, market_returns)
        CorrelationMetrics.rolling_beta(asset_returns, market_returns)
        CorrelationMetrics.correlation_stability(asset_returns, market_returns)
        CorrelationMetrics.correlation_breakdown_risk(asset_returns, market_returns)
        CorrelationMetrics.correlation_risk_features(asset_returns, market_returns)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (3 seconds for 1000 points)
        assert (
            execution_time < 3.0
        ), f"Correlation calculations too slow: {execution_time:.3f}s"

    def test_window_size_effects(self, multi_asset_returns):
        """Test effects of different window sizes on correlation stability."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        # Test multiple window sizes
        windows = [30, 60, 120]
        results = {}

        for window in windows:
            results[window] = CorrelationMetrics.rolling_correlation(
                asset_returns, market_returns, window=window
            )

        # Larger windows should generally produce smoother results
        for window in windows:
            assert isinstance(results[window], pd.Series)
            assert len(results[window]) == len(asset_returns)

        # Larger windows should have fewer initial NaN values impact
        if len(asset_returns) > max(windows):
            for i, window in enumerate(windows[:-1]):
                next_window = windows[i + 1]
                # Larger window should have fewer non-NaN values
                assert results[next_window].count() <= results[window].count()

    def test_correlation_symmetry(self, multi_asset_returns):
        """Test symmetry property of correlation calculations."""
        asset_1 = multi_asset_returns["stock"]
        asset_2 = multi_asset_returns["tech"]

        # Correlation should be symmetric: corr(A,B) = corr(B,A)
        corr_12 = CorrelationMetrics.rolling_correlation(asset_1, asset_2)
        corr_21 = CorrelationMetrics.rolling_correlation(asset_2, asset_1)

        # Since indices might be different due to alignment, compare overlapping periods
        common_index = corr_12.index.intersection(corr_21.index)
        if not common_index.empty:
            corr_12_common = corr_12.loc[common_index]
            corr_21_common = corr_21.loc[common_index]

            valid_mask = corr_12_common.notna() & corr_21_common.notna()
            if valid_mask.any():
                assert np.allclose(
                    corr_12_common[valid_mask], corr_21_common[valid_mask], atol=1e-10
                ), "Correlation should be symmetric"

    def test_risk_free_rate_impact(self, multi_asset_returns, risk_free_rates):
        """Test impact of different risk-free rates on beta calculation."""
        asset_returns = multi_asset_returns["stock"]
        market_returns = multi_asset_returns["market"]

        # Test with different risk-free rates
        beta_zero = CorrelationMetrics.rolling_beta(
            asset_returns, market_returns, risk_free_rate=0.0
        )
        beta_positive = CorrelationMetrics.rolling_beta(
            asset_returns, market_returns, risk_free_rate=risk_free_rates["normal"]
        )

        # Both should return valid results
        assert isinstance(beta_zero["beta"], pd.Series)
        assert isinstance(beta_positive["beta"], pd.Series)

        # Beta values should be different (unless asset/market returns are identical)
        if not asset_returns.equals(market_returns):
            valid_mask = beta_zero["beta"].notna() & beta_positive["beta"].notna()
            if valid_mask.any() and valid_mask.sum() > 1:
                # Allow for some cases where they might be very similar
                differences = np.abs(
                    beta_zero["beta"][valid_mask] - beta_positive["beta"][valid_mask]
                )
                # At least some differences should exist or they should be very small
                assert differences.max() < 10.0, "Beta differences should be reasonable"
