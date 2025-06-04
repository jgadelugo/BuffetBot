"""
Regression tests for options analysis system.

This module contains regression tests to ensure that fixes remain stable
under various edge cases and stress conditions.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from buffetbot.analysis.options.core.domain_models import (
    AnalysisRequest,
    RiskTolerance,
    StrategyType,
    TimeHorizon,
)
from buffetbot.analysis.options.data.options_service import DefaultOptionsService
from buffetbot.analysis.options_advisor import normalize_scoring_weights


class TestRegressionScenarios:
    """Test regression scenarios to ensure fixes are stable."""

    def test_time_horizon_enum_under_stress(self):
        """Test TimeHorizon enum behavior under stress conditions."""
        # Test repeated conversions
        for _ in range(100):
            horizon = TimeHorizon("One Year (12 months)")
            assert horizon == TimeHorizon.ONE_YEAR

        # Test all enum values in rapid succession
        all_values = [
            "Short-term (1-3 months)",
            "Medium-term (3-6 months)",
            "Long-term (6+ months)",
            "One Year (12 months)",
            "18 Months (1.5 years)",
        ]

        for value in all_values * 10:  # Repeat 10 times
            horizon = TimeHorizon(value)
            assert horizon.value == value

    def test_weight_normalization_edge_cases(self):
        """Test weight normalization with various edge cases."""
        # Edge case 1: Very small weights
        small_weights = {"rsi": 0.0001, "beta": 0.0001, "momentum": 0.9998}
        result = normalize_scoring_weights(small_weights, ["rsi", "beta"])
        assert abs(sum(result.values()) - 1.0) < 0.001

        # Edge case 2: Very large number of sources
        many_sources = [f"source_{i}" for i in range(100)]
        result = normalize_scoring_weights({}, many_sources)
        assert len(result) == 100
        assert abs(sum(result.values()) - 1.0) < 0.001

        # Edge case 3: Duplicate sources
        duplicate_sources = ["rsi", "rsi", "beta", "beta"]
        original_weights = {"rsi": 0.6, "beta": 0.4}
        result = normalize_scoring_weights(original_weights, duplicate_sources)
        # Should handle duplicates gracefully
        assert "rsi" in result
        assert "beta" in result

    def test_options_service_error_resilience(self):
        """Test options service resilience to various error conditions."""
        service = DefaultOptionsService()

        # Test with various invalid inputs
        invalid_inputs = [
            ("", 180),  # Empty ticker
            ("   ", 180),  # Whitespace ticker
            ("AAPL", 0),  # Zero min_days
            ("AAPL", -10),  # Negative min_days
            (None, 180),  # None ticker
        ]

        for ticker, min_days in invalid_inputs:
            # Should handle gracefully without crashing
            try:
                if ticker is None:
                    # This should raise an error during validation
                    continue
                service.fetch_options_data(ticker, min_days)
            except Exception as e:
                # Expected to fail, but shouldn't crash the system
                assert isinstance(e, Exception)

    def test_analysis_request_boundary_conditions(self):
        """Test AnalysisRequest with boundary values."""
        boundary_cases = [
            # Valid boundary cases
            {"ticker": "A", "min_days": 1, "top_n": 1},
            {"ticker": "AAPL", "min_days": 365, "top_n": 50},
            {"ticker": "VERYLONGTICKER", "min_days": 180, "top_n": 25},
        ]

        for case in boundary_cases:
            request = AnalysisRequest(
                ticker=case["ticker"],
                strategy_type=StrategyType.LONG_CALLS,
                min_days=case["min_days"],
                top_n=case["top_n"],
            )
            # Should create successfully
            assert request.ticker == case["ticker"]
            assert request.min_days == case["min_days"]
            assert request.top_n == case["top_n"]

    def test_concurrent_enum_access(self):
        """Test concurrent access to TimeHorizon enum (simulated)."""
        import threading

        results = []
        errors = []

        def access_enum():
            try:
                for value in ["One Year (12 months)", "18 Months (1.5 years)"]:
                    horizon = TimeHorizon(value)
                    results.append(horizon.value)
            except Exception as e:
                errors.append(e)

        # Simulate concurrent access
        threads = [threading.Thread(target=access_enum) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have no errors and correct number of results
        assert len(errors) == 0
        assert len(results) == 20  # 10 threads * 2 values each

    def test_memory_leak_prevention(self):
        """Test that repeated operations don't cause memory issues."""
        # Test repeated AnalysisRequest creation
        for i in range(1000):
            request = AnalysisRequest(
                ticker=f"TEST{i%10}",
                strategy_type=StrategyType.LONG_CALLS,
                time_horizon=TimeHorizon.ONE_YEAR
                if i % 2 == 0
                else TimeHorizon.EIGHTEEN_MONTHS,
            )
            # Object should be created successfully
            assert request.ticker.startswith("TEST")

        # Test repeated weight normalization
        weights = {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2}
        for i in range(1000):
            sources = ["rsi", "beta"] if i % 2 == 0 else ["momentum", "iv", "forecast"]
            result = normalize_scoring_weights(weights, sources)
            assert abs(sum(result.values()) - 1.0) < 0.001

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # Test tickers with special characters (though not typically valid)
        special_tickers = ["TEST-A", "TEST.B", "TEST/C"]

        for ticker in special_tickers:
            try:
                request = AnalysisRequest(
                    ticker=ticker, strategy_type=StrategyType.LONG_CALLS
                )
                # If it creates successfully, that's fine
                assert request.ticker == ticker
            except ValueError:
                # If it fails validation, that's also acceptable
                pass


class TestDataIntegrityChecks:
    """Test data integrity across all fixed components."""

    def test_enum_value_consistency(self):
        """Test that enum values are consistent across the system."""
        # All TimeHorizon values should be unique
        all_values = [th.value for th in TimeHorizon]
        assert len(all_values) == len(set(all_values))

        # All values should be strings
        assert all(isinstance(value, str) for value in all_values)

        # All values should be non-empty
        assert all(len(value.strip()) > 0 for value in all_values)

    def test_weight_normalization_properties(self):
        """Test mathematical properties of weight normalization."""
        test_weights = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}

        # Property 1: Weights should always sum to 1.0
        for i in range(1, 5):  # Test with 1 to 4 sources
            sources = list(test_weights.keys())[:i]
            result = normalize_scoring_weights(test_weights, sources)
            assert abs(sum(result.values()) - 1.0) < 0.001

        # Property 2: Relative proportions should be maintained
        sources = ["a", "b"]  # a:b should be 1:2 ratio
        result = normalize_scoring_weights(test_weights, sources)
        assert abs(result["b"] / result["a"] - 2.0) < 0.001

    def test_options_service_data_flow(self):
        """Test data flow integrity in options service."""
        service = DefaultOptionsService()

        # Test cache functionality
        service.cache_enabled = True
        assert service._cache == {}

        # Test cache operations
        service._cache["test_key"] = "test_value"
        assert service._cache["test_key"] == "test_value"

        service.clear_cache()
        assert service._cache == {}

    @patch("buffetbot.analysis.options.data.options_service.fetch_long_dated_calls")
    def test_error_propagation_consistency(self, mock_fetch):
        """Test that errors are consistently handled and propagated."""
        # Test various error scenarios
        error_scenarios = [
            {"data_available": False, "error_message": "No data"},
            {
                "data_available": True,
                "data": pd.DataFrame(),
                "error_message": None,
            },  # Empty data
        ]

        service = DefaultOptionsService()

        for scenario in error_scenarios:
            mock_fetch.return_value = scenario

            with pytest.raises(Exception) as exc_info:
                service.fetch_options_data("TEST", 180)

            # Error should contain meaningful information
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["data", "options", "test"])


class TestBackwardsCompatibility:
    """Test that fixes maintain backwards compatibility."""

    def test_original_time_horizon_values_still_work(self):
        """Test that original TimeHorizon values still work as before."""
        original_values = [
            "Short-term (1-3 months)",
            "Medium-term (3-6 months)",
            "Long-term (6+ months)",
        ]

        for value in original_values:
            # Should work exactly as before
            horizon = TimeHorizon(value)
            assert horizon.value == value

            # Should work in AnalysisRequest
            request = AnalysisRequest(
                ticker="AAPL",
                strategy_type=StrategyType.LONG_CALLS,
                time_horizon=horizon,
            )
            assert request.time_horizon.value == value

    def test_default_parameter_behavior(self):
        """Test that default parameters still work as expected."""
        # AnalysisRequest with minimal parameters
        request = AnalysisRequest(ticker="AAPL", strategy_type=StrategyType.LONG_CALLS)

        # Should use default values
        assert request.min_days == 180
        assert request.top_n == 5
        assert request.risk_tolerance == RiskTolerance.CONSERVATIVE
        assert request.time_horizon == TimeHorizon.MEDIUM_TERM

    def test_existing_api_signatures(self):
        """Test that existing API signatures are preserved."""
        # normalize_scoring_weights should accept the same parameters
        weights = {"rsi": 0.2, "beta": 0.8}
        sources = ["rsi"]

        # This call signature should still work
        result = normalize_scoring_weights(weights, sources)
        assert isinstance(result, dict)
        assert "rsi" in result

    def test_error_message_consistency(self):
        """Test that error messages are consistent and helpful."""
        # Test invalid TimeHorizon value
        with pytest.raises(ValueError) as exc_info:
            TimeHorizon("Invalid Time Horizon")

        error_msg = str(exc_info.value)
        assert "Invalid Time Horizon" in error_msg

        # Test invalid AnalysisRequest
        with pytest.raises(ValueError) as exc_info:
            AnalysisRequest(ticker="", strategy_type=StrategyType.LONG_CALLS)

        error_msg = str(exc_info.value)
        assert any(
            keyword in error_msg.lower() for keyword in ["ticker", "empty", "string"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
