#!/usr/bin/env python3
"""
Test script for the options strategy dispatcher.

This script validates that the strategy dispatcher correctly routes to different
options strategies and that all supported strategies work as expected.
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the current directory to path to import our modules
sys.path.append(".")

try:
    from buffetbot.analysis.options_advisor import (
        InsufficientDataError,
        OptionsAdvisorError,
        analyze_options_strategy,
        recommend_bull_call_spread,
        recommend_cash_secured_put,
        recommend_covered_call,
        recommend_long_calls,
    )

    print(
        "✅ Successfully imported strategy dispatcher and individual strategy functions"
    )
except ImportError as e:
    print(f"❌ Failed to import strategy functions: {e}")
    sys.exit(1)


class TestStrategyDispatcher(unittest.TestCase):
    """Test cases for the options strategy dispatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_ticker = "AAPL"
        self.min_days = 90
        self.top_n = 3
        self.risk_tolerance = "Conservative"
        self.time_horizon = "Medium-term (3-6 months)"

        # Sample options data for mocking
        self.sample_options_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 3,
                "strike": [150.0, 155.0, 160.0],
                "expiry": ["2024-12-20", "2024-12-20", "2024-12-20"],
                "lastPrice": [5.0, 3.5, 2.0],
                "IV": [0.25, 0.28, 0.30],
                "RSI": [45.0, 45.0, 45.0],
                "Beta": [1.2, 1.2, 1.2],
                "Momentum": [0.02, 0.02, 0.02],
                "ForecastConfidence": [0.7, 0.7, 0.7],
                "CompositeScore": [0.75, 0.70, 0.65],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    }
                ]
                * 3,
                "daysToExpiry": [180, 180, 180],
            }
        )

    def test_supported_strategies_validation(self):
        """Test that only supported strategies are accepted."""
        print("\n🧪 Testing strategy validation...")

        supported_strategies = [
            "Long Calls",
            "Bull Call Spread",
            "Covered Call",
            "Cash-Secured Put",
        ]

        # Test unsupported strategy
        with self.assertRaises(OptionsAdvisorError) as context:
            analyze_options_strategy(
                strategy_type="Unsupported Strategy",
                ticker=self.test_ticker,
                min_days=self.min_days,
                top_n=self.top_n,
            )

        self.assertIn("Unsupported strategy", str(context.exception))
        print("✅ Correctly rejected unsupported strategy")

        # Test all supported strategies are in the validation list
        for strategy in supported_strategies:
            try:
                # This will fail for other reasons, but not strategy validation
                analyze_options_strategy(
                    strategy_type=strategy,
                    ticker="INVALID_TICKER_FOR_TEST",
                    min_days=self.min_days,
                    top_n=self.top_n,
                )
            except OptionsAdvisorError as e:
                # Should not be a strategy validation error
                self.assertNotIn("Unsupported strategy", str(e))
                print(f"✅ Strategy '{strategy}' passed validation")
            except Exception:
                # Other exceptions are expected during testing
                print(f"✅ Strategy '{strategy}' passed validation")

    @patch("buffetbot.analysis.options_advisor.recommend_long_calls")
    def test_long_calls_routing(self, mock_recommend_long_calls):
        """Test that Long Calls strategy routes correctly."""
        print("\n🧪 Testing Long Calls routing...")

        mock_recommend_long_calls.return_value = self.sample_options_data.copy()

        result = analyze_options_strategy(
            strategy_type="Long Calls",
            ticker=self.test_ticker,
            min_days=self.min_days,
            top_n=self.top_n,
            risk_tolerance=self.risk_tolerance,
            time_horizon=self.time_horizon,
        )

        # Verify the correct function was called
        mock_recommend_long_calls.assert_called_once_with(
            self.test_ticker, self.min_days, self.top_n, "Conservative"
        )

        # Verify metadata was added
        self.assertIn("strategy_type", result.columns)
        self.assertIn("risk_tolerance", result.columns)
        self.assertIn("time_horizon", result.columns)
        self.assertIn("analysis_date", result.columns)

        # Verify metadata values
        self.assertEqual(result["strategy_type"].iloc[0], "Long Calls")
        self.assertEqual(result["risk_tolerance"].iloc[0], self.risk_tolerance)
        self.assertEqual(result["time_horizon"].iloc[0], self.time_horizon)

        print("✅ Long Calls strategy routing successful")

    @patch("buffetbot.analysis.options_advisor.recommend_bull_call_spread")
    def test_bull_call_spread_routing(self, mock_recommend_bull_call_spread):
        """Test that Bull Call Spread strategy routes correctly."""
        print("\n🧪 Testing Bull Call Spread routing...")

        # Create spread-specific data
        spread_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 2,
                "expiry": ["2024-12-20", "2024-12-20"],
                "long_strike": [150.0, 155.0],
                "short_strike": [155.0, 160.0],
                "long_price": [5.0, 3.5],
                "short_price": [3.0, 2.0],
                "net_premium": [2.0, 1.5],
                "max_profit": [3.0, 3.5],
                "max_loss": [2.0, 1.5],
                "profit_ratio": [1.5, 2.33],
                "RSI": [45.0, 45.0],
                "Beta": [1.2, 1.2],
                "Momentum": [0.02, 0.02],
                "IV": [0.265, 0.29],
                "ForecastConfidence": [0.7, 0.7],
                "CompositeScore": [0.75, 0.70],
                "score_details": [
                    {
                        "rsi": 0.2,
                        "beta": 0.2,
                        "momentum": 0.2,
                        "iv": 0.2,
                        "forecast": 0.2,
                    }
                ]
                * 2,
                "daysToExpiry": [180, 180],
            }
        )

        mock_recommend_bull_call_spread.return_value = spread_data

        result = analyze_options_strategy(
            strategy_type="Bull Call Spread",
            ticker=self.test_ticker,
            min_days=self.min_days,
            top_n=self.top_n,
            risk_tolerance=self.risk_tolerance,
            time_horizon=self.time_horizon,
        )

        # Verify the correct function was called with adjusted min_days
        called_args = mock_recommend_bull_call_spread.call_args
        self.assertEqual(called_args[0][0], self.test_ticker)  # ticker
        self.assertLessEqual(called_args[0][1], 90)  # min_days should be <= 90
        self.assertEqual(called_args[0][2], self.top_n)  # top_n
        self.assertEqual(called_args[0][3], "Conservative")  # risk_tolerance

        # Verify spread-specific columns exist
        self.assertIn("long_strike", result.columns)
        self.assertIn("short_strike", result.columns)
        self.assertIn("net_premium", result.columns)
        self.assertIn("max_profit", result.columns)
        self.assertIn("profit_ratio", result.columns)

        # Verify metadata
        self.assertEqual(result["strategy_type"].iloc[0], "Bull Call Spread")

        print("✅ Bull Call Spread strategy routing successful")

    @patch("buffetbot.analysis.options_advisor.recommend_covered_call")
    def test_covered_call_routing(self, mock_recommend_covered_call):
        """Test that Covered Call strategy routes correctly."""
        print("\n🧪 Testing Covered Call routing...")

        # Create covered call-specific data
        covered_call_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 2,
                "strike": [155.0, 160.0],
                "expiry": ["2024-09-20", "2024-09-20"],
                "lastPrice": [2.0, 1.0],
                "IV": [0.25, 0.22],
                "RSI": [45.0, 45.0],
                "Beta": [1.2, 1.2],
                "Momentum": [0.02, 0.02],
                "ForecastConfidence": [0.7, 0.7],
                "CompositeScore": [0.75, 0.70],
                "premium_yield": [1.3, 0.65],
                "annualized_yield": [15.6, 7.8],
                "upside_capture": [3.3, 6.7],
                "total_return": [4.6, 7.35],
                "daysToExpiry": [30, 30],
            }
        )

        mock_recommend_covered_call.return_value = covered_call_data

        result = analyze_options_strategy(
            strategy_type="Covered Call",
            ticker=self.test_ticker,
            min_days=90,  # Should be adjusted down for covered calls
            top_n=self.top_n,
            risk_tolerance=self.risk_tolerance,
            time_horizon=self.time_horizon,
        )

        # Verify the correct function was called with adjusted min_days
        # Covered calls should have min_days adjusted to max 90
        called_args = mock_recommend_covered_call.call_args
        self.assertEqual(called_args[0][0], self.test_ticker)  # ticker
        self.assertLessEqual(called_args[0][1], 90)  # min_days should be <= 90
        self.assertEqual(called_args[0][2], self.top_n)  # top_n
        self.assertEqual(called_args[0][3], "Conservative")  # risk_tolerance

        # Verify covered call-specific columns exist
        self.assertIn("premium_yield", result.columns)
        self.assertIn("annualized_yield", result.columns)
        self.assertIn("upside_capture", result.columns)

        # Verify metadata
        self.assertEqual(result["strategy_type"].iloc[0], "Covered Call")

        print("✅ Covered Call strategy routing successful")

    @patch("buffetbot.analysis.options_advisor.recommend_cash_secured_put")
    def test_cash_secured_put_routing(self, mock_recommend_cash_secured_put):
        """Test that Cash-Secured Put strategy routes correctly."""
        print("\n🧪 Testing Cash-Secured Put routing...")

        # Create cash-secured put-specific data
        csp_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 2,
                "strike": [145.0, 140.0],
                "expiry": ["2024-09-20", "2024-09-20"],
                "lastPrice": [1.5, 0.8],
                "IV": [0.28, 0.25],
                "RSI": [35.0, 35.0],
                "Beta": [1.2, 1.2],
                "Momentum": [-0.01, -0.01],
                "ForecastConfidence": [0.7, 0.7],
                "CompositeScore": [0.70, 0.65],
                "premium_yield": [1.03, 0.57],
                "annualized_yield": [12.5, 6.9],
                "assignment_discount": [3.3, 6.7],
                "effective_cost": [143.5, 139.2],
                "discount_to_current": [4.3, 7.2],
                "daysToExpiry": [30, 30],
            }
        )

        mock_recommend_cash_secured_put.return_value = csp_data

        result = analyze_options_strategy(
            strategy_type="Cash-Secured Put",
            ticker=self.test_ticker,
            min_days=90,  # Should be adjusted down for CSPs
            top_n=self.top_n,
            risk_tolerance=self.risk_tolerance,
            time_horizon=self.time_horizon,
        )

        # Verify the correct function was called with adjusted min_days
        called_args = mock_recommend_cash_secured_put.call_args
        self.assertEqual(called_args[0][0], self.test_ticker)  # ticker
        self.assertLessEqual(called_args[0][1], 90)  # min_days should be <= 90
        self.assertEqual(called_args[0][2], self.top_n)  # top_n
        self.assertEqual(called_args[0][3], "Conservative")  # risk_tolerance

        # Verify CSP-specific columns exist
        self.assertIn("premium_yield", result.columns)
        self.assertIn("annualized_yield", result.columns)
        self.assertIn("assignment_discount", result.columns)
        self.assertIn("effective_cost", result.columns)

        # Verify metadata
        self.assertEqual(result["strategy_type"].iloc[0], "Cash-Secured Put")

        print("✅ Cash-Secured Put strategy routing successful")

    def test_risk_tolerance_adjustments(self):
        """Test that risk tolerance properly adjusts parameters."""
        print("\n🧪 Testing risk tolerance adjustments...")

        # Mock the underlying functions to test parameter passing
        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_long_calls:
            mock_long_calls.return_value = self.sample_options_data.copy()

            # Test Conservative adjustment (should increase min_days)
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=30,  # Low value to test adjustment
                top_n=self.top_n,
                risk_tolerance="Conservative",
            )

            called_args = mock_long_calls.call_args
            conservative_min_days = called_args[0][1]
            self.assertGreaterEqual(conservative_min_days, 60)  # Should be at least 60
            print(
                f"✅ Conservative risk tolerance adjusted min_days to {conservative_min_days}"
            )

            # Test Aggressive adjustment (should increase top_n)
            mock_long_calls.reset_mock()
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=self.min_days,
                top_n=3,  # Low value to test adjustment
                risk_tolerance="Aggressive",
            )

            called_args = mock_long_calls.call_args
            aggressive_top_n = called_args[0][2]
            self.assertGreater(aggressive_top_n, 3)  # Should be increased
            self.assertLessEqual(aggressive_top_n, 10)  # But capped at 10
            print(f"✅ Aggressive risk tolerance adjusted top_n to {aggressive_top_n}")

    def test_parameter_validation(self):
        """Test that invalid parameters are handled appropriately."""
        print("\n🧪 Testing parameter validation...")

        # Test with empty/invalid ticker - should be caught by underlying functions
        with self.assertRaises((OptionsAdvisorError, InsufficientDataError, Exception)):
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker="",  # Invalid ticker
                min_days=self.min_days,
                top_n=self.top_n,
            )
        print("✅ Invalid ticker properly rejected")

    def test_metadata_addition(self):
        """Test that strategy metadata is properly added to results."""
        print("\n🧪 Testing metadata addition...")

        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_long_calls:
            test_data = self.sample_options_data.copy()
            mock_long_calls.return_value = test_data

            custom_risk_tolerance = "Aggressive"
            custom_time_horizon = "One Year (12 months)"

            result = analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=self.min_days,
                top_n=self.top_n,
                risk_tolerance=custom_risk_tolerance,
                time_horizon=custom_time_horizon,
            )

            # Verify all metadata columns exist
            expected_metadata = [
                "strategy_type",
                "risk_tolerance",
                "time_horizon",
                "analysis_date",
            ]
            for col in expected_metadata:
                self.assertIn(col, result.columns, f"Missing metadata column: {col}")

            # Verify metadata values
            self.assertEqual(result["strategy_type"].iloc[0], "Long Calls")
            self.assertEqual(result["risk_tolerance"].iloc[0], custom_risk_tolerance)
            self.assertEqual(result["time_horizon"].iloc[0], custom_time_horizon)

            # Verify analysis_date is recent (within last minute)
            analysis_date = datetime.strptime(
                result["analysis_date"].iloc[0], "%Y-%m-%d %H:%M:%S"
            )
            time_diff = datetime.now() - analysis_date
            self.assertLess(time_diff.total_seconds(), 60)  # Within last minute

            print("✅ All metadata properly added to results")

    def test_empty_results_handling(self):
        """Test that empty results are handled gracefully."""
        print("\n🧪 Testing empty results handling...")

        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_long_calls:
            # Return empty DataFrame
            empty_df = pd.DataFrame()
            mock_long_calls.return_value = empty_df

            result = analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=self.min_days,
                top_n=self.top_n,
                risk_tolerance=self.risk_tolerance,
                time_horizon=self.time_horizon,
            )

            # Should return empty DataFrame without crashing
            self.assertTrue(result.empty)
            print("✅ Empty results handled gracefully")


def main():
    """Run the strategy dispatcher tests."""
    print("=" * 60)
    print("TESTING OPTIONS STRATEGY DISPATCHER")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategyDispatcher)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("STRATEGY DISPATCHER TEST SUMMARY")
    print("=" * 60)

    if result.wasSuccessful():
        print("✅ All strategy dispatcher tests passed!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ Some tests failed!")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
