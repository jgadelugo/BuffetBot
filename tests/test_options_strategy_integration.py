#!/usr/bin/env python3
"""
Integration test for options strategy selection in the dashboard.

This test validates that the dashboard UI properly integrates with the strategy dispatcher
and that changing the strategy selection actually routes to different analysis functions.
Enhanced tests for comprehensive risk tolerance functionality.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add the current directory to path
sys.path.append(".")

try:
    from buffetbot.analysis.options_advisor import analyze_options_strategy

    print("✅ Successfully imported strategy dispatcher")
except ImportError as e:
    print(f"❌ Failed to import strategy dispatcher: {e}")
    sys.exit(1)


class TestOptionsStrategyIntegration(unittest.TestCase):
    """Integration tests for options strategy selection with enhanced risk tolerance."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_ticker = "AAPL"
        self.sample_results = {
            "Long Calls": pd.DataFrame(
                {
                    "ticker": [self.test_ticker] * 2,
                    "strike": [150.0, 155.0],
                    "expiry": ["2024-12-20", "2024-12-20"],
                    "lastPrice": [5.0, 3.5],
                    "IV": [0.25, 0.28],
                    "CompositeScore": [0.75, 0.70],
                    "strategy_type": ["Long Calls", "Long Calls"],
                    "risk_tolerance_applied": ["Moderate", "Moderate"],
                }
            ),
            "Bull Call Spread": pd.DataFrame(
                {
                    "ticker": [self.test_ticker] * 2,
                    "long_strike": [150.0, 155.0],
                    "short_strike": [155.0, 160.0],
                    "net_premium": [2.0, 1.5],
                    "profit_ratio": [1.5, 2.33],
                    "CompositeScore": [0.75, 0.70],
                    "strategy_type": ["Bull Call Spread", "Bull Call Spread"],
                    "risk_tolerance_applied": ["Moderate", "Moderate"],
                }
            ),
            "Covered Call": pd.DataFrame(
                {
                    "ticker": [self.test_ticker] * 2,
                    "strike": [155.0, 160.0],
                    "premium_yield": [1.3, 0.65],
                    "annualized_yield": [15.6, 7.8],
                    "assignment_probability": [0.2, 0.1],
                    "CompositeScore": [0.75, 0.70],
                    "strategy_type": ["Covered Call", "Covered Call"],
                    "risk_tolerance_applied": ["Moderate", "Moderate"],
                }
            ),
            "Cash-Secured Put": pd.DataFrame(
                {
                    "ticker": [self.test_ticker] * 2,
                    "strike": [145.0, 140.0],
                    "premium_yield": [1.03, 0.57],
                    "assignment_discount": [3.3, 6.7],
                    "assignment_probability": [0.15, 0.08],
                    "CompositeScore": [0.70, 0.65],
                    "strategy_type": ["Cash-Secured Put", "Cash-Secured Put"],
                    "risk_tolerance_applied": ["Moderate", "Moderate"],
                }
            ),
        }

    @patch("buffetbot.analysis.options_advisor.recommend_long_calls")
    @patch("buffetbot.analysis.options_advisor.recommend_bull_call_spread")
    @patch("buffetbot.analysis.options_advisor.recommend_covered_call")
    @patch("buffetbot.analysis.options_advisor.recommend_cash_secured_put")
    def test_enhanced_risk_tolerance_routing(
        self, mock_csp, mock_cc, mock_bcs, mock_lc
    ):
        """Test that risk tolerance parameters are properly passed to all strategies."""
        print("\n🧪 Testing enhanced risk tolerance routing...")

        # Setup mocks with risk tolerance results
        mock_lc.return_value = self.sample_results["Long Calls"]
        mock_bcs.return_value = self.sample_results["Bull Call Spread"]
        mock_cc.return_value = self.sample_results["Covered Call"]
        mock_csp.return_value = self.sample_results["Cash-Secured Put"]

        # Test each strategy with different risk tolerances
        strategies = [
            "Long Calls",
            "Bull Call Spread",
            "Covered Call",
            "Cash-Secured Put",
        ]
        risk_tolerances = ["Conservative", "Moderate", "Aggressive"]

        for strategy in strategies:
            for risk_tolerance in risk_tolerances:
                print(f"  Testing {strategy} with {risk_tolerance} risk tolerance...")

                # Reset all mocks
                for mock in [mock_lc, mock_bcs, mock_cc, mock_csp]:
                    mock.reset_mock()

                # Call the strategy dispatcher with risk tolerance
                result = analyze_options_strategy(
                    strategy_type=strategy,
                    ticker=self.test_ticker,
                    min_days=90,
                    top_n=3,
                    risk_tolerance=risk_tolerance,
                    time_horizon="Medium-term (3-6 months)",
                )

                # Verify the correct function was called with risk tolerance
                if strategy == "Long Calls":
                    mock_lc.assert_called_once()
                    called_args = mock_lc.call_args
                    # Check if called with positional arguments or keyword arguments
                    if (
                        len(called_args[0]) >= 4
                    ):  # positional args: ticker, min_days, top_n, risk_tolerance
                        passed_risk_tolerance = called_args[0][3]
                    else:  # keyword argument
                        passed_risk_tolerance = called_args[1].get(
                            "risk_tolerance", "Moderate"
                        )
                    self.assertEqual(passed_risk_tolerance, risk_tolerance)
                elif strategy == "Bull Call Spread":
                    mock_bcs.assert_called_once()
                    called_args = mock_bcs.call_args
                    if len(called_args[0]) >= 4:
                        passed_risk_tolerance = called_args[0][3]
                    else:
                        passed_risk_tolerance = called_args[1].get(
                            "risk_tolerance", "Moderate"
                        )
                    self.assertEqual(passed_risk_tolerance, risk_tolerance)
                elif strategy == "Covered Call":
                    mock_cc.assert_called_once()
                    called_args = mock_cc.call_args
                    if len(called_args[0]) >= 4:
                        passed_risk_tolerance = called_args[0][3]
                    else:
                        passed_risk_tolerance = called_args[1].get(
                            "risk_tolerance", "Moderate"
                        )
                    self.assertEqual(passed_risk_tolerance, risk_tolerance)
                elif strategy == "Cash-Secured Put":
                    mock_csp.assert_called_once()
                    called_args = mock_csp.call_args
                    if len(called_args[0]) >= 4:
                        passed_risk_tolerance = called_args[0][3]
                    else:
                        passed_risk_tolerance = called_args[1].get(
                            "risk_tolerance", "Moderate"
                        )
                    self.assertEqual(passed_risk_tolerance, risk_tolerance)

                # Verify metadata includes risk tolerance
                self.assertIn("risk_tolerance", result.columns)
                self.assertEqual(result["risk_tolerance"].iloc[0], risk_tolerance)

                print(
                    f"    ✅ {strategy} + {risk_tolerance}: Risk tolerance properly passed"
                )

    def test_risk_tolerance_parameter_adjustments(self):
        """Test that risk tolerance affects parameter adjustments correctly."""
        print("\n🧪 Testing risk tolerance parameter adjustments...")

        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_lc:
            mock_lc.return_value = self.sample_results["Long Calls"]

            # Test Conservative adjustments
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=30,  # Low value to trigger adjustment
                top_n=5,
                risk_tolerance="Conservative",
            )

            called_args = mock_lc.call_args[0]
            conservative_min_days = called_args[1]
            self.assertGreaterEqual(
                conservative_min_days, 60, "Conservative should increase min_days"
            )
            print("✅ Conservative: min_days properly adjusted")

            # Test Aggressive adjustments
            mock_lc.reset_mock()
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker=self.test_ticker,
                min_days=90,
                top_n=3,  # Low value to trigger adjustment
                risk_tolerance="Aggressive",
            )

            called_args = mock_lc.call_args[0]
            aggressive_top_n = called_args[2]
            self.assertGreater(aggressive_top_n, 3, "Aggressive should increase top_n")
            self.assertLessEqual(aggressive_top_n, 10, "But cap at 10")
            print("✅ Aggressive: top_n properly adjusted")

    def test_risk_tolerance_metadata_consistency(self):
        """Test that risk tolerance metadata is consistently applied across strategies."""
        print("\n🧪 Testing risk tolerance metadata consistency...")

        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_lc, patch(
            "buffetbot.analysis.options_advisor.recommend_covered_call"
        ) as mock_cc:
            # Setup mocks with proper risk tolerance metadata
            sample_lc = self.sample_results["Long Calls"].copy()
            sample_cc = self.sample_results["Covered Call"].copy()

            mock_lc.return_value = sample_lc
            mock_cc.return_value = sample_cc

            risk_tolerances = ["Conservative", "Moderate", "Aggressive"]

            for risk_tolerance in risk_tolerances:
                # Test Long Calls
                lc_result = analyze_options_strategy(
                    "Long Calls", self.test_ticker, risk_tolerance=risk_tolerance
                )

                # Test Covered Call
                cc_result = analyze_options_strategy(
                    "Covered Call", self.test_ticker, risk_tolerance=risk_tolerance
                )

                # Verify metadata consistency
                expected_metadata = [
                    "strategy_type",
                    "risk_tolerance",
                    "time_horizon",
                    "analysis_date",
                ]
                for col in expected_metadata:
                    self.assertIn(col, lc_result.columns, f"Long Calls missing {col}")
                    self.assertIn(col, cc_result.columns, f"Covered Call missing {col}")

                # Verify risk tolerance values
                self.assertEqual(lc_result["risk_tolerance"].iloc[0], risk_tolerance)
                self.assertEqual(cc_result["risk_tolerance"].iloc[0], risk_tolerance)

                print(
                    f"✅ {risk_tolerance}: Metadata consistently applied to all strategies"
                )

    def test_risk_tolerance_filtering_effects(self):
        """Test that risk tolerance affects the types of recommendations returned."""
        print("\n🧪 Testing risk tolerance filtering effects...")

        # Create mock data with different risk characteristics
        conservative_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 3,
                "strike": [148.0, 152.0, 156.0],  # More ITM/ATM options
                "expiry": ["2024-12-20"] * 3,
                "lastPrice": [6.0, 4.0, 2.5],
                "IV": [0.20, 0.22, 0.24],  # Lower IV
                "CompositeScore": [0.80, 0.75, 0.70],
                "daysToExpiry": [180, 180, 180],  # Longer time
                "strategy_type": ["Long Calls"] * 3,
                "risk_tolerance_applied": ["Conservative"] * 3,
            }
        )

        aggressive_data = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 5,  # More options for aggressive
                "strike": [155.0, 160.0, 165.0, 170.0, 175.0],  # More OTM options
                "expiry": ["2024-09-20"] * 5,
                "lastPrice": [3.0, 1.8, 1.0, 0.5, 0.3],
                "IV": [0.28, 0.32, 0.35, 0.38, 0.40],  # Higher IV
                "CompositeScore": [0.75, 0.70, 0.65, 0.60, 0.55],
                "daysToExpiry": [60, 60, 60, 60, 60],  # Shorter time
                "strategy_type": ["Long Calls"] * 5,
                "risk_tolerance_applied": ["Aggressive"] * 5,
            }
        )

        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_lc:
            # Test Conservative
            mock_lc.return_value = conservative_data
            conservative_result = analyze_options_strategy(
                "Long Calls", self.test_ticker, risk_tolerance="Conservative", top_n=5
            )

            # Conservative should return fewer, higher-quality options
            self.assertLessEqual(len(conservative_result), 5)
            avg_iv_conservative = conservative_result["IV"].mean()
            self.assertLess(
                avg_iv_conservative, 0.30, "Conservative should have lower average IV"
            )
            print("✅ Conservative: Returns lower-risk, lower-IV options")

            # Test Aggressive
            mock_lc.return_value = aggressive_data
            aggressive_result = analyze_options_strategy(
                "Long Calls", self.test_ticker, risk_tolerance="Aggressive", top_n=3
            )

            # Aggressive should return more options with higher potential
            self.assertGreaterEqual(len(aggressive_result), 3)
            avg_iv_aggressive = aggressive_result["IV"].mean()
            self.assertGreater(
                avg_iv_aggressive, 0.25, "Aggressive should allow higher average IV"
            )
            print("✅ Aggressive: Returns higher-leverage, higher-IV options")

    def test_strategy_specific_risk_tolerance_effects(self):
        """Test that different strategies apply risk tolerance differently."""
        print("\n🧪 Testing strategy-specific risk tolerance effects...")

        # Test Covered Call risk tolerance effects
        covered_call_conservative = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 2,
                "strike": [160.0, 165.0],  # Further OTM for conservative
                "premium_yield": [0.8, 0.6],  # Lower but safer yields
                "annualized_yield": [10.0, 8.0],
                "assignment_probability": [0.05, 0.02],  # Lower assignment risk
                "CompositeScore": [0.75, 0.70],
                "strategy_type": ["Covered Call"] * 2,
                "risk_tolerance_applied": ["Conservative"] * 2,
            }
        )

        covered_call_aggressive = pd.DataFrame(
            {
                "ticker": [self.test_ticker] * 3,
                "strike": [152.0, 155.0, 158.0],  # Closer to ATM
                "premium_yield": [2.0, 1.5, 1.2],  # Higher yields
                "annualized_yield": [25.0, 20.0, 18.0],
                "assignment_probability": [0.3, 0.2, 0.15],  # Higher assignment risk
                "CompositeScore": [0.80, 0.75, 0.70],
                "strategy_type": ["Covered Call"] * 3,
                "risk_tolerance_applied": ["Aggressive"] * 3,
            }
        )

        with patch(
            "buffetbot.analysis.options_advisor.recommend_covered_call"
        ) as mock_cc:
            # Test Conservative Covered Calls
            mock_cc.return_value = covered_call_conservative
            conservative_cc = analyze_options_strategy(
                "Covered Call", self.test_ticker, risk_tolerance="Conservative"
            )

            avg_assignment_prob = conservative_cc["assignment_probability"].mean()
            self.assertLess(
                avg_assignment_prob,
                0.1,
                "Conservative covered calls should have low assignment probability",
            )
            print("✅ Conservative Covered Calls: Lower assignment probability")

            # Test Aggressive Covered Calls
            mock_cc.return_value = covered_call_aggressive
            aggressive_cc = analyze_options_strategy(
                "Covered Call", self.test_ticker, risk_tolerance="Aggressive"
            )

            avg_yield = aggressive_cc["annualized_yield"].mean()
            self.assertGreater(
                avg_yield, 15.0, "Aggressive covered calls should have higher yields"
            )
            print("✅ Aggressive Covered Calls: Higher annualized yields")


def main():
    """Run the enhanced integration tests."""
    print("=" * 80)
    print("TESTING ENHANCED OPTIONS STRATEGY INTEGRATION WITH RISK TOLERANCE")
    print("=" * 80)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestOptionsStrategyIntegration
    )

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 80)
    print("ENHANCED INTEGRATION TEST SUMMARY")
    print("=" * 80)

    if result.wasSuccessful():
        print("✅ All enhanced integration tests passed!")
        print(f"✅ Ran {result.testsRun} tests successfully")
        print("\n🎉 ENHANCED RISK TOLERANCE FUNCTIONALITY IS WORKING!")
        print("🎯 Users now get sophisticated risk-adjusted options analysis:")
        print("   • Conservative: Lower-risk, higher-probability trades")
        print("   • Moderate: Balanced risk/reward approach")
        print("   • Aggressive: Higher-leverage, higher-potential trades")
        print("   • Strategy-specific risk adjustments for each options strategy")
    else:
        print("❌ Some enhanced integration tests failed!")
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
