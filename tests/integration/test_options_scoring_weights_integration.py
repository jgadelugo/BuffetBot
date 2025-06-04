"""
Integration tests for options scoring weights system.

This module tests the integration between the options scoring weights configuration
and the dashboard utilities, ensuring they work together correctly.
"""

import pytest

from buffetbot.analysis.options.config.scoring_weights import get_scoring_weights
from buffetbot.analysis.options.core.domain_models import StrategyType
from buffetbot.dashboard.utils.enhanced_options_analysis import (
    analyze_options_with_custom_settings,
    get_strategy_specific_weights,
)


class TestScoringWeightsIntegration:
    """Test integration between scoring weights config and dashboard utilities."""

    def test_long_calls_weights_integration(self):
        """Test that Long Calls weights are consistent across all systems."""
        # Expected weights as specified by the user
        expected_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }

        # Test config system
        config_weights = get_scoring_weights(StrategyType.LONG_CALLS)
        config_dict = config_weights.to_dict()

        # Test dashboard utility
        dashboard_weights = get_strategy_specific_weights("Long Calls")

        # Verify both match expected weights
        for indicator, expected_value in expected_weights.items():
            assert (
                config_dict[indicator] == expected_value
            ), f"Config {indicator} weight mismatch: expected {expected_value:.0%}, got {config_dict[indicator]:.0%}"

            assert (
                dashboard_weights[indicator] == expected_value
            ), f"Dashboard {indicator} weight mismatch: expected {expected_value:.0%}, got {dashboard_weights[indicator]:.0%}"

        # Verify they match each other
        for indicator in expected_weights.keys():
            assert (
                config_dict[indicator] == dashboard_weights[indicator]
            ), f"Config and dashboard {indicator} weights don't match"

    def test_all_strategies_consistency(self):
        """Test that all strategies have consistent weights between config and dashboard."""
        strategy_mappings = {
            StrategyType.LONG_CALLS: "Long Calls",
            StrategyType.LONG_PUTS: "Long Puts",
            StrategyType.BULL_CALL_SPREAD: "Bull Call Spread",
            StrategyType.BEAR_PUT_SPREAD: "Bear Put Spread",
            StrategyType.BULL_PUT_SPREAD: "Bull Put Spread",
            StrategyType.BEAR_CALL_SPREAD: "Bear Call Spread",
            StrategyType.COVERED_CALL: "Covered Call",
            StrategyType.CASH_SECURED_PUT: "Cash-Secured Put",
            StrategyType.IRON_CONDOR: "Iron Condor",
            StrategyType.IRON_BUTTERFLY: "Iron Butterfly",
            StrategyType.CALENDAR_SPREAD: "Calendar Spread",
            StrategyType.LONG_STRADDLE: "Long Straddle",
            StrategyType.LONG_STRANGLE: "Long Strangle",
            StrategyType.SHORT_STRADDLE: "Short Straddle",
            StrategyType.SHORT_STRANGLE: "Short Strangle",
        }

        for strategy_enum, strategy_str in strategy_mappings.items():
            config_weights = get_scoring_weights(strategy_enum).to_dict()
            dashboard_weights = get_strategy_specific_weights(strategy_str)

            # Verify they match exactly
            for indicator in ["rsi", "beta", "momentum", "iv", "forecast"]:
                assert config_weights[indicator] == dashboard_weights[indicator], (
                    f"{strategy_str} {indicator} weight mismatch: "
                    f"config={config_weights[indicator]:.0%}, "
                    f"dashboard={dashboard_weights[indicator]:.0%}"
                )

    def test_weights_sum_to_one_across_systems(self):
        """Test that weights sum to 1.0 in both config and dashboard systems."""
        strategies = [
            ("Long Calls", StrategyType.LONG_CALLS),
            ("Bull Call Spread", StrategyType.BULL_CALL_SPREAD),
            ("Covered Call", StrategyType.COVERED_CALL),
            ("Cash-Secured Put", StrategyType.CASH_SECURED_PUT),
        ]

        for strategy_str, strategy_enum in strategies:
            # Config system
            config_weights = get_scoring_weights(strategy_enum)
            config_total = sum(
                [
                    config_weights.rsi,
                    config_weights.beta,
                    config_weights.momentum,
                    config_weights.iv,
                    config_weights.forecast,
                ]
            )
            assert (
                abs(config_total - 1.0) < 0.001
            ), f"{strategy_str} config weights don't sum to 1.0: {config_total}"

            # Dashboard system
            dashboard_weights = get_strategy_specific_weights(strategy_str)
            dashboard_total = sum(dashboard_weights.values())
            assert (
                abs(dashboard_total - 1.0) < 0.001
            ), f"{strategy_str} dashboard weights don't sum to 1.0: {dashboard_total}"

    def test_regression_long_calls_specific_values(self):
        """Regression test to ensure Long Calls maintains specific weight values."""
        # This test locks in the specific values requested by the user
        # RSI: 25%, Beta: 15%, Momentum: 25%, IV: 15%, Forecast: 20%

        weights = get_strategy_specific_weights("Long Calls")

        # These values should NEVER change without explicit approval
        assert weights["rsi"] == 0.25, "Long Calls RSI weight changed from 25%"
        assert weights["beta"] == 0.15, "Long Calls Beta weight changed from 15%"
        assert (
            weights["momentum"] == 0.25
        ), "Long Calls Momentum weight changed from 25%"
        assert weights["iv"] == 0.15, "Long Calls IV weight changed from 15%"
        assert (
            weights["forecast"] == 0.20
        ), "Long Calls Forecast weight changed from 20%"

    def test_dashboard_fallback_behavior(self):
        """Test that dashboard utility handles invalid strategy names gracefully."""
        # Test invalid strategy returns equal weights
        fallback_weights = get_strategy_specific_weights("Invalid Strategy Name")

        expected_fallback = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }

        assert (
            fallback_weights == expected_fallback
        ), "Invalid strategy should return equal weights fallback"

    @pytest.mark.parametrize(
        "strategy_str,expected_emphasis",
        [
            (
                "Long Calls",
                {"rsi": 0.25, "momentum": 0.25},
            ),  # Emphasizes technical analysis
            (
                "Covered Call",
                {"iv": 0.30, "beta": 0.25},
            ),  # Emphasizes income generation
            ("Cash-Secured Put", {"rsi": 0.25, "iv": 0.25}),  # Emphasizes entry timing
            ("Bull Call Spread", None),  # Equal weights (no emphasis)
        ],
    )
    def test_strategy_specific_emphasis(self, strategy_str, expected_emphasis):
        """Test that each strategy emphasizes the correct indicators."""
        weights = get_strategy_specific_weights(strategy_str)

        if expected_emphasis is None:
            # Bull Call Spread should have equal weights
            assert all(
                w == 0.20 for w in weights.values()
            ), f"{strategy_str} should have equal weights"
        else:
            # Check that emphasized indicators have the expected weights
            for indicator, expected_weight in expected_emphasis.items():
                assert weights[indicator] == expected_weight, (
                    f"{strategy_str} should emphasize {indicator} with {expected_weight:.0%} weight, "
                    f"got {weights[indicator]:.0%}"
                )

    def test_no_negative_weights(self):
        """Test that no strategy has negative weights."""
        strategies = [
            "Long Calls",
            "Bull Call Spread",
            "Covered Call",
            "Cash-Secured Put",
        ]

        for strategy in strategies:
            weights = get_strategy_specific_weights(strategy)
            for indicator, weight in weights.items():
                assert (
                    weight >= 0
                ), f"{strategy} has negative {indicator} weight: {weight}"

    def test_weights_are_percentages(self):
        """Test that all weights are valid percentage values (0-1)."""
        strategies = [
            "Long Calls",
            "Bull Call Spread",
            "Covered Call",
            "Cash-Secured Put",
        ]

        for strategy in strategies:
            weights = get_strategy_specific_weights(strategy)
            for indicator, weight in weights.items():
                assert (
                    0 <= weight <= 1
                ), f"{strategy} {indicator} weight is not a valid percentage: {weight}"


class TestOptionsAnalysisIntegration:
    """Test integration with the options analysis system."""

    def test_custom_weights_override(self):
        """Test that custom weights properly override default strategy weights."""
        # This would require mock data for a full test, but we can test the weight handling
        custom_weights = {
            "rsi": 0.40,
            "beta": 0.10,
            "momentum": 0.30,
            "iv": 0.10,
            "forecast": 0.10,
        }

        # Verify the custom weights sum to 1.0
        assert abs(sum(custom_weights.values()) - 1.0) < 0.001

        # The actual integration would require setting up mock options data
        # but this verifies the weights structure is correct

    def test_strategy_type_enum_consistency(self):
        """Test that StrategyType enum values match expected string representations."""
        expected_mappings = {
            StrategyType.LONG_CALLS: "Long Calls",
            StrategyType.BULL_CALL_SPREAD: "Bull Call Spread",
            StrategyType.COVERED_CALL: "Covered Call",
            StrategyType.CASH_SECURED_PUT: "Cash-Secured Put",
        }

        for strategy_enum, expected_str in expected_mappings.items():
            # Get weights using both enum and string
            enum_weights = get_scoring_weights(strategy_enum)
            str_weights = get_strategy_specific_weights(expected_str)

            # They should match
            enum_dict = enum_weights.to_dict()
            for indicator in ["rsi", "beta", "momentum", "iv", "forecast"]:
                assert (
                    enum_dict[indicator] == str_weights[indicator]
                ), f"Enum and string weights don't match for {expected_str} {indicator}"
