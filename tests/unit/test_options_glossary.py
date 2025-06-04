"""
Unit tests for options strategy glossary functionality.

This module tests the comprehensive options strategy definitions,
search capabilities, and utility functions for the glossary.
"""

import pytest

from buffetbot.glossary import (
    OPTIONS_STRATEGIES,
    get_all_definitions,
    get_metrics_by_category,
    get_options_strategy_info,
    get_strategies_by_outlook,
    get_strategies_by_risk_profile,
    search_options_strategies,
)


class TestOptionsStrategiesGlossary:
    """Test the options strategies glossary functionality."""

    def test_all_strategies_present(self):
        """Test that all expected strategies are defined in the glossary."""
        expected_strategies = {
            "long_calls",
            "long_puts",
            "covered_call",
            "cash_secured_put",
            "bull_call_spread",
            "bear_put_spread",
            "bull_put_spread",
            "bear_call_spread",
            "iron_condor",
            "iron_butterfly",
            "calendar_spread",
            "long_straddle",
            "long_strangle",
            "short_straddle",
            "short_strangle",
        }

        actual_strategies = set(OPTIONS_STRATEGIES.keys())
        assert actual_strategies == expected_strategies, (
            f"Missing strategies: {expected_strategies - actual_strategies}, "
            f"Extra strategies: {actual_strategies - expected_strategies}"
        )

    def test_strategy_structure_completeness(self):
        """Test that all strategies have complete information."""
        required_fields = {
            "name",
            "category",
            "description",
            "objective",
            "market_outlook",
            "risk_profile",
            "default_weights",
            "weights_rationale",
            "max_profit",
            "max_loss",
            "breakeven",
        }

        for strategy_key, strategy in OPTIONS_STRATEGIES.items():
            actual_fields = set(strategy.keys())
            assert (
                actual_fields == required_fields
            ), f"{strategy_key} missing fields: {required_fields - actual_fields}"

    def test_all_categories_are_options(self):
        """Test that all strategies are categorized as 'options'."""
        for strategy_key, strategy in OPTIONS_STRATEGIES.items():
            assert (
                strategy["category"] == "options"
            ), f"{strategy_key} has incorrect category: {strategy['category']}"

    def test_default_weights_structure(self):
        """Test that all strategies have proper default weights structure."""
        required_indicators = {"rsi", "beta", "momentum", "iv", "forecast"}

        for strategy_key, strategy in OPTIONS_STRATEGIES.items():
            weights = strategy["default_weights"]

            # Check all indicators present
            actual_indicators = set(weights.keys())
            assert (
                actual_indicators == required_indicators
            ), f"{strategy_key} missing indicators: {required_indicators - actual_indicators}"

            # Check weights sum to 1.0
            total_weight = sum(weights.values())
            assert (
                abs(total_weight - 1.0) < 0.001
            ), f"{strategy_key} weights don't sum to 1.0: {total_weight}"

            # Check all weights are valid percentages
            for indicator, weight in weights.items():
                assert (
                    0 <= weight <= 1
                ), f"{strategy_key} {indicator} weight invalid: {weight}"

    def test_long_calls_definition(self):
        """Test the Long Calls strategy definition in detail."""
        long_calls = OPTIONS_STRATEGIES["long_calls"]

        assert long_calls["name"] == "Long Calls"
        assert "bullish" in long_calls["market_outlook"].lower()
        assert "unlimited" in long_calls["max_profit"].lower()
        assert "limited" in long_calls["max_loss"].lower()

        # Check specific weights
        weights = long_calls["default_weights"]
        assert weights["rsi"] == 0.25
        assert weights["beta"] == 0.15
        assert weights["momentum"] == 0.25
        assert weights["iv"] == 0.15
        assert weights["forecast"] == 0.20

    def test_volatility_strategies_weights(self):
        """Test that volatility strategies emphasize IV appropriately."""
        volatility_strategies = ["long_straddle", "long_strangle", "calendar_spread"]

        for strategy_key in volatility_strategies:
            strategy = OPTIONS_STRATEGIES[strategy_key]
            weights = strategy["default_weights"]

            # Volatility strategies should have high IV weights
            assert (
                weights["iv"] >= 0.35
            ), f"{strategy_key} should emphasize IV but has weight: {weights['iv']}"

    def test_income_strategies_characteristics(self):
        """Test that income strategies have appropriate characteristics."""
        income_strategies = [
            "covered_call",
            "cash_secured_put",
            "iron_condor",
            "iron_butterfly",
        ]

        for strategy_key in income_strategies:
            strategy = OPTIONS_STRATEGIES[strategy_key]

            # Income strategies should mention premium or income
            description = strategy["description"].lower()
            objective = strategy["objective"].lower()

            assert (
                "income" in description
                or "income" in objective
                or "premium" in description
                or "premium" in objective
            ), f"{strategy_key} should be income-focused but description doesn't mention income/premium"


class TestGlossaryUtilityFunctions:
    """Test the utility functions for the glossary."""

    def test_get_metrics_by_category_options(self):
        """Test getting options strategies by category."""
        options_strategies = get_metrics_by_category("options")

        assert isinstance(options_strategies, dict)
        assert len(options_strategies) == len(OPTIONS_STRATEGIES)
        assert options_strategies == OPTIONS_STRATEGIES

    def test_get_all_definitions_includes_options(self):
        """Test that get_all_definitions includes options strategies."""
        all_definitions = get_all_definitions()

        # Check that options strategies are included
        for strategy_key in OPTIONS_STRATEGIES:
            assert strategy_key in all_definitions
            assert all_definitions[strategy_key] == OPTIONS_STRATEGIES[strategy_key]

    def test_get_options_strategy_info_valid(self):
        """Test getting info for a valid strategy."""
        strategy_info = get_options_strategy_info("long_calls")

        assert strategy_info["name"] == "Long Calls"
        assert strategy_info == OPTIONS_STRATEGIES["long_calls"]

    def test_get_options_strategy_info_invalid(self):
        """Test getting info for an invalid strategy raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_options_strategy_info("invalid_strategy")

        assert "invalid_strategy" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_search_options_strategies_by_name(self):
        """Test searching strategies by name."""
        results = search_options_strategies("call")

        expected_keys = {
            "long_calls",
            "covered_call",
            "bull_call_spread",
            "bear_call_spread",
        }
        actual_keys = set(results.keys())

        assert expected_keys.issubset(
            actual_keys
        ), f"Missing expected strategies: {expected_keys - actual_keys}"

    def test_search_options_strategies_by_description(self):
        """Test searching strategies by description content."""
        results = search_options_strategies("volatility")

        # Should find strategies that mention volatility
        assert len(results) > 0

        # Verify all results actually contain the search term
        for strategy in results.values():
            content = (
                strategy["description"]
                + " "
                + strategy["objective"]
                + " "
                + strategy["market_outlook"]
            ).lower()
            assert "volatility" in content

    def test_get_strategies_by_outlook_bullish(self):
        """Test filtering strategies by bullish outlook."""
        bullish_strategies = get_strategies_by_outlook("bullish")

        assert len(bullish_strategies) > 0

        # Check that all returned strategies have bullish outlook
        for strategy in bullish_strategies.values():
            outlook = strategy["market_outlook"].lower()
            assert "bullish" in outlook or "bull" in outlook

    def test_get_strategies_by_outlook_bearish(self):
        """Test filtering strategies by bearish outlook."""
        bearish_strategies = get_strategies_by_outlook("bearish")

        assert len(bearish_strategies) > 0

        # Check that all returned strategies have bearish outlook
        for strategy in bearish_strategies.values():
            outlook = strategy["market_outlook"].lower()
            assert "bearish" in outlook or "bear" in outlook

    def test_get_strategies_by_outlook_neutral(self):
        """Test filtering strategies by neutral outlook."""
        neutral_strategies = get_strategies_by_outlook("neutral")

        assert len(neutral_strategies) > 0

        # Check that all returned strategies have neutral outlook
        for strategy in neutral_strategies.values():
            outlook = strategy["market_outlook"].lower()
            assert (
                "neutral" in outlook
                or "range-bound" in outlook
                or "low volatility" in outlook
            )

    def test_get_strategies_by_risk_profile_limited(self):
        """Test filtering strategies by limited risk profile."""
        limited_risk_strategies = get_strategies_by_risk_profile("limited")

        assert len(limited_risk_strategies) > 0

        # Check that all returned strategies mention limited risk
        for strategy in limited_risk_strategies.values():
            risk_profile = strategy["risk_profile"].lower()
            assert "limited" in risk_profile

    def test_get_strategies_by_risk_profile_unlimited(self):
        """Test filtering strategies by unlimited risk/reward profile."""
        unlimited_strategies = get_strategies_by_risk_profile("unlimited")

        assert len(unlimited_strategies) > 0

        # Check that all returned strategies mention unlimited
        for strategy in unlimited_strategies.values():
            risk_profile = strategy["risk_profile"].lower()
            assert "unlimited" in risk_profile


class TestStrategyClassifications:
    """Test strategy classifications and groupings."""

    def test_single_leg_strategies(self):
        """Test single-leg strategy characteristics."""
        single_leg = ["long_calls", "long_puts", "covered_call", "cash_secured_put"]

        for strategy_key in single_leg:
            strategy = OPTIONS_STRATEGIES[strategy_key]

            # Single-leg strategies should have clear directional bias or income focus
            objective = strategy["objective"].lower()
            assert (
                "profit" in objective
                or "income" in objective
                or "upward" in objective
                or "downward" in objective
            ), f"{strategy_key} should have clear objective"

    def test_spread_strategies_have_defined_risk(self):
        """Test that spread strategies have defined risk/reward."""
        spread_strategies = [
            "bull_call_spread",
            "bear_put_spread",
            "bull_put_spread",
            "bear_call_spread",
            "iron_condor",
            "iron_butterfly",
        ]

        for strategy_key in spread_strategies:
            strategy = OPTIONS_STRATEGIES[strategy_key]
            risk_profile = strategy["risk_profile"].lower()

            # Spreads should have limited/defined risk
            assert (
                "limited" in risk_profile or "defined" in risk_profile
            ), f"{strategy_key} should have limited/defined risk but has: {risk_profile}"

    def test_volatility_strategies_risk_characteristics(self):
        """Test volatility strategies have appropriate risk characteristics."""
        long_vol = ["long_straddle", "long_strangle"]
        short_vol = ["short_straddle", "short_strangle"]

        # Long volatility strategies should have limited risk, unlimited profit
        for strategy_key in long_vol:
            strategy = OPTIONS_STRATEGIES[strategy_key]
            max_loss = strategy["max_loss"].lower()
            max_profit = strategy["max_profit"].lower()

            # Check for limited loss (premium paid is limited)
            assert (
                "limited" in max_loss or "premium paid" in max_loss
            ), f"{strategy_key} should have limited risk but max_loss is: {strategy['max_loss']}"
            assert "unlimited" in max_profit

        # Short volatility strategies should have limited profit, unlimited risk
        for strategy_key in short_vol:
            strategy = OPTIONS_STRATEGIES[strategy_key]
            max_loss = strategy["max_loss"].lower()
            assert "unlimited" in max_loss


@pytest.mark.parametrize(
    "strategy_key,expected_emphasis",
    [
        ("long_calls", "momentum"),  # Should emphasize momentum and RSI
        ("covered_call", "iv"),  # Should emphasize IV for income
        ("iron_condor", "iv"),  # Should emphasize IV for volatility selling
        ("calendar_spread", "iv"),  # Should emphasize IV for volatility differences
        ("long_straddle", "iv"),  # Should emphasize IV for volatility buying
    ],
)
def test_strategy_weight_emphasis(strategy_key, expected_emphasis):
    """Test that strategies emphasize the expected indicators."""
    strategy = OPTIONS_STRATEGIES[strategy_key]
    weights = strategy["default_weights"]

    # The emphasized indicator should have one of the higher weights
    emphasized_weight = weights[expected_emphasis]
    other_weights = [w for k, w in weights.items() if k != expected_emphasis]

    # Should be at least tied for highest weight
    max_other_weight = max(other_weights)
    assert emphasized_weight >= max_other_weight, (
        f"{strategy_key} should emphasize {expected_emphasis} but {expected_emphasis}="
        f"{emphasized_weight:.0%} vs max_other={max_other_weight:.0%}"
    )
