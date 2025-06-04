"""
Unit tests for options scoring weights functionality.

This module tests the scoring weights configuration, strategy-specific weights,
and the dashboard utilities that display and use these weights.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from buffetbot.analysis.options.config.scoring_weights import (
    DEFAULT_SCORING_WEIGHTS_CONFIG,
    get_scoring_weights,
    normalize_scoring_weights,
)
from buffetbot.analysis.options.core.domain_models import ScoringWeights, StrategyType
from buffetbot.dashboard.utils.enhanced_options_analysis import (
    get_scoring_indicator_names,
    get_strategy_specific_weights,
    get_total_scoring_indicators,
    normalize_weights,
    validate_custom_weights,
)


class TestScoringWeightsConfiguration:
    """Test the core scoring weights configuration system."""

    def test_default_weights_structure(self):
        """Test that default weights are properly structured."""
        default_weights = DEFAULT_SCORING_WEIGHTS_CONFIG.default_weights

        # Test that default weights sum to 1.0
        total = sum(
            [
                default_weights.rsi,
                default_weights.beta,
                default_weights.momentum,
                default_weights.iv,
                default_weights.forecast,
            ]
        )
        assert (
            abs(total - 1.0) < 0.001
        ), f"Default weights should sum to 1.0, got {total}"

        # Test that all weights are positive
        assert default_weights.rsi > 0
        assert default_weights.beta > 0
        assert default_weights.momentum > 0
        assert default_weights.iv > 0
        assert default_weights.forecast > 0

    def test_long_calls_strategy_weights(self):
        """Test that Long Calls strategy has the correct specific weights."""
        weights = get_scoring_weights(StrategyType.LONG_CALLS)

        # Verify the specific weights as requested by the user
        assert weights.rsi == 0.25, f"RSI weight should be 25%, got {weights.rsi:.0%}"
        assert (
            weights.beta == 0.15
        ), f"Beta weight should be 15%, got {weights.beta:.0%}"
        assert (
            weights.momentum == 0.25
        ), f"Momentum weight should be 25%, got {weights.momentum:.0%}"
        assert weights.iv == 0.15, f"IV weight should be 15%, got {weights.iv:.0%}"
        assert (
            weights.forecast == 0.20
        ), f"Forecast weight should be 20%, got {weights.forecast:.0%}"

        # Verify they sum to 1.0
        total = sum(
            [weights.rsi, weights.beta, weights.momentum, weights.iv, weights.forecast]
        )
        assert (
            abs(total - 1.0) < 0.001
        ), f"Long Calls weights should sum to 1.0, got {total}"

    def test_bull_call_spread_strategy_weights(self):
        """Test that Bull Call Spread strategy has balanced weights."""
        weights = get_scoring_weights(StrategyType.BULL_CALL_SPREAD)

        # Bull Call Spread should have equal weights (20% each)
        assert weights.rsi == 0.20, f"RSI weight should be 20%, got {weights.rsi:.0%}"
        assert (
            weights.beta == 0.20
        ), f"Beta weight should be 20%, got {weights.beta:.0%}"
        assert (
            weights.momentum == 0.20
        ), f"Momentum weight should be 20%, got {weights.momentum:.0%}"
        assert weights.iv == 0.20, f"IV weight should be 20%, got {weights.iv:.0%}"
        assert (
            weights.forecast == 0.20
        ), f"Forecast weight should be 20%, got {weights.forecast:.0%}"

    def test_covered_call_strategy_weights(self):
        """Test that Covered Call strategy emphasizes income-focused metrics."""
        weights = get_scoring_weights(StrategyType.COVERED_CALL)

        # Covered calls should emphasize IV (income generation)
        assert (
            weights.iv == 0.30
        ), f"IV weight should be 30% for covered calls, got {weights.iv:.0%}"
        assert (
            weights.beta == 0.25
        ), f"Beta weight should be 25% for covered calls, got {weights.beta:.0%}"

        # Verify they sum to 1.0
        total = sum(
            [weights.rsi, weights.beta, weights.momentum, weights.iv, weights.forecast]
        )
        assert (
            abs(total - 1.0) < 0.001
        ), f"Covered Call weights should sum to 1.0, got {total}"

    def test_cash_secured_put_strategy_weights(self):
        """Test that Cash-Secured Put strategy emphasizes entry timing metrics."""
        weights = get_scoring_weights(StrategyType.CASH_SECURED_PUT)

        # CSPs should emphasize RSI and IV
        assert (
            weights.rsi == 0.25
        ), f"RSI weight should be 25% for CSPs, got {weights.rsi:.0%}"
        assert (
            weights.iv == 0.25
        ), f"IV weight should be 25% for CSPs, got {weights.iv:.0%}"

        # Verify they sum to 1.0
        total = sum(
            [weights.rsi, weights.beta, weights.momentum, weights.iv, weights.forecast]
        )
        assert abs(total - 1.0) < 0.001, f"CSP weights should sum to 1.0, got {total}"

    def test_get_weights_without_strategy(self):
        """Test getting default weights when no strategy is specified."""
        weights = get_scoring_weights(None)

        # Should return default weights (equal distribution)
        assert weights.rsi == 0.20
        assert weights.beta == 0.20
        assert weights.momentum == 0.20
        assert weights.iv == 0.20
        assert weights.forecast == 0.20

    def test_scoring_weights_to_dict(self):
        """Test converting ScoringWeights to dictionary."""
        weights = get_scoring_weights(StrategyType.LONG_CALLS)
        weights_dict = weights.to_dict()

        # Test structure
        assert isinstance(weights_dict, dict)
        assert set(weights_dict.keys()) == {"rsi", "beta", "momentum", "iv", "forecast"}

        # Test values match
        assert weights_dict["rsi"] == weights.rsi
        assert weights_dict["beta"] == weights.beta
        assert weights_dict["momentum"] == weights.momentum
        assert weights_dict["iv"] == weights.iv
        assert weights_dict["forecast"] == weights.forecast


class TestDashboardUtilities:
    """Test the dashboard utility functions for scoring weights."""

    def test_get_strategy_specific_weights(self):
        """Test the dashboard utility function for getting strategy weights."""
        # Test Long Calls
        weights = get_strategy_specific_weights("Long Calls")

        assert isinstance(weights, dict)
        assert weights["rsi"] == 0.25
        assert weights["beta"] == 0.15
        assert weights["momentum"] == 0.25
        assert weights["iv"] == 0.15
        assert weights["forecast"] == 0.20

    def test_get_strategy_specific_weights_bull_call_spread(self):
        """Test getting weights for Bull Call Spread."""
        weights = get_strategy_specific_weights("Bull Call Spread")

        # Should have equal weights
        assert all(w == 0.20 for w in weights.values())

    def test_get_strategy_specific_weights_invalid_strategy(self):
        """Test getting weights for an invalid strategy returns fallback."""
        weights = get_strategy_specific_weights("Invalid Strategy")

        # Should return equal weights as fallback
        assert weights["rsi"] == 0.20
        assert weights["beta"] == 0.20
        assert weights["momentum"] == 0.20
        assert weights["iv"] == 0.20
        assert weights["forecast"] == 0.20

    def test_get_total_scoring_indicators(self):
        """Test getting the total number of scoring indicators."""
        total = get_total_scoring_indicators()
        assert total == 5

    def test_get_scoring_indicator_names(self):
        """Test getting the names of scoring indicators."""
        names = get_scoring_indicator_names()

        assert isinstance(names, list)
        assert len(names) == 5
        assert set(names) == {"rsi", "beta", "momentum", "iv", "forecast"}

    def test_validate_custom_weights_valid(self):
        """Test validating valid custom weights."""
        valid_weights = {
            "rsi": 0.30,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.15,
            "forecast": 0.15,
        }

        is_valid, error_message = validate_custom_weights(valid_weights)
        assert is_valid == True
        assert error_message == ""

    def test_validate_custom_weights_missing_keys(self):
        """Test validating custom weights with missing keys."""
        invalid_weights = {
            "rsi": 0.30,
            "beta": 0.20,
            # Missing momentum, iv, forecast
        }

        is_valid, error_message = validate_custom_weights(invalid_weights)
        assert is_valid == False
        assert "Missing keys" in error_message

    def test_validate_custom_weights_extra_keys(self):
        """Test validating custom weights with extra keys."""
        invalid_weights = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
            "extra_key": 0.10,  # Extra key
        }

        is_valid, error_message = validate_custom_weights(invalid_weights)
        assert is_valid == False
        assert "Extra keys" in error_message

    def test_validate_custom_weights_invalid_sum(self):
        """Test validating custom weights that don't sum to 1.0."""
        invalid_weights = {
            "rsi": 0.50,
            "beta": 0.50,
            "momentum": 0.50,
            "iv": 0.00,
            "forecast": 0.00,
        }

        is_valid, error_message = validate_custom_weights(invalid_weights)
        assert is_valid == False
        assert "must sum to 1.0" in error_message

    def test_validate_custom_weights_negative_values(self):
        """Test validating custom weights with negative values."""
        invalid_weights = {
            "rsi": -0.10,
            "beta": 0.30,
            "momentum": 0.30,
            "iv": 0.30,
            "forecast": 0.20,
        }

        is_valid, error_message = validate_custom_weights(invalid_weights)
        assert is_valid == False
        assert "must be a non-negative number" in error_message

    def test_normalize_weights(self):
        """Test normalizing weights to sum to 1.0."""
        unnormalized_weights = {
            "rsi": 0.50,
            "beta": 0.30,
            "momentum": 0.20,
            "iv": 0.10,
            "forecast": 0.00,
        }

        normalized = normalize_weights(unnormalized_weights)

        # Should sum to 1.0
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001

        # Proportions should be maintained
        original_total = sum(unnormalized_weights.values())
        assert abs(normalized["rsi"] - (0.50 / original_total)) < 0.001

    def test_normalize_weights_all_zero(self):
        """Test normalizing weights when all are zero."""
        zero_weights = {
            "rsi": 0.0,
            "beta": 0.0,
            "momentum": 0.0,
            "iv": 0.0,
            "forecast": 0.0,
        }

        normalized = normalize_weights(zero_weights)

        # Should distribute equally
        assert all(abs(w - 0.20) < 0.001 for w in normalized.values())


class TestWeightNormalization:
    """Test the weight normalization functionality."""

    def test_normalize_scoring_weights_all_available(self):
        """Test normalizing weights when all sources are available."""
        input_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }
        available_sources = ["rsi", "beta", "momentum", "iv", "forecast"]

        normalized = normalize_scoring_weights(input_weights, available_sources)

        # Should be unchanged since all sources are available
        assert normalized == input_weights
        assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_normalize_scoring_weights_missing_sources(self):
        """Test normalizing weights when some sources are missing."""
        input_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }
        available_sources = ["rsi", "beta", "momentum", "iv"]  # forecast missing

        normalized = normalize_scoring_weights(input_weights, available_sources)

        # Should redistribute weights proportionally
        assert "forecast" not in normalized
        assert abs(sum(normalized.values()) - 1.0) < 0.001

        # Original proportions should be maintained among available sources
        available_weight = sum(input_weights[k] for k in available_sources)
        expected_rsi = input_weights["rsi"] / available_weight
        assert abs(normalized["rsi"] - expected_rsi) < 0.001

    def test_normalize_scoring_weights_single_source(self):
        """Test normalizing weights when only one source is available."""
        input_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }
        available_sources = ["rsi"]

        normalized = normalize_scoring_weights(input_weights, available_sources)

        # Should give 100% weight to the single available source
        assert normalized == {"rsi": 1.0}

    def test_normalize_scoring_weights_no_sources(self):
        """Test normalizing weights when no sources are available."""
        input_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }
        available_sources = []

        normalized = normalize_scoring_weights(input_weights, available_sources)

        # Should return empty dict
        assert normalized == {}


class TestIntegrationWithConfig:
    """Test integration between different components of the scoring system."""

    def test_config_matches_dashboard_utility(self):
        """Test that config weights match what dashboard utility returns."""
        strategy_types = [
            ("Long Calls", StrategyType.LONG_CALLS),
            ("Bull Call Spread", StrategyType.BULL_CALL_SPREAD),
            ("Covered Call", StrategyType.COVERED_CALL),
            ("Cash-Secured Put", StrategyType.CASH_SECURED_PUT),
        ]

        for strategy_str, strategy_enum in strategy_types:
            config_weights = get_scoring_weights(strategy_enum)
            utility_weights = get_strategy_specific_weights(strategy_str)

            # Should match exactly
            assert utility_weights["rsi"] == config_weights.rsi
            assert utility_weights["beta"] == config_weights.beta
            assert utility_weights["momentum"] == config_weights.momentum
            assert utility_weights["iv"] == config_weights.iv
            assert utility_weights["forecast"] == config_weights.forecast

    def test_long_calls_weights_regression(self):
        """Regression test to ensure Long Calls weights don't change unexpectedly."""
        weights = get_scoring_weights(StrategyType.LONG_CALLS)

        # These are the specific weights requested by the user
        expected_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }

        for indicator, expected_value in expected_weights.items():
            actual_value = getattr(weights, indicator)
            assert actual_value == expected_value, (
                f"Long Calls {indicator} weight changed! "
                f"Expected {expected_value:.0%}, got {actual_value:.0%}"
            )

    def test_all_strategies_have_valid_weights(self):
        """Test that all strategy types have valid weights configuration."""
        for strategy in StrategyType:
            weights = get_scoring_weights(strategy)

            # Test that weights sum to 1.0
            total = sum(
                [
                    weights.rsi,
                    weights.beta,
                    weights.momentum,
                    weights.iv,
                    weights.forecast,
                ]
            )
            assert (
                abs(total - 1.0) < 0.001
            ), f"{strategy.value} weights don't sum to 1.0: {total}"

            # Test that all weights are non-negative
            assert weights.rsi >= 0, f"{strategy.value} has negative RSI weight"
            assert weights.beta >= 0, f"{strategy.value} has negative Beta weight"
            assert (
                weights.momentum >= 0
            ), f"{strategy.value} has negative Momentum weight"
            assert weights.iv >= 0, f"{strategy.value} has negative IV weight"
            assert (
                weights.forecast >= 0
            ), f"{strategy.value} has negative Forecast weight"


@pytest.mark.parametrize(
    "strategy_name,expected_weights",
    [
        (
            "Long Calls",
            {"rsi": 0.25, "beta": 0.15, "momentum": 0.25, "iv": 0.15, "forecast": 0.20},
        ),
        (
            "Long Puts",
            {"rsi": 0.30, "beta": 0.20, "momentum": 0.20, "iv": 0.15, "forecast": 0.15},
        ),
        (
            "Bull Call Spread",
            {"rsi": 0.20, "beta": 0.20, "momentum": 0.20, "iv": 0.20, "forecast": 0.20},
        ),
        (
            "Bear Put Spread",
            {"rsi": 0.30, "beta": 0.15, "momentum": 0.20, "iv": 0.20, "forecast": 0.15},
        ),
        (
            "Bull Put Spread",
            {"rsi": 0.20, "beta": 0.25, "momentum": 0.15, "iv": 0.25, "forecast": 0.15},
        ),
        (
            "Bear Call Spread",
            {"rsi": 0.25, "beta": 0.15, "momentum": 0.20, "iv": 0.25, "forecast": 0.15},
        ),
        (
            "Covered Call",
            {"rsi": 0.15, "beta": 0.25, "momentum": 0.15, "iv": 0.30, "forecast": 0.15},
        ),
        (
            "Cash-Secured Put",
            {"rsi": 0.25, "beta": 0.20, "momentum": 0.15, "iv": 0.25, "forecast": 0.15},
        ),
        (
            "Iron Condor",
            {"rsi": 0.15, "beta": 0.30, "momentum": 0.10, "iv": 0.35, "forecast": 0.10},
        ),
        (
            "Iron Butterfly",
            {"rsi": 0.15, "beta": 0.25, "momentum": 0.15, "iv": 0.35, "forecast": 0.10},
        ),
        (
            "Calendar Spread",
            {"rsi": 0.15, "beta": 0.20, "momentum": 0.10, "iv": 0.40, "forecast": 0.15},
        ),
        (
            "Long Straddle",
            {"rsi": 0.15, "beta": 0.15, "momentum": 0.20, "iv": 0.40, "forecast": 0.10},
        ),
        (
            "Long Strangle",
            {"rsi": 0.15, "beta": 0.15, "momentum": 0.20, "iv": 0.40, "forecast": 0.10},
        ),
        (
            "Short Straddle",
            {"rsi": 0.20, "beta": 0.25, "momentum": 0.15, "iv": 0.30, "forecast": 0.10},
        ),
        (
            "Short Strangle",
            {"rsi": 0.20, "beta": 0.25, "momentum": 0.15, "iv": 0.30, "forecast": 0.10},
        ),
    ],
)
def test_strategy_weights_parametrized(strategy_name, expected_weights):
    """Parametrized test for all strategy weights."""
    actual_weights = get_strategy_specific_weights(strategy_name)

    for indicator, expected_value in expected_weights.items():
        assert actual_weights[indicator] == expected_value, (
            f"{strategy_name} {indicator} weight mismatch: "
            f"expected {expected_value:.0%}, got {actual_weights[indicator]:.0%}"
        )
