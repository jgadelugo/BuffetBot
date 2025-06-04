"""Test suite for scoring weights type safety fix."""

import unittest
from unittest.mock import MagicMock

from buffetbot.analysis.options.core.domain_models import ScoringWeights
from buffetbot.dashboard.views.options_advisor import safe_get_weight


class TestScoringWeightsTypeSafety:
    """Test cases to verify the scoring weights type safety fix."""

    def test_safe_get_weight_with_scoring_weights_object(self):
        """Test safe_get_weight with ScoringWeights object."""
        weights = ScoringWeights(
            rsi=0.25, beta=0.15, momentum=0.25, iv=0.15, forecast=0.20
        )

        # Test successful retrieval
        assert safe_get_weight(weights, "rsi") == 0.25
        assert safe_get_weight(weights, "beta") == 0.15
        assert safe_get_weight(weights, "momentum") == 0.25
        assert safe_get_weight(weights, "iv") == 0.15
        assert safe_get_weight(weights, "forecast") == 0.20

        # Test missing indicator with default
        assert safe_get_weight(weights, "unknown", 0.5) == 0.5
        assert safe_get_weight(weights, "unknown") == 0.0

    def test_safe_get_weight_with_dictionary(self):
        """Test safe_get_weight with dictionary."""
        weights = {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }

        # Test successful retrieval
        assert safe_get_weight(weights, "rsi") == 0.20
        assert safe_get_weight(weights, "beta") == 0.20
        assert safe_get_weight(weights, "momentum") == 0.20
        assert safe_get_weight(weights, "iv") == 0.20
        assert safe_get_weight(weights, "forecast") == 0.20

        # Test missing indicator with default
        assert safe_get_weight(weights, "unknown", 0.5) == 0.5
        assert safe_get_weight(weights, "unknown") == 0.0

    def test_safe_get_weight_with_mock_object(self):
        """Test safe_get_weight with mock object that has attributes."""

        # Create a simple object with attributes (not a full mock)
        class SimpleWeights:
            def __init__(self):
                self.rsi = 0.30
                self.beta = 0.10

        mock_weights = SimpleWeights()

        # Should use attribute access
        assert safe_get_weight(mock_weights, "rsi") == 0.30
        assert safe_get_weight(mock_weights, "beta") == 0.10

        # Should return default for missing attributes
        assert safe_get_weight(mock_weights, "unknown", 0.7) == 0.7

    def test_safe_get_weight_with_none(self):
        """Test safe_get_weight with None input."""
        assert safe_get_weight(None, "rsi") == 0.0
        assert safe_get_weight(None, "rsi", 0.5) == 0.5

    def test_safe_get_weight_with_invalid_input(self):
        """Test safe_get_weight with invalid input types."""
        # String input
        assert safe_get_weight("invalid", "rsi") == 0.0
        assert safe_get_weight("invalid", "rsi", 0.5) == 0.5

        # Number input
        assert safe_get_weight(123, "rsi") == 0.0
        assert safe_get_weight(123, "rsi", 0.5) == 0.5

    def test_scoring_weights_integration(self):
        """Integration test to ensure ScoringWeights and safe_get_weight work together."""
        # Test default ScoringWeights
        default_weights = ScoringWeights()

        # All weights should be 0.20 (equal distribution)
        for indicator in ["rsi", "beta", "momentum", "iv", "forecast"]:
            weight = safe_get_weight(default_weights, indicator)
            assert (
                weight == 0.20
            ), f"Default {indicator} weight should be 0.20, got {weight}"

        # Test custom ScoringWeights
        custom_weights = ScoringWeights(
            rsi=0.30, beta=0.10, momentum=0.30, iv=0.10, forecast=0.20
        )

        assert safe_get_weight(custom_weights, "rsi") == 0.30
        assert safe_get_weight(custom_weights, "beta") == 0.10
        assert safe_get_weight(custom_weights, "momentum") == 0.30

    def test_backwards_compatibility(self):
        """Test that the fix maintains backwards compatibility."""
        # Test old-style dictionary access still works
        old_style_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.15,
            "forecast": 0.20,
        }

        # Should work exactly like the old .get() method
        assert safe_get_weight(old_style_weights, "rsi", 0) == 0.25
        assert safe_get_weight(old_style_weights, "unknown", 0) == 0

        # Test new-style ScoringWeights object
        new_style_weights = ScoringWeights(
            rsi=0.25, beta=0.15, momentum=0.25, iv=0.15, forecast=0.20
        )

        # Should return the same values
        for indicator in ["rsi", "beta", "momentum", "iv", "forecast"]:
            old_value = safe_get_weight(old_style_weights, indicator, 0)
            new_value = safe_get_weight(new_style_weights, indicator, 0)
            assert old_value == new_value, f"Values should match for {indicator}"


if __name__ == "__main__":
    # Simple test runner
    test_suite = TestScoringWeightsTypeSafety()

    print("🔍 Testing scoring weights type safety fix...")

    try:
        test_suite.test_safe_get_weight_with_scoring_weights_object()
        print("✅ ScoringWeights object test passed")

        test_suite.test_safe_get_weight_with_dictionary()
        print("✅ Dictionary test passed")

        test_suite.test_safe_get_weight_with_mock_object()
        print("✅ Mock object test passed")

        test_suite.test_safe_get_weight_with_none()
        print("✅ None input test passed")

        test_suite.test_safe_get_weight_with_invalid_input()
        print("✅ Invalid input test passed")

        test_suite.test_scoring_weights_integration()
        print("✅ Integration test passed")

        test_suite.test_backwards_compatibility()
        print("✅ Backwards compatibility test passed")

        print("🎉 All scoring weights fix tests passed!")
        print("🔧 ScoringWeights error should be resolved in production")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
