"""
Unit tests for the modular options analysis architecture.

This module tests individual components of the new modular architecture
in isolation, following pytest best practices.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from buffetbot.analysis.options.config.risk_profiles import get_risk_profile
from buffetbot.analysis.options.config.scoring_weights import get_scoring_weights
from buffetbot.analysis.options.core.domain_models import (
    AnalysisRequest,
    AnalysisResult,
    MarketData,
    RiskProfile,
    RiskTolerance,
    ScoringWeights,
    StrategyType,
    TechnicalIndicators,
    TimeHorizon,
)
from buffetbot.analysis.options.core.exceptions import (
    ErrorContext,
    InsufficientDataError,
    OptionsAdvisorError,
    StrategyValidationError,
)
from buffetbot.analysis.options.scoring.composite_scorer import CompositeScorer
from buffetbot.analysis.options.scoring.technical_indicators import (
    TechnicalIndicatorsCalculator,
)
from buffetbot.analysis.options.scoring.weight_normalizer import (
    normalize_weights,
    redistribute_weight,
    validate_weights,
)


class TestDomainModels:
    """Test domain model validation and behavior."""

    def test_analysis_request_validation(self):
        """Test AnalysisRequest validation logic."""
        # Valid request
        request = AnalysisRequest(
            ticker="AAPL", strategy_type=StrategyType.LONG_CALLS, min_days=180, top_n=5
        )
        assert request.ticker == "AAPL"
        assert request.strategy_type == StrategyType.LONG_CALLS
        assert request.min_days == 180
        assert request.top_n == 5

        # Test default values
        assert request.risk_tolerance == RiskTolerance.CONSERVATIVE
        assert request.time_horizon == TimeHorizon.MEDIUM_TERM

    def test_analysis_request_invalid_ticker(self):
        """Test AnalysisRequest with invalid ticker."""
        with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
            AnalysisRequest(ticker="", strategy_type=StrategyType.LONG_CALLS)

    def test_analysis_request_invalid_min_days(self):
        """Test AnalysisRequest with invalid min_days."""
        with pytest.raises(ValueError, match="min_days must be positive"):
            AnalysisRequest(
                ticker="AAPL", strategy_type=StrategyType.LONG_CALLS, min_days=-1
            )

    def test_analysis_request_string_conversion(self):
        """Test AnalysisRequest enum string conversion."""
        request = AnalysisRequest(
            ticker="AAPL",
            strategy_type="Long Calls",  # String instead of enum
            risk_tolerance="Moderate",  # String instead of enum
            time_horizon="Short-term (1-3 months)",  # String instead of enum
        )

        assert request.strategy_type == StrategyType.LONG_CALLS
        assert request.risk_tolerance == RiskTolerance.MODERATE
        assert request.time_horizon == TimeHorizon.SHORT_TERM

    def test_market_data_validation(self):
        """Test MarketData validation."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        stock_prices = pd.Series([100 + i for i in range(10)], index=dates)
        spy_prices = pd.Series([400 + i for i in range(10)], index=dates)
        options_data = pd.DataFrame({"Strike": [100, 105], "IV": [0.2, 0.3]})

        market_data = MarketData(
            ticker="AAPL",
            stock_prices=stock_prices,
            spy_prices=spy_prices,
            options_data=options_data,
            current_price=105.0,
        )

        assert market_data.ticker == "AAPL"
        assert len(market_data.stock_prices) == 10
        assert market_data.current_price == 105.0

    def test_market_data_empty_prices(self):
        """Test MarketData with empty price data."""
        with pytest.raises(ValueError, match="Stock prices cannot be empty"):
            MarketData(
                ticker="AAPL",
                stock_prices=pd.Series(),
                spy_prices=pd.Series([400]),
                options_data=pd.DataFrame({"Strike": [100]}),
                current_price=105.0,
            )

    def test_technical_indicators_validation(self):
        """Test TechnicalIndicators validation."""
        indicators = TechnicalIndicators(
            rsi=45.0, beta=1.2, momentum=0.05, avg_iv=0.23, forecast_confidence=0.7
        )

        assert indicators.rsi == 45.0
        assert indicators.beta == 1.2
        assert indicators.forecast_confidence == 0.7

    def test_technical_indicators_invalid_rsi(self):
        """Test TechnicalIndicators with invalid RSI."""
        with pytest.raises(ValueError, match="RSI must be between 0 and 100"):
            TechnicalIndicators(
                rsi=150.0,  # Invalid RSI
                beta=1.2,
                momentum=0.05,
                avg_iv=0.23,
                forecast_confidence=0.7,
            )

    def test_scoring_weights_validation(self):
        """Test ScoringWeights validation."""
        weights = ScoringWeights(
            rsi=0.25, beta=0.15, momentum=0.25, iv=0.15, forecast=0.20
        )

        assert abs(sum(weights.to_dict().values()) - 1.0) < 0.001

    def test_scoring_weights_invalid_sum(self):
        """Test ScoringWeights with invalid sum."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ScoringWeights(
                rsi=0.5, beta=0.5, momentum=0.5, iv=0.0, forecast=0.0  # Sum > 1.0
            )

    def test_analysis_result_properties(self):
        """Test AnalysisResult computed properties."""
        request = AnalysisRequest(ticker="AAPL", strategy_type=StrategyType.LONG_CALLS)
        indicators = TechnicalIndicators(
            rsi=45.0, beta=1.2, momentum=0.05, avg_iv=0.23, forecast_confidence=0.7
        )

        # Test successful result
        recommendations = pd.DataFrame({"Strike": [100, 105], "Score": [0.8, 0.7]})
        result = AnalysisResult(
            request=request,
            recommendations=recommendations,
            technical_indicators=indicators,
            execution_time_seconds=1.5,
        )

        assert result.is_successful == True
        assert result.recommendation_count == 2

        # Test empty result
        empty_result = AnalysisResult(
            request=request,
            recommendations=pd.DataFrame(),
            technical_indicators=indicators,
            execution_time_seconds=1.0,
        )

        assert empty_result.is_successful == False
        assert empty_result.recommendation_count == 0


class TestScoringComponents:
    """Test scoring components in isolation."""

    @pytest.fixture
    def sample_technical_indicators(self):
        """Sample technical indicators for testing."""
        return TechnicalIndicators(
            rsi=45.0,
            beta=1.2,
            momentum=0.05,
            avg_iv=0.23,
            forecast_confidence=0.7,
            data_availability={
                "rsi": True,
                "beta": True,
                "momentum": True,
                "iv": True,
                "forecast": True,
            },
        )

    def test_weight_normalization(self):
        """Test weight normalization utility."""
        weights = {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2}
        available_sources = ["rsi", "beta", "momentum"]

        normalized = normalize_weights(weights, available_sources)

        assert len(normalized) == 3
        assert abs(sum(normalized.values()) - 1.0) < 0.001
        assert all(source in normalized for source in available_sources)

    def test_weight_normalization_no_matching_sources(self):
        """Test weight normalization with no matching sources."""
        weights = {"x": 0.5, "y": 0.5}
        available_sources = ["rsi", "beta", "momentum"]

        normalized = normalize_weights(weights, available_sources)

        # Should fallback to equal weights
        expected_weight = 1.0 / 3
        assert all(
            abs(weight - expected_weight) < 0.001 for weight in normalized.values()
        )

    def test_weight_validation(self):
        """Test weight validation utility."""
        valid_weights = {"rsi": 0.5, "beta": 0.5}
        invalid_weights = {"rsi": 0.3, "beta": 0.3}

        assert validate_weights(valid_weights) == True
        assert validate_weights(invalid_weights) == False

    def test_weight_redistribution(self):
        """Test weight redistribution utility."""
        weights = {"rsi": 0.4, "beta": 0.3, "momentum": 0.3}
        available_sources = ["rsi", "beta"]

        redistributed = redistribute_weight(weights, "momentum", available_sources)

        assert "momentum" not in redistributed
        assert len(redistributed) == 2
        assert abs(sum(redistributed.values()) - 1.0) < 0.001

    def test_composite_scorer(self, sample_technical_indicators):
        """Test CompositeScorer functionality."""
        scoring_weights = ScoringWeights()
        scorer = CompositeScorer(scoring_weights)

        options_df = pd.DataFrame(
            {
                "Strike": [95, 100, 105],
                "IV": [0.25, 0.22, 0.20],
                "Delta": [0.8, 0.6, 0.4],
                "Volume": [100, 200, 150],
                "OpenInterest": [500, 800, 600],
            }
        )

        scored_df = scorer.calculate_composite_scores(
            options_df, sample_technical_indicators, "Long Calls"
        )

        assert "CompositeScore" in scored_df.columns
        assert "iv_score" in scored_df.columns
        assert "score_details" in scored_df.columns
        assert all(0 <= score <= 1 for score in scored_df["CompositeScore"])
        assert len(scored_df) == 3

    def test_technical_indicators_calculator(self):
        """Test TechnicalIndicatorsCalculator with realistic data."""
        # Create market data with proper column names
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        stock_prices = pd.Series([100 + i * 0.1 for i in range(50)], index=dates)
        spy_prices = pd.Series([400 + i * 0.05 for i in range(50)], index=dates)
        options_data = pd.DataFrame(
            {
                "Strike": [100, 105],
                "impliedVolatility": [0.2, 0.3],  # Use correct column name
                "volume": [100, 200],  # Add volume for weighting
            }
        )

        market_data = MarketData(
            ticker="AAPL",
            stock_prices=stock_prices,
            spy_prices=spy_prices,
            options_data=options_data,
            current_price=105.0,
        )

        calculator = TechnicalIndicatorsCalculator()
        indicators = calculator.calculate_all_indicators(market_data, 0.7)

        # Verify structure rather than exact values
        assert hasattr(indicators, "rsi")
        assert hasattr(indicators, "beta")
        assert hasattr(indicators, "momentum")
        assert hasattr(indicators, "avg_iv")
        assert indicators.forecast_confidence == 0.7

        # Verify data availability tracking
        assert isinstance(indicators.data_availability, dict)
        assert "rsi" in indicators.data_availability
        assert "beta" in indicators.data_availability
        assert "momentum" in indicators.data_availability
        assert "iv" in indicators.data_availability
        assert "forecast" in indicators.data_availability

        # RSI should be between 0 and 100
        assert 0 <= indicators.rsi <= 100

        # Beta should be positive for normal stocks
        assert indicators.beta > 0

        # Average IV should be reasonable
        assert 0 < indicators.avg_iv < 2.0


class TestConfigurationManagement:
    """Test configuration management components."""

    def test_scoring_weights_configuration(self):
        """Test scoring weights configuration retrieval."""
        # Test default weights
        default_weights = get_scoring_weights()
        assert abs(sum(default_weights.to_dict().values()) - 1.0) < 0.001

        # Test strategy-specific weights
        long_calls_weights = get_scoring_weights(StrategyType.LONG_CALLS)
        covered_call_weights = get_scoring_weights(StrategyType.COVERED_CALL)

        # Different strategies should have different weights
        assert long_calls_weights.to_dict() != covered_call_weights.to_dict()
        assert abs(sum(long_calls_weights.to_dict().values()) - 1.0) < 0.001
        assert abs(sum(covered_call_weights.to_dict().values()) - 1.0) < 0.001

    def test_risk_profiles_configuration(self):
        """Test risk profiles configuration."""
        conservative_profile = get_risk_profile(RiskTolerance.CONSERVATIVE)
        aggressive_profile = get_risk_profile(RiskTolerance.AGGRESSIVE)

        # Conservative should be more restrictive
        assert (
            conservative_profile.max_delta_threshold
            < aggressive_profile.max_delta_threshold
        )
        assert (
            conservative_profile.min_days_to_expiry
            > aggressive_profile.min_days_to_expiry
        )
        assert (
            conservative_profile.volume_threshold > aggressive_profile.volume_threshold
        )
        assert (
            conservative_profile.max_bid_ask_spread
            < aggressive_profile.max_bid_ask_spread
        )


class TestErrorHandling:
    """Test error handling and exception hierarchy."""

    def test_error_context(self):
        """Test ErrorContext functionality."""
        context = ErrorContext(
            ticker="AAPL", strategy="Long Calls", additional_data={"test": "data"}
        )

        assert context.ticker == "AAPL"
        assert context.strategy == "Long Calls"
        assert context.additional_data["test"] == "data"
        assert context.timestamp is not None

    def test_options_advisor_error_with_context(self):
        """Test OptionsAdvisorError with context."""
        context = ErrorContext(ticker="AAPL", strategy="Long Calls")
        error = OptionsAdvisorError("Test error", context=context)

        assert "AAPL" in str(error)
        assert error.context.ticker == "AAPL"
        assert error.message == "Test error"

    def test_insufficient_data_error(self):
        """Test InsufficientDataError specific attributes."""
        context = ErrorContext(ticker="AAPL")
        error = InsufficientDataError(
            "Insufficient data", context=context, data_points=10, required_points=30
        )

        assert error.data_points == 10
        assert error.required_points == 30
        assert isinstance(error, OptionsAdvisorError)

    def test_strategy_validation_error(self):
        """Test StrategyValidationError specific attributes."""
        context = ErrorContext(ticker="AAPL")
        validation_errors = {
            "min_days": "Must be positive",
            "top_n": "Must be positive",
        }

        error = StrategyValidationError(
            "Validation failed", context=context, validation_errors=validation_errors
        )

        assert error.validation_errors == validation_errors
        assert len(error.validation_errors) == 2
        assert isinstance(error, OptionsAdvisorError)


class TestEnumerations:
    """Test enumeration types and their behavior."""

    def test_strategy_type_enum(self):
        """Test StrategyType enumeration."""
        # Test all strategy types are defined correctly
        assert StrategyType.LONG_CALLS.value == "Long Calls"
        assert StrategyType.BULL_CALL_SPREAD.value == "Bull Call Spread"
        assert StrategyType.COVERED_CALL.value == "Covered Call"
        assert StrategyType.CASH_SECURED_PUT.value == "Cash-Secured Put"

        # Test string conversion
        assert StrategyType("Long Calls") == StrategyType.LONG_CALLS
        assert StrategyType("Bull Call Spread") == StrategyType.BULL_CALL_SPREAD

    def test_risk_tolerance_enum(self):
        """Test RiskTolerance enumeration."""
        assert RiskTolerance.CONSERVATIVE.value == "Conservative"
        assert RiskTolerance.MODERATE.value == "Moderate"
        assert RiskTolerance.AGGRESSIVE.value == "Aggressive"

        # Test string conversion
        assert RiskTolerance("Conservative") == RiskTolerance.CONSERVATIVE
        assert RiskTolerance("Moderate") == RiskTolerance.MODERATE

    def test_time_horizon_enum(self):
        """Test TimeHorizon enumeration."""
        assert TimeHorizon.SHORT_TERM.value == "Short-term (1-3 months)"
        assert TimeHorizon.MEDIUM_TERM.value == "Medium-term (3-6 months)"
        assert TimeHorizon.LONG_TERM.value == "Long-term (6+ months)"

        # Test string conversion
        assert TimeHorizon("Short-term (1-3 months)") == TimeHorizon.SHORT_TERM
