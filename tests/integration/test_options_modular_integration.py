"""
Integration tests for the modular options analysis architecture.

This module tests the complete workflow of the modular architecture,
including data fetching, processing, and backward compatibility.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from buffetbot.analysis.options import analyze_options_strategy
from buffetbot.analysis.options.core.domain_models import (
    AnalysisRequest,
    RiskTolerance,
    StrategyType,
    TimeHorizon,
)
from buffetbot.analysis.options.core.strategy_dispatcher import (
    execute_strategy_analysis,
)


class TestModularArchitectureIntegration:
    """Test the complete modular architecture integration."""

    @pytest.fixture
    def mock_data_services(self):
        """Mock all data services for integration testing."""
        with patch(
            "buffetbot.analysis.options.data.price_service.YFinancePriceService"
        ), patch(
            "buffetbot.analysis.options.data.options_service.DefaultOptionsService"
        ), patch(
            "buffetbot.analysis.options.data.forecast_service.DefaultForecastService"
        ):
            yield

    @pytest.fixture
    def sample_market_data_for_integration(self):
        """Create comprehensive sample data for integration testing."""
        from buffetbot.analysis.options.core.domain_models import MarketData
        from buffetbot.data.options_fetcher import OptionsResult

        # Price data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        stock_prices = pd.Series(
            [100 + i * 0.5 + (-1 if i % 7 == 0 else 1) * (i % 3) for i in range(100)],
            index=dates,
            name="Close",
        )
        spy_prices = pd.Series(
            [400 + i * 0.2 + (-1 if i % 5 == 0 else 1) * (i % 2) for i in range(100)],
            index=dates,
            name="Close",
        )

        # Options data
        options_data = pd.DataFrame(
            {
                "Strike": [95, 100, 105, 110, 115, 120],
                "IV": [0.28, 0.25, 0.22, 0.20, 0.23, 0.26],
                "Delta": [0.85, 0.70, 0.55, 0.40, 0.25, 0.15],
                "Volume": [150, 200, 300, 180, 100, 50],
                "OpenInterest": [800, 1200, 1500, 900, 600, 300],
                "Bid": [8.5, 5.0, 2.8, 1.5, 0.8, 0.3],
                "Ask": [8.8, 5.3, 3.1, 1.8, 1.1, 0.6],
                "DaysToExpiry": [45, 45, 45, 45, 45, 45],
                "LastPrice": [8.65, 5.15, 2.95, 1.65, 0.95, 0.45],
            }
        )

        return {
            "market_data": MarketData(
                ticker="AAPL",
                stock_prices=stock_prices,
                spy_prices=spy_prices,
                options_data=options_data,
                current_price=149.5,
            ),
            "options_result": OptionsResult(
                options_df=options_data,
                current_price=149.5,
                status="success",
                data_source="test",
            ),
        }

    @patch("buffetbot.analysis.options_advisor.recommend_long_calls")
    def test_backward_compatibility_long_calls(
        self, mock_recommend, mock_data_services, sample_market_data_for_integration
    ):
        """Test backward compatibility for Long Calls strategy."""
        # Setup mock return value
        expected_result = pd.DataFrame(
            {
                "Strike": [100, 105, 110],
                "CompositeScore": [0.85, 0.78, 0.72],
                "IV": [0.25, 0.22, 0.20],
                "Delta": [0.70, 0.55, 0.40],
                "Volume": [200, 300, 180],
                "OpenInterest": [1200, 1500, 900],
            }
        )
        mock_recommend.return_value = expected_result

        # Test backward compatible API
        result = analyze_options_strategy(
            strategy_type="Long Calls",
            ticker="AAPL",
            min_days=180,
            top_n=5,
            risk_tolerance="Conservative",
        )

        # Verify the legacy function was called
        mock_recommend.assert_called_once()

        # Verify backward compatibility metadata is added
        assert "strategy_type" in result.columns
        assert "risk_tolerance" in result.columns
        assert "analysis_date" in result.columns
        assert result["strategy_type"].iloc[0] == "Long Calls"
        assert result["risk_tolerance"].iloc[0] == "Conservative"

        # Verify core data is preserved
        assert len(result) == 3
        assert "CompositeScore" in result.columns
        assert all(result["CompositeScore"] == expected_result["CompositeScore"])

    @patch("buffetbot.analysis.options_advisor.recommend_bull_call_spread")
    def test_backward_compatibility_bull_call_spread(
        self, mock_recommend, mock_data_services
    ):
        """Test backward compatibility for Bull Call Spread strategy."""
        expected_result = pd.DataFrame(
            {
                "LongStrike": [100, 105],
                "ShortStrike": [105, 110],
                "NetDebit": [2.5, 2.8],
                "MaxProfit": [2.5, 2.2],
                "CompositeScore": [0.82, 0.75],
            }
        )
        mock_recommend.return_value = expected_result

        result = analyze_options_strategy(
            strategy_type="Bull Call Spread",
            ticker="MSFT",
            min_days=90,
            top_n=3,
            risk_tolerance="Moderate",
        )

        mock_recommend.assert_called_once()
        assert len(result) == 2
        assert "strategy_type" in result.columns
        assert result["strategy_type"].iloc[0] == "Bull Call Spread"

    @patch("buffetbot.analysis.options_advisor.recommend_covered_call")
    def test_income_strategy_min_days_adjustment(
        self, mock_recommend, mock_data_services
    ):
        """Test that income strategies get min_days adjustment."""
        expected_result = pd.DataFrame(
            {
                "Strike": [155, 160],
                "Yield": [0.08, 0.06],
                "CompositeScore": [0.88, 0.81],
            }
        )
        mock_recommend.return_value = expected_result

        # Request with high min_days for covered call
        result = analyze_options_strategy(
            strategy_type="Covered Call",
            ticker="GOOGL",
            min_days=365,  # Should be adjusted down for income strategies
            top_n=5,
            risk_tolerance="Aggressive",
        )

        # Verify the function was called with adjusted parameters
        mock_recommend.assert_called_once()
        call_args = mock_recommend.call_args

        # The min_days should be adjusted down for income strategies
        assert call_args[1]["min_days"] <= 90

    def test_analysis_request_end_to_end(
        self, mock_data_services, sample_market_data_for_integration
    ):
        """Test end-to-end AnalysisRequest processing."""
        request = AnalysisRequest(
            ticker="AAPL",
            strategy_type=StrategyType.LONG_CALLS,
            min_days=120,
            top_n=5,
            risk_tolerance=RiskTolerance.MODERATE,
            time_horizon=TimeHorizon.MEDIUM_TERM,
        )

        # Mock the data repository methods
        with patch(
            "buffetbot.analysis.options.data.repositories.DefaultDataRepository"
        ) as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_market_data.return_value = sample_market_data_for_integration[
                "market_data"
            ]
            mock_repo.get_forecast_data.return_value = 0.7

            # Mock the legacy strategy function
            with patch(
                "buffetbot.analysis.options_advisor.recommend_long_calls"
            ) as mock_strategy:
                expected_recommendations = pd.DataFrame(
                    {
                        "Strike": [100, 105],
                        "CompositeScore": [0.85, 0.78],
                        "IV": [0.25, 0.22],
                    }
                )
                mock_strategy.return_value = expected_recommendations

                # Execute the analysis
                result = execute_strategy_analysis(request)

                # Verify the result structure
                assert result.request == request
                assert not result.recommendations.empty
                assert result.technical_indicators is not None
                assert result.execution_time_seconds > 0
                assert result.is_successful
                assert result.recommendation_count == 2

                # Verify metadata
                assert "data_sources_used" in result.metadata
                assert "scoring_weights" in result.metadata
                assert "market_data_timestamp" in result.metadata

    def test_risk_tolerance_parameter_adjustment(self, mock_data_services):
        """Test that risk tolerance affects parameter adjustment."""
        with patch(
            "buffetbot.analysis.options_advisor.recommend_long_calls"
        ) as mock_recommend:
            mock_recommend.return_value = pd.DataFrame(
                {"Strike": [100], "CompositeScore": [0.8]}
            )

            # Test conservative risk tolerance
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker="AAPL",
                min_days=30,  # Short timeframe
                risk_tolerance="Conservative",
            )

            # Conservative should increase min_days
            call_args = mock_recommend.call_args
            assert call_args[1]["min_days"] >= 60

            mock_recommend.reset_mock()

            # Test aggressive risk tolerance
            analyze_options_strategy(
                strategy_type="Long Calls",
                ticker="AAPL",
                min_days=200,
                top_n=3,
                risk_tolerance="Aggressive",
            )

            # Aggressive should allow more results
            call_args = mock_recommend.call_args
            # top_n might be increased for aggressive (up to 10)
            assert call_args[1]["top_n"] >= 3

    def test_technical_indicators_integration(
        self, mock_data_services, sample_market_data_for_integration
    ):
        """Test technical indicators calculation integration."""
        from buffetbot.analysis.options.core.domain_models import TechnicalIndicators
        from buffetbot.analysis.options.scoring.technical_indicators import (
            TechnicalIndicatorsCalculator,
        )

        # Mock the calculator method directly
        with patch.object(
            TechnicalIndicatorsCalculator, "calculate_all_indicators"
        ) as mock_calculate:
            mock_indicators = TechnicalIndicators(
                rsi=42.5,
                beta=1.15,
                momentum=0.08,
                avg_iv=0.24,
                forecast_confidence=0.68,
                data_availability={
                    "rsi": True,
                    "beta": True,
                    "momentum": True,
                    "iv": True,
                    "forecast": True,
                },
            )
            mock_calculate.return_value = mock_indicators

            # Mock data repository
            with patch(
                "buffetbot.analysis.options.data.repositories.DefaultDataRepository"
            ) as mock_repo_class:
                mock_repo = mock_repo_class.return_value
                mock_repo.get_market_data.return_value = (
                    sample_market_data_for_integration["market_data"]
                )
                mock_repo.get_forecast_data.return_value = 0.68

                with patch(
                    "buffetbot.analysis.options_advisor.recommend_long_calls"
                ) as mock_strategy:
                    mock_strategy.return_value = pd.DataFrame(
                        {"Strike": [100], "CompositeScore": [0.8]}
                    )

                    request = AnalysisRequest(
                        ticker="AAPL", strategy_type=StrategyType.LONG_CALLS
                    )

                    result = execute_strategy_analysis(request)

                    # Verify technical indicators were calculated
                    mock_calculate.assert_called_once()
                    assert result.technical_indicators.rsi == 42.5
                    assert result.technical_indicators.beta == 1.15
                    assert result.technical_indicators.forecast_confidence == 0.68

    def test_error_propagation(self, mock_data_services):
        """Test that errors are properly propagated through the system."""
        from buffetbot.analysis.options.core.exceptions import DataSourceError

        # Mock a data source error
        with patch(
            "buffetbot.analysis.options.data.repositories.DefaultDataRepository"
        ) as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_market_data.side_effect = DataSourceError(
                "Failed to fetch data"
            )

            request = AnalysisRequest(
                ticker="INVALID", strategy_type=StrategyType.LONG_CALLS
            )

            # Should propagate the error with context
            with pytest.raises(Exception) as exc_info:
                execute_strategy_analysis(request)

            # Verify error handling (either DataSourceError or wrapped OptionsAdvisorError)
            assert "Failed to fetch data" in str(exc_info.value) or "INVALID" in str(
                exc_info.value
            )

    def test_strategy_type_enumeration_integration(
        self, mock_data_services, sample_market_data_for_integration
    ):
        """Test that all strategy types are properly handled."""
        strategy_mocks = {
            StrategyType.LONG_CALLS: "buffetbot.analysis.options_advisor.recommend_long_calls",
            StrategyType.BULL_CALL_SPREAD: "buffetbot.analysis.options_advisor.recommend_bull_call_spread",
            StrategyType.COVERED_CALL: "buffetbot.analysis.options_advisor.recommend_covered_call",
            StrategyType.CASH_SECURED_PUT: "buffetbot.analysis.options_advisor.recommend_cash_secured_put",
        }

        # Mock data repository to provide valid data instead of trying to fetch from TEST ticker
        with patch(
            "buffetbot.analysis.options.data.repositories.DefaultDataRepository"
        ) as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.get_market_data.return_value = sample_market_data_for_integration[
                "market_data"
            ]
            mock_repo.get_forecast_data.return_value = 0.75

            for strategy_type, mock_path in strategy_mocks.items():
                with patch(mock_path) as mock_strategy:
                    mock_strategy.return_value = pd.DataFrame(
                        {"Strike": [100], "CompositeScore": [0.8]}
                    )

                    # Test string-based API (backward compatibility) with mock data
                    result = analyze_options_strategy(
                        strategy_type=strategy_type.value,
                        ticker="AAPL",  # Use AAPL instead of TEST
                        min_days=90,
                        top_n=1,
                    )

                    assert not result.empty
                    assert "strategy_type" in result.columns
                    assert result["strategy_type"].iloc[0] == strategy_type.value
                    mock_strategy.assert_called_once()


class TestDataLayerIntegration:
    """Test data layer integration with mocked external services."""

    def test_data_repository_composition(self):
        """Test that data repository properly composes services."""
        from buffetbot.analysis.options.data.forecast_service import (
            DefaultForecastService,
        )
        from buffetbot.analysis.options.data.options_service import (
            DefaultOptionsService,
        )
        from buffetbot.analysis.options.data.price_service import YFinancePriceService
        from buffetbot.analysis.options.data.repositories import DefaultDataRepository

        # Create services
        price_service = YFinancePriceService()
        options_service = DefaultOptionsService()
        forecast_service = DefaultForecastService()

        # Create repository
        repo = DefaultDataRepository(
            options_repo=options_service,
            price_repo=price_service,
            forecast_repo=forecast_service,
        )

        # Verify composition
        assert repo.options_repo == options_service
        assert repo.price_repo == price_service
        assert repo.forecast_repo == forecast_service

    def test_service_caching_behavior(self):
        """Test that services properly implement caching."""
        from buffetbot.analysis.options.data.forecast_service import (
            DefaultForecastService,
        )
        from buffetbot.analysis.options.data.options_service import (
            DefaultOptionsService,
        )
        from buffetbot.analysis.options.data.price_service import YFinancePriceService

        # Test that services can be created with caching enabled/disabled
        price_service_cached = YFinancePriceService(cache_enabled=True)
        price_service_no_cache = YFinancePriceService(cache_enabled=False)

        assert price_service_cached.cache_enabled == True
        assert price_service_no_cache.cache_enabled == False

        options_service_cached = DefaultOptionsService(cache_enabled=True)
        options_service_no_cache = DefaultOptionsService(cache_enabled=False)

        assert options_service_cached.cache_enabled == True
        assert options_service_no_cache.cache_enabled == False

        forecast_service_cached = DefaultForecastService(cache_enabled=True)
        forecast_service_no_cache = DefaultForecastService(cache_enabled=False)

        assert forecast_service_cached.cache_enabled == True
        assert forecast_service_no_cache.cache_enabled == False


class TestConfigurationIntegration:
    """Test configuration system integration."""

    def test_scoring_weights_strategy_integration(self):
        """Test that scoring weights integrate properly with strategies."""
        from buffetbot.analysis.options.config.scoring_weights import (
            get_scoring_weights,
        )

        # Test that each strategy type has unique weights
        strategies = [
            StrategyType.LONG_CALLS,
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.COVERED_CALL,
            StrategyType.CASH_SECURED_PUT,
        ]

        strategy_weights = {}
        for strategy in strategies:
            weights = get_scoring_weights(strategy)
            strategy_weights[strategy] = weights.to_dict()

            # Each strategy should have valid weights
            assert abs(sum(weights.to_dict().values()) - 1.0) < 0.001

        # Income strategies should have higher IV weights
        income_strategies = [StrategyType.COVERED_CALL, StrategyType.CASH_SECURED_PUT]
        growth_strategies = [StrategyType.LONG_CALLS, StrategyType.BULL_CALL_SPREAD]

        for income_strategy in income_strategies:
            for growth_strategy in growth_strategies:
                income_iv_weight = strategy_weights[income_strategy]["iv"]
                growth_iv_weight = strategy_weights[growth_strategy]["iv"]

                # Income strategies should generally weight IV more heavily
                assert income_iv_weight >= growth_iv_weight

    def test_risk_profile_integration(self):
        """Test that risk profiles integrate with analysis parameters."""
        from buffetbot.analysis.options.config.risk_profiles import get_risk_profile

        conservative = get_risk_profile(RiskTolerance.CONSERVATIVE)
        moderate = get_risk_profile(RiskTolerance.MODERATE)
        aggressive = get_risk_profile(RiskTolerance.AGGRESSIVE)

        # Conservative should be most restrictive
        assert (
            conservative.max_delta_threshold
            <= moderate.max_delta_threshold
            <= aggressive.max_delta_threshold
        )
        assert (
            conservative.min_days_to_expiry
            >= moderate.min_days_to_expiry
            >= aggressive.min_days_to_expiry
        )
        assert (
            conservative.volume_threshold
            >= moderate.volume_threshold
            >= aggressive.volume_threshold
        )
        assert (
            conservative.max_bid_ask_spread
            <= moderate.max_bid_ask_spread
            <= aggressive.max_bid_ask_spread
        )
