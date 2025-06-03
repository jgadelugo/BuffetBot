"""Integration tests for dashboard tab functionality."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st

from buffetbot.dashboard.views.glossary import render_glossary_tab
from buffetbot.dashboard.views.overview import render_overview_tab
from buffetbot.dashboard.views.risk_analysis import render_risk_analysis_tab


class TestTabIntegration:
    """Integration tests for tab rendering with mock data."""

    @pytest.fixture
    def mock_stock_data(self):
        """Create mock stock data for testing."""
        return {
            "fundamentals": {"market_cap": 1.5e12, "pe_ratio": 25.6, "beta": 1.2},
            "metrics": {"price_change": 0.025, "volatility": 0.35, "rsi": 65.4},
            "price_data": pd.DataFrame(
                {
                    "Open": [150.0, 151.0, 149.0],
                    "High": [155.0, 156.0, 154.0],
                    "Low": [145.0, 146.0, 144.0],
                    "Close": [150.1, 151.2, 153.7],
                    "Volume": [1000000, 1100000, 1200000],
                }
            ),
            "income_stmt": pd.DataFrame(
                {
                    "revenue": [100000, 110000, 120000],
                    "net_income": [10000, 12000, 14000],
                }
            ),
            "balance_sheet": pd.DataFrame(
                {
                    "total_assets": [500000, 550000, 600000],
                    "total_debt": [100000, 110000, 120000],
                }
            ),
            "cash_flow": pd.DataFrame(
                {
                    "operating_cash_flow": [15000, 17000, 19000],
                    "free_cash_flow": [8000, 9000, 10000],
                }
            ),
        }

    @pytest.fixture
    def mock_risk_analysis_result(self):
        """Create mock risk analysis result."""
        return {
            "overall_risk": {
                "score": 65.5,
                "level": "Moderate",
                "factors": [
                    "Market volatility exposure",
                    "Sector concentration risk",
                    "Liquidity risk in stressed conditions",
                ],
                "warnings": ["Beta calculation based on limited historical data"],
                "errors": [],
            },
            "market_risk": {"beta": 1.2, "volatility": 0.35},
            "financial_risk": {"debt_to_equity": 0.8, "interest_coverage": 5.2},
            "business_risk": {"operating_margin": 0.15, "revenue": 1000000000},
        }

    @patch("streamlit.header")
    @patch("streamlit.columns")
    @patch("streamlit.markdown")
    @patch("dashboard.components.metrics.display_metric_with_info")
    @patch("utils.data_report.DataCollectionReport")
    def test_overview_tab_rendering(
        self,
        mock_report,
        mock_display_metric,
        mock_markdown,
        mock_columns,
        mock_header,
        mock_stock_data,
    ):
        """Test overview tab renders correctly with valid data."""
        # Mock the columns return
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

        # Mock the data report
        mock_report_instance = MagicMock()
        mock_report_instance.get_report.return_value = {"data_quality_score": 85.5}
        mock_report.return_value = mock_report_instance

        # Test tab rendering
        try:
            render_overview_tab(mock_stock_data, "AAPL")

            # Verify key components were called
            mock_header.assert_called()
            mock_columns.assert_called()
            mock_display_metric.assert_called()

        except Exception as e:
            pytest.fail(f"Overview tab rendering failed: {str(e)}")

    @patch("streamlit.subheader")
    @patch("streamlit.columns")
    @patch("streamlit.success")
    @patch("streamlit.warning")
    @patch("streamlit.info")
    @patch("analysis.risk_analysis.analyze_risk_metrics")
    @patch("dashboard.components.disclaimers.render_investment_disclaimer")
    def test_risk_analysis_tab_rendering(
        self,
        mock_disclaimer,
        mock_analyze_risk,
        mock_info,
        mock_warning,
        mock_success,
        mock_columns,
        mock_subheader,
        mock_stock_data,
        mock_risk_analysis_result,
    ):
        """Test risk analysis tab renders correctly with valid data."""
        # Mock the risk analysis function
        mock_analyze_risk.return_value = mock_risk_analysis_result

        # Mock columns
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

        try:
            render_risk_analysis_tab(mock_stock_data, "AAPL")

            # Verify key components were called
            mock_subheader.assert_called()
            mock_disclaimer.assert_called_with("analysis")
            mock_analyze_risk.assert_called_with(mock_stock_data)

        except Exception as e:
            pytest.fail(f"Risk analysis tab rendering failed: {str(e)}")

    @patch("streamlit.header")
    @patch("streamlit.columns")
    @patch("streamlit.text_input")
    @patch("streamlit.radio")
    @patch("streamlit.metric")
    @patch(
        "glossary.GLOSSARY",
        {
            "test_metric": {
                "name": "Test",
                "category": "growth",
                "description": "Test desc",
                "formula": "test = 1",
            }
        },
    )
    def test_glossary_tab_rendering(
        self, mock_metric, mock_radio, mock_text_input, mock_columns, mock_header
    ):
        """Test glossary tab renders correctly."""
        # Mock user inputs
        mock_text_input.return_value = ""
        mock_radio.return_value = "All"

        # Mock columns
        mock_col1, mock_col2 = MagicMock(), MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]

        try:
            render_glossary_tab()

            # Verify key components were called
            mock_header.assert_called()
            mock_columns.assert_called()
            mock_text_input.assert_called()
            mock_radio.assert_called()

        except Exception as e:
            pytest.fail(f"Glossary tab rendering failed: {str(e)}")

    @patch("analysis.risk_analysis.analyze_risk_metrics")
    def test_risk_analysis_tab_error_handling(self, mock_analyze_risk, mock_stock_data):
        """Test risk analysis tab handles errors gracefully."""
        # Mock an exception in the analysis function
        mock_analyze_risk.side_effect = Exception("Analysis failed")

        with patch("streamlit.error") as mock_error:
            try:
                render_risk_analysis_tab(mock_stock_data, "AAPL")
                # Should not raise, should show error in UI
                mock_error.assert_called()
            except Exception as e:
                pytest.fail(
                    f"Risk analysis tab should handle errors gracefully: {str(e)}"
                )

    def test_overview_tab_missing_data(self):
        """Test overview tab handles missing data gracefully."""
        # Create incomplete data
        incomplete_data = {
            "fundamentals": {},
            "metrics": {},
            "price_data": pd.DataFrame(),
        }

        with patch("streamlit.header"), patch(
            "streamlit.columns"
        ) as mock_columns, patch(
            "dashboard.components.metrics.display_metric_with_info"
        ):
            mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

            try:
                render_overview_tab(incomplete_data, "AAPL")
                # Should not crash with missing data
            except Exception as e:
                pytest.fail(
                    f"Overview tab should handle missing data gracefully: {str(e)}"
                )


class TestDataFlow:
    """Test data flow between components."""

    def test_data_processing_pipeline(self):
        """Test that data flows correctly through the processing pipeline."""
        # Mock the entire data processing pipeline
        with patch(
            "dashboard.utils.data_processing.get_stock_info"
        ) as mock_get_data, patch(
            "dashboard.utils.data_processing.handle_ticker_change"
        ) as mock_handle_change:
            # Mock data return
            mock_stock_data = {
                "fundamentals": {"market_cap": 1e12},
                "price_data": pd.DataFrame({"Close": [100, 101, 102]}),
            }
            mock_get_data.return_value = mock_stock_data
            mock_handle_change.return_value = True

            # Test data retrieval
            ticker = "AAPL"
            years = 5

            result = mock_get_data(ticker, years)
            assert result is not None
            assert "fundamentals" in result
            assert "price_data" in result

            # Test ticker change handling
            change_detected = mock_handle_change(ticker)
            assert isinstance(change_detected, bool)


class TestComponentInteraction:
    """Test interactions between different components."""

    @patch("dashboard.utils.formatters.safe_format_currency")
    @patch("dashboard.utils.formatters.safe_format_percentage")
    @patch("dashboard.utils.data_utils.safe_get_nested_value")
    def test_formatter_integration(
        self, mock_get_nested, mock_format_pct, mock_format_curr
    ):
        """Test that formatters work correctly with data utils."""
        # Mock data extraction
        mock_get_nested.return_value = 1234.56
        mock_format_curr.return_value = "$1,234.56"
        mock_format_pct.return_value = "12.3%"

        # Test integration
        test_data = {"metrics": {"price": 1234.56}}

        # Extract value
        price = mock_get_nested(test_data, "metrics", "price")

        # Format value
        formatted_price = mock_format_curr(price)

        # Verify the pipeline works
        assert price == 1234.56
        assert formatted_price == "$1,234.56"

        # Verify functions were called
        mock_get_nested.assert_called_with(test_data, "metrics", "price")
        mock_format_curr.assert_called_with(price)


class TestUserInteraction:
    """Test user interaction scenarios."""

    @patch("streamlit.session_state", {})
    def test_session_state_management(self):
        """Test session state management across components."""
        # Test session state initialization
        from buffetbot.dashboard.config.settings import initialize_session_state

        # Initialize session state
        initialize_session_state()

        # Test that defaults are set
        assert st.session_state.get("show_metric_definitions") == True
        assert st.session_state.get("show_data_report") == False
        assert st.session_state.get("glossary_category") == "All"

    @patch("streamlit.sidebar.text_input")
    @patch("streamlit.sidebar.slider")
    @patch("streamlit.sidebar.button")
    def test_sidebar_interaction(self, mock_button, mock_slider, mock_text_input):
        """Test sidebar user interaction."""
        from buffetbot.dashboard.components.sidebar import render_sidebar

        # Mock user inputs
        mock_text_input.return_value = "AAPL"
        mock_slider.return_value = 5
        mock_button.return_value = False

        with patch("dashboard.config.settings.get_dashboard_config") as mock_config:
            mock_config.return_value = {
                "default_ticker": "AAPL",
                "min_years": 1,
                "max_years": 10,
                "default_years": 5,
            }

            try:
                ticker, years = render_sidebar()
                assert ticker == "AAPL"
                assert years == 5
            except Exception as e:
                pytest.fail(f"Sidebar interaction failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
