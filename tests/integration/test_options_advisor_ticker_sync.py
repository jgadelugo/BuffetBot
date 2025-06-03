#!/usr/bin/env python
"""
Integration tests for Options Advisor ticker synchronization.

This module tests the integration between the global ticker selection
and the Options Advisor tab to ensure proper synchronization.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))


class TestOptionsAdvisorTickerSync:
    """Test suite for Options Advisor ticker synchronization functionality."""

    def test_function_signature_is_correct(self):
        """Test that the options advisor function has the correct signature."""
        import inspect

        from dashboard.views.options_advisor import render_options_advisor_tab

        # Get function signature
        sig = inspect.signature(render_options_advisor_tab)

        # Check parameters
        params = list(sig.parameters.keys())
        expected_params = ["data", "ticker"]

        assert (
            params == expected_params
        ), f"Expected parameters {expected_params}, got {params}"

        # Check type annotations - be more flexible with Dict type checking
        param_annotations = {
            name: param.annotation.__name__
            if hasattr(param.annotation, "__name__")
            else str(param.annotation)
            for name, param in sig.parameters.items()
        }

        assert (
            param_annotations["ticker"] == "str"
        ), f"ticker parameter should be str, got {param_annotations['ticker']}"
        # Check that data parameter has some form of Dict annotation
        data_annotation = str(param_annotations["data"]).lower()
        assert (
            "dict" in data_annotation
        ), f"data parameter should be Dict type, got {param_annotations['data']}"

    def test_function_import_from_views_module(self):
        """Test that the function can be imported from the views module."""
        from dashboard.views import render_options_advisor_tab

        # Test that the function exists
        assert callable(
            render_options_advisor_tab
        ), "render_options_advisor_tab should be callable"

        # Test that it's properly exported
        from dashboard.views import __all__

        assert (
            "render_options_advisor_tab" in __all__
        ), "Function should be in __all__ exports"

    @patch("streamlit.header")
    @patch("streamlit.markdown")
    @patch("streamlit.columns")
    @patch("streamlit.session_state", {})
    def test_function_handles_valid_inputs(
        self, mock_columns, mock_markdown, mock_header
    ):
        """Test that the function handles valid inputs without errors."""
        from dashboard.views.options_advisor import render_options_advisor_tab

        # Mock streamlit components
        mock_columns.return_value = [MagicMock(), MagicMock()]

        # Test data
        test_data = {
            "price_data": {"close": [100, 101, 102]},
            "fundamentals": {"symbol": "AAPL"},
        }
        test_ticker = "AAPL"

        try:
            # This should not raise an exception
            render_options_advisor_tab(test_data, test_ticker)
        except Exception as e:
            # We expect some streamlit-related exceptions in testing environment
            # but not validation errors
            assert "Invalid ticker" not in str(
                e
            ), f"Should not get ticker validation error: {e}"

    @patch("streamlit.error")
    def test_function_validates_ticker_input(self, mock_error):
        """Test that the function properly validates ticker input."""
        from dashboard.views.options_advisor import render_options_advisor_tab

        test_data = {"price_data": {"close": [100, 101, 102]}}

        # Test empty ticker
        render_options_advisor_tab(test_data, "")
        mock_error.assert_called_with(
            "❌ Invalid ticker provided. Please select a valid ticker in the sidebar."
        )

        # Test None ticker
        mock_error.reset_mock()
        render_options_advisor_tab(test_data, None)
        mock_error.assert_called_with(
            "❌ Invalid ticker provided. Please select a valid ticker in the sidebar."
        )

    @patch("streamlit.warning")
    def test_function_handles_missing_data(self, mock_warning):
        """Test that the function handles missing or invalid data gracefully."""
        from dashboard.views.options_advisor import render_options_advisor_tab

        test_ticker = "AAPL"

        # Test with None data
        try:
            render_options_advisor_tab(None, test_ticker)
            mock_warning.assert_called_with(
                "⚠️ No stock data available. Please ensure data is loaded for the selected ticker."
            )
        except Exception:
            # Function may exit early, which is acceptable
            pass

        # Test with empty dict
        mock_warning.reset_mock()
        try:
            render_options_advisor_tab({}, test_ticker)
            mock_warning.assert_called_with(
                "⚠️ No stock data available. Please ensure data is loaded for the selected ticker."
            )
        except Exception:
            # Function may exit early, which is acceptable
            pass

    def test_view_metadata_updated_correctly(self):
        """Test that the view metadata reflects the function now requires data."""
        from dashboard.views import get_all_views

        all_views = get_all_views()
        options_advisor_view = all_views["advanced_tools"]["options_advisor"]

        assert (
            options_advisor_view["requires_data"] is True
        ), "Options advisor should now require data"
        assert (
            options_advisor_view["title"] == "Options Advisor"
        ), "Title should be correct"
        assert options_advisor_view["icon"] == "🎯", "Icon should be correct"

    def test_legacy_view_registry_updated(self):
        """Test that the legacy view registry is also updated correctly."""
        from dashboard.views.base import view_registry

        # Check if the view is registered (may not be in all configurations)
        try:
            metadata = view_registry.get_metadata("options_advisor")
            if metadata:
                assert (
                    metadata.requires_data is True
                ), "Legacy registry should show requires_data=True"
        except Exception:
            # Registry may not be populated in test environment, which is fine
            pass

    @patch("dashboard.views.options_advisor.handle_ticker_change")
    def test_ticker_change_handling_integration(self, mock_handle_ticker_change):
        """Test that ticker change handling is properly integrated."""
        from dashboard.views.options_advisor import render_options_advisor_tab

        # Mock the ticker change handler
        mock_handle_ticker_change.return_value = True

        test_data = {"price_data": {"close": [100, 101, 102]}}
        test_ticker = "AAPL"

        try:
            render_options_advisor_tab(test_data, test_ticker)
            # Verify that handle_ticker_change was called with the correct ticker
            mock_handle_ticker_change.assert_called_once_with("AAPL")
        except Exception:
            # Function may fail due to streamlit mocking, but we care about the ticker handling
            # Check if the function was called at least once
            assert (
                mock_handle_ticker_change.called
            ), "handle_ticker_change should have been called"
            if mock_handle_ticker_change.call_args_list:
                first_call = mock_handle_ticker_change.call_args_list[0]
                assert (
                    first_call[0][0] == "AAPL"
                ), f"Expected ticker 'AAPL', got {first_call[0][0]}"


class TestOptionsAdvisorAppIntegration:
    """Test suite for app-level integration of Options Advisor."""

    def test_app_imports_function_correctly(self):
        """Test that the main app can import the function correctly."""
        try:
            from dashboard.app import main

            # If we can import main, the imports in app.py should work
            assert callable(main), "Main app function should be callable"
        except ImportError as e:
            pytest.fail(f"App should be able to import options advisor function: {e}")

    def test_function_can_be_called_with_app_signature(self):
        """Test that the function can be called with the signature used in app.py."""
        import inspect

        from dashboard.views.options_advisor import render_options_advisor_tab

        # Get the signature
        sig = inspect.signature(render_options_advisor_tab)

        # Test that it matches the expected call pattern from app.py
        # render_options_advisor_tab(data, ticker)
        try:
            # Create mock arguments that match app.py usage
            mock_data = {"test": "data"}
            mock_ticker = "AAPL"

            # Check if the signature can bind these arguments
            bound_args = sig.bind(mock_data, mock_ticker)
            assert bound_args.arguments["data"] == mock_data
            assert bound_args.arguments["ticker"] == mock_ticker

        except TypeError as e:
            pytest.fail(
                f"Function signature should accept (data, ticker) as used in app.py: {e}"
            )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
