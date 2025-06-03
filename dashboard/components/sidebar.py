"""Sidebar components for the dashboard."""

from typing import Tuple

import streamlit as st

from dashboard.config.settings import get_dashboard_config
from dashboard.utils.data_processing import clear_cache


def render_sidebar() -> tuple[str, int]:
    """Render the sidebar with input parameters and controls.

    Returns:
        Tuple of (ticker, years)
    """
    config = get_dashboard_config()

    # Input Parameters
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", config["default_ticker"]).upper()

    years = st.sidebar.slider(
        "Years of Historical Data",
        config["min_years"],
        config["max_years"],
        config["default_years"],
    )

    # Cache Management
    st.sidebar.markdown("---")
    st.sidebar.header("Cache Management")
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and refresh"):
        clear_cache()

    # Display Settings
    st.sidebar.markdown("---")
    st.sidebar.header("Display Settings")

    # Toggle for metric definitions
    st.session_state.show_metric_definitions = st.sidebar.checkbox(
        "Show Metric Definitions",
        value=st.session_state.get("show_metric_definitions", True),
        help="Toggle to show/hide metric descriptions and formulas throughout the dashboard",
    )

    return ticker, years
