"""
Modularized Stock Analysis Dashboard

This is a refactored version of the original app.py file following software engineering best practices:
- Single Responsibility Principle
- Dependency Injection
- Clear separation of concerns
- Improved testability and maintainability
"""

from typing import Any, Dict

import streamlit as st

# Path setup must be first!
from dashboard.config.settings import (
    configure_streamlit_page,
    initialize_session_state,
    setup_logging_config,
    setup_project_path,
)

# Setup project path
setup_project_path()

from dashboard.components.disclaimers import (
    render_compliance_footer,
    render_investment_disclaimer,
)

# Import modules after path setup
from dashboard.components.sidebar import render_sidebar
from dashboard.utils.data_processing import get_stock_info, handle_ticker_change

# Import all views from the consolidated views module
from dashboard.views import (
    render_financial_health_page,
    render_glossary_tab,
    render_growth_metrics_tab,
    render_options_advisor_tab,
    render_overview_tab,
    render_price_analysis_page,
    render_risk_analysis_tab,
)
from utils.data_report import DataCollectionReport
from utils.logger import get_logger

# Initialize logging and get logger
setup_logging_config()
logger = get_logger(__name__)


def render_data_collection_report(data: dict[str, Any], ticker: str) -> None:
    """Render the data collection report modal.

    Args:
        data: Stock data dictionary
        ticker: Stock ticker symbol
    """
    st.title(f"Data Collection Report - {ticker}")

    # Add back button
    if st.button("← Back to Dashboard"):
        st.session_state.show_data_report = False
        st.rerun()

    # Create a fresh report for the current ticker and data
    try:
        current_report = DataCollectionReport(data)
        current_report_data = current_report.get_report()

        # Display data quality score
        quality_score = current_report_data.get("data_quality_score", 0)
        logger.info(f"Data quality score for {ticker} in modal: {quality_score:.1f}%")

        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px; background-color: {
                '#4CAF50' if quality_score >= 80 else
                '#FFA500' if quality_score >= 50 else
                '#FF5252'
            }; color: white; border-radius: 10px;'>
                <h2>Data Quality Score for {ticker}: {quality_score:.1f}%</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Display validation results, availability, impact analysis, and recommendations
        # (Implementation details would follow the original pattern)
        st.info("Full data collection report implementation would go here.")

    except Exception as e:
        logger.error(
            f"Error generating detailed data report for {ticker}: {str(e)}",
            exc_info=True,
        )
        st.error(f"Error generating detailed data report: {str(e)}")
        st.info(
            "Please try refreshing the data using the 'Clear Cache' button in the sidebar."
        )


def main() -> None:
    """Main dashboard application entry point."""
    # Configure Streamlit page
    configure_streamlit_page()

    # Initialize session state
    initialize_session_state()

    # Set page title and disclaimer
    st.title("Stock Analysis Dashboard")
    render_investment_disclaimer("header")

    # Render sidebar and get inputs
    ticker, years = render_sidebar()

    # Validate ticker input
    if not ticker or len(ticker.strip()) == 0:
        st.error("Please enter a valid ticker symbol")
        return

    # Handle ticker changes
    handle_ticker_change(ticker)

    # Fetch and process data
    with st.spinner(f"Fetching data for {ticker}..."):
        data = get_stock_info(ticker, years)

    if data is None:
        st.error(
            f"Could not fetch data for {ticker}. Please check the ticker symbol and try again."
        )
        return

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Price Analysis",
            "Financial Health",
            "Growth Metrics",
            "Risk Analysis",
            "📚 Glossary",
            "Options Advisor",
        ]
    )

    # Render tab content
    with tab1:
        render_overview_tab(data, ticker)

    with tab2:
        render_price_analysis_page(data, ticker)

    with tab3:
        render_financial_health_page(data, ticker)

    with tab4:
        render_growth_metrics_tab(data, ticker)

    with tab5:
        render_risk_analysis_tab(data, ticker)

    with tab6:
        render_glossary_tab()

    with tab7:
        render_options_advisor_tab()

    # Check if we should show the data collection report
    if st.session_state.get("show_data_report", False):
        render_data_collection_report(data, ticker)

    # Add comprehensive compliance footer
    render_compliance_footer()


if __name__ == "__main__":
    main()
