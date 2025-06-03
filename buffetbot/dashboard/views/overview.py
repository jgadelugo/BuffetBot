"""Overview tab rendering module."""

from typing import Any, Dict

import streamlit as st

from buffetbot.dashboard.components.metrics import display_metric_with_info
from buffetbot.dashboard.dashboard_utils.data_utils import (
    safe_get_last_price,
    safe_get_nested_value,
)
from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from buffetbot.utils.data_report import DataCollectionReport
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def render_overview_tab(data: dict[str, Any], ticker: str) -> None:
    """Render the overview tab content.

    Args:
        data: Stock data dictionary
        ticker: Stock ticker symbol
    """
    # Display basic information
    st.header(f"{ticker} Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        display_metric_with_info(
            "Current Price",
            safe_format_currency(safe_get_last_price(data["price_data"])),
            safe_format_percentage(
                safe_get_nested_value(data, "metrics", "price_change")
            ),
            metric_key="latest_price",
        )

    with col2:
        display_metric_with_info(
            "Market Cap",
            safe_format_currency(
                safe_get_nested_value(data, "fundamentals", "market_cap")
            ),
            f"P/E: {safe_format_number(safe_get_nested_value(data, 'fundamentals', 'pe_ratio'))}",
            metric_key="market_cap",
        )

    with col3:
        display_metric_with_info(
            "Volatility",
            safe_format_percentage(
                safe_get_nested_value(data, "metrics", "volatility")
            ),
            f"RSI: {safe_format_number(safe_get_nested_value(data, 'metrics', 'rsi'))}",
            metric_key="volatility",
        )

    # Add link to data collection report
    st.markdown("---")
    st.subheader("Data Quality")

    # Create a fresh DataCollectionReport for the current ticker data
    try:
        report = DataCollectionReport(data)
        report_data = report.get_report()
        quality_score = report_data.get("data_quality_score", 0)

        # Log the data quality score for debugging
        logger.info(f"Data quality score for {ticker}: {quality_score:.1f}%")

        score_color = (
            "green"
            if quality_score >= 80
            else "orange"
            if quality_score >= 50
            else "red"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 20px; background-color: {score_color}20; border-radius: 10px;'>
                    <h2 style='color: {score_color}; margin: 0;'>Data Quality Score</h2>
                    <h1 style='color: {score_color}; margin: 10px 0;'>{quality_score:.1f}%</h1>
                    <p style='color: {score_color}; margin: 5px 0; font-size: 0.9em;'>Ticker: {ticker}</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                ### Data Collection Report for {ticker}
                View detailed information about the collected data, including:
                - Data availability status
                - Missing columns and metrics
                - Data quality indicators
                - Impact on analysis
                - Recommendations for improvement
            """
            )
            if st.button("View Data Collection Report", key="view_data_report"):
                st.session_state["show_data_report"] = True
                st.rerun()

    except Exception as e:
        logger.error(
            f"Error generating data quality report for {ticker}: {str(e)}",
            exc_info=True,
        )
        st.error(f"Error generating data quality report: {str(e)}")
        # Show a fallback quality score
        st.markdown(
            """
            <div style='text-align: center; padding: 20px; background-color: orange20; border-radius: 10px;'>
                <h2 style='color: orange; margin: 0;'>Data Quality Score</h2>
                <h1 style='color: orange; margin: 10px 0;'>N/A</h1>
                <p style='color: orange; margin: 5px 0; font-size: 0.9em;'>Error calculating score</p>
            </div>
        """,
            unsafe_allow_html=True,
        )
