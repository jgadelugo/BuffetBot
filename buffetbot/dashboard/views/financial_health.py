"""Enhanced Financial Health Analysis page with improved UI/UX and features."""

# Path setup must be first!
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from typing import Dict, Optional

import pandas as pd
import streamlit as st

# Import analysis functions
from buffetbot.analysis.health_analysis import analyze_financial_health

# Import components using absolute imports
from buffetbot.dashboard.components import (
    create_comparison_table,
    create_progress_indicator,
    display_metric_with_status,
    display_metrics_grid_enhanced,
)

# Import disclaimer components
from buffetbot.dashboard.components.disclaimers import render_investment_disclaimer

logger = logging.getLogger(__name__)


def render_health_scores(health_metrics_result: dict) -> None:
    """Render the health scores section."""
    try:
        st.subheader("🏥 Health Scores")

        # Create columns for scores
        col1, col2 = st.columns(2)

        with col1:
            if "piotroski_score" in health_metrics_result:
                score = health_metrics_result["piotroski_score"]
                status = "good" if score >= 7 else "warning" if score >= 4 else "bad"
                display_metric_with_status(
                    label="Piotroski F-Score",
                    value=f"{score}/9",
                    status=status,
                    help_text="Measures financial strength based on 9 criteria",
                    metric_type="score",
                )

        with col2:
            if "altman_z_score" in health_metrics_result:
                score = health_metrics_result["altman_z_score"]
                status = (
                    "good" if score > 2.99 else "warning" if score > 1.81 else "bad"
                )
                display_metric_with_status(
                    label="Altman Z-Score",
                    value=f"{score:.2f}",
                    status=status,
                    help_text="Predicts probability of bankruptcy",
                    metric_type="score",
                )

    except Exception as e:
        logger.error(f"Error rendering health scores: {str(e)}")
        st.error("Error displaying health scores")


def render_financial_ratios(health_metrics_result: dict) -> None:
    """Render the financial ratios section with an enhanced table layout."""
    try:
        st.subheader("📊 Financial Ratios")
        ratios = health_metrics_result.get("financial_ratios", {})
        calculation_status = health_metrics_result.get("calculation_status", {})

        # Create tabs for different ratio categories
        tab1, tab2, tab3 = st.tabs(["Liquidity", "Leverage", "Profitability"])

        def create_ratio_table(metrics_dict: dict) -> None:
            """Helper function to create a styled ratio table."""
            # Create DataFrame for the table
            data = []
            for name, info in metrics_dict.items():
                value = info["value"]
                status = info["status"]
                help_text = info["help_text"]

                # Check if value is None, NaN, or calculation failed
                if (
                    value is None
                    or pd.isna(value)
                    or (
                        isinstance(value, (int, float))
                        and value == 0
                        and "calculation_failed" in info
                    )
                ):
                    formatted_value = "N/A"
                    status_emoji = "❓"
                else:
                    # Format value based on type
                    if info["type"] == "percentage":
                        formatted_value = f"{value:.1%}"
                    else:
                        formatted_value = f"{value:.2f}"

                    # Add status indicator
                    status_emoji = (
                        "✅"
                        if status == "good"
                        else "⚠️"
                        if status == "warning"
                        else "❌"
                    )

                data.append(
                    {
                        "Metric": name,
                        "Value": formatted_value,
                        "Status": status_emoji,
                        "Description": help_text,
                    }
                )

            # Create DataFrame
            df = pd.DataFrame(data)

            # Style the table
            st.dataframe(
                df,
                column_config={
                    "Metric": st.column_config.TextColumn(
                        "Metric", width="medium", help="Financial ratio name"
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value",
                        width="small",
                        help="Current value (N/A indicates missing or failed calculation)",
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        width="small",
                        help="Health indicator (❓ indicates data unavailable)",
                    ),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large", help="What this ratio means"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

        with tab1:
            st.markdown("### 💧 Liquidity Ratios")
            st.markdown(
                "Measures the company's ability to meet its short-term obligations"
            )

            liquidity_metrics = {
                "Current Ratio": {
                    "value": ratios.get("current_ratio", 0),
                    "metric_key": "current_ratio",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("current_ratio", 0) > 1
                    else "warning",
                    "help_text": "Measures ability to pay short-term obligations",
                    "calculation_failed": not calculation_status.get(
                        "current_ratio", True
                    ),
                },
                "Quick Ratio": {
                    "value": ratios.get("quick_ratio", 0),
                    "metric_key": "quick_ratio",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("quick_ratio", 0) > 0.8
                    else "warning",
                    "help_text": "Measures ability to meet short-term obligations with most liquid assets",
                    "calculation_failed": not calculation_status.get(
                        "quick_ratio", True
                    ),
                },
                "Cash Ratio": {
                    "value": ratios.get("cash_ratio", 0),
                    "metric_key": "cash_ratio",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("cash_ratio", 0) > 0.5
                    else "warning",
                    "help_text": "Measures ability to pay short-term obligations with cash and cash equivalents",
                    "calculation_failed": not calculation_status.get(
                        "cash_ratio", True
                    ),
                },
            }
            create_ratio_table(liquidity_metrics)

        with tab2:
            st.markdown("### ⚖️ Leverage Ratios")
            st.markdown("Measures the company's debt levels and financial leverage")

            leverage_metrics = {
                "Debt to Equity": {
                    "value": ratios.get("debt_to_equity", 0),
                    "metric_key": "debt_to_equity",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("debt_to_equity", 0) < 2
                    else "warning",
                    "help_text": "Measures financial leverage and risk",
                    "calculation_failed": not calculation_status.get(
                        "debt_to_equity", True
                    ),
                },
                "Debt to Assets": {
                    "value": ratios.get("debt_to_assets", 0),
                    "metric_key": "debt_to_assets",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("debt_to_assets", 0) < 0.5
                    else "warning",
                    "help_text": "Measures percentage of assets financed by debt",
                    "calculation_failed": not calculation_status.get(
                        "debt_to_assets", True
                    ),
                },
                "Interest Coverage": {
                    "value": ratios.get("interest_coverage", 0),
                    "metric_key": "interest_coverage",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("interest_coverage", 0) > 3
                    else "warning",
                    "help_text": "Measures ability to pay interest on debt",
                    "calculation_failed": not calculation_status.get(
                        "interest_coverage", True
                    ),
                },
                "Long-term Debt to Equity": {
                    "value": ratios.get("long_term_debt_to_equity", 0),
                    "metric_key": "long_term_debt_to_equity",
                    "type": "ratio",
                    "status": "good"
                    if ratios.get("long_term_debt_to_equity", 0) < 1
                    else "warning",
                    "help_text": "Measures long-term financial leverage",
                    "calculation_failed": not calculation_status.get(
                        "long_term_debt_to_equity", True
                    ),
                },
            }
            create_ratio_table(leverage_metrics)

        with tab3:
            st.markdown("### 💰 Profitability Ratios")
            st.markdown("Measures the company's ability to generate profits")

            profitability_metrics = {
                "Return on Equity": {
                    "value": ratios.get("return_on_equity", 0),
                    "metric_key": "return_on_equity",
                    "type": "percentage",
                    "status": "good"
                    if ratios.get("return_on_equity", 0) > 0.15
                    else "warning",
                    "help_text": "Measures profitability relative to shareholder equity",
                    "calculation_failed": not calculation_status.get(
                        "return_on_equity", True
                    ),
                },
                "Return on Assets": {
                    "value": ratios.get("return_on_assets", 0),
                    "metric_key": "return_on_assets",
                    "type": "percentage",
                    "status": "good"
                    if ratios.get("return_on_assets", 0) > 0.05
                    else "warning",
                    "help_text": "Measures profitability relative to total assets",
                    "calculation_failed": not calculation_status.get(
                        "return_on_assets", True
                    ),
                },
                "Gross Margin": {
                    "value": ratios.get("gross_margin", 0),
                    "metric_key": "gross_margin",
                    "type": "percentage",
                    "status": "good"
                    if ratios.get("gross_margin", 0) > 0.3
                    else "warning",
                    "help_text": "Measures profitability after cost of goods sold",
                    "calculation_failed": not calculation_status.get(
                        "gross_margin", True
                    ),
                },
                "Operating Margin": {
                    "value": ratios.get("operating_margin", 0),
                    "metric_key": "operating_margin",
                    "type": "percentage",
                    "status": "good"
                    if ratios.get("operating_margin", 0) > 0.15
                    else "warning",
                    "help_text": "Measures profitability from operations",
                    "calculation_failed": not calculation_status.get(
                        "operating_margin", True
                    ),
                },
                "Net Margin": {
                    "value": ratios.get("net_margin", 0),
                    "metric_key": "net_margin",
                    "type": "percentage",
                    "status": "good"
                    if ratios.get("net_margin", 0) > 0.1
                    else "warning",
                    "help_text": "Measures overall profitability",
                    "calculation_failed": not calculation_status.get(
                        "net_margin", True
                    ),
                },
            }
            create_ratio_table(profitability_metrics)

    except Exception as e:
        logger.error(f"Error rendering financial ratios: {str(e)}")
        st.error("Error displaying financial ratios")


def render_health_indicators(health_metrics_result: dict) -> None:
    """Render the health indicators section."""
    try:
        st.subheader("🔍 Health Indicators")

        # Get health flags
        health_flags = health_metrics_result.get("health_flags", [])

        if health_flags:
            # Group flags by severity
            critical_flags = [
                flag for flag in health_flags if "critical" in flag.lower()
            ]
            warning_flags = [flag for flag in health_flags if "warning" in flag.lower()]
            info_flags = [
                flag
                for flag in health_flags
                if flag not in critical_flags + warning_flags
            ]

            # Display flags in expandable sections
            if critical_flags:
                with st.expander("⚠️ Critical Issues", expanded=True):
                    for flag in critical_flags:
                        st.error(flag)

            if warning_flags:
                with st.expander("⚠️ Warnings", expanded=True):
                    for flag in warning_flags:
                        st.warning(flag)

            if info_flags:
                with st.expander("ℹ️ Additional Information", expanded=False):
                    for flag in info_flags:
                        st.info(flag)
        else:
            st.success("No health issues detected")

    except Exception as e:
        logger.error(f"Error rendering health indicators: {str(e)}")
        st.error("Error displaying health indicators")


def render_health_trends(health_metrics_result: dict) -> None:
    """Render the health trends section."""
    try:
        st.subheader("📈 Health Trends")

        # Get trend data
        trends = health_metrics_result.get("trends", {})

        if trends:
            # Create comparison table for key metrics
            metrics = []

            for metric, data in trends.items():
                current = data.get("current", 0)
                previous = data.get("previous", 0)
                change = (current - previous) / previous if previous != 0 else 0

                status = "good" if change > 0 else "bad" if change < 0 else "neutral"
                metrics.append((metric, current, previous, status))

            create_comparison_table(
                metrics, headers=["Metric", "Current", "Previous", "Status"]
            )
        else:
            st.info("No trend data available")

    except Exception as e:
        logger.error(f"Error rendering health trends: {str(e)}")
        st.error("Error displaying health trends")


def render_financial_health_page(data: dict, ticker: str) -> None:
    """Render the complete financial health analysis page."""
    try:
        # Analyze financial health
        health_metrics_result = analyze_financial_health(data)

        # Display page header
        st.header(f"Financial Health Analysis: {ticker}")

        # Add disclaimer for financial analysis
        render_investment_disclaimer("analysis")

        # Render each section
        render_health_scores(health_metrics_result)
        st.markdown("---")
        render_financial_ratios(health_metrics_result)
        st.markdown("---")
        render_health_indicators(health_metrics_result)
        st.markdown("---")
        render_health_trends(health_metrics_result)

    except Exception as e:
        logger.error(f"Error in financial health analysis: {str(e)}", exc_info=True)
        st.error(f"Error in financial health analysis: {str(e)}")
