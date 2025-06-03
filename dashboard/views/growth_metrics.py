"""Growth metrics tab rendering module."""

from typing import Any, Dict

import streamlit as st

from analysis.growth_analysis import analyze_growth_metrics
from dashboard.components.disclaimers import render_investment_disclaimer
from dashboard.components.metrics import display_metric_with_info, display_metrics_grid
from dashboard.utils.formatters import safe_format_number, safe_format_percentage
from utils.logger import get_logger

logger = get_logger(__name__)


def render_growth_metrics_tab(data: dict[str, Any], ticker: str) -> None:
    """Render the growth metrics tab content.

    Args:
        data: Stock data dictionary
        ticker: Stock ticker symbol
    """
    try:
        # Analyze growth metrics
        growth_metrics_result = analyze_growth_metrics(data)

        if growth_metrics_result:
            # Display growth metrics
            st.subheader("Growth Metrics")

            # Add disclaimer for growth analysis
            render_investment_disclaimer("analysis")

            growth_metrics = {
                "Revenue Growth": {
                    "value": safe_format_percentage(
                        growth_metrics_result.get("revenue_growth")
                    ),
                    "metric_key": "revenue_growth",
                },
                "Earnings Growth": {
                    "value": safe_format_percentage(
                        growth_metrics_result.get("earnings_growth")
                    ),
                    "metric_key": "earnings_growth",
                },
                "EPS Growth": {
                    "value": safe_format_percentage(
                        growth_metrics_result.get("eps_growth")
                    ),
                    "metric_key": "eps_growth",
                },
            }

            display_metrics_grid(growth_metrics, cols=3)

            # Display growth score if available
            if "growth_score" in growth_metrics_result:
                st.markdown("---")
                display_metric_with_info(
                    "Growth Score",
                    safe_format_number(growth_metrics_result.get("growth_score")),
                    "Overall Growth Assessment",
                    metric_key="growth_score",
                )
        else:
            st.warning(
                "Could not calculate growth metrics. Some required financial data may be missing."
            )
    except Exception as e:
        logger.error(f"Error in growth metrics analysis: {str(e)}")
        st.error(f"Error in growth metrics analysis: {str(e)}")
