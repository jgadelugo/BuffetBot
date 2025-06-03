"""Risk analysis tab rendering module."""

from typing import Any, Dict

import streamlit as st

from analysis.risk_analysis import analyze_risk_metrics
from dashboard.components.disclaimers import render_investment_disclaimer
from dashboard.components.metrics import display_metric_with_info, display_metrics_grid
from dashboard.utils.data_utils import safe_get_nested_value
from dashboard.utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def render_risk_analysis_tab(data: dict[str, Any], ticker: str) -> None:
    """Render the risk analysis tab content.

    Args:
        data: Stock data dictionary
        ticker: Stock ticker symbol
    """
    try:
        # Analyze risk metrics
        risk_metrics_result = analyze_risk_metrics(data)

        if risk_metrics_result:
            # Display risk metrics
            st.subheader("Risk Metrics")

            # Add disclaimer for risk analysis
            render_investment_disclaimer("analysis")

            # Check if we have overall risk data
            if (
                "overall_risk" in risk_metrics_result
                and risk_metrics_result["overall_risk"]
            ):
                # Overall risk score with color coding
                risk_score = risk_metrics_result["overall_risk"].get("score", 0)
                risk_level = risk_metrics_result["overall_risk"].get("level", "Unknown")

                # Log for debugging
                logger.info(
                    f"Risk Analysis for {ticker}: Score={safe_format_number(risk_score)}%, Level={risk_level}"
                )

                # Create columns for risk score and level
                col1, col2 = st.columns(2)

                with col1:
                    # Risk score gauge
                    display_metric_with_info(
                        "Risk Score",
                        f"{safe_format_number(risk_score)}%",
                        delta=None,
                        metric_key="overall_risk_score",
                    )

                    # Color-coded risk level
                    if risk_level == "High":
                        st.error(f"Risk Level: {risk_level}")
                    elif risk_level == "Moderate":
                        st.warning(f"Risk Level: {risk_level}")
                    elif risk_level == "Low":
                        st.success(f"Risk Level: {risk_level}")
                    else:
                        st.info(f"Risk Level: {risk_level}")

                with col2:
                    # Risk factors
                    st.write("Risk Factors:")
                    factors = risk_metrics_result["overall_risk"].get("factors", [])
                    if factors:
                        for factor in factors[:5]:  # Show first 5 factors
                            st.write(f"• {factor}")
                        if len(factors) > 5:
                            with st.expander(f"Show all {len(factors)} factors"):
                                for factor in factors[5:]:
                                    st.write(f"• {factor}")
                    else:
                        st.write("• No specific risk factors identified")

                # Display warnings and errors if any
                warnings = risk_metrics_result["overall_risk"].get("warnings", [])
                if warnings:
                    with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                        for warning in warnings:
                            st.warning(warning)

                errors = risk_metrics_result["overall_risk"].get("errors", [])
                if errors:
                    with st.expander(f"❌ Errors ({len(errors)})", expanded=True):
                        for error in errors:
                            st.error(error)

                # Add data availability check
                st.markdown("---")
                st.subheader("Data Availability Check")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if (
                        "price_data" in data
                        and data["price_data"] is not None
                        and not data["price_data"].empty
                    ):
                        st.success("✓ Price Data Available")
                    else:
                        st.error("✗ Price Data Missing")

                with col2:
                    # Check beta availability
                    beta_val = safe_get_nested_value(data, "fundamentals", "beta")
                    if beta_val is not None:
                        st.success(f"✓ Beta Available ({safe_format_number(beta_val)})")
                    else:
                        st.warning("⚠ Beta is null")

                with col3:
                    if (
                        "income_stmt" in data
                        and data["income_stmt"] is not None
                        and not data["income_stmt"].empty
                    ):
                        st.success("✓ Financial Data Available")
                    else:
                        st.error("✗ Financial Data Missing")

            else:
                st.warning("Overall risk assessment data is not available")

            # Market Risk
            st.subheader("Market Risk")
            market_risk = risk_metrics_result.get("market_risk", {})
            if market_risk and any(market_risk.values()):
                market_metrics = {}

                if "beta" in market_risk and market_risk["beta"] is not None:
                    market_metrics["Beta"] = {
                        "value": safe_format_number(market_risk["beta"]),
                        "metric_key": "beta",
                    }

                if (
                    "volatility" in market_risk
                    and market_risk["volatility"] is not None
                ):
                    market_metrics["Annualized Volatility"] = {
                        "value": safe_format_percentage(market_risk["volatility"]),
                        "metric_key": "volatility",
                    }

                if market_metrics:
                    display_metrics_grid(market_metrics, cols=2)
                else:
                    st.info("Market risk metrics are not available or have zero values")
            else:
                st.info("No market risk metrics available")

            # Financial Risk
            st.subheader("Financial Risk")
            financial_risk = risk_metrics_result.get("financial_risk", {})
            if financial_risk and any(financial_risk.values()):
                financial_metrics = {}

                if (
                    "debt_to_equity" in financial_risk
                    and financial_risk["debt_to_equity"] is not None
                ):
                    financial_metrics["Debt to Equity"] = {
                        "value": safe_format_number(financial_risk["debt_to_equity"]),
                        "metric_key": "debt_to_equity",
                    }

                if (
                    "interest_coverage" in financial_risk
                    and financial_risk["interest_coverage"] is not None
                ):
                    financial_metrics["Interest Coverage"] = {
                        "value": safe_format_number(
                            financial_risk["interest_coverage"]
                        ),
                        "metric_key": "interest_coverage",
                    }

                if financial_metrics:
                    display_metrics_grid(financial_metrics, cols=2)
                else:
                    st.info(
                        "Financial risk metrics are not available or have zero values"
                    )
            else:
                st.info("No financial risk metrics available")

            # Business Risk
            st.subheader("Business Risk")
            business_risk = risk_metrics_result.get("business_risk", {})
            if business_risk and any(business_risk.values()):
                business_metrics = {}

                if (
                    "operating_margin" in business_risk
                    and business_risk["operating_margin"] is not None
                ):
                    business_metrics["Operating Margin"] = {
                        "value": safe_format_percentage(
                            business_risk["operating_margin"]
                        ),
                        "metric_key": "operating_margin",
                    }

                if (
                    "revenue" in business_risk
                    and business_risk["revenue"] is not None
                    and business_risk["revenue"] > 0
                ):
                    business_metrics["Revenue"] = {
                        "value": safe_format_currency(business_risk["revenue"]),
                        "metric_key": "revenue",
                    }

                if business_metrics:
                    display_metrics_grid(business_metrics, cols=2)
                else:
                    st.info(
                        "Business risk metrics are not available or have zero values"
                    )
            else:
                st.info("No business risk metrics available")
        else:
            st.warning(
                "Could not calculate risk metrics. Some required data may be missing."
            )
            st.info(
                "Try clearing the cache using the button in the sidebar and refreshing the data."
            )
    except Exception as e:
        logger.error(f"Error in risk metrics analysis: {str(e)}", exc_info=True)
        st.error(f"Error in risk metrics analysis: {str(e)}")
        st.info(
            "Try clearing the cache using the button in the sidebar and refreshing the data."
        )
