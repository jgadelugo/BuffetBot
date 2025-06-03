# Path setup must be first!
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import io
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Debug path issues
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: Path(__file__).parent = {Path(__file__).parent}")
print(f"DEBUG: Path(__file__).parent.parent = {Path(__file__).parent.parent}")

# Try to import, if it fails, add parent to path and try again
try:
    from utils.logger import get_logger, setup_logging
except ImportError as e:
    print(f"DEBUG: First import failed: {e}")
    # Add parent directory to path
    parent_path = str(Path(__file__).parent.parent.absolute())
    print(f"DEBUG: Adding to sys.path: {parent_path}")
    sys.path.insert(0, parent_path)
    print(f"DEBUG: sys.path[0] is now: {sys.path[0]}")

    # Check if utils directory exists
    utils_path = Path(parent_path) / "utils"
    print(f"DEBUG: utils directory exists: {utils_path.exists()}")
    if utils_path.exists():
        print(
            f"DEBUG: utils/__init__.py exists: {(utils_path / '__init__.py').exists()}"
        )
        print(f"DEBUG: utils/logger.py exists: {(utils_path / 'logger.py').exists()}")

    try:
        from utils.logger import get_logger, setup_logging

        print("DEBUG: Second import successful!")
    except ImportError as e2:
        print(f"DEBUG: Second import also failed: {e2}")
        raise

from analysis.growth_analysis import analyze_growth_metrics
from analysis.health_analysis import analyze_financial_health
from analysis.options_advisor import (
    InsufficientDataError,
    OptionsAdvisorError,
    get_scoring_weights,
    recommend_long_calls,
)
from analysis.risk_analysis import analyze_risk_metrics
from analysis.value_analysis import calculate_intrinsic_value

# Import new modular components using absolute imports from dashboard
from dashboard.components import (
    create_comparison_table,
    create_progress_indicator,
    display_metric_with_status,
    display_metrics_grid_enhanced,
)

# Import disclaimer components
from dashboard.components.disclaimers import (
    render_compliance_footer,
    render_educational_notice,
    render_investment_disclaimer,
)
from dashboard.components.forecast_panel import render_forecast_panel
from dashboard.pages import render_financial_health_page, render_price_analysis_page
from data.cleaner import clean_financial_data

# Import from BuffetBot modules
from data.fetcher import fetch_stock_data

# Import glossary functions
from glossary import (
    GLOSSARY,
    MetricDefinition,
    get_metric_info,
    get_metrics_by_category,
    search_metrics,
)
from recommend.recommender import generate_recommendation
from utils.data_report import DataCollectionReport

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_format_currency(value: float | None, decimal_places: int = 2) -> str:
    """Safely format a value as currency, handling None values."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"${value:,.{decimal_places}f}"
    except (ValueError, TypeError):
        return "N/A"


def safe_format_percentage(value: float | None, decimal_places: int = 1) -> str:
    """Safely format a value as percentage, handling None values."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{value:.{decimal_places}%}"
    except (ValueError, TypeError):
        return "N/A"


def safe_format_number(value: float | None, decimal_places: int = 2) -> str:
    """Safely format a number, handling None values."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{value:.{decimal_places}f}"
    except (ValueError, TypeError):
        return "N/A"


def safe_get_nested_value(data: dict[str, Any], *keys) -> Any:
    """Safely get a nested dictionary value, returning None if any key is missing."""
    try:
        result = data
        for key in keys:
            if result is None or not isinstance(result, dict) or key not in result:
                return None
            result = result[key]
        return result
    except (KeyError, TypeError, AttributeError):
        return None


def safe_get_last_price(price_data: pd.DataFrame | None) -> float | None:
    """Safely get the last closing price from price data."""
    try:
        if (
            price_data is None
            or not isinstance(price_data, pd.DataFrame)
            or price_data.empty
            or "Close" not in price_data.columns
        ):
            return None
        return float(price_data["Close"].iloc[-1])
    except (IndexError, KeyError, ValueError, TypeError, AttributeError):
        return None


def display_metric_with_info(
    label: str, value: str, delta=None, metric_key: str = None, help_text: str = None
):
    """Display a metric with optional glossary information.

    Args:
        label: The metric label
        value: The metric value
        delta: Optional delta value
        metric_key: Optional key to look up in glossary
        help_text: Optional custom help text (overrides glossary)
    """
    # Check if we should show definitions
    show_definitions = st.session_state.get("show_metric_definitions", True)

    # Get glossary info if available and definitions are enabled
    if show_definitions and metric_key and not help_text:
        try:
            metric_info = get_metric_info(metric_key)
            help_text = (
                f"{metric_info['description']} Formula: {metric_info['formula']}"
            )
        except KeyError:
            help_text = None
    elif not show_definitions:
        help_text = None

    # Display metric with help
    st.metric(label=label, value=value, delta=delta, help=help_text)


def display_table_with_info(df: pd.DataFrame, metric_keys: dict = None):
    """Display a table with help text for metrics.

    Args:
        df: DataFrame with 'Metric'/'Ratio' and 'Value' columns
        metric_keys: Optional dictionary mapping metric names to glossary keys
    """
    # Create a cleaner display without buttons
    st.markdown("---")

    # Determine the metric column name
    metric_col = (
        "Metric"
        if "Metric" in df.columns
        else "Ratio"
        if "Ratio" in df.columns
        else None
    )

    if metric_col is None:
        st.dataframe(df)  # Fallback to standard display
        return

    # Display each metric as a row with help text
    for idx, row in df.iterrows():
        metric_name = row[metric_col]
        metric_value = row["Value"]
        metric_key = metric_keys.get(metric_name) if metric_keys else None

        # Get help text if available
        help_text = None
        if metric_key:
            try:
                metric_info = get_metric_info(metric_key)
                help_text = (
                    f"{metric_info['description']} Formula: {metric_info['formula']}"
                )
            except KeyError:
                pass

        # Create two columns for metric and value
        col1, col2 = st.columns([2, 1])

        with col1:
            if help_text:
                st.write(metric_name, help=help_text)
            else:
                st.write(metric_name)

        with col2:
            st.write(f"**{metric_value}**")


def display_metrics_grid(metrics_dict: dict, cols: int = 3):
    """Display metrics in a grid layout with help text.

    Args:
        metrics_dict: Dictionary of metrics with structure:
            {
                'metric_name': {
                    'value': 'displayed value',
                    'metric_key': 'glossary key',
                    'delta': 'optional delta value'
                }
            }
        cols: Number of columns in the grid
    """
    # Create columns
    columns = st.columns(cols)

    # Display metrics
    for idx, (metric_name, metric_data) in enumerate(metrics_dict.items()):
        col_idx = idx % cols

        with columns[col_idx]:
            # Check if we should show definitions
            show_definitions = st.session_state.get("show_metric_definitions", True)

            # Get help text if available and definitions are enabled
            help_text = None
            if show_definitions and "metric_key" in metric_data:
                try:
                    metric_info = get_metric_info(metric_data["metric_key"])
                    help_text = f"{metric_info['description']} Formula: {metric_info['formula']}"
                except KeyError:
                    pass

            # Display metric
            st.metric(
                label=metric_name,
                value=metric_data["value"],
                delta=metric_data.get("delta"),
                help=help_text,
            )


# Cache stock data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(ticker: str, years: int = 5):
    """Fetch and process stock data with caching."""
    try:
        # Fetch raw data
        raw_data = fetch_stock_data(ticker, years)

        # Clean and process data
        cleaned_data = clean_financial_data(
            {
                "income_stmt": raw_data["income_stmt"],
                "balance_sheet": raw_data["balance_sheet"],
                "cash_flow": raw_data["cash_flow"],
            }
        )

        # Add price data and fundamentals
        cleaned_data["price_data"] = raw_data["price_data"]
        cleaned_data["fundamentals"] = raw_data["fundamentals"]
        cleaned_data["metrics"] = raw_data["metrics"]

        return cleaned_data
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def create_price_gauge(current_price: float, intrinsic_value: float) -> go.Figure:
    """Create a gauge chart for price comparison."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_price,
            title={"text": "Current Price vs Intrinsic Value"},
            gauge={
                "axis": {"range": [0, max(current_price, intrinsic_value) * 1.2]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, intrinsic_value], "color": "lightgray"},
                    {
                        "range": [intrinsic_value, intrinsic_value * 1.2],
                        "color": "gray",
                    },
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": intrinsic_value,
                },
            },
        )
    )
    return fig


def create_growth_chart(price_data: pd.DataFrame) -> go.Figure:
    """Create a growth chart with moving averages and Bollinger Bands."""
    try:
        # Calculate technical indicators
        df = price_data.copy()

        # Calculate moving averages
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()

        # Calculate Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Std"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
        df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)

        # Create figure
        fig = go.Figure()

        # Add price line
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="blue"))
        )

        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                name="20-day MA",
                line=dict(color="orange", dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA50"],
                name="50-day MA",
                line=dict(color="green", dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA200"],
                name="200-day MA",
                line=dict(color="red", dash="dash"),
            )
        )

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="gray", dash="dot"),
                fill=None,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="gray", dash="dot"),
                fill="tonexty",
            )
        )

        fig.update_layout(
            title="Price History with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating growth chart: {str(e)}")
        # Return a simple price chart if technical indicators fail
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"],
                name="Price",
                line=dict(color="blue"),
            )
        )
        fig.update_layout(
            title="Price History", xaxis_title="Date", yaxis_title="Price"
        )
        return fig


def render_metric_card(key: str, metric: MetricDefinition):
    """Render a single metric as a styled card."""
    category_class = f"category-{metric['category']}"

    card_html = f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 1.2em; font-weight: bold; color: #1f77b4; margin-bottom: 10px;">{metric['name']}</div>
        <span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; margin-bottom: 10px; background-color: {'#d4edda' if metric['category'] == 'growth' else '#cce5ff' if metric['category'] == 'value' else '#fff3cd' if metric['category'] == 'health' else '#f8d7da'}; color: {'#155724' if metric['category'] == 'growth' else '#004085' if metric['category'] == 'value' else '#856404' if metric['category'] == 'health' else '#721c24'};">{metric['category'].upper()}</span>
        <div style="color: #495057; line-height: 1.6; margin: 10px 0;">{metric['description']}</div>
        <div style="margin-top: 15px;">
            <strong>Formula:</strong>
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; color: #212529;">{metric['formula']}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    st.title("Stock Analysis Dashboard")

    # Add educational disclaimer at the top
    render_investment_disclaimer("header")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)

    # Add cache management section
    st.sidebar.markdown("---")
    st.sidebar.header("Cache Management")
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and refresh"):
        get_stock_info.clear()
        st.success("Cache cleared! Data will be refreshed.")
        st.rerun()

    # Add metric definitions toggle
    st.sidebar.markdown("---")
    st.sidebar.header("Display Settings")

    # Initialize session state for metric definitions
    if "show_metric_definitions" not in st.session_state:
        st.session_state.show_metric_definitions = True

    # Toggle for metric definitions
    st.session_state.show_metric_definitions = st.sidebar.checkbox(
        "Show Metric Definitions",
        value=st.session_state.show_metric_definitions,
        help="Toggle to show/hide metric descriptions and formulas throughout the dashboard",
    )

    # Fetch and process data
    data = get_stock_info(ticker, years)

    if data is None:
        st.error(f"Could not fetch data for {ticker}")
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

    with tab1:
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
        report = DataCollectionReport(data)
        report_data = report.get_report()
        quality_score = report_data.get("data_quality_score", 0)
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
                    <h1 style='color: {score_color}; margin: 10px 0;'>{safe_format_percentage(quality_score)}</h1>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                ### Data Collection Report
                View detailed information about the collected data, including:
                - Data availability status
                - Missing columns and metrics
                - Data quality indicators
                - Impact on analysis
                - Recommendations for improvement
            """
            )
            if st.button("View Data Collection Report"):
                st.session_state["show_data_report"] = True
                st.rerun()

    with tab2:
        # Use the new enhanced Price Analysis page
        render_price_analysis_page(data, ticker)

    with tab3:
        # Use the new enhanced Financial Health page
        render_financial_health_page(data, ticker)

    with tab4:
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

    with tab5:
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
                    risk_level = risk_metrics_result["overall_risk"].get(
                        "level", "Unknown"
                    )

                    # Log for debugging
                    logger.info(
                        f"Risk Analysis for {ticker}: Score={risk_score:.2f}%, Level={risk_level}"
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
                        with st.expander(
                            f"⚠️ Warnings ({len(warnings)})", expanded=False
                        ):
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
                            st.success(
                                f"✓ Beta Available ({safe_format_number(beta_val)})"
                            )
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
                        st.info(
                            "Market risk metrics are not available or have zero values"
                        )
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
                            "value": safe_format_number(
                                financial_risk["debt_to_equity"]
                            ),
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

    with tab6:
        # Glossary header
        st.header("📚 Financial Metrics Glossary")
        st.markdown("Comprehensive guide to financial metrics used in this analysis")

        # Create two columns for layout
        col1, col2 = st.columns([1, 3])

        with col1:
            # Search and filter controls
            st.subheader("🔍 Search & Filter")

            # Search box
            search_term = st.text_input(
                "Search metrics", placeholder="Enter term...", key="glossary_search"
            )

            # Category filter
            st.subheader("Categories")
            categories = ["All", "Growth", "Value", "Health", "Risk"]

            # Use session state for selected category
            if "glossary_category" not in st.session_state:
                st.session_state.glossary_category = "All"

            selected_category = st.radio(
                "Filter by category",
                categories,
                index=categories.index(st.session_state.glossary_category),
                key="glossary_category_radio",
            )

            # Quick stats
            st.subheader("📊 Statistics")
            total_metrics = len(GLOSSARY)
            st.metric("Total Metrics", total_metrics)

            # Category counts
            for cat in ["growth", "value", "health", "risk"]:
                count = len(get_metrics_by_category(cat))
                st.caption(f"{cat.title()}: {count}")

        with col2:
            # Apply filters
            if search_term:
                filtered_metrics = search_metrics(search_term)
                st.caption(
                    f"Found {len(filtered_metrics)} metrics matching '{search_term}'"
                )
            else:
                if selected_category == "All":
                    filtered_metrics = GLOSSARY
                else:
                    filtered_metrics = get_metrics_by_category(
                        selected_category.lower()
                    )

            # Display metrics
            if filtered_metrics:
                # Group by category if showing all
                if not search_term and selected_category == "All":
                    for category in ["growth", "value", "health", "risk"]:
                        category_metrics = {
                            k: v
                            for k, v in filtered_metrics.items()
                            if v["category"] == category
                        }

                        if category_metrics:
                            # Category header
                            emoji_map = {
                                "growth": "📈",
                                "value": "💰",
                                "health": "💪",
                                "risk": "⚠️",
                            }

                            with st.expander(
                                f"{emoji_map.get(category, '📊')} {category.upper()} METRICS ({len(category_metrics)} items)",
                                expanded=True,
                            ):
                                for key, metric in category_metrics.items():
                                    render_metric_card(key, metric)
                else:
                    # Display filtered results without grouping
                    for key, metric in filtered_metrics.items():
                        render_metric_card(key, metric)
            else:
                st.info("No metrics found matching your criteria.")

            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")

            # Prepare data for export
            export_data = []
            for key, metric in GLOSSARY.items():
                export_data.append(
                    {
                        "Key": key,
                        "Name": metric["name"],
                        "Category": metric["category"],
                        "Description": metric["description"],
                        "Formula": metric["formula"],
                    }
                )

            df = pd.DataFrame(export_data)

            col1_export, col2_export = st.columns(2)

            with col1_export:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="financial_metrics_glossary.csv",
                    mime="text/csv",
                )

            with col2_export:
                # JSON download
                json_str = json.dumps(GLOSSARY, indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name="financial_metrics_glossary.json",
                    mime="application/json",
                )

    with tab7:
        # Options Advisor
        st.header("🎯 Options Advisor")

        # Add informative description with tooltips
        st.markdown(
            """
        Analyze long-dated call options using comprehensive technical scoring.
        This tool combines RSI, Beta, Momentum, and Implied Volatility to recommend the best option contracts.
        """
        )

        # Add prominent options trading disclaimer
        render_investment_disclaimer("options")

        # Create input section
        st.subheader("📊 Analysis Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            options_ticker = (
                st.text_input(
                    "Stock Ticker",
                    value="AAPL",
                    help="Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
                )
                .upper()
                .strip()
            )

        with col2:
            # Get help text for days to expiry based on metric definitions toggle
            show_definitions = st.session_state.get("show_metric_definitions", True)
            days_help_text = "Minimum number of days until option expiration. Longer terms provide more time value."

            if show_definitions:
                try:
                    metric_info = get_metric_info("days_to_expiry")
                    days_help_text = f"{metric_info['description']} Formula: {metric_info['formula']}"
                except KeyError:
                    pass

            min_days = st.slider(
                "Minimum Days to Expiry",
                min_value=90,
                max_value=720,
                value=180,
                help=days_help_text,
            )

        with col3:
            top_n = st.slider(
                "Number of Recommendations",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top-ranked option recommendations to display",
            )

        # CSV download checkbox
        download_csv = st.checkbox(
            "📄 Include CSV Download",
            help="Generate a downloadable CSV file of the recommendations",
        )

        # Add Forecast Insight Panel
        st.markdown("---")

        # Render the forecast panel if we have a ticker
        forecast_data = None
        if options_ticker:
            forecast_data = render_forecast_panel(options_ticker)

        st.markdown("---")

        # Analyze button
        if st.button("🔍 Analyze Options", type="primary"):
            if not options_ticker:
                st.error("Please enter a valid ticker symbol")
            else:
                # Log the interaction
                logger.info(
                    f"Options analysis requested for {options_ticker} - min_days={min_days}, top_n={top_n}"
                )

                # Create loading placeholder
                with st.spinner(f"🔄 Analyzing options for {options_ticker}..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Update progress
                        status_text.text("Fetching options data...")
                        progress_bar.progress(25)

                        # Call the recommend_long_calls function
                        status_text.text("Computing technical indicators...")
                        progress_bar.progress(50)

                        recommendations = recommend_long_calls(
                            ticker=options_ticker, min_days=min_days, top_n=top_n
                        )

                        status_text.text("Calculating composite scores...")
                        progress_bar.progress(75)

                        if recommendations.empty:
                            progress_bar.progress(100)
                            status_text.empty()
                            st.warning(
                                f"⚠️ No options found for {options_ticker} with minimum {min_days} days to expiry"
                            )
                            logger.warning(
                                f"No options found for {options_ticker} with min_days={min_days}"
                            )
                        else:
                            progress_bar.progress(100)
                            status_text.text("Analysis complete!")

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()

                            # Display success message
                            st.success(
                                f"✅ Found {len(recommendations)} option recommendations for {options_ticker}"
                            )
                            logger.info(
                                f"Options analysis completed for {options_ticker} - returned {len(recommendations)} recommendations"
                            )

                            # Display results section
                            st.subheader("📈 Top Option Recommendations")

                            # Display forecast summary if available
                            if forecast_data:
                                st.markdown("#### 🎯 Forecast Context")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(
                                        "🎯 Analyst Target",
                                        safe_format_currency(
                                            forecast_data["mean_target"]
                                        ),
                                        help="Average analyst price target",
                                    )

                                with col2:
                                    st.metric(
                                        "🔒 Forecast Confidence",
                                        safe_format_percentage(
                                            forecast_data["confidence"]
                                        ),
                                        help="Analyst consensus confidence score",
                                    )

                                with col3:
                                    st.metric(
                                        "👥 Analysts",
                                        str(forecast_data["num_analysts"]),
                                        help="Number of analysts providing targets",
                                    )

                                st.markdown("---")

                            # Display key metrics with tooltips
                            st.markdown("#### Key Technical Indicators")

                            # Get scoring weights for display
                            scoring_weights = get_scoring_weights()

                            # Create columns for key metrics display
                            (
                                met_col1,
                                met_col2,
                                met_col3,
                                met_col4,
                                met_col5,
                            ) = st.columns(5)

                            # Get the current values for display (from first recommendation)
                            if not recommendations.empty:
                                first_row = recommendations.iloc[0]

                                with met_col1:
                                    display_metric_with_info(
                                        "RSI",
                                        safe_format_number(first_row["RSI"]),
                                        delta=None,
                                        metric_key="rsi",
                                    )

                                with met_col2:
                                    display_metric_with_info(
                                        "Beta",
                                        safe_format_number(first_row["Beta"]),
                                        delta=None,
                                        metric_key="beta",
                                    )

                                with met_col3:
                                    display_metric_with_info(
                                        "Momentum",
                                        safe_format_number(first_row["Momentum"]),
                                        delta=None,
                                        metric_key="momentum",
                                    )

                                with met_col4:
                                    display_metric_with_info(
                                        "Avg IV",
                                        safe_format_percentage(first_row["IV"]),
                                        delta=None,
                                        metric_key="implied_volatility",
                                    )

                                with met_col5:
                                    if "ForecastConfidence" in first_row:
                                        display_metric_with_info(
                                            "Forecast",
                                            safe_format_percentage(
                                                first_row["ForecastConfidence"]
                                            ),
                                            delta=None,
                                            help_text="Analyst forecast confidence score",
                                        )

                            st.markdown("---")

                            # Create a table with metric-aware headers
                            st.markdown("#### Options Recommendations")

                            # Create column headers with tooltips using metric info
                            show_definitions = st.session_state.get(
                                "show_metric_definitions", True
                            )

                            if show_definitions:
                                col_headers = []
                                col_headers.append("Strike Price")
                                col_headers.append("Expiration")
                                col_headers.append("Option Price")
                                col_headers.append("RSI")
                                col_headers.append("IV")
                                col_headers.append("Momentum")
                                col_headers.append("Forecast")
                                col_headers.append("Composite Score")

                                # Create tooltips for headers
                                header_help = {
                                    "Strike Price": "option_strike",
                                    "Expiration": "option_expiry",
                                    "Option Price": "option_price",
                                    "RSI": "rsi",
                                    "IV": "implied_volatility",
                                    "Momentum": "momentum",
                                    "Forecast": "forecast_confidence",
                                    "Composite Score": "composite_score",
                                }

                                # Display headers with help text in a table-like format
                                st.markdown("**Column Definitions:**")
                                help_cols = st.columns(len(col_headers))
                                for i, (header, metric_key) in enumerate(
                                    zip(col_headers, header_help.values())
                                ):
                                    with help_cols[i]:
                                        try:
                                            metric_info = get_metric_info(metric_key)
                                            help_text = f"{metric_info['description']}"
                                            st.markdown(
                                                f"**{header}** ℹ️", help=help_text
                                            )
                                        except KeyError:
                                            st.markdown(f"**{header}**")

                            # Format the data for display
                            display_df = recommendations.copy()

                            # Format numerical columns
                            display_df["Strike"] = display_df["strike"].apply(
                                lambda x: safe_format_currency(x)
                            )
                            display_df["Price"] = display_df["lastPrice"].apply(
                                lambda x: safe_format_currency(x)
                            )
                            display_df["RSI"] = display_df["RSI"].apply(
                                lambda x: safe_format_number(x)
                            )
                            display_df["Beta"] = display_df["Beta"].apply(
                                lambda x: safe_format_number(x)
                            )
                            display_df["Momentum"] = display_df["Momentum"].apply(
                                lambda x: safe_format_number(x)
                            )
                            display_df["IV"] = display_df["IV"].apply(
                                lambda x: safe_format_percentage(x)
                            )
                            display_df["Forecast"] = display_df[
                                "ForecastConfidence"
                            ].apply(lambda x: safe_format_percentage(x))
                            display_df["Score"] = display_df["CompositeScore"].apply(
                                lambda x: safe_format_number(x)
                            )

                            # Select and rename columns for display
                            display_df = display_df[
                                [
                                    "Strike",
                                    "expiry",
                                    "Price",
                                    "RSI",
                                    "IV",
                                    "Momentum",
                                    "Forecast",
                                    "Score",
                                ]
                            ]
                            display_df.columns = [
                                "Strike",
                                "Expiry",
                                "Price",
                                "RSI",
                                "IV",
                                "Momentum",
                                "Forecast",
                                "Composite Score",
                            ]

                            # Style the dataframe with conditional formatting
                            def highlight_rsi(val):
                                """Color code RSI values"""
                                try:
                                    numeric_val = float(val)
                                    if numeric_val > 70:
                                        return "background-color: #ffcdd2; color: #d32f2f"  # Red for overbought
                                    elif numeric_val < 30:
                                        return "background-color: #c8e6c9; color: #2e7d32"  # Green for oversold
                                    else:
                                        return "background-color: #fff3e0; color: #f57c00"  # Orange for neutral
                                except:
                                    return ""

                            def highlight_score(val):
                                """Color code composite scores"""
                                try:
                                    numeric_val = float(val)
                                    if numeric_val >= 0.7:
                                        return "background-color: #c8e6c9; color: #2e7d32; font-weight: bold"  # Strong green
                                    elif numeric_val >= 0.5:
                                        return "background-color: #fff3e0; color: #f57c00; font-weight: bold"  # Orange
                                    else:
                                        return "background-color: #ffcdd2; color: #d32f2f"  # Light red
                                except:
                                    return ""

                            def highlight_iv(val):
                                """Color code IV values - lower is generally better for buying calls"""
                                try:
                                    # Extract percentage value
                                    numeric_val = float(val.strip("%")) / 100
                                    if numeric_val <= 0.3:
                                        return "background-color: #c8e6c9; color: #2e7d32"  # Green for low IV
                                    elif numeric_val <= 0.5:
                                        return "background-color: #fff3e0; color: #f57c00"  # Orange for medium IV
                                    else:
                                        return "background-color: #ffcdd2; color: #d32f2f"  # Red for high IV
                                except:
                                    return ""

                            def highlight_forecast(val):
                                """Color code forecast confidence values"""
                                try:
                                    numeric_val = float(val.strip("%")) / 100
                                    if numeric_val >= 0.7:
                                        return "background-color: #c8e6c9; color: #2e7d32"  # Green for high confidence
                                    elif numeric_val >= 0.4:
                                        return "background-color: #fff3e0; color: #f57c00"  # Orange for medium confidence
                                    else:
                                        return "background-color: #ffcdd2; color: #d32f2f"  # Red for low confidence
                                except:
                                    return ""

                            # Apply styling
                            styled_df = (
                                display_df.style.applymap(highlight_rsi, subset=["RSI"])
                                .applymap(highlight_score, subset=["Composite Score"])
                                .applymap(highlight_iv, subset=["IV"])
                                .applymap(highlight_forecast, subset=["Forecast"])
                                .format(
                                    {
                                        "Expiry": lambda x: pd.to_datetime(x).strftime(
                                            "%Y-%m-%d"
                                        )
                                        if pd.notna(x)
                                        else ""
                                    }
                                )
                            )

                            # Display the styled dataframe
                            st.dataframe(
                                styled_df, use_container_width=True, hide_index=True
                            )

                            # Add color legend
                            st.markdown(
                                """
                            **Color Legend:**
                            - 🟢 **Green**: Favorable values (Low RSI/IV, High Score/Forecast Confidence)
                            - 🟠 **Orange**: Neutral/Moderate values
                            - 🔴 **Red**: Less favorable values (High RSI/IV, Low Score/Forecast Confidence)
                            """
                            )

                            # CSV Download functionality
                            if download_csv:
                                st.subheader("📄 Download Data")

                                # Prepare raw data for CSV
                                csv_df = recommendations.copy()
                                csv_df["analysis_date"] = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                csv_df[
                                    "parameters"
                                ] = f"min_days={min_days}, top_n={top_n}"

                                # Convert to CSV
                                csv_buffer = io.StringIO()
                                csv_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()

                                # Create download button
                                filename = f"{options_ticker}_options_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=filename,
                                    mime="text/csv",
                                    help="Download the complete options analysis results as a CSV file",
                                )

                                logger.info(
                                    f"CSV download prepared for {options_ticker} options analysis"
                                )

                                # Show preview of CSV data
                                with st.expander("📋 CSV Preview", expanded=False):
                                    st.dataframe(
                                        csv_df.head(), use_container_width=True
                                    )

                    except OptionsAdvisorError as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"⚠️ Options analysis error: {str(e)}")
                        logger.error(
                            f"Options analysis error for {options_ticker}: {str(e)}"
                        )

                        # Provide user-friendly suggestions
                        if "No long-dated call options found" in str(e):
                            st.info(
                                """
                            **Suggestions:**
                            - Try a different ticker symbol
                            - Reduce the minimum days to expiry
                            - Check if the stock has active options trading
                            """
                            )

                    except InsufficientDataError as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning(f"📊 Insufficient data: {str(e)}")
                        logger.warning(
                            f"Insufficient data for {options_ticker}: {str(e)}"
                        )

                        st.info(
                            """
                        **This might happen if:**
                        - The stock is newly listed
                        - Limited trading history available
                        - Market data is temporarily unavailable
                        """
                        )

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"🚨 Unexpected error occurred: {str(e)}")
                        logger.error(
                            f"Unexpected error in options analysis for {options_ticker}: {str(e)}",
                            exc_info=True,
                        )

                        st.info(
                            """
                        **If this error persists:**
                        - Try refreshing the page
                        - Check your internet connection
                        - Contact support if the issue continues
                        """
                        )

        # Add helpful information section
        st.markdown("---")
        st.subheader("💡 How It Works")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Technical Analysis Framework:**
            1. **RSI Analysis** - Identifies momentum conditions
            2. **Beta Calculation** - Measures market correlation
            3. **Price Momentum** - Evaluates trend strength
            4. **Implied Volatility** - Assesses option pricing
            5. **Analyst Forecasts** - Incorporates forward-looking sentiment
            """
            )

        with col2:
            st.markdown(
                """
            **Scoring Methodology:**
            - Each metric is normalized to 0-1 scale
            - Weighted composite score combines all factors
            - Analyst forecast confidence included in scoring
            - Time-scoped forecast filtering available
            - Higher scores indicate more attractive options
            - Results ranked by composite score
            """
            )

        # Add new forecast methodology section
        st.markdown("---")
        st.subheader("🧠 Forecast Analysis Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
            **📊 Forecast Metrics:**
            - Mean & Median Targets
            - High & Low Target Range
            - Standard Deviation
            - Number of Analysts
            - Confidence Score (0-100%)
            """
            )

        with col2:
            st.markdown(
                """
            **🗓️ Time-Scoped Analysis:**
            - All forecasts (default)
            - Last 1 month
            - Last 3 months
            - Last 6 months
            - Recency-adjusted confidence
            """
            )

        with col3:
            st.markdown(
                """
            **📈 Visualization:**
            - Target distribution charts
            - Forecast evolution trends
            - Confidence level indicators
            - Interactive time filtering
            """
            )

        # Add disclaimer
        st.markdown(
            """
        ---
        ⚠️ **Disclaimer**: This tool is for educational and research purposes only.
        Options trading involves significant risk and may not be suitable for all investors.
        Analyst forecasts are opinions and may not reflect actual future performance.
        Always consult with a qualified financial advisor before making investment decisions.
        """
        )

    # Check if we should show the data collection report
    if st.session_state.get("show_data_report", False):
        st.title("Data Collection Report")

        # Add back button
        if st.button("← Back to Dashboard"):
            st.session_state.show_data_report = False
            st.rerun()

        # Display data quality score
        quality_score = report_data.get("data_quality_score", 0)
        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px; background-color: {
                '#4CAF50' if quality_score >= 80 else
                '#FFA500' if quality_score >= 50 else
                '#FF5252'
            }; color: white; border-radius: 10px;'>
                <h2>Data Quality Score: {safe_format_percentage(quality_score)}</h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Display validation results
        st.subheader("Data Validation")
        validation = report_data.get("data_validation", {})

        for statement, status in validation.items():
            with st.expander(f"{statement.replace('_', ' ').title()} Validation"):
                if status["is_valid"]:
                    st.success("✓ Data structure is valid")
                else:
                    st.error("✗ Data structure has issues")

                if status["errors"]:
                    st.error("Errors:")
                    for error in status["errors"]:
                        st.write(f"- {error}")

                if status["warnings"]:
                    st.warning("Warnings:")
                    for warning in status["warnings"]:
                        st.write(f"- {warning}")

        # Display data availability
        st.subheader("Data Availability")
        availability = report_data.get("data_availability", {})

        for statement, status in availability.items():
            with st.expander(f"{statement.replace('_', ' ').title()} Availability"):
                if status["available"]:
                    st.success("✓ Data is available")
                    st.write(
                        f"Completeness: {safe_format_percentage(status['completeness'])}"
                    )
                    st.write(f"Last available date: {status['last_available_date']}")

                    if status["missing_columns"]:
                        st.warning("Missing columns:")
                        for col in status["missing_columns"]:
                            st.write(f"- {col}")

                    if status["data_quality_issues"]:
                        st.warning("Data quality issues:")
                        for issue in status["data_quality_issues"]:
                            st.write(f"- {issue}")
                else:
                    st.error("✗ Data is not available")
                    if "collection_status" in status:
                        st.error(f"Error: {status['collection_status']['error']}")
                        st.error(f"Reason: {status['collection_status']['reason']}")
                        if "details" in status["collection_status"]:
                            st.error(
                                f"Details: {status['collection_status']['details']}"
                            )

        # Display impact analysis
        st.subheader("Impact Analysis")
        impact = report_data.get("impact_analysis", {})

        for category, metrics in impact.items():
            with st.expander(f"{category.replace('_', ' ').title()} Impact"):
                if metrics:
                    st.warning(f"The following {category} metrics are affected:")
                    for metric in metrics:
                        st.write(f"- {metric}")
                else:
                    st.success(f"No {category} metrics are affected")

        # Display recommendations
        st.subheader("Recommendations")
        recommendations = report_data.get("recommendations", [])

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(
                    f"""
                    <div style='padding: 10px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px;'>
                        <strong>Recommendation {i}:</strong> {rec}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No recommendations - all required data is available and valid")

    # Add comprehensive compliance footer
    render_compliance_footer()


if __name__ == "__main__":
    main()
