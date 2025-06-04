"""Analyst Forecast tab rendering module."""

import io
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from buffetbot.dashboard.components.disclaimers import render_investment_disclaimer
from buffetbot.dashboard.components.forecast_panel import render_forecast_panel
from buffetbot.dashboard.components.metrics import display_metric_with_info
from buffetbot.dashboard.config.settings import get_dashboard_config
from buffetbot.dashboard.dashboard_utils.data_processing import handle_ticker_change
from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from buffetbot.data.forecast_fetcher import ForecastFetchError, get_analyst_forecast
from buffetbot.glossary import get_metric_info
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def render_analyst_forecast_tab(data: dict[str, Any], ticker: str) -> None:
    """Render the analyst forecast tab content.

    Args:
        data: Stock data dictionary from the main application
        ticker: Stock ticker symbol from global state
    """
    # Validate inputs
    if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
        st.error(
            "❌ Invalid ticker provided. Please select a valid ticker in the sidebar."
        )
        return

    ticker = ticker.upper().strip()  # Normalize ticker format

    # Handle ticker changes and cache management
    ticker_changed = handle_ticker_change(ticker)

    if not data or not isinstance(data, dict):
        st.warning(
            "⚠️ No stock data available. Please ensure data is loaded for the selected ticker."
        )
        # Continue with limited functionality

    # Analyst Forecast Header
    st.header("🔮 Analyst Forecast Analysis")

    # Check if ticker has changed and provide feedback
    previous_ticker = st.session_state.get("analyst_forecast_previous_ticker", None)
    if previous_ticker and previous_ticker != ticker:
        st.info(
            f"🔄 Ticker updated from {previous_ticker} to {ticker}. The forecast analysis below will reflect the new selection."
        )

    # Store current ticker for next comparison
    st.session_state.analyst_forecast_previous_ticker = ticker

    # Display current ticker prominently with sync status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Analyzing Forecasts for: {ticker}**")
    with col2:
        st.success("🔗 Synced")  # Visual indicator that ticker is synced

    # Add informative description
    st.markdown(
        """
    Comprehensive analyst forecast analysis combining multiple data points including price targets,
    recommendations, earnings estimates, and consensus ratings. This analysis helps you understand
    Wall Street's collective view on the stock's future performance.
    """
    )

    # Add investment disclaimer specific to forecasts
    render_investment_disclaimer("forecast")

    # Enhanced Forecast Panel with additional controls
    st.subheader("📊 Forecast Analysis Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            options=["Standard", "Detailed", "Expert"],
            index=0,
            help="Choose the level of detail for the forecast analysis",
        )

    with col2:
        comparison_mode = st.checkbox(
            "Compare with Historical",
            help="Compare current forecasts with historical accuracy",
        )

    with col3:
        export_data = st.checkbox(
            "📄 Enable Data Export", help="Generate downloadable forecast data"
        )

    st.markdown("---")

    # Main Forecast Analysis
    try:
        # Render the enhanced forecast panel
        forecast_data = render_forecast_panel(ticker)

        if forecast_data:
            # Additional analysis sections
            render_forecast_analytics(forecast_data, ticker, analysis_depth)

            if comparison_mode:
                render_historical_comparison(forecast_data, ticker)

            render_consensus_analysis(forecast_data, ticker)

            if export_data:
                render_export_options(forecast_data, ticker)
        else:
            st.info("📊 No forecast data available for detailed analysis.")

    except Exception as e:
        logger.error(f"Error in analyst forecast tab for {ticker}: {str(e)}")
        st.error(f"An error occurred while analyzing forecasts: {str(e)}")

    # Additional Resources Section
    st.markdown("---")
    render_forecast_resources()


def render_forecast_analytics(
    forecast_data: dict, ticker: str, analysis_depth: str
) -> None:
    """Render advanced forecast analytics based on depth level."""

    st.subheader("📈 Advanced Forecast Analytics")

    if analysis_depth == "Standard":
        render_standard_analytics(forecast_data, ticker)
    elif analysis_depth == "Detailed":
        render_detailed_analytics(forecast_data, ticker)
    else:  # Expert
        render_expert_analytics(forecast_data, ticker)


def render_standard_analytics(forecast_data: dict, ticker: str) -> None:
    """Render standard level analytics."""

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Forecast Reliability",
            f"{forecast_data.get('confidence', 0) * 100:.1f}%",
            help="Based on analyst consensus and historical accuracy",
        )

    with col2:
        upside_potential = forecast_data.get("mean_target", 0) - forecast_data.get(
            "current_price", 0
        )
        st.metric(
            "Upside Potential",
            safe_format_currency(upside_potential),
            help="Difference between mean target and current price",
        )


def render_detailed_analytics(forecast_data: dict, ticker: str) -> None:
    """Render detailed level analytics."""

    render_standard_analytics(forecast_data, ticker)

    # Price target distribution
    st.markdown("#### 🎯 Price Target Distribution")

    targets = [
        forecast_data.get("low_target", 0),
        forecast_data.get("mean_target", 0),
        forecast_data.get("high_target", 0),
    ]

    if any(targets):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Low Target", "Mean Target", "High Target"],
                y=targets,
                marker_color=["red", "blue", "green"],
            )
        )
        fig.update_layout(
            title=f"Analyst Price Targets for {ticker}",
            yaxis_title="Price ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_expert_analytics(forecast_data: dict, ticker: str) -> None:
    """Render expert level analytics."""

    render_detailed_analytics(forecast_data, ticker)

    # Advanced metrics
    st.markdown("#### 🧠 Expert Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Analyst Dispersion",
            f"{(forecast_data.get('high_target', 0) - forecast_data.get('low_target', 0)) / forecast_data.get('mean_target', 1) * 100:.1f}%",
            help="Measure of disagreement among analysts",
        )

    with col2:
        revision_trend = forecast_data.get("revision_trend", "Neutral")
        st.metric(
            "Revision Trend",
            revision_trend,
            help="Recent direction of analyst revisions",
        )

    with col3:
        coverage_quality = min(forecast_data.get("num_analysts", 0) / 10, 1.0)
        st.metric(
            "Coverage Quality",
            f"{coverage_quality * 100:.0f}%",
            help="Quality score based on number of covering analysts",
        )


def render_historical_comparison(forecast_data: dict, ticker: str) -> None:
    """Render historical forecast accuracy comparison."""

    st.subheader("📊 Historical Accuracy Analysis")

    # This would ideally fetch historical data
    # For now, we'll show a placeholder with simulated data
    st.info(
        "📈 Historical accuracy data would be displayed here with actual implementation"
    )

    # Simulated accuracy metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("6-Month Accuracy", "72%", help="Historical accuracy over 6 months")

    with col2:
        st.metric("12-Month Accuracy", "68%", help="Historical accuracy over 12 months")

    with col3:
        st.metric(
            "Bias Direction", "Optimistic", help="Tendency of forecasts vs actual"
        )


def render_consensus_analysis(forecast_data: dict, ticker: str) -> None:
    """Render consensus analysis section."""

    st.subheader("🤝 Consensus Analysis")

    # Recommendation breakdown
    recommendations = forecast_data.get("recommendations", {})

    if recommendations:
        # Create a pie chart for recommendations
        labels = list(recommendations.keys())
        values = list(recommendations.values())

        fig = px.pie(
            values=values, names=labels, title=f"Analyst Recommendations for {ticker}"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Recommendation data not available")


def render_export_options(forecast_data: dict, ticker: str) -> None:
    """Render data export options."""

    st.subheader("📤 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Download Forecast Data (CSV)"):
            # Convert forecast data to DataFrame
            df = pd.DataFrame([forecast_data])

            # Create CSV download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="💾 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{ticker}_analyst_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("📋 Generate Report Summary"):
            summary = generate_forecast_summary(forecast_data, ticker)
            st.text_area("Forecast Summary", summary, height=200)


def generate_forecast_summary(forecast_data: dict, ticker: str) -> str:
    """Generate a text summary of the forecast analysis."""

    mean_target = forecast_data.get("mean_target", 0)
    num_analysts = forecast_data.get("num_analysts", 0)
    confidence = forecast_data.get("confidence", 0) * 100

    summary = f"""
Analyst Forecast Summary for {ticker}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Metrics:
- Mean Price Target: ${mean_target:.2f}
- Number of Analysts: {num_analysts}
- Confidence Score: {confidence:.1f}%

Analysis:
Based on {num_analysts} analysts covering {ticker}, the consensus price target is ${mean_target:.2f}.
The forecast confidence level is {confidence:.1f}%, indicating {'high' if confidence > 70 else 'moderate' if confidence > 40 else 'low'} reliability.

Disclaimer: This analysis is for informational purposes only and should not be considered as investment advice.
    """

    return summary.strip()


def render_forecast_resources() -> None:
    """Render additional resources and educational content."""

    with st.expander("📚 Understanding Analyst Forecasts", expanded=False):
        st.markdown(
            """
        **How to Interpret Analyst Forecasts:**

        1. **Price Targets**: Analyst's estimate of where the stock price will be in 12 months
        2. **Confidence Score**: Our proprietary measure of forecast reliability
        3. **Consensus**: Average of all analyst opinions
        4. **Dispersion**: How much analysts disagree (high dispersion = more uncertainty)

        **Key Considerations:**
        - Forecasts are opinions, not guarantees
        - Recent revisions can indicate changing sentiment
        - Number of analysts affects reliability
        - Consider your own research alongside analyst views
        """
        )

    with st.expander("⚠️ Forecast Limitations", expanded=False):
        st.markdown(
            """
        **Important Limitations:**

        - Analysts may have conflicts of interest
        - Forecasts can be influenced by recent news or trends
        - Historical accuracy varies by analyst and sector
        - Market conditions can change rapidly
        - Past forecast accuracy doesn't guarantee future performance

        **Best Practices:**

        - Use forecasts as one input among many
        - Consider the track record of individual analysts
        - Look for consensus trends rather than single opinions
        - Combine with your own fundamental analysis
        """
        )
