"""Enhanced Price Analysis page with improved UI/UX and features."""

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
from buffetbot.analysis.value_analysis import calculate_intrinsic_value

# Import components using absolute imports
from buffetbot.dashboard.components import (
    PriceValuationCard,
    create_comparison_table,
    create_enhanced_price_gauge,
    create_progress_indicator,
    create_technical_analysis_chart,
    create_valuation_metrics_chart,
    create_valuation_summary,
    display_metric_with_status,
    display_metrics_grid_enhanced,
)

# Import disclaimer components
from buffetbot.dashboard.components.disclaimers import render_investment_disclaimer

logger = logging.getLogger(__name__)


def render_price_analysis_page(data: dict, ticker: str) -> None:
    """Render the enhanced Price Analysis page.

    Args:
        data: Stock data dictionary
        ticker: Stock ticker symbol
    """
    try:
        # Calculate intrinsic value
        intrinsic_value_result = calculate_intrinsic_value(data)

        if (
            intrinsic_value_result
            and intrinsic_value_result.get("intrinsic_value") is not None
        ):
            current_price = data["price_data"]["Close"].iloc[-1]
            intrinsic_value = intrinsic_value_result["intrinsic_value"]

            # Display prominent valuation card at the top
            st.markdown("---")
            valuation_card = PriceValuationCard(current_price, intrinsic_value, ticker)
            valuation_card.render()

            # Add valuation disclaimer after the card
            render_investment_disclaimer("price_valuation")

            # Create tabs for different analysis views
            analysis_tabs = st.tabs(
                [
                    "📊 Valuation Overview",
                    "📈 Technical Analysis",
                    "💰 Detailed Metrics",
                    "🎯 Investment Summary",
                ]
            )

            with analysis_tabs[0]:
                render_valuation_overview(
                    current_price, intrinsic_value, intrinsic_value_result, data
                )

            with analysis_tabs[1]:
                render_technical_analysis(data["price_data"])

            with analysis_tabs[2]:
                render_detailed_metrics(
                    data, intrinsic_value_result, current_price, intrinsic_value
                )

            with analysis_tabs[3]:
                render_investment_summary(
                    data, current_price, intrinsic_value, intrinsic_value_result
                )

        else:
            # Handle case where intrinsic value cannot be calculated
            st.error("❌ Unable to calculate intrinsic value")
            st.warning("Some required financial data may be missing.")

            if intrinsic_value_result and "errors" in intrinsic_value_result:
                with st.expander("🔍 Error Details", expanded=True):
                    for error in intrinsic_value_result["errors"]:
                        st.error(error)

            # Still show available data
            st.markdown("---")
            st.subheader("📊 Available Price Data")
            render_basic_price_metrics(data)

    except Exception as e:
        logger.error(f"Error rendering price analysis page: {str(e)}", exc_info=True)
        st.error(f"Error in price analysis: {str(e)}")


def render_valuation_overview(
    current_price: float,
    intrinsic_value: float,
    intrinsic_value_result: dict,
    data: dict,
) -> None:
    """Render the valuation overview section."""
    try:
        logger.info("Starting valuation overview rendering")
        logger.info(
            f"Input data - current_price: {current_price}, intrinsic_value: {intrinsic_value}"
        )
        logger.info(
            f"Intrinsic value result keys: {list(intrinsic_value_result.keys())}"
        )
        logger.info(f"Data keys: {list(data.keys())}")

        # Create valuation card
        try:
            logger.info("Creating valuation card")
            valuation_card = PriceValuationCard(
                current_price=current_price,
                intrinsic_value=intrinsic_value,
                ticker=data.get("ticker", ""),
            )
            valuation_card.render()
            logger.info("Valuation card rendered successfully")
        except Exception as e:
            logger.error(f"Error creating valuation card: {str(e)}", exc_info=True)
            st.error("Error displaying valuation card")
            return

        # Display valuation summary
        try:
            logger.info("Creating valuation summary")
            create_valuation_summary(
                current_price=current_price,
                intrinsic_value=intrinsic_value,
                pe_ratio=data["fundamentals"].get("pe_ratio"),
                peg_ratio=data["fundamentals"].get("peg_ratio"),
                price_to_book=data["fundamentals"].get("price_to_book"),
            )
            logger.info("Valuation summary created successfully")
        except Exception as e:
            logger.error(f"Error creating valuation summary: {str(e)}", exc_info=True)
            st.error("Error displaying valuation summary")
            return

        # Display key assumptions
        try:
            logger.info("Displaying key assumptions")
            st.subheader("📊 Key Assumptions")

            assumptions = intrinsic_value_result.get("assumptions", {})
            logger.info(f"Available assumptions: {assumptions}")

            assumptions_metrics = {
                "Growth Rate": {
                    "value": assumptions.get("growth_rate", 0),
                    "status": "good"
                    if assumptions.get("growth_rate", 0) > 0.05
                    else "warning",
                    "help_text": "Expected annual growth rate used in DCF calculation",
                    "type": "percentage",
                },
                "Discount Rate": {
                    "value": assumptions.get("discount_rate", 0),
                    "status": "neutral",
                    "help_text": "Required rate of return (WACC)",
                    "type": "percentage",
                },
                "Terminal Growth": {
                    "value": assumptions.get("terminal_growth_rate", 0.025),
                    "status": "neutral",
                    "help_text": "Perpetual growth rate after forecast period",
                    "type": "percentage",
                },
            }

            display_metrics_grid_enhanced(assumptions_metrics, cols=3)
            logger.info("Key assumptions displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying key assumptions: {str(e)}", exc_info=True)
            st.error("Error displaying key assumptions")
            return

        # Display warnings if any
        try:
            if intrinsic_value_result.get("warnings"):
                logger.info("Displaying analysis warnings")
                with st.expander("⚠️ Analysis Warnings", expanded=False):
                    for warning in intrinsic_value_result["warnings"]:
                        st.warning(warning)
                logger.info("Analysis warnings displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying warnings: {str(e)}", exc_info=True)
            st.error("Error displaying analysis warnings")
            return

        # Add valuation metrics radar chart
        try:
            logger.info("Creating valuation metrics radar chart")
            st.markdown("---")
            st.subheader("🎯 Valuation Metrics Overview")

            # Calculate normalized metrics for radar chart (0-100 scale)
            margin_of_safety = intrinsic_value_result.get("margin_of_safety", 0)
            logger.info(f"Margin of safety: {margin_of_safety}")

            # Log available data for metrics calculation
            logger.info(f"Fundamentals data: {data.get('fundamentals', {})}")
            logger.info(f"Metrics data: {data.get('metrics', {})}")

            metrics_normalized = {
                "Value Score": min(margin_of_safety * 200, 100)
                if margin_of_safety is not None and margin_of_safety > 0
                else 0,
                "Growth Score": min(
                    data["fundamentals"].get("revenue_growth", 0) * 500, 100
                )
                if data["fundamentals"].get("revenue_growth", 0) > 0
                else 50,
                "Quality Score": min(data["fundamentals"].get("roe", 0) * 300, 100)
                if data["fundamentals"].get("roe", 0) > 0
                else 50,
                "Financial Health": 80
                if data["fundamentals"].get("debt_to_equity", 1) < 0.5
                else 50,
                "Momentum": min(data["metrics"].get("momentum", 0) * 200 + 50, 100)
                if data["metrics"].get("momentum", 0) > 0
                else 50,
            }
            logger.info(f"Normalized metrics: {metrics_normalized}")

            radar_chart = create_valuation_metrics_chart(metrics_normalized)
            st.plotly_chart(
                radar_chart, use_container_width=True, key="valuation_metrics_radar"
            )
            logger.info("Valuation metrics radar chart created successfully")
        except Exception as e:
            logger.error(
                f"Error creating valuation metrics chart: {str(e)}", exc_info=True
            )
            st.error("Error displaying valuation metrics chart")
            return

    except Exception as e:
        logger.error(f"Error rendering valuation overview: {str(e)}", exc_info=True)
        st.error("Error displaying valuation overview")
        # Display detailed error information in an expander
        with st.expander("Error Details", expanded=True):
            st.error(f"Error type: {type(e).__name__}")
            st.error(f"Error message: {str(e)}")
            st.error("Please check the logs for more details.")


def render_technical_analysis(price_data: pd.DataFrame) -> None:
    """Render the technical analysis section."""
    st.subheader("📈 Technical Analysis")

    # Add technical analysis disclaimer
    render_investment_disclaimer("technical_analysis")

    # Technical indicators selection
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Settings")
        indicators = st.multiselect(
            "Select Indicators",
            ["SMA", "EMA", "BB", "RSI", "MACD"],
            default=["SMA", "BB", "RSI"],
        )

        chart_type = st.radio(
            "Chart Type",
            ["candlestick", "line"],
            index=0
            if all(
                col in price_data.columns for col in ["Open", "High", "Low", "Close"]
            )
            else 1,
        )

    with col1:
        # Display technical analysis chart
        tech_chart = create_technical_analysis_chart(
            price_data, indicators=indicators, chart_type=chart_type
        )
        st.plotly_chart(
            tech_chart, use_container_width=True, key="technical_analysis_chart"
        )

    # Display technical signals
    st.markdown("---")
    st.subheader("🚦 Technical Signals")

    # Calculate technical signals
    signals = calculate_technical_signals(price_data)

    signal_cols = st.columns(4)
    for idx, (signal_name, signal_data) in enumerate(signals.items()):
        with signal_cols[idx % 4]:
            display_metric_with_status(
                signal_name,
                signal_data["value"],
                status=signal_data["status"],
                help_text=signal_data["description"],
            )


def render_detailed_metrics(
    data: dict,
    intrinsic_value_result: dict,
    current_price: float,
    intrinsic_value: float,
) -> None:
    """Render detailed metrics section."""
    try:
        st.subheader("📊 Detailed Metrics")

        # Price metrics
        st.markdown("#### Price Metrics")
        price_metrics = {
            "Latest Price": {
                "value": data["metrics"]["latest_price"],
                "status": "neutral",
                "metric_key": "latest_price",
                "type": "currency",
            },
            "Price Change": {
                "value": data["metrics"]["price_change"],
                "status": "good" if data["metrics"]["price_change"] > 0 else "bad",
                "metric_key": "price_change",
                "type": "percentage",
            },
            "Volatility": {
                "value": data["metrics"]["volatility"],
                "status": "warning" if data["metrics"]["volatility"] > 0.3 else "good",
                "metric_key": "volatility",
                "type": "percentage",
            },
            "RSI": {
                "value": data["metrics"]["rsi"],
                "status": "warning"
                if data["metrics"]["rsi"] > 70 or data["metrics"]["rsi"] < 30
                else "good",
                "metric_key": "rsi",
                "type": "score",
            },
            "Momentum": {
                "value": data["metrics"]["momentum"],
                "status": "good" if data["metrics"]["momentum"] > 0 else "bad",
                "metric_key": "momentum",
                "type": "percentage",
            },
        }

        display_metrics_grid_enhanced(price_metrics, cols=3, show_status_colors=True)

        # Valuation methods breakdown
        if "valuation_breakdown" in intrinsic_value_result:
            st.markdown("---")
            st.subheader("🔍 Valuation Methods Breakdown")

            breakdown = intrinsic_value_result["valuation_breakdown"]
            breakdown_df = pd.DataFrame(breakdown.items(), columns=["Method", "Value"])
            breakdown_df["Value"] = breakdown_df["Value"].apply(
                lambda x: f"${x:.2f}" if x else "N/A"
            )

            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    except Exception as e:
        logger.error(f"Error rendering detailed metrics: {str(e)}")
        st.error("Error displaying detailed metrics")


def render_investment_summary(
    data: dict,
    current_price: float,
    intrinsic_value: float,
    intrinsic_value_result: dict,
) -> None:
    """Render investment summary section."""

    st.subheader("🎯 Investment Summary")

    # Add disclaimer at the top of investment summary
    render_investment_disclaimer("analysis")

    # Calculate key metrics
    margin_of_safety = (
        ((intrinsic_value - current_price) / intrinsic_value)
        if intrinsic_value > 0
        else 0
    )
    upside_potential = (
        ((intrinsic_value - current_price) / current_price) if current_price > 0 else 0
    )

    # Create summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        display_metric_with_status(
            "Upside Potential",
            f"{upside_potential:.1%}",
            status="good"
            if upside_potential > 0.15
            else "warning"
            if upside_potential > 0
            else "bad",
            help_text="Potential gain if price reaches intrinsic value",
        )

    with col2:
        display_metric_with_status(
            "Investment Rating",
            get_investment_rating(margin_of_safety),
            status="good"
            if margin_of_safety > 0.15
            else "warning"
            if margin_of_safety > 0
            else "bad",
            help_text="Overall investment recommendation",
        )

    with col3:
        display_metric_with_status(
            "Risk Level",
            get_risk_level(data["metrics"]["volatility"]),
            status="good"
            if data["metrics"]["volatility"] < 0.2
            else "warning"
            if data["metrics"]["volatility"] < 0.4
            else "bad",
            help_text="Based on historical volatility",
        )

    # Investment thesis
    st.markdown("---")
    st.markdown("### 📝 Investment Thesis")

    thesis = generate_investment_thesis(
        margin_of_safety, data["fundamentals"], data["metrics"], intrinsic_value_result
    )

    for point in thesis:
        st.write(f"• {point}")

    # Radar chart of valuation metrics
    st.markdown("---")
    st.markdown("### 🎯 Valuation Metrics Overview")

    # Normalize metrics for radar chart (0-100 scale)
    metrics_normalized = {
        "Value Score": min(margin_of_safety * 200, 100) if margin_of_safety > 0 else 0,
        "Growth Score": min(data["fundamentals"].get("revenue_growth", 0) * 500, 100)
        if data["fundamentals"].get("revenue_growth", 0) > 0
        else 50,
        "Quality Score": min(data["fundamentals"].get("roe", 0) * 300, 100)
        if data["fundamentals"].get("roe", 0) > 0
        else 50,
        "Financial Health": 80
        if data["fundamentals"].get("debt_to_equity", 1) < 0.5
        else 50,
        "Momentum": min(data["metrics"].get("momentum", 0) * 200 + 50, 100)
        if data["metrics"].get("momentum", 0) > 0
        else 50,
    }

    radar_chart = create_valuation_metrics_chart(metrics_normalized)
    st.plotly_chart(radar_chart, use_container_width=True)

    # Add additional disclaimer for investment summary
    render_investment_disclaimer("price_valuation")


def render_basic_price_metrics(data: dict) -> None:
    """Render basic price metrics when intrinsic value is not available."""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"${data['price_data']['Close'].iloc[-1]:.2f}",
            f"{data['metrics']['price_change']:.1%}",
        )

    with col2:
        st.metric("52-Week High", f"${data['price_data']['Close'].tail(252).max():.2f}")

    with col3:
        st.metric("52-Week Low", f"${data['price_data']['Close'].tail(252).min():.2f}")

    # Show available chart
    st.plotly_chart(
        create_technical_analysis_chart(data["price_data"], indicators=["SMA"]),
        use_container_width=True,
        key="basic_price_chart",
    )


def calculate_technical_signals(price_data: pd.DataFrame) -> dict:
    """Calculate technical trading signals."""

    signals = {}

    try:
        # RSI Signal
        if "rsi" in price_data.columns or len(price_data) >= 14:
            # Calculate RSI if not present
            delta = price_data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if current_rsi > 70:
                signals["RSI Signal"] = {
                    "value": "Overbought",
                    "status": "warning",
                    "description": f"RSI at {current_rsi:.1f} - Consider selling",
                }
            elif current_rsi < 30:
                signals["RSI Signal"] = {
                    "value": "Oversold",
                    "status": "good",
                    "description": f"RSI at {current_rsi:.1f} - Consider buying",
                }
            else:
                signals["RSI Signal"] = {
                    "value": "Neutral",
                    "status": "neutral",
                    "description": f"RSI at {current_rsi:.1f}",
                }

        # Moving Average Signal
        if len(price_data) >= 50:
            sma_20 = price_data["Close"].rolling(window=20).mean().iloc[-1]
            sma_50 = price_data["Close"].rolling(window=50).mean().iloc[-1]
            current_price = price_data["Close"].iloc[-1]

            if current_price > sma_20 > sma_50:
                signals["MA Signal"] = {
                    "value": "Bullish",
                    "status": "good",
                    "description": "Price above rising moving averages",
                }
            elif current_price < sma_20 < sma_50:
                signals["MA Signal"] = {
                    "value": "Bearish",
                    "status": "bad",
                    "description": "Price below falling moving averages",
                }
            else:
                signals["MA Signal"] = {
                    "value": "Mixed",
                    "status": "warning",
                    "description": "Mixed moving average signals",
                }

        # Volume Signal
        if "Volume" in price_data.columns:
            avg_volume = price_data["Volume"].rolling(window=20).mean().iloc[-1]
            current_volume = price_data["Volume"].iloc[-1]

            if current_volume > avg_volume * 1.5:
                signals["Volume Signal"] = {
                    "value": "High Volume",
                    "status": "warning",
                    "description": "Unusual trading activity",
                }
            else:
                signals["Volume Signal"] = {
                    "value": "Normal",
                    "status": "neutral",
                    "description": "Normal trading volume",
                }

        # Trend Signal
        if len(price_data) >= 20:
            recent_trend = (
                price_data["Close"].iloc[-1] - price_data["Close"].iloc[-20]
            ) / price_data["Close"].iloc[-20]

            if recent_trend > 0.05:
                signals["Trend"] = {
                    "value": "Uptrend",
                    "status": "good",
                    "description": f"+{recent_trend:.1%} over 20 days",
                }
            elif recent_trend < -0.05:
                signals["Trend"] = {
                    "value": "Downtrend",
                    "status": "bad",
                    "description": f"{recent_trend:.1%} over 20 days",
                }
            else:
                signals["Trend"] = {
                    "value": "Sideways",
                    "status": "neutral",
                    "description": f"{recent_trend:+.1%} over 20 days",
                }

    except Exception as e:
        logger.error(f"Error calculating technical signals: {str(e)}")

    return signals


def get_investment_rating(margin_of_safety: float) -> str:
    """Get investment rating based on margin of safety."""

    if margin_of_safety >= 0.25:
        return "STRONG BUY"
    elif margin_of_safety >= 0.15:
        return "BUY"
    elif margin_of_safety >= 0.05:
        return "HOLD"
    elif margin_of_safety >= -0.10:
        return "WATCH"
    else:
        return "AVOID"


def get_risk_level(volatility: float) -> str:
    """Get risk level based on volatility."""

    if volatility < 0.15:
        return "Low"
    elif volatility < 0.25:
        return "Moderate"
    elif volatility < 0.40:
        return "High"
    else:
        return "Very High"


def generate_investment_thesis(
    margin_of_safety: float,
    fundamentals: dict,
    metrics: dict,
    intrinsic_value_result: dict,
) -> list:
    """Generate investment thesis points."""

    thesis = []

    # Valuation thesis
    if margin_of_safety >= 0.15:
        thesis.append(
            f"Stock appears undervalued with {margin_of_safety:.1%} margin of safety"
        )
    elif margin_of_safety >= 0:
        thesis.append(
            f"Stock is fairly valued with limited {margin_of_safety:.1%} margin of safety"
        )
    else:
        thesis.append(f"Stock appears overvalued by {abs(margin_of_safety):.1%}")

    # Growth thesis
    if fundamentals.get("revenue_growth", 0) > 0.15:
        thesis.append(
            f"Strong revenue growth of {fundamentals['revenue_growth']:.1%} indicates business expansion"
        )
    elif fundamentals.get("revenue_growth", 0) > 0.05:
        thesis.append(
            f"Moderate revenue growth of {fundamentals['revenue_growth']:.1%} shows stable business"
        )

    # Quality thesis
    if fundamentals.get("roe", 0) > 0.15:
        thesis.append(
            f"High ROE of {fundamentals['roe']:.1%} demonstrates efficient capital utilization"
        )

    # Financial health thesis
    if fundamentals.get("debt_to_equity", 1) < 0.5:
        thesis.append(
            "Strong balance sheet with low debt levels provides financial flexibility"
        )
    elif fundamentals.get("debt_to_equity", 1) > 1.5:
        thesis.append("High debt levels may pose financial risk in economic downturns")

    # Momentum thesis
    if metrics.get("momentum", 0) > 0.1:
        thesis.append(
            f"Positive price momentum of {metrics['momentum']:.1%} suggests market confidence"
        )
    elif metrics.get("momentum", 0) < -0.1:
        thesis.append(
            f"Negative momentum of {metrics['momentum']:.1%} may indicate market concerns"
        )

    # Risk thesis
    if metrics.get("volatility", 0) > 0.3:
        thesis.append(
            f"High volatility of {metrics['volatility']:.1%} suggests increased investment risk"
        )

    return thesis
