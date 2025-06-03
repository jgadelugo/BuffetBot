"""Enhanced chart components for financial visualization."""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def create_enhanced_price_gauge(
    current_price: float, intrinsic_value: float, show_zones: bool = True
) -> go.Figure:
    """Create an enhanced gauge chart with valuation zones.

    Args:
        current_price: Current market price
        intrinsic_value: Calculated intrinsic value
        show_zones: Whether to show valuation zones

    Returns:
        Plotly figure object
    """
    try:
        # Calculate range for gauge
        max_value = max(current_price, intrinsic_value) * 1.3

        # Calculate percentage difference
        price_diff_pct = ((current_price - intrinsic_value) / intrinsic_value) * 100

        # Determine gauge color based on valuation
        if current_price < intrinsic_value * 0.75:
            gauge_color = "#006400"  # Deep green - deeply undervalued
        elif current_price < intrinsic_value * 0.9:
            gauge_color = "#228B22"  # Green - undervalued
        elif current_price < intrinsic_value:
            gauge_color = "#32CD32"  # Light green - slightly undervalued
        elif current_price < intrinsic_value * 1.1:
            gauge_color = "#FFA500"  # Orange - fairly valued
        elif current_price < intrinsic_value * 1.25:
            gauge_color = "#FF6347"  # Tomato - slightly overvalued
        else:
            gauge_color = "#DC143C"  # Red - overvalued

        # Create gauge figure
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_price,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": f"Current Price vs Intrinsic Value<br><span style='font-size:14px'>Difference: {price_diff_pct:+.1f}%</span>",
                    "font": {"size": 20},
                },
                delta={
                    "reference": intrinsic_value,
                    "relative": True,
                    "valueformat": ".1%",
                },
                gauge={
                    "axis": {
                        "range": [0, max_value],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": gauge_color, "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": []
                    if not show_zones
                    else [
                        {
                            "range": [0, intrinsic_value * 0.75],
                            "color": "#90EE90",
                        },  # Light green - deep value zone
                        {
                            "range": [intrinsic_value * 0.75, intrinsic_value * 0.9],
                            "color": "#98FB98",
                        },  # Pale green - value zone
                        {
                            "range": [intrinsic_value * 0.9, intrinsic_value],
                            "color": "#F0E68C",
                        },  # Khaki - slight value zone
                        {
                            "range": [intrinsic_value, intrinsic_value * 1.1],
                            "color": "#FFE4B5",
                        },  # Moccasin - fair value zone
                        {
                            "range": [intrinsic_value * 1.1, intrinsic_value * 1.25],
                            "color": "#FFA07A",
                        },  # Light salmon - caution zone
                        {
                            "range": [intrinsic_value * 1.25, max_value],
                            "color": "#FFB6C1",
                        },  # Light pink - overvalued zone
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": intrinsic_value,
                    },
                },
            )
        )

        # Update layout
        fig.update_layout(
            height=400,
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            paper_bgcolor="white",
            font={"color": "darkblue", "family": "Arial"},
        )

        # Add annotations for zones if enabled
        if show_zones:
            fig.add_annotation(
                text="Deep Value",
                xref="paper",
                yref="paper",
                x=0.15,
                y=0.1,
                showarrow=False,
                font=dict(size=10, color="green"),
            )
            fig.add_annotation(
                text="Fair Value",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.1,
                showarrow=False,
                font=dict(size=10, color="orange"),
            )
            fig.add_annotation(
                text="Overvalued",
                xref="paper",
                yref="paper",
                x=0.85,
                y=0.1,
                showarrow=False,
                font=dict(size=10, color="red"),
            )

        return fig

    except Exception as e:
        logger.error(f"Error creating enhanced price gauge: {str(e)}")
        # Return a simple gauge as fallback
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=current_price,
                title={"text": "Current Price vs Intrinsic Value"},
                gauge={
                    "axis": {"range": [0, max(current_price, intrinsic_value) * 1.2]},
                    "bar": {"color": "darkblue"},
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": intrinsic_value,
                    },
                },
            )
        )
        return fig


def create_technical_analysis_chart(
    price_data: pd.DataFrame,
    indicators: list[str] = ["SMA", "EMA", "BB", "MACD", "RSI"],
    chart_type: str = "candlestick",
) -> go.Figure:
    """Create an advanced technical analysis chart.

    Args:
        price_data: DataFrame with OHLC data
        indicators: List of indicators to display
        chart_type: Type of price chart ('candlestick' or 'line')

    Returns:
        Plotly figure object
    """
    try:
        # Create figure with subplots
        from plotly.subplots import make_subplots

        # Determine number of subplots needed
        subplot_count = 1  # Main price chart
        if "RSI" in indicators:
            subplot_count += 1
        if "MACD" in indicators:
            subplot_count += 1

        # Create subplot heights
        heights = [0.6] + [0.2] * (subplot_count - 1)

        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=heights,
            subplot_titles=["Price"]
            + (["RSI"] if "RSI" in indicators else [])
            + (["MACD"] if "MACD" in indicators else []),
        )

        # Prepare data
        df = price_data.copy()

        # Add main price chart
        if chart_type == "candlestick" and all(
            col in df.columns for col in ["Open", "High", "Low", "Close"]
        ):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Price",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name="Close Price",
                    line=dict(color="blue", width=2),
                ),
                row=1,
                col=1,
            )

        # Add Simple Moving Averages
        if "SMA" in indicators:
            for period in [20, 50, 200]:
                if len(df) >= period:
                    df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"SMA_{period}"],
                            mode="lines",
                            name=f"SMA {period}",
                            line=dict(width=1.5, dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )

        # Add Exponential Moving Averages
        if "EMA" in indicators:
            for period in [12, 26]:
                if len(df) >= period:
                    df[f"EMA_{period}"] = (
                        df["Close"].ewm(span=period, adjust=False).mean()
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[f"EMA_{period}"],
                            mode="lines",
                            name=f"EMA {period}",
                            line=dict(width=1.5),
                        ),
                        row=1,
                        col=1,
                    )

        # Add Bollinger Bands
        if "BB" in indicators and len(df) >= 20:
            df["BB_Middle"] = df["Close"].rolling(window=20).mean()
            df["BB_Std"] = df["Close"].rolling(window=20).std()
            df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
            df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)

            # Add upper band
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Upper"],
                    mode="lines",
                    name="BB Upper",
                    line=dict(color="gray", width=1),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add lower band with fill
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Lower"],
                    mode="lines",
                    name="Bollinger Bands",
                    line=dict(color="gray", width=1),
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.2)",
                ),
                row=1,
                col=1,
            )

        # Current row for additional indicators
        current_row = 2

        # Add RSI
        if "RSI" in indicators and len(df) >= 14:
            # Calculate RSI
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Plot RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["RSI"],
                    mode="lines",
                    name="RSI",
                    line=dict(color="purple", width=2),
                ),
                row=current_row,
                col=1,
            )

            # Add RSI levels
            fig.add_hline(
                y=70, line_dash="dash", line_color="red", row=current_row, col=1
            )
            fig.add_hline(
                y=30, line_dash="dash", line_color="green", row=current_row, col=1
            )

            current_row += 1

        # Add MACD
        if "MACD" in indicators and len(df) >= 26:
            # Calculate MACD
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["Histogram"] = df["MACD"] - df["Signal"]

            # Plot MACD line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD"],
                    mode="lines",
                    name="MACD",
                    line=dict(color="blue", width=2),
                ),
                row=current_row,
                col=1,
            )

            # Plot Signal line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Signal"],
                    mode="lines",
                    name="Signal",
                    line=dict(color="red", width=2),
                ),
                row=current_row,
                col=1,
            )

            # Plot Histogram
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Histogram"],
                    name="MACD Histogram",
                    marker_color="gray",
                ),
                row=current_row,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title="Technical Analysis",
            height=600 + (200 * (subplot_count - 1)),
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        # Update x-axis for the last subplot
        fig.update_xaxes(title_text="Date", row=subplot_count, col=1)

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if "RSI" in indicators:
            rsi_row = 2
            fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, range=[0, 100])
        if "MACD" in indicators:
            macd_row = subplot_count
            fig.update_yaxes(title_text="MACD", row=macd_row, col=1)

        return fig

    except Exception as e:
        logger.error(f"Error creating technical analysis chart: {str(e)}")
        # Return simple price chart as fallback
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data["Close"],
                mode="lines",
                name="Price",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_layout(
            title="Price History", xaxis_title="Date", yaxis_title="Price", height=400
        )
        return fig


def create_valuation_metrics_chart(metrics: dict[str, float]) -> go.Figure:
    """Create a radar chart for valuation metrics.

    Args:
        metrics: Dictionary of metric names and values (normalized 0-100)

    Returns:
        Plotly figure object
    """
    try:
        # Prepare data
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(0, 123, 255, 0.2)",
                line=dict(color="rgb(0, 123, 255)", width=2),
                name="Current Valuation",
            )
        )

        # Add reference line at 50 (neutral)
        fig.add_trace(
            go.Scatterpolar(
                r=[50] * len(categories),
                theta=categories,
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                name="Neutral",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            showlegend=True,
            title="Valuation Metrics Overview",
            height=400,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating valuation metrics chart: {str(e)}")
        # Return empty figure
        return go.Figure()


def create_price_gauge(current_price: float, intrinsic_value: float) -> go.Figure:
    """Create a gauge chart for price comparison.

    Args:
        current_price: Current stock price
        intrinsic_value: Calculated intrinsic value

    Returns:
        Plotly Figure object
    """
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
    """Create a growth chart with moving averages and Bollinger Bands.

    Args:
        price_data: DataFrame containing price data with 'Close' column

    Returns:
        Plotly Figure object
    """
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
