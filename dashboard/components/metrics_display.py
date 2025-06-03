"""Enhanced metrics display components with visual status indicators."""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def format_metric_value(value: any, metric_type: str = None) -> str:
    """Format metric value with appropriate decimal places.

    Args:
        value: The value to format
        metric_type: Optional type of metric to determine formatting

    Returns:
        Formatted string value
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            # Format based on metric type
            if metric_type == "currency":
                return f"${value:,.2f}"
            elif metric_type == "percentage":
                return f"{value:.1%}"
            elif metric_type == "ratio":
                return f"{value:.2f}"
            elif metric_type == "score":
                return f"{value:.1f}"
            else:
                # Default formatting
                if abs(value) >= 1000:
                    return f"{value:,.0f}"
                elif abs(value) >= 1:
                    return f"{value:.2f}"
                else:
                    return f"{value:.2f}"
        else:
            return str(value)

    except Exception as e:
        logger.error(f"Error formatting metric value: {str(e)}")
        return str(value)


def display_metrics_grid_enhanced(
    metrics_dict: dict[str, dict], cols: int = 3, show_status_colors: bool = True
) -> None:
    """Display metrics in an enhanced grid layout with status indicators.

    Args:
        metrics_dict: Dictionary of metrics with structure:
            {
                'metric_name': {
                    'value': 'displayed value',
                    'metric_key': 'glossary key',
                    'delta': 'optional delta value',
                    'status': 'good/warning/bad/neutral',
                    'help_text': 'optional custom help text',
                    'type': 'optional metric type (currency/percentage/ratio/score)'
                }
            }
        cols: Number of columns in the grid
        show_status_colors: Whether to show color-coded status
    """
    try:
        # Create columns
        columns = st.columns(cols)

        # Display metrics
        for idx, (metric_name, metric_data) in enumerate(metrics_dict.items()):
            col_idx = idx % cols

            with columns[col_idx]:
                # Get status and determine emoji
                status = metric_data.get("status", "neutral")
                status_emoji = {
                    "good": "✅",
                    "warning": "⚠️",
                    "bad": "❌",
                    "neutral": "📊",
                }.get(status, "📊")

                # Format label with emoji if showing status colors
                label = (
                    f"{status_emoji} {metric_name}"
                    if show_status_colors
                    else metric_name
                )

                # Format value
                value = metric_data["value"]
                metric_type = metric_data.get("type")
                formatted_value = format_metric_value(value, metric_type)

                # Format delta if present
                delta = metric_data.get("delta")
                if delta is not None:
                    delta = format_metric_value(delta, "percentage")

                # Use standard streamlit metric
                help_text = metric_data.get("help_text")
                st.metric(
                    label=label, value=formatted_value, delta=delta, help=help_text
                )

    except Exception as e:
        logger.error(f"Error displaying metrics grid: {str(e)}")
        st.error("Error displaying metrics")


def display_metric_with_status(
    label: str,
    value: any,
    status: str = "neutral",
    delta: any | None = None,
    help_text: str | None = None,
    show_trend: bool = True,
    metric_type: str = None,
) -> None:
    """Display a single metric with visual status indicator.

    Args:
        label: Metric label
        value: Metric value
        status: Status indicator (good/warning/bad/neutral)
        delta: Optional delta value
        help_text: Optional help text
        show_trend: Whether to show trend arrow
        metric_type: Optional type of metric (currency/percentage/ratio/score)
    """
    try:
        # Status configurations
        status_config = {
            "good": {"icon": "✅", "delta_color": "normal"},
            "warning": {"icon": "⚠️", "delta_color": "normal"},
            "bad": {"icon": "❌", "delta_color": "inverse"},
            "neutral": {"icon": "📊", "delta_color": "off"},
        }

        config = status_config.get(status, status_config["neutral"])

        # Format label with status icon
        formatted_label = f"{config['icon']} {label}"

        # Format values
        formatted_value = format_metric_value(value, metric_type)
        formatted_delta = (
            format_metric_value(delta, "percentage") if delta is not None else None
        )

        # Display using Streamlit metric
        st.metric(
            label=formatted_label,
            value=formatted_value,
            delta=formatted_delta,
            delta_color=config["delta_color"] if delta else "off",
            help=help_text,
        )

    except Exception as e:
        logger.error(f"Error displaying metric with status: {str(e)}")
        # Fallback to standard metric
        st.metric(
            label=label,
            value=str(value),
            delta=str(delta) if delta else None,
            help=help_text,
        )


def create_comparison_table(
    metrics: list[tuple[str, any, any, str]],
    headers: list[str] = ["Metric", "Actual", "Expected", "Status"],
) -> None:
    """Create a comparison table with color-coded status.

    Args:
        metrics: List of tuples (metric_name, actual_value, expected_value, status)
        headers: Table headers
    """
    try:
        # Create data for DataFrame
        data = []
        for metric_name, actual, expected, status in metrics:
            # Status icons
            status_icons = {
                "good": "✅ Good",
                "warning": "⚠️ Warning",
                "bad": "❌ Bad",
                "neutral": "➖ Neutral",
            }
            status_display = status_icons.get(status, status)

            # Format values
            formatted_actual = format_metric_value(actual)
            formatted_expected = format_metric_value(expected)

            data.append(
                {
                    headers[0]: metric_name,
                    headers[1]: formatted_actual,
                    headers[2]: formatted_expected,
                    headers[3]: status_display,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(data)

        # Display with custom styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                headers[0]: st.column_config.TextColumn(headers[0], width="medium"),
                headers[1]: st.column_config.TextColumn(headers[1], width="small"),
                headers[2]: st.column_config.TextColumn(headers[2], width="small"),
                headers[3]: st.column_config.TextColumn(headers[3], width="medium"),
            },
        )

    except Exception as e:
        logger.error(f"Error creating comparison table: {str(e)}")
        # Fallback to simple dataframe
        df_data = {
            headers[0]: [m[0] for m in metrics],
            headers[1]: [format_metric_value(m[1]) for m in metrics],
            headers[2]: [format_metric_value(m[2]) for m in metrics],
            headers[3]: [m[3] for m in metrics],
        }
        st.dataframe(pd.DataFrame(df_data), hide_index=True)


def create_progress_indicator(
    label: str,
    value: float,
    max_value: float = 100,
    status: str = "neutral",
    show_percentage: bool = True,
) -> None:
    """Create a progress bar with status coloring.

    Args:
        label: Progress bar label
        value: Current value
        max_value: Maximum value
        status: Status for color coding
        show_percentage: Whether to show percentage
    """
    try:
        # Calculate percentage
        percentage = (value / max_value * 100) if max_value > 0 else 0
        percentage = min(max(percentage, 0), 100)  # Clamp between 0 and 100

        # Status emojis
        status_emojis = {"good": "✅", "warning": "⚠️", "bad": "❌", "neutral": "📊"}
        emoji = status_emojis.get(status, "📊")

        # Display label with percentage
        if show_percentage:
            st.write(f"**{emoji} {label}** - {percentage:.1f}%")
        else:
            st.write(f"**{emoji} {label}**")

        # Display progress bar
        st.progress(percentage / 100)

    except Exception as e:
        logger.error(f"Error creating progress indicator: {str(e)}")
        st.progress(percentage / 100)
