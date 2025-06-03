"""Enhanced metrics display components for the dashboard."""

from typing import Any, Dict, Optional, Union

import pandas as pd
import streamlit as st

from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from glossary import get_metric_info


def display_metric_with_info(
    label: str,
    value: str,
    delta: str | None = None,
    metric_key: str | None = None,
    help_text: str | None = None,
) -> None:
    """Display a metric with optional tooltip and definition.

    Args:
        label: The metric label
        value: The formatted metric value
        delta: Optional delta value for change indication
        metric_key: Optional key for glossary lookup
        help_text: Optional help text (overrides glossary lookup)
    """
    # Get help text from glossary if available and not provided
    if help_text is None and metric_key:
        try:
            metric_info = get_metric_info(metric_key)
            help_text = (
                f"{metric_info['description']} | Formula: {metric_info['formula']}"
            )
        except KeyError:
            help_text = None

    # Display the metric with help text if available
    if help_text:
        st.metric(label=label, value=value, delta=delta, help=help_text)
    else:
        st.metric(label=label, value=value, delta=delta)


def display_table_with_info(
    df: pd.DataFrame, metric_keys: dict[str, str] | None = None
) -> None:
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


def display_metrics_grid(metrics: dict[str, dict[str, Any]], cols: int = 3) -> None:
    """Display metrics in a grid layout.

    Args:
        metrics: Dictionary of metrics with their config
        cols: Number of columns in the grid
    """
    if not metrics:
        st.info("No metrics available to display")
        return

    # Create columns
    columns = st.columns(cols)

    # Display metrics
    for i, (key, config) in enumerate(metrics.items()):
        with columns[i % cols]:
            display_metric_with_info(
                label=key,
                value=config.get("value", "N/A"),
                delta=config.get("delta"),
                metric_key=config.get("metric_key"),
                help_text=config.get("help_text"),
            )


def display_metric_with_status(
    label: str,
    value: str,
    status: str,
    help_text: str = None,
    metric_type: str = "metric",
) -> None:
    """Display a metric with status indicator.

    Args:
        label: The metric label
        value: The formatted metric value
        status: Status level (good, warning, bad)
        help_text: Optional help text
        metric_type: Type of metric for styling
    """
    # Define status colors and icons
    status_config = {
        "good": {"color": "#28a745", "icon": "✅"},
        "warning": {"color": "#ffc107", "icon": "⚠️"},
        "bad": {"color": "#dc3545", "icon": "❌"},
        "neutral": {"color": "#6c757d", "icon": "ℹ️"},
    }

    config = status_config.get(status, status_config["neutral"])

    # Create styled metric display
    st.markdown(
        f"""
        <div style="background-color: white; padding: 1rem; border-radius: 0.5rem;
                    border-left: 4px solid {config['color']}; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #333; font-size: 1.1rem;">{label}</h3>
                    <p style="margin: 0; font-size: 1.8rem; font-weight: bold; color: {config['color']};">
                        {value}
                    </p>
                    {f'<small style="color: #666;">{help_text}</small>' if help_text else ''}
                </div>
                <div style="font-size: 2rem;">
                    {config['icon']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metrics_grid_enhanced(
    metrics: dict[str, dict[str, Any]], title: str = None, cols: int = 2
) -> None:
    """Display metrics in an enhanced grid with status indicators.

    Args:
        metrics: Dictionary of metrics with enhanced config
        title: Optional section title
        cols: Number of columns
    """
    if title:
        st.subheader(title)

    if not metrics:
        st.info("No metrics available to display")
        return

    # Create columns
    columns = st.columns(cols)

    # Display metrics with status
    for i, (key, config) in enumerate(metrics.items()):
        with columns[i % cols]:
            display_metric_with_status(
                label=config.get("label", key),
                value=config.get("value", "N/A"),
                status=config.get("status", "neutral"),
                help_text=config.get("help_text"),
                metric_type=config.get("type", "metric"),
            )


def create_comparison_table(
    data: dict[str, Any],
    title: str = "Comparison Table",
    format_functions: dict[str, callable] = None,
) -> None:
    """Create a comparison table with formatted values.

    Args:
        data: Dictionary of data to display
        title: Table title
        format_functions: Optional formatting functions for specific columns
    """
    if not data:
        st.info(f"No data available for {title}")
        return

    st.subheader(title)

    # Convert to DataFrame
    df_data = []
    for key, value in data.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            for sub_key, sub_value in value.items():
                formatted_value = sub_value
                if format_functions and f"{key}_{sub_key}" in format_functions:
                    formatted_value = format_functions[f"{key}_{sub_key}"](sub_value)
                elif isinstance(sub_value, float):
                    formatted_value = safe_format_number(sub_value)

                df_data.append(
                    {"Category": key, "Metric": sub_key, "Value": formatted_value}
                )
        else:
            # Handle direct values
            formatted_value = value
            if format_functions and key in format_functions:
                formatted_value = format_functions[key](value)
            elif isinstance(value, float):
                formatted_value = safe_format_number(value)

            df_data.append({"Metric": key, "Value": formatted_value})

    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def create_progress_indicator(
    value: float,
    min_value: float = 0,
    max_value: float = 100,
    label: str = "Progress",
    show_percentage: bool = True,
) -> None:
    """Create a progress indicator with value display.

    Args:
        value: Current value
        min_value: Minimum value for scale
        max_value: Maximum value for scale
        label: Progress label
        show_percentage: Whether to show percentage
    """
    # Calculate percentage
    if max_value == min_value:
        percentage = 0
    else:
        percentage = max(
            0, min(100, ((value - min_value) / (max_value - min_value)) * 100)
        )

    # Create progress bar
    st.markdown(f"**{label}**")

    progress_html = f"""
    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 5px 0;">
        <div style="background-color: #1f77b4; height: 20px; border-radius: 7px;
                    width: {percentage}%; transition: width 0.3s ease;">
        </div>
    </div>
    """

    st.markdown(progress_html, unsafe_allow_html=True)

    if show_percentage:
        st.caption(f"{safe_format_number(value)} ({percentage:.1f}%)")
    else:
        st.caption(f"{safe_format_number(value)}")
