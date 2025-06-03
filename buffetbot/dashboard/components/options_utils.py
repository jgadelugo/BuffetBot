"""Options advisor utility functions for the dashboard."""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_number,
    safe_format_percentage,
)


def render_score_details_popover(score_details: dict[str, Any], row_index: int) -> None:
    """
    Render an expandable section showing scoring breakdown details.

    Args:
        score_details: Dictionary containing scoring weights for each indicator
        row_index: Row index for unique key generation
    """
    if not score_details:
        st.write("No scoring details available")
        return

    total_indicators = 5  # RSI, Beta, Momentum, IV, Forecast
    available_indicators = len(score_details)

    # Create color-coded indicator count
    if available_indicators == total_indicators:
        indicator_badge = f"🟢 {available_indicators}/{total_indicators} indicators"
    elif available_indicators >= 3:
        indicator_badge = f"🟡 {available_indicators}/{total_indicators} indicators"
    else:
        indicator_badge = f"🔴 {available_indicators}/{total_indicators} indicators"

    st.markdown(f"**Data Score:** {indicator_badge}")

    # Display weight breakdown
    st.markdown("**Scoring Weights:**")
    for indicator, weight in score_details.items():
        indicator_display = {
            "rsi": "📈 RSI",
            "beta": "📊 Beta",
            "momentum": "🚀 Momentum",
            "iv": "💨 Implied Volatility",
            "forecast": "🔮 Analyst Forecast",
        }.get(indicator, indicator.upper())

        st.markdown(f"- {indicator_display}: {weight:.1%}")

    # Show missing indicators if any
    all_indicators = {"rsi", "beta", "momentum", "iv", "forecast"}
    missing_indicators = all_indicators - set(score_details.keys())

    if missing_indicators:
        st.markdown("**Missing Data:**")
        for indicator in missing_indicators:
            indicator_display = {
                "rsi": "📈 RSI",
                "beta": "📊 Beta",
                "momentum": "🚀 Momentum",
                "iv": "💨 Implied Volatility",
                "forecast": "🔮 Analyst Forecast",
            }.get(indicator, indicator.upper())
            st.markdown(f"- {indicator_display}")


def get_data_score_badge(score_details: dict[str, Any]) -> str:
    """
    Generate a data score badge showing how many indicators were used.

    Args:
        score_details: Dictionary containing scoring weights

    Returns:
        str: Formatted badge string
    """
    # Handle invalid input types
    if not isinstance(score_details, dict) or not score_details:
        return "❓ 0/5"

    total_indicators = 5
    available_indicators = len(score_details)

    if available_indicators == total_indicators:
        return f"🟢 {available_indicators}/5"
    elif available_indicators >= 3:
        return f"🟡 {available_indicators}/5"
    else:
        return f"🔴 {available_indicators}/5"


def check_for_partial_data(recommendations: pd.DataFrame) -> bool:
    """
    Check if any recommendations are based on partial data.

    Args:
        recommendations: DataFrame containing options recommendations

    Returns:
        bool: True if any recommendations use fewer than 5 indicators
    """
    if recommendations.empty or "score_details" not in recommendations.columns:
        return False

    for _, row in recommendations.iterrows():
        score_details = row.get("score_details", {})
        if isinstance(score_details, dict) and len(score_details) < 5:
            return True

    return False


def create_styling_functions():
    """Create styling functions for the options recommendations table."""

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
                return (
                    "background-color: #fff3e0; color: #f57c00"  # Orange for medium IV
                )
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

    return highlight_rsi, highlight_score, highlight_iv, highlight_forecast
