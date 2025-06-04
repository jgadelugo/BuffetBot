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
    Render an expandable section showing comprehensive scoring breakdown details.

    Args:
        score_details: Dictionary containing scoring weights for each indicator
        row_index: Row index for unique key generation
    """
    if not score_details:
        st.write("No scoring details available")
        return

    # Import here to avoid circular imports
    try:
        from buffetbot.dashboard.utils.enhanced_options_analysis import (
            get_scoring_indicator_names,
            get_total_scoring_indicators,
        )

        total_indicators = get_total_scoring_indicators()
        all_indicator_names = set(get_scoring_indicator_names())
    except ImportError:
        # Fallback to hardcoded values if import fails
        total_indicators = 5
        all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}

    # Separate actual scoring indicators from metadata
    actual_indicators = {
        k: v for k, v in score_details.items() if k in all_indicator_names
    }
    metadata_fields = {
        k: v for k, v in score_details.items() if k not in all_indicator_names
    }

    available_indicators = len(actual_indicators)

    # Create color-coded indicator count with correct total
    if available_indicators == total_indicators:
        indicator_badge = f"🟢 {available_indicators}/{total_indicators} indicators"
    elif available_indicators >= (total_indicators * 0.6):  # 60% or more
        indicator_badge = f"🟡 {available_indicators}/{total_indicators} indicators"
    else:
        indicator_badge = f"🔴 {available_indicators}/{total_indicators} indicators"

    st.markdown(f"**Data Availability:** {indicator_badge}")

    # Display weight breakdown for actual indicators
    if actual_indicators:
        st.markdown("**📊 Scoring Component Weights:**")

        # Enhanced indicator display with descriptions and values
        indicator_info = {
            "rsi": {
                "icon": "📈",
                "name": "RSI (Relative Strength Index)",
                "description": "Measures overbought/oversold conditions",
            },
            "beta": {
                "icon": "📊",
                "name": "Beta",
                "description": "Stock volatility relative to market",
            },
            "momentum": {
                "icon": "🚀",
                "name": "Momentum",
                "description": "Recent price movement trend",
            },
            "iv": {
                "icon": "💨",
                "name": "Implied Volatility",
                "description": "Market's expectation of future volatility",
            },
            "forecast": {
                "icon": "🔮",
                "name": "Analyst Forecast",
                "description": "Wall Street consensus confidence",
            },
        }

        for indicator, weight in actual_indicators.items():
            info = indicator_info.get(
                indicator,
                {
                    "icon": "📋",
                    "name": indicator.upper(),
                    "description": "Technical indicator",
                },
            )

            st.markdown(
                f"- {info['icon']} **{info['name']}**: {weight:.1%} "
                f"<small>({info['description']})</small>",
                unsafe_allow_html=True,
            )

    # Show missing indicators if any
    missing_indicators = all_indicator_names - set(actual_indicators.keys())
    if missing_indicators:
        st.markdown("**❌ Missing Data Sources:**")
        for indicator in sorted(missing_indicators):
            info = {
                "rsi": {"icon": "📈", "name": "RSI (Relative Strength Index)"},
                "beta": {"icon": "📊", "name": "Beta"},
                "momentum": {"icon": "🚀", "name": "Momentum"},
                "iv": {"icon": "💨", "name": "Implied Volatility"},
                "forecast": {"icon": "🔮", "name": "Analyst Forecast"},
            }.get(indicator, {"icon": "📋", "name": indicator.upper()})

            st.markdown(f"- {info['icon']} {info['name']}")

    # Display metadata fields if present
    if metadata_fields:
        st.markdown("**⚙️ Analysis Configuration:**")
        for field, value in metadata_fields.items():
            if field == "risk_tolerance":
                st.markdown(f"- 🎯 **Risk Tolerance**: {value}")
            else:
                # Format field name nicely
                field_name = field.replace("_", " ").title()
                st.markdown(f"- 📋 **{field_name}**: {value}")

    # Add data quality assessment
    data_quality = (
        "Excellent"
        if available_indicators == total_indicators
        else "Good"
        if available_indicators >= (total_indicators * 0.8)
        else "Moderate"
        if available_indicators >= (total_indicators * 0.6)
        else "Limited"
    )

    quality_color = {"Excellent": "🟢", "Good": "🟡", "Moderate": "🟠", "Limited": "🔴"}
    st.markdown(
        f"**📋 Data Quality**: {quality_color.get(data_quality, '⚪')} {data_quality}"
    )


def get_data_score_badge(score_details: dict[str, Any]) -> str:
    """
    Generate a data score badge showing how many indicators were used.

    Now dynamically calculates the total based on SCORING_WEIGHTS instead of hardcoding.

    Args:
        score_details: Dictionary containing scoring weights

    Returns:
        str: Formatted badge string (e.g., "🟢 5/5" or "🟡 4/5")
    """
    # Handle invalid input types
    if not isinstance(score_details, dict) or not score_details:
        # Import here to avoid circular imports
        try:
            from buffetbot.dashboard.utils.enhanced_options_analysis import (
                get_total_scoring_indicators,
            )

            total_indicators = get_total_scoring_indicators()
        except ImportError:
            total_indicators = 5  # Fallback
        return f"❓ 0/{total_indicators}"

    # Import here to avoid circular imports
    try:
        from buffetbot.dashboard.utils.enhanced_options_analysis import (
            get_scoring_indicator_names,
            get_total_scoring_indicators,
        )

        total_indicators = get_total_scoring_indicators()
        all_indicator_names = set(get_scoring_indicator_names())
    except ImportError:
        # Fallback to hardcoded values if import fails
        total_indicators = 5
        all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}

    # Count only actual scoring indicators (exclude metadata like risk_tolerance)
    actual_indicators = {
        k: v for k, v in score_details.items() if k in all_indicator_names
    }
    available_indicators = len(actual_indicators)

    # Generate badge with dynamic total
    if available_indicators == total_indicators:
        return f"🟢 {available_indicators}/{total_indicators}"
    elif available_indicators >= (total_indicators * 0.6):  # 60% or more
        return f"🟡 {available_indicators}/{total_indicators}"
    else:
        return f"🔴 {available_indicators}/{total_indicators}"


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
