"""
Advanced settings component for options analysis.

This module provides UI components for managing options analysis settings,
including custom scoring weights, risk profiles, and analysis parameters.
"""

from typing import Any, Dict

import streamlit as st

from buffetbot.analysis.options.config.scoring_weights import get_scoring_weights
from buffetbot.analysis.options.core.domain_models import StrategyType
from buffetbot.dashboard.config.settings import (
    clear_analysis_cache,
    get_options_setting,
    settings_have_changed,
    update_options_setting,
    validate_scoring_weights,
)
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def render_advanced_settings_panel() -> bool:
    """
    Render the advanced settings panel for options analysis.

    Returns:
        bool: True if settings were changed, False otherwise
    """
    with st.expander("⚙️ Advanced Settings & Custom Scoring", expanded=False):
        st.markdown(
            """
        **Customize Analysis Parameters**

        Fine-tune the options analysis by adjusting scoring weights and risk parameters.
        Changes will take effect when you click "Analyze Options" again.
        """
        )

        # Settings change indicator
        if settings_have_changed():
            st.info(
                "🔄 Settings have changed. Click 'Analyze Options' to apply the new configuration."
            )

        # Custom scoring weights section
        render_custom_scoring_weights()

        # Risk profile adjustments
        render_risk_profile_settings()

        # Analysis behavior settings
        render_analysis_behavior_settings()

        # Settings reset
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🔄 Reset to Defaults", help="Reset all settings to default values"
            ):
                reset_to_defaults()
                st.rerun()

        with col2:
            if st.button(
                "🗑️ Clear Analysis Cache", help="Force recalculation on next analysis"
            ):
                clear_analysis_cache()
                st.success("Cache cleared!")

        return settings_have_changed()


def render_custom_scoring_weights() -> None:
    """Render the custom scoring weights configuration UI."""
    st.markdown("#### 📊 Custom Scoring Weights")

    use_custom = st.checkbox(
        "Use Custom Scoring Weights",
        value=get_options_setting("use_custom_weights", False),
        help="Override default strategy-specific weights with custom values",
    )
    update_options_setting("use_custom_weights", use_custom)

    if use_custom:
        st.markdown(
            """
        **Adjust the importance of each factor in option scoring:**
        - **RSI**: Relative Strength Index (momentum)
        - **Beta**: Stock correlation with market
        - **Momentum**: Price trend strength
        - **IV**: Implied Volatility level
        - **Forecast**: Analyst forecast confidence
        """
        )

        # Get current custom weights
        current_weights = get_options_setting(
            "custom_scoring_weights",
            {"rsi": 0.20, "beta": 0.20, "momentum": 0.20, "iv": 0.20, "forecast": 0.20},
        )

        # Create weight sliders
        col1, col2 = st.columns(2)

        with col1:
            rsi_weight = st.slider(
                "📈 RSI Weight",
                0.0,
                1.0,
                current_weights.get("rsi", 0.20),
                0.05,
                help="Weight for Relative Strength Index (momentum indicator)",
            )

            beta_weight = st.slider(
                "📊 Beta Weight",
                0.0,
                1.0,
                current_weights.get("beta", 0.20),
                0.05,
                help="Weight for market correlation coefficient",
            )

            momentum_weight = st.slider(
                "🚀 Momentum Weight",
                0.0,
                1.0,
                current_weights.get("momentum", 0.20),
                0.05,
                help="Weight for price trend strength",
            )

        with col2:
            iv_weight = st.slider(
                "💨 IV Weight",
                0.0,
                1.0,
                current_weights.get("iv", 0.20),
                0.05,
                help="Weight for Implied Volatility level",
            )

            forecast_weight = st.slider(
                "🔮 Forecast Weight",
                0.0,
                1.0,
                current_weights.get("forecast", 0.20),
                0.05,
                help="Weight for analyst forecast confidence",
            )

        # Build new weights dict
        new_weights = {
            "rsi": rsi_weight,
            "beta": beta_weight,
            "momentum": momentum_weight,
            "iv": iv_weight,
            "forecast": forecast_weight,
        }

        # Validate weights
        is_valid, error_msg = validate_scoring_weights(new_weights)

        if not is_valid:
            st.error(f"❌ {error_msg}")

            # Auto-normalize button
            if st.button("🔧 Auto-Normalize Weights"):
                total = sum(new_weights.values())
                if total > 0:
                    normalized_weights = {k: v / total for k, v in new_weights.items()}
                    update_options_setting("custom_scoring_weights", normalized_weights)
                    st.rerun()
        else:
            # Show current total
            total = sum(new_weights.values())
            st.success(f"✅ Weights sum to {total:.3f}")

            # Update settings if weights changed
            if new_weights != current_weights:
                update_options_setting("custom_scoring_weights", new_weights)

    else:
        # Show strategy-specific default weights
        strategy_type = get_options_setting("strategy_type", "Long Calls")
        try:
            strategy_enum = StrategyType(strategy_type)
            default_weights = get_scoring_weights(strategy_enum)

            st.info(
                f"""
            **Using default weights for {strategy_type}:**
            - RSI: {default_weights.rsi:.0%}
            - Beta: {default_weights.beta:.0%}
            - Momentum: {default_weights.momentum:.0%}
            - IV: {default_weights.iv:.0%}
            - Forecast: {default_weights.forecast:.0%}
            """
            )
        except Exception as e:
            logger.error(f"Error getting default weights: {e}")


def render_risk_profile_settings() -> None:
    """Render risk profile adjustment settings."""
    st.markdown("#### ⚡ Risk Profile Adjustments")

    col1, col2 = st.columns(2)

    with col1:
        delta_threshold = st.slider(
            "Max Delta Threshold",
            0.0,
            1.0,
            get_options_setting("delta_threshold", 0.7),
            0.05,
            help="Maximum delta value for option recommendations",
        )
        update_options_setting("delta_threshold", delta_threshold)

        volume_threshold = st.number_input(
            "Min Volume Threshold",
            min_value=0,
            value=get_options_setting("volume_threshold", 50),
            help="Minimum daily volume for option recommendations",
        )
        update_options_setting("volume_threshold", volume_threshold)

    with col2:
        bid_ask_spread = st.slider(
            "Max Bid-Ask Spread",
            0.0,
            2.0,
            get_options_setting("bid_ask_spread", 0.5),
            0.05,
            help="Maximum bid-ask spread (in dollars) for recommendations",
        )
        update_options_setting("bid_ask_spread", bid_ask_spread)

        open_interest = st.number_input(
            "Min Open Interest",
            min_value=0,
            value=get_options_setting("open_interest", 100),
            help="Minimum open interest for option recommendations",
        )
        update_options_setting("open_interest", open_interest)


def render_analysis_behavior_settings() -> None:
    """Render analysis behavior and performance settings."""
    st.markdown("#### 🔧 Analysis Behavior")

    col1, col2 = st.columns(2)

    with col1:
        enable_caching = st.checkbox(
            "Enable Result Caching",
            value=get_options_setting("enable_caching", True),
            help="Cache analysis results to improve performance",
        )
        update_options_setting("enable_caching", enable_caching)

        auto_refresh = st.checkbox(
            "Auto-refresh on Setting Change",
            value=get_options_setting("auto_refresh", False),
            help="Automatically re-run analysis when settings change",
        )
        update_options_setting("auto_refresh", auto_refresh)

    with col2:
        detailed_logging = st.checkbox(
            "Detailed Logging",
            value=get_options_setting("detailed_logging", False),
            help="Enable detailed logging for troubleshooting",
        )
        update_options_setting("detailed_logging", detailed_logging)

        parallel_processing = st.checkbox(
            "Parallel Processing",
            value=get_options_setting("parallel_processing", True),
            help="Use parallel processing for faster analysis",
        )
        update_options_setting("parallel_processing", parallel_processing)


def render_settings_impact_documentation() -> None:
    """Render documentation explaining how settings impact metrics."""
    with st.expander("📚 How Settings Impact Your Analysis", expanded=False):
        st.markdown(
            """
        ### Understanding Options Analysis Settings

        #### 📊 **Scoring Weights Impact**
        Each technical indicator contributes to the final option score:

        - **RSI (Relative Strength Index)**: Higher weight = prioritize momentum trades
          - *High RSI weight*: Favors options on stocks with strong price momentum
          - *Low RSI weight*: Less emphasis on short-term price movements

        - **Beta (Market Correlation)**: Higher weight = factor in market sensitivity
          - *High Beta weight*: Considers how much the stock moves with the market
          - *Low Beta weight*: Focuses more on stock-specific factors

        - **Momentum**: Higher weight = prioritize trending stocks
          - *High Momentum weight*: Favors stocks with sustained price trends
          - *Low Momentum weight*: Less emphasis on recent price direction

        - **IV (Implied Volatility)**: Higher weight = factor in option pricing
          - *High IV weight*: Considers option expensiveness/cheapness more heavily
          - *Low IV weight*: Focuses less on volatility levels

        - **Forecast**: Higher weight = trust analyst predictions more
          - *High Forecast weight*: Prioritizes options on stocks with strong analyst backing
          - *Low Forecast weight*: Less reliance on Wall Street opinions

        #### ⚡ **Risk Tolerance Impact**

        - **Conservative**: Stricter filtering, longer expiry times, higher volume requirements
        - **Moderate**: Balanced approach with moderate risk parameters
        - **Aggressive**: Looser filtering, shorter expiry allowed, lower volume requirements

        #### 📅 **Time Horizon Impact**

        - **Short-term**: Emphasizes momentum and technical factors
        - **Medium-term**: Balanced approach across all factors
        - **Long-term**: Emphasizes fundamentals and analyst forecasts

        #### 🎯 **Strategy-Specific Behavior**

        - **Long Calls**: Emphasizes momentum and forecast confidence
        - **Bull Call Spreads**: Balanced scoring with moderate risk
        - **Covered Calls**: Higher IV weight for income optimization
        - **Cash-Secured Puts**: Emphasizes RSI and IV for entry timing

        > **💡 Pro Tip**: Start with default settings and gradually adjust based on your trading style and market outlook.
        """
        )


def reset_to_defaults() -> None:
    """Reset all options settings to their default values."""
    defaults = {
        "strategy_type": "Long Calls",
        "risk_tolerance": "Conservative",
        "time_horizon": "Medium-term (3-6 months)",
        "min_days": 180,
        "top_n": 5,
        "include_greeks": True,
        "volatility_analysis": False,
        "download_csv": False,
        "use_custom_weights": False,
        "custom_scoring_weights": {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        },
        "delta_threshold": 0.7,
        "volume_threshold": 50,
        "bid_ask_spread": 0.5,
        "open_interest": 100,
        "enable_caching": True,
        "auto_refresh": False,
        "detailed_logging": False,
        "parallel_processing": True,
    }

    for key, value in defaults.items():
        update_options_setting(key, value)

    clear_analysis_cache()
    logger.info("Options settings reset to defaults")
    st.success("🔄 Settings reset to defaults!")


def get_analysis_settings() -> dict[str, Any]:
    """
    Get the current analysis settings formatted for use in options analysis.

    Returns:
        Dict containing all current analysis settings
    """
    return {
        "strategy_type": get_options_setting("strategy_type", "Long Calls"),
        "risk_tolerance": get_options_setting("risk_tolerance", "Conservative"),
        "time_horizon": get_options_setting("time_horizon", "Medium-term (3-6 months)"),
        "min_days": get_options_setting("min_days", 180),
        "top_n": get_options_setting("top_n", 5),
        "use_custom_weights": get_options_setting("use_custom_weights", False),
        "custom_scoring_weights": get_options_setting("custom_scoring_weights", {}),
        "delta_threshold": get_options_setting("delta_threshold", 0.7),
        "volume_threshold": get_options_setting("volume_threshold", 50),
        "bid_ask_spread": get_options_setting("bid_ask_spread", 0.5),
        "open_interest": get_options_setting("open_interest", 100),
    }
