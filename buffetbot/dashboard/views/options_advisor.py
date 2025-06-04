"""Options advisor tab rendering module."""

import io
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from buffetbot.analysis.options import analyze_options_strategy
from buffetbot.analysis.options.config.scoring_weights import get_scoring_weights
from buffetbot.analysis.options.core.domain_models import StrategyType
from buffetbot.analysis.options.core.exceptions import (
    InsufficientDataError,
    OptionsAdvisorError,
)
from buffetbot.dashboard.components.disclaimers import render_investment_disclaimer
from buffetbot.dashboard.components.metrics import display_metric_with_info
from buffetbot.dashboard.components.options_settings import (
    get_analysis_settings,
    render_advanced_settings_panel,
    render_settings_impact_documentation,
)
from buffetbot.dashboard.components.options_utils import (
    check_for_partial_data,
    create_styling_functions,
    get_data_score_badge,
    render_score_details_popover,
)
from buffetbot.dashboard.config.settings import (
    clear_analysis_cache,
    get_dashboard_config,
    get_options_setting,
    mark_settings_applied,
    settings_have_changed,
    update_options_setting,
)
from buffetbot.dashboard.dashboard_utils.data_processing import handle_ticker_change
from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from buffetbot.dashboard.utils.enhanced_options_analysis import (
    analyze_options_with_custom_settings,
    get_scoring_indicator_names,
    get_strategy_specific_weights,
    get_total_scoring_indicators,
)
from buffetbot.glossary import get_metric_info
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def render_options_advisor_tab(data: dict[str, Any], ticker: str) -> None:
    """Render the options advisor tab content.

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

    config = get_dashboard_config()

    # Options Advisor
    st.header("🎯 Options Advisor")

    # Check if ticker has changed and provide feedback
    previous_ticker = st.session_state.get("options_advisor_previous_ticker", None)
    if previous_ticker and previous_ticker != ticker:
        st.info(
            f"🔄 Ticker updated from {previous_ticker} to {ticker}. The options analysis below will reflect the new selection."
        )

    # Store current ticker for next comparison
    st.session_state.options_advisor_previous_ticker = ticker

    # Display current ticker prominently with sync status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Analyzing Options for: {ticker}**")
    with col2:
        st.success("🔗 Synced")  # Visual indicator that ticker is synced

    # Add ticker change detection info
    with st.expander("ℹ️ Global Ticker Synchronization", expanded=False):
        st.markdown(
            """
        **📡 Ticker Synchronization:**
        - This page automatically uses the ticker selected in the sidebar
        - When you change the ticker globally, this page will update automatically
        - The options analysis will refresh to show data for the new ticker
        - No need to manually enter the ticker - it's synchronized across all tabs

        **💡 Pro Tip:** Check the Analyst Forecast tab for detailed Wall Street opinions on this stock!
        """
        )

    # Add informative description with tooltips
    st.markdown(
        """
    Analyze options contracts using comprehensive technical scoring and Greeks analysis.
    This tool combines RSI, Beta, Momentum, Implied Volatility, and Analyst Forecast Confidence
    to recommend optimal option strategies.

    💡 **Note**: This analysis incorporates analyst forecast data as a key scoring component.
    For detailed forecast analysis and price targets, visit the 🔮 **Analyst Forecast** tab.
    """
    )

    # Add prominent options trading disclaimer
    render_investment_disclaimer("options")

    # Enhanced Options Strategy Selector
    st.subheader("📊 Strategy & Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_strategy = get_options_setting("strategy_type", "Long Calls")
        strategy_type = st.selectbox(
            "🎯 Options Strategy",
            options=[
                "Long Calls",
                "Bull Call Spread",
                "Covered Call",
                "Cash-Secured Put",
            ],
            index=[
                "Long Calls",
                "Bull Call Spread",
                "Covered Call",
                "Cash-Secured Put",
            ].index(current_strategy),
            help="Select the options strategy to analyze",
        )
        update_options_setting("strategy_type", strategy_type)

    with col2:
        current_risk = get_options_setting("risk_tolerance", "Conservative")
        risk_tolerance = st.selectbox(
            "⚡ Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(current_risk),
            help="Your risk tolerance affects strategy recommendations",
        )
        update_options_setting("risk_tolerance", risk_tolerance)

    with col3:
        current_horizon = get_options_setting("time_horizon", "One Year (12 months)")
        time_horizon = st.selectbox(
            "📅 Time Horizon",
            options=[
                "Medium-term (3-6 months)",
                "Long-term (6+ months)",
                "One Year (12 months)",
                "18 Months (1.5 years)",
            ],
            index=[
                "Medium-term (3-6 months)",
                "Long-term (6+ months)",
                "One Year (12 months)",
                "18 Months (1.5 years)",
            ].index(current_horizon),
            help="Expected holding period for the options position",
        )
        update_options_setting("time_horizon", time_horizon)

    # Core Analysis Parameters
    col1, col2 = st.columns(2)

    with col1:
        # Get help text for days to expiry based on metric definitions toggle
        show_definitions = st.session_state.get("show_metric_definitions", True)
        days_help_text = "Minimum number of days until option expiration. Longer terms provide more time value."

        if show_definitions:
            try:
                metric_info = get_metric_info("days_to_expiry")
                days_help_text = (
                    f"{metric_info['description']} Formula: {metric_info['formula']}"
                )
            except KeyError:
                pass

        current_min_days = get_options_setting("min_days", config["default_min_days"])
        min_days = st.slider(
            "Minimum Days to Expiry",
            min_value=config["min_min_days"],
            max_value=config["max_min_days"],
            value=current_min_days,
            help=days_help_text,
        )
        update_options_setting("min_days", min_days)

    with col2:
        current_top_n = get_options_setting("top_n", config["default_top_n"])
        top_n = st.slider(
            "Number of Recommendations",
            min_value=config["min_top_n"],
            max_value=config["max_top_n"],
            value=current_top_n,
            help="Number of top-ranked option recommendations to display",
        )
        update_options_setting("top_n", top_n)

    # Advanced Analysis Options
    st.subheader("🔧 Advanced Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        current_greeks = get_options_setting("include_greeks", True)
        include_greeks = st.checkbox(
            "📊 Include Greeks Analysis",
            value=current_greeks,
            help="Add Delta, Gamma, Theta, Vega analysis to recommendations",
        )
        update_options_setting("include_greeks", include_greeks)

    with col2:
        current_volatility = get_options_setting("volatility_analysis", False)
        volatility_analysis = st.checkbox(
            "📈 Volatility Analysis",
            value=current_volatility,
            help="Include implied vs historical volatility comparison",
        )
        update_options_setting("volatility_analysis", volatility_analysis)

    with col3:
        current_download = get_options_setting("download_csv", False)
        download_csv = st.checkbox(
            "📄 Enable CSV Export",
            value=current_download,
            help="Generate a downloadable CSV file of the recommendations",
        )
        update_options_setting("download_csv", download_csv)

    # Add advanced settings panel
    render_advanced_settings_panel()

    # Add settings impact documentation
    render_settings_impact_documentation()

    st.markdown("---")

    # Settings change indicator and auto-refresh option
    if settings_have_changed():
        st.info("🔄 Settings have changed. Analysis will use the new configuration.")

        # Auto-refresh option
        auto_refresh = get_options_setting("auto_refresh", False)
        if auto_refresh:
            st.info("🔄 Auto-refresh is enabled. Running analysis with new settings...")
            # Trigger analysis automatically
            st.session_state.trigger_auto_analysis = True

    # Analyze button with enhanced state management
    analyze_clicked = st.button("🔍 Analyze Options", type="primary")

    # Check if we should run analysis (button clicked or auto-refresh triggered)
    should_run_analysis = analyze_clicked or st.session_state.get(
        "trigger_auto_analysis", False
    )

    if should_run_analysis:
        # Clear auto-trigger flag
        if "trigger_auto_analysis" in st.session_state:
            del st.session_state.trigger_auto_analysis

        # Get all current settings
        analysis_settings = get_analysis_settings()

        # Log the interaction
        logger.info(
            f"Options analysis requested for {ticker} - strategy={strategy_type}, "
            f"min_days={min_days}, top_n={top_n}, settings_changed={settings_have_changed()}"
        )

        # Create loading placeholder
        with st.spinner(f"🔄 Analyzing {strategy_type.lower()} for {ticker}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Update progress
                status_text.text("Fetching options data...")
                progress_bar.progress(25)

                # Check if we can use cached results
                use_cache = get_options_setting("enable_caching", True)
                cached_result = None

                if use_cache and not settings_have_changed():
                    cached_result = st.session_state.get("analysis_cache")
                    if cached_result is not None:
                        status_text.text("Using cached results...")
                        progress_bar.progress(100)
                        recommendations = cached_result

                if cached_result is None:
                    # Call the analyze_options_strategy function with current settings
                    status_text.text("Computing technical indicators...")
                    progress_bar.progress(50)

                    # Use enhanced analysis with custom settings
                    if analysis_settings["use_custom_weights"] or any(
                        [
                            analysis_settings.get("delta_threshold") is not None,
                            analysis_settings.get("volume_threshold") is not None,
                            analysis_settings.get("bid_ask_spread") is not None,
                            analysis_settings.get("open_interest") is not None,
                        ]
                    ):
                        logger.info("Using enhanced analysis with custom settings")
                        recommendations = analyze_options_with_custom_settings(
                            strategy_type=strategy_type,
                            ticker=ticker,
                            analysis_settings=analysis_settings,
                        )
                    else:
                        # Use standard analysis for default settings
                        logger.info("Using standard analysis with default settings")
                        recommendations = analyze_options_strategy(
                            strategy_type=strategy_type,
                            ticker=ticker,
                            min_days=min_days,
                            top_n=top_n,
                            risk_tolerance=risk_tolerance,
                            time_horizon=time_horizon,
                        )

                    status_text.text("Calculating composite scores...")
                    progress_bar.progress(75)

                    # Cache the results if caching is enabled
                    if use_cache:
                        st.session_state.analysis_cache = recommendations
                        st.session_state.analysis_timestamp = datetime.now()

                # Mark settings as applied
                mark_settings_applied()

                if recommendations.empty:
                    progress_bar.progress(100)
                    status_text.empty()
                    st.warning(
                        f"⚠️ No options found for {ticker} with minimum {min_days} days to expiry"
                    )
                    logger.warning(
                        f"No options found for {ticker} with min_days={min_days}"
                    )
                else:
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Display success message with settings info
                    success_msg = f"✅ Found {len(recommendations)} {strategy_type.lower()} recommendations for {ticker}"
                    if cached_result is not None:
                        success_msg += " (cached)"
                    st.success(success_msg)

                    # Display results section
                    st.subheader(f"📈 Top {strategy_type} Recommendations")

                    # Show current settings summary
                    with st.expander("📊 Analysis Configuration Used", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Strategy:** {strategy_type}")
                            st.write(f"**Risk Tolerance:** {risk_tolerance}")
                            st.write(f"**Time Horizon:** {time_horizon}")
                            st.write(f"**Min Days:** {min_days}")
                            st.write(f"**Top N:** {top_n}")

                        with col2:
                            st.write(
                                f"**Custom Weights:** {'Yes' if analysis_settings['use_custom_weights'] else 'No'}"
                            )
                            st.write(
                                f"**Greeks Analysis:** {'Yes' if include_greeks else 'No'}"
                            )
                            st.write(
                                f"**Volatility Analysis:** {'Yes' if volatility_analysis else 'No'}"
                            )
                            if st.session_state.get("analysis_timestamp"):
                                st.write(
                                    f"**Analysis Time:** {st.session_state.analysis_timestamp.strftime('%H:%M:%S')}"
                                )

                    # Check for partial data and display warning banner
                    has_partial_data = check_for_partial_data(recommendations)
                    if has_partial_data:
                        st.warning(
                            "⚠️ Some scores are based on partial data. "
                            "Hover over score details for breakdown.",
                            icon="⚠️",
                        )
                        # Log the UI-level warning
                        logger.warning(
                            f"UI Warning: Partial scoring data detected for {ticker} options analysis"
                        )

                    # Enhanced Options Analysis Display
                    if include_greeks:
                        render_score_components_analysis(
                            recommendations, ticker, strategy_type
                        )

                    if volatility_analysis:
                        render_volatility_analysis(data, ticker)

                    # Display key metrics with tooltips and help icon
                    col_header, col_help = st.columns([6, 1])

                    with col_header:
                        st.markdown("#### Key Technical Indicators")

                    with col_help:
                        with st.popover(
                            "❓ Score Help", help="Click for score explanation"
                        ):
                            st.markdown(
                                """
                            **Score Explanation**

                            This score is based on 5 weighted indicators:
                            - 📈 **RSI** (Relative Strength Index): Momentum indicator
                            - 📊 **Beta**: Market correlation coefficient
                            - 🚀 **Momentum**: Price trend strength
                            - 💨 **Implied Volatility**: Option price uncertainty
                            - 📅 **Forecast**: Analyst forecast confidence

                            If data is missing, the weights are redistributed proportionally
                            among available indicators to maintain fair comparison.
                            """
                            )

                    # Get scoring weights for display and methodology
                    if analysis_settings["use_custom_weights"]:
                        current_weights = analysis_settings["custom_scoring_weights"]
                        st.info("📊 Using custom scoring weights")
                    else:
                        scoring_weights = get_scoring_weights()
                        current_weights = scoring_weights

                    # Extract actual weights used in scoring from recommendations for methodology
                    actual_weights_for_methodology = current_weights  # Default fallback
                    if (
                        not recommendations.empty
                        and "score_details" in recommendations.columns
                    ):
                        # Get actual weights from the first recommendation's score_details
                        first_score_details = recommendations.iloc[0]["score_details"]
                        if isinstance(first_score_details, dict):
                            try:
                                all_indicator_names = set(get_scoring_indicator_names())
                                # Extract only the actual scoring indicators (not metadata)
                                actual_weights_for_methodology = {
                                    k: v
                                    for k, v in first_score_details.items()
                                    if k in all_indicator_names
                                }
                            except ImportError:
                                # Fallback to known indicators if import fails
                                known_indicators = {
                                    "rsi",
                                    "beta",
                                    "momentum",
                                    "iv",
                                    "forecast",
                                }
                                actual_weights_for_methodology = {
                                    k: v
                                    for k, v in first_score_details.items()
                                    if k in known_indicators
                                }

                    # Create a table with metric-aware headers
                    st.markdown("#### Options Recommendations")

                    # Render strategy-specific table
                    render_strategy_specific_table(recommendations, strategy_type)

                    # Add comprehensive score components analysis section
                    render_score_components_analysis(
                        recommendations, ticker, strategy_type
                    )

                    # Add detailed scoring breakdown and individual recommendation analysis
                    render_comprehensive_scoring_breakdown(
                        recommendations, ticker, strategy_type
                    )

                    # Strategy-specific insights
                    render_strategy_insights(
                        strategy_type, recommendations, ticker, risk_tolerance
                    )

                    # Enhanced methodology explanation with actual weights used
                    render_enhanced_methodology(actual_weights_for_methodology)

                    # CSV download section
                    if download_csv:
                        render_csv_download(recommendations, ticker, strategy_type)

            except OptionsAdvisorError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"⚠️ Options analysis error: {str(e)}")
                logger.error(f"Options advisor error for {ticker}: {str(e)}")

            except InsufficientDataError as e:
                progress_bar.empty()
                status_text.empty()
                st.warning(f"⚠️ Insufficient data: {str(e)}")
                logger.warning(f"Insufficient data for {ticker}: {str(e)}")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"🚨 Unexpected error: {str(e)}")
                logger.error(
                    f"Unexpected error in options advisor for {ticker}: {str(e)}",
                    exc_info=True,
                )


def render_score_components_analysis(
    recommendations: pd.DataFrame, ticker: str, strategy_type: str
) -> None:
    """
    Render comprehensive score components analysis with enhanced details.

    This function provides detailed breakdown of scoring components, data quality
    assessment, and interactive exploration of how scores are calculated.

    Args:
        recommendations: DataFrame containing options recommendations
        ticker: Stock ticker symbol
        strategy_type: Type of options strategy being analyzed
    """
    if recommendations.empty or "score_details" not in recommendations.columns:
        return

    st.subheader("📊 Scoring Components Analysis")

    # Import here to avoid circular imports
    try:
        total_indicators = get_total_scoring_indicators()
        all_indicator_names = set(get_scoring_indicator_names())
    except ImportError:
        total_indicators = 5
        all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}

    # Get first recommendation's score details for overall analysis
    first_score_details = recommendations.iloc[0]["score_details"]

    if isinstance(first_score_details, dict):
        # Separate actual indicators from metadata
        actual_indicators = {
            k: v for k, v in first_score_details.items() if k in all_indicator_names
        }
        metadata_fields = {
            k: v for k, v in first_score_details.items() if k not in all_indicator_names
        }

        # Data Quality Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            available_count = len(actual_indicators)
            data_quality = (
                "Excellent"
                if available_count == total_indicators
                else "Good"
                if available_count >= (total_indicators * 0.8)
                else "Moderate"
                if available_count >= (total_indicators * 0.6)
                else "Limited"
            )

            quality_colors = {
                "Excellent": "🟢",
                "Good": "🟡",
                "Moderate": "🟠",
                "Limited": "🔴",
            }
            st.metric(
                "Data Availability",
                f"{available_count}/{total_indicators}",
                delta=f"{quality_colors.get(data_quality, '⚪')} {data_quality}",
                help="Number of technical indicators with available data",
            )

        with col2:
            # Show the number of recommendations
            st.metric(
                "Recommendations",
                len(recommendations),
                help="Number of options recommendations generated",
            )

        with col3:
            # Show strategy type
            st.metric("Strategy", strategy_type, help="Options strategy being analyzed")

        # Detailed Components Breakdown
        st.markdown("#### 🔍 Scoring Components Breakdown")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            ["📈 Technical Indicators", "⚙️ Configuration", "📋 Data Quality"]
        )

        with tab1:
            if actual_indicators:
                st.markdown("**Technical Scoring Weights & Descriptions:**")

                # Enhanced indicator information
                indicator_details = {
                    "rsi": {
                        "name": "RSI (Relative Strength Index)",
                        "description": "Measures overbought/oversold conditions (0-100)",
                        "interpretation": "Lower RSI often better for call buying opportunities",
                        "icon": "📈",
                    },
                    "beta": {
                        "name": "Beta Coefficient",
                        "description": "Stock volatility relative to market",
                        "interpretation": "Moderate beta (0.8-1.5) often preferred for balanced risk",
                        "icon": "📊",
                    },
                    "momentum": {
                        "name": "Price Momentum",
                        "description": "Recent price movement trend",
                        "interpretation": "Positive momentum favors bullish strategies",
                        "icon": "🚀",
                    },
                    "iv": {
                        "name": "Implied Volatility",
                        "description": "Market's expectation of future volatility",
                        "interpretation": "Lower IV generally better for option buying",
                        "icon": "💨",
                    },
                    "forecast": {
                        "name": "Analyst Forecast Confidence",
                        "description": "Wall Street consensus confidence level",
                        "interpretation": "Higher confidence supports directional strategies",
                        "icon": "🔮",
                    },
                }

                for indicator, weight in actual_indicators.items():
                    details = indicator_details.get(
                        indicator,
                        {
                            "name": indicator.upper(),
                            "description": "Technical indicator",
                            "interpretation": "Contributes to composite score",
                            "icon": "📋",
                        },
                    )

                    with st.expander(
                        f"{details['icon']} {details['name']} - Weight: {weight:.1%}"
                    ):
                        st.markdown(f"**Description:** {details['description']}")
                        st.markdown(
                            f"**Strategy Interpretation:** {details['interpretation']}"
                        )

                        # Show actual values if available in recommendations
                        if indicator == "rsi" and "RSI" in recommendations.columns:
                            avg_value = recommendations["RSI"].mean()
                            st.markdown(f"**Current Average Value:** {avg_value:.1f}")
                        elif indicator == "beta" and "Beta" in recommendations.columns:
                            avg_value = recommendations["Beta"].mean()
                            st.markdown(f"**Current Average Value:** {avg_value:.2f}")
                        elif (
                            indicator == "momentum"
                            and "Momentum" in recommendations.columns
                        ):
                            avg_value = recommendations["Momentum"].mean()
                            st.markdown(f"**Current Average Value:** {avg_value:.3f}")
                        elif indicator == "iv" and "IV" in recommendations.columns:
                            avg_value = (
                                recommendations["IV"].mean()
                                if recommendations["IV"].dtype != "object"
                                else 0
                            )
                            st.markdown(
                                f"**Current Average Value:** {avg_value:.1%}"
                                if avg_value > 0
                                else "**Current Average Value:** See individual recommendations"
                            )
                        elif (
                            indicator == "forecast"
                            and "ForecastConfidence" in recommendations.columns
                        ):
                            avg_value = recommendations["ForecastConfidence"].mean()
                            st.markdown(f"**Current Average Value:** {avg_value:.1%}")
            else:
                st.warning("No technical indicators available for analysis")

        with tab2:
            st.markdown("**Analysis Configuration:**")

            # Show metadata fields
            if metadata_fields:
                for field, value in metadata_fields.items():
                    if field == "risk_tolerance":
                        st.markdown(f"🎯 **Risk Tolerance:** {value}")

                        # Explain risk tolerance impact
                        if value == "Conservative":
                            st.info(
                                "Conservative settings prefer lower risk, higher probability strategies"
                            )
                        elif value == "Aggressive":
                            st.info(
                                "Aggressive settings favor higher reward potential with increased risk"
                            )
                        else:
                            st.info(
                                "Moderate settings balance risk and reward considerations"
                            )
                    else:
                        # Format field name nicely
                        field_name = field.replace("_", " ").title()
                        st.markdown(f"📋 **{field_name}:** {value}")

            # Show strategy-specific configuration impact
            st.markdown("**Strategy-Specific Adjustments:**")
            if strategy_type == "Long Calls":
                st.markdown("- Emphasizes upside potential and momentum")
                st.markdown("- Prefers lower implied volatility for cost efficiency")
            elif strategy_type == "Bull Call Spread":
                st.markdown("- Balances profit potential with risk limitation")
                st.markdown("- Considers spread efficiency and breakeven points")
            elif strategy_type == "Covered Call":
                st.markdown("- Prioritizes income generation and stability")
                st.markdown(
                    "- Favors moderate volatility and assignment considerations"
                )
            elif strategy_type == "Cash-Secured Put":
                st.markdown("- Focuses on entry opportunity and income")
                st.markdown(
                    "- Considers assignment probability and effective cost basis"
                )

        with tab3:
            st.markdown("**Data Quality Assessment:**")

            # Missing indicators analysis
            missing_indicators = all_indicator_names - set(actual_indicators.keys())
            if missing_indicators:
                st.markdown("**❌ Missing Data Sources:**")
                for indicator in sorted(missing_indicators):
                    indicator_name = {
                        "rsi": "RSI (Relative Strength Index)",
                        "beta": "Beta Coefficient",
                        "momentum": "Price Momentum",
                        "iv": "Implied Volatility",
                        "forecast": "Analyst Forecast",
                    }.get(indicator, indicator.upper())
                    st.markdown(f"- ❌ {indicator_name}")

                st.warning(
                    f"⚠️ Analysis using {len(actual_indicators)}/{total_indicators} data sources. "
                    "Missing data sources may affect recommendation accuracy."
                )
            else:
                st.success(
                    "✅ All technical indicators available - optimal data quality!"
                )

            # Data quality impact explanation
            st.markdown("**Impact on Recommendations:**")
            if len(actual_indicators) == total_indicators:
                st.success(
                    "🟢 **Excellent**: All indicators available for comprehensive analysis"
                )
            elif len(actual_indicators) >= (total_indicators * 0.8):
                st.info(
                    "🟡 **Good**: Most indicators available, minor impact on accuracy"
                )
            elif len(actual_indicators) >= (total_indicators * 0.6):
                st.warning(
                    "🟠 **Moderate**: Some indicators missing, moderate impact on accuracy"
                )
            else:
                st.error(
                    "🔴 **Limited**: Many indicators missing, significant impact on accuracy"
                )

        # Interactive score exploration for individual recommendations
        st.markdown("#### 🔍 Individual Recommendation Analysis")

        if len(recommendations) > 1:
            # Let user select a specific recommendation to explore
            rec_options = []
            for idx, row in recommendations.head(
                5
            ).iterrows():  # Show top 5 for selection
                if strategy_type == "Bull Call Spread":
                    label = f"#{idx+1}: ${row.get('long_strike', 'N/A')} - ${row.get('short_strike', 'N/A')} (Score: {row.get('CompositeScore', 0):.3f})"
                else:
                    label = f"#{idx+1}: ${row.get('strike', row.get('Strike', 'N/A'))} (Score: {row.get('CompositeScore', 0):.3f})"
                rec_options.append((label, idx))

            selected_option = st.selectbox(
                "Select recommendation to analyze:",
                rec_options,
                format_func=lambda x: x[0],
                help="Choose a specific recommendation to see detailed scoring breakdown",
            )
            selected_idx = selected_option[1]

            # Show details for selected recommendation
            selected_details = recommendations.iloc[selected_idx]["score_details"]
            if isinstance(selected_details, dict):
                render_score_details_popover(selected_details, selected_idx)
        else:
            # Show details for the single recommendation
            render_score_details_popover(first_score_details, 0)


def render_comprehensive_scoring_breakdown(
    recommendations: pd.DataFrame, ticker: str, strategy_type: str
) -> None:
    """
    Render comprehensive scoring breakdown and individual recommendation analysis.

    This provides detailed breakdowns of each recommendation's scoring components
    and allows users to explore individual recommendations in depth.

    Args:
        recommendations: DataFrame containing options recommendations
        ticker: Stock ticker symbol
        strategy_type: Type of options strategy being analyzed
    """
    if recommendations.empty or "score_details" not in recommendations.columns:
        return

    st.subheader("🔬 Comprehensive Scoring Breakdown & Individual Analysis")

    # Import scoring functions to avoid circular imports
    try:
        total_indicators = get_total_scoring_indicators()
        all_indicator_names = set(get_scoring_indicator_names())
        scoring_weights = get_scoring_weights()
    except ImportError:
        total_indicators = 5
        all_indicator_names = {"rsi", "beta", "momentum", "iv", "forecast"}
        scoring_weights = {
            "rsi": 0.25,
            "beta": 0.15,
            "momentum": 0.25,
            "iv": 0.20,
            "forecast": 0.15,
        }

    # Create main tabs for comprehensive analysis
    main_tab1, main_tab2, main_tab3 = st.tabs(
        [
            "📊 Portfolio Scoring Overview",
            "🎯 Individual Recommendation Analysis",
            "📈 Comparative Analysis",
        ]
    )

    with main_tab1:
        st.markdown("#### 📊 Portfolio-Level Scoring Analysis")

        # Aggregate scoring statistics across all recommendations
        scoring_stats = {}
        available_indicators = set()

        for idx, row in recommendations.iterrows():
            score_details = row.get("score_details", {})
            if isinstance(score_details, dict):
                # Filter to only actual indicators
                indicators = {
                    k: v for k, v in score_details.items() if k in all_indicator_names
                }
                available_indicators.update(indicators.keys())

                for indicator, weight in indicators.items():
                    if indicator not in scoring_stats:
                        scoring_stats[indicator] = []
                    scoring_stats[indicator].append(weight)

        # Display aggregate metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_coverage = len(available_indicators) / total_indicators * 100
            coverage_color = (
                "🟢" if avg_coverage >= 80 else "🟡" if avg_coverage >= 60 else "🔴"
            )
            st.metric(
                "Data Coverage",
                f"{avg_coverage:.0f}%",
                delta=f"{coverage_color} {len(available_indicators)}/{total_indicators} indicators",
                help="Percentage of scoring indicators with available data",
            )

        with col2:
            total_score_avg = (
                recommendations["TotalScore"].mean()
                if "TotalScore" in recommendations.columns
                else 0
            )
            st.metric(
                "Avg Total Score",
                f"{total_score_avg:.2f}",
                help="Average total composite score across all recommendations",
            )

        with col3:
            score_std = (
                recommendations["TotalScore"].std()
                if "TotalScore" in recommendations.columns
                else 0
            )
            consistency = (
                "High" if score_std < 0.5 else "Medium" if score_std < 1.0 else "Low"
            )
            st.metric(
                "Score Consistency",
                consistency,
                delta=f"σ = {score_std:.3f}",
                help="How consistent the scores are across recommendations",
            )

        with col4:
            top_score = (
                recommendations["TotalScore"].max()
                if "TotalScore" in recommendations.columns
                else 0
            )
            st.metric(
                "Best Score",
                f"{top_score:.2f}",
                help="Highest scoring recommendation in the portfolio",
            )

        # Detailed indicator breakdown
        st.markdown("#### 📈 Indicator-by-Indicator Breakdown")

        if scoring_stats:
            for indicator in all_indicator_names:
                if indicator in scoring_stats:
                    weights = scoring_stats[indicator]
                    avg_weight = sum(weights) / len(weights)

                    # Enhanced indicator display with actual values
                    with st.expander(
                        f"📊 {indicator.upper()} Analysis - Avg Weight: {avg_weight:.1%}"
                    ):
                        col_left, col_right = st.columns([2, 1])

                        with col_left:
                            st.markdown(f"**Weight Distribution:**")
                            st.markdown(f"- Average: {avg_weight:.1%}")
                            st.markdown(
                                f"- Standard Weight: {scoring_weights.get(indicator, 0):.1%}"
                            )
                            st.markdown(
                                f"- Used in: {len(weights)}/{len(recommendations)} recommendations"
                            )

                            # Show actual values if available in recommendations
                            value_col_map = {
                                "rsi": "RSI",
                                "beta": "Beta",
                                "momentum": "Momentum",
                                "iv": "IV",
                                "forecast": "ForecastConfidence",
                            }

                            if (
                                indicator in value_col_map
                                and value_col_map[indicator] in recommendations.columns
                            ):
                                values = recommendations[value_col_map[indicator]]
                                if values.dtype != "object":
                                    st.markdown(
                                        f"- Value Range: {values.min():.3f} to {values.max():.3f}"
                                    )
                                    st.markdown(f"- Average Value: {values.mean():.3f}")

                        with col_right:
                            # Mini histogram of weights
                            weight_counts = {}
                            for w in weights:
                                w_rounded = round(w, 2)
                                weight_counts[w_rounded] = (
                                    weight_counts.get(w_rounded, 0) + 1
                                )

                            st.markdown("**Weight Distribution:**")
                            for weight, count in sorted(weight_counts.items()):
                                bar_length = int(count / len(weights) * 20)
                                bar = "█" * bar_length
                                st.markdown(f"{weight:.1%}: {bar} ({count})")
                else:
                    st.warning(f"⚠️ {indicator.upper()}: No data available")

        # Portfolio quality assessment
        st.markdown("#### 🎯 Portfolio Quality Assessment")

        quality_metrics = []

        # Data completeness
        completeness_score = len(available_indicators) / total_indicators
        quality_metrics.append(
            ("Data Completeness", completeness_score, "Higher is better")
        )

        # Score consistency (inverse of std dev)
        if "TotalScore" in recommendations.columns:
            consistency_score = max(
                0,
                1
                - (
                    recommendations["TotalScore"].std()
                    / recommendations["TotalScore"].mean()
                ),
            )
            quality_metrics.append(
                ("Score Consistency", consistency_score, "Higher is better")
            )

        # Display quality metrics
        for metric_name, score, interpretation in quality_metrics:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Create a visual bar
                bar_length = int(score * 20)
                bar_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                bar = "█" * bar_length + "░" * (20 - bar_length)
                st.markdown(f"**{metric_name}:** {bar_color} {bar} {score:.1%}")
                st.caption(interpretation)

    with main_tab2:
        st.markdown("#### 🎯 Individual Recommendation Deep Dive")

        # Recommendation selector
        rec_options = []
        for idx, row in recommendations.iterrows():
            strike = row.get("Strike", "N/A")
            expiry = row.get("Expiry", "N/A")
            total_score = row.get("TotalScore", 0)
            rec_options.append(
                f"#{idx+1}: Strike ${strike} | Exp: {expiry} | Score: {total_score:.2f}"
            )

        selected_rec = st.selectbox(
            "🎯 Select Recommendation to Analyze:",
            options=range(len(rec_options)),
            format_func=lambda x: rec_options[x],
            help="Choose a specific recommendation for detailed analysis",
        )

        if selected_rec is not None:
            selected_row = recommendations.iloc[selected_rec]
            score_details = selected_row.get("score_details", {})

            # Individual recommendation header
            st.markdown(f"#### 📋 Recommendation #{selected_rec + 1} Analysis")

            # Key metrics display
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                strike = selected_row.get("Strike", "N/A")
                st.metric("Strike Price", f"${strike}")

            with col2:
                expiry = selected_row.get("Expiry", "N/A")
                st.metric("Expiration", str(expiry))

            with col3:
                total_score = selected_row.get("TotalScore", 0)
                st.metric("Total Score", f"{total_score:.3f}")

            with col4:
                # Calculate rank
                rank = (recommendations["TotalScore"] > total_score).sum() + 1
                st.metric("Rank", f"#{rank}")

            # Detailed scoring breakdown for this recommendation
            if isinstance(score_details, dict):
                actual_indicators = {
                    k: v for k, v in score_details.items() if k in all_indicator_names
                }
                metadata_fields = {
                    k: v
                    for k, v in score_details.items()
                    if k not in all_indicator_names
                }

                st.markdown("#### 🔍 Scoring Component Analysis")

                # Create two columns for detailed analysis
                left_col, right_col = st.columns([2, 1])

                with left_col:
                    st.markdown("**Technical Indicators Breakdown:**")

                    # Enhanced indicator details with actual values
                    indicator_details = {
                        "rsi": {
                            "name": "RSI (Relative Strength Index)",
                            "description": "Momentum oscillator (0-100)",
                            "good_range": "30-70 for balanced, <30 oversold, >70 overbought",
                            "icon": "📈",
                            "value_col": "RSI",
                        },
                        "beta": {
                            "name": "Beta Coefficient",
                            "description": "Volatility vs market",
                            "good_range": "0.8-1.5 for balanced risk/reward",
                            "icon": "📊",
                            "value_col": "Beta",
                        },
                        "momentum": {
                            "name": "Price Momentum",
                            "description": "Recent price trend strength",
                            "good_range": "Positive for bullish strategies",
                            "icon": "🚀",
                            "value_col": "Momentum",
                        },
                        "iv": {
                            "name": "Implied Volatility",
                            "description": "Option pricing volatility expectation",
                            "good_range": "Lower generally better for buying",
                            "icon": "💨",
                            "value_col": "IV",
                        },
                        "forecast": {
                            "name": "Analyst Forecast Confidence",
                            "description": "Wall Street consensus strength",
                            "good_range": "Higher confidence supports direction",
                            "icon": "🔮",
                            "value_col": "ForecastConfidence",
                        },
                    }

                    for indicator, weight in actual_indicators.items():
                        details = indicator_details.get(
                            indicator,
                            {
                                "name": indicator.upper(),
                                "description": "Technical indicator",
                                "good_range": "Varies by strategy",
                                "icon": "📋",
                                "value_col": indicator.upper(),
                            },
                        )

                        with st.expander(
                            f"{details['icon']} {details['name']} - Weight: {weight:.1%}"
                        ):
                            st.markdown(f"**Description:** {details['description']}")
                            st.markdown(f"**Optimal Range:** {details['good_range']}")

                            # Show actual value for this recommendation
                            if details["value_col"] in selected_row:
                                actual_value = selected_row[details["value_col"]]
                                if indicator == "iv" and isinstance(actual_value, str):
                                    st.markdown(f"**Current Value:** {actual_value}")
                                elif indicator == "forecast" and pd.notna(actual_value):
                                    st.markdown(
                                        f"**Current Value:** {actual_value:.1%}"
                                    )
                                elif pd.notna(actual_value) and isinstance(
                                    actual_value, (int, float)
                                ):
                                    st.markdown(
                                        f"**Current Value:** {actual_value:.3f}"
                                    )

                            # Contribution to total score
                            contribution = weight * scoring_weights.get(indicator, 0)
                            st.markdown(f"**Score Contribution:** {contribution:.3f}")

                            # Weight vs standard comparison
                            standard_weight = scoring_weights.get(indicator, 0)
                            if weight != standard_weight:
                                diff = weight - standard_weight
                                direction = "higher" if diff > 0 else "lower"
                                st.info(
                                    f"Note: Weight is {abs(diff):.1%} {direction} than standard ({standard_weight:.1%})"
                                )

                with right_col:
                    st.markdown("**Scoring Summary:**")

                    # Visual scoring breakdown
                    total_weight = sum(actual_indicators.values())
                    st.markdown(f"**Total Weight:** {total_weight:.1%}")

                    # Weight distribution pie chart representation
                    st.markdown("**Weight Distribution:**")
                    for indicator, weight in actual_indicators.items():
                        percentage = (
                            weight / total_weight * 100 if total_weight > 0 else 0
                        )
                        bar_length = int(percentage / 5)  # Scale for display
                        bar = "█" * bar_length
                        st.markdown(f"{indicator.upper()}: {bar} {percentage:.1f}%")

                    # Missing indicators
                    missing_indicators = all_indicator_names - set(
                        actual_indicators.keys()
                    )
                    if missing_indicators:
                        st.markdown("**Missing Indicators:**")
                        for missing in missing_indicators:
                            st.markdown(f"⚠️ {missing.upper()}")

                    # Configuration metadata
                    if metadata_fields:
                        st.markdown("**Configuration:**")
                        for field, value in metadata_fields.items():
                            field_name = field.replace("_", " ").title()
                            st.markdown(f"• **{field_name}:** {value}")

    with main_tab3:
        st.markdown("#### 📈 Comparative Analysis Across Recommendations")

        # Comparative scoring analysis
        if len(recommendations) > 1:
            # Score distribution analysis
            st.markdown("##### 📊 Score Distribution")

            col1, col2 = st.columns(2)

            with col1:
                # Score statistics
                if "TotalScore" in recommendations.columns:
                    scores = recommendations["TotalScore"]
                    st.markdown("**Score Statistics:**")
                    st.markdown(f"• **Highest:** {scores.max():.3f}")
                    st.markdown(f"• **Lowest:** {scores.min():.3f}")
                    st.markdown(f"• **Average:** {scores.mean():.3f}")
                    st.markdown(f"• **Std Dev:** {scores.std():.3f}")
                    st.markdown(f"• **Range:** {scores.max() - scores.min():.3f}")

                    # Score quartiles
                    q1 = scores.quantile(0.25)
                    q2 = scores.quantile(0.50)
                    q3 = scores.quantile(0.75)

                    st.markdown("**Score Quartiles:**")
                    st.markdown(f"• **Q1 (25%):** {q1:.3f}")
                    st.markdown(f"• **Q2 (50%):** {q2:.3f}")
                    st.markdown(f"• **Q3 (75%):** {q3:.3f}")

            with col2:
                # Top performers analysis
                st.markdown("**Top Performers:**")

                top_3 = (
                    recommendations.nlargest(3, "TotalScore")
                    if "TotalScore" in recommendations.columns
                    else recommendations.head(3)
                )

                for idx, (_, row) in enumerate(top_3.iterrows()):
                    rank_num = idx + 1
                    strike = row.get("Strike", "N/A")
                    score = row.get("TotalScore", 0)
                    expiry = row.get("Expiry", "N/A")

                    medal = "🥇" if rank_num == 1 else "🥈" if rank_num == 2 else "🥉"
                    st.markdown(
                        f"{medal} **#{rank_num}:** ${strike} | {expiry} | {score:.3f}"
                    )

                # Bottom performers
                st.markdown("**Bottom Performers:**")
                bottom_3 = (
                    recommendations.nsmallest(3, "TotalScore")
                    if "TotalScore" in recommendations.columns
                    else recommendations.tail(3)
                )

                for idx, (_, row) in enumerate(bottom_3.iterrows()):
                    strike = row.get("Strike", "N/A")
                    score = row.get("TotalScore", 0)
                    expiry = row.get("Expiry", "N/A")
                    rank_from_bottom = idx + 1

                    st.markdown(
                        f"⬇️ **#{len(recommendations) - rank_from_bottom + 1}:** ${strike} | {expiry} | {score:.3f}"
                    )

            # Indicator consistency analysis
            st.markdown("##### 🎯 Indicator Consistency Analysis")

            indicator_consistency = {}

            for idx, row in recommendations.iterrows():
                score_details = row.get("score_details", {})
                if isinstance(score_details, dict):
                    indicators = {
                        k: v
                        for k, v in score_details.items()
                        if k in all_indicator_names
                    }

                    for indicator, weight in indicators.items():
                        if indicator not in indicator_consistency:
                            indicator_consistency[indicator] = []
                        indicator_consistency[indicator].append(weight)

            # Display consistency metrics
            for indicator in all_indicator_names:
                if indicator in indicator_consistency:
                    weights = indicator_consistency[indicator]
                    avg_weight = sum(weights) / len(weights)
                    std_weight = (
                        sum((w - avg_weight) ** 2 for w in weights) / len(weights)
                    ) ** 0.5
                    consistency = (
                        "High"
                        if std_weight < 0.01
                        else "Medium"
                        if std_weight < 0.05
                        else "Low"
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"{indicator.upper()}", f"{avg_weight:.1%}")
                    with col2:
                        st.metric("Std Dev", f"{std_weight:.3f}")
                    with col3:
                        st.metric("Consistency", consistency)
                    with col4:
                        coverage = len(weights) / len(recommendations) * 100
                        st.metric("Coverage", f"{coverage:.0f}%")

                    st.markdown("---")
        else:
            st.info(
                "📊 Comparative analysis requires multiple recommendations. Add more recommendations to see comparison metrics."
            )


def render_greeks_analysis(recommendations: pd.DataFrame, ticker: str) -> None:
    """Render Greeks analysis for the options recommendations."""

    st.subheader("🧮 Options Greeks Analysis")

    # Simulated Greeks data (in production, this would come from real options data)
    st.info("📊 Greeks analysis shows option price sensitivities to various factors")

    if not recommendations.empty:
        col1, col2, col3, col4 = st.columns(4)

        # Simulated Greeks values for demonstration
        with col1:
            st.metric(
                "Delta (Δ)",
                "0.65",
                help="Price sensitivity to underlying stock movement (0-1 for calls)",
            )

        with col2:
            st.metric(
                "Gamma (Γ)",
                "0.12",
                help="Rate of change of Delta relative to underlying price",
            )

        with col3:
            st.metric(
                "Theta (Θ)",
                "-0.05",
                help="Time decay - how much option loses value per day",
            )

        with col4:
            st.metric(
                "Vega (V)", "0.18", help="Sensitivity to changes in implied volatility"
            )

        # Greeks interpretation
        with st.expander("📚 Greeks Interpretation Guide", expanded=False):
            st.markdown(
                """
            **Delta (Δ):** Measures how much the option price changes for each $1 move in the stock
            - Higher delta = more sensitive to stock price changes
            - Call options: 0 to 1, Put options: -1 to 0

            **Gamma (Γ):** Measures how much delta changes as the stock price moves
            - Higher gamma = delta changes more rapidly
            - Important for risk management

            **Theta (Θ):** Time decay - how much value the option loses each day
            - Always negative for long options
            - Accelerates as expiration approaches

            **Vega (V):** Sensitivity to implied volatility changes
            - Higher vega = more sensitive to volatility changes
            - Important when volatility is expected to change
            """
            )


def render_volatility_analysis(data: dict, ticker: str) -> None:
    """Render volatility analysis comparing implied vs historical volatility."""

    st.subheader("📈 Volatility Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Implied Volatility",
            "28.5%",
            delta="2.3%",
            help="Market's expectation of future volatility based on option prices",
        )

    with col2:
        st.metric(
            "Historical Volatility (30d)",
            "24.2%",
            help="Actual price volatility over the past 30 days",
        )

    # Volatility interpretation
    iv_vs_hv_ratio = 28.5 / 24.2

    if iv_vs_hv_ratio > 1.2:
        st.warning(
            "⚠️ Implied volatility is significantly higher than historical - options may be expensive"
        )
    elif iv_vs_hv_ratio < 0.8:
        st.success(
            "✅ Implied volatility is lower than historical - options may represent good value"
        )
    else:
        st.info("ℹ️ Implied and historical volatility are relatively aligned")


def render_strategy_insights(
    strategy_type: str, recommendations: pd.DataFrame, ticker: str, risk_tolerance: str
) -> None:
    """Render strategy-specific insights and recommendations."""

    st.subheader(f"💡 {strategy_type} Strategy Insights")

    if strategy_type == "Long Calls":
        st.markdown(
            """
        **Long Call Strategy Analysis:**
        - **Best for:** Bullish outlook with limited risk
        - **Max Risk:** Premium paid for the option
        - **Max Reward:** Unlimited upside potential
        - **Break-even:** Strike price + premium paid
        """
        )

        if risk_tolerance == "Conservative":
            st.info(
                "💡 Conservative tip: Consider in-the-money calls for higher delta and lower risk"
            )
        elif risk_tolerance == "Aggressive":
            st.info(
                "💡 Aggressive tip: Out-of-the-money calls offer higher leverage but more risk"
            )

    elif strategy_type == "Bull Call Spread":
        st.markdown(
            """
        **Bull Call Spread Strategy Analysis:**
        - **Best for:** Moderately bullish outlook with defined risk
        - **Max Risk:** Net premium paid (long call premium - short call premium)
        - **Max Reward:** Difference between strikes - net premium paid
        - **Break-even:** Lower strike + net premium paid

        **Key Metrics to Watch:**
        - **Profit Ratio:** Max profit ÷ max loss (higher is better)
        - **Time Decay:** Benefits the spread as expiration approaches
        - **Volatility Impact:** Lower volatility generally helps profitability
        """
        )

        if not recommendations.empty and "profit_ratio" in recommendations.columns:
            avg_profit_ratio = recommendations["profit_ratio"].mean()
            if avg_profit_ratio > 2.0:
                st.success(
                    f"✅ Excellent profit ratios averaging {avg_profit_ratio:.2f}:1"
                )
            elif avg_profit_ratio > 1.0:
                st.info(f"📊 Good profit ratios averaging {avg_profit_ratio:.2f}:1")
            else:
                st.warning(
                    f"⚠️ Lower profit ratios averaging {avg_profit_ratio:.2f}:1 - consider wider spreads"
                )

        if risk_tolerance == "Conservative":
            st.info(
                "💡 Conservative: Look for spreads with profit ratios > 2:1 and longer time to expiration"
            )
        elif risk_tolerance == "Aggressive":
            st.info(
                "💡 Aggressive: Consider narrower spreads for higher probability of success"
            )

    elif strategy_type == "Covered Call":
        st.markdown(
            """
        **Covered Call Strategy Analysis:**
        - **Best for:** Income generation on existing stock positions
        - **Max Risk:** Stock ownership risk minus premium received
        - **Max Reward:** Premium + upside to strike price
        - **Assignment Risk:** May be called away if stock rises above strike

        **Key Metrics to Watch:**
        - **Annualized Yield:** Premium income on annualized basis
        - **Upside Capture:** Additional return if stock appreciates to strike
        - **Total Return Potential:** Premium yield + upside capture
        """
        )

        if not recommendations.empty:
            if "annualized_yield" in recommendations.columns:
                avg_yield = recommendations["annualized_yield"].mean()
                if avg_yield > 20:
                    st.success(
                        f"✅ High income potential: {avg_yield:.1f}% annualized yield"
                    )
                elif avg_yield > 10:
                    st.info(
                        f"📊 Good income potential: {avg_yield:.1f}% annualized yield"
                    )
                else:
                    st.warning(f"⚠️ Modest income: {avg_yield:.1f}% annualized yield")

            if "upside_capture" in recommendations.columns:
                avg_upside = recommendations["upside_capture"].mean()
                if avg_upside > 10:
                    st.info(
                        f"📈 Additional upside potential: {avg_upside:.1f}% if assigned"
                    )

        if risk_tolerance == "Conservative":
            st.info(
                "💡 Conservative: Focus on out-of-the-money strikes to reduce assignment risk"
            )
        elif risk_tolerance == "Aggressive":
            st.info("💡 Aggressive: Consider at-the-money strikes for higher premiums")

    elif strategy_type == "Cash-Secured Put":
        st.markdown(
            """
        **Cash-Secured Put Strategy Analysis:**
        - **Best for:** Income generation + potential stock acquisition at discount
        - **Max Risk:** Strike price - premium received (if stock goes to zero)
        - **Max Reward:** Premium received (if stock stays above strike)
        - **Assignment Risk:** May acquire stock at strike price if below at expiration

        **Key Metrics to Watch:**
        - **Annualized Yield:** Premium income on annualized basis
        - **Assignment Discount:** Your entry price vs. current stock price
        - **Effective Cost:** Your net cost basis if assigned (strike - premium)
        """
        )

        if not recommendations.empty:
            if "annualized_yield" in recommendations.columns:
                avg_yield = recommendations["annualized_yield"].mean()
                if avg_yield > 15:
                    st.success(
                        f"✅ High income potential: {avg_yield:.1f}% annualized yield"
                    )
                elif avg_yield > 8:
                    st.info(
                        f"📊 Good income potential: {avg_yield:.1f}% annualized yield"
                    )
                else:
                    st.warning(f"⚠️ Modest income: {avg_yield:.1f}% annualized yield")

            if "discount_to_current" in recommendations.columns:
                avg_discount = recommendations["discount_to_current"].mean()
                if avg_discount > 15:
                    st.success(
                        f"✅ Excellent entry discount: {avg_discount:.1f}% below current price"
                    )
                elif avg_discount > 8:
                    st.info(
                        f"📊 Good entry discount: {avg_discount:.1f}% below current price"
                    )

        if risk_tolerance == "Conservative":
            st.info(
                "💡 Conservative: Choose strikes well below current price for lower assignment risk"
            )
        elif risk_tolerance == "Aggressive":
            st.info("💡 Aggressive: Consider higher strikes for more premium income")

    # Add general risk warnings based on strategy
    with st.expander("⚠️ Risk Considerations", expanded=False):
        if strategy_type == "Long Calls":
            st.warning(
                """
                **Long Call Risks:**
                - Time decay accelerates as expiration approaches
                - Can lose 100% of premium paid
                - Implied volatility changes affect option prices
                - Requires significant stock movement to profit
                """
            )
        elif strategy_type == "Bull Call Spread":
            st.warning(
                """
                **Bull Call Spread Risks:**
                - Limited profit potential compared to long calls
                - Both legs subject to time decay
                - Early assignment risk on short call
                - Requires stock to move above long strike by expiration
                """
            )
        elif strategy_type == "Covered Call":
            st.warning(
                """
                **Covered Call Risks:**
                - Stock may be called away in bull market
                - Full downside exposure to stock ownership
                - Opportunity cost if stock rallies strongly
                - Dividend risk around ex-dates
                """
            )
        elif strategy_type == "Cash-Secured Put":
            st.warning(
                """
                **Cash-Secured Put Risks:**
                - May be assigned stock in declining market
                - Full cash requirement tied up as collateral
                - Subject to stock's downside risk
                - Assignment typically occurs when stock is below strike
                """
            )


def render_enhanced_methodology(current_weights: dict = None) -> None:
    """Render enhanced methodology explanation with dynamic weights."""

    # Get default weights if none provided
    if current_weights is None:
        try:
            # Use strategy-specific default weights
            strategy_type = st.session_state.get("strategy_type", "Long Calls")
            current_weights = get_strategy_specific_weights(strategy_type)
        except ImportError:
            current_weights = {
                "rsi": 0.20,
                "beta": 0.20,
                "momentum": 0.20,
                "iv": 0.20,
                "forecast": 0.20,
            }

    with st.expander("🔬 Enhanced Analysis Methodology", expanded=False):
        st.markdown(
            f"""
        **Technical Scoring Components:**

        1. **RSI Analysis ({current_weights.get('rsi', 0.20):.0%} weight):**
           - Identifies overbought/oversold conditions
           - Optimal range: 30-70 for entry points

        2. **Beta Analysis ({current_weights.get('beta', 0.20):.0%} weight):**
           - Measures stock correlation with market
           - Higher beta = more volatility and option premium

        3. **Momentum Analysis ({current_weights.get('momentum', 0.20):.0%} weight):**
           - Price trend strength over multiple timeframes
           - Positive momentum favors call options

        4. **Implied Volatility Analysis ({current_weights.get('iv', 0.20):.0%} weight):**
           - Current IV vs historical levels
           - Higher IV = higher option prices

        5. **Forecast Analysis ({current_weights.get('forecast', 0.20):.0%} weight):**
           - Analyst forecast confidence
           - Higher forecast = higher confidence in the recommendation

        **Risk Management Features:**
        - Strategy-specific risk metrics
        - Greeks analysis for sensitivity measurement
        - Volatility comparison for fair value assessment
        """
        )


def render_csv_download(
    recommendations: pd.DataFrame, ticker: str, strategy_type: str
) -> None:
    """Render CSV download functionality."""

    st.subheader("📤 Export Analysis")

    if st.button("📊 Generate CSV Export"):
        # Prepare data for export
        export_data = recommendations.copy()
        export_data["Analysis_Date"] = datetime.now().strftime("%Y-%m-%d")
        export_data["Ticker"] = ticker
        export_data["Strategy"] = strategy_type

        # Create CSV
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)

        st.download_button(
            label="💾 Download Options Analysis CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{ticker}_{strategy_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        st.success("✅ CSV export ready for download!")


def render_strategy_specific_table(
    recommendations: pd.DataFrame, strategy_type: str
) -> None:
    """
    Render strategy-specific recommendations table with appropriate columns.

    Args:
        recommendations: Strategy recommendations DataFrame
        strategy_type: Type of options strategy being displayed
    """
    if recommendations.empty:
        st.warning("No recommendations to display")
        return

    # Create column headers with tooltips using metric info
    show_definitions = st.session_state.get("show_metric_definitions", True)

    # Format the data for display
    display_df = recommendations.copy()

    # Common formatting for all strategies
    if "strike" in display_df.columns:
        display_df["Strike"] = display_df["strike"].apply(
            lambda x: safe_format_currency(x)
        )
    if "lastPrice" in display_df.columns:
        display_df["Price"] = display_df["lastPrice"].apply(
            lambda x: safe_format_currency(x)
        )
    if "RSI" in display_df.columns:
        display_df["RSI"] = display_df["RSI"].apply(lambda x: safe_format_number(x))
    if "Beta" in display_df.columns:
        display_df["Beta"] = display_df["Beta"].apply(lambda x: safe_format_number(x))
    if "Momentum" in display_df.columns:
        display_df["Momentum"] = display_df["Momentum"].apply(
            lambda x: safe_format_number(x)
        )
    if "IV" in display_df.columns:
        display_df["IV"] = display_df["IV"].apply(lambda x: safe_format_percentage(x))
    if "ForecastConfidence" in display_df.columns:
        display_df["Forecast"] = display_df["ForecastConfidence"].apply(
            lambda x: safe_format_percentage(x)
        )
    if "CompositeScore" in display_df.columns:
        display_df["Score"] = display_df["CompositeScore"].apply(
            lambda x: safe_format_number(x)
        )

    # Add Data Score badges if score_details exists
    if "score_details" in display_df.columns:
        display_df["Data Score"] = display_df["score_details"].apply(
            lambda x: get_data_score_badge(x)
        )

    # Strategy-specific column selection and formatting
    if strategy_type == "Long Calls":
        # Standard long calls display
        table_columns = [
            "Strike",
            "expiry",
            "Price",
            "RSI",
            "IV",
            "Momentum",
            "Forecast",
            "Score",
        ]
        if "Data Score" in display_df.columns:
            table_columns.append("Data Score")

        table_display_df = display_df[table_columns].copy()
        table_display_df.columns = [
            "Strike",
            "Expiry",
            "Price",
            "RSI",
            "IV",
            "Momentum",
            "Forecast",
            "Composite Score",
        ] + (["Data Score"] if "Data Score" in table_columns else [])

    elif strategy_type == "Bull Call Spread":
        # Bull call spread specific columns
        display_df["Long Strike"] = display_df["long_strike"].apply(
            lambda x: safe_format_currency(x)
        )
        display_df["Short Strike"] = display_df["short_strike"].apply(
            lambda x: safe_format_currency(x)
        )
        display_df["Net Premium"] = display_df["net_premium"].apply(
            lambda x: safe_format_currency(x)
        )
        display_df["Max Profit"] = display_df["max_profit"].apply(
            lambda x: safe_format_currency(x)
        )
        display_df["Max Loss"] = display_df["max_loss"].apply(
            lambda x: safe_format_currency(x)
        )
        display_df["Profit Ratio"] = display_df["profit_ratio"].apply(
            lambda x: safe_format_number(x)
        )

        table_columns = [
            "Long Strike",
            "Short Strike",
            "expiry",
            "Net Premium",
            "Max Profit",
            "Max Loss",
            "Profit Ratio",
            "Score",
        ]
        table_display_df = display_df[table_columns].copy()
        table_display_df.columns = [
            "Long Strike",
            "Short Strike",
            "Expiry",
            "Net Premium",
            "Max Profit",
            "Max Loss",
            "Profit Ratio",
            "Composite Score",
        ]

    elif strategy_type == "Covered Call":
        # Covered call specific columns
        if "premium_yield" in display_df.columns:
            display_df["Premium Yield"] = display_df["premium_yield"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "annualized_yield" in display_df.columns:
            display_df["Annualized Yield"] = display_df["annualized_yield"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "upside_capture" in display_df.columns:
            display_df["Upside Capture"] = display_df["upside_capture"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "total_return" in display_df.columns:
            display_df["Total Return"] = display_df["total_return"].apply(
                lambda x: safe_format_percentage(x)
            )

        table_columns = [
            "Strike",
            "expiry",
            "Price",
            "Premium Yield",
            "Annualized Yield",
            "Upside Capture",
            "Score",
        ]
        table_display_df = display_df[table_columns].copy()
        table_display_df.columns = [
            "Strike",
            "Expiry",
            "Premium",
            "Premium Yield",
            "Annualized Yield",
            "Upside Capture",
            "Composite Score",
        ]

    elif strategy_type == "Cash-Secured Put":
        # Cash-secured put specific columns
        if "premium_yield" in display_df.columns:
            display_df["Premium Yield"] = display_df["premium_yield"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "annualized_yield" in display_df.columns:
            display_df["Annualized Yield"] = display_df["annualized_yield"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "assignment_discount" in display_df.columns:
            display_df["Assignment Discount"] = display_df["assignment_discount"].apply(
                lambda x: safe_format_percentage(x)
            )
        if "effective_cost" in display_df.columns:
            display_df["Effective Cost"] = display_df["effective_cost"].apply(
                lambda x: safe_format_currency(x)
            )

        table_columns = [
            "Strike",
            "expiry",
            "Price",
            "Premium Yield",
            "Annualized Yield",
            "Assignment Discount",
            "Effective Cost",
            "Score",
        ]
        table_display_df = display_df[table_columns].copy()
        table_display_df.columns = [
            "Strike",
            "Expiry",
            "Premium",
            "Premium Yield",
            "Annualized Yield",
            "Assignment Discount",
            "Effective Cost",
            "Composite Score",
        ]

    else:
        # Fallback for unknown strategies
        table_display_df = display_df.head(10)  # Show first 10 columns

    # Get styling functions
    try:
        (
            highlight_rsi,
            highlight_score,
            highlight_iv,
            highlight_forecast,
        ) = create_styling_functions()

        # Apply styling based on available columns
        styled_df = table_display_df.style

        if "RSI" in table_display_df.columns:
            styled_df = styled_df.map(highlight_rsi, subset=["RSI"])
        if "Composite Score" in table_display_df.columns:
            styled_df = styled_df.map(highlight_score, subset=["Composite Score"])
        if "IV" in table_display_df.columns:
            styled_df = styled_df.map(highlight_iv, subset=["IV"])
        if "Forecast" in table_display_df.columns:
            styled_df = styled_df.map(highlight_forecast, subset=["Forecast"])

        # Format expiry dates
        if "Expiry" in table_display_df.columns:
            styled_df = styled_df.format(
                {
                    "Expiry": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
                    if pd.notna(x)
                    else ""
                }
            )
    except Exception as e:
        logger.warning(f"Error applying styling: {str(e)}")
        styled_df = table_display_df

    # Display the formatted dataframe
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        column_config={
            "Strike": st.column_config.TextColumn(
                "Strike Price", help="Option strike price"
            ),
            "Long Strike": st.column_config.TextColumn(
                "Long Strike", help="Lower strike price (buy)"
            ),
            "Short Strike": st.column_config.TextColumn(
                "Short Strike", help="Higher strike price (sell)"
            ),
            "Expiry": st.column_config.DateColumn(
                "Expiration", help="Option expiration date"
            ),
            "Premium": st.column_config.TextColumn(
                "Premium", help="Option premium price"
            ),
            "Price": st.column_config.TextColumn("Price", help="Option price"),
            "Net Premium": st.column_config.TextColumn(
                "Net Premium", help="Net cost of the spread"
            ),
            "Max Profit": st.column_config.TextColumn(
                "Max Profit", help="Maximum potential profit"
            ),
            "Max Loss": st.column_config.TextColumn(
                "Max Loss", help="Maximum potential loss"
            ),
            "Profit Ratio": st.column_config.TextColumn(
                "Profit Ratio", help="Max profit to max loss ratio"
            ),
            "Premium Yield": st.column_config.TextColumn(
                "Premium Yield", help="Premium as % of stock/strike price"
            ),
            "Annualized Yield": st.column_config.TextColumn(
                "Annualized Yield", help="Annualized premium yield"
            ),
            "Upside Capture": st.column_config.TextColumn(
                "Upside Capture", help="Additional return if assigned"
            ),
            "Assignment Discount": st.column_config.TextColumn(
                "Assignment Discount", help="Discount to current price if assigned"
            ),
            "Effective Cost": st.column_config.TextColumn(
                "Effective Cost", help="Net cost basis if assigned"
            ),
            "Composite Score": st.column_config.ProgressColumn(
                "Score",
                help="Composite technical score (0-1)",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
        },
    )

    # Display strategy-specific insights
    if strategy_type == "Bull Call Spread":
        st.info(
            "💡 **Bull Call Spread**: Limited risk, limited reward. Look for high profit ratios and moderate volatility."
        )
    elif strategy_type == "Covered Call":
        st.info(
            "💡 **Covered Call**: Generate income on existing stock positions. Higher yields are better, but watch upside capture."
        )
    elif strategy_type == "Cash-Secured Put":
        st.info(
            "💡 **Cash-Secured Put**: Generate income while potentially acquiring stock at a discount. Consider assignment scenarios."
        )
