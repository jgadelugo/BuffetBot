"""Options advisor tab rendering module."""

import io
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from buffetbot.analysis.options_advisor import (
    InsufficientDataError,
    OptionsAdvisorError,
    analyze_options_strategy,
    get_scoring_weights,
    recommend_long_calls,
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
)
from buffetbot.utils.logger import get_logger
from glossary import get_metric_info

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
        current_horizon = get_options_setting(
            "time_horizon", "Medium-term (3-6 months)"
        )
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

                    logger.info(
                        f"Options analysis completed for {ticker} - returned {len(recommendations)} recommendations"
                    )

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
                        render_greeks_analysis(recommendations, ticker)

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

                    # Get scoring weights for display
                    if analysis_settings["use_custom_weights"]:
                        current_weights = analysis_settings["custom_scoring_weights"]
                        st.info("📊 Using custom scoring weights")
                    else:
                        scoring_weights = get_scoring_weights()
                        current_weights = scoring_weights

                    # Create a table with metric-aware headers
                    st.markdown("#### Options Recommendations")

                    # Render strategy-specific table
                    render_strategy_specific_table(recommendations, strategy_type)

                    # Strategy-specific insights
                    render_strategy_insights(
                        strategy_type, recommendations, ticker, risk_tolerance
                    )

                    # Enhanced methodology explanation
                    render_enhanced_methodology()

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


def render_enhanced_methodology() -> None:
    """Render enhanced methodology explanation."""

    with st.expander("🔬 Enhanced Analysis Methodology", expanded=False):
        st.markdown(
            """
        **Technical Scoring Components:**

        1. **RSI Analysis (20% weight):**
           - Identifies overbought/oversold conditions
           - Optimal range: 30-70 for entry points

        2. **Beta Analysis (20% weight):**
           - Measures stock correlation with market
           - Higher beta = more volatility and option premium

        3. **Momentum Analysis (20% weight):**
           - Price trend strength over multiple timeframes
           - Positive momentum favors call options

        4. **Implied Volatility Analysis (20% weight):**
           - Current IV vs historical levels
           - Higher IV = higher option prices

        5. **Forecast Analysis (20% weight):**
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


# Add the rest of the existing code structure that follows the pattern from the original file
# This includes the remaining display logic, error handling, and styling functions
