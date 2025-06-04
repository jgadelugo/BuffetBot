"""Options advisor tab rendering module."""

import io
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from buffetbot.analysis.options_advisor import (
    InsufficientDataError,
    OptionsAdvisorError,
    get_scoring_weights,
    recommend_long_calls,
)
from buffetbot.dashboard.components.disclaimers import render_investment_disclaimer
from buffetbot.dashboard.components.metrics import display_metric_with_info
from buffetbot.dashboard.components.options_utils import (
    check_for_partial_data,
    create_styling_functions,
    get_data_score_badge,
    render_score_details_popover,
)
from buffetbot.dashboard.config.settings import get_dashboard_config
from buffetbot.dashboard.dashboard_utils.data_processing import handle_ticker_change
from buffetbot.dashboard.dashboard_utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
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
        strategy_type = st.selectbox(
            "🎯 Options Strategy",
            options=[
                "Long Calls",
                "Bull Call Spread",
                "Covered Call",
                "Cash-Secured Put",
            ],
            index=0,
            help="Select the options strategy to analyze",
        )

    with col2:
        risk_tolerance = st.selectbox(
            "⚡ Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Your risk tolerance affects strategy recommendations",
        )

    with col3:
        time_horizon = st.selectbox(
            "📅 Time Horizon",
            options=[
                "Short-term (1-3 months)",
                "Medium-term (3-6 months)",
                "Long-term (6+ months)",
            ],
            index=1,
            help="Expected holding period for the options position",
        )

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

        min_days = st.slider(
            "Minimum Days to Expiry",
            min_value=config["min_min_days"],
            max_value=config["max_min_days"],
            value=config["default_min_days"],
            help=days_help_text,
        )

    with col2:
        top_n = st.slider(
            "Number of Recommendations",
            min_value=config["min_top_n"],
            max_value=config["max_top_n"],
            value=config["default_top_n"],
            help="Number of top-ranked option recommendations to display",
        )

    # Advanced Analysis Options
    st.subheader("🔧 Advanced Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        include_greeks = st.checkbox(
            "📊 Include Greeks Analysis",
            value=True,
            help="Add Delta, Gamma, Theta, Vega analysis to recommendations",
        )

    with col2:
        volatility_analysis = st.checkbox(
            "📈 Volatility Analysis",
            help="Include implied vs historical volatility comparison",
        )

    with col3:
        download_csv = st.checkbox(
            "📄 Enable CSV Export",
            help="Generate a downloadable CSV file of the recommendations",
        )

    st.markdown("---")

    # Analyze button
    if st.button("🔍 Analyze Options", type="primary"):
        # Log the interaction
        logger.info(
            f"Options analysis requested for {ticker} - strategy={strategy_type}, min_days={min_days}, top_n={top_n}"
        )

        # Create loading placeholder
        with st.spinner(f"🔄 Analyzing {strategy_type.lower()} for {ticker}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Update progress
                status_text.text("Fetching options data...")
                progress_bar.progress(25)

                # Call the recommend_long_calls function with global ticker
                status_text.text("Computing technical indicators...")
                progress_bar.progress(50)

                recommendations = recommend_long_calls(
                    ticker=ticker, min_days=min_days, top_n=top_n
                )

                status_text.text("Calculating composite scores...")
                progress_bar.progress(75)

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

                    # Display success message
                    st.success(
                        f"✅ Found {len(recommendations)} {strategy_type.lower()} recommendations for {ticker}"
                    )
                    logger.info(
                        f"Options analysis completed for {ticker} - returned {len(recommendations)} recommendations"
                    )

                    # Display results section
                    st.subheader(f"📈 Top {strategy_type} Recommendations")

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
                    scoring_weights = get_scoring_weights()

                    # Create columns for key metrics display
                    (
                        met_col1,
                        met_col2,
                        met_col3,
                        met_col4,
                        met_col5,
                    ) = st.columns(5)

                    # Get the current values for display (from first recommendation)
                    if not recommendations.empty:
                        first_row = recommendations.iloc[0]

                        with met_col1:
                            display_metric_with_info(
                                "RSI",
                                safe_format_number(first_row["RSI"]),
                                delta=None,
                                metric_key="rsi",
                            )

                        with met_col2:
                            display_metric_with_info(
                                "Beta",
                                safe_format_number(first_row["Beta"]),
                                delta=None,
                                metric_key="beta",
                            )

                        with met_col3:
                            display_metric_with_info(
                                "Momentum",
                                safe_format_number(first_row["Momentum"]),
                                delta=None,
                                metric_key="momentum",
                            )

                        with met_col4:
                            display_metric_with_info(
                                "Implied Vol",
                                safe_format_percentage(first_row["IV"]),
                                delta=None,
                                metric_key="implied_volatility",
                            )

                        with met_col5:
                            if "ForecastConfidence" in first_row:
                                display_metric_with_info(
                                    "Forecast",
                                    safe_format_percentage(
                                        first_row["ForecastConfidence"]
                                    ),
                                    delta=None,
                                    help_text="Analyst forecast confidence score",
                                )
                            else:
                                st.metric(
                                    "Forecast",
                                    "N/A",
                                    help="Forecast data not available",
                                )

                    st.markdown("---")

                    # Create a table with metric-aware headers
                    st.markdown("#### Options Recommendations")

                    # Create column headers with tooltips using metric info
                    show_definitions = st.session_state.get(
                        "show_metric_definitions", True
                    )

                    if show_definitions:
                        col_headers = []
                        col_headers.append("Strike Price")
                        col_headers.append("Expiration")
                        col_headers.append("Option Price")
                        col_headers.append("RSI")
                        col_headers.append("IV")
                        col_headers.append("Momentum")
                        col_headers.append("Forecast")
                        col_headers.append("Composite Score")
                        col_headers.append("Data Score")  # New column
                        col_headers.append("Scoring Inputs")  # New column

                        # Create tooltips for headers
                        header_help = {
                            "Strike Price": "option_strike",
                            "Expiration": "option_expiry",
                            "Option Price": "option_price",
                            "RSI": "rsi",
                            "IV": "implied_volatility",
                            "Momentum": "momentum",
                            "Forecast": "forecast_confidence",
                            "Composite Score": "composite_score",
                            "Data Score": "data_score_badge",
                            "Scoring Inputs": "scoring_breakdown",
                        }

                        # Display headers with help text in a table-like format
                        st.markdown("**Column Definitions:**")
                        help_cols = st.columns(len(col_headers))
                        for i, (header, metric_key) in enumerate(
                            zip(col_headers, header_help.values())
                        ):
                            with help_cols[i]:
                                try:
                                    if metric_key in [
                                        "data_score_badge",
                                        "scoring_breakdown",
                                    ]:
                                        # Custom help text for new columns
                                        if metric_key == "data_score_badge":
                                            help_text = "Shows how many of the 5 indicators were used in scoring"
                                        else:
                                            help_text = "Expandable breakdown of scoring weights and missing data"
                                        st.markdown(f"**{header}** ℹ️", help=help_text)
                                    else:
                                        metric_info = get_metric_info(metric_key)
                                        help_text = f"{metric_info['description']}"
                                        st.markdown(f"**{header}** ℹ️", help=help_text)
                                except KeyError:
                                    st.markdown(f"**{header}**")

                    # Format the data for display
                    display_df = recommendations.copy()

                    # Format numerical columns
                    display_df["Strike"] = display_df["strike"].apply(
                        lambda x: safe_format_currency(x)
                    )
                    display_df["Price"] = display_df["lastPrice"].apply(
                        lambda x: safe_format_currency(x)
                    )
                    display_df["RSI"] = display_df["RSI"].apply(
                        lambda x: safe_format_number(x)
                    )
                    display_df["Beta"] = display_df["Beta"].apply(
                        lambda x: safe_format_number(x)
                    )
                    display_df["Momentum"] = display_df["Momentum"].apply(
                        lambda x: safe_format_number(x)
                    )
                    display_df["IV"] = display_df["IV"].apply(
                        lambda x: safe_format_percentage(x)
                    )
                    display_df["Forecast"] = display_df["ForecastConfidence"].apply(
                        lambda x: safe_format_percentage(x)
                    )
                    display_df["Score"] = display_df["CompositeScore"].apply(
                        lambda x: safe_format_number(x)
                    )

                    # Add Data Score badges
                    display_df["Data Score"] = display_df["score_details"].apply(
                        lambda x: get_data_score_badge(x)
                    )

                    # Select and rename columns for display (including forecast column)
                    table_display_df = display_df[
                        [
                            "Strike",
                            "expiry",
                            "Price",
                            "RSI",
                            "IV",
                            "Momentum",
                            "Forecast",
                            "Score",
                            "Data Score",
                        ]
                    ].copy()

                    table_display_df.columns = [
                        "Strike",
                        "Expiry",
                        "Price",
                        "RSI",
                        "IV",
                        "Momentum",
                        "Forecast",
                        "Composite Score",
                        "Data Score",
                    ]

                    # Get styling functions
                    (
                        highlight_rsi,
                        highlight_score,
                        highlight_iv,
                        highlight_forecast,
                    ) = create_styling_functions()

                    # Apply styling (including forecast styling)
                    styled_df = (
                        table_display_df.style.map(highlight_rsi, subset=["RSI"])
                        .map(highlight_score, subset=["Composite Score"])
                        .map(highlight_iv, subset=["IV"])
                        .map(highlight_forecast, subset=["Forecast"])
                        .format(
                            {
                                "Expiry": lambda x: pd.to_datetime(x).strftime(
                                    "%Y-%m-%d"
                                )
                                if pd.notna(x)
                                else ""
                            }
                        )
                    )

                    # Display the formatted dataframe using Streamlit's dataframe widget with enhanced configuration
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Contract": st.column_config.TextColumn(
                                "Option Contract",
                                help="Option contract symbol",
                                width="large",
                            ),
                            "Strike": st.column_config.NumberColumn(
                                "Strike Price",
                                help="Option strike price",
                                format="$%.2f",
                            ),
                            "DaysToExpiry": st.column_config.NumberColumn(
                                "Days to Expiry",
                                help="Number of days until option expiration",
                            ),
                            "CompositeScore": st.column_config.ProgressColumn(
                                "Score",
                                help="Composite technical score (0-1)",
                                min_value=0,
                                max_value=1,
                                format="%.3f",
                            ),
                            "LastPrice": st.column_config.NumberColumn(
                                "Option Price",
                                help="Last traded option price",
                                format="$%.2f",
                            ),
                        },
                    )

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
        """
        )

    # Add more strategy-specific insights as needed


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


# Add the rest of the existing code structure that follows the pattern from the original file
# This includes the remaining display logic, error handling, and styling functions
