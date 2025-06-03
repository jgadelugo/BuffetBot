"""Options advisor tab rendering module."""

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from analysis.options_advisor import (
    InsufficientDataError,
    OptionsAdvisorError,
    get_scoring_weights,
    recommend_long_calls,
)
from dashboard.components.disclaimers import render_investment_disclaimer
from dashboard.components.forecast_panel import render_forecast_panel
from dashboard.components.metrics import display_metric_with_info
from dashboard.components.options_utils import (
    check_for_partial_data,
    create_styling_functions,
    get_data_score_badge,
    render_score_details_popover,
)
from dashboard.config.settings import get_dashboard_config
from dashboard.utils.formatters import (
    safe_format_currency,
    safe_format_number,
    safe_format_percentage,
)
from glossary import get_metric_info
from utils.logger import get_logger

logger = get_logger(__name__)


def render_options_advisor_tab() -> None:
    """Render the options advisor tab content."""
    config = get_dashboard_config()

    # Options Advisor
    st.header("🎯 Options Advisor")

    # Add informative description with tooltips
    st.markdown(
        """
    Analyze long-dated call options using comprehensive technical scoring.
    This tool combines RSI, Beta, Momentum, and Implied Volatility to recommend the best option contracts.
    """
    )

    # Add prominent options trading disclaimer
    render_investment_disclaimer("options")

    # Create input section
    st.subheader("📊 Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        options_ticker = (
            st.text_input(
                "Stock Ticker",
                value="AAPL",
                help="Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
            )
            .upper()
            .strip()
        )

    with col2:
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

    with col3:
        top_n = st.slider(
            "Number of Recommendations",
            min_value=config["min_top_n"],
            max_value=config["max_top_n"],
            value=config["default_top_n"],
            help="Number of top-ranked option recommendations to display",
        )

    # CSV download checkbox
    download_csv = st.checkbox(
        "📄 Include CSV Download",
        help="Generate a downloadable CSV file of the recommendations",
    )

    # Add Forecast Insight Panel
    st.markdown("---")

    # Render the forecast panel if we have a ticker
    forecast_data = None
    if options_ticker:
        forecast_data = render_forecast_panel(options_ticker)

    st.markdown("---")

    # Analyze button
    if st.button("🔍 Analyze Options", type="primary"):
        if not options_ticker:
            st.error("Please enter a valid ticker symbol")
        else:
            # Log the interaction
            logger.info(
                f"Options analysis requested for {options_ticker} - min_days={min_days}, top_n={top_n}"
            )

            # Create loading placeholder
            with st.spinner(f"🔄 Analyzing options for {options_ticker}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Update progress
                    status_text.text("Fetching options data...")
                    progress_bar.progress(25)

                    # Call the recommend_long_calls function
                    status_text.text("Computing technical indicators...")
                    progress_bar.progress(50)

                    recommendations = recommend_long_calls(
                        ticker=options_ticker, min_days=min_days, top_n=top_n
                    )

                    status_text.text("Calculating composite scores...")
                    progress_bar.progress(75)

                    if recommendations.empty:
                        progress_bar.progress(100)
                        status_text.empty()
                        st.warning(
                            f"⚠️ No options found for {options_ticker} with minimum {min_days} days to expiry"
                        )
                        logger.warning(
                            f"No options found for {options_ticker} with min_days={min_days}"
                        )
                    else:
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")

                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()

                        # Display success message
                        st.success(
                            f"✅ Found {len(recommendations)} option recommendations for {options_ticker}"
                        )
                        logger.info(
                            f"Options analysis completed for {options_ticker} - returned {len(recommendations)} recommendations"
                        )

                        # Display results section
                        st.subheader("📈 Top Option Recommendations")

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
                                f"UI Warning: Partial scoring data detected for {options_ticker} options analysis"
                            )

                        # Display forecast summary if available
                        if forecast_data:
                            st.markdown("#### 🎯 Forecast Context")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "🎯 Analyst Target",
                                    safe_format_currency(forecast_data["mean_target"]),
                                    help="Average analyst price target",
                                )

                            with col2:
                                st.metric(
                                    "🔒 Forecast Confidence",
                                    safe_format_percentage(forecast_data["confidence"]),
                                    help="Analyst consensus confidence score",
                                )

                            with col3:
                                st.metric(
                                    "👥 Analysts",
                                    str(forecast_data["num_analysts"]),
                                    help="Number of analysts providing targets",
                                )

                            st.markdown("---")

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
                                - 🔮 **Forecast**: Analyst consensus confidence

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
                                    "Avg IV",
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
                                            st.markdown(
                                                f"**{header}** ℹ️", help=help_text
                                            )
                                        else:
                                            metric_info = get_metric_info(metric_key)
                                            help_text = f"{metric_info['description']}"
                                            st.markdown(
                                                f"**{header}** ℹ️", help=help_text
                                            )
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

                        # Select and rename columns for display (excluding the expandable column for now)
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

                        # Apply styling (same as before but include Data Score column)
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

                        # Display the styled dataframe
                        st.dataframe(
                            styled_df, use_container_width=True, hide_index=True
                        )

                        # Add expandable scoring details section
                        st.markdown("#### 📊 Detailed Scoring Breakdown")
                        st.markdown(
                            "Expand any row below to see the detailed scoring inputs and weights:"
                        )

                        for idx, row in display_df.iterrows():
                            strike = row["strike"]
                            expiry = row["expiry"]
                            score_details = row.get("score_details", {})

                            # Format expiry date
                            try:
                                expiry_formatted = pd.to_datetime(expiry).strftime(
                                    "%Y-%m-%d"
                                )
                            except:
                                expiry_formatted = str(expiry)

                            data_badge = get_data_score_badge(score_details)

                            with st.expander(
                                f"${strike:.2f} Strike • {expiry_formatted} • {data_badge}",
                                expanded=False,
                            ):
                                render_score_details_popover(score_details, idx)

                        # Add color legend
                        st.markdown(
                            """
                        **Color Legend:**
                        - 🟢 **Green**: Favorable values (Low RSI/IV, High Score/Forecast Confidence)
                        - 🟠 **Orange**: Neutral/Moderate values
                        - 🔴 **Red**: Less favorable values (High RSI/IV, Low Score/Forecast Confidence)

                        **Data Score Legend:**
                        - 🟢 **5/5**: All indicators available (full scoring)
                        - 🟡 **3-4/5**: Most indicators available (good scoring)
                        - 🔴 **1-2/5**: Few indicators available (limited scoring)
                        """
                        )

                        # CSV Download functionality
                        if download_csv:
                            st.subheader("📄 Download Data")

                            # Prepare raw data for CSV
                            csv_df = recommendations.copy()
                            csv_df["analysis_date"] = datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            csv_df["parameters"] = f"min_days={min_days}, top_n={top_n}"

                            # Convert to CSV
                            csv_buffer = io.StringIO()
                            csv_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()

                            # Create download button
                            filename = f"{options_ticker}_options_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                help="Download the complete options analysis results as a CSV file",
                            )

                            logger.info(
                                f"CSV download prepared for {options_ticker} options analysis"
                            )

                            # Show preview of CSV data
                            with st.expander("📋 CSV Preview", expanded=False):
                                st.dataframe(csv_df.head(), use_container_width=True)

                except OptionsAdvisorError as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"⚠️ Options analysis error: {str(e)}")
                    logger.error(
                        f"Options analysis error for {options_ticker}: {str(e)}"
                    )

                    # Provide user-friendly suggestions
                    if "No long-dated call options found" in str(e):
                        st.info(
                            """
                        **Suggestions:**
                        - Try a different ticker symbol
                        - Reduce the minimum days to expiry
                        - Check if the stock has active options trading
                        """
                        )

                except InsufficientDataError as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning(f"📊 Insufficient data: {str(e)}")
                    logger.warning(f"Insufficient data for {options_ticker}: {str(e)}")

                    st.info(
                        """
                    **This might happen if:**
                    - The stock is newly listed
                    - Limited trading history available
                    - Market data is temporarily unavailable
                    """
                    )

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"🚨 Unexpected error occurred: {str(e)}")
                    logger.error(
                        f"Unexpected error in options analysis for {options_ticker}: {str(e)}",
                        exc_info=True,
                    )

                    st.info(
                        """
                    **If this error persists:**
                    - Try refreshing the page
                    - Check your internet connection
                    - Contact support if the issue continues
                    """
                    )

    # Add helpful information section
    st.markdown("---")
    st.subheader("💡 How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Technical Analysis Framework:**
        1. **RSI Analysis** - Identifies momentum conditions
        2. **Beta Calculation** - Measures market correlation
        3. **Price Momentum** - Evaluates trend strength
        4. **Implied Volatility** - Assesses option pricing
        5. **Analyst Forecasts** - Incorporates forward-looking sentiment
        """
        )

    with col2:
        st.markdown(
            """
        **Scoring Methodology:**
        - Each metric is normalized to 0-1 scale
        - Weighted composite score combines all factors
        - Analyst forecast confidence included in scoring
        - Time-scoped forecast filtering available
        - Higher scores indicate more attractive options
        - Results ranked by composite score
        """
        )

    # Add new forecast methodology section
    st.markdown("---")
    st.subheader("🧠 Forecast Analysis Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **📊 Forecast Metrics:**
        - Mean & Median Targets
        - High & Low Target Range
        - Standard Deviation
        - Number of Analysts
        - Confidence Score (0-100%)
        """
        )

    with col2:
        st.markdown(
            """
        **🗓️ Time-Scoped Analysis:**
        - All forecasts (default)
        - Last 1 month
        - Last 3 months
        - Last 6 months
        - Recency-adjusted confidence
        """
        )

    with col3:
        st.markdown(
            """
        **📈 Visualization:**
        - Target distribution charts
        - Forecast evolution trends
        - Confidence level indicators
        - Interactive time filtering
        """
        )

    # Add disclaimer
    st.markdown(
        """
    ---
    ⚠️ **Disclaimer**: This tool is for educational and research purposes only.
    Options trading involves significant risk and may not be suitable for all investors.
    Analyst forecasts are opinions and may not reflect actual future performance.
    Always consult with a qualified financial advisor before making investment decisions.
    """
    )
