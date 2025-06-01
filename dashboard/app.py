# Path setup must be first!
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Debug path issues
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: Path(__file__).parent = {Path(__file__).parent}")
print(f"DEBUG: Path(__file__).parent.parent = {Path(__file__).parent.parent}")

# Try to import, if it fails, add parent to path and try again
try:
    from utils.logger import setup_logging, get_logger
except ImportError as e:
    print(f"DEBUG: First import failed: {e}")
    # Add parent directory to path
    parent_path = str(Path(__file__).parent.parent.absolute())
    print(f"DEBUG: Adding to sys.path: {parent_path}")
    sys.path.insert(0, parent_path)
    print(f"DEBUG: sys.path[0] is now: {sys.path[0]}")
    
    # Check if utils directory exists
    utils_path = Path(parent_path) / "utils"
    print(f"DEBUG: utils directory exists: {utils_path.exists()}")
    if utils_path.exists():
        print(f"DEBUG: utils/__init__.py exists: {(utils_path / '__init__.py').exists()}")
        print(f"DEBUG: utils/logger.py exists: {(utils_path / 'logger.py').exists()}")
    
    try:
        from utils.logger import setup_logging, get_logger
        print("DEBUG: Second import successful!")
    except ImportError as e2:
        print(f"DEBUG: Second import also failed: {e2}")
        raise

# Import from BuffetBot modules
from data.fetcher import fetch_stock_data
from data.cleaner import clean_financial_data
from analysis.value_analysis import calculate_intrinsic_value
from analysis.health_analysis import analyze_financial_health
from analysis.growth_analysis import analyze_growth_metrics
from analysis.risk_analysis import analyze_risk_metrics
from recommend.recommender import generate_recommendation
from utils.data_report import DataCollectionReport

# Import glossary functions
from glossary import (
    GLOSSARY,
    get_metrics_by_category,
    search_metrics,
    MetricDefinition,
    get_metric_info
)

# Import new modular components using absolute imports from dashboard
from dashboard.components import (
    display_metrics_grid_enhanced,
    display_metric_with_status,
    create_comparison_table,
    create_progress_indicator
)
from dashboard.pages import (
    render_price_analysis_page,
    render_financial_health_page
)

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_metric_with_info(label: str, value: str, delta=None, metric_key: str = None, help_text: str = None):
    """Display a metric with optional glossary information.
    
    Args:
        label: The metric label
        value: The metric value
        delta: Optional delta value
        metric_key: Optional key to look up in glossary
        help_text: Optional custom help text (overrides glossary)
    """
    # Check if we should show definitions
    show_definitions = st.session_state.get('show_metric_definitions', True)
    
    # Get glossary info if available and definitions are enabled
    if show_definitions and metric_key and not help_text:
        try:
            metric_info = get_metric_info(metric_key)
            help_text = f"{metric_info['description']} Formula: {metric_info['formula']}"
        except KeyError:
            help_text = None
    elif not show_definitions:
        help_text = None
    
    # Display metric with help
    st.metric(label=label, value=value, delta=delta, help=help_text)


def display_table_with_info(df: pd.DataFrame, metric_keys: dict = None):
    """Display a table with help text for metrics.
    
    Args:
        df: DataFrame with 'Metric'/'Ratio' and 'Value' columns
        metric_keys: Optional dictionary mapping metric names to glossary keys
    """
    # Create a cleaner display without buttons
    st.markdown("---")
    
    # Determine the metric column name
    metric_col = 'Metric' if 'Metric' in df.columns else 'Ratio' if 'Ratio' in df.columns else None
    
    if metric_col is None:
        st.dataframe(df)  # Fallback to standard display
        return
    
    # Display each metric as a row with help text
    for idx, row in df.iterrows():
        metric_name = row[metric_col]
        metric_value = row['Value']
        metric_key = metric_keys.get(metric_name) if metric_keys else None
        
        # Get help text if available
        help_text = None
        if metric_key:
            try:
                metric_info = get_metric_info(metric_key)
                help_text = f"{metric_info['description']} Formula: {metric_info['formula']}"
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


def display_metrics_grid(metrics_dict: dict, cols: int = 3):
    """Display metrics in a grid layout with help text.
    
    Args:
        metrics_dict: Dictionary of metrics with structure:
            {
                'metric_name': {
                    'value': 'displayed value',
                    'metric_key': 'glossary key',
                    'delta': 'optional delta value'
                }
            }
        cols: Number of columns in the grid
    """
    # Create columns
    columns = st.columns(cols)
    
    # Display metrics
    for idx, (metric_name, metric_data) in enumerate(metrics_dict.items()):
        col_idx = idx % cols
        
        with columns[col_idx]:
            # Check if we should show definitions
            show_definitions = st.session_state.get('show_metric_definitions', True)
            
            # Get help text if available and definitions are enabled
            help_text = None
            if show_definitions and 'metric_key' in metric_data:
                try:
                    metric_info = get_metric_info(metric_data['metric_key'])
                    help_text = f"{metric_info['description']} Formula: {metric_info['formula']}"
                except KeyError:
                    pass
            
            # Display metric
            st.metric(
                label=metric_name,
                value=metric_data['value'],
                delta=metric_data.get('delta'),
                help=help_text
            )


# Cache stock data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(ticker: str, years: int = 5):
    """Fetch and process stock data with caching."""
    try:
        # Fetch raw data
        raw_data = fetch_stock_data(ticker, years)
        
        # Clean and process data
        cleaned_data = clean_financial_data({
            'income_stmt': raw_data['income_stmt'],
            'balance_sheet': raw_data['balance_sheet'],
            'cash_flow': raw_data['cash_flow']
        })
        
        # Add price data and fundamentals
        cleaned_data['price_data'] = raw_data['price_data']
        cleaned_data['fundamentals'] = raw_data['fundamentals']
        cleaned_data['metrics'] = raw_data['metrics']
        
        return cleaned_data
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def create_price_gauge(current_price: float, intrinsic_value: float) -> go.Figure:
    """Create a gauge chart for price comparison."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_price,
        title={'text': "Current Price vs Intrinsic Value"},
        gauge={
            'axis': {'range': [0, max(current_price, intrinsic_value) * 1.2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, intrinsic_value], 'color': "lightgray"},
                {'range': [intrinsic_value, intrinsic_value * 1.2], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': intrinsic_value
            }
        }
    ))
    return fig

def create_growth_chart(price_data: pd.DataFrame) -> go.Figure:
    """Create a growth chart with moving averages and Bollinger Bands."""
    try:
        # Calculate technical indicators
        df = price_data.copy()
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Price',
            line=dict(color='blue')
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='20-day MA',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            name='50-day MA',
            line=dict(color='green', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA200'],
            name='200-day MA',
            line=dict(color='red', dash='dash')
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dot'),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dot'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Price History with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating growth chart: {str(e)}")
        # Return a simple price chart if technical indicators fail
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            name='Price',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='Price History',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        return fig

def render_metric_card(key: str, metric: MetricDefinition):
    """Render a single metric as a styled card."""
    category_class = f"category-{metric['category']}"
    
    card_html = f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 1.2em; font-weight: bold; color: #1f77b4; margin-bottom: 10px;">{metric['name']}</div>
        <span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; margin-bottom: 10px; background-color: {'#d4edda' if metric['category'] == 'growth' else '#cce5ff' if metric['category'] == 'value' else '#fff3cd' if metric['category'] == 'health' else '#f8d7da'}; color: {'#155724' if metric['category'] == 'growth' else '#004085' if metric['category'] == 'value' else '#856404' if metric['category'] == 'health' else '#721c24'};">{metric['category'].upper()}</span>
        <div style="color: #495057; line-height: 1.6; margin: 10px 0;">{metric['description']}</div>
        <div style="margin-top: 15px;">
            <strong>Formula:</strong>
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; color: #212529;">{metric['formula']}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    """Main dashboard function."""
    st.title("Stock Analysis Dashboard")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
    
    # Add cache management section
    st.sidebar.markdown("---")
    st.sidebar.header("Cache Management")
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and refresh"):
        get_stock_info.clear()
        st.success("Cache cleared! Data will be refreshed.")
        st.rerun()
    
    # Add metric definitions toggle
    st.sidebar.markdown("---")
    st.sidebar.header("Display Settings")
    
    # Initialize session state for metric definitions
    if 'show_metric_definitions' not in st.session_state:
        st.session_state.show_metric_definitions = True
    
    # Toggle for metric definitions
    st.session_state.show_metric_definitions = st.sidebar.checkbox(
        "Show Metric Definitions",
        value=st.session_state.show_metric_definitions,
        help="Toggle to show/hide metric descriptions and formulas throughout the dashboard"
    )
    
    # Fetch and process data
    data = get_stock_info(ticker, years)
    
    if data is None:
        st.error(f"Could not fetch data for {ticker}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Price Analysis", "Financial Health", 
        "Growth Metrics", "Risk Analysis", "📚 Glossary"
    ])
    
    with tab1:
        # Display basic information
        st.header(f"{ticker} Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_metric_with_info(
                "Current Price",
                f"${data['price_data']['Close'].iloc[-1]:.2f}",
                f"{data['metrics']['price_change']:.1%}",
                metric_key='latest_price'
            )
            
        with col2:
            display_metric_with_info(
                "Market Cap",
                f"${data['fundamentals']['market_cap']:,.0f}",
                f"P/E: {data['fundamentals']['pe_ratio']:.1f}",
                metric_key='market_cap'
            )
            
        with col3:
            display_metric_with_info(
                "Volatility",
                f"{data['metrics']['volatility']:.1%}",
                f"RSI: {data['metrics']['rsi']:.1f}",
                metric_key='volatility'
            )
        
        # Add link to data collection report
        st.markdown("---")
        st.subheader("Data Quality")
        report = DataCollectionReport(data)
        report_data = report.get_report()
        quality_score = report_data.get('data_quality_score', 0)
        score_color = "green" if quality_score >= 80 else "orange" if quality_score >= 50 else "red"
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: {score_color}20; border-radius: 10px;'>
                    <h2 style='color: {score_color}; margin: 0;'>Data Quality Score</h2>
                    <h1 style='color: {score_color}; margin: 10px 0;'>{quality_score:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                ### Data Collection Report
                View detailed information about the collected data, including:
                - Data availability status
                - Missing columns and metrics
                - Data quality indicators
                - Impact on analysis
                - Recommendations for improvement
            """)
            if st.button("View Data Collection Report"):
                st.session_state['show_data_report'] = True
                st.rerun()
    
    with tab2:
        # Use the new enhanced Price Analysis page
        render_price_analysis_page(data, ticker)
    
    with tab3:
        # Use the new enhanced Financial Health page
        render_financial_health_page(data, ticker)
    
    with tab4:
        try:
            # Analyze growth metrics
            growth_metrics_result = analyze_growth_metrics(data)
            
            if growth_metrics_result:
                # Display growth metrics
                st.subheader("Growth Metrics")
                
                growth_metrics = {
                    'Revenue Growth': {
                        'value': f"{growth_metrics_result.get('revenue_growth', 0):.1%}",
                        'metric_key': 'revenue_growth'
                    },
                    'Earnings Growth': {
                        'value': f"{growth_metrics_result.get('earnings_growth', 0):.1%}",
                        'metric_key': 'earnings_growth'
                    },
                    'EPS Growth': {
                        'value': f"{growth_metrics_result.get('eps_growth', 0):.1%}",
                        'metric_key': 'eps_growth'
                    }
                }
                
                display_metrics_grid(growth_metrics, cols=3)
                
                # Display growth score if available
                if 'growth_score' in growth_metrics_result:
                    st.markdown("---")
                    display_metric_with_info(
                        "Growth Score",
                        f"{growth_metrics_result['growth_score']:.1f}",
                        "Overall Growth Assessment",
                        metric_key='growth_score'
                    )
            else:
                st.warning("Could not calculate growth metrics. Some required financial data may be missing.")
        except Exception as e:
            logger.error(f"Error in growth metrics analysis: {str(e)}")
            st.error(f"Error in growth metrics analysis: {str(e)}")
        
    with tab5:
        try:
            # Analyze risk metrics
            risk_metrics_result = analyze_risk_metrics(data)
            
            if risk_metrics_result:
                # Display risk metrics
                st.subheader("Risk Metrics")
                
                # Check if we have overall risk data
                if 'overall_risk' in risk_metrics_result and risk_metrics_result['overall_risk']:
                    # Overall risk score with color coding
                    risk_score = risk_metrics_result['overall_risk'].get('score', 0)
                    risk_level = risk_metrics_result['overall_risk'].get('level', 'Unknown')
                    
                    # Log for debugging
                    logger.info(f"Risk Analysis for {ticker}: Score={risk_score:.2f}%, Level={risk_level}")
                    
                    # Create columns for risk score and level
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk score gauge
                        display_metric_with_info(
                            "Risk Score",
                            f"{risk_score:.1f}%",
                            delta=None,
                            metric_key='overall_risk_score'
                        )
                        
                        # Color-coded risk level
                        if risk_level == "High":
                            st.error(f"Risk Level: {risk_level}")
                        elif risk_level == "Moderate":
                            st.warning(f"Risk Level: {risk_level}")
                        elif risk_level == "Low":
                            st.success(f"Risk Level: {risk_level}")
                        else:
                            st.info(f"Risk Level: {risk_level}")
                    
                    with col2:
                        # Risk factors
                        st.write("Risk Factors:")
                        factors = risk_metrics_result['overall_risk'].get('factors', [])
                        if factors:
                            for factor in factors[:5]:  # Show first 5 factors
                                st.write(f"• {factor}")
                            if len(factors) > 5:
                                with st.expander(f"Show all {len(factors)} factors"):
                                    for factor in factors[5:]:
                                        st.write(f"• {factor}")
                        else:
                            st.write("• No specific risk factors identified")
                    
                    # Display warnings and errors if any
                    warnings = risk_metrics_result['overall_risk'].get('warnings', [])
                    if warnings:
                        with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                            for warning in warnings:
                                st.warning(warning)
                    
                    errors = risk_metrics_result['overall_risk'].get('errors', [])
                    if errors:
                        with st.expander(f"❌ Errors ({len(errors)})", expanded=True):
                            for error in errors:
                                st.error(error)
                    
                    # Add data availability check
                    st.markdown("---")
                    st.subheader("Data Availability Check")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'price_data' in data and data['price_data'] is not None and not data['price_data'].empty:
                            st.success("✓ Price Data Available")
                        else:
                            st.error("✗ Price Data Missing")
                    
                    with col2:
                        if 'fundamentals' in data and data['fundamentals'] and 'beta' in data['fundamentals']:
                            beta_val = data['fundamentals']['beta']
                            if beta_val is not None:
                                st.success(f"✓ Beta Available ({beta_val:.3f})")
                            else:
                                st.warning("⚠ Beta is null")
                        else:
                            st.error("✗ Beta Missing")
                    
                    with col3:
                        if 'income_stmt' in data and data['income_stmt'] is not None and not data['income_stmt'].empty:
                            st.success("✓ Financial Data Available")
                        else:
                            st.error("✗ Financial Data Missing")
                            
                else:
                    st.warning("Overall risk assessment data is not available")
                
                # Market Risk
                st.subheader("Market Risk")
                market_risk = risk_metrics_result.get('market_risk', {})
                if market_risk and any(market_risk.values()):
                    market_metrics = {}
                    
                    if 'beta' in market_risk and market_risk['beta'] is not None:
                        market_metrics['Beta'] = {
                            'value': f"{market_risk['beta']:.2f}",
                            'metric_key': 'beta'
                        }
                    
                    if 'volatility' in market_risk and market_risk['volatility'] is not None:
                        market_metrics['Annualized Volatility'] = {
                            'value': f"{market_risk['volatility']:.1%}",
                            'metric_key': 'volatility'
                        }
                    
                    if market_metrics:
                        display_metrics_grid(market_metrics, cols=2)
                    else:
                        st.info("Market risk metrics are not available or have zero values")
                else:
                    st.info("No market risk metrics available")
                
                # Financial Risk
                st.subheader("Financial Risk")
                financial_risk = risk_metrics_result.get('financial_risk', {})
                if financial_risk and any(financial_risk.values()):
                    financial_metrics = {}
                    
                    if 'debt_to_equity' in financial_risk and financial_risk['debt_to_equity'] is not None:
                        financial_metrics['Debt to Equity'] = {
                            'value': f"{financial_risk['debt_to_equity']:.2f}",
                            'metric_key': 'debt_to_equity'
                        }
                    
                    if 'interest_coverage' in financial_risk and financial_risk['interest_coverage'] is not None:
                        financial_metrics['Interest Coverage'] = {
                            'value': f"{financial_risk['interest_coverage']:.2f}",
                            'metric_key': 'interest_coverage'
                        }
                    
                    if financial_metrics:
                        display_metrics_grid(financial_metrics, cols=2)
                    else:
                        st.info("Financial risk metrics are not available or have zero values")
                else:
                    st.info("No financial risk metrics available")
                
                # Business Risk
                st.subheader("Business Risk")
                business_risk = risk_metrics_result.get('business_risk', {})
                if business_risk and any(business_risk.values()):
                    business_metrics = {}
                    
                    if 'operating_margin' in business_risk and business_risk['operating_margin'] is not None:
                        business_metrics['Operating Margin'] = {
                            'value': f"{business_risk['operating_margin']:.1%}",
                            'metric_key': 'operating_margin'
                        }
                    
                    if 'revenue' in business_risk and business_risk['revenue'] is not None and business_risk['revenue'] > 0:
                        business_metrics['Revenue'] = {
                            'value': f"${business_risk['revenue']:,.0f}",
                            'metric_key': 'revenue'
                        }
                    
                    if business_metrics:
                        display_metrics_grid(business_metrics, cols=2)
                    else:
                        st.info("Business risk metrics are not available or have zero values")
                else:
                    st.info("No business risk metrics available")
            else:
                st.warning("Could not calculate risk metrics. Some required data may be missing.")
                st.info("Try clearing the cache using the button in the sidebar and refreshing the data.")
        except Exception as e:
            logger.error(f"Error in risk metrics analysis: {str(e)}", exc_info=True)
            st.error(f"Error in risk metrics analysis: {str(e)}")
            st.info("Try clearing the cache using the button in the sidebar and refreshing the data.")

    with tab6:
        # Glossary header
        st.header("📚 Financial Metrics Glossary")
        st.markdown("Comprehensive guide to financial metrics used in this analysis")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Search and filter controls
            st.subheader("🔍 Search & Filter")
            
            # Search box
            search_term = st.text_input(
                "Search metrics",
                placeholder="Enter term...",
                key="glossary_search"
            )
            
            # Category filter
            st.subheader("Categories")
            categories = ["All", "Growth", "Value", "Health", "Risk"]
            
            # Use session state for selected category
            if 'glossary_category' not in st.session_state:
                st.session_state.glossary_category = "All"
            
            selected_category = st.radio(
                "Filter by category",
                categories,
                index=categories.index(st.session_state.glossary_category),
                key="glossary_category_radio"
            )
            
            # Quick stats
            st.subheader("📊 Statistics")
            total_metrics = len(GLOSSARY)
            st.metric("Total Metrics", total_metrics)
            
            # Category counts
            for cat in ["growth", "value", "health", "risk"]:
                count = len(get_metrics_by_category(cat))
                st.caption(f"{cat.title()}: {count}")
        
        with col2:
            # Apply filters
            if search_term:
                filtered_metrics = search_metrics(search_term)
                st.caption(f"Found {len(filtered_metrics)} metrics matching '{search_term}'")
            else:
                if selected_category == "All":
                    filtered_metrics = GLOSSARY
                else:
                    filtered_metrics = get_metrics_by_category(selected_category.lower())
            
            # Display metrics
            if filtered_metrics:
                # Group by category if showing all
                if not search_term and selected_category == "All":
                    for category in ["growth", "value", "health", "risk"]:
                        category_metrics = {k: v for k, v in filtered_metrics.items() 
                                          if v["category"] == category}
                        
                        if category_metrics:
                            # Category header
                            emoji_map = {
                                "growth": "📈",
                                "value": "💰", 
                                "health": "💪",
                                "risk": "⚠️"
                            }
                            
                            with st.expander(
                                f"{emoji_map.get(category, '📊')} {category.upper()} METRICS ({len(category_metrics)} items)",
                                expanded=True
                            ):
                                for key, metric in category_metrics.items():
                                    render_metric_card(key, metric)
                else:
                    # Display filtered results without grouping
                    for key, metric in filtered_metrics.items():
                        render_metric_card(key, metric)
            else:
                st.info("No metrics found matching your criteria.")
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")
            
            # Prepare data for export
            export_data = []
            for key, metric in GLOSSARY.items():
                export_data.append({
                    "Key": key,
                    "Name": metric["name"],
                    "Category": metric["category"],
                    "Description": metric["description"],
                    "Formula": metric["formula"]
                })
            
            df = pd.DataFrame(export_data)
            
            col1_export, col2_export = st.columns(2)
            
            with col1_export:
                # CSV download
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="financial_metrics_glossary.csv",
                    mime="text/csv"
                )
            
            with col2_export:
                # JSON download
                json_str = json.dumps(GLOSSARY, indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name="financial_metrics_glossary.json",
                    mime="application/json"
                )

    # Check if we should show the data collection report
    if st.session_state.get('show_data_report', False):
        st.title("Data Collection Report")
        
        # Add back button
        if st.button("← Back to Dashboard"):
            st.session_state.show_data_report = False
            st.rerun()
        
        # Display data quality score
        quality_score = report_data.get('data_quality_score', 0)
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: {
                '#4CAF50' if quality_score >= 80 else
                '#FFA500' if quality_score >= 50 else
                '#FF5252'
            }; color: white; border-radius: 10px;'>
                <h2>Data Quality Score: {quality_score:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Display validation results
        st.subheader("Data Validation")
        validation = report_data.get('data_validation', {})
        
        for statement, status in validation.items():
            with st.expander(f"{statement.replace('_', ' ').title()} Validation"):
                if status['is_valid']:
                    st.success("✓ Data structure is valid")
                else:
                    st.error("✗ Data structure has issues")
                
                if status['errors']:
                    st.error("Errors:")
                    for error in status['errors']:
                        st.write(f"- {error}")
                
                if status['warnings']:
                    st.warning("Warnings:")
                    for warning in status['warnings']:
                        st.write(f"- {warning}")
        
        # Display data availability
        st.subheader("Data Availability")
        availability = report_data.get('data_availability', {})
        
        for statement, status in availability.items():
            with st.expander(f"{statement.replace('_', ' ').title()} Availability"):
                if status['available']:
                    st.success("✓ Data is available")
                    st.write(f"Completeness: {status['completeness']:.1f}%")
                    st.write(f"Last available date: {status['last_available_date']}")
                    
                    if status['missing_columns']:
                        st.warning("Missing columns:")
                        for col in status['missing_columns']:
                            st.write(f"- {col}")
                    
                    if status['data_quality_issues']:
                        st.warning("Data quality issues:")
                        for issue in status['data_quality_issues']:
                            st.write(f"- {issue}")
                else:
                    st.error("✗ Data is not available")
                    if 'collection_status' in status:
                        st.error(f"Error: {status['collection_status']['error']}")
                        st.error(f"Reason: {status['collection_status']['reason']}")
                        if 'details' in status['collection_status']:
                            st.error(f"Details: {status['collection_status']['details']}")
        
        # Display impact analysis
        st.subheader("Impact Analysis")
        impact = report_data.get('impact_analysis', {})
        
        for category, metrics in impact.items():
            with st.expander(f"{category.replace('_', ' ').title()} Impact"):
                if metrics:
                    st.warning(f"The following {category} metrics are affected:")
                    for metric in metrics:
                        st.write(f"- {metric}")
                else:
                    st.success(f"No {category} metrics are affected")
        
        # Display recommendations
        st.subheader("Recommendations")
        recommendations = report_data.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div style='padding: 10px; margin: 5px 0; background-color: #f0f2f6; border-radius: 5px;'>
                        <strong>Recommendation {i}:</strong> {rec}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No recommendations - all required data is available and valid")

if __name__ == "__main__":
    main() 