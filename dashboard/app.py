import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.logger import setup_logging, get_logger

from data.fetcher import fetch_stock_data
from data.cleaner import clean_financial_data
from analysis.value_analysis import calculate_intrinsic_value
from analysis.health_analysis import analyze_financial_health
from analysis.growth_analysis import analyze_growth_metrics
from analysis.risk_analysis import analyze_risk_metrics
from recommend.recommender import generate_recommendation
from utils.data_report import DataCollectionReport

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

def main():
    """Main dashboard function."""
    st.title("Stock Analysis Dashboard")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)
    
    # Fetch and process data
    data = get_stock_info(ticker, years)
    
    if data is None:
        st.error(f"Could not fetch data for {ticker}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Price Analysis", "Financial Health", 
        "Growth Metrics", "Risk Analysis"
    ])
    
    with tab1:
        # Display basic information
        st.header(f"{ticker} Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${data['price_data']['Close'].iloc[-1]:.2f}",
                f"{data['metrics']['price_change']:.1%}"
            )
            
        with col2:
            st.metric(
                "Market Cap",
                f"${data['fundamentals']['market_cap']:,.0f}",
                f"P/E: {data['fundamentals']['pe_ratio']:.1f}"
            )
            
        with col3:
            st.metric(
                "Volatility",
                f"{data['metrics']['volatility']:.1%}",
                f"RSI: {data['metrics']['rsi']:.1f}"
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
        try:
            # Calculate intrinsic value
            intrinsic_value_result = calculate_intrinsic_value(data)
            
            if intrinsic_value_result and intrinsic_value_result.get('intrinsic_value') is not None:
                current_price = data['price_data']['Close'].iloc[-1]
                intrinsic_value = intrinsic_value_result['intrinsic_value']
                
                # Calculate margin of safety
                if intrinsic_value > 0:
                    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
                else:
                    margin_of_safety = None
                    logger.warning(f"Invalid intrinsic value: {intrinsic_value}")
                
                # Display price gauge
                st.plotly_chart(create_price_gauge(
                    current_price,
                    intrinsic_value
                ))
                
                # Display intrinsic value metrics
                st.subheader("Intrinsic Value Analysis")
                iv_metrics = pd.DataFrame({
                    'Metric': ['Intrinsic Value', 'Current Price', 'Margin of Safety', 'Growth Rate', 'Discount Rate'],
                    'Value': [
                        f"${intrinsic_value:.2f}",
                        f"${current_price:.2f}",
                        f"{margin_of_safety:.1%}" if margin_of_safety is not None else "N/A",
                        f"{intrinsic_value_result.get('assumptions', {}).get('growth_rate', 0):.1%}",
                        f"{intrinsic_value_result.get('assumptions', {}).get('discount_rate', 0):.1%}"
                    ]
                })
                st.table(iv_metrics)
                
                # Display warnings if any
                if intrinsic_value_result.get('warnings'):
                    with st.expander("⚠️ Warnings", expanded=True):
                        for warning in intrinsic_value_result['warnings']:
                            st.warning(warning)
            else:
                st.warning("Could not calculate intrinsic value. Some required financial data may be missing.")
                if intrinsic_value_result and 'errors' in intrinsic_value_result:
                    with st.expander("❌ Errors", expanded=True):
                        for error in intrinsic_value_result['errors']:
                            st.error(error)
        except Exception as e:
            logger.error(f"Error in intrinsic value calculation: {str(e)}", exc_info=True)
            st.error(f"Error in intrinsic value calculation: {str(e)}")
        
        # Display growth chart
        st.plotly_chart(create_growth_chart(data['price_data']))
        
        # Display price metrics
        st.subheader("Price Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Latest Price', 'Price Change', 'Volatility', 'RSI', 'Momentum'],
            'Value': [
                f"${data['metrics']['latest_price']:.2f}",
                f"{data['metrics']['price_change']:.1%}",
                f"{data['metrics']['volatility']:.1%}",
                f"{data['metrics']['rsi']:.1f}",
                f"{data['metrics']['momentum']:.1%}"
            ]
        })
        st.table(metrics_df)
    
    with tab3:
        try:
            # Analyze financial health
            health_metrics_result = analyze_financial_health(data)
            
            # Display health metrics
            st.subheader("Financial Health Metrics")
            
            # Create metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                if 'piotroski_score' in health_metrics_result:
                    st.metric(
                        "Piotroski F-Score",
                        f"{health_metrics_result['piotroski_score']}/9",
                        "Financial Health Indicator"
                    )
                if 'altman_z_score' in health_metrics_result:
                    st.metric(
                        "Altman Z-Score",
                        f"{health_metrics_result['altman_z_score']:.2f}",
                        "Bankruptcy Risk Indicator"
                    )
            
            with col2:
                # Display key financial ratios
                ratios = health_metrics_result.get('financial_ratios', {})
                st.write("Key Financial Ratios")
                ratios_df = pd.DataFrame({
                    'Ratio': [
                        'Current Ratio',
                        'Debt to Equity',
                        'Debt to Assets',
                        'Interest Coverage',
                        'Return on Equity',
                        'Return on Assets',
                        'Gross Margin',
                        'Operating Margin',
                        'Net Margin'
                    ],
                    'Value': [
                        f"{ratios.get('current_ratio', 'N/A')}",
                        f"{ratios.get('debt_to_equity', 'N/A')}",
                        f"{ratios.get('debt_to_assets', 'N/A')}",
                        f"{ratios.get('interest_coverage', 'N/A')}",
                        f"{ratios.get('return_on_equity', 'N/A')}",
                        f"{ratios.get('return_on_assets', 'N/A')}",
                        f"{ratios.get('gross_margin', 'N/A')}",
                        f"{ratios.get('operating_margin', 'N/A')}",
                        f"{ratios.get('net_margin', 'N/A')}"
                    ]
                })
                st.table(ratios_df)
            
            # Display health flags
            st.subheader("Health Indicators")
            for flag in health_metrics_result.get('health_flags', []):
                st.write(f"• {flag}")
        except Exception as e:
            logger.error(f"Error in financial health analysis: {str(e)}", exc_info=True)
            st.error(f"Error in financial health analysis: {str(e)}")
        
    with tab4:
        try:
            # Analyze growth metrics
            growth_metrics_result = analyze_growth_metrics(data)
            
            if growth_metrics_result:
                # Display growth metrics
                st.subheader("Growth Metrics")
                growth_df = pd.DataFrame({
                    'Metric': ['Revenue Growth', 'Earnings Growth', 'EPS Growth'],
                    'Value': [
                        f"{growth_metrics_result.get('revenue_growth', 0):.1%}",
                        f"{growth_metrics_result.get('earnings_growth', 0):.1%}",
                        f"{growth_metrics_result.get('eps_growth', 0):.1%}"
                    ]
                })
                st.table(growth_df)
                
                # Display growth score if available
                if 'growth_score' in growth_metrics_result:
                    st.metric(
                        "Growth Score",
                        f"{growth_metrics_result['growth_score']:.1f}",
                        "Overall Growth Assessment"
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
                
                # Overall risk score with color coding
                risk_score = risk_metrics_result['overall_risk']['score']
                risk_level = risk_metrics_result['overall_risk']['level']
                
                # Create columns for risk score and level
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk score gauge
                    st.metric(
                        "Risk Score",
                        f"{risk_score:.1f}%",
                        delta=None,
                        delta_color="off"
                    )
                    
                    # Color-coded risk level
                    if risk_level == "High":
                        st.error(f"Risk Level: {risk_level}")
                    elif risk_level == "Moderate":
                        st.warning(f"Risk Level: {risk_level}")
                    else:
                        st.success(f"Risk Level: {risk_level}")
                
                with col2:
                    # Risk factors
                    st.write("Risk Factors:")
                    for factor in risk_metrics_result['overall_risk']['factors']:
                        st.write(f"• {factor}")
                
                # Display warnings and errors if any
                if risk_metrics_result['overall_risk']['warnings']:
                    with st.expander("⚠️ Warnings", expanded=True):
                        for warning in risk_metrics_result['overall_risk']['warnings']:
                            st.warning(warning)
                
                if risk_metrics_result['overall_risk']['errors']:
                    with st.expander("❌ Errors", expanded=True):
                        for error in risk_metrics_result['overall_risk']['errors']:
                            st.error(error)
                
                # Market Risk
                st.subheader("Market Risk")
                market_risk = risk_metrics_result['market_risk']
                if market_risk:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'beta' in market_risk:
                            st.metric(
                                "Beta",
                                f"{market_risk['beta']:.2f}",
                                delta=None,
                                delta_color="off"
                            )
                    with col2:
                        if 'volatility' in market_risk:
                            st.metric(
                                "Annualized Volatility",
                                f"{market_risk['volatility']:.1%}",
                                delta=None,
                                delta_color="off"
                            )
                else:
                    st.info("No market risk metrics available")
                
                # Financial Risk
                st.subheader("Financial Risk")
                financial_risk = risk_metrics_result['financial_risk']
                if financial_risk:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'debt_to_equity' in financial_risk:
                            st.metric(
                                "Debt to Equity",
                                f"{financial_risk['debt_to_equity']:.2f}",
                                delta=None,
                                delta_color="off"
                            )
                    with col2:
                        if 'interest_coverage' in financial_risk:
                            st.metric(
                                "Interest Coverage",
                                f"{financial_risk['interest_coverage']:.2f}",
                                delta=None,
                                delta_color="off"
                            )
                else:
                    st.info("No financial risk metrics available")
                
                # Business Risk
                st.subheader("Business Risk")
                business_risk = risk_metrics_result['business_risk']
                if business_risk:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'operating_margin' in business_risk:
                            st.metric(
                                "Operating Margin",
                                f"{business_risk['operating_margin']:.1%}",
                                delta=None,
                                delta_color="off"
                            )
                    with col2:
                        if 'revenue' in business_risk:
                            st.metric(
                                "Revenue",
                                f"${business_risk['revenue']:,.0f}",
                                delta=None,
                                delta_color="off"
                            )
                else:
                    st.info("No business risk metrics available")
            else:
                st.warning("Could not calculate risk metrics. Some required data may be missing.")
        except Exception as e:
            logger.error(f"Error in risk metrics analysis: {str(e)}", exc_info=True)
            st.error(f"Error in risk metrics analysis: {str(e)}")
        
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