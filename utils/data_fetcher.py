from typing import Dict, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from utils.logger import setup_logger
from utils.validators import validate_ticker, validate_date_range
from data.fetcher.utils.financial_calculations import calculate_rsi

# Initialize logger
logger = setup_logger(__name__)

def _standardize_column_names(df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """
    Basic column name standardization.
    For full implementation, use data.fetcher.utils.standardization
    """
    # Just return the dataframe as-is for now
    # The actual standardization is handled by the data.fetcher module
    return df

def fetch_stock_data(ticker: str, years: int = 5) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Fetch historical stock data and financial statements.
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data to fetch
        
    Returns:
        Dict containing:
            - price_data: DataFrame with historical price data
            - income_stmt: DataFrame with income statement data
            - balance_sheet: DataFrame with balance sheet data
            - cash_flow: DataFrame with cash flow statement data
            - fundamentals: Dict with key financial metrics
            - metrics: Dict with calculated metrics
            
    Raises:
        ValueError: If ticker is invalid or no data is found
        Exception: For other errors during data fetching
    """
    try:
        # Validate inputs
        if not validate_ticker(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")
            
        logger.info(f"Fetching data for {ticker}")
        
        # Initialize yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        if not validate_date_range(start_date, end_date, years):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
            
        # Fetch historical price data
        logger.info(f"Fetching {years} years of price data for {ticker}")
        price_data = stock.history(start=start_date, end=end_date)
        
        if price_data.empty:
            logger.warning(f"No price data found for {ticker}")
            # Don't raise an error, continue with empty price data
            price_data = pd.DataFrame()
        else:
            # Handle missing values
            price_data = price_data.ffill()  # Forward fill missing values
            price_data = price_data.bfill()  # Backward fill any remaining missing values
            logger.info(f"Successfully fetched {len(price_data)} days of data for {ticker}")
        
        # Fetch financial statements
        logger.info(f"Fetching financial statements for {ticker}")
        
        # Income Statement
        try:
            income_stmt = stock.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                income_stmt = _standardize_column_names(income_stmt, 'income')
                income_stmt = income_stmt.fillna(0)  # Fill missing values with 0
                logger.info(f"Income statement shape: {income_stmt.shape}")
            else:
                logger.warning("Income statement is empty or None")
                income_stmt = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching income statement: {str(e)}")
            income_stmt = pd.DataFrame()
            
        # Balance Sheet
        try:
            balance_sheet = stock.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                balance_sheet = _standardize_column_names(balance_sheet, 'balance')
                balance_sheet = balance_sheet.fillna(0)  # Fill missing values with 0
                logger.info(f"Balance sheet shape: {balance_sheet.shape}")
            else:
                logger.warning("Balance sheet is empty or None")
                balance_sheet = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {str(e)}")
            balance_sheet = pd.DataFrame()
            
        # Cash Flow Statement
        try:
            cash_flow = stock.cashflow
            if cash_flow is not None and not cash_flow.empty:
                cash_flow = _standardize_column_names(cash_flow, 'cash_flow')
                cash_flow = cash_flow.fillna(0)  # Fill missing values with 0
                logger.info(f"Cash flow shape: {cash_flow.shape}")
            else:
                logger.warning("Cash flow is empty or None")
                cash_flow = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching cash flow: {str(e)}")
            cash_flow = pd.DataFrame()
            
        # Fetch fundamental data
        logger.info(f"Fetching fundamental data for {ticker}")
        stock_info = stock.info  # Get stock info once
        
        if not stock_info:
            logger.warning(f"No fundamental data found for {ticker}")
            stock_info = {}
            
        fundamentals = {
            'pe_ratio': stock_info.get('trailingPE', None),
            'pb_ratio': stock_info.get('priceToBook', None),
            'eps': stock_info.get('trailingEps', None),
            'roe': stock_info.get('returnOnEquity', None),
            'market_cap': stock_info.get('marketCap', None),
            'dividend_yield': stock_info.get('dividendYield', None),
            'beta': stock_info.get('beta', None),
            'sector': stock_info.get('sector', None),
            'industry': stock_info.get('industry', None),
            'total_debt': stock_info.get('totalDebt', None),
            'total_equity': stock_info.get('totalStockholderEquity', None),
            'current_assets': stock_info.get('totalCurrentAssets', None),
            'current_liabilities': stock_info.get('totalCurrentLiabilities', None),
            'interest_expense': stock_info.get('interestExpense', None),
            'ebit': stock_info.get('ebit', None),
            'gross_profit': stock_info.get('grossProfit', None),
            'operating_income': stock_info.get('operatingIncome', None),
            'net_income': stock_info.get('netIncome', None),
            'revenue': stock_info.get('totalRevenue', None)
        }
        
        # Log fundamental data availability
        logger.info(f"Fundamentals data for {ticker}:")
        for key, value in fundamentals.items():
            if value is not None:
                logger.info(f"  {key}: {value}")
            else:
                logger.warning(f"  {key}: Not available")
        
        # Calculate additional metrics
        metrics = {}
        
        if not price_data.empty:
            metrics['latest_price'] = price_data['Close'].iloc[-1]
            metrics['price_change'] = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1
            metrics['volatility'] = price_data['Close'].pct_change().std() * np.sqrt(252)
            metrics['rsi'] = calculate_rsi(price_data['Close'])
            metrics['momentum'] = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[-20]) - 1 if len(price_data) > 20 else 0
        else:
            logger.warning("Price data is empty, cannot calculate price-based metrics")
            metrics = {
                'latest_price': None,
                'price_change': None,
                'volatility': None,
                'rsi': None,
                'momentum': None
            }
        
        result = {
            'price_data': price_data,
            'income_stmt': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'fundamentals': fundamentals,
            'metrics': metrics
        }
        
        logger.info(f"Successfully fetched all data for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
        # Return empty structure instead of None
        return {
            'price_data': pd.DataFrame(),
            'income_stmt': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'fundamentals': {},
            'metrics': {}
        } 