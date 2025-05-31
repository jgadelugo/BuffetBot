from typing import Dict, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from utils.logger import setup_logger
from utils.validators import validate_financial_data

# Initialize logger
logger = setup_logger(__name__)

# Growth classification thresholds
GROWTH_THRESHOLDS = {
    'high': 0.15,    # 15% or higher
    'moderate': 0.05  # 5% or higher
}

def _normalize_column_name(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Find the actual column name from a list of possible names.
    
    Args:
        df: DataFrame to search in
        possible_names: List of possible column names
        
    Returns:
        str: Found column name or None
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        start_value: Initial value
        end_value: Final value
        periods: Number of periods
        
    Returns:
        float: CAGR as a decimal
    """
    try:
        logger.info(f"Calculating CAGR with start_value={start_value}, end_value={end_value}, periods={periods}")
        if start_value <= 0 or end_value <= 0:
            logger.warning(f"Non-positive values in CAGR calculation: start_value={start_value}, end_value={end_value}")
            return 0.0
        cagr = (end_value / start_value) ** (1 / periods) - 1
        logger.info(f"Calculated CAGR: {cagr:.2%}")
        return cagr
    except Exception as e:
        logger.error(f"Error calculating CAGR: {str(e)}")
        return 0.0

def compute_growth_trends(financials: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze year-over-year growth trends for key financial metrics.
    
    Args:
        financials: DataFrame with historical financial data
        
    Returns:
        Dict containing:
            - CAGR values for each metric
            - Average YOY growth rates
    """
    try:
        logger.info("Computing growth trends")
        
        # Normalize column names
        revenue_col = _normalize_column_name(financials, ['revenue', 'Revenue', 'Total Revenue'])
        net_income_col = _normalize_column_name(financials, ['net_income', 'Net Income', 'NetIncome'])
        eps_col = _normalize_column_name(financials, ['eps', 'EPS', 'Earnings Per Share'])
        fcf_col = _normalize_column_name(financials, ['free_cash_flow', 'Free Cash Flow', 'FCF'])
        
        if not revenue_col or not net_income_col:
            logger.error("Missing required columns for growth analysis")
            return {}
            
        results = {}
        
        # Calculate CAGR for each metric
        metrics = {
            'revenue': revenue_col,
            'net_income': net_income_col,
            'eps': eps_col,
            'fcf': fcf_col
        }
        
        for metric_name, col_name in metrics.items():
            if col_name and col_name in financials.columns:
                # Get start and end values
                start_value = financials[col_name].iloc[0]
                end_value = financials[col_name].iloc[-1]
                periods = len(financials) - 1
                
                # Calculate CAGR
                cagr = calculate_cagr(start_value, end_value, periods)
                results[f'{metric_name}_cagr'] = cagr
                
                # Calculate average YOY growth
                yoy_growth = financials[col_name].pct_change().dropna()
                avg_growth = yoy_growth.mean()
                results[f'{metric_name}_avg_yoy'] = avg_growth
                
                # Log growth metrics
                logger.info(f"{metric_name.title()} CAGR: {cagr:.2%}")
                logger.info(f"{metric_name.title()} Avg YOY Growth: {avg_growth:.2%}")
                
                # Check for growth stability
                growth_std = yoy_growth.std()
                if growth_std > 0.5:  # 50% standard deviation
                    logger.warning(f"High volatility in {metric_name} growth")
            else:
                logger.warning(f"Missing data for {metric_name}")
                
        return results
        
    except Exception as e:
        logger.error(f"Error computing growth trends: {str(e)}")
        return {}

def assess_growth_quality(growth_data: Dict[str, float]) -> Dict[str, Union[str, float]]:
    """
    Assess the quality and sustainability of growth metrics.
    
    Args:
        growth_data: Dictionary of growth metrics from compute_growth_trends
        
    Returns:
        Dict containing:
            - growth_score: 0-5 score
            - growth_classification: Text classification
            - growth_stability: Stability assessment
    """
    try:
        logger.info("Assessing growth quality")
        
        if not growth_data:
            logger.error("No growth data provided for assessment")
            return {
                'growth_score': 0,
                'growth_classification': 'Insufficient Data',
                'growth_stability': 'Unknown'
            }
            
        # Calculate overall growth score
        score = 0
        stability_flags = []
        
        # Assess revenue growth
        if 'revenue_cagr' in growth_data:
            revenue_cagr = growth_data['revenue_cagr']
            if revenue_cagr >= GROWTH_THRESHOLDS['high']:
                score += 2
            elif revenue_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 1
                
            # Check revenue stability
            if 'revenue_avg_yoy' in growth_data:
                revenue_std = abs(growth_data['revenue_avg_yoy'] - revenue_cagr)
                if revenue_std > 0.1:  # 10% deviation
                    stability_flags.append("Unstable revenue growth")
                    
        # Assess earnings growth
        if 'net_income_cagr' in growth_data:
            net_income_cagr = growth_data['net_income_cagr']
            if net_income_cagr >= GROWTH_THRESHOLDS['high']:
                score += 2
            elif net_income_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 1
                
        # Assess cash flow growth
        if 'fcf_cagr' in growth_data:
            fcf_cagr = growth_data['fcf_cagr']
            if fcf_cagr >= GROWTH_THRESHOLDS['high']:
                score += 1
            elif fcf_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 0.5
                
        # Determine growth classification
        if score >= 4:
            classification = "High Growth"
        elif score >= 2:
            classification = "Moderate Growth"
        else:
            classification = "Low/Negative Growth"
            
        # Determine stability
        if stability_flags:
            stability = "Unstable Growth"
        else:
            stability = "Stable Growth"
            
        logger.info(f"Growth assessment: {classification} (Score: {score})")
        if stability_flags:
            logger.warning(f"Growth stability issues: {', '.join(stability_flags)}")
            
        return {
            'growth_score': score,
            'growth_classification': classification,
            'growth_stability': stability,
            'stability_flags': stability_flags
        }
        
    except Exception as e:
        logger.error(f"Error assessing growth quality: {str(e)}")
        return {
            'growth_score': 0,
            'growth_classification': 'Error in Assessment',
            'growth_stability': 'Unknown'
        }

def analyze_growth(financials: pd.DataFrame) -> Dict[str, Union[float, str, List[str]]]:
    """
    Perform comprehensive growth analysis.
    
    Args:
        financials: DataFrame with historical financial data
        
    Returns:
        Dict containing:
            - Growth metrics (CAGR, YOY)
            - Growth quality assessment
            - Analysis metadata
    """
    try:
        logger.info("Starting comprehensive growth analysis")
        
        # Calculate growth trends
        growth_trends = compute_growth_trends(financials)
        
        # Assess growth quality
        growth_quality = assess_growth_quality(growth_trends)
        
        # Combine results
        analysis = {
            **growth_trends,
            **growth_quality,
            'analysis_date': datetime.now().isoformat(),
            'status': 'Success'
        }
        
        logger.info("Successfully completed growth analysis")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in growth analysis: {str(e)}")
        return {
            'growth_score': 0,
            'growth_classification': 'Error in Analysis',
            'growth_stability': 'Unknown',
            'analysis_date': datetime.now().isoformat(),
            'status': f'Error: {str(e)}'
        }

def analyze_growth_metrics(
    financials: Dict[str, Union[pd.DataFrame, Dict]],
    price_data: Optional[pd.DataFrame] = None
) -> Dict[str, Union[float, str]]:
    """
    Analyze growth metrics and trends.
    
    Args:
        financials: Dictionary containing financial statements and fundamentals
        price_data: Optional price history DataFrame
        
    Returns:
        Dictionary containing growth metrics and analysis
    """
    try:
        logger.info("Starting growth metrics analysis")
        
        # Extract required data
        income_stmt = financials.get('income_stmt')
        balance_sheet = financials.get('balance_sheet')
        cash_flow = financials.get('cash_flow')
        fundamentals = financials.get('fundamentals', {})
        
        if income_stmt is None or income_stmt.empty:
            logger.warning("No income statement data available")
            return {
                'revenue_growth': 0,
                'earnings_growth': 0,
                'eps_growth': 0,
                'growth_score': 0,
                'status': 'Error: No income statement data available'
            }
        
        # Initialize results
        results = {
            'revenue_growth': 0,
            'earnings_growth': 0,
            'eps_growth': 0,
            'growth_score': 0,
            'status': 'Success'
        }
        
        # Calculate growth metrics using helper functions
        results['revenue_growth'] = _calculate_revenue_growth(income_stmt)
        results['earnings_growth'] = _calculate_earnings_growth(income_stmt)
        results['eps_growth'] = _calculate_eps_growth(income_stmt, fundamentals)
        
        # Calculate overall growth score
        try:
            growth_metrics = [
                results['revenue_growth'],
                results['earnings_growth'],
                results['eps_growth']
            ]
            
            # Filter out zero values and calculate average
            valid_metrics = [m for m in growth_metrics if m != 0]
            if valid_metrics:
                results['growth_score'] = sum(valid_metrics) / len(valid_metrics)
                logger.info(f"Calculated growth score: {results['growth_score']:.2%}")
            else:
                logger.warning("All growth metrics are zero")
                results['growth_score'] = 0
        except Exception as e:
            logger.warning(f"Error calculating growth score: {str(e)}")
        
        logger.info("Successfully completed growth metrics analysis")
        return results
        
    except Exception as e:
        logger.error(f"Error in growth metrics analysis: {str(e)}")
        return {
            'revenue_growth': 0,
            'earnings_growth': 0,
            'eps_growth': 0,
            'growth_score': 0,
            'status': f'Error: {str(e)}'
        }

def _calculate_revenue_growth(income_stmt: pd.DataFrame) -> float:
    """Calculate year-over-year revenue growth."""
    try:
        # Check if Total Revenue is in columns
        if 'Total Revenue' in income_stmt.columns:
            revenue = income_stmt['Total Revenue']
            if len(revenue) >= 2:
                # Calculate YoY growth (most recent year vs previous year)
                growth = (revenue.iloc[0] - revenue.iloc[1]) / abs(revenue.iloc[1])
                logger.info(f"Calculated revenue growth: {growth:.2%}")
                return float(growth)
            else:
                logger.warning("Insufficient revenue data for growth calculation")
                return 0.0
        else:
            logger.warning("Total Revenue not found in income statement columns")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating revenue growth: {str(e)}")
        return 0.0

def _calculate_earnings_growth(income_stmt: pd.DataFrame) -> float:
    """Calculate year-over-year earnings growth."""
    try:
        if 'Net Income' in income_stmt.columns:
            earnings = income_stmt['Net Income']
            if len(earnings) >= 2:
                # Calculate YoY growth (most recent year vs previous year)
                growth = (earnings.iloc[0] - earnings.iloc[1]) / abs(earnings.iloc[1])
                logger.info(f"Calculated earnings growth: {growth:.2%}")
                return float(growth)
            else:
                logger.warning("Insufficient earnings data for growth calculation")
                return 0.0
        else:
            logger.warning("Net Income not found in income statement columns")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating earnings growth: {str(e)}")
        return 0.0

def _calculate_eps_growth(income_stmt: pd.DataFrame, fundamentals: Dict) -> float:
    """Calculate year-over-year EPS growth."""
    try:
        # Try EPS directly from income statement
        if 'EPS' in income_stmt.columns:
            eps = income_stmt['EPS']
            if len(eps) >= 2:
                growth = (eps.iloc[0] - eps.iloc[1]) / abs(eps.iloc[1])
                logger.info(f"Calculated EPS growth from EPS column: {growth:.2%}")
                return float(growth)
        
        # Calculate EPS from Net Income and Shares Outstanding
        if 'Net Income' in income_stmt.columns:
            net_income = income_stmt['Net Income']
            if 'Shares Outstanding' in income_stmt.columns:
                shares = income_stmt['Shares Outstanding']
                if len(net_income) >= 2 and len(shares) >= 2:
                    eps_current = net_income.iloc[0] / shares.iloc[0] if shares.iloc[0] != 0 else 0
                    eps_previous = net_income.iloc[1] / shares.iloc[1] if shares.iloc[1] != 0 else 0
                    if eps_previous != 0:
                        growth = (eps_current - eps_previous) / abs(eps_previous)
                        logger.info(f"Calculated EPS growth from Net Income/Shares: {growth:.2%}")
                        return float(growth)
        
        logger.warning("Could not calculate EPS growth")
        return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating EPS growth: {str(e)}")
        return 0.0

def _calculate_fcf_growth(cash_flow: pd.DataFrame) -> float:
    """Calculate year-over-year free cash flow growth."""
    try:
        if 'Free Cash Flow' in cash_flow.columns:
            fcf = cash_flow['Free Cash Flow']
        elif 'Free Cash Flow' in cash_flow.index:
            fcf = cash_flow.loc['Free Cash Flow']
        else:
            logger.warning("Free Cash Flow not found in cash flow statement")
            return 0.0
            
        if len(fcf) < 2:
            logger.warning("Insufficient FCF data for growth calculation")
            return 0.0
            
        growth = (fcf.iloc[0] - fcf.iloc[1]) / abs(fcf.iloc[1])
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating FCF growth: {str(e)}")
        return 0.0

def _calculate_asset_growth(balance_sheet: pd.DataFrame) -> float:
    """Calculate year-over-year total asset growth."""
    try:
        if 'Total Assets' in balance_sheet.columns:
            assets = balance_sheet['Total Assets']
        elif 'Total Assets' in balance_sheet.index:
            assets = balance_sheet.loc['Total Assets']
        else:
            logger.warning("Total Assets not found in balance sheet")
            return 0.0
            
        if len(assets) < 2:
            logger.warning("Insufficient asset data for growth calculation")
            return 0.0
            
        growth = (assets.iloc[0] - assets.iloc[1]) / abs(assets.iloc[1])
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating asset growth: {str(e)}")
        return 0.0

def _calculate_equity_growth(balance_sheet: pd.DataFrame) -> float:
    """Calculate year-over-year shareholder equity growth."""
    try:
        if 'Total Stockholder Equity' in balance_sheet.columns:
            equity = balance_sheet['Total Stockholder Equity']
        elif 'Total Stockholder Equity' in balance_sheet.index:
            equity = balance_sheet.loc['Total Stockholder Equity']
        else:
            logger.warning("Total Stockholder Equity not found in balance sheet")
            return 0.0
            
        if len(equity) < 2:
            logger.warning("Insufficient equity data for growth calculation")
            return 0.0
            
        growth = (equity.iloc[0] - equity.iloc[1]) / abs(equity.iloc[1])
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating equity growth: {str(e)}")
        return 0.0

def _calculate_growth_score(metrics: Dict[str, float]) -> float:
    """Calculate overall growth score (0-100)."""
    try:
        # Define weights for different growth metrics
        weights = {
            'revenue_growth': 0.3,
            'earnings_growth': 0.3,
            'eps_growth': 0.2,
            'fcf_growth': 0.1,
            'asset_growth': 0.05,
            'equity_growth': 0.05
        }
        
        # Calculate weighted score
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize growth rate to 0-100 scale
                growth = metrics[metric]
                normalized_score = min(max(growth * 100, 0), 100)
                score += normalized_score * weight
                total_weight += weight
                
        # Normalize final score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0
            
        return float(final_score)
        
    except Exception as e:
        logger.error(f"Error calculating growth score: {str(e)}")
        return 0.0

def _classify_growth(score: float) -> str:
    """Classify growth based on score."""
    if score >= 80:
        return "High Growth"
    elif score >= 50:
        return "Moderate Growth"
    else:
        return "Low Growth" 