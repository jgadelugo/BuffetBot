from typing import Dict, List, Union, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from utils.logger import setup_logger
from utils.validators import validate_financial_data

# Initialize logger
logger = logging.getLogger(__name__)

class GrowthClassification(Enum):
    """Enumeration for growth classification levels."""
    HIGH = "High Growth"
    MODERATE = "Moderate Growth"
    LOW = "Low/Negative Growth"
    INSUFFICIENT = "Insufficient Data"
    ERROR = "Error in Assessment"

@dataclass
class GrowthMetrics:
    """Data class to hold growth metrics."""
    revenue_cagr: float
    net_income_cagr: float
    eps_cagr: Optional[float]
    fcf_cagr: Optional[float]
    revenue_avg_yoy: float
    net_income_avg_yoy: float
    eps_avg_yoy: Optional[float]
    fcf_avg_yoy: Optional[float]

# Growth classification thresholds
GROWTH_THRESHOLDS = {
    'high': 0.15,    # 15% or higher
    'moderate': 0.05  # 5% or higher
}

def _normalize_column_name(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find the actual column name from a list of possible names.
    
    Args:
        df: DataFrame to search in
        possible_names: List of possible column names to check
        
    Returns:
        Optional[str]: Found column name or None if no match found
    """
    for name in possible_names:
        if name in df.columns:
            return name
    logger.warning(f"No matching column found in {possible_names}")
    return None

def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        start_value: Initial value
        end_value: Final value
        periods: Number of periods between start and end
        
    Returns:
        float: CAGR as a decimal (e.g., 0.15 for 15% growth)
        
    Raises:
        ValueError: If periods is less than 1
    """
    if periods < 1:
        raise ValueError("Periods must be at least 1")
        
    try:
        logger.debug(f"Calculating CAGR: start={start_value}, end={end_value}, periods={periods}")
        
        if start_value <= 0 or end_value <= 0:
            logger.warning(f"Non-positive values in CAGR calculation: start={start_value}, end={end_value}")
            return 0.0
            
        cagr = (end_value / start_value) ** (1 / periods) - 1
        logger.debug(f"CAGR calculated: {cagr:.2%}")
        return cagr
        
    except Exception as e:
        logger.error(f"Error calculating CAGR: {str(e)}")
        return 0.0

def compute_growth_trends(financials: pd.DataFrame) -> Dict[str, float]:
    """Analyze year-over-year growth trends for key financial metrics.
    
    Args:
        financials: DataFrame with historical financial data, sorted by date
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - CAGR values for each metric (e.g., 'revenue_cagr')
            - Average YOY growth rates (e.g., 'revenue_avg_yoy')
            
    Raises:
        ValueError: If required columns are missing
    """
    try:
        logger.info("Computing growth trends from financial data")
        
        # Normalize column names
        column_mappings = {
            'revenue': ['revenue', 'Revenue', 'Total Revenue'],
            'net_income': ['net_income', 'Net Income', 'NetIncome'],
            'eps': ['eps', 'EPS', 'Earnings Per Share'],
            'fcf': ['free_cash_flow', 'Free Cash Flow', 'FCF']
        }
        
        normalized_columns = {
            metric: _normalize_column_name(financials, names)
            for metric, names in column_mappings.items()
        }
        
        # Validate required columns
        if not normalized_columns['revenue'] or not normalized_columns['net_income']:
            raise ValueError("Missing required columns for growth analysis")
            
        results: Dict[str, float] = {}
        
        for metric_name, col_name in normalized_columns.items():
            if col_name and col_name in financials.columns:
                # Calculate CAGR
                start_value = financials[col_name].iloc[0]
                end_value = financials[col_name].iloc[-1]
                periods = len(financials) - 1
                
                cagr = calculate_cagr(start_value, end_value, periods)
                results[f'{metric_name}_cagr'] = cagr
                
                # Calculate YOY growth
                yoy_growth = financials[col_name].pct_change().dropna()
                avg_growth = yoy_growth.mean()
                results[f'{metric_name}_avg_yoy'] = avg_growth
                
                # Log metrics
                logger.info(f"{metric_name.title()} - CAGR: {cagr:.2%}, Avg YOY: {avg_growth:.2%}")
                
                # Check growth stability
                growth_std = yoy_growth.std()
                if growth_std > 0.5:  # 50% standard deviation
                    logger.warning(f"High volatility in {metric_name} growth (std: {growth_std:.2%})")
            else:
                logger.warning(f"Missing data for {metric_name}")
                
        return results
        
    except Exception as e:
        logger.error(f"Error computing growth trends: {str(e)}")
        raise

def assess_growth_quality(growth_data: Dict[str, float]) -> Dict[str, Union[str, float, List[str]]]:
    """Assess the quality and sustainability of growth metrics.
    
    Args:
        growth_data: Dictionary of growth metrics from compute_growth_trends
        
    Returns:
        Dict containing:
            - growth_score: float (0-5 score)
            - growth_classification: str (GrowthClassification enum value)
            - growth_stability: str
            - stability_flags: List[str]
            
    Raises:
        ValueError: If growth_data is empty
    """
    try:
        logger.info("Assessing growth quality")
        
        if not growth_data:
            raise ValueError("No growth data provided for assessment")
            
        score = 0.0
        stability_flags: List[str] = []
        
        # Assess revenue growth
        if 'revenue_cagr' in growth_data:
            revenue_cagr = growth_data['revenue_cagr']
            if revenue_cagr >= GROWTH_THRESHOLDS['high']:
                score += 2.0
            elif revenue_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 1.0
                
            # Check revenue stability
            if 'revenue_avg_yoy' in growth_data:
                revenue_std = abs(growth_data['revenue_avg_yoy'] - revenue_cagr)
                if revenue_std > 0.1:  # 10% deviation
                    stability_flags.append("Unstable revenue growth")
                    
        # Assess earnings growth
        if 'net_income_cagr' in growth_data:
            net_income_cagr = growth_data['net_income_cagr']
            if net_income_cagr >= GROWTH_THRESHOLDS['high']:
                score += 2.0
            elif net_income_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 1.0
                
        # Assess cash flow growth
        if 'fcf_cagr' in growth_data:
            fcf_cagr = growth_data['fcf_cagr']
            if fcf_cagr >= GROWTH_THRESHOLDS['high']:
                score += 1.0
            elif fcf_cagr >= GROWTH_THRESHOLDS['moderate']:
                score += 0.5
                
        # Determine classification
        if score >= 4.0:
            classification = GrowthClassification.HIGH.value
        elif score >= 2.0:
            classification = GrowthClassification.MODERATE.value
        else:
            classification = GrowthClassification.LOW.value
            
        # Determine stability
        stability = "Stable Growth" if not stability_flags else "Unstable Growth"
        
        logger.info(f"Growth assessment: {classification} (Score: {score:.1f})")
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
            'growth_score': 0.0,
            'growth_classification': GrowthClassification.ERROR.value,
            'growth_stability': 'Unknown',
            'stability_flags': [str(e)]
        }

def analyze_growth(financials: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive growth analysis.
    
    Args:
        financials: DataFrame with historical financial data
        
    Returns:
        Dict containing:
            - Growth metrics (CAGR, YOY)
            - Growth quality assessment
            - Analysis metadata
            
    Raises:
        ValueError: If financials DataFrame is empty or invalid
    """
    try:
        logger.info("Starting comprehensive growth analysis")
        
        if financials.empty:
            raise ValueError("Empty financials DataFrame provided")
            
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
            'status': 'Error',
            'error': str(e),
            'analysis_date': datetime.now().isoformat()
        }

def analyze_growth_metrics(
    financials: Dict[str, Union[pd.DataFrame, Dict[str, Any]]],
    price_data: Optional[pd.DataFrame] = None
) -> Dict[str, Union[float, str]]:
    """Analyze growth metrics and trends from financial statements.
    
    Args:
        financials: Dictionary containing:
            - income_stmt: Income statement DataFrame
            - balance_sheet: Balance sheet DataFrame
            - cash_flow: Cash flow statement DataFrame
            - fundamentals: Dictionary of fundamental metrics
        price_data: Optional DataFrame containing historical price data
        
    Returns:
        Dict[str, Union[float, str]] containing:
            - revenue_growth: Year-over-year revenue growth rate
            - earnings_growth: Year-over-year earnings growth rate
            - eps_growth: Year-over-year EPS growth rate
            - growth_score: Overall growth score (0-100)
            - status: Analysis status message
            
    Raises:
        ValueError: If required financial data is missing
    """
    try:
        logger.info("Starting growth metrics analysis")
        
        # Extract required data
        income_stmt = financials.get('income_stmt')
        balance_sheet = financials.get('balance_sheet')
        cash_flow = financials.get('cash_flow')
        fundamentals = financials.get('fundamentals', {})
        
        if income_stmt is None or income_stmt.empty:
            raise ValueError("No income statement data available")
        
        # Initialize results with type hints
        results: Dict[str, Union[float, str]] = {
            'revenue_growth': 0.0,
            'earnings_growth': 0.0,
            'eps_growth': 0.0,
            'growth_score': 0.0,
            'status': 'Success'
        }
        
        # Calculate growth metrics
        results['revenue_growth'] = _calculate_revenue_growth(income_stmt)
        results['earnings_growth'] = _calculate_earnings_growth(income_stmt)
        results['eps_growth'] = _calculate_eps_growth(income_stmt, fundamentals)
        
        # Calculate overall growth score
        growth_metrics = {
            'revenue_growth': results['revenue_growth'],
            'earnings_growth': results['earnings_growth'],
            'eps_growth': results['eps_growth']
        }
        
        results['growth_score'] = _calculate_growth_score(growth_metrics)
        logger.info(f"Calculated growth score: {results['growth_score']:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in growth metrics analysis: {str(e)}")
        return {
            'revenue_growth': 0.0,
            'earnings_growth': 0.0,
            'eps_growth': 0.0,
            'growth_score': 0.0,
            'status': f'Error: {str(e)}'
        }

def _calculate_revenue_growth(income_stmt: pd.DataFrame) -> float:
    """Calculate year-over-year revenue growth rate.
    
    Args:
        income_stmt: DataFrame containing income statement data
        
    Returns:
        float: Year-over-year revenue growth rate as a decimal
        
    Raises:
        ValueError: If required revenue data is missing or invalid
    """
    try:
        revenue_col = _normalize_column_name(income_stmt, ['Total Revenue', 'Revenue', 'revenue'])
        if not revenue_col:
            raise ValueError("Revenue data not found in income statement")
            
        revenue = income_stmt[revenue_col]
        if len(revenue) < 2:
            raise ValueError("Insufficient revenue data for growth calculation")
            
        # Calculate YoY growth
        growth = (revenue.iloc[0] - revenue.iloc[1]) / abs(revenue.iloc[1])
        logger.info(f"Calculated revenue growth: {growth:.2%}")
        return float(growth)
            
    except Exception as e:
        logger.error(f"Error calculating revenue growth: {str(e)}")
        return 0.0

def _calculate_earnings_growth(income_stmt: pd.DataFrame) -> float:
    """Calculate year-over-year earnings growth rate.
    
    Args:
        income_stmt: DataFrame containing income statement data
        
    Returns:
        float: Year-over-year earnings growth rate as a decimal
        
    Raises:
        ValueError: If required earnings data is missing or invalid
    """
    try:
        earnings_col = _normalize_column_name(income_stmt, ['Net Income', 'NetIncome', 'net_income'])
        if not earnings_col:
            raise ValueError("Earnings data not found in income statement")
            
        earnings = income_stmt[earnings_col]
        if len(earnings) < 2:
            raise ValueError("Insufficient earnings data for growth calculation")
            
        # Calculate YoY growth
        growth = (earnings.iloc[0] - earnings.iloc[1]) / abs(earnings.iloc[1])
        logger.info(f"Calculated earnings growth: {growth:.2%}")
        return float(growth)
            
    except Exception as e:
        logger.error(f"Error calculating earnings growth: {str(e)}")
        return 0.0

def _calculate_eps_growth(
    income_stmt: pd.DataFrame,
    fundamentals: Dict[str, Any]
) -> float:
    """Calculate year-over-year EPS growth rate.
    
    Args:
        income_stmt: DataFrame containing income statement data
        fundamentals: Dictionary containing fundamental metrics
        
    Returns:
        float: Year-over-year EPS growth rate as a decimal
        
    Raises:
        ValueError: If required EPS data is missing or invalid
    """
    try:
        # Try EPS directly from income statement
        eps_col = _normalize_column_name(income_stmt, ['EPS', 'Earnings Per Share', 'eps'])
        if eps_col:
            eps = income_stmt[eps_col]
            if len(eps) >= 2:
                growth = (eps.iloc[0] - eps.iloc[1]) / abs(eps.iloc[1])
                logger.info(f"Calculated EPS growth from EPS column: {growth:.2%}")
                return float(growth)
        
        # Calculate EPS from Net Income and Shares Outstanding
        net_income_col = _normalize_column_name(income_stmt, ['Net Income', 'NetIncome', 'net_income'])
        shares_col = _normalize_column_name(income_stmt, ['Shares Outstanding', 'shares_outstanding'])
        
        if net_income_col and shares_col:
            net_income = income_stmt[net_income_col]
            shares = income_stmt[shares_col]
            
            if len(net_income) >= 2 and len(shares) >= 2:
                eps_current = net_income.iloc[0] / shares.iloc[0] if shares.iloc[0] != 0 else 0
                eps_previous = net_income.iloc[1] / shares.iloc[1] if shares.iloc[1] != 0 else 0
                
                if eps_previous != 0:
                    growth = (eps_current - eps_previous) / abs(eps_previous)
                    logger.info(f"Calculated EPS growth from Net Income/Shares: {growth:.2%}")
                    return float(growth)
        
        raise ValueError("Could not calculate EPS growth from available data")
            
    except Exception as e:
        logger.error(f"Error calculating EPS growth: {str(e)}")
        return 0.0

def _calculate_fcf_growth(cash_flow: pd.DataFrame) -> float:
    """Calculate year-over-year free cash flow growth rate.
    
    Args:
        cash_flow: DataFrame containing cash flow statement data
        
    Returns:
        float: Year-over-year FCF growth rate as a decimal
        
    Raises:
        ValueError: If required FCF data is missing or invalid
    """
    try:
        fcf_col = _normalize_column_name(cash_flow, ['Free Cash Flow', 'FCF', 'free_cash_flow'])
        if not fcf_col:
            raise ValueError("Free Cash Flow data not found in cash flow statement")
            
        fcf = cash_flow[fcf_col]
        if len(fcf) < 2:
            raise ValueError("Insufficient FCF data for growth calculation")
            
        growth = (fcf.iloc[0] - fcf.iloc[1]) / abs(fcf.iloc[1])
        logger.info(f"Calculated FCF growth: {growth:.2%}")
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating FCF growth: {str(e)}")
        return 0.0

def _calculate_asset_growth(balance_sheet: pd.DataFrame) -> float:
    """Calculate year-over-year total asset growth rate.
    
    Args:
        balance_sheet: DataFrame containing balance sheet data
        
    Returns:
        float: Year-over-year asset growth rate as a decimal
        
    Raises:
        ValueError: If required asset data is missing or invalid
    """
    try:
        assets_col = _normalize_column_name(balance_sheet, ['Total Assets', 'total_assets'])
        if not assets_col:
            raise ValueError("Total Assets data not found in balance sheet")
            
        assets = balance_sheet[assets_col]
        if len(assets) < 2:
            raise ValueError("Insufficient asset data for growth calculation")
            
        growth = (assets.iloc[0] - assets.iloc[1]) / abs(assets.iloc[1])
        logger.info(f"Calculated asset growth: {growth:.2%}")
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating asset growth: {str(e)}")
        return 0.0

def _calculate_equity_growth(balance_sheet: pd.DataFrame) -> float:
    """Calculate year-over-year shareholder equity growth rate.
    
    Args:
        balance_sheet: DataFrame containing balance sheet data
        
    Returns:
        float: Year-over-year equity growth rate as a decimal
        
    Raises:
        ValueError: If required equity data is missing or invalid
    """
    try:
        equity_col = _normalize_column_name(
            balance_sheet,
            ['Total Stockholder Equity', 'Stockholders Equity', 'total_equity']
        )
        if not equity_col:
            raise ValueError("Total Stockholder Equity data not found in balance sheet")
            
        equity = balance_sheet[equity_col]
        if len(equity) < 2:
            raise ValueError("Insufficient equity data for growth calculation")
            
        growth = (equity.iloc[0] - equity.iloc[1]) / abs(equity.iloc[1])
        logger.info(f"Calculated equity growth: {growth:.2%}")
        return float(growth)
        
    except Exception as e:
        logger.error(f"Error calculating equity growth: {str(e)}")
        return 0.0

def _calculate_growth_score(metrics: Dict[str, float]) -> float:
    """Calculate overall growth score based on weighted metrics.
    
    Args:
        metrics: Dictionary containing growth metrics:
            - revenue_growth: Revenue growth rate
            - earnings_growth: Earnings growth rate
            - eps_growth: EPS growth rate
            - fcf_growth: Free cash flow growth rate
            - asset_growth: Asset growth rate
            - equity_growth: Equity growth rate
            
    Returns:
        float: Overall growth score (0-100)
        
    Raises:
        ValueError: If no valid metrics are provided
    """
    try:
        # Define weights for different growth metrics
        weights: Dict[str, float] = {
            'revenue_growth': 0.3,
            'earnings_growth': 0.3,
            'eps_growth': 0.2,
            'fcf_growth': 0.1,
            'asset_growth': 0.05,
            'equity_growth': 0.05
        }
        
        # Calculate weighted score
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] != 0:
                # Normalize growth rate to 0-100 scale
                growth = metrics[metric]
                normalized_score = min(max(growth * 100, 0), 100)
                score += normalized_score * weight
                total_weight += weight
                
        if total_weight == 0:
            raise ValueError("No valid growth metrics available")
            
        final_score = score / total_weight
        logger.info(f"Calculated overall growth score: {final_score:.2f}")
        return float(final_score)
        
    except Exception as e:
        logger.error(f"Error calculating growth score: {str(e)}")
        return 0.0

def _classify_growth(score: float) -> str:
    """Classify growth level based on growth score.
    
    Args:
        score: Growth score (0-100)
        
    Returns:
        str: Growth classification (High/Moderate/Low)
    """
    if score >= 80:
        classification = GrowthClassification.HIGH.value
    elif score >= 50:
        classification = GrowthClassification.MODERATE.value
    else:
        classification = GrowthClassification.LOW.value
        
    logger.info(f"Classified growth as: {classification} (Score: {score:.2f})")
    return classification 