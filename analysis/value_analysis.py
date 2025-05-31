from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

def calculate_intrinsic_value(
    financials: Dict[str, Union[pd.DataFrame, Dict]],
    growth_rate: float = 0.03,
    discount_rate: float = 0.10,
    years: int = 10
) -> Dict[str, Union[float, Dict, None]]:
    """
    Calculate the intrinsic value of a stock using Discounted Cash Flow (DCF) analysis.
    
    Args:
        financials (Dict): Dictionary containing financial data:
            - income_stmt: Income statement DataFrame
            - balance_sheet: Balance sheet DataFrame
            - cash_flow: Cash flow statement DataFrame
            - fundamentals: Dictionary of fundamental metrics
        growth_rate (float, optional): Expected growth rate. Defaults to 0.03 (3%).
        discount_rate (float, optional): Discount rate (required rate of return). Defaults to 0.10 (10%).
        years (int, optional): Number of years to project. Defaults to 10.
        
    Returns:
        Dict containing:
            - intrinsic_value: Calculated intrinsic value per share
            - margin_of_safety: Margin of safety percentage (None if not computable)
            - assumptions: Dictionary of assumptions used
            - components: Dictionary of DCF components
            
    Raises:
        ValueError: If input data is invalid or missing required components
        Exception: For other errors during calculation
    """
    try:
        logger.info("Starting intrinsic value calculation")
        
        # Input validation
        if not isinstance(financials, dict):
            logger.error("Invalid financials data structure: expected dict, got %s", type(financials))
            raise ValueError("Invalid financials data structure")
        if not 0 <= growth_rate <= 1:
            logger.error("Growth rate must be between 0 and 1, got %s", growth_rate)
            raise ValueError("Growth rate must be between 0 and 1")
        if not 0 <= discount_rate <= 1:
            logger.error("Discount rate must be between 0 and 1, got %s", discount_rate)
            raise ValueError("Discount rate must be between 0 and 1")
        if not isinstance(years, int) or years <= 0:
            logger.error("Years must be a positive integer, got %s", years)
            raise ValueError("Years must be a positive integer")
            
        # Extract required data
        income_stmt = financials.get('income_stmt')
        balance_sheet = financials.get('balance_sheet')
        cash_flow = financials.get('cash_flow')
        fundamentals = financials.get('fundamentals', {})
        price_data = financials.get('price_data')
        
        # Log available keys and types
        logger.info(f"Available financials keys: {list(financials.keys())}")
        logger.info(f"income_stmt: {type(income_stmt)}, balance_sheet: {type(balance_sheet)}, cash_flow: {type(cash_flow)}")
        if income_stmt is not None:
            logger.info(f"income_stmt columns: {getattr(income_stmt, 'columns', None)}")
        if balance_sheet is not None:
            logger.info(f"balance_sheet columns: {getattr(balance_sheet, 'columns', None)}")
        if cash_flow is not None:
            logger.info(f"cash_flow columns: {getattr(cash_flow, 'columns', None)}")

        missing = []
        if income_stmt is None or income_stmt.empty:
            logger.error("Missing or empty required financial statement: income_stmt")
            missing.append('income_stmt')
        if balance_sheet is None or balance_sheet.empty:
            logger.error("Missing or empty required financial statement: balance_sheet")
            missing.append('balance_sheet')
        if cash_flow is None or cash_flow.empty:
            logger.error("Missing or empty required financial statement: cash_flow")
            missing.append('cash_flow')
        if missing:
            raise ValueError(f"Missing required financial statements: {', '.join(missing)}")
            
        # Calculate free cash flow
        fcf = _calculate_free_cash_flow(income_stmt, balance_sheet, cash_flow)
        if fcf is None:
            logger.error("Could not calculate free cash flow. See previous logs for missing fields and alternatives checked.")
            raise ValueError("Could not calculate free cash flow. Check logs for missing fields and alternatives checked.")
            
        # Calculate terminal value
        terminal_value = _calculate_terminal_value(fcf, growth_rate, discount_rate)
        
        # Calculate present value of projected cash flows
        projected_cash_flows = _project_cash_flows(fcf, growth_rate, years)
        present_values = _calculate_present_values(projected_cash_flows, discount_rate)
        
        # Calculate total enterprise value
        enterprise_value = sum(present_values) + terminal_value
        
        # Calculate equity value
        equity_value = _calculate_equity_value(enterprise_value, balance_sheet)
        
        # Calculate per share value
        shares_outstanding = _get_shares_outstanding(fundamentals, balance_sheet)
        intrinsic_value = equity_value / shares_outstanding if shares_outstanding else None
        
        # Get current price from fundamentals or fallback to price_data
        current_price = fundamentals.get('latest_price')
        if (current_price is None or current_price <= 0) and price_data is not None and not price_data.empty:
            try:
                current_price = price_data['Close'].iloc[-1]
                logger.info(f"Using fallback current price from price_data: {current_price}")
            except Exception as e:
                logger.warning(f"Could not get current price from price_data: {str(e)}")
                current_price = None
        
        # Calculate margin of safety only if both values are valid
        if intrinsic_value is not None and intrinsic_value > 0 and current_price is not None and current_price > 0:
            margin_of_safety = _calculate_margin_of_safety(current_price, intrinsic_value)
        else:
            logger.warning(f"Cannot calculate margin of safety: intrinsic_value={intrinsic_value}, current_price={current_price}")
            margin_of_safety = None
        
        # Prepare results
        results = {
            'intrinsic_value': intrinsic_value,
            'margin_of_safety': margin_of_safety,
            'assumptions': {
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'projection_years': years
            },
            'components': {
                'free_cash_flow': fcf,
                'terminal_value': terminal_value,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value
            }
        }
        
        logger.info("Successfully calculated intrinsic value")
        return results
        
    except ValueError as ve:
        logger.error(f"Validation error in intrinsic value calculation: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error calculating intrinsic value: {str(e)}")
        raise

def _calculate_free_cash_flow(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame
) -> Optional[float]:
    """
    Calculate free cash flow from financial statements.
    
    Args:
        income_stmt: Income statement DataFrame
        balance_sheet: Balance sheet DataFrame
        cash_flow: Cash flow statement DataFrame
        
    Returns:
        float: Free cash flow value, or None if calculation fails
        
    Note:
        Tries multiple methods to calculate FCF:
        1. Direct from cash flow statement (Operating Cash Flow - Capex)
        2. From income statement (EBIT * (1-tax) + Depreciation)
        3. From net income (Net Income + Depreciation)
    """
    try:
        logger.info("Calculating free cash flow")
        
        # Log available columns for debugging
        logger.info(f"Cash Flow Statement columns: {cash_flow.columns.tolist() if cash_flow is not None else 'None'}")
        logger.info(f"Income Statement columns: {income_stmt.columns.tolist() if income_stmt is not None else 'None'}")
        logger.info(f"Balance Sheet columns: {balance_sheet.columns.tolist() if balance_sheet is not None else 'None'}")
        
        # Method 1: Direct from cash flow statement
        try:
            ocf_columns = [
                'Operating Cash Flow', 'operating_cash_flow', 'Cash Flow From Operations',
                'Total Cash From Operating Activities', 'total_cash_from_operating_activities',
                'Cash Flow From Operations', 'cash_flow_from_operations',
                'Net Cash Provided by Operating Activities', 'net_cash_provided_by_operating_activities',
                'Cash Flow From Operating Activities', 'cash_flow_from_operating_activities',
                'Operating Activities', 'operating_activities'
            ]
            capex_columns = [
                'Capital Expenditure', 'capital_expenditure', 'Purchase Of Equipment',
                'Capital Expenditures', 'capital_expenditures', 'Purchase of PPE',
                'Purchase of Property Plant and Equipment', 'Purchase of Fixed Assets',
                'Capital Expenditure - Fixed Assets', 'Capital Expenditure - Fixed Assets',
                'Investments In Property Plant And Equipment', 'investments_in_property_plant_and_equipment',
                'Purchase of Equipment', 'purchase_of_equipment'
            ]
            logger.info(f"Checking for operating cash flow columns: {ocf_columns}")
            logger.info(f"Checking for capex columns: {capex_columns}")
            operating_cash_flow = None
            for col in ocf_columns:
                if col in cash_flow.columns:
                    # Get the most recent value (first row since dates are in index)
                    operating_cash_flow = cash_flow[col].iloc[0] if not cash_flow[col].empty else None
                    logger.info(f"Found operating cash flow in column: {col} = {operating_cash_flow}")
                    break
            if operating_cash_flow is None:
                logger.error(f"None of the expected operating cash flow columns found for FCF calculation. Tried: {ocf_columns}. Available columns: {list(cash_flow.columns)}")
            capex = None
            for col in capex_columns:
                if col in cash_flow.columns:
                    # Get the most recent value (first row since dates are in index)
                    capex = cash_flow[col].iloc[0] if not cash_flow[col].empty else None
                    logger.info(f"Found capital expenditure in column: {col} = {capex}")
                    break
            if capex is None:
                logger.error(f"None of the expected capex columns found for FCF calculation. Tried: {capex_columns}. Available columns: {list(cash_flow.columns)}")
            if operating_cash_flow is not None and capex is not None:
                fcf = operating_cash_flow - abs(capex)
                logger.info(f"Calculated FCF from cash flow statement: {fcf}")
                return fcf
            else:
                logger.warning(f"Missing required data for Method 1: operating_cash_flow={operating_cash_flow}, capex={capex}")
        except Exception as e:
            logger.warning(f"Failed to calculate FCF from cash flow statement: {str(e)}")
        
        # Method 2: From income statement and balance sheet
        try:
            ebit_columns = [
                'Operating Income', 'operating_income', 'EBIT', 'ebit',
                'Earnings Before Interest and Taxes', 'earnings_before_interest_and_taxes',
                'Operating Profit', 'operating_profit', 'Income Before Tax', 'income_before_tax',
                'Operating Income Loss', 'operating_income_loss'
            ]
            logger.info(f"Checking for EBIT columns: {ebit_columns}")
            ebit = None
            for col in ebit_columns:
                if col in income_stmt.columns:
                    ebit = income_stmt[col].iloc[0] if not income_stmt[col].empty else None
                    logger.info(f"Found EBIT in column: {col} = {ebit}")
                    break
            if ebit is None:
                logger.error(f"None of the expected EBIT columns found for FCF calculation. Tried: {ebit_columns}. Available columns: {list(income_stmt.columns)}")
            if ebit is not None:
                tax_rate = 0.21
                tax_columns = [
                    'Income Tax Expense', 'income_tax_expense', 'Provision for Income Taxes',
                    'Income Tax', 'income_tax', 'Tax Expense', 'tax_expense',
                    'Income Tax Expense', 'income_tax_expense'
                ]
                logger.info(f"Checking for tax columns: {tax_columns}")
                for col in tax_columns:
                    if col in income_stmt.columns:
                        tax_expense = income_stmt[col].iloc[0] if not income_stmt[col].empty else None
                        if tax_expense and ebit != 0:
                            tax_rate = tax_expense / ebit
                        logger.info(f"Found tax rate from column: {col} = {tax_rate:.2%}")
                        break
                        
                # Check for depreciation in cash flow statement first, then income statement
                dep_columns = [
                    'Depreciation and Amortization', 'Depreciation & Amortization',
                    'depreciation_and_amortization', 'Depreciation', 'depreciation',
                    'Depreciation And Amortization', 'Depreciation Expense', 'depreciation_expense'
                ]
                logger.info(f"Checking for depreciation columns: {dep_columns}")
                depreciation = 0
                # Check cash flow statement first
                if cash_flow is not None:
                    for col in dep_columns:
                        if col in cash_flow.columns:
                            depreciation = cash_flow[col].iloc[0] if not cash_flow[col].empty else 0
                            logger.info(f"Found depreciation in cash flow column: {col} = {depreciation}")
                            break
                # If not found, check income statement
                if depreciation == 0:
                    for col in dep_columns:
                        if col in income_stmt.columns:
                            depreciation = income_stmt[col].iloc[0] if not income_stmt[col].empty else 0
                            logger.info(f"Found depreciation in income statement column: {col} = {depreciation}")
                            break
                            
                fcf = ebit * (1 - tax_rate) + depreciation
                logger.info(f"Calculated FCF from income statement: {fcf}")
                return fcf
            else:
                logger.warning("Missing EBIT data for Method 2")
        except Exception as e:
            logger.warning(f"Failed to calculate FCF from income statement: {str(e)}")
        
        # Method 3: From net income
        try:
            net_income_columns = [
                'Net Income', 'net_income', 'Net Earnings', 'net_earnings',
                'Profit', 'profit', 'Net Profit', 'net_profit',
                'Net Income Common Stockholders', 'net_income_common_stockholders'
            ]
            logger.info(f"Checking for net income columns: {net_income_columns}")
            net_income = None
            for col in net_income_columns:
                if col in income_stmt.columns:
                    net_income = income_stmt[col].iloc[0] if not income_stmt[col].empty else None
                    logger.info(f"Found net income in column: {col} = {net_income}")
                    break
            if net_income is None:
                logger.error(f"None of the expected net income columns found for FCF calculation. Tried: {net_income_columns}. Available columns: {list(income_stmt.columns)}")
            if net_income is not None:
                # Check for depreciation in cash flow statement first, then income statement
                dep_columns = [
                    'Depreciation and Amortization', 'Depreciation & Amortization',
                    'depreciation_and_amortization', 'Depreciation', 'depreciation',
                    'Depreciation And Amortization', 'Depreciation Expense', 'depreciation_expense'
                ]
                logger.info(f"Checking for depreciation columns: {dep_columns}")
                depreciation = 0
                # Check cash flow statement first
                if cash_flow is not None:
                    for col in dep_columns:
                        if col in cash_flow.columns:
                            depreciation = cash_flow[col].iloc[0] if not cash_flow[col].empty else 0
                            logger.info(f"Found depreciation in cash flow column: {col} = {depreciation}")
                            break
                # If not found, check income statement
                if depreciation == 0:
                    for col in dep_columns:
                        if col in income_stmt.columns:
                            depreciation = income_stmt[col].iloc[0] if not income_stmt[col].empty else 0
                            logger.info(f"Found depreciation in income statement column: {col} = {depreciation}")
                            break
                            
                fcf = net_income + depreciation
                logger.info(f"Calculated FCF from net income: {fcf}")
                return fcf
            else:
                logger.warning("Missing net income data for Method 3")
        except Exception as e:
            logger.warning(f"Failed to calculate FCF from net income: {str(e)}")
        
        logger.error("Could not calculate free cash flow using any method. Please check logs for missing fields and alternatives checked.")
        return None
        
    except Exception as e:
        logger.error(f"Error in free cash flow calculation: {str(e)}")
        return None

def _calculate_terminal_value(fcf: float, growth_rate: float, discount_rate: float) -> float:
    """
    Calculate terminal value using Gordon Growth Model.
    
    Args:
        fcf: Free cash flow
        growth_rate: Expected growth rate
        discount_rate: Discount rate
        
    Returns:
        float: Terminal value
    """
    try:
        logger.info("Calculating terminal value")
        
        if discount_rate <= growth_rate:
            raise ValueError("Discount rate must be greater than growth rate")
            
        terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
        logger.info(f"Calculated terminal value: {terminal_value}")
        return terminal_value
        
    except Exception as e:
        logger.error(f"Error calculating terminal value: {str(e)}")
        raise

def _project_cash_flows(fcf: float, growth_rate: float, years: int) -> list:
    """
    Project future cash flows.
    
    Args:
        fcf: Current free cash flow
        growth_rate: Expected growth rate
        years: Number of years to project
        
    Returns:
        list: Projected cash flows
    """
    try:
        logger.info(f"Projecting cash flows for {years} years")
        
        projected_flows = []
        current_fcf = fcf
        
        for year in range(years):
            current_fcf *= (1 + growth_rate)
            projected_flows.append(current_fcf)
            
        logger.info(f"Projected {len(projected_flows)} years of cash flows")
        return projected_flows
        
    except Exception as e:
        logger.error(f"Error projecting cash flows: {str(e)}")
        raise

def _calculate_present_values(cash_flows: list, discount_rate: float) -> list:
    """
    Calculate present values of projected cash flows.
    
    Args:
        cash_flows: List of projected cash flows
        discount_rate: Discount rate
        
    Returns:
        list: Present values
    """
    try:
        logger.info("Calculating present values")
        
        present_values = []
        for year, cash_flow in enumerate(cash_flows, 1):
            present_value = cash_flow / ((1 + discount_rate) ** year)
            present_values.append(present_value)
            
        logger.info(f"Calculated {len(present_values)} present values")
        return present_values
        
    except Exception as e:
        logger.error(f"Error calculating present values: {str(e)}")
        raise

def _calculate_equity_value(enterprise_value: float, balance_sheet: pd.DataFrame) -> float:
    """
    Calculate equity value from enterprise value.
    
    Args:
        enterprise_value: Total enterprise value
        balance_sheet: Balance sheet DataFrame
        
    Returns:
        float: Equity value
    """
    try:
        logger.info("Calculating equity value")
        
        # Get net debt
        total_debt = 0
        cash = 0
        
        # Check for Total Debt in columns
        debt_columns = ['Total Debt', 'Long Term Debt', 'Total Liabilities']
        for col in debt_columns:
            if col in balance_sheet.columns:
                total_debt = balance_sheet[col].iloc[0] if not balance_sheet[col].empty else 0
                logger.info(f"Found total debt in column: {col} = {total_debt}")
                break
        
        # Check for Cash in columns
        cash_columns = ['Cash and Equivalents', 'Cash', 'Cash And Cash Equivalents', 'Cash Financial']
        for col in cash_columns:
            if col in balance_sheet.columns:
                cash = balance_sheet[col].iloc[0] if not balance_sheet[col].empty else 0
                logger.info(f"Found cash in column: {col} = {cash}")
                break
        
        net_debt = total_debt - cash
        equity_value = enterprise_value - net_debt
        
        logger.info(f"Total debt: {total_debt}, Cash: {cash}, Net debt: {net_debt}")
        logger.info(f"Calculated equity value: {equity_value}")
        return equity_value
        
    except Exception as e:
        logger.error(f"Error calculating equity value: {str(e)}")
        raise

def _get_shares_outstanding(fundamentals: Dict, balance_sheet: pd.DataFrame = None) -> Optional[float]:
    """
    Get number of shares outstanding.
    
    Args:
        fundamentals: Dictionary of fundamental metrics
        balance_sheet: Balance sheet DataFrame (optional)
        
    Returns:
        float: Number of shares outstanding, or None if not available
    """
    try:
        logger.info("Getting shares outstanding")
        
        # Check fundamentals first
        shares = fundamentals.get('shares_outstanding')
        if shares is None:
            # Try market cap / price calculation
            market_cap = fundamentals.get('market_cap')
            latest_price = fundamentals.get('latest_price')
            if market_cap and latest_price and latest_price > 0:
                shares = market_cap / latest_price
                logger.info(f"Calculated shares from market cap: {shares:,.0f}")
                return float(shares)
            
            # Try balance sheet
            if balance_sheet is not None and not balance_sheet.empty:
                # Check for Shares Outstanding column
                shares_columns = ['Shares Outstanding', 'Common Shares Outstanding', 
                                'Ordinary Shares Number', 'Share Issued']
                for col in shares_columns:
                    if col in balance_sheet.columns and not balance_sheet[col].empty:
                        shares_value = balance_sheet[col].iloc[0]
                        if not pd.isna(shares_value) and shares_value > 0:
                            logger.info(f"Found shares in balance sheet column '{col}': {shares_value:,.0f}")
                            return float(shares_value)
            
            logger.warning("Shares outstanding not found in fundamentals or balance sheet")
            logger.warning("Using default shares outstanding: 10000000000")
            return 10000000000  # Default to 10B shares
        
        logger.info(f"Shares outstanding from fundamentals: {shares:,.0f}")
        return float(shares)
    
    except Exception as e:
        logger.error(f"Error getting shares outstanding: {e}")
        return None

def _calculate_margin_of_safety(current_price: float, intrinsic_value: float) -> Optional[float]:
    """
    Calculate margin of safety.
    
    Args:
        current_price: Current stock price
        intrinsic_value: Calculated intrinsic value
        
    Returns:
        Optional[float]: Margin of safety as a percentage, or None if inputs are invalid
    """
    try:
        logger.info("Calculating margin of safety")
        
        if current_price is None or intrinsic_value is None or current_price <= 0 or intrinsic_value <= 0:
            logger.warning(f"Invalid input(s) for margin of safety calculation: current_price={current_price}, intrinsic_value={intrinsic_value}")
            return None
        
        margin = ((intrinsic_value - current_price) / intrinsic_value) * 100
        logger.info(f"Calculated margin of safety: {margin}%")
        return margin
        
    except Exception as e:
        logger.error(f"Error calculating margin of safety: {str(e)}")
        return None

def calculate_margin_of_safety(
    intrinsic_value: float,
    current_price: float
) -> float:
    """
    Calculate margin of safety between intrinsic value and current price.
    
    Args:
        intrinsic_value: Calculated intrinsic value per share
        current_price: Current market price per share
        
    Returns:
        float: Margin of safety as a decimal (e.g., 0.25 for 25%)
    """
    try:
        logger.info("Calculating margin of safety")
        
        if intrinsic_value <= 0:
            logger.error("Invalid intrinsic value (must be positive)")
            return 0.0
            
        if current_price <= 0:
            logger.error("Invalid current price (must be positive)")
            return 0.0
        
        margin = (intrinsic_value - current_price) / intrinsic_value
        
        # Log appropriate message based on margin
        if margin <= 0:
            logger.warning(f"Negative margin of safety: {margin:.2%}")
        elif margin < 0.2:
            logger.warning(f"Low margin of safety: {margin:.2%}")
        else:
            logger.info(f"Margin of safety: {margin:.2%}")
            
        return margin
        
    except Exception as e:
        logger.error(f"Error calculating margin of safety: {str(e)}")
        return 0.0

def get_projected_cash_flows(
    financial_data: pd.DataFrame,
    years: int = 5,
    conservative_factor: float = 0.8
) -> List[float]:
    """
    Project future cash flows based on historical data and growth rates.
    
    Args:
        financial_data: DataFrame with historical financial data
        years: Number of years to project (default: 5)
        conservative_factor: Factor to reduce growth rate (default: 0.8)
        
    Returns:
        List[float]: Projected free cash flows for next N years
    """
    try:
        logger.info(f"Projecting cash flows for next {years} years")
        
        if financial_data.empty:
            logger.error("No financial data provided for projection")
            return []
            
        # Calculate historical growth rates
        if 'free_cash_flow' not in financial_data.columns:
            logger.error("Missing free cash flow data")
            return []
            
        fcf = financial_data['free_cash_flow']
        if len(fcf) < 2:
            logger.warning("Insufficient historical data for growth calculation")
            return [fcf.iloc[-1]] * years
            
        # Calculate year-over-year growth rates
        growth_rates = fcf.pct_change().dropna()
        
        # Use conservative growth rate
        avg_growth = growth_rates.mean() * conservative_factor
        logger.info(f"Using conservative growth rate: {avg_growth:.2%}")
        
        # Project future cash flows
        last_fcf = fcf.iloc[-1]
        projected_flows = []
        
        for year in range(years):
            next_fcf = last_fcf * (1 + avg_growth)
            projected_flows.append(next_fcf)
            last_fcf = next_fcf
            
        logger.info(f"Successfully projected {len(projected_flows)} years of cash flows")
        return projected_flows
        
    except Exception as e:
        logger.error(f"Error projecting cash flows: {str(e)}")
        return []

def calculate_intrinsic_value_dcf(
    projected_flows: List[float],
    discount_rate: float,
    terminal_growth_rate: float
) -> float:
    """
    Calculate intrinsic value using Discounted Cash Flow (DCF) method.
    
    Args:
        projected_flows: List of projected free cash flows
        discount_rate: Required rate of return
        terminal_growth_rate: Long-term growth rate
        
    Returns:
        float: Calculated intrinsic value per share
    """
    try:
        logger.info("Calculating DCF intrinsic value")
        
        if not projected_flows:
            raise ValueError("No projected cash flows provided")
            
        # Calculate present value of projected cash flows
        present_values = []
        for year, cash_flow in enumerate(projected_flows, 1):
            present_value = cash_flow / ((1 + discount_rate) ** year)
            present_values.append(present_value)
            
        # Calculate terminal value using Gordon Growth Model
        terminal_fcf = projected_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        terminal_pv = terminal_value / ((1 + discount_rate) ** len(projected_flows))
        
        # Calculate total value
        total_value = sum(present_values) + terminal_pv
        
        logger.info(f"Calculated DCF intrinsic value: {total_value}")
        return total_value
        
    except Exception as e:
        logger.error(f"Error calculating DCF intrinsic value: {str(e)}")
        raise

def analyze_value_metrics(
    financial_data: pd.DataFrame,
    current_price: float,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.03,
    projection_years: int = 5
) -> Dict[str, Union[float, str]]:
    """
    Perform comprehensive value analysis and return key metrics.
    
    Args:
        financial_data: DataFrame with historical financial data
        current_price: Current market price per share
        discount_rate: Required rate of return (default: 10%)
        terminal_growth_rate: Long-term growth rate (default: 3%)
        projection_years: Years to project cash flows (default: 5)
        
    Returns:
        Dict containing:
            - intrinsic_value: Calculated DCF value
            - margin_of_safety: Safety margin as decimal
            - projected_cash_flows: List of projected flows
            - analysis_date: Timestamp of analysis
            - status: Success/error message
    """
    try:
        logger.info("Starting comprehensive value analysis")
        
        # Get projected cash flows
        projected_flows = get_projected_cash_flows(
            financial_data,
            years=projection_years
        )
        
        if not projected_flows:
            return {
                'intrinsic_value': 0.0,
                'margin_of_safety': 0.0,
                'projected_cash_flows': [],
                'analysis_date': datetime.now().isoformat(),
                'status': 'Error: Could not project cash flows'
            }
        
        # Calculate intrinsic value
        intrinsic_value = calculate_intrinsic_value_dcf(
            projected_flows,
            discount_rate,
            terminal_growth_rate
        )
        
        # Calculate margin of safety
        margin = calculate_margin_of_safety(intrinsic_value, current_price)
        
        return {
            'intrinsic_value': intrinsic_value,
            'margin_of_safety': margin,
            'projected_cash_flows': projected_flows,
            'analysis_date': datetime.now().isoformat(),
            'status': 'Success'
        }
        
    except Exception as e:
        logger.error(f"Error in value analysis: {str(e)}")
        return {
            'intrinsic_value': 0.0,
            'margin_of_safety': 0.0,
            'projected_cash_flows': [],
            'analysis_date': datetime.now().isoformat(),
            'status': f'Error: {str(e)}'
        } 