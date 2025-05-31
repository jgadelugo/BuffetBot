"""
Data Collection Report Generator

This module provides functionality to generate comprehensive reports about the data collection
process, including what data was successfully collected, what data was missing, and how this
affects various financial metrics and analyses.

The report includes:
- Data availability status for each financial statement
- Missing columns and their impact
- Affected metrics and calculations
- Data quality indicators
- Recommendations for data collection improvements
"""

from typing import Dict, List, Union, Optional, Any
import pandas as pd
from datetime import datetime
import json
import numpy as np

from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

class DataCollectionReport:
    """
    Generates comprehensive reports about data collection status and quality.
    
    This class analyzes the collected financial data and generates detailed reports
    about what data was successfully collected, what data is missing, and how this
    affects various financial metrics and analyses.
    
    Attributes:
        financials (Dict): Dictionary containing financial statements and fundamentals
        price_data (Optional[pd.DataFrame]): Historical price data if available
        report_data (Dict): Structured report data
    """
    
    def __init__(
        self,
        financials: Dict[str, Union[pd.DataFrame, Dict]],
        price_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the DataCollectionReport.
        
        Args:
            financials: Dictionary containing financial statements and fundamentals
            price_data: Optional price history DataFrame
        """
        self.financials = financials
        self.price_data = price_data
        self.report_data = self._generate_report()
        
    def _check_data_availability(self) -> Dict[str, Any]:
        """Check data availability and quality."""
        logger.info("Starting data availability check")
        availability = {}
        
        # Define required columns for each statement type with alternative names
        required_columns = {
            'income_stmt': [
                'Total Revenue',  # Alternative: 'Revenue', 'Sales', 'Net Sales'
                'Gross Profit',   # Alternative: 'Gross Income', 'Gross Margin'
                'Operating Income', # Alternative: 'Operating Profit', 'EBIT'
                'Net Income'      # Alternative: 'Net Earnings', 'Net Profit'
            ],
            'balance_sheet': [
                'Total Assets',
                'Total Liabilities',
                'Total Stockholder Equity'
            ],
            'cash_flow': [
                'Operating Cash Flow',
                'Capital Expenditure',
                'Free Cash Flow'
            ]
        }
        
        # Log the structure of the financials dictionary
        logger.info("Financial data structure:")
        for key, value in self.financials.items():
            if isinstance(value, pd.DataFrame):
                logger.info(f"{key}: DataFrame with shape {value.shape}")
                logger.info(f"{key} columns: {list(value.columns)}")
                logger.info(f"{key} index: {list(value.index)}")
            elif isinstance(value, dict):
                logger.info(f"{key}: Dictionary with keys {list(value.keys())}")
            else:
                logger.info(f"{key}: {type(value)}")
        
        # Check each financial statement
        for statement_type in ['income_stmt', 'balance_sheet', 'cash_flow']:
            statement_data = self.financials.get(statement_type)
            if statement_data is None or statement_data.empty:
                logger.warning(f"No data available for {statement_type}")
                availability[statement_type] = {
                    'available': False,
                    'collection_status': {
                        'error': 'No data available',
                        'reason': 'Data not collected or empty'
                    }
                }
                continue
            
            # Log available columns for debugging
            logger.info(f"Available columns for {statement_type}: {list(statement_data.columns)}")
            
            # Check completeness
            total_rows = len(statement_data)
            non_null_rows = statement_data.count().min()
            completeness = (non_null_rows / total_rows) * 100 if total_rows > 0 else 0
            
            # Check for missing required columns
            missing_columns = [col for col in required_columns[statement_type] 
                             if col not in statement_data.columns]
            
            # Check for negative values in key columns
            negative_values = {}
            for col in required_columns[statement_type]:
                if col in statement_data.columns:
                    neg_count = (statement_data[col] < 0).sum()
                    if neg_count > 0:
                        negative_values[col] = int(neg_count)
            
            # Get last available date with enhanced error handling
            try:
                # Log the index type and first few values for debugging
                logger.info(f"Index type for {statement_type}: {type(statement_data.index)}")
                logger.info(f"First few index values for {statement_type}: {statement_data.index[:5].tolist()}")
                
                # For balance sheet, we don't require a datetime index
                if statement_type == 'balance_sheet':
                    logger.info("Balance sheet data found - using first row as latest data")
                    last_date = None
                else:
                    # For other statements, check if index is datetime
                    if not isinstance(statement_data.index, pd.DatetimeIndex):
                        logger.info(f"Index for {statement_type} is not datetime, skipping date processing")
                        last_date = None
                    else:
                        last_date = statement_data.index.max()
                        if pd.isna(last_date):
                            logger.warning(f"Could not determine last date for {statement_type} - all dates are invalid")
                            last_date = None
                        else:
                            logger.info(f"Successfully determined last date for {statement_type}: {last_date}")
            except Exception as e:
                logger.error(f"Error processing date for {statement_type}: {str(e)}")
                logger.error(f"Index values causing error: {statement_data.index[:5].tolist()}")
                last_date = None
            
            availability[statement_type] = {
                'available': True,
                'completeness': completeness,
                'periods_available': total_rows,
                'last_available_date': last_date,
                'missing_columns': missing_columns,
                'data_quality_issues': [
                    f"Negative values in {col}: {count} occurrences" 
                    for col, count in negative_values.items()
                ] if negative_values else []
            }
            
            # Log detailed information about the statement
            logger.info(f"Data availability for {statement_type}:")
            logger.info(f"- Completeness: {completeness:.1f}%")
            logger.info(f"- Periods available: {total_rows}")
            logger.info(f"- Last available date: {last_date}")
            if missing_columns:
                logger.warning(f"- Missing columns: {missing_columns}")
            if negative_values:
                logger.warning(f"- Negative values found: {negative_values}")
        
        # Check fundamentals
        fundamentals = self.financials.get('fundamentals', {})
        if fundamentals:
            # Check for outliers in fundamental metrics
            try:
                for metric, value in fundamentals.items():
                    if isinstance(value, (int, float)):
                        # Log the actual value for debugging
                        logger.info(f"Fundamental metric {metric}: {value}")
                        # Add more sophisticated outlier detection if needed
                        if value < 0:
                            logger.warning(f"Negative value detected in fundamental metric {metric}: {value}")
            except Exception as e:
                logger.error(f"Error checking fundamental metrics: {str(e)}")
        
        logger.info("Completed data availability check")
        return availability
        
    def _analyze_impact(self, availability: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Analyze impact of missing data on various metrics and analyses.
        
        Args:
            availability: Dictionary containing data availability status
            
        Returns:
            Dict containing affected metrics and analyses
        """
        try:
            logger.info("Starting impact analysis")
            impact = {
                'value_metrics': [],
                'health_metrics': [],
                'growth_metrics': [],
                'risk_metrics': []
            }
            
            # Check fundamentals data specifically for P/E Ratio
            fundamentals = self.financials.get('fundamentals', {})
            if fundamentals:
                logger.info(f"Available fundamental metrics: {list(fundamentals.keys())}")
                if 'pe_ratio' in fundamentals:
                    pe_value = fundamentals['pe_ratio']
                    logger.info(f"P/E Ratio value found: {pe_value}")
                    if pe_value is None or pd.isna(pe_value):
                        logger.warning("P/E Ratio is present but has no valid value")
                        impact['value_metrics'].append('P/E Ratio')
                else:
                    logger.warning("P/E Ratio not found in fundamentals")
                    impact['value_metrics'].append('P/E Ratio')
            else:
                logger.warning("No fundamentals data available")
                impact['value_metrics'].append('P/E Ratio')
            
            # Check for Market Risk metrics
            if fundamentals:
                if 'beta' in fundamentals:
                    beta_value = fundamentals['beta']
                    logger.info(f"Beta value found: {beta_value}")
                    if beta_value is None or pd.isna(beta_value):
                        logger.warning("Beta is present but has no valid value")
                        impact['risk_metrics'].append('Market Risk')
                else:
                    logger.warning("Beta not found in fundamentals")
                    impact['risk_metrics'].append('Market Risk')
            else:
                logger.warning("No fundamentals data available for market risk")
                impact['risk_metrics'].append('Market Risk')
            
            # Check impact on other value metrics
            if not availability.get('cash_flow', {}).get('available', False):
                impact['value_metrics'].extend(['Free Cash Flow', 'Intrinsic Value (DCF)'])
            elif 'Free Cash Flow' in availability.get('cash_flow', {}).get('missing_columns', []):
                impact['value_metrics'].extend(['Free Cash Flow', 'Intrinsic Value (DCF)'])
            
            # Check impact on health metrics
            if not availability.get('balance_sheet', {}).get('available', False):
                impact['health_metrics'].extend(['Current Ratio', 'Debt-to-Equity Ratio', 'Altman Z-Score'])
            elif any(col in availability.get('balance_sheet', {}).get('missing_columns', []) 
                    for col in ['Total Assets', 'Total Current Assets', 'Total Liabilities', 'Total Current Liabilities', 'Total Stockholder Equity']):
                impact['health_metrics'].extend(['Current Ratio', 'Debt-to-Equity Ratio', 'Altman Z-Score'])
            
            if not availability.get('income_stmt', {}).get('available', False):
                impact['health_metrics'].extend(['Operating Margin', 'Net Margin', 'Piotroski Score'])
            elif any(col in availability.get('income_stmt', {}).get('missing_columns', [])
                    for col in ['Operating Income', 'Net Income']):
                impact['health_metrics'].extend(['Operating Margin', 'Net Margin', 'Piotroski Score'])
            
            # Check impact on growth metrics
            if not availability.get('income_stmt', {}).get('available', False):
                impact['growth_metrics'].append('Revenue Growth')
            elif 'Total Revenue' in availability.get('income_stmt', {}).get('missing_columns', []):
                impact['growth_metrics'].append('Revenue Growth')
            
            if not availability.get('income_stmt', {}).get('available', False):
                impact['growth_metrics'].extend(['Earnings Growth', 'EPS Growth'])
            elif 'Net Income' in availability.get('income_stmt', {}).get('missing_columns', []):
                impact['growth_metrics'].extend(['Earnings Growth', 'EPS Growth'])
            
            # Check impact on other risk metrics
            if not availability.get('balance_sheet', {}).get('available', False):
                impact['risk_metrics'].extend(['Financial Risk', 'Business Risk'])
            elif any(col in availability.get('balance_sheet', {}).get('missing_columns', [])
                    for col in ['Total Assets', 'Total Liabilities', 'Total Stockholder Equity']):
                impact['risk_metrics'].extend(['Financial Risk', 'Business Risk'])
            
            logger.info("Completed impact analysis")
            logger.info(f"Impact summary: {impact}")
            return impact
            
        except Exception as e:
            logger.error(f"Error analyzing impact: {str(e)}", exc_info=True)
            return {
                'value_metrics': [],
                'health_metrics': [],
                'growth_metrics': [],
                'risk_metrics': []
            }
        
    def _generate_recommendations(self, availability: Dict[str, Dict], impact: Dict[str, List[str]]) -> List[str]:
        """
        Generate detailed recommendations for improving data collection.
        
        Args:
            availability: Dictionary containing data availability status
            impact: Dictionary containing affected metrics and analyses
            
        Returns:
            List of specific and actionable recommendations
        """
        recommendations = []
        
        # Check for missing statements
        for statement, status in availability.items():
            if not status.get('available', False):
                recommendations.append(
                    f"Collect {statement.replace('_', ' ').title()} data to enable "
                    f"calculation of {', '.join(impact.get(statement.replace('_', ' ').lower(), []))}"
                )
        
        # Check for missing columns and their impact
        for statement, status in availability.items():
            missing_columns = status.get('missing_columns', [])
            if missing_columns:
                # Group missing columns by their impact
                value_impact = [col for col in missing_columns 
                              if any(col in metrics for metrics in impact.get('value_metrics', []))]
                health_impact = [col for col in missing_columns 
                               if any(col in metrics for metrics in impact.get('health_metrics', []))]
                growth_impact = [col for col in missing_columns 
                               if any(col in metrics for metrics in impact.get('growth_metrics', []))]
                
                if value_impact:
                    recommendations.append(
                        f"Add {', '.join(value_impact)} to {statement.replace('_', ' ').title()} "
                        "to improve value metrics calculation"
                    )
                if health_impact:
                    recommendations.append(
                        f"Add {', '.join(health_impact)} to {statement.replace('_', ' ').title()} "
                        "to improve financial health analysis"
                    )
                if growth_impact:
                    recommendations.append(
                        f"Add {', '.join(growth_impact)} to {statement.replace('_', ' ').title()} "
                        "to improve growth metrics calculation"
                    )
        
        # Check for data quality issues
        for statement, status in availability.items():
            if status.get('available', False):
                data_quality = status.get('data_quality_issues', [])
                if data_quality:
                    recommendations.append(
                        f"Review {statement.replace('_', ' ').title()} for {', '.join(data_quality)}"
                    )
        
        # Check for optional columns that could enhance analysis
        for statement, status in availability.items():
            if status.get('available', False) and 'optional_columns' in status:
                missing_optional = [col for col in status.get('optional_columns', []) 
                                  if col not in status.get('available_columns', [])]
                if missing_optional:
                    recommendations.append(
                        f"Consider adding optional columns to {statement.replace('_', ' ').title()}: "
                        f"{', '.join(missing_optional)}"
                    )
        
        # Check for fundamentals
        if availability.get('fundamentals', {}).get('missing_metrics', []):
            recommendations.append(
                "Collect missing fundamental metrics: "
                f"{', '.join(availability['fundamentals']['missing_metrics'])}"
            )
        
        # Add specific recommendations based on impact
        for metric_type, metrics in impact.items():
            if metrics:
                recommendations.append(
                    f"Improve {metric_type.replace('_', ' ')} calculation by collecting: "
                    f"{', '.join(metrics)}"
                )
        
        # Add recommendations for data freshness
        for statement, status in availability.items():
            if status.get('available', False):
                last_date = status.get('last_available_date')
                if last_date:
                    try:
                        last_date = pd.to_datetime(last_date)
                        if (datetime.now() - last_date).days > 90:
                            recommendations.append(
                                f"Update {statement.replace('_', ' ').title()} data "
                                f"(last available: {last_date.strftime('%Y-%m-%d')})"
                            )
                    except Exception as e:
                        logger.warning(f"Error processing date for {statement}: {str(e)}")
        
        return recommendations
        
    def _validate_data_structure(self) -> Dict[str, Dict]:
        """
        Validate the structure of the financial data.
        
        Returns:
            Dict containing validation results for each statement with:
                - is_valid: Whether the data structure is valid
                - errors: List of validation errors
                - warnings: List of validation warnings
        """
        logger.info("Starting data structure validation")
        validation = {
            'income_stmt': {
                'is_valid': False,
                'errors': [],
                'warnings': []
            },
            'balance_sheet': {
                'is_valid': False,
                'errors': [],
                'warnings': []
            },
            'cash_flow': {
                'is_valid': False,
                'errors': [],
                'warnings': []
            },
            'fundamentals': {
                'is_valid': False,
                'errors': [],
                'warnings': []
            }
        }
        
        # Validate fundamentals with enhanced logging
        logger.info("Validating fundamental metrics structure")
        fundamentals = self.financials.get('fundamentals', {})
        if fundamentals:
            try:
                # Check if it's a dictionary
                if not isinstance(fundamentals, dict):
                    validation['fundamentals']['errors'].append(
                        "Fundamentals is not a dictionary"
                    )
                else:
                    # Log all available fundamental metrics
                    logger.info(f"Available fundamental metrics: {list(fundamentals.keys())}")
                    
                    # Define which metrics should be numeric and which should be strings
                    numeric_metrics = [
                        'pe_ratio', 'pb_ratio', 'eps', 'roe', 'market_cap',
                        'dividend_yield', 'beta', 'total_debt', 'total_equity',
                        'current_assets', 'current_liabilities', 'interest_expense',
                        'ebit', 'gross_profit', 'operating_income', 'net_income',
                        'revenue'
                    ]
                    string_metrics = ['sector', 'industry']
                    
                    # Check for required metrics with detailed logging
                    required_metrics = numeric_metrics + string_metrics
                    missing_metrics = [m for m in required_metrics if m not in fundamentals]
                    if missing_metrics:
                        logger.warning(f"Missing required metrics: {missing_metrics}")
                        validation['fundamentals']['errors'].append(
                            f"Missing required metrics: {', '.join(missing_metrics)}"
                        )
                    
                    # Check data types and values with detailed logging
                    for metric, value in fundamentals.items():
                        logger.info(f"Checking metric {metric}: value={value}, type={type(value)}")
                        if value is not None:
                            if metric in numeric_metrics:
                                if not isinstance(value, (int, float)):
                                    logger.warning(f"Metric '{metric}' is not numeric: {value} ({type(value)})")
                                    validation['fundamentals']['warnings'].append(
                                        f"Metric '{metric}' is not numeric: {value} ({type(value)})"
                                    )
                                elif pd.isna(value):
                                    logger.warning(f"Metric '{metric}' is NaN")
                                    validation['fundamentals']['warnings'].append(
                                        f"Metric '{metric}' is NaN"
                                    )
                                else:
                                    logger.info(f"Metric '{metric}' is valid: {value}")
                            elif metric in string_metrics:
                                if not isinstance(value, str):
                                    logger.warning(f"Metric '{metric}' is not a string: {value} ({type(value)})")
                                    validation['fundamentals']['warnings'].append(
                                        f"Metric '{metric}' is not a string: {value} ({type(value)})"
                                    )
                                else:
                                    logger.info(f"Metric '{metric}' is valid: {value}")
                    
                    validation['fundamentals']['is_valid'] = len(validation['fundamentals']['errors']) == 0
            except Exception as e:
                logger.error(f"Error validating fundamental metrics: {str(e)}", exc_info=True)
                validation['fundamentals']['errors'].append(f"Validation error: {str(e)}")
        
        # Validate income statement
        logger.info("Validating income statement structure")
        income_stmt = self.financials.get('income_stmt')
        if income_stmt is not None:
            try:
                # Check if it's a DataFrame
                if not isinstance(income_stmt, pd.DataFrame):
                    validation['income_stmt']['errors'].append(
                        "Income statement is not a pandas DataFrame"
                    )
                else:
                    # Check index type (should be datetime)
                    if not isinstance(income_stmt.index, pd.DatetimeIndex):
                        validation['income_stmt']['errors'].append(
                            "Income statement index is not datetime"
                        )
                    
                    # Check for required columns
                    required_cols = [
                        'Total Revenue', 'Gross Profit', 'Operating Income',
                        'Net Income', 'Basic EPS'
                    ]
                    missing_cols = [col for col in required_cols if col not in income_stmt.columns]
                    if missing_cols:
                        validation['income_stmt']['errors'].append(
                            f"Missing required columns: {', '.join(missing_cols)}"
                        )
                    
                    # Check data types
                    for col in income_stmt.columns:
                        if not pd.api.types.is_numeric_dtype(income_stmt[col]):
                            validation['income_stmt']['warnings'].append(
                                f"Column '{col}' is not numeric"
                            )
                    
                    # Check for missing values
                    missing_values = income_stmt.isnull().sum()
                    if missing_values.any():
                        validation['income_stmt']['warnings'].append(
                            f"Found missing values in columns: {', '.join(missing_values[missing_values > 0].index)}"
                        )
                    
                    validation['income_stmt']['is_valid'] = len(validation['income_stmt']['errors']) == 0
            except Exception as e:
                logger.error(f"Error validating income statement: {str(e)}")
                validation['income_stmt']['errors'].append(f"Validation error: {str(e)}")
        
        # Validate balance sheet
        logger.info("Validating balance sheet structure")
        balance_sheet = self.financials.get('balance_sheet')
        if balance_sheet is not None:
            try:
                # Check if it's a DataFrame
                if not isinstance(balance_sheet, pd.DataFrame):
                    validation['balance_sheet']['errors'].append(
                        "Balance sheet is not a pandas DataFrame"
                    )
                else:
                    # Check index type (should be datetime)
                    if not isinstance(balance_sheet.index, pd.DatetimeIndex):
                        validation['balance_sheet']['errors'].append(
                            "Balance sheet index is not datetime"
                        )
                    
                    # Check for required columns
                    required_cols = [
                        'Total Assets', 'Total Current Assets', 'Total Liabilities',
                        'Total Current Liabilities', 'Total Stockholder Equity'
                    ]
                    missing_cols = [col for col in required_cols if col not in balance_sheet.columns]
                    if missing_cols:
                        validation['balance_sheet']['errors'].append(
                            f"Missing required columns: {', '.join(missing_cols)}"
                        )
                    
                    # Check data types
                    for col in balance_sheet.columns:
                        if not pd.api.types.is_numeric_dtype(balance_sheet[col]):
                            validation['balance_sheet']['warnings'].append(
                                f"Column '{col}' is not numeric"
                            )
                    
                    # Check for missing values
                    missing_values = balance_sheet.isnull().sum()
                    if missing_values.any():
                        validation['balance_sheet']['warnings'].append(
                            f"Found missing values in columns: {', '.join(missing_values[missing_values > 0].index)}"
                        )
                    
                    validation['balance_sheet']['is_valid'] = len(validation['balance_sheet']['errors']) == 0
            except Exception as e:
                logger.error(f"Error validating balance sheet: {str(e)}")
                validation['balance_sheet']['errors'].append(f"Validation error: {str(e)}")
        
        # Validate cash flow
        logger.info("Validating cash flow statement structure")
        cash_flow = self.financials.get('cash_flow')
        if cash_flow is not None:
            try:
                # Check if it's a DataFrame
                if not isinstance(cash_flow, pd.DataFrame):
                    validation['cash_flow']['errors'].append(
                        "Cash flow statement is not a pandas DataFrame"
                    )
                else:
                    # Check index type (should be datetime)
                    if not isinstance(cash_flow.index, pd.DatetimeIndex):
                        validation['cash_flow']['errors'].append(
                            "Cash flow statement index is not datetime"
                        )
                    
                    # Check for required columns
                    required_cols = [
                        'Operating Cash Flow', 'Capital Expenditure',
                        'Free Cash Flow'
                    ]
                    missing_cols = [col for col in required_cols if col not in cash_flow.columns]
                    if missing_cols:
                        validation['cash_flow']['errors'].append(
                            f"Missing required columns: {', '.join(missing_cols)}"
                        )
                    
                    # Check data types
                    for col in cash_flow.columns:
                        if not pd.api.types.is_numeric_dtype(cash_flow[col]):
                            validation['cash_flow']['warnings'].append(
                                f"Column '{col}' is not numeric"
                            )
                    
                    # Check for missing values
                    missing_values = cash_flow.isnull().sum()
                    if missing_values.any():
                        validation['cash_flow']['warnings'].append(
                            f"Found missing values in columns: {', '.join(missing_values[missing_values > 0].index)}"
                        )
                    
                    validation['cash_flow']['is_valid'] = len(validation['cash_flow']['errors']) == 0
            except Exception as e:
                logger.error(f"Error validating cash flow statement: {str(e)}")
                validation['cash_flow']['errors'].append(f"Validation error: {str(e)}")
        
        logger.info("Completed data structure validation")
        return validation
        
    def _generate_report(self) -> Dict:
        """
        Generate comprehensive data collection report.
        
        Returns:
            Dict containing structured report data with:
                - timestamp: When the report was generated
                - data_availability: Status of each financial statement
                - data_validation: Validation results for each statement
                - impact_analysis: Impact of missing data on metrics
                - recommendations: List of recommendations
                - data_quality_score: Overall data quality score
                - summary: Summary statistics
        """
        try:
            logger.info("Generating data collection report")
            
            # Validate data structure
            validation = self._validate_data_structure()
            
            # Check data availability
            availability = self._check_data_availability()
            
            # Analyze impact
            impact = self._analyze_impact(availability)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(availability, impact)
            
            # Calculate data quality score
            total_required = 0
            total_available = 0
            
            # Define required columns for each statement
            required_columns = {
                'income_stmt': [
                    'Total Revenue', 'Gross Profit', 'Operating Income',
                    'Net Income', 'Basic EPS'
                ],
                'balance_sheet': [
                    'Total Assets', 'Total Current Assets', 'Total Liabilities',
                    'Total Current Liabilities', 'Total Stockholder Equity'
                ],
                'cash_flow': [
                    'Operating Cash Flow', 'Capital Expenditure',
                    'Free Cash Flow'
                ],
                'fundamentals': [
                    'pe_ratio', 'pb_ratio', 'eps', 'roe', 'market_cap',
                    'dividend_yield', 'beta'
                ]
            }
            
            # Calculate quality score
            for statement, status in availability.items():
                if statement in required_columns:
                    if statement == 'fundamentals':
                        # For fundamentals, check each required metric
                        for metric in required_columns[statement]:
                            total_required += 1
                            if metric in self.financials.get('fundamentals', {}):
                                total_available += 1
                    else:
                        # For financial statements, check each required column
                        df = self.financials.get(statement)
                        if df is not None and not df.empty:
                            for col in required_columns[statement]:
                                total_required += 1
                                if col in df.columns:
                                    total_available += 1
            
            # Calculate quality score as percentage of available required data
            quality_score = (total_available / total_required * 100) if total_required > 0 else 0
            logger.info(f"Calculated data quality score: {quality_score:.1f}% (Available: {total_available}, Required: {total_required})")
            
            # Compile report
            report = {
                'timestamp': datetime.now().isoformat(),
                'data_availability': availability,
                'data_validation': validation,
                'impact_analysis': impact,
                'recommendations': recommendations,
                'data_quality_score': quality_score,
                'summary': {
                    'total_statements': len(availability),
                    'available_statements': sum(1 for status in availability.values() if status.get('available', False)),
                    'total_required_columns': total_required,
                    'missing_columns': total_required - total_available,
                    'affected_metrics': sum(len(metrics) for metrics in impact.values())
                }
            }
            
            logger.info(f"Successfully generated data collection report with quality score: {quality_score:.1f}%")
            return report
            
        except Exception as e:
            logger.error(f"Error generating data collection report: {str(e)}", exc_info=True)
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'Error',
                'data_quality_score': 0,
                'summary': {
                    'total_statements': 0,
                    'available_statements': 0,
                    'total_required_columns': 0,
                    'missing_columns': 0,
                    'affected_metrics': 0
                }
            }
            
    def get_report(self) -> Dict:
        """
        Get the generated data collection report.
        
        Returns:
            Dict containing the complete report
        """
        return self.report_data
        
    def get_summary(self) -> Dict:
        """
        Get a summary of the data collection status.
        
        Returns:
            Dict containing summary information
        """
        return self.report_data.get('summary', {})
        
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for improving data collection.
        
        Returns:
            List of recommendations
        """
        return self.report_data.get('recommendations', [])
        
    def to_json(self) -> str:
        """
        Convert the report to JSON format.
        
        Returns:
            str: JSON string representation of the report
        """
        return json.dumps(self.report_data, indent=2) 