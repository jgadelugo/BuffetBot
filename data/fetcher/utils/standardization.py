"""
Data standardization utilities for financial statements.
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches
from typing import Dict, List, Optional, Union
from ..mappings.column_mappings import COLUMN_MAPPINGS
from utils.errors import DataError, ErrorSeverity
from utils.logger import setup_logger

logger = setup_logger(__name__)

def standardize_column_names(df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """
    Standardize column names across different financial statements.
    
    Args:
        df: DataFrame containing financial statement data
        statement_type: Type of financial statement ('income', 'balance', or 'cash_flow')
        
    Returns:
        DataFrame with standardized column names
    """
    logger.info(f"Starting standardization for {statement_type} statement")
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Input DataFrame index type: {type(df.index)}")
    
    if df.empty:
        logger.warning(f"Empty DataFrame received for {statement_type} statement")
        return df
        
    # Get the appropriate mapping dictionary
    mappings = COLUMN_MAPPINGS.get(statement_type, {})
    if not mappings:
        error_msg = f"Invalid statement type: {statement_type}"
        logger.error(error_msg)
        raise DataError(error_msg, ErrorSeverity.ERROR)
    
    # Create a new DataFrame with standardized columns
    standardized_df = pd.DataFrame(index=df.index)
    logger.info(f"Created empty standardized DataFrame with index: {type(standardized_df.index)}")
    
    # Create a mapping of all possible variations to standard names
    name_mapping = {}
    for std_name, variations in mappings.items():
        name_mapping[std_name] = std_name  # Add the standard name itself
        for var in variations:
            name_mapping[var] = std_name
            name_mapping[var.lower()] = std_name
            name_mapping[var.upper()] = std_name
            for sep in [' ', '_', '-', '.']:
                name_mapping[var.replace(' ', sep)] = std_name
                name_mapping[var.replace(' ', sep).lower()] = std_name
                name_mapping[var.replace(' ', sep).upper()] = std_name
    
    logger.info(f"Created name mapping with {len(name_mapping)} variations")
    
    # Map each column to its standard name
    logger.info(f"Mapping columns for {statement_type} statement:")
    for col in df.columns:
        # Skip timestamp columns
        if isinstance(col, pd.Timestamp):
            logger.debug(f"Skipping timestamp column: {col}")
            continue
            
        # Try exact match first
        if col in name_mapping:
            std_name = name_mapping[col]
            if std_name in standardized_df.columns:
                standardized_df[std_name] = standardized_df[std_name].fillna(df[col])
                logger.info(f"  Combined values for '{std_name}' from '{col}'")
            else:
                standardized_df[std_name] = df[col]
                logger.info(f"  Added new column '{std_name}' from '{col}'")
            continue
            
        # Try fuzzy matching
        matches = get_close_matches(col, name_mapping.keys(), n=1, cutoff=0.8)
        if matches:
            std_name = name_mapping[matches[0]]
            if std_name in standardized_df.columns:
                standardized_df[std_name] = standardized_df[std_name].fillna(df[col])
                logger.info(f"  Fuzzy mapping: '{col}' -> '{std_name}' (via '{matches[0]}')")
            else:
                standardized_df[std_name] = df[col]
                logger.info(f"  Fuzzy mapping: '{col}' -> '{std_name}' (via '{matches[0]}')")
        else:
            # If no match found, preserve the original column
            standardized_df[col] = df[col]
            logger.warning(f"  No mapping found for '{col}'")
    
    # Log unmapped columns
    unmapped = [col for col in df.columns if not isinstance(col, pd.Timestamp) and col not in name_mapping and not any(col in vars for vars in mappings.values())]
    if unmapped:
        logger.warning(f"Unmapped columns in {statement_type} statement: {unmapped}")
        logger.warning("These columns will be preserved as-is")
    
    # Log fuzzy mapping suggestions
    fuzzy_matches = {}
    for col in unmapped:
        matches = get_close_matches(col, name_mapping.keys(), n=1, cutoff=0.6)
        if matches:
            fuzzy_matches[col] = name_mapping[matches[0]]
    if fuzzy_matches:
        logger.warning(f"Fuzzy mapping suggestions for {statement_type} statement: {fuzzy_matches}")
    
    # Add missing standard columns with default values
    for std_name in mappings.keys():
        if std_name not in standardized_df.columns:
            standardized_df[std_name] = np.nan
            logger.info(f"  Added missing standard column '{std_name}' with NaN values")
    
    logger.info(f"Standardization completed for {statement_type} statement")
    logger.info(f"Final DataFrame shape: {standardized_df.shape}")
    logger.info(f"Final DataFrame columns: {standardized_df.columns.tolist()}")
    
    return standardized_df

def standardize_financial_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Standardize all financial statements in a dataset.
    
    Args:
        data: Dictionary containing financial statements
        
    Returns:
        Dictionary with standardized financial statements
    """
    logger.info("Starting financial data standardization")
    logger.info(f"Input data keys: {list(data.keys())}")
    
    standardized_data = {}
    
    # Standardize each statement
    for statement_type in ['income', 'balance', 'cash_flow']:
        if statement_type in data:
            try:
                df = data[statement_type]
                logger.info(f"Processing {statement_type} statement")
                logger.info(f"DataFrame type: {type(df)}")
                
                if df is None:
                    logger.warning(f"None DataFrame received for {statement_type} statement")
                    standardized_data[statement_type] = pd.DataFrame()
                    continue
                    
                if df.empty:
                    logger.warning(f"Empty DataFrame received for {statement_type} statement")
                    standardized_data[statement_type] = pd.DataFrame()
                    continue
                
                standardized_data[statement_type] = standardize_column_names(
                    df,
                    statement_type
                )
                logger.info(f"Successfully standardized {statement_type} statement")
                
            except Exception as e:
                logger.error(f"Error standardizing {statement_type} statement: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                standardized_data[statement_type] = pd.DataFrame()
    
    logger.info("Financial data standardization completed")
    logger.info(f"Output data keys: {list(standardized_data.keys())}")
    
    return standardized_data 