"""
Data package for financial data fetching and processing.
"""

from .fetcher import DataFetcher
from .fetcher.utils.financial_calculations import (
    calculate_rsi,
    calculate_working_capital,
    calculate_debt_to_equity,
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_roa,
    calculate_roe,
    calculate_gross_margin,
    calculate_operating_margin,
    calculate_net_margin
)

__all__ = [
    'DataFetcher',
    'calculate_rsi',
    'calculate_working_capital',
    'calculate_debt_to_equity',
    'calculate_current_ratio',
    'calculate_quick_ratio',
    'calculate_roa',
    'calculate_roe',
    'calculate_gross_margin',
    'calculate_operating_margin',
    'calculate_net_margin'
] 