"""
Data package for financial data fetching and processing.
"""

from .fetcher import DataFetcher, fetch_stock_data
from .fetcher.utils.financial_calculations import (
    calculate_current_ratio,
    calculate_debt_to_equity,
    calculate_gross_margin,
    calculate_net_margin,
    calculate_operating_margin,
    calculate_quick_ratio,
    calculate_roa,
    calculate_roe,
    calculate_rsi,
    calculate_working_capital,
)
from .source_status import (
    get_data_availability_status,
    get_source_health_summary,
    print_data_status,
)

__all__ = [
    "DataFetcher",
    "fetch_stock_data",
    "calculate_rsi",
    "calculate_working_capital",
    "calculate_debt_to_equity",
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_roa",
    "calculate_roe",
    "calculate_gross_margin",
    "calculate_operating_margin",
    "calculate_net_margin",
    "get_data_availability_status",
    "print_data_status",
    "get_source_health_summary",
]
