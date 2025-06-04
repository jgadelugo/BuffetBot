"""
BuffetBot - A modular Python toolkit for evaluating companies and stocks using value investing principles.

This package provides comprehensive tools for stock analysis, financial data fetching,
and investment decision making based on fundamental analysis principles.
"""

__version__ = "1.0.0"
__author__ = "Jose Alvarez de Lugo"
__email__ = "josepluton+buffetbot@gmail.com"

# Core imports for easy access
from buffetbot.data import fetch_stock_data

# Glossary imports
from buffetbot.glossary import (
    GLOSSARY,
    get_metric_info,
    get_metrics_by_category,
    search_metrics,
)
from buffetbot.utils.errors import DataError, DataValidationError
from buffetbot.utils.logger import get_logger


# Dashboard entry point
def run_dashboard():
    """Launch the BuffetBot dashboard."""
    from buffetbot.dashboard.app import main

    main()


__all__ = [
    "fetch_stock_data",
    "DataError",
    "DataValidationError",
    "get_logger",
    "run_dashboard",
    "GLOSSARY",
    "get_metrics_by_category",
    "search_metrics",
    "get_metric_info",
]
