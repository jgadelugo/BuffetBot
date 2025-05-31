"""BuffetBot - A modular Python toolkit for value investing analysis.

This package provides tools for evaluating companies and stocks using
value investing principles inspired by Warren Buffett and Benjamin Graham.
"""

__version__ = "1.0.0"
__author__ = "Jose Alvarez de Lugo"
__email__ = "josepluton+buffetbot@gmail.com"

# Import main components for easier access
from buffetbot.glossary import (
    GLOSSARY,
    search_metrics,
    get_metrics_by_category,
    get_metric_info,
    MetricDefinition,
)

__all__ = [
    "GLOSSARY",
    "search_metrics",
    "get_metrics_by_category", 
    "get_metric_info",
    "MetricDefinition",
] 