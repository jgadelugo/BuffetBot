"""
Schema Management Module

Data validation, versioning, and schema evolution for all BuffetBot data types.
"""

from .forecasts import FORECAST_SCHEMAS
from .manager import SchemaManager, ValidationError, ValidationResult
from .market_data import MARKET_DATA_SCHEMAS

__all__ = [
    "SchemaManager",
    "ValidationResult",
    "ValidationError",
    "MARKET_DATA_SCHEMAS",
    "FORECAST_SCHEMAS",
]
