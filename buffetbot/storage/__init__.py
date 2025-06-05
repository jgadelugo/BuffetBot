"""
BuffetBot Storage Module

Provides Google Cloud Storage integration for all BuffetBot data types including:
- Market data (prices, volumes, technical indicators)
- Options data (chains, Greeks, implied volatility)
- Forecast data (ML model predictions)
- Ecosystem metrics (sentiment, news analysis)
"""

from .formatters.parquet_formatter import ParquetFormatter
from .gcs.manager import GCSStorageManager
from .query.optimizer import DataQuery, QueryOptimizer
from .schemas.manager import SchemaManager, ValidationResult

__version__ = "1.0.0"

__all__ = [
    "GCSStorageManager",
    "SchemaManager",
    "ValidationResult",
    "ParquetFormatter",
    "QueryOptimizer",
    "DataQuery",
]
