"""
Error handling utilities for the data fetcher.
"""

from utils.errors import (
    ErrorSeverity,
    DataError,
    DataFetcherError,
    DataValidationError,
    DataCleaningError,
    handle_data_error
)

__all__ = [
    'ErrorSeverity',
    'DataError',
    'DataFetcherError',
    'DataValidationError',
    'DataCleaningError',
    'handle_data_error'
] 