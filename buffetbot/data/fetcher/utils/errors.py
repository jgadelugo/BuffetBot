"""
Error handling utilities for the data fetcher.
"""

# Path setup to ensure proper imports
import sys
from pathlib import Path

# Ensure project root is in path for absolute imports
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from buffetbot.utils.errors import (
    DataCleaningError,
    DataError,
    DataFetcherError,
    DataValidationError,
    ErrorSeverity,
    handle_data_error,
)

__all__ = [
    "ErrorSeverity",
    "DataError",
    "DataFetcherError",
    "DataValidationError",
    "DataCleaningError",
    "handle_data_error",
]
