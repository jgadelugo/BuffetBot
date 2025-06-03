from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataError:
    code: str
    message: str
    severity: ErrorSeverity
    details: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class DataFetcherError(Exception):
    def __init__(self, error: DataError):
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class DataValidationError(Exception):
    def __init__(self, error: DataError):
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class DataCleaningError(Exception):
    def __init__(self, error: DataError):
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


def handle_data_error(error: DataError, logger) -> None:
    """Handle data errors with appropriate logging based on severity."""
    log_method = {
        ErrorSeverity.LOW: logger.warning,
        ErrorSeverity.MEDIUM: logger.error,
        ErrorSeverity.HIGH: logger.error,
        ErrorSeverity.CRITICAL: logger.critical,
    }

    log_method[error.severity](
        f"{error.code}: {error.message}",
        extra={
            "error_details": error.details,
            "severity": error.severity.value,
            "timestamp": error.timestamp.isoformat(),
        },
    )
