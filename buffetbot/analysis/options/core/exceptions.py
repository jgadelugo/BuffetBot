"""
Custom exception hierarchy for options advisor module.

This module provides a comprehensive error handling system with meaningful
error messages and proper inheritance hierarchy.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ErrorContext:
    """Context information for errors to aid in debugging and monitoring."""

    ticker: str
    strategy: Optional[str] = None
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    additional_data: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OptionsAdvisorError(Exception):
    """Base exception for options advisor module errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context
        self.message = message

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (Ticker: {self.context.ticker})"
        return self.message


class InsufficientDataError(OptionsAdvisorError):
    """Raised when there's insufficient data for analysis."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        data_points: Optional[int] = None,
        required_points: Optional[int] = None,
    ):
        super().__init__(message, context)
        self.data_points = data_points
        self.required_points = required_points


class StrategyValidationError(OptionsAdvisorError):
    """Raised when strategy validation fails."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        validation_errors: Optional[dict[str, str]] = None,
    ):
        super().__init__(message, context)
        self.validation_errors = validation_errors or {}


class RiskFilteringError(OptionsAdvisorError):
    """Raised when risk filtering fails."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        risk_tolerance: Optional[str] = None,
    ):
        super().__init__(message, context)
        self.risk_tolerance = risk_tolerance


class CalculationError(OptionsAdvisorError):
    """Raised when technical indicator calculations fail."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        calculation_type: Optional[str] = None,
    ):
        super().__init__(message, context)
        self.calculation_type = calculation_type


class DataSourceError(OptionsAdvisorError):
    """Raised when data source operations fail."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        source_name: Optional[str] = None,
    ):
        super().__init__(message, context)
        self.source_name = source_name


class ConfigurationError(OptionsAdvisorError):
    """Raised when configuration validation fails."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        config_key: Optional[str] = None,
    ):
        super().__init__(message, context)
        self.config_key = config_key


class ScoringError(OptionsAdvisorError):
    """Raised when scoring calculations fail."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        scoring_component: Optional[str] = None,
    ):
        super().__init__(message, context)
        self.scoring_component = scoring_component
