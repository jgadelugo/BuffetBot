"""
Options Analysis Module

This module provides a modular, extensible architecture for analyzing different
options strategies. It follows enterprise-level software engineering practices
and SOLID principles.

Main entry point: analyze_options_strategy()
"""

from .core.domain_models import (
    AnalysisRequest,
    AnalysisResult,
    MarketData,
    RiskTolerance,
    StrategyType,
    TechnicalIndicators,
    TimeHorizon,
)
from .core.exceptions import (
    InsufficientDataError,
    OptionsAdvisorError,
    RiskFilteringError,
    StrategyValidationError,
)
from .core.strategy_dispatcher import analyze_options_strategy

__all__ = [
    # Main entry point
    "analyze_options_strategy",
    # Domain models
    "AnalysisRequest",
    "AnalysisResult",
    "RiskTolerance",
    "StrategyType",
    "TimeHorizon",
    "MarketData",
    "TechnicalIndicators",
    # Exceptions
    "OptionsAdvisorError",
    "InsufficientDataError",
    "StrategyValidationError",
    "RiskFilteringError",
]

__version__ = "2.0.0"
