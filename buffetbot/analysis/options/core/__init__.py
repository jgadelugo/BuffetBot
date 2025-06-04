"""Core module for options analysis architecture."""

from .domain_models import (
    AnalysisRequest,
    AnalysisResult,
    MarketData,
    RiskTolerance,
    ScoringResult,
    StrategyType,
)
from .exceptions import (
    CalculationError,
    InsufficientDataError,
    OptionsAdvisorError,
    RiskFilteringError,
    StrategyValidationError,
)
from .strategy_dispatcher import analyze_options_strategy
from .strategy_registry import StrategyRegistry

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "RiskTolerance",
    "StrategyType",
    "MarketData",
    "ScoringResult",
    "OptionsAdvisorError",
    "InsufficientDataError",
    "StrategyValidationError",
    "RiskFilteringError",
    "CalculationError",
    "analyze_options_strategy",
    "StrategyRegistry",
]
