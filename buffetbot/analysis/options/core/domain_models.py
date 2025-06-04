"""
Core domain models for options analysis.

This module contains type-safe domain objects that represent the core
business concepts and data structures used throughout the options analysis system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class StrategyType(Enum):
    """Supported options strategies."""

    # Single-leg strategies
    LONG_CALLS = "Long Calls"
    LONG_PUTS = "Long Puts"
    COVERED_CALL = "Covered Call"
    CASH_SECURED_PUT = "Cash-Secured Put"

    # Vertical spreads
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    BULL_PUT_SPREAD = "Bull Put Spread"
    BEAR_CALL_SPREAD = "Bear Call Spread"

    # Income strategies
    IRON_CONDOR = "Iron Condor"
    IRON_BUTTERFLY = "Iron Butterfly"
    CALENDAR_SPREAD = "Calendar Spread"

    # Volatility strategies
    LONG_STRADDLE = "Long Straddle"
    LONG_STRANGLE = "Long Strangle"
    SHORT_STRADDLE = "Short Straddle"
    SHORT_STRANGLE = "Short Strangle"


class RiskTolerance(Enum):
    """Risk tolerance levels for strategy analysis."""

    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


class TimeHorizon(Enum):
    """Investment time horizons."""

    SHORT_TERM = "Short-term (1-3 months)"
    MEDIUM_TERM = "Medium-term (3-6 months)"
    LONG_TERM = "Long-term (6+ months)"
    ONE_YEAR = "One Year (12 months)"
    EIGHTEEN_MONTHS = "18 Months (1.5 years)"


@dataclass
class AnalysisRequest:
    """Request object for options strategy analysis."""

    ticker: str
    strategy_type: StrategyType
    min_days: int = 180
    top_n: int = 5
    risk_tolerance: RiskTolerance = RiskTolerance.CONSERVATIVE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Validate request parameters."""
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string")

        if self.min_days <= 0:
            raise ValueError("min_days must be positive")

        if self.top_n <= 0:
            raise ValueError("top_n must be positive")

        # Convert strings to enums if necessary
        if isinstance(self.strategy_type, str):
            self.strategy_type = StrategyType(self.strategy_type)

        if isinstance(self.risk_tolerance, str):
            self.risk_tolerance = RiskTolerance(self.risk_tolerance)

        if isinstance(self.time_horizon, str):
            self.time_horizon = TimeHorizon(self.time_horizon)


@dataclass
class MarketData:
    """Container for market data used in analysis."""

    ticker: str
    stock_prices: pd.Series
    spy_prices: pd.Series
    options_data: pd.DataFrame
    current_price: float
    data_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate market data."""
        if self.stock_prices.empty:
            raise ValueError("Stock prices cannot be empty")

        if self.spy_prices.empty:
            raise ValueError("SPY prices cannot be empty")

        if self.options_data.empty:
            raise ValueError("Options data cannot be empty")

        if self.current_price <= 0:
            raise ValueError("Current price must be positive")


@dataclass
class OptionsData:
    """Container for options data with metadata."""

    options_df: pd.DataFrame
    total_volume: float
    avg_iv: float
    source: str
    fetch_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate options data."""
        if self.options_df.empty:
            raise ValueError("Options DataFrame cannot be empty")

        if self.total_volume < 0:
            raise ValueError("Total volume must be non-negative")

        if self.avg_iv < 0:
            raise ValueError("Average IV must be non-negative")


@dataclass
class TechnicalIndicators:
    """Container for calculated technical indicators."""

    rsi: float
    beta: float
    momentum: float
    avg_iv: float
    forecast_confidence: float
    data_availability: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Validate technical indicators."""
        if not (0 <= self.rsi <= 100):
            raise ValueError("RSI must be between 0 and 100")

        if self.beta < 0:
            raise ValueError("Beta must be non-negative")


@dataclass
class ScoringResult:
    """Result of scoring calculations."""

    composite_score: float
    individual_scores: dict[str, float]
    weights_used: dict[str, float]
    score_details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate scoring result."""
        if not (0 <= self.composite_score <= 1):
            raise ValueError("Composite score must be between 0 and 1")


@dataclass
class StrategyParameters:
    """Strategy-specific parameters."""

    strategy_type: StrategyType
    risk_tolerance: RiskTolerance
    min_days: int
    time_horizon: TimeHorizon
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of options strategy analysis."""

    request: AnalysisRequest
    recommendations: pd.DataFrame
    technical_indicators: TechnicalIndicators
    execution_time_seconds: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate analysis result."""
        if self.execution_time_seconds < 0:
            raise ValueError("Execution time must be non-negative")

    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return not self.recommendations.empty

    @property
    def recommendation_count(self) -> int:
        """Get number of recommendations."""
        return len(self.recommendations)


@dataclass
class RiskProfile:
    """Risk profile configuration for different tolerance levels."""

    max_delta_threshold: float
    min_days_to_expiry: int
    max_moneyness_range: float
    volume_threshold: int
    open_interest_threshold: int
    max_bid_ask_spread: float

    def __post_init__(self):
        """Validate risk profile parameters."""
        if not (0 <= self.max_delta_threshold <= 1):
            raise ValueError("Delta threshold must be between 0 and 1")

        if self.min_days_to_expiry <= 0:
            raise ValueError("Min days to expiry must be positive")


@dataclass
class ScoringWeights:
    """Scoring weights configuration."""

    rsi: float = 0.20
    beta: float = 0.20
    momentum: float = 0.20
    iv: float = 0.20
    forecast: float = 0.20

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = sum([self.rsi, self.beta, self.momentum, self.iv, self.forecast])
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "rsi": self.rsi,
            "beta": self.beta,
            "momentum": self.momentum,
            "iv": self.iv,
            "forecast": self.forecast,
        }
