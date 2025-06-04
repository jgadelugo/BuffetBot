"""
Strategy-specific configuration for options analysis.

This module provides configuration settings that are specific to each
options strategy type.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.domain_models import RiskTolerance, StrategyType


@dataclass
class StrategyConfig:
    """Configuration for a specific options strategy."""

    name: str
    description: str
    default_min_days: int
    strategy_specific_params: dict[str, Any]
    risk_adjustments: dict[RiskTolerance, dict[str, Any]]

    def get_risk_adjusted_params(self, risk_tolerance: RiskTolerance) -> dict[str, Any]:
        """Get risk-adjusted parameters for this strategy."""
        base_params = self.strategy_specific_params.copy()
        risk_params = self.risk_adjustments.get(risk_tolerance, {})
        base_params.update(risk_params)
        return base_params


# Strategy configurations
STRATEGY_CONFIGS = {
    StrategyType.LONG_CALLS: StrategyConfig(
        name="Long Calls",
        description="Bullish strategy with unlimited upside potential",
        default_min_days=180,
        strategy_specific_params={
            "prefer_otm": True,
            "max_delta": 0.80,
            "min_volume": 10,
            "min_open_interest": 25,
        },
        risk_adjustments={
            RiskTolerance.CONSERVATIVE: {
                "max_delta": 0.60,
                "min_volume": 50,
                "min_open_interest": 100,
                "min_days_adjustment": 30,  # Add 30 days to default
            },
            RiskTolerance.MODERATE: {
                "max_delta": 0.70,
                "min_volume": 25,
                "min_open_interest": 50,
                "min_days_adjustment": 0,
            },
            RiskTolerance.AGGRESSIVE: {
                "max_delta": 0.90,
                "min_volume": 5,
                "min_open_interest": 10,
                "min_days_adjustment": -60,  # Allow shorter timeframes
            },
        },
    ),
    StrategyType.BULL_CALL_SPREAD: StrategyConfig(
        name="Bull Call Spread",
        description="Limited risk/reward bullish strategy",
        default_min_days=180,
        strategy_specific_params={
            "spread_width": 5.0,  # Default $5 spread width
            "min_credit_ratio": 0.30,  # Minimum 30% of spread width as credit
            "max_long_delta": 0.70,
            "min_short_delta": 0.30,
        },
        risk_adjustments={
            RiskTolerance.CONSERVATIVE: {
                "spread_width": 2.5,
                "min_credit_ratio": 0.40,
                "max_long_delta": 0.60,
                "min_short_delta": 0.35,
            },
            RiskTolerance.MODERATE: {
                "spread_width": 5.0,
                "min_credit_ratio": 0.30,
                "max_long_delta": 0.70,
                "min_short_delta": 0.30,
            },
            RiskTolerance.AGGRESSIVE: {
                "spread_width": 10.0,
                "min_credit_ratio": 0.20,
                "max_long_delta": 0.80,
                "min_short_delta": 0.20,
            },
        },
    ),
    StrategyType.COVERED_CALL: StrategyConfig(
        name="Covered Call",
        description="Income generation strategy for existing positions",
        default_min_days=30,
        strategy_specific_params={
            "prefer_otm": True,
            "target_delta": 0.30,  # Target 30 delta for covered calls
            "min_annualized_yield": 0.08,  # Minimum 8% annualized yield
            "max_delta": 0.50,
        },
        risk_adjustments={
            RiskTolerance.CONSERVATIVE: {
                "target_delta": 0.20,
                "min_annualized_yield": 0.06,
                "max_delta": 0.30,
            },
            RiskTolerance.MODERATE: {
                "target_delta": 0.30,
                "min_annualized_yield": 0.08,
                "max_delta": 0.50,
            },
            RiskTolerance.AGGRESSIVE: {
                "target_delta": 0.45,
                "min_annualized_yield": 0.12,
                "max_delta": 0.70,
            },
        },
    ),
    StrategyType.CASH_SECURED_PUT: StrategyConfig(
        name="Cash-Secured Put",
        description="Income generation with potential stock acquisition",
        default_min_days=30,
        strategy_specific_params={
            "prefer_otm": True,
            "target_delta": 0.25,  # Target 25 delta for CSPs
            "min_annualized_yield": 0.08,  # Minimum 8% annualized yield
            "max_delta": 0.45,
        },
        risk_adjustments={
            RiskTolerance.CONSERVATIVE: {
                "target_delta": 0.15,
                "min_annualized_yield": 0.06,
                "max_delta": 0.25,
            },
            RiskTolerance.MODERATE: {
                "target_delta": 0.25,
                "min_annualized_yield": 0.08,
                "max_delta": 0.45,
            },
            RiskTolerance.AGGRESSIVE: {
                "target_delta": 0.40,
                "min_annualized_yield": 0.12,
                "max_delta": 0.60,
            },
        },
    ),
}


def get_strategy_config(strategy_type: StrategyType) -> StrategyConfig:
    """
    Get configuration for a specific strategy.

    Args:
        strategy_type: Strategy type to get configuration for

    Returns:
        StrategyConfig: Configuration for the strategy
    """
    return STRATEGY_CONFIGS[strategy_type]


def get_all_supported_strategies() -> list[StrategyType]:
    """
    Get list of all supported strategy types.

    Returns:
        List[StrategyType]: List of supported strategies
    """
    return list(STRATEGY_CONFIGS.keys())
