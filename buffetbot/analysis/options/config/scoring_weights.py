"""
Scoring weights configuration for options analysis.

This module provides centralized, type-safe configuration for scoring weights
used in technical indicator analysis.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..core.domain_models import ScoringWeights, StrategyType


@dataclass
class ScoringWeightsConfig:
    """Configuration for scoring weights across different strategies."""

    default_weights: ScoringWeights
    strategy_specific_weights: dict[StrategyType, ScoringWeights]

    def get_weights_for_strategy(self, strategy_type: StrategyType) -> ScoringWeights:
        """Get scoring weights for a specific strategy."""
        return self.strategy_specific_weights.get(strategy_type, self.default_weights)

    def validate(self) -> None:
        """Validate all scoring weights."""
        # Default weights are validated in ScoringWeights __post_init__
        for strategy, weights in self.strategy_specific_weights.items():
            # Weights are validated in ScoringWeights __post_init__
            pass


# Default configuration
DEFAULT_SCORING_WEIGHTS_CONFIG = ScoringWeightsConfig(
    default_weights=ScoringWeights(),
    strategy_specific_weights={
        StrategyType.LONG_CALLS: ScoringWeights(
            rsi=0.25, beta=0.15, momentum=0.25, iv=0.15, forecast=0.20
        ),
        StrategyType.BULL_CALL_SPREAD: ScoringWeights(
            rsi=0.20, beta=0.20, momentum=0.20, iv=0.20, forecast=0.20
        ),
        StrategyType.COVERED_CALL: ScoringWeights(
            rsi=0.15,
            beta=0.25,  # Lower beta preferred for covered calls
            momentum=0.15,
            iv=0.30,  # Higher IV weight for income strategies
            forecast=0.15,
        ),
        StrategyType.CASH_SECURED_PUT: ScoringWeights(
            rsi=0.25,  # RSI more important for CSPs
            beta=0.20,
            momentum=0.15,
            iv=0.25,  # IV important for income
            forecast=0.15,
        ),
    },
)


def get_scoring_weights(strategy_type: StrategyType = None) -> ScoringWeights:
    """
    Get scoring weights for a strategy type.

    Args:
        strategy_type: Strategy to get weights for. If None, returns default weights.

    Returns:
        ScoringWeights: Configured weights for the strategy
    """
    if strategy_type is None:
        return DEFAULT_SCORING_WEIGHTS_CONFIG.default_weights

    return DEFAULT_SCORING_WEIGHTS_CONFIG.get_weights_for_strategy(strategy_type)


def normalize_scoring_weights(
    input_weights: dict[str, float], available_sources: list[str]
) -> dict[str, float]:
    """
    Normalize scoring weights based on available data sources.

    Args:
        input_weights: Original weights dictionary
        available_sources: List of available data sources

    Returns:
        Dict[str, float]: Normalized weights that sum to 1.0
    """
    # Filter weights to only include available sources
    available_weights = {
        k: v for k, v in input_weights.items() if k in available_sources
    }

    if not available_weights:
        # Fallback: equal weights for all available sources
        weight_per_source = 1.0 / len(available_sources)
        return {source: weight_per_source for source in available_sources}

    # Normalize to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        weight_per_source = 1.0 / len(available_sources)
        return {source: weight_per_source for source in available_sources}

    normalized = {k: v / total_weight for k, v in available_weights.items()}

    # Ensure we have entries for all available sources
    for source in available_sources:
        if source not in normalized:
            normalized[source] = 0.0

    return normalized
