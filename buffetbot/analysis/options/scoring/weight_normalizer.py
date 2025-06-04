"""
Weight normalization utilities for scoring.

This module provides utilities for normalizing scoring weights based on
available data sources and ensuring weights sum to 1.0.
"""

import logging
from typing import Dict, List

from buffetbot.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/weight_normalizer.log")


def normalize_weights(
    input_weights: dict[str, float], available_sources: list[str]
) -> dict[str, float]:
    """
    Normalize scoring weights based on available data sources.

    Args:
        input_weights: Original weights dictionary
        available_sources: List of available data sources

    Returns:
        Dict[str, float]: Normalized weights that sum to 1.0

    Raises:
        ValueError: If no available sources provided
    """
    if not available_sources:
        raise ValueError("At least one data source must be available")

    logger.debug(f"Normalizing weights for sources: {available_sources}")

    # Filter weights to only include available sources
    available_weights = {
        k: v for k, v in input_weights.items() if k in available_sources
    }

    if not available_weights:
        # Fallback: equal weights for all available sources
        weight_per_source = 1.0 / len(available_sources)
        normalized = {source: weight_per_source for source in available_sources}
        logger.info(f"No matching weights found, using equal weights: {normalized}")
        return normalized

    # Normalize to sum to 1.0
    total_weight = sum(available_weights.values())
    if total_weight == 0:
        weight_per_source = 1.0 / len(available_sources)
        normalized = {source: weight_per_source for source in available_sources}
        logger.warning(f"Total weight is zero, using equal weights: {normalized}")
        return normalized

    normalized = {k: v / total_weight for k, v in available_weights.items()}

    # Ensure we have entries for all available sources
    for source in available_sources:
        if source not in normalized:
            normalized[source] = 0.0

    # Verify normalization
    total_normalized = sum(normalized.values())
    if abs(total_normalized - 1.0) > 0.001:
        logger.warning(f"Normalized weights sum to {total_normalized}, expected 1.0")

    logger.debug(f"Normalized weights: {normalized}")
    return normalized


def validate_weights(weights: dict[str, float], tolerance: float = 0.001) -> bool:
    """
    Validate that weights sum to approximately 1.0.

    Args:
        weights: Dictionary of weights to validate
        tolerance: Acceptable deviation from 1.0

    Returns:
        bool: True if weights are valid
    """
    if not weights:
        return False

    total = sum(weights.values())
    is_valid = abs(total - 1.0) <= tolerance

    if not is_valid:
        logger.warning(f"Invalid weights sum: {total}, expected 1.0 ± {tolerance}")

    return is_valid


def redistribute_weight(
    weights: dict[str, float], removed_source: str, available_sources: list[str]
) -> dict[str, float]:
    """
    Redistribute weight from a removed source to remaining sources.

    Args:
        weights: Current weights dictionary
        removed_source: Source being removed
        available_sources: Remaining available sources

    Returns:
        Dict[str, float]: Redistributed weights
    """
    if removed_source not in weights:
        return {k: v for k, v in weights.items() if k in available_sources}

    removed_weight = weights[removed_source]
    remaining_weights = {
        k: v
        for k, v in weights.items()
        if k != removed_source and k in available_sources
    }

    if not remaining_weights:
        # All remaining sources get equal weight
        weight_per_source = 1.0 / len(available_sources)
        return {source: weight_per_source for source in available_sources}

    # Redistribute proportionally to achieve total of 1.0
    remaining_total = sum(remaining_weights.values())

    if remaining_total > 0:
        # Scale up the remaining weights to sum to 1.0
        scale_factor = 1.0 / remaining_total
        redistributed = {k: v * scale_factor for k, v in remaining_weights.items()}
    else:
        # Equal redistribution
        weight_per_source = 1.0 / len(available_sources)
        redistributed = {source: weight_per_source for source in available_sources}

    logger.debug(
        f"Redistributed weight {removed_weight} from {removed_source} to {redistributed}"
    )
    return redistributed
