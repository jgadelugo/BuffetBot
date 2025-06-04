"""
Enhanced options analysis with custom settings support.

This module provides wrapper functions for the options analysis that can
apply custom scoring weights and advanced settings from the dashboard.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from buffetbot.analysis.options.config.scoring_weights import get_scoring_weights
from buffetbot.analysis.options.core.domain_models import StrategyType
from buffetbot.analysis.options_advisor import OptionsAdvisorError
from buffetbot.analysis.options_advisor import analyze_options_strategy as base_analyze
from buffetbot.analysis.options_advisor import get_scoring_weights as get_base_weights
from buffetbot.analysis.options_advisor import update_scoring_weights
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_options_with_custom_settings(
    strategy_type: str, ticker: str, analysis_settings: dict[str, Any]
) -> pd.DataFrame:
    """
    Analyze options with custom settings and scoring weights.

    Args:
        strategy_type: Options strategy to analyze
        ticker: Stock ticker symbol
        analysis_settings: Dictionary containing all analysis settings

    Returns:
        pd.DataFrame: Analysis results with custom settings applied

    Raises:
        OptionsAdvisorError: If analysis fails
    """
    logger.info(
        f"Starting enhanced options analysis for {ticker} with strategy {strategy_type}"
    )

    # Extract settings
    min_days = analysis_settings.get("min_days", 180)
    top_n = analysis_settings.get("top_n", 5)
    risk_tolerance = analysis_settings.get("risk_tolerance", "Conservative")
    time_horizon = analysis_settings.get("time_horizon", "Medium-term (3-6 months)")
    use_custom_weights = analysis_settings.get("use_custom_weights", False)
    custom_weights = analysis_settings.get("custom_scoring_weights", {})

    try:
        # Apply custom scoring weights if enabled
        original_weights = None
        if use_custom_weights and custom_weights:
            logger.info(f"Applying custom scoring weights: {custom_weights}")

            # Store original weights for restoration
            original_weights = get_base_weights()

            # Update global scoring weights
            update_scoring_weights(custom_weights)

        # Perform the analysis with current settings
        recommendations = base_analyze(
            strategy_type=strategy_type,
            ticker=ticker,
            min_days=min_days,
            top_n=top_n,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
        )

        # Apply additional filtering based on custom settings
        if not recommendations.empty:
            recommendations = apply_custom_filtering(recommendations, analysis_settings)

        # Add metadata about settings used
        if not recommendations.empty:
            recommendations = add_analysis_metadata(recommendations, analysis_settings)

        logger.info(
            f"Enhanced analysis completed for {ticker}, returned {len(recommendations)} recommendations"
        )
        return recommendations

    except Exception as e:
        logger.error(f"Enhanced options analysis failed for {ticker}: {str(e)}")
        raise OptionsAdvisorError(f"Enhanced analysis failed: {str(e)}")

    finally:
        # Restore original weights if they were modified
        if original_weights is not None:
            logger.info("Restoring original scoring weights")
            update_scoring_weights(original_weights)


def apply_custom_filtering(
    recommendations: pd.DataFrame, analysis_settings: dict[str, Any]
) -> pd.DataFrame:
    """
    Apply custom filtering based on advanced settings.

    Args:
        recommendations: Original recommendations DataFrame
        analysis_settings: Dictionary containing filtering settings

    Returns:
        pd.DataFrame: Filtered recommendations
    """
    filtered_recs = recommendations.copy()

    # Apply delta threshold filtering
    delta_threshold = analysis_settings.get("delta_threshold")
    if delta_threshold is not None and "Delta" in filtered_recs.columns:
        initial_count = len(filtered_recs)
        filtered_recs = filtered_recs[filtered_recs["Delta"] <= delta_threshold]
        if len(filtered_recs) < initial_count:
            logger.info(
                f"Delta filtering: {initial_count} -> {len(filtered_recs)} recommendations"
            )

    # Apply volume threshold filtering
    volume_threshold = analysis_settings.get("volume_threshold")
    if volume_threshold is not None and "Volume" in filtered_recs.columns:
        initial_count = len(filtered_recs)
        filtered_recs = filtered_recs[filtered_recs["Volume"] >= volume_threshold]
        if len(filtered_recs) < initial_count:
            logger.info(
                f"Volume filtering: {initial_count} -> {len(filtered_recs)} recommendations"
            )

    # Apply bid-ask spread filtering
    bid_ask_spread = analysis_settings.get("bid_ask_spread")
    if (
        bid_ask_spread is not None
        and "Bid" in filtered_recs.columns
        and "Ask" in filtered_recs.columns
    ):
        initial_count = len(filtered_recs)
        spread = filtered_recs["Ask"] - filtered_recs["Bid"]
        filtered_recs = filtered_recs[spread <= bid_ask_spread]
        if len(filtered_recs) < initial_count:
            logger.info(
                f"Bid-ask spread filtering: {initial_count} -> {len(filtered_recs)} recommendations"
            )

    # Apply open interest filtering
    open_interest = analysis_settings.get("open_interest")
    if open_interest is not None and "OpenInterest" in filtered_recs.columns:
        initial_count = len(filtered_recs)
        filtered_recs = filtered_recs[filtered_recs["OpenInterest"] >= open_interest]
        if len(filtered_recs) < initial_count:
            logger.info(
                f"Open interest filtering: {initial_count} -> {len(filtered_recs)} recommendations"
            )

    return filtered_recs


def add_analysis_metadata(
    recommendations: pd.DataFrame, analysis_settings: dict[str, Any]
) -> pd.DataFrame:
    """
    Add metadata columns about the analysis settings used.

    Args:
        recommendations: Recommendations DataFrame
        analysis_settings: Settings used in analysis

    Returns:
        pd.DataFrame: DataFrame with metadata columns added
    """
    enhanced_recs = recommendations.copy()

    # Add settings metadata
    enhanced_recs["CustomWeights"] = (
        "Yes" if analysis_settings.get("use_custom_weights", False) else "No"
    )
    enhanced_recs["RiskTolerance"] = analysis_settings.get(
        "risk_tolerance", "Conservative"
    )
    enhanced_recs["TimeHorizon"] = analysis_settings.get("time_horizon", "Medium-term")

    # Add filtering metadata
    filtering_applied = []
    if analysis_settings.get("delta_threshold") is not None:
        filtering_applied.append(f"Delta≤{analysis_settings['delta_threshold']}")
    if analysis_settings.get("volume_threshold") is not None:
        filtering_applied.append(f"Vol≥{analysis_settings['volume_threshold']}")
    if analysis_settings.get("bid_ask_spread") is not None:
        filtering_applied.append(f"Spread≤{analysis_settings['bid_ask_spread']}")
    if analysis_settings.get("open_interest") is not None:
        filtering_applied.append(f"OI≥{analysis_settings['open_interest']}")

    enhanced_recs["FilteringApplied"] = (
        "; ".join(filtering_applied) if filtering_applied else "None"
    )

    return enhanced_recs


def get_strategy_specific_weights(strategy_type: str) -> dict[str, float]:
    """
    Get strategy-specific default weights.

    Args:
        strategy_type: Strategy type string

    Returns:
        Dict[str, float]: Strategy-specific weights
    """
    try:
        strategy_enum = StrategyType(strategy_type)
        weights_obj = get_scoring_weights(strategy_enum)
        return weights_obj.to_dict()
    except Exception as e:
        logger.error(f"Error getting strategy weights for {strategy_type}: {e}")
        # Return balanced weights as fallback
        return {
            "rsi": 0.20,
            "beta": 0.20,
            "momentum": 0.20,
            "iv": 0.20,
            "forecast": 0.20,
        }


def validate_custom_weights(weights: dict[str, float]) -> tuple[bool, str]:
    """
    Validate custom scoring weights.

    Args:
        weights: Dictionary of weights to validate

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    required_keys = {"rsi", "beta", "momentum", "iv", "forecast"}

    # Check if all required keys are present
    if not required_keys.issubset(weights.keys()):
        missing = required_keys - weights.keys()
        return False, f"Missing required weights: {missing}"

    # Check if values are valid
    for key, value in weights.items():
        if not isinstance(value, (int, float)):
            return False, f"Weight for {key} must be a number"
        if not 0 <= value <= 1:
            return False, f"Weight for {key} must be between 0 and 1"

    # Check if weights sum to approximately 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        return False, f"Weights must sum to 1.0 (current sum: {total:.3f})"

    return True, ""


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Normalize weights to sum to 1.0.

    Args:
        weights: Dictionary of weights

    Returns:
        Dict[str, float]: Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        # Return equal weights if total is 0
        num_weights = len(weights)
        return {k: 1.0 / num_weights for k in weights.keys()}

    return {k: v / total for k, v in weights.items()}
