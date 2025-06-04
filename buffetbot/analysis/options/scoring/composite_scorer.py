"""
Composite scoring engine for options analysis.

This module provides the main scoring orchestrator that combines
technical indicators into composite scores using configurable weights.
"""

import logging
from typing import Any, Dict

import pandas as pd

from buffetbot.utils.logger import setup_logger

from ..core.domain_models import ScoringResult, ScoringWeights, TechnicalIndicators
from ..core.exceptions import ErrorContext, ScoringError
from .weight_normalizer import normalize_weights

logger = setup_logger(__name__, "logs/composite_scorer.log")


class CompositeScorer:
    """Main scoring orchestrator for options analysis."""

    def __init__(self, scoring_weights: ScoringWeights):
        self.scoring_weights = scoring_weights

    def calculate_composite_scores(
        self,
        options_df: pd.DataFrame,
        indicators: TechnicalIndicators,
        strategy_type: str = None,
    ) -> pd.DataFrame:
        """
        Calculate composite scores for options data.

        Args:
            options_df: Options data DataFrame
            indicators: Technical indicators
            strategy_type: Strategy type for logging

        Returns:
            pd.DataFrame: Options data with composite scores

        Raises:
            ScoringError: If scoring calculation fails
        """
        logger.info(f"Calculating composite scores for {len(options_df)} options")

        try:
            scored_df = options_df.copy()

            # Get available data sources
            available_sources = [
                k for k, v in indicators.data_availability.items() if v
            ]

            # Normalize weights based on available data
            weights_dict = self.scoring_weights.to_dict()
            normalized_weights = normalize_weights(weights_dict, available_sources)

            # Calculate individual component scores
            score_components = self._calculate_individual_scores(indicators)

            # Calculate IV scores for each option
            if indicators.data_availability.get("iv", False):
                # Handle both column names for flexibility (original uses 'impliedVolatility')
                iv_column = None
                if "impliedVolatility" in scored_df.columns:
                    iv_column = "impliedVolatility"
                elif "IV" in scored_df.columns:
                    iv_column = "IV"

                if iv_column:
                    scored_df["iv_score"] = scored_df[iv_column].apply(
                        lambda iv: self._normalize_score(
                            iv, indicators.avg_iv * 0.8, indicators.avg_iv * 1.5
                        )
                    )
                else:
                    logger.warning("No IV column found, using neutral IV score")
                    scored_df["iv_score"] = 0.5
            else:
                scored_df["iv_score"] = 0.5

            # Calculate composite score
            scored_df["CompositeScore"] = 0.0

            for source, weight in normalized_weights.items():
                if source == "iv":
                    scored_df["CompositeScore"] += weight * scored_df["iv_score"]
                elif source in score_components:
                    scored_df["CompositeScore"] += weight * score_components[source]

            # Ensure scores are in valid range
            scored_df["CompositeScore"] = scored_df["CompositeScore"].clip(0, 1)

            # Add scoring metadata
            scored_df["score_details"] = scored_df.apply(
                lambda row: {
                    "weights_used": normalized_weights,
                    "available_sources": available_sources,
                    "individual_scores": score_components,
                },
                axis=1,
            )

            logger.info(
                f"Composite scores calculated successfully using sources: {available_sources}"
            )
            return scored_df

        except Exception as e:
            context = ErrorContext(
                ticker=getattr(indicators, "ticker", "unknown"),
                strategy=strategy_type,
                additional_data={"available_sources": available_sources},
            )
            raise ScoringError(
                f"Failed to calculate composite scores: {str(e)}",
                context=context,
                scoring_component="composite",
            )

    def _calculate_individual_scores(
        self, indicators: TechnicalIndicators
    ) -> dict[str, float]:
        """Calculate individual component scores."""
        scores = {}

        if indicators.data_availability.get("rsi", False):
            scores["rsi"] = self._calculate_rsi_score(indicators.rsi)

        if indicators.data_availability.get("beta", False):
            scores["beta"] = self._calculate_beta_score(indicators.beta)

        if indicators.data_availability.get("momentum", False):
            scores["momentum"] = self._calculate_momentum_score(indicators.momentum)

        if indicators.data_availability.get("forecast", False):
            scores["forecast"] = indicators.forecast_confidence

        return scores

    def _calculate_rsi_score(self, rsi: float) -> float:
        """Calculate RSI-based score."""
        # RSI between 30-70 is considered good, with 50 being optimal
        if rsi <= 30:
            return 1.0  # Oversold - good for buying
        elif rsi >= 70:
            return 0.2  # Overbought - not ideal for buying
        else:
            # Linear interpolation between 30-70
            return 1.0 - (rsi - 30) / 40 * 0.8

    def _calculate_beta_score(self, beta: float) -> float:
        """Calculate Beta-based score."""
        # Moderate beta (0.8-1.2) is generally preferred
        if 0.8 <= beta <= 1.2:
            return 1.0
        elif beta < 0.8:
            return 0.7  # Low beta - less volatile but also less upside
        else:
            # High beta - more volatile
            return max(0.3, 1.0 - (beta - 1.2) * 0.2)

    def _calculate_momentum_score(self, momentum: float) -> float:
        """Calculate Momentum-based score."""
        # Positive momentum is good, but not too extreme
        if momentum > 0.1:
            return 0.8  # Very high momentum might indicate overbought
        elif momentum > 0:
            return 1.0  # Positive momentum is ideal
        elif momentum > -0.05:
            return 0.6  # Slight negative momentum is acceptable
        else:
            return 0.2  # Strong negative momentum is not good

    def _normalize_score(
        self, value: float, min_val: float, max_val: float, invert: bool = False
    ) -> float:
        """
        Normalize a value to a 0-1 score.

        Args:
            value: Value to normalize
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            invert: Whether to invert the score (higher value = lower score)

        Returns:
            float: Normalized score between 0 and 1
        """
        if max_val <= min_val:
            return 0.5  # Default neutral score

        # Clamp value to range
        clamped_value = max(min_val, min(max_val, value))

        # Normalize to 0-1
        normalized = (clamped_value - min_val) / (max_val - min_val)

        # Invert if requested
        if invert:
            normalized = 1.0 - normalized

        return normalized


def calculate_strategy_specific_scores(
    options_df: pd.DataFrame,
    indicators: TechnicalIndicators,
    strategy_type: str,
    scoring_weights: ScoringWeights,
) -> pd.DataFrame:
    """
    Calculate strategy-specific composite scores.

    Args:
        options_df: Options data DataFrame
        indicators: Technical indicators
        strategy_type: Strategy type for customization
        scoring_weights: Strategy-specific weights

    Returns:
        pd.DataFrame: Options data with strategy-specific scores
    """
    scorer = CompositeScorer(scoring_weights)
    return scorer.calculate_composite_scores(options_df, indicators, strategy_type)
