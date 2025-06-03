"""
Correlation Math Utilities

This module provides mathematical functions for calculating correlations
between stocks and normalizing ecosystem scores for the ecosystem mapping feature.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from buffetbot.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class CorrelationResult:
    """Data class to hold correlation analysis results."""

    ticker1: str
    ticker2: str
    correlation: float
    p_value: float
    sample_size: int
    correlation_type: str = "pearson"

    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant (p < 0.05)."""
        return self.p_value < 0.05

    @property
    def strength_category(self) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"


@dataclass
class EcosystemScore:
    """Data class to hold ecosystem analysis results."""

    ticker: str
    peers: list[str]
    avg_correlation: float
    normalized_score: float
    individual_correlations: dict[str, float]
    confidence_score: float
    sample_size: int


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Series of prices
        method: "simple" or "log" returns

    Returns:
        pd.Series: Series of returns

    Raises:
        ValueError: If method is not supported or prices are invalid
    """
    if prices.empty:
        raise ValueError("Price series cannot be empty")

    if method == "simple":
        returns = prices.pct_change().dropna()
    elif method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unsupported return method: {method}")

    return returns


def calculate_correlation(
    series1: pd.Series,
    series2: pd.Series,
    method: str = "pearson",
    min_periods: int = 30,
) -> CorrelationResult:
    """
    Calculate correlation between two time series.

    Args:
        series1: First time series (e.g., stock returns)
        series2: Second time series (e.g., peer stock returns)
        method: Correlation method ("pearson", "spearman", "kendall")
        min_periods: Minimum number of observations required

    Returns:
        CorrelationResult: Correlation analysis results

    Raises:
        ValueError: If inputs are invalid or insufficient data
    """
    if series1.empty or series2.empty:
        raise ValueError("Input series cannot be empty")

    # Align series by index
    aligned_data = pd.concat([series1, series2], axis=1).dropna()

    if len(aligned_data) < min_periods:
        raise ValueError(
            f"Insufficient data: {len(aligned_data)} < {min_periods} minimum periods"
        )

    s1_aligned = aligned_data.iloc[:, 0]
    s2_aligned = aligned_data.iloc[:, 1]

    # Calculate correlation based on method
    if method == "pearson":
        correlation, p_value = stats.pearsonr(s1_aligned, s2_aligned)
    elif method == "spearman":
        correlation, p_value = stats.spearmanr(s1_aligned, s2_aligned)
    elif method == "kendall":
        correlation, p_value = stats.kendalltau(s1_aligned, s2_aligned)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    # Handle NaN results
    if np.isnan(correlation):
        logger.warning("Correlation calculation resulted in NaN")
        correlation = 0.0
        p_value = 1.0

    return CorrelationResult(
        ticker1=series1.name or "series1",
        ticker2=series2.name or "series2",
        correlation=correlation,
        p_value=p_value,
        sample_size=len(aligned_data),
        correlation_type=method,
    )


def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 30,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Calculate rolling correlation between two series.

    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        min_periods: Minimum periods for correlation calculation

    Returns:
        pd.Series: Rolling correlation values
    """
    if min_periods is None:
        min_periods = max(10, window // 2)

    # Align series
    aligned_data = pd.concat([series1, series2], axis=1).dropna()

    if len(aligned_data) < window:
        logger.warning(
            f"Insufficient data for rolling correlation: {len(aligned_data)} < {window}"
        )
        return pd.Series(dtype=float)

    s1_aligned = aligned_data.iloc[:, 0]
    s2_aligned = aligned_data.iloc[:, 1]

    return s1_aligned.rolling(window=window, min_periods=min_periods).corr(s2_aligned)


def calculate_ecosystem_correlations(
    target_returns: pd.Series,
    peer_returns_dict: dict[str, pd.Series],
    method: str = "pearson",
    min_periods: int = 30,
) -> dict[str, CorrelationResult]:
    """
    Calculate correlations between target stock and all peer stocks.

    Args:
        target_returns: Returns of the target stock
        peer_returns_dict: Dictionary of peer ticker -> returns series
        method: Correlation method
        min_periods: Minimum periods required

    Returns:
        Dict[str, CorrelationResult]: Dictionary of peer ticker -> correlation result
    """
    correlations = {}

    logger.info(
        f"Calculating ecosystem correlations for {len(peer_returns_dict)} peers"
    )

    for peer_ticker, peer_returns in peer_returns_dict.items():
        try:
            correlation_result = calculate_correlation(
                target_returns, peer_returns, method=method, min_periods=min_periods
            )
            correlations[peer_ticker] = correlation_result

            logger.debug(
                f"Correlation with {peer_ticker}: {correlation_result.correlation:.3f} "
                f"(p={correlation_result.p_value:.3f})"
            )

        except Exception as e:
            logger.warning(
                f"Failed to calculate correlation with {peer_ticker}: {str(e)}"
            )
            # Create a default result with zero correlation
            correlations[peer_ticker] = CorrelationResult(
                ticker1=target_returns.name or "target",
                ticker2=peer_ticker,
                correlation=0.0,
                p_value=1.0,
                sample_size=0,
                correlation_type=method,
            )

    return correlations


def normalize_score(
    value: float,
    min_value: float = -1.0,
    max_value: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    Normalize a value to a target range.

    Args:
        value: Value to normalize
        min_value: Minimum possible input value
        max_value: Maximum possible input value
        target_min: Minimum target value
        target_max: Maximum target value

    Returns:
        float: Normalized value in target range
    """
    if max_value == min_value:
        return (target_min + target_max) / 2

    # Clamp value to input range
    clamped_value = max(min_value, min(max_value, value))

    # Normalize to target range
    normalized = (clamped_value - min_value) / (max_value - min_value)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized


def calculate_ecosystem_score(
    correlations: dict[str, CorrelationResult],
    weight_by_significance: bool = True,
    min_sample_size: int = 30,
) -> EcosystemScore:
    """
    Calculate a normalized ecosystem score based on peer correlations.

    Args:
        correlations: Dictionary of peer correlations
        weight_by_significance: Whether to weight by statistical significance
        min_sample_size: Minimum sample size to consider reliable

    Returns:
        EcosystemScore: Comprehensive ecosystem analysis
    """
    if not correlations:
        raise ValueError("No correlations provided")

    target_ticker = next(iter(correlations.values())).ticker1
    peer_tickers = list(correlations.keys())

    # Extract correlation values and weights
    correlation_values = []
    weights = []
    individual_correlations = {}

    for peer_ticker, corr_result in correlations.items():
        correlation = abs(corr_result.correlation)  # Use absolute value for strength
        individual_correlations[peer_ticker] = corr_result.correlation

        # Calculate weight based on significance and sample size
        weight = 1.0

        if weight_by_significance:
            # Higher weight for significant correlations
            if corr_result.is_significant:
                weight *= 1.5
            else:
                weight *= 0.5

        # Weight by sample size reliability
        if corr_result.sample_size >= min_sample_size:
            weight *= 1.0
        else:
            weight *= max(0.1, corr_result.sample_size / min_sample_size)

        correlation_values.append(correlation)
        weights.append(weight)

    # Calculate weighted average correlation
    if sum(weights) > 0:
        avg_correlation = np.average(correlation_values, weights=weights)
    else:
        avg_correlation = np.mean(correlation_values) if correlation_values else 0.0

    # Normalize score to 0-1 range
    # Strong positive correlation (close to 1) should give high ecosystem score
    normalized_score = normalize_score(avg_correlation, 0.0, 1.0, 0.0, 1.0)

    # Calculate confidence score based on:
    # 1. Number of peers
    # 2. Average sample size
    # 3. Proportion of significant correlations

    num_peers = len(correlations)
    avg_sample_size = np.mean([corr.sample_size for corr in correlations.values()])
    significant_ratio = (
        sum(1 for corr in correlations.values() if corr.is_significant) / num_peers
    )

    # Confidence components (each 0-1)
    peer_confidence = min(
        1.0, num_peers / 5
    )  # Confidence increases with more peers (up to 5)
    sample_confidence = min(
        1.0, avg_sample_size / 100
    )  # Confidence with sample size (up to 100)
    significance_confidence = significant_ratio

    # Overall confidence (weighted average)
    confidence_score = (
        0.4 * peer_confidence + 0.3 * sample_confidence + 0.3 * significance_confidence
    )

    ecosystem_score = EcosystemScore(
        ticker=target_ticker,
        peers=peer_tickers,
        avg_correlation=avg_correlation,
        normalized_score=normalized_score,
        individual_correlations=individual_correlations,
        confidence_score=confidence_score,
        sample_size=int(avg_sample_size),
    )

    logger.info(
        f"Ecosystem score for {target_ticker}: {normalized_score:.3f} "
        f"(confidence: {confidence_score:.3f})"
    )

    return ecosystem_score


def detect_correlation_regime_changes(
    correlations: pd.Series, window: int = 20, threshold: float = 0.3
) -> list[tuple[pd.Timestamp, float]]:
    """
    Detect significant changes in correlation patterns.

    Args:
        correlations: Time series of correlation values
        window: Window for detecting changes
        threshold: Minimum change threshold to flag

    Returns:
        List of (timestamp, correlation_change) tuples for regime changes
    """
    if len(correlations) < window * 2:
        return []

    changes = []

    # Calculate rolling mean to smooth out noise
    rolling_mean = correlations.rolling(window=window, center=True).mean()

    # Calculate changes between windows
    for i in range(window, len(rolling_mean) - window):
        prev_window = rolling_mean.iloc[i - window : i].mean()
        next_window = rolling_mean.iloc[i : i + window].mean()

        change = abs(next_window - prev_window)

        if change >= threshold:
            timestamp = correlations.index[i]
            changes.append((timestamp, change))

    return changes


def calculate_portfolio_correlation_matrix(
    returns_dict: dict[str, pd.Series], method: str = "pearson"
) -> pd.DataFrame:
    """
    Calculate correlation matrix for a portfolio of stocks.

    Args:
        returns_dict: Dictionary of ticker -> returns series
        method: Correlation method

    Returns:
        pd.DataFrame: Correlation matrix
    """
    if not returns_dict:
        return pd.DataFrame()

    # Combine all series
    combined_df = pd.DataFrame(returns_dict)

    # Calculate correlation matrix
    if method == "pearson":
        corr_matrix = combined_df.corr()
    elif method == "spearman":
        corr_matrix = combined_df.corr(method="spearman")
    elif method == "kendall":
        corr_matrix = combined_df.corr(method="kendall")
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    return corr_matrix
