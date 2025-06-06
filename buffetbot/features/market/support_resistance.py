"""
Support and Resistance Analysis Module

Professional support and resistance level identification using multiple
methods including pivot points, volume profile, and psychological levels.
These levels are crucial for understanding market structure and improving
ML model accuracy.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# Configure logging
logger = logging.getLogger(__name__)


class SupportResistance:
    """
    Professional support and resistance level identification.

    This class provides multiple methods for identifying key price levels
    that act as support (price floors) and resistance (price ceilings).
    """

    @staticmethod
    def pivot_point_levels(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
    ) -> dict[str, pd.Series]:
        """
        Calculate pivot point support and resistance levels.

        Pivot points are calculated using high, low, and close prices over
        a specified period. They provide objective levels where price
        reactions are likely to occur.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Lookback period for pivot calculation (default: 20)

        Returns:
            Dictionary containing:
            - pivot: Main pivot point levels
            - support1: First support level
            - support2: Second support level
            - resistance1: First resistance level
            - resistance2: Second resistance level

        Example:
            >>> levels = SupportResistance.pivot_point_levels(high, low, close)
            >>> current_resistance = levels['resistance1'].iloc[-1]
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if not all(len(series) == len(close) for series in [high, low]):
                raise ValueError("All series must have the same length")

            if len(close) < period:
                logger.warning(
                    f"Insufficient data for pivot points: {len(close)} < {period}"
                )
                empty_series = pd.Series(index=close.index, dtype=float)
                return {
                    "pivot": empty_series,
                    "support1": empty_series,
                    "support2": empty_series,
                    "resistance1": empty_series,
                    "resistance2": empty_series,
                }

            # Calculate rolling pivot points
            high_rolling = high.rolling(window=period)
            low_rolling = low.rolling(window=period)
            close_rolling = close.rolling(window=period)

            # Standard pivot point formula
            pivot = (
                high_rolling.max() + low_rolling.min() + close_rolling.iloc[-1]
            ) / 3

            # Support and resistance levels
            high_max = high_rolling.max()
            low_min = low_rolling.min()

            support1 = 2 * pivot - high_max
            support2 = pivot - (high_max - low_min)
            resistance1 = 2 * pivot - low_min
            resistance2 = pivot + (high_max - low_min)

            return {
                "pivot": pivot,
                "support1": support1,
                "support2": support2,
                "resistance1": resistance1,
                "resistance2": resistance2,
            }

        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            empty_series = pd.Series(index=close.index, dtype=float)
            return {
                "pivot": empty_series,
                "support1": empty_series,
                "support2": empty_series,
                "resistance1": empty_series,
                "resistance2": empty_series,
            }

    @staticmethod
    def psychological_levels(
        close: pd.Series, round_numbers: list[int] = None
    ) -> dict[str, list[float]]:
        """
        Identify psychological support and resistance levels.

        Psychological levels are round numbers that traders pay attention to,
        such as $100, $50, etc. These often act as support or resistance.

        Args:
            close: Series of closing prices
            round_numbers: List of round number intervals to check (default: [1, 5, 10, 25, 50, 100])

        Returns:
            Dictionary containing:
            - nearby_levels: Psychological levels near current price
            - support_levels: Psychological levels below current price
            - resistance_levels: Psychological levels above current price

        Example:
            >>> levels = SupportResistance.psychological_levels(close)
            >>> nearby_resistance = levels['resistance_levels'][:3]  # Top 3
        """
        try:
            if round_numbers is None:
                round_numbers = [1, 5, 10, 25, 50, 100]

            if close.empty:
                return {
                    "nearby_levels": [],
                    "support_levels": [],
                    "resistance_levels": [],
                }

            current_price = close.iloc[-1]
            price_range = close.max() - close.min()

            # Find relevant psychological levels
            psychological_levels = []

            for round_num in round_numbers:
                # Skip if round number is too large for price range
                if round_num > price_range:
                    continue

                # Find levels within reasonable range of current price
                search_range = max(price_range * 0.5, round_num * 10)

                lower_bound = max(0, current_price - search_range)
                upper_bound = current_price + search_range

                # Generate levels
                start_level = int(lower_bound / round_num) * round_num
                level = start_level

                while level <= upper_bound:
                    if level > 0 and lower_bound <= level <= upper_bound:
                        psychological_levels.append(level)
                    level += round_num

            # Remove duplicates and sort
            psychological_levels = sorted(list(set(psychological_levels)))

            # Categorize levels
            support_levels = [
                level for level in psychological_levels if level < current_price
            ]
            resistance_levels = [
                level for level in psychological_levels if level > current_price
            ]

            # Get closest levels
            nearby_levels = []
            if support_levels:
                nearby_levels.extend(support_levels[-3:])  # 3 closest support levels
            if resistance_levels:
                nearby_levels.extend(
                    resistance_levels[:3]
                )  # 3 closest resistance levels

            return {
                "nearby_levels": sorted(nearby_levels),
                "support_levels": sorted(support_levels, reverse=True),  # Closest first
                "resistance_levels": sorted(resistance_levels),  # Closest first
            }

        except Exception as e:
            logger.error(f"Error identifying psychological levels: {str(e)}")
            return {"nearby_levels": [], "support_levels": [], "resistance_levels": []}

    @staticmethod
    def peak_trough_levels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        min_distance: int = 10,
        prominence: float = 0.02,
    ) -> dict[str, pd.DataFrame]:
        """
        Identify support and resistance using peak and trough analysis.

        This method finds significant peaks (resistance) and troughs (support)
        in the price data using scipy's peak detection algorithm.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            min_distance: Minimum distance between peaks/troughs (default: 10)
            prominence: Minimum prominence as fraction of price range (default: 0.02)

        Returns:
            Dictionary containing:
            - resistance_levels: DataFrame with resistance levels and dates
            - support_levels: DataFrame with support levels and dates

        Example:
            >>> levels = SupportResistance.peak_trough_levels(high, low, close)
            >>> recent_resistance = levels['resistance_levels'].tail(5)
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if len(close) < min_distance * 2:
                logger.warning("Insufficient data for peak/trough analysis")
                return {
                    "resistance_levels": pd.DataFrame(
                        columns=["price", "date", "strength"]
                    ),
                    "support_levels": pd.DataFrame(
                        columns=["price", "date", "strength"]
                    ),
                }

            # Calculate prominence threshold
            price_range = close.max() - close.min()
            prominence_threshold = price_range * prominence

            # Find peaks (resistance levels)
            peaks, peak_properties = find_peaks(
                high.values, distance=min_distance, prominence=prominence_threshold
            )

            # Find troughs (support levels)
            troughs, trough_properties = find_peaks(
                -low.values,  # Invert for trough detection
                distance=min_distance,
                prominence=prominence_threshold,
            )

            # Create resistance levels DataFrame
            resistance_data = []
            if len(peaks) > 0:
                for i, peak_idx in enumerate(peaks):
                    resistance_data.append(
                        {
                            "price": high.iloc[peak_idx],
                            "date": high.index[peak_idx],
                            "strength": peak_properties["prominences"][i] / price_range,
                        }
                    )

            resistance_df = pd.DataFrame(resistance_data)

            # Create support levels DataFrame
            support_data = []
            if len(troughs) > 0:
                for i, trough_idx in enumerate(troughs):
                    support_data.append(
                        {
                            "price": low.iloc[trough_idx],
                            "date": low.index[trough_idx],
                            "strength": trough_properties["prominences"][i]
                            / price_range,
                        }
                    )

            support_df = pd.DataFrame(support_data)

            return {
                "resistance_levels": resistance_df.sort_values("date"),
                "support_levels": support_df.sort_values("date"),
            }

        except Exception as e:
            logger.error(f"Error in peak/trough analysis: {str(e)}")
            return {
                "resistance_levels": pd.DataFrame(
                    columns=["price", "date", "strength"]
                ),
                "support_levels": pd.DataFrame(columns=["price", "date", "strength"]),
            }

    @staticmethod
    def volume_profile_levels(
        close: pd.Series, volume: pd.Series, num_levels: int = 10
    ) -> dict[str, pd.DataFrame]:
        """
        Identify support and resistance using volume profile analysis.

        Volume profile shows where most trading activity occurred, which
        often creates strong support and resistance levels.

        Args:
            close: Series of closing prices
            volume: Series of trading volumes
            num_levels: Number of top volume levels to identify (default: 10)

        Returns:
            Dictionary containing:
            - volume_levels: DataFrame with price levels and volume traded
            - support_candidates: Levels below current price
            - resistance_candidates: Levels above current price

        Example:
            >>> levels = SupportResistance.volume_profile_levels(close, volume)
            >>> high_volume_resistance = levels['resistance_candidates'].head(3)
        """
        try:
            # Input validation
            if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
                raise ValueError("close and volume must be pandas Series")

            if len(close) != len(volume):
                raise ValueError("close and volume must have the same length")

            if len(close) < num_levels:
                logger.warning("Insufficient data for volume profile analysis")
                return {
                    "volume_levels": pd.DataFrame(
                        columns=["price", "volume", "volume_percent"]
                    ),
                    "support_candidates": pd.DataFrame(
                        columns=["price", "volume", "volume_percent"]
                    ),
                    "resistance_candidates": pd.DataFrame(
                        columns=["price", "volume", "volume_percent"]
                    ),
                }

            # Create price bins
            price_min = close.min()
            price_max = close.max()
            price_range = price_max - price_min

            # Use appropriate number of bins
            num_bins = min(50, len(close) // 2)
            bin_size = price_range / num_bins

            # Create bins
            bins = np.linspace(price_min, price_max, num_bins + 1)

            # Assign each price to a bin
            price_bins = pd.cut(close, bins=bins, include_lowest=True)

            # Sum volume for each price bin
            volume_profile = volume.groupby(price_bins).sum()

            # Get bin centers as price levels
            bin_centers = [
                (interval.left + interval.right) / 2
                for interval in volume_profile.index
            ]

            # Create volume levels DataFrame
            volume_levels_data = []
            total_volume = volume.sum()

            for i, (interval, vol) in enumerate(volume_profile.items()):
                if not pd.isna(vol) and vol > 0:
                    volume_levels_data.append(
                        {
                            "price": bin_centers[i],
                            "volume": vol,
                            "volume_percent": (vol / total_volume) * 100,
                        }
                    )

            volume_levels_df = pd.DataFrame(volume_levels_data)

            if volume_levels_df.empty:
                empty_df = pd.DataFrame(columns=["price", "volume", "volume_percent"])
                return {
                    "volume_levels": empty_df,
                    "support_candidates": empty_df,
                    "resistance_candidates": empty_df,
                }

            # Sort by volume and get top levels
            volume_levels_df = volume_levels_df.sort_values(
                "volume", ascending=False
            ).head(num_levels)

            # Current price for support/resistance classification
            current_price = close.iloc[-1]

            # Classify as support or resistance
            support_candidates = volume_levels_df[
                volume_levels_df["price"] < current_price
            ].sort_values("price", ascending=False)

            resistance_candidates = volume_levels_df[
                volume_levels_df["price"] > current_price
            ].sort_values("price", ascending=True)

            return {
                "volume_levels": volume_levels_df.sort_values(
                    "volume", ascending=False
                ),
                "support_candidates": support_candidates,
                "resistance_candidates": resistance_candidates,
            }

        except Exception as e:
            logger.error(f"Error in volume profile analysis: {str(e)}")
            empty_df = pd.DataFrame(columns=["price", "volume", "volume_percent"])
            return {
                "volume_levels": empty_df,
                "support_candidates": empty_df,
                "resistance_candidates": empty_df,
            }

    @staticmethod
    def level_strength_score(
        price_level: float,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tolerance: float = 0.01,
    ) -> dict[str, float]:
        """
        Calculate strength score for a support/resistance level.

        Strength is determined by:
        - Number of times price tested the level
        - Volume at those test points
        - Time since last test
        - Price reaction strength

        Args:
            price_level: The support/resistance level to analyze
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            tolerance: Price tolerance as percentage (default: 1%)

        Returns:
            Dictionary with strength metrics:
            - test_count: Number of times level was tested
            - strength_score: Overall strength score (0-100)
            - last_test_days: Days since last test
            - avg_reaction: Average price reaction from level

        Example:
            >>> strength = SupportResistance.level_strength_score(100.0, high, low, close)
            >>> if strength['strength_score'] > 70:
            >>>     print("Strong level")
        """
        try:
            # Input validation
            if not all(isinstance(series, pd.Series) for series in [high, low, close]):
                raise ValueError("high, low, and close must be pandas Series")

            if price_level <= 0:
                raise ValueError("price_level must be positive")

            # Calculate price tolerance range
            tolerance_range = price_level * tolerance
            level_low = price_level - tolerance_range
            level_high = price_level + tolerance_range

            # Find tests of the level
            # Support tests: low prices near the level
            support_tests = (low >= level_low) & (low <= level_high)
            # Resistance tests: high prices near the level
            resistance_tests = (high >= level_low) & (high <= level_high)

            # Combine all tests
            all_tests = support_tests | resistance_tests
            test_count = all_tests.sum()

            if test_count == 0:
                return {
                    "test_count": 0,
                    "strength_score": 0.0,
                    "last_test_days": 999,
                    "avg_reaction": 0.0,
                }

            # Calculate days since last test
            test_dates = close.index[all_tests]
            if len(test_dates) > 0:
                last_test_date = test_dates[-1]
                current_date = close.index[-1]
                last_test_days = (
                    (current_date - last_test_date).days
                    if hasattr(current_date, "days")
                    else 0
                )
            else:
                last_test_days = 999

            # Calculate average reaction strength
            reactions = []
            test_indices = close.index[all_tests]

            for test_idx in test_indices:
                try:
                    idx_loc = close.index.get_loc(test_idx)

                    # Look at price movement after test (next 5 days)
                    future_window = close.iloc[idx_loc : idx_loc + 6]

                    if len(future_window) > 1:
                        price_at_test = close.iloc[idx_loc]
                        max_move = abs(future_window.max() - price_at_test)
                        min_move = abs(future_window.min() - price_at_test)
                        reaction = max(max_move, min_move)
                        reactions.append(
                            reaction / price_at_test * 100
                        )  # As percentage
                except:
                    continue

            avg_reaction = np.mean(reactions) if reactions else 0.0

            # Calculate overall strength score
            # Factors: test count, recency, reaction strength
            test_score = min(test_count * 10, 40)  # Max 40 points for tests
            recency_score = max(
                0, 30 - (last_test_days * 0.5)
            )  # Max 30 points for recency
            reaction_score = min(avg_reaction * 3, 30)  # Max 30 points for reaction

            strength_score = test_score + recency_score + reaction_score

            return {
                "test_count": int(test_count),
                "strength_score": float(min(strength_score, 100.0)),
                "last_test_days": int(last_test_days),
                "avg_reaction": float(avg_reaction),
            }

        except Exception as e:
            logger.error(f"Error calculating level strength: {str(e)}")
            return {
                "test_count": 0,
                "strength_score": 0.0,
                "last_test_days": 999,
                "avg_reaction": 0.0,
            }


def identify_key_levels(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series = None
) -> dict[str, Union[pd.DataFrame, list, dict]]:
    """
    Comprehensive support and resistance level identification.

    Combines multiple methods to identify the most significant support
    and resistance levels for ML model features.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        volume: Optional series of trading volumes

    Returns:
        Dictionary containing:
        - pivot_levels: Pivot point support/resistance levels
        - psychological_levels: Round number levels
        - peak_trough_levels: Technical analysis levels
        - volume_levels: Volume-based levels (if volume provided)
        - consolidated_levels: Combined and ranked levels

    Example:
        >>> levels = identify_key_levels(high, low, close, volume)
        >>> top_resistance = levels['consolidated_levels']['resistance'][:3]
    """
    try:
        results = {}

        # Pivot point levels
        results["pivot_levels"] = SupportResistance.pivot_point_levels(high, low, close)

        # Psychological levels
        results["psychological_levels"] = SupportResistance.psychological_levels(close)

        # Peak and trough levels
        results["peak_trough_levels"] = SupportResistance.peak_trough_levels(
            high, low, close
        )

        # Volume profile levels (if volume data available)
        if volume is not None:
            results["volume_levels"] = SupportResistance.volume_profile_levels(
                close, volume
            )

        # Consolidate all levels
        current_price = close.iloc[-1]
        all_support = []
        all_resistance = []

        # Add pivot levels
        pivot_data = results["pivot_levels"]
        if not pivot_data["support1"].empty:
            latest_support1 = pivot_data["support1"].iloc[-1]
            latest_support2 = pivot_data["support2"].iloc[-1]
            latest_resistance1 = pivot_data["resistance1"].iloc[-1]
            latest_resistance2 = pivot_data["resistance2"].iloc[-1]

            if latest_support1 < current_price:
                all_support.append(
                    {"price": latest_support1, "type": "pivot", "strength": 70}
                )
            if latest_support2 < current_price:
                all_support.append(
                    {"price": latest_support2, "type": "pivot", "strength": 60}
                )
            if latest_resistance1 > current_price:
                all_resistance.append(
                    {"price": latest_resistance1, "type": "pivot", "strength": 70}
                )
            if latest_resistance2 > current_price:
                all_resistance.append(
                    {"price": latest_resistance2, "type": "pivot", "strength": 60}
                )

        # Add psychological levels
        psych_data = results["psychological_levels"]
        for level in psych_data["support_levels"][:5]:  # Top 5
            all_support.append(
                {"price": level, "type": "psychological", "strength": 50}
            )
        for level in psych_data["resistance_levels"][:5]:  # Top 5
            all_resistance.append(
                {"price": level, "type": "psychological", "strength": 50}
            )

        # Add peak/trough levels
        peak_trough_data = results["peak_trough_levels"]
        for _, row in peak_trough_data["support_levels"].tail(10).iterrows():
            if row["price"] < current_price:
                strength = min(90, 50 + row["strength"] * 100)
                all_support.append(
                    {"price": row["price"], "type": "technical", "strength": strength}
                )

        for _, row in peak_trough_data["resistance_levels"].tail(10).iterrows():
            if row["price"] > current_price:
                strength = min(90, 50 + row["strength"] * 100)
                all_resistance.append(
                    {"price": row["price"], "type": "technical", "strength": strength}
                )

        # Sort and deduplicate
        all_support = sorted(all_support, key=lambda x: x["price"], reverse=True)
        all_resistance = sorted(all_resistance, key=lambda x: x["price"])

        # Remove levels too close to each other
        consolidated_support = _consolidate_levels(all_support, tolerance=0.005)
        consolidated_resistance = _consolidate_levels(all_resistance, tolerance=0.005)

        results["consolidated_levels"] = {
            "support": consolidated_support[:10],  # Top 10
            "resistance": consolidated_resistance[:10],  # Top 10
        }

        return results

    except Exception as e:
        logger.error(f"Error in comprehensive level identification: {str(e)}")
        return {}


def _consolidate_levels(levels: list[dict], tolerance: float = 0.005) -> list[dict]:
    """
    Consolidate nearby levels to avoid redundancy.

    Args:
        levels: List of level dictionaries
        tolerance: Price tolerance for consolidation (default: 0.5%)

    Returns:
        List of consolidated levels
    """
    if not levels:
        return []

    consolidated = []

    for level in levels:
        price = level["price"]

        # Check if this level is close to any existing consolidated level
        is_duplicate = False
        for existing in consolidated:
            existing_price = existing["price"]
            price_diff = abs(price - existing_price) / existing_price

            if price_diff <= tolerance:
                # Merge with existing level (keep higher strength)
                if level["strength"] > existing["strength"]:
                    existing.update(level)
                is_duplicate = True
                break

        if not is_duplicate:
            consolidated.append(level.copy())

    # Sort by strength
    return sorted(consolidated, key=lambda x: x["strength"], reverse=True)
