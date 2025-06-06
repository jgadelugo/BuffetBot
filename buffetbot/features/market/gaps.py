"""
Gap Analysis Module

Professional gap detection and analysis for market structure understanding.
Identifies different types of gaps (common, breakaway, runaway, exhaustion)
and provides gap statistics that are valuable for ML model features.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class GapAnalysis:
    """
    Professional gap analysis for market structure features.

    This class provides comprehensive gap detection, classification,
    and statistical analysis for enhancing ML model performance.
    """

    @staticmethod
    def detect_gaps(
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        min_gap_percent: float = 0.5,
    ) -> pd.DataFrame:
        """
        Detect price gaps in market data.

        A gap occurs when the opening price is significantly different
        from the previous day's closing price, creating a 'gap' in the
        price chart.

        Args:
            open_prices: Series of opening prices
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            min_gap_percent: Minimum gap size as percentage (default: 0.5%)

        Returns:
            DataFrame with gap information including:
            - gap_size: Size of gap in price points
            - gap_percent: Gap size as percentage
            - gap_direction: 'up' or 'down'
            - gap_type: Classification of gap type
            - is_filled: Whether gap has been filled
            - days_to_fill: Days taken to fill gap (if filled)

        Example:
            >>> gaps = GapAnalysis.detect_gaps(open_prices, high, low, close)
            >>> up_gaps = gaps[gaps['gap_direction'] == 'up']
        """
        try:
            # Input validation
            if not all(
                isinstance(series, pd.Series)
                for series in [open_prices, high, low, close]
            ):
                raise ValueError("All inputs must be pandas Series")

            if not all(
                len(series) == len(close) for series in [open_prices, high, low]
            ):
                raise ValueError("All series must have the same length")

            if len(close) < 2:
                logger.warning("Insufficient data for gap analysis")
                return pd.DataFrame()

            # Calculate gaps
            prev_close = close.shift(1)
            gap_size = open_prices - prev_close
            gap_percent = (gap_size / prev_close) * 100

            # Filter significant gaps
            significant_gaps = abs(gap_percent) >= min_gap_percent

            if significant_gaps.sum() == 0:
                logger.info("No significant gaps found")
                return pd.DataFrame()

            # Create gap DataFrame
            gaps_df = pd.DataFrame(
                {
                    "date": open_prices.index,
                    "gap_size": gap_size,
                    "gap_percent": gap_percent,
                    "gap_direction": np.where(gap_size > 0, "up", "down"),
                    "open_price": open_prices,
                    "prev_close": prev_close,
                    "high": high,
                    "low": low,
                    "close": close,
                }
            )

            # Filter for significant gaps
            gaps_df = gaps_df[significant_gaps].copy()

            if len(gaps_df) == 0:
                return gaps_df

            # Classify gap types
            gaps_df["gap_type"] = gaps_df.apply(
                lambda row: GapAnalysis._classify_gap_type(
                    row, close.loc[: row["date"]]
                ),
                axis=1,
            )

            # Check if gaps are filled
            gap_fill_info = []
            for idx, gap_row in gaps_df.iterrows():
                fill_info = GapAnalysis._check_gap_filled(
                    gap_row, high[gap_row["date"] :], low[gap_row["date"] :]
                )
                gap_fill_info.append(fill_info)

            gap_fill_df = pd.DataFrame(gap_fill_info)
            gaps_df = pd.concat([gaps_df, gap_fill_df], axis=1)

            return gaps_df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in gap detection: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _classify_gap_type(gap_row: pd.Series, historical_closes: pd.Series) -> str:
        """
        Classify the type of gap based on market context.

        Gap types:
        - common: Small gaps that occur frequently
        - breakaway: Gaps that break out of trading ranges
        - runaway: Gaps in the middle of strong trends
        - exhaustion: Gaps near the end of trends
        """
        try:
            gap_percent = abs(gap_row["gap_percent"])

            # Get recent price history (last 20 days)
            recent_history = historical_closes.tail(20)

            if len(recent_history) < 10:
                return "common"

            # Calculate trend and volatility context
            price_trend = (
                (recent_history.iloc[-1] - recent_history.iloc[0])
                / recent_history.iloc[0]
                * 100
            )
            price_volatility = recent_history.pct_change().std() * 100

            # Classification logic
            if gap_percent < 1.0:
                return "common"
            elif gap_percent > 5.0 and abs(price_trend) > 10:
                return "exhaustion"
            elif gap_percent > 3.0 and abs(price_trend) > 5:
                return "runaway"
            elif gap_percent > 2.0:
                return "breakaway"
            else:
                return "common"

        except Exception as e:
            logger.warning(f"Error classifying gap type: {str(e)}")
            return "common"

    @staticmethod
    def _check_gap_filled(
        gap_row: pd.Series, future_highs: pd.Series, future_lows: pd.Series
    ) -> dict[str, Union[bool, int, float]]:
        """
        Check if a gap has been filled and calculate fill statistics.
        """
        try:
            gap_direction = gap_row["gap_direction"]
            gap_fill_level = gap_row["prev_close"]

            days_to_fill = None
            is_filled = False
            fill_percentage = 0.0

            if gap_direction == "up":
                # For up gaps, check if price dropped back to previous close
                fill_candidates = future_lows <= gap_fill_level
            else:
                # For down gaps, check if price rose back to previous close
                fill_candidates = future_highs >= gap_fill_level

            if fill_candidates.any():
                is_filled = True
                fill_date = fill_candidates.idxmax()
                days_to_fill = (
                    future_lows.index.get_loc(fill_date)
                    if gap_direction == "up"
                    else future_highs.index.get_loc(fill_date)
                )
                fill_percentage = 100.0
            else:
                # Calculate partial fill percentage
                if gap_direction == "up":
                    closest_approach = future_lows.min()
                    gap_size = gap_row["open_price"] - gap_row["prev_close"]
                    filled_amount = gap_row["open_price"] - closest_approach
                else:
                    closest_approach = future_highs.max()
                    gap_size = gap_row["prev_close"] - gap_row["open_price"]
                    filled_amount = closest_approach - gap_row["open_price"]

                if gap_size > 0:
                    fill_percentage = min(
                        100.0, max(0.0, (filled_amount / gap_size) * 100)
                    )

            return {
                "is_filled": is_filled,
                "days_to_fill": days_to_fill,
                "fill_percentage": fill_percentage,
            }

        except Exception as e:
            logger.warning(f"Error checking gap fill: {str(e)}")
            return {"is_filled": False, "days_to_fill": None, "fill_percentage": 0.0}

    @staticmethod
    def gap_statistics(gaps_df: pd.DataFrame) -> dict[str, float]:
        """
        Calculate comprehensive gap statistics for ML features.

        Args:
            gaps_df: DataFrame from detect_gaps()

        Returns:
            Dictionary with gap statistics including:
            - total_gaps: Total number of gaps
            - up_gaps_ratio: Percentage of upward gaps
            - avg_gap_size: Average gap size percentage
            - gap_fill_rate: Percentage of gaps that get filled
            - avg_days_to_fill: Average days for gaps to be filled
            - gap_volatility: Standard deviation of gap sizes

        Example:
            >>> stats = GapAnalysis.gap_statistics(gaps_df)
            >>> print(f"Gap fill rate: {stats['gap_fill_rate']:.1f}%")
        """
        try:
            if gaps_df.empty:
                return {
                    "total_gaps": 0,
                    "up_gaps_ratio": 0.0,
                    "down_gaps_ratio": 0.0,
                    "avg_gap_size": 0.0,
                    "gap_fill_rate": 0.0,
                    "avg_days_to_fill": 0.0,
                    "gap_volatility": 0.0,
                    "largest_gap": 0.0,
                    "common_gaps_ratio": 0.0,
                    "breakaway_gaps_ratio": 0.0,
                }

            total_gaps = len(gaps_df)
            up_gaps = (gaps_df["gap_direction"] == "up").sum()
            down_gaps = (gaps_df["gap_direction"] == "down").sum()

            # Basic statistics
            stats = {
                "total_gaps": total_gaps,
                "up_gaps_ratio": (up_gaps / total_gaps) * 100,
                "down_gaps_ratio": (down_gaps / total_gaps) * 100,
                "avg_gap_size": abs(gaps_df["gap_percent"]).mean(),
                "gap_volatility": abs(gaps_df["gap_percent"]).std(),
                "largest_gap": abs(gaps_df["gap_percent"]).max(),
            }

            # Fill statistics
            filled_gaps = gaps_df["is_filled"].sum()
            stats["gap_fill_rate"] = (filled_gaps / total_gaps) * 100

            filled_gaps_df = gaps_df[gaps_df["is_filled"] == True]
            if not filled_gaps_df.empty:
                stats["avg_days_to_fill"] = filled_gaps_df["days_to_fill"].mean()
            else:
                stats["avg_days_to_fill"] = 0.0

            # Gap type distribution
            gap_types = gaps_df["gap_type"].value_counts()
            stats["common_gaps_ratio"] = (gap_types.get("common", 0) / total_gaps) * 100
            stats["breakaway_gaps_ratio"] = (
                gap_types.get("breakaway", 0) / total_gaps
            ) * 100

            return stats

        except Exception as e:
            logger.error(f"Error calculating gap statistics: {str(e)}")
            return {}

    @staticmethod
    def recent_gap_features(
        gaps_df: pd.DataFrame, lookback_days: int = 30
    ) -> dict[str, float]:
        """
        Extract recent gap features for ML model input.

        Focuses on recent market behavior to capture current gap patterns
        that may influence future price movements.

        Args:
            gaps_df: DataFrame from detect_gaps()
            lookback_days: Number of recent days to analyze (default: 30)

        Returns:
            Dictionary with recent gap features:
            - recent_gaps_count: Number of gaps in lookback period
            - recent_gap_intensity: Average gap size in recent period
            - gap_momentum: Trend in gap frequency
            - unfilled_gaps_nearby: Number of unfilled gaps close to current price

        Example:
            >>> recent_features = GapAnalysis.recent_gap_features(gaps_df, 20)
            >>> gap_intensity = recent_features['recent_gap_intensity']
        """
        try:
            if gaps_df.empty:
                return {
                    "recent_gaps_count": 0,
                    "recent_gap_intensity": 0.0,
                    "gap_momentum": 0.0,
                    "unfilled_gaps_nearby": 0,
                    "days_since_last_gap": 999,
                    "recent_up_gap_bias": 0.0,
                }

            # Filter for recent gaps
            cutoff_date = gaps_df["date"].max() - pd.Timedelta(days=lookback_days)
            recent_gaps = gaps_df[gaps_df["date"] >= cutoff_date]

            features = {
                "recent_gaps_count": len(recent_gaps),
                "recent_gap_intensity": abs(recent_gaps["gap_percent"]).mean()
                if not recent_gaps.empty
                else 0.0,
                "unfilled_gaps_nearby": (recent_gaps["is_filled"] == False).sum(),
            }

            # Gap momentum (trend in gap frequency)
            if len(recent_gaps) >= 2:
                recent_gaps_sorted = recent_gaps.sort_values("date")
                gap_dates = pd.to_datetime(recent_gaps_sorted["date"])
                gap_frequency = gap_dates.diff().dt.days.mean()
                features["gap_momentum"] = (
                    1.0 / gap_frequency if gap_frequency > 0 else 0.0
                )
            else:
                features["gap_momentum"] = 0.0

            # Days since last gap
            if not gaps_df.empty:
                last_gap_date = gaps_df["date"].max()
                current_date = gaps_df["date"].max()  # Assuming this is current
                features[
                    "days_since_last_gap"
                ] = 0  # In real implementation, use actual current date
            else:
                features["days_since_last_gap"] = 999

            # Recent gap direction bias
            if not recent_gaps.empty:
                up_gaps_recent = (recent_gaps["gap_direction"] == "up").sum()
                features["recent_up_gap_bias"] = (
                    up_gaps_recent / len(recent_gaps)
                ) * 100
            else:
                features["recent_up_gap_bias"] = 50.0  # Neutral

            return features

        except Exception as e:
            logger.error(f"Error calculating recent gap features: {str(e)}")
            return {}


def analyze_gap_patterns(
    open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, Union[pd.DataFrame, dict]]:
    """
    Comprehensive gap pattern analysis for ML features.

    This function combines all gap analysis methods to provide a complete
    set of gap-based features for machine learning models.

    Args:
        open_prices: Series of opening prices
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices

    Returns:
        Dictionary containing:
        - gaps_df: DataFrame with detailed gap information
        - statistics: Overall gap statistics
        - recent_features: Recent gap pattern features

    Example:
        >>> analysis = analyze_gap_patterns(open_prices, high, low, close)
        >>> ml_features = analysis['recent_features']
    """
    try:
        # Detect gaps
        gaps_df = GapAnalysis.detect_gaps(open_prices, high, low, close)

        # Calculate statistics
        statistics = GapAnalysis.gap_statistics(gaps_df)

        # Extract recent features
        recent_features = GapAnalysis.recent_gap_features(gaps_df)

        return {
            "gaps_df": gaps_df,
            "statistics": statistics,
            "recent_features": recent_features,
        }

    except Exception as e:
        logger.error(f"Error in comprehensive gap analysis: {str(e)}")
        return {"gaps_df": pd.DataFrame(), "statistics": {}, "recent_features": {}}
