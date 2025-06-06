"""
Drawdown Analysis Module

Professional drawdown calculation and analysis for risk-based ML features.
Provides maximum drawdown, recovery analysis, underwater curves, and
drawdown clustering that capture portfolio stress behavior.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class DrawdownAnalysis:
    """
    Professional drawdown analysis for risk assessment.

    This class provides comprehensive drawdown metrics including
    maximum drawdown, recovery analysis, and underwater periods.
    """

    @staticmethod
    def calculate_drawdowns(prices: pd.Series) -> dict[str, pd.Series]:
        """
        Calculate comprehensive drawdown metrics from price series.

        Drawdown represents the decline from a peak to a trough in
        the value of an investment, expressed as a percentage.

        Args:
            prices: Series of prices (cumulative returns or price levels)

        Returns:
            Dictionary containing:
            - drawdown: Current drawdown from peak (%)
            - cumulative_max: Running maximum (peak) values
            - underwater: Boolean series indicating underwater periods
            - drawdown_duration: Days in current drawdown

        Example:
            >>> prices = (1 + returns).cumprod()
            >>> dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
            >>> max_dd = dd_metrics['drawdown'].min()
        """
        try:
            # Input validation
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) == 0:
                logger.warning("Empty price series provided")
                empty_series = pd.Series(dtype=float)
                return {
                    "drawdown": empty_series,
                    "cumulative_max": empty_series,
                    "underwater": pd.Series(dtype=bool),
                    "drawdown_duration": pd.Series(dtype=int),
                }

            # Remove any NaN values
            clean_prices = prices.dropna()

            if len(clean_prices) == 0:
                logger.warning("No valid prices after removing NaN")
                empty_series = pd.Series(index=prices.index, dtype=float)
                return {
                    "drawdown": empty_series,
                    "cumulative_max": empty_series,
                    "underwater": pd.Series(index=prices.index, dtype=bool),
                    "drawdown_duration": pd.Series(index=prices.index, dtype=int),
                }

            # Calculate running maximum (peaks)
            cumulative_max = clean_prices.expanding().max()

            # Calculate drawdown as percentage decline from peak
            drawdown = (clean_prices - cumulative_max) / cumulative_max * 100

            # Identify underwater periods (when in drawdown)
            underwater = drawdown < 0

            # Calculate drawdown duration
            drawdown_duration = pd.Series(index=clean_prices.index, dtype=int)
            current_duration = 0

            for i, is_underwater in enumerate(underwater):
                if is_underwater:
                    current_duration += 1
                else:
                    current_duration = 0
                drawdown_duration.iloc[i] = current_duration

            # Expand results to original index with NaN padding
            result_drawdown = pd.Series(index=prices.index, dtype=float)
            result_cummax = pd.Series(index=prices.index, dtype=float)
            result_underwater = pd.Series(index=prices.index, dtype=bool)
            result_duration = pd.Series(index=prices.index, dtype=int)

            result_drawdown.loc[clean_prices.index] = drawdown
            result_cummax.loc[clean_prices.index] = cumulative_max
            result_underwater.loc[clean_prices.index] = underwater
            result_duration.loc[clean_prices.index] = drawdown_duration

            return {
                "drawdown": result_drawdown,
                "cumulative_max": result_cummax,
                "underwater": result_underwater,
                "drawdown_duration": result_duration,
            }

        except Exception as e:
            logger.error(f"Error calculating drawdowns: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                "drawdown": empty_series,
                "cumulative_max": empty_series,
                "underwater": pd.Series(index=prices.index, dtype=bool),
                "drawdown_duration": pd.Series(index=prices.index, dtype=int),
            }

    @staticmethod
    def maximum_drawdown_analysis(
        prices: pd.Series,
    ) -> dict[str, Union[float, pd.Timestamp, int]]:
        """
        Analyze maximum drawdown characteristics.

        Identifies the worst drawdown period and provides detailed
        statistics about the peak-to-trough decline and recovery.

        Args:
            prices: Series of prices

        Returns:
            Dictionary containing:
            - max_drawdown: Maximum drawdown percentage
            - peak_date: Date of peak before max drawdown
            - trough_date: Date of trough (max drawdown point)
            - recovery_date: Date of recovery (if recovered)
            - drawdown_duration: Days from peak to trough
            - recovery_duration: Days from trough to recovery
            - total_duration: Total days from peak to recovery

        Example:
            >>> max_dd_info = DrawdownAnalysis.maximum_drawdown_analysis(prices)
            >>> worst_loss = max_dd_info['max_drawdown']
        """
        try:
            # Calculate basic drawdown metrics
            dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
            drawdown = dd_metrics["drawdown"]
            cumulative_max = dd_metrics["cumulative_max"]

            if drawdown.empty or drawdown.isna().all():
                return {
                    "max_drawdown": 0.0,
                    "peak_date": None,
                    "trough_date": None,
                    "recovery_date": None,
                    "drawdown_duration": 0,
                    "recovery_duration": 0,
                    "total_duration": 0,
                }

            # Find maximum drawdown
            max_drawdown = drawdown.min()
            trough_date = drawdown.idxmin()

            # Find peak date (last time cumulative max increased before trough)
            peak_date = None
            trough_idx = prices.index.get_loc(trough_date)

            for i in range(trough_idx, -1, -1):
                if i == 0 or cumulative_max.iloc[i] > cumulative_max.iloc[i - 1]:
                    peak_date = prices.index[i]
                    break

            # Find recovery date (first time price reaches peak level after trough)
            recovery_date = None
            if peak_date is not None:
                peak_value = cumulative_max.loc[peak_date]
                post_trough_prices = prices.loc[trough_date:]

                recovery_candidates = post_trough_prices[
                    post_trough_prices >= peak_value
                ]
                if not recovery_candidates.empty:
                    recovery_date = recovery_candidates.index[0]

            # Calculate durations
            drawdown_duration = 0
            recovery_duration = 0
            total_duration = 0

            if peak_date is not None and trough_date is not None:
                drawdown_duration = (trough_date - peak_date).days

                if recovery_date is not None:
                    recovery_duration = (recovery_date - trough_date).days
                    total_duration = (recovery_date - peak_date).days
                else:
                    # Still underwater
                    total_duration = (prices.index[-1] - peak_date).days

            return {
                "max_drawdown": float(max_drawdown),
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "drawdown_duration": max(0, drawdown_duration),
                "recovery_duration": max(0, recovery_duration),
                "total_duration": max(0, total_duration),
            }

        except Exception as e:
            logger.error(f"Error in maximum drawdown analysis: {str(e)}")
            return {
                "max_drawdown": 0.0,
                "peak_date": None,
                "trough_date": None,
                "recovery_date": None,
                "drawdown_duration": 0,
                "recovery_duration": 0,
                "total_duration": 0,
            }

    @staticmethod
    def rolling_max_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling maximum drawdown over specified window.

        Provides time-varying maximum drawdown that captures
        changing risk characteristics over time.

        Args:
            prices: Series of prices
            window: Rolling window size in periods (default: 252 for 1 year)

        Returns:
            Series of rolling maximum drawdown values

        Example:
            >>> rolling_dd = DrawdownAnalysis.rolling_max_drawdown(prices, 60)
            >>> recent_worst_dd = rolling_dd.iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(prices, pd.Series):
                raise ValueError("prices must be a pandas Series")

            if len(prices) < window:
                logger.warning(
                    f"Insufficient data for rolling max drawdown: {len(prices)} < {window}"
                )
                return pd.Series(index=prices.index, dtype=float)

            rolling_max_dd = []

            for i in range(window, len(prices) + 1):
                window_prices = prices.iloc[i - window : i]

                # Calculate drawdown for window
                window_cummax = window_prices.expanding().max()
                window_drawdown = (window_prices - window_cummax) / window_cummax * 100

                # Get maximum drawdown in window
                max_dd = window_drawdown.min()
                rolling_max_dd.append(max_dd)

            # Create result series
            result = pd.Series(index=prices.index, dtype=float)
            result.iloc[window - 1 :] = rolling_max_dd

            return result

        except Exception as e:
            logger.error(f"Error calculating rolling max drawdown: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)

    @staticmethod
    def drawdown_clusters(
        prices: pd.Series, min_recovery: float = 0.1
    ) -> dict[str, Union[pd.DataFrame, int]]:
        """
        Identify and analyze drawdown clusters.

        Groups consecutive drawdown periods and analyzes their
        characteristics to understand stress period patterns.

        Args:
            prices: Series of prices
            min_recovery: Minimum recovery percentage to end a cluster (default: 0.1%)

        Returns:
            Dictionary containing:
            - clusters: DataFrame with cluster details
            - cluster_count: Number of clusters identified
            - avg_cluster_duration: Average duration of clusters
            - avg_cluster_depth: Average maximum depth of clusters

        Example:
            >>> clusters = DrawdownAnalysis.drawdown_clusters(prices)
            >>> stress_periods = clusters['clusters']
        """
        try:
            # Calculate basic drawdown metrics
            dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
            drawdown = dd_metrics["drawdown"]
            underwater = dd_metrics["underwater"]

            if drawdown.empty:
                return {
                    "clusters": pd.DataFrame(
                        columns=["start_date", "end_date", "max_drawdown", "duration"]
                    ),
                    "cluster_count": 0,
                    "avg_cluster_duration": 0.0,
                    "avg_cluster_depth": 0.0,
                }

            clusters = []
            cluster_start = None
            cluster_max_dd = 0.0

            for i, (date, is_underwater) in enumerate(underwater.items()):
                current_dd = drawdown.iloc[i] if i < len(drawdown) else 0

                if is_underwater and cluster_start is None:
                    # Start new cluster
                    cluster_start = date
                    cluster_max_dd = current_dd

                elif is_underwater and cluster_start is not None:
                    # Continue cluster, update max drawdown
                    cluster_max_dd = min(cluster_max_dd, current_dd)

                elif not is_underwater and cluster_start is not None:
                    # Check if recovery is sufficient to end cluster
                    recovery = abs(current_dd)  # How much we've recovered

                    if recovery >= min_recovery or i == len(underwater) - 1:
                        # End cluster
                        cluster_end = date
                        cluster_duration = (cluster_end - cluster_start).days

                        clusters.append(
                            {
                                "start_date": cluster_start,
                                "end_date": cluster_end,
                                "max_drawdown": cluster_max_dd,
                                "duration": cluster_duration,
                            }
                        )

                        cluster_start = None
                        cluster_max_dd = 0.0

            # Handle case where we end still in a cluster
            if cluster_start is not None:
                clusters.append(
                    {
                        "start_date": cluster_start,
                        "end_date": prices.index[-1],
                        "max_drawdown": cluster_max_dd,
                        "duration": (prices.index[-1] - cluster_start).days,
                    }
                )

            # Create DataFrame
            clusters_df = pd.DataFrame(clusters)

            # Calculate summary statistics
            cluster_count = len(clusters_df)
            avg_duration = (
                clusters_df["duration"].mean() if not clusters_df.empty else 0.0
            )
            avg_depth = (
                clusters_df["max_drawdown"].mean() if not clusters_df.empty else 0.0
            )

            return {
                "clusters": clusters_df,
                "cluster_count": cluster_count,
                "avg_cluster_duration": float(avg_duration),
                "avg_cluster_depth": float(avg_depth),
            }

        except Exception as e:
            logger.error(f"Error analyzing drawdown clusters: {str(e)}")
            return {
                "clusters": pd.DataFrame(
                    columns=["start_date", "end_date", "max_drawdown", "duration"]
                ),
                "cluster_count": 0,
                "avg_cluster_duration": 0.0,
                "avg_cluster_depth": 0.0,
            }

    @staticmethod
    def recovery_analysis(prices: pd.Series) -> dict[str, Union[pd.Series, float]]:
        """
        Analyze recovery patterns from drawdown periods.

        Studies how quickly and consistently the investment
        recovers from drawdown periods.

        Args:
            prices: Series of prices

        Returns:
            Dictionary containing:
            - recovery_factor: How much of drawdown has been recovered
            - time_to_recovery: Rolling estimate of recovery time
            - recovery_rate: Speed of recovery (% per period)
            - avg_recovery_time: Average historical recovery time

        Example:
            >>> recovery_info = DrawdownAnalysis.recovery_analysis(prices)
            >>> current_recovery_factor = recovery_info['recovery_factor'].iloc[-1]
        """
        try:
            # Calculate drawdown metrics
            dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
            drawdown = dd_metrics["drawdown"]
            cumulative_max = dd_metrics["cumulative_max"]
            underwater = dd_metrics["underwater"]

            if drawdown.empty:
                empty_series = pd.Series(dtype=float)
                return {
                    "recovery_factor": empty_series,
                    "time_to_recovery": empty_series,
                    "recovery_rate": empty_series,
                    "avg_recovery_time": 0.0,
                }

            # Calculate recovery factor (how much of drawdown recovered)
            recovery_factor = pd.Series(index=prices.index, dtype=float)

            for i in range(len(prices)):
                if underwater.iloc[i]:
                    # Currently underwater
                    recovery_factor.iloc[i] = 0.0
                else:
                    # Not underwater, full recovery
                    recovery_factor.iloc[i] = 1.0

            # Calculate time to recovery estimates
            time_to_recovery = pd.Series(index=prices.index, dtype=float)
            recovery_rate = pd.Series(index=prices.index, dtype=float)

            # Analyze historical recoveries
            recovery_times = []

            # Identify completed recovery cycles
            clusters = DrawdownAnalysis.drawdown_clusters(prices)
            completed_recoveries = clusters["clusters"][
                clusters["clusters"]["end_date"] < prices.index[-1]
            ]

            for _, cluster in completed_recoveries.iterrows():
                start_date = cluster["start_date"]
                end_date = cluster["end_date"]
                recovery_time = cluster["duration"]
                recovery_times.append(recovery_time)

            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0

            # Calculate current recovery rate for underwater periods
            for i in range(1, len(prices)):
                if underwater.iloc[i]:
                    # Calculate recovery rate as change in drawdown
                    dd_change = drawdown.iloc[i] - drawdown.iloc[i - 1]
                    recovery_rate.iloc[i] = abs(dd_change) if dd_change > 0 else 0.0

                    # Estimate time to recovery based on current rate
                    current_dd = abs(drawdown.iloc[i])
                    current_rate = recovery_rate.iloc[i]
                    if current_rate > 0:
                        estimated_time = current_dd / current_rate
                        time_to_recovery.iloc[i] = min(
                            estimated_time, avg_recovery_time * 2
                        )
                    else:
                        time_to_recovery.iloc[i] = avg_recovery_time
                else:
                    recovery_rate.iloc[i] = 0.0
                    time_to_recovery.iloc[i] = 0.0

            return {
                "recovery_factor": recovery_factor,
                "time_to_recovery": time_to_recovery,
                "recovery_rate": recovery_rate,
                "avg_recovery_time": float(avg_recovery_time),
            }

        except Exception as e:
            logger.error(f"Error in recovery analysis: {str(e)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                "recovery_factor": empty_series,
                "time_to_recovery": empty_series,
                "recovery_rate": empty_series,
                "avg_recovery_time": 0.0,
            }

    @staticmethod
    def drawdown_risk_features(prices: pd.Series) -> dict[str, float]:
        """
        Extract drawdown-based risk features for ML models.

        Combines multiple drawdown metrics into a comprehensive
        set of risk features suitable for machine learning.

        Args:
            prices: Series of prices

        Returns:
            Dictionary with drawdown risk features:
            - max_drawdown: Historical maximum drawdown
            - current_drawdown: Current drawdown level
            - avg_drawdown_duration: Average duration of drawdowns
            - drawdown_frequency: Frequency of significant drawdowns
            - recovery_factor: Current recovery progress
            - underwater_periods: Percentage of time underwater

        Example:
            >>> dd_features = DrawdownAnalysis.drawdown_risk_features(prices)
            >>> current_risk = dd_features['current_drawdown']
        """
        try:
            # Calculate comprehensive drawdown analysis
            dd_metrics = DrawdownAnalysis.calculate_drawdowns(prices)
            max_dd_info = DrawdownAnalysis.maximum_drawdown_analysis(prices)
            clusters_info = DrawdownAnalysis.drawdown_clusters(prices)
            recovery_info = DrawdownAnalysis.recovery_analysis(prices)

            # Extract features
            features = {}

            # Basic drawdown metrics
            features["max_drawdown"] = abs(max_dd_info["max_drawdown"])
            features["current_drawdown"] = (
                abs(dd_metrics["drawdown"].iloc[-1])
                if not dd_metrics["drawdown"].empty
                else 0.0
            )

            # Duration metrics
            features["max_drawdown_duration"] = max_dd_info["drawdown_duration"]
            features["avg_drawdown_duration"] = clusters_info["avg_cluster_duration"]

            # Frequency metrics
            total_days = len(prices)
            underwater_days = (
                dd_metrics["underwater"].sum()
                if not dd_metrics["underwater"].empty
                else 0
            )
            features["underwater_periods"] = (
                (underwater_days / total_days) * 100 if total_days > 0 else 0.0
            )

            # Cluster metrics
            features["drawdown_frequency"] = (
                clusters_info["cluster_count"] / (total_days / 252)
                if total_days > 0
                else 0.0
            )
            features["avg_cluster_depth"] = abs(clusters_info["avg_cluster_depth"])

            # Recovery metrics
            features["avg_recovery_time"] = recovery_info["avg_recovery_time"]
            features["current_recovery_factor"] = (
                recovery_info["recovery_factor"].iloc[-1]
                if not recovery_info["recovery_factor"].empty
                else 1.0
            )

            # Derived risk metrics
            features["drawdown_to_recovery_ratio"] = features[
                "max_drawdown_duration"
            ] / max(features["avg_recovery_time"], 1)

            # Risk score (composite measure)
            risk_score = (
                features["max_drawdown"] * 0.3
                + features["current_drawdown"] * 0.2
                + features["underwater_periods"] * 0.2
                + features["drawdown_frequency"] * 10 * 0.3
            )
            features["drawdown_risk_score"] = min(100, max(0, risk_score))

            return features

        except Exception as e:
            logger.error(f"Error extracting drawdown risk features: {str(e)}")
            return {
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown_duration": 0.0,
                "underwater_periods": 0.0,
                "drawdown_frequency": 0.0,
                "avg_cluster_depth": 0.0,
                "avg_recovery_time": 0.0,
                "current_recovery_factor": 1.0,
                "drawdown_to_recovery_ratio": 0.0,
                "drawdown_risk_score": 0.0,
            }
