"""
Value at Risk (VaR) Metrics Module

Professional VaR calculation and analysis for risk-based ML features.
Includes historical VaR, parametric VaR, Expected Shortfall, and
conditional risk measures that capture tail risk behavior.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class VaRMetrics:
    """
    Professional Value at Risk and Expected Shortfall calculations.

    This class provides multiple VaR methodologies and related risk metrics
    for capturing downside risk patterns in financial data.
    """

    @staticmethod
    def historical_var(
        returns: pd.Series, confidence_levels: list[float] = None, window: int = 252
    ) -> dict[str, pd.Series]:
        """
        Calculate historical Value at Risk using empirical distribution.

        Historical VaR uses actual return distribution without assuming
        any particular statistical distribution, making it robust to
        non-normal return patterns.

        Args:
            returns: Series of returns (daily, weekly, etc.)
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            window: Rolling window size (default: 252 for 1 year)

        Returns:
            Dictionary containing:
            - var_95: 95% VaR values
            - var_99: 99% VaR values
            - var_custom: Custom confidence level VaR (if specified)

        Example:
            >>> returns = stock_prices.pct_change().dropna()
            >>> var_metrics = VaRMetrics.historical_var(returns)
            >>> current_var_95 = var_metrics['var_95'].iloc[-1]
        """
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]

            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for VaR calculation: {len(returns)} < {window}"
                )
                empty_series = pd.Series(index=returns.index, dtype=float)
                result = {}
                for conf_level in confidence_levels:
                    key = f"var_{int(conf_level*100)}"
                    result[key] = empty_series
                return result

            # Calculate rolling historical VaR
            result = {}

            for conf_level in confidence_levels:
                var_values = []
                alpha = 1 - conf_level  # e.g., 0.05 for 95% confidence

                for i in range(window, len(returns) + 1):
                    window_returns = returns.iloc[i - window : i]

                    # Calculate VaR as the alpha-quantile of returns
                    var_value = np.percentile(window_returns, alpha * 100)
                    var_values.append(var_value)

                # Create series with appropriate index
                var_series = pd.Series(
                    var_values,
                    index=returns.index[window - 1 :],
                    name=f"VaR_{int(conf_level*100)}%",
                )

                # Pad with NaN for early periods
                full_var_series = pd.Series(index=returns.index, dtype=float)
                full_var_series.loc[var_series.index] = var_series

                key = f"var_{int(conf_level*100)}"
                result[key] = full_var_series

            return result

        except Exception as e:
            logger.error(f"Error calculating historical VaR: {str(e)}")
            empty_series = pd.Series(dtype=float)

            result = {}
            for conf in confidence_levels:
                conf_str = f"var_{int(conf * 100)}"
                result[conf_str] = empty_series

            return result

    @staticmethod
    def parametric_var(
        returns: pd.Series,
        confidence_levels: list[float] = None,
        window: int = 252,
        distribution: str = "normal",
    ) -> dict[str, pd.Series]:
        """
        Calculate parametric VaR assuming a specific distribution.

        Parametric VaR fits returns to a statistical distribution and
        uses the theoretical quantiles to estimate VaR.

        Args:
            returns: Series of returns
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            window: Rolling window size (default: 252)
            distribution: Distribution to fit ('normal', 't', 'skewnorm')

        Returns:
            Dictionary with parametric VaR series for each confidence level

        Example:
            >>> var_metrics = VaRMetrics.parametric_var(returns, distribution='t')
            >>> t_dist_var = var_metrics['var_95'].iloc[-1]
        """
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]

            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < 30:  # Use minimum 30 observations
                logger.warning(
                    f"Insufficient data for parametric VaR: {len(returns)} < 30"
                )
                empty_series = pd.Series(dtype=float)
                result = {}
                for conf_level in confidence_levels:
                    key = f"var_{int(conf_level*100)}"
                    result[key] = empty_series
                return result

            # Use smaller window to ensure we get meaningful rolling results
            # For datasets with exactly the window size, use a smaller window to get multiple points
            if len(returns) <= window:
                effective_window = max(
                    30, min(60, len(returns) // 2)
                )  # Use smaller window for rolling
            else:
                effective_window = window

            result = {}

            for conf_level in confidence_levels:
                var_values = []
                alpha = 1 - conf_level

                for i in range(effective_window, len(returns) + 1):
                    window_returns = returns.iloc[i - effective_window : i]

                    try:
                        # Check for edge cases
                        if len(window_returns) < 30:  # Insufficient data
                            var_value = np.nan
                        elif window_returns.std() == 0:  # No volatility
                            var_value = (
                                window_returns.mean()
                            )  # Return the constant value
                        else:
                            if distribution == "normal":
                                # Normal distribution VaR
                                mean = window_returns.mean()
                                std = window_returns.std()
                                if std > 0:
                                    var_value = stats.norm.ppf(
                                        alpha, loc=mean, scale=std
                                    )
                                else:
                                    var_value = mean

                            elif distribution == "t":
                                # Student's t-distribution VaR
                                params = stats.t.fit(window_returns)
                                var_value = stats.t.ppf(alpha, *params)

                            elif distribution == "skewnorm":
                                # Skewed normal distribution VaR
                                params = stats.skewnorm.fit(window_returns)
                                var_value = stats.skewnorm.ppf(alpha, *params)

                            else:
                                # Default to normal if unknown distribution
                                mean = window_returns.mean()
                                std = window_returns.std()
                                if std > 0:
                                    var_value = stats.norm.ppf(
                                        alpha, loc=mean, scale=std
                                    )
                                else:
                                    var_value = mean

                    except Exception as fit_error:
                        # Fallback to historical VaR if fitting fails
                        try:
                            var_value = np.percentile(window_returns, alpha * 100)
                        except:
                            var_value = np.nan

                    # Validate result
                    if not np.isfinite(var_value):
                        var_value = np.nan

                    var_values.append(var_value)

                # Create series - only include valid values, no NaN padding
                if var_values:
                    var_series = pd.Series(
                        var_values,
                        index=returns.index[effective_window - 1 :],
                        name=f"Parametric_VaR_{int(conf_level*100)}%",
                    )
                    # Don't pad with NaN, just return the valid values
                    full_var_series = var_series
                else:
                    # If no valid values, return empty series
                    full_var_series = pd.Series(dtype=float)

                key = f"var_{int(conf_level*100)}"
                result[key] = full_var_series

            return result

        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {str(e)}")
            empty_series = pd.Series(dtype=float)
            result = {}
            for conf_level in confidence_levels:
                key = f"var_{int(conf_level*100)}"
                result[key] = empty_series
            return result

    @staticmethod
    def expected_shortfall(
        returns: pd.Series, confidence_levels: list[float] = None, window: int = 252
    ) -> dict[str, pd.Series]:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Expected Shortfall measures the expected loss given that
        the loss exceeds the VaR threshold. It provides insight
        into tail risk beyond what VaR captures.

        Args:
            returns: Series of returns
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            window: Rolling window size (default: 252)

        Returns:
            Dictionary with Expected Shortfall series for each confidence level

        Example:
            >>> es_metrics = VaRMetrics.expected_shortfall(returns)
            >>> current_es_99 = es_metrics['es_99'].iloc[-1]
        """
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]

            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for ES calculation: {len(returns)} < {window}"
                )
                empty_series = pd.Series(index=returns.index, dtype=float)
                result = {}
                for conf_level in confidence_levels:
                    key = f"es_{int(conf_level*100)}"
                    result[key] = empty_series
                return result

            result = {}

            for conf_level in confidence_levels:
                es_values = []
                alpha = 1 - conf_level

                for i in range(window, len(returns) + 1):
                    window_returns = returns.iloc[i - window : i]

                    # Calculate VaR first
                    var_threshold = np.percentile(window_returns, alpha * 100)

                    # Calculate Expected Shortfall (mean of returns below VaR)
                    tail_returns = window_returns[window_returns <= var_threshold]

                    if len(tail_returns) > 0:
                        es_value = tail_returns.mean()
                    else:
                        # If no returns below VaR, use VaR itself
                        es_value = var_threshold

                    es_values.append(es_value)

                # Create series
                es_series = pd.Series(
                    es_values,
                    index=returns.index[window - 1 :],
                    name=f"ES_{int(conf_level*100)}%",
                )

                # Pad with NaN
                full_es_series = pd.Series(index=returns.index, dtype=float)
                full_es_series.loc[es_series.index] = es_series

                key = f"es_{int(conf_level*100)}"
                result[key] = full_es_series

            return result

        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            empty_series = pd.Series(dtype=float)
            result = {}
            for conf_level in confidence_levels:
                key = f"es_{int(conf_level*100)}"
                result[key] = empty_series
            return result

    @staticmethod
    def var_breach_analysis(
        returns: pd.Series, var_estimates: pd.Series, confidence_level: float = 0.95
    ) -> dict[str, Union[pd.Series, float, int]]:
        """
        Analyze VaR model performance through breach analysis.

        Examines how often actual losses exceed VaR estimates
        and clusters of breaches which indicate model inadequacy.

        Args:
            returns: Series of actual returns
            var_estimates: Series of VaR estimates
            confidence_level: Confidence level of VaR estimates (default: 0.95)

        Returns:
            Dictionary containing:
            - breaches: Boolean series of VaR breaches
            - breach_rate: Actual breach rate vs expected
            - breach_clusters: Identification of breach clustering
            - kupiec_test: Kupiec test p-value for breach rate

        Example:
            >>> var_data = VaRMetrics.historical_var(returns)
            >>> breach_analysis = VaRMetrics.var_breach_analysis(
            ...     returns, var_data['var_95'])
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series) or not isinstance(
                var_estimates, pd.Series
            ):
                raise ValueError("returns and var_estimates must be pandas Series")

            # Align series
            aligned_data = pd.concat([returns, var_estimates], axis=1, join="inner")
            if aligned_data.empty:
                logger.warning("No overlapping data for breach analysis")
                return {
                    "breaches": pd.Series(dtype=bool),
                    "breach_rate": 0.0,
                    "expected_breach_rate": 1 - confidence_level,
                    "breach_clusters": pd.Series(dtype=int),
                    "kupiec_test_pvalue": np.nan,
                }

            actual_returns = aligned_data.iloc[:, 0]
            var_values = aligned_data.iloc[:, 1]

            # Identify breaches (returns worse than VaR)
            breaches = actual_returns < var_values

            # Calculate breach statistics
            total_observations = len(breaches)
            total_breaches = breaches.sum()
            actual_breach_rate = total_breaches / total_observations
            expected_breach_rate = 1 - confidence_level

            # Identify breach clusters
            breach_clusters = _identify_breach_clusters(breaches)

            # Kupiec unconditional coverage test
            kupiec_p_value = _kupiec_test(
                total_breaches, total_observations, expected_breach_rate
            )

            return {
                "breaches": breaches,
                "breach_rate": actual_breach_rate,
                "expected_breach_rate": expected_breach_rate,
                "breach_clusters": breach_clusters,
                "kupiec_test_pvalue": kupiec_p_value,
                "total_breaches": int(total_breaches),
                "breach_rate_ratio": actual_breach_rate / expected_breach_rate
                if expected_breach_rate > 0
                else np.inf,
            }

        except Exception as e:
            logger.error(f"Error in VaR breach analysis: {str(e)}")
            return {
                "breaches": pd.Series(dtype=bool),
                "breach_rate": 0.0,
                "expected_breach_rate": 1 - confidence_level,
                "breach_clusters": pd.Series(dtype=int),
                "kupiec_test_pvalue": np.nan,
            }

    @staticmethod
    def tail_risk_features(
        returns: pd.Series, window: int = 252
    ) -> dict[str, pd.Series]:
        """
        Extract comprehensive tail risk features for ML models.

        Combines multiple tail risk measures to capture extreme
        market behavior patterns.

        Args:
            returns: Series of returns
            window: Rolling window size (default: 252)

        Returns:
            Dictionary with tail risk features:
            - skewness: Rolling skewness of returns
            - kurtosis: Rolling kurtosis (excess)
            - tail_ratio: Ratio of tail losses to total volatility
            - extreme_loss_freq: Frequency of extreme losses
            - tail_expectation: Expected value of tail losses

        Example:
            >>> tail_features = VaRMetrics.tail_risk_features(returns)
            >>> current_skewness = tail_features['skewness'].iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for tail risk features: {len(returns)} < {window}"
                )
                empty_series = pd.Series(index=returns.index, dtype=float)
                return {
                    "skewness": empty_series,
                    "kurtosis": empty_series,
                    "tail_ratio": empty_series,
                    "extreme_loss_freq": empty_series,
                    "tail_expectation": empty_series,
                }

            # Calculate rolling statistics
            rolling_skewness = returns.rolling(window=window).skew()
            rolling_kurtosis = returns.rolling(window=window).kurt()  # Excess kurtosis

            # Calculate tail-specific features
            tail_ratio_values = []
            extreme_loss_freq_values = []
            tail_expectation_values = []

            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i - window : i]

                # Tail ratio: std of bottom 5% / total std
                bottom_5_percent = np.percentile(window_returns, 5)
                tail_returns = window_returns[window_returns <= bottom_5_percent]
                if len(tail_returns) > 1:
                    tail_std = tail_returns.std()
                    total_std = window_returns.std()
                    tail_ratio = tail_std / total_std if total_std > 0 else 0
                else:
                    tail_ratio = 0

                # Extreme loss frequency (losses > 2 standard deviations)
                mean_return = window_returns.mean()
                std_return = window_returns.std()
                extreme_threshold = mean_return - 2 * std_return
                extreme_losses = (window_returns < extreme_threshold).sum()
                extreme_loss_freq = extreme_losses / len(window_returns)

                # Tail expectation (mean of worst 5% returns)
                worst_5_percent = window_returns[window_returns <= bottom_5_percent]
                tail_expectation = (
                    worst_5_percent.mean()
                    if len(worst_5_percent) > 0
                    else bottom_5_percent
                )

                tail_ratio_values.append(tail_ratio)
                extreme_loss_freq_values.append(extreme_loss_freq)
                tail_expectation_values.append(tail_expectation)

            # Create series for calculated features
            tail_ratio_series = pd.Series(index=returns.index, dtype=float)
            extreme_loss_freq_series = pd.Series(index=returns.index, dtype=float)
            tail_expectation_series = pd.Series(index=returns.index, dtype=float)

            tail_ratio_series.iloc[window - 1 :] = tail_ratio_values
            extreme_loss_freq_series.iloc[window - 1 :] = extreme_loss_freq_values
            tail_expectation_series.iloc[window - 1 :] = tail_expectation_values

            return {
                "skewness": rolling_skewness,
                "kurtosis": rolling_kurtosis,
                "tail_ratio": tail_ratio_series,
                "extreme_loss_freq": extreme_loss_freq_series,
                "tail_expectation": tail_expectation_series,
            }

        except Exception as e:
            logger.error(f"Error calculating tail risk features: {str(e)}")
            empty_series = pd.Series(index=returns.index, dtype=float)
            return {
                "skewness": empty_series,
                "kurtosis": empty_series,
                "tail_ratio": empty_series,
                "extreme_loss_freq": empty_series,
                "tail_expectation": empty_series,
            }


def _identify_breach_clusters(breaches: pd.Series, max_gap: int = 5) -> pd.Series:
    """
    Identify clusters of VaR breaches.

    Breaches occurring within max_gap periods are considered part
    of the same cluster, indicating model inadequacy during stress periods.
    """
    try:
        clusters = pd.Series(index=breaches.index, dtype=int)
        cluster_id = 0
        in_cluster = False
        days_since_breach = 0

        for i, is_breach in enumerate(breaches):
            if is_breach:
                if not in_cluster or days_since_breach <= max_gap:
                    if not in_cluster:
                        cluster_id += 1
                        in_cluster = True
                    clusters.iloc[i] = cluster_id
                    days_since_breach = 0
                else:
                    # Start new cluster
                    cluster_id += 1
                    clusters.iloc[i] = cluster_id
                    days_since_breach = 0
            else:
                days_since_breach += 1
                if days_since_breach > max_gap:
                    in_cluster = False
                clusters.iloc[i] = cluster_id if in_cluster else 0

        return clusters

    except Exception as e:
        logger.warning(f"Error identifying breach clusters: {str(e)}")
        return pd.Series(index=breaches.index, dtype=int)


def _kupiec_test(
    observed_breaches: int, total_observations: int, expected_rate: float
) -> float:
    """
    Perform Kupiec unconditional coverage test for VaR model validation.

    Tests whether the observed breach rate is statistically different
    from the expected breach rate.
    """
    try:
        if total_observations == 0 or expected_rate <= 0 or expected_rate >= 1:
            return np.nan

        expected_breaches = total_observations * expected_rate

        if expected_breaches == 0 or expected_breaches == total_observations:
            return np.nan

        # Likelihood ratio test statistic
        if observed_breaches == 0:
            lr_stat = -2 * total_observations * np.log(1 - expected_rate)
        elif observed_breaches == total_observations:
            lr_stat = -2 * total_observations * np.log(expected_rate)
        else:
            p_observed = observed_breaches / total_observations
            lr_stat = -2 * (
                observed_breaches * np.log(expected_rate / p_observed)
                + (total_observations - observed_breaches)
                * np.log((1 - expected_rate) / (1 - p_observed))
            )

        # P-value from chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        return p_value

    except Exception as e:
        logger.warning(f"Error in Kupiec test: {str(e)}")
        return np.nan
