"""
Risk-Adjusted Returns Module

Professional risk-adjusted performance metrics for ML features.
Provides Sharpe ratio, Sortino ratio, Calmar ratio, and other
risk-adjusted measures that capture return efficiency.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class RiskAdjustedReturns:
    """
    Professional risk-adjusted return calculations.

    This class provides comprehensive risk-adjusted performance metrics
    that measure return efficiency relative to various risk measures.
    """

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, window: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Sharpe ratio measures excess return per unit of total risk (volatility).
        Higher values indicate better risk-adjusted performance.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized, default: 0.0)
            window: Rolling window size (default: 252 for 1 year)

        Returns:
            Series of rolling Sharpe ratios

        Example:
            >>> returns = stock_prices.pct_change().dropna()
            >>> sharpe = RiskAdjustedReturns.sharpe_ratio(returns, 0.02)
            >>> current_sharpe = sharpe.iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for Sharpe ratio: {len(returns)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            # Convert risk-free rate to period rate
            period_rf_rate = risk_free_rate / 252  # Assuming daily data

            # Calculate rolling Sharpe ratio
            excess_returns = returns - period_rf_rate
            rolling_mean = excess_returns.rolling(window=window).mean()
            rolling_std = excess_returns.rolling(window=window).std()

            # Annualize and calculate Sharpe ratio
            annualized_return = rolling_mean * 252
            annualized_volatility = rolling_std * np.sqrt(252)

            sharpe_ratio = annualized_return / annualized_volatility

            # Handle division by zero
            sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], np.nan)

            return sharpe_ratio

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, window: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Sortino ratio.

        Sortino ratio measures excess return per unit of downside risk.
        It only considers negative volatility, making it more relevant
        for downside risk assessment.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized, default: 0.0)
            window: Rolling window size (default: 252)

        Returns:
            Series of rolling Sortino ratios

        Example:
            >>> sortino = RiskAdjustedReturns.sortino_ratio(returns)
            >>> current_sortino = sortino.iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for Sortino ratio: {len(returns)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            # Convert risk-free rate to period rate
            period_rf_rate = risk_free_rate / 252

            # Calculate excess returns
            excess_returns = returns - period_rf_rate

            # Calculate rolling Sortino ratio
            sortino_ratios = []

            for i in range(window, len(returns) + 1):
                window_excess = excess_returns.iloc[i - window : i]

                # Calculate downside deviation (only negative returns)
                downside_returns = window_excess[window_excess < 0]

                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std()
                else:
                    downside_deviation = 0.0

                # Calculate mean excess return
                mean_excess = window_excess.mean()

                # Calculate Sortino ratio
                if downside_deviation > 0:
                    sortino = (mean_excess * 252) / (downside_deviation * np.sqrt(252))
                else:
                    sortino = np.inf if mean_excess > 0 else 0.0

                sortino_ratios.append(sortino)

            # Create result series
            result = pd.Series(index=returns.index, dtype=float)
            result.iloc[window - 1 :] = sortino_ratios

            # Handle infinite values
            result = result.replace([np.inf, -np.inf], np.nan)

            return result

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def calmar_ratio(returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling Calmar ratio.

        Calmar ratio measures annualized return relative to maximum drawdown.
        It focuses on the worst-case scenario for risk assessment.

        Args:
            returns: Series of returns
            window: Rolling window size (default: 252)

        Returns:
            Series of rolling Calmar ratios

        Example:
            >>> calmar = RiskAdjustedReturns.calmar_ratio(returns)
            >>> current_calmar = calmar.iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for Calmar ratio: {len(returns)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            # Calculate cumulative returns for drawdown calculation
            cumulative_returns = (1 + returns).cumprod()

            calmar_ratios = []

            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i - window : i]
                window_cumulative = cumulative_returns.iloc[i - window : i]

                # Calculate annualized return
                total_return = (
                    window_cumulative.iloc[-1] / window_cumulative.iloc[0] - 1
                )
                annualized_return = (1 + total_return) ** (
                    252 / len(window_returns)
                ) - 1

                # Calculate maximum drawdown
                running_max = window_cumulative.expanding().max()
                drawdown = (window_cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min())

                # Calculate Calmar ratio
                if max_drawdown > 0:
                    calmar = annualized_return / max_drawdown
                else:
                    calmar = np.inf if annualized_return > 0 else 0.0

                calmar_ratios.append(calmar)

            # Create result series
            result = pd.Series(index=returns.index, dtype=float)
            result.iloc[window - 1 :] = calmar_ratios

            # Handle infinite values
            result = result.replace([np.inf, -np.inf], np.nan)

            return result

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def information_ratio(
        returns: pd.Series, benchmark_returns: pd.Series, window: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Information ratio.

        Information ratio measures active return per unit of tracking error.
        It's useful for evaluating performance relative to a benchmark.

        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            window: Rolling window size (default: 252)

        Returns:
            Series of rolling Information ratios

        Example:
            >>> info_ratio = RiskAdjustedReturns.information_ratio(
            ...     portfolio_returns, benchmark_returns)
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series) or not isinstance(
                benchmark_returns, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Align series
            aligned_data = pd.concat(
                [returns, benchmark_returns], axis=1, join="inner"
            ).dropna()

            if len(aligned_data) < window:
                logger.warning(
                    f"Insufficient data for Information ratio: {len(aligned_data)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            portfolio_rets = aligned_data.iloc[:, 0]
            benchmark_rets = aligned_data.iloc[:, 1]

            # Calculate active returns (excess returns over benchmark)
            active_returns = portfolio_rets - benchmark_rets

            # Calculate rolling Information ratio
            rolling_mean = active_returns.rolling(window=window).mean()
            rolling_std = active_returns.rolling(window=window).std()

            # Annualize
            annualized_active_return = rolling_mean * 252
            annualized_tracking_error = rolling_std * np.sqrt(252)

            information_ratio = annualized_active_return / annualized_tracking_error

            # Handle division by zero
            information_ratio = information_ratio.replace([np.inf, -np.inf], np.nan)

            # Expand to original index
            result = pd.Series(index=returns.index, dtype=float)
            result.loc[information_ratio.index] = information_ratio

            return result

        except Exception as e:
            logger.error(f"Error calculating Information ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def omega_ratio(
        returns: pd.Series, threshold: float = 0.0, window: int = 252
    ) -> pd.Series:
        """
        Calculate rolling Omega ratio.

        Omega ratio measures the probability-weighted ratio of gains
        to losses relative to a threshold return.

        Args:
            returns: Series of returns
            threshold: Threshold return (default: 0.0)
            window: Rolling window size (default: 252)

        Returns:
            Series of rolling Omega ratios

        Example:
            >>> omega = RiskAdjustedReturns.omega_ratio(returns, threshold=0.01)
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for Omega ratio: {len(returns)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            omega_ratios = []

            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i - window : i]

                # Calculate gains and losses relative to threshold
                excess_returns = window_returns - threshold
                gains = excess_returns[excess_returns > 0]
                losses = excess_returns[excess_returns < 0]

                # Calculate Omega ratio
                gains_sum = gains.sum() if len(gains) > 0 else 0.0
                losses_sum = abs(losses.sum()) if len(losses) > 0 else 0.0

                if losses_sum > 0:
                    omega = gains_sum / losses_sum
                else:
                    omega = np.inf if gains_sum > 0 else 1.0

                omega_ratios.append(omega)

            # Create result series
            result = pd.Series(index=returns.index, dtype=float)
            result.iloc[window - 1 :] = omega_ratios

            # Handle infinite values
            result = result.replace([np.inf, -np.inf], np.nan)

            return result

        except Exception as e:
            logger.error(f"Error calculating Omega ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def sterling_ratio(returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling Sterling ratio.

        Sterling ratio measures annualized return relative to the
        average maximum drawdown, providing a more stable measure
        than Calmar ratio.

        Args:
            returns: Series of returns
            window: Rolling window size (default: 252)

        Returns:
            Series of rolling Sterling ratios
        """
        try:
            # Input validation
            if not isinstance(returns, pd.Series):
                raise ValueError("returns must be a pandas Series")

            if len(returns) < window:
                logger.warning(
                    f"Insufficient data for Sterling ratio: {len(returns)} < {window}"
                )
                return pd.Series(index=returns.index, dtype=float)

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()

            sterling_ratios = []

            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i - window : i]
                window_cumulative = cumulative_returns.iloc[i - window : i]

                # Calculate annualized return
                total_return = (
                    window_cumulative.iloc[-1] / window_cumulative.iloc[0] - 1
                )
                annualized_return = (1 + total_return) ** (
                    252 / len(window_returns)
                ) - 1

                # Calculate drawdowns
                running_max = window_cumulative.expanding().max()
                drawdowns = (window_cumulative - running_max) / running_max

                # Calculate average maximum drawdown (absolute value)
                # Split into drawdown periods and find max of each
                drawdown_periods = []
                current_dd = 0.0

                for dd in drawdowns:
                    if dd < 0:
                        current_dd = min(current_dd, dd)
                    else:
                        if current_dd < 0:
                            drawdown_periods.append(abs(current_dd))
                            current_dd = 0.0

                # Add final drawdown if still in one
                if current_dd < 0:
                    drawdown_periods.append(abs(current_dd))

                avg_max_drawdown = (
                    np.mean(drawdown_periods) if drawdown_periods else 0.01
                )  # Avoid division by zero

                # Calculate Sterling ratio
                sterling = (
                    annualized_return / avg_max_drawdown
                    if avg_max_drawdown > 0
                    else 0.0
                )
                sterling_ratios.append(sterling)

            # Create result series
            result = pd.Series(index=returns.index, dtype=float)
            result.iloc[window - 1 :] = sterling_ratios

            return result

        except Exception as e:
            logger.error(f"Error calculating Sterling ratio: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def comprehensive_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.0,
    ) -> dict[str, Union[pd.Series, float]]:
        """
        Calculate comprehensive risk-adjusted performance metrics.

        Provides a complete suite of risk-adjusted measures for
        thorough performance evaluation and ML feature extraction.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for relative metrics
            risk_free_rate: Risk-free rate (annualized, default: 0.0)

        Returns:
            Dictionary containing all risk-adjusted metrics and current values

        Example:
            >>> metrics = RiskAdjustedReturns.comprehensive_metrics(
            ...     returns, benchmark_returns, 0.02)
            >>> current_sharpe = metrics['current_sharpe']
        """
        return RiskAdjustedReturns.comprehensive_risk_adjusted_metrics(
            returns, benchmark_returns, risk_free_rate
        )

    @staticmethod
    def comprehensive_risk_adjusted_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.0,
    ) -> dict[str, Union[pd.Series, float]]:
        """
        Calculate comprehensive risk-adjusted performance metrics.

        Provides a complete suite of risk-adjusted measures for
        thorough performance evaluation and ML feature extraction.

        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for relative metrics
            risk_free_rate: Risk-free rate (annualized, default: 0.0)

        Returns:
            Dictionary containing all risk-adjusted metrics and current values

        Example:
            >>> metrics = RiskAdjustedReturns.comprehensive_risk_adjusted_metrics(
            ...     returns, benchmark_returns, 0.02)
            >>> current_sharpe = metrics['current_sharpe']
        """
        try:
            # Calculate all risk-adjusted ratios
            sharpe = RiskAdjustedReturns.sharpe_ratio(returns, risk_free_rate)
            sortino = RiskAdjustedReturns.sortino_ratio(returns, risk_free_rate)
            calmar = RiskAdjustedReturns.calmar_ratio(returns)
            omega = RiskAdjustedReturns.omega_ratio(returns)
            sterling = RiskAdjustedReturns.sterling_ratio(returns)

            metrics = {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "omega_ratio": omega,
                "sterling_ratio": sterling,
            }

            # Add benchmark-relative metrics if benchmark provided
            if benchmark_returns is not None:
                info_ratio = RiskAdjustedReturns.information_ratio(
                    returns, benchmark_returns
                )
                metrics["information_ratio"] = info_ratio
                metrics["current_information_ratio"] = (
                    info_ratio.iloc[-1] if not info_ratio.empty else 0.0
                )

            # Extract current values for ML features
            metrics["current_sharpe"] = sharpe.iloc[-1] if not sharpe.empty else 0.0
            metrics["current_sortino"] = sortino.iloc[-1] if not sortino.empty else 0.0
            metrics["current_calmar"] = calmar.iloc[-1] if not calmar.empty else 0.0
            metrics["current_omega"] = omega.iloc[-1] if not omega.empty else 1.0
            metrics["current_sterling"] = (
                sterling.iloc[-1] if not sterling.empty else 0.0
            )

            # Calculate summary statistics
            metrics["avg_sharpe"] = sharpe.mean() if not sharpe.empty else 0.0
            metrics["sharpe_volatility"] = sharpe.std() if not sharpe.empty else 0.0
            metrics["sharpe_trend"] = (
                _calculate_trend(sharpe) if not sharpe.empty else 0.0
            )

            # Risk-adjusted performance classification
            current_sharpe = metrics["current_sharpe"]
            if current_sharpe > 2.0:
                metrics["performance_class"] = "excellent"
            elif current_sharpe > 1.0:
                metrics["performance_class"] = "good"
            elif current_sharpe > 0.5:
                metrics["performance_class"] = "average"
            elif current_sharpe > 0.0:
                metrics["performance_class"] = "poor"
            else:
                metrics["performance_class"] = "very_poor"

            # Risk efficiency score (composite measure)
            efficiency_score = (
                min(metrics["current_sharpe"], 3.0) * 20
                + min(metrics["current_sortino"], 3.0) * 15  # Sharpe component (max 60)
                + min(metrics["current_calmar"], 2.0)  # Sortino component (max 45)
                * 5  # Calmar component (max 10)
            )
            metrics["risk_efficiency_score"] = min(100, max(0, efficiency_score))

            return metrics

        except Exception as e:
            logger.error(
                f"Error calculating comprehensive risk-adjusted metrics: {str(e)}"
            )
            return {
                "sharpe_ratio": pd.Series(dtype=float),
                "current_sharpe": 0.0,
                "performance_class": "unknown",
                "risk_efficiency_score": 0.0,
            }


def _calculate_trend(series: pd.Series, window: int = 60) -> float:
    """
    Calculate trend in a time series using linear regression slope.

    Args:
        series: Time series to analyze
        window: Window for trend calculation

    Returns:
        Trend slope (positive = improving, negative = deteriorating)
    """
    try:
        if len(series) < window:
            return 0.0

        recent_data = series.tail(window).dropna()

        if len(recent_data) < 10:
            return 0.0

        x = np.arange(len(recent_data))
        y = recent_data.values

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

    except Exception as e:
        logger.warning(f"Error calculating trend: {str(e)}")
        return 0.0
