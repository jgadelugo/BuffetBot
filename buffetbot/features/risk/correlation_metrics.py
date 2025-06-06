"""
Correlation Metrics Module

Professional correlation and beta analysis for risk-based ML features.
Provides rolling correlations, market beta, correlation clustering,
and relationship stability metrics for market analysis.

Author: BuffetBot Development Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class CorrelationMetrics:
    """
    Professional correlation and beta analysis.

    This class provides comprehensive correlation metrics including
    rolling correlations, market beta, and correlation stability measures.
    """

    @staticmethod
    def rolling_correlation(
        returns1: pd.Series, returns2: pd.Series, window: int = 60
    ) -> pd.Series:
        """
        Calculate rolling correlation between two return series.

        Rolling correlation captures the time-varying relationship
        between assets, which is crucial for understanding
        changing market dynamics.

        Args:
            returns1: First return series (e.g., stock returns)
            returns2: Second return series (e.g., market returns)
            window: Rolling window size (default: 60 periods)

        Returns:
            Series of rolling correlation coefficients

        Example:
            >>> stock_returns = stock_prices.pct_change()
            >>> market_returns = market_index.pct_change()
            >>> rolling_corr = CorrelationMetrics.rolling_correlation(
            ...     stock_returns, market_returns)
        """
        try:
            # Input validation
            if not isinstance(returns1, pd.Series) or not isinstance(
                returns2, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Align series
            aligned_data = pd.concat(
                [returns1, returns2], axis=1, join="inner"
            ).dropna()

            if len(aligned_data) < window:
                logger.warning(
                    f"Insufficient data for rolling correlation: {len(aligned_data)} < {window}"
                )
                return pd.Series(dtype=float)

            # Calculate rolling correlation
            rolling_corr = (
                aligned_data.iloc[:, 0]
                .rolling(window=window)
                .corr(aligned_data.iloc[:, 1])
            )

            # Expand to original index
            result = pd.Series(index=returns1.index, dtype=float)
            result.loc[rolling_corr.index] = rolling_corr

            return result

        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {str(e)}")
            return pd.Series(dtype=float)

    @staticmethod
    def rolling_beta(
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60,
        risk_free_rate: float = 0.0,
    ) -> dict[str, pd.Series]:
        """
        Calculate rolling beta and related regression metrics.

        Beta measures systematic risk relative to the market.
        Higher beta indicates higher sensitivity to market movements.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            window: Rolling window size (default: 60)
            risk_free_rate: Risk-free rate for excess returns (default: 0.0)

        Returns:
            Dictionary containing:
            - beta: Rolling beta coefficients
            - alpha: Rolling alpha (intercept)
            - r_squared: Rolling R-squared values
            - tracking_error: Rolling tracking error

        Example:
            >>> beta_metrics = CorrelationMetrics.rolling_beta(
            ...     stock_returns, market_returns)
            >>> current_beta = beta_metrics['beta'].iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(asset_returns, pd.Series) or not isinstance(
                market_returns, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Align series and calculate excess returns
            aligned_data = pd.concat(
                [asset_returns, market_returns], axis=1, join="inner"
            ).dropna()

            if len(aligned_data) < window:
                logger.warning(
                    f"Insufficient data for rolling beta: {len(aligned_data)} < {window}"
                )
                empty_series = pd.Series(index=asset_returns.index, dtype=float)
                return {
                    "beta": empty_series,
                    "alpha": empty_series,
                    "r_squared": empty_series,
                    "tracking_error": empty_series,
                }

            asset_excess = aligned_data.iloc[:, 0] - risk_free_rate
            market_excess = aligned_data.iloc[:, 1] - risk_free_rate

            # Calculate rolling regression metrics
            betas = []
            alphas = []
            r_squareds = []
            tracking_errors = []

            for i in range(window, len(aligned_data) + 1):
                window_asset = asset_excess.iloc[i - window : i]
                window_market = market_excess.iloc[i - window : i]

                try:
                    # Calculate covariance and variance
                    covariance = np.cov(window_asset, window_market)[0, 1]
                    market_variance = np.var(window_market, ddof=1)

                    if market_variance > 0:
                        beta = covariance / market_variance
                        alpha = window_asset.mean() - beta * window_market.mean()

                        # R-squared
                        correlation = np.corrcoef(window_asset, window_market)[0, 1]
                        r_squared = (
                            correlation**2 if not np.isnan(correlation) else 0.0
                        )

                        # Tracking error (standard deviation of excess returns)
                        excess_returns = window_asset - beta * window_market
                        tracking_error = excess_returns.std()

                    else:
                        beta = 0.0
                        alpha = window_asset.mean()
                        r_squared = 0.0
                        tracking_error = window_asset.std()

                except Exception as calc_error:
                    beta = 0.0
                    alpha = 0.0
                    r_squared = 0.0
                    tracking_error = 0.0

                betas.append(beta)
                alphas.append(alpha)
                r_squareds.append(r_squared)
                tracking_errors.append(tracking_error)

            # Create result series
            result_index = aligned_data.index[window - 1 :]

            beta_series = pd.Series(betas, index=result_index, name="Beta")
            alpha_series = pd.Series(alphas, index=result_index, name="Alpha")
            r_squared_series = pd.Series(
                r_squareds, index=result_index, name="R_Squared"
            )
            tracking_error_series = pd.Series(
                tracking_errors, index=result_index, name="Tracking_Error"
            )

            # Expand to original index
            result_beta = pd.Series(index=asset_returns.index, dtype=float)
            result_alpha = pd.Series(index=asset_returns.index, dtype=float)
            result_r2 = pd.Series(index=asset_returns.index, dtype=float)
            result_te = pd.Series(index=asset_returns.index, dtype=float)

            result_beta.loc[beta_series.index] = beta_series
            result_alpha.loc[alpha_series.index] = alpha_series
            result_r2.loc[r_squared_series.index] = r_squared_series
            result_te.loc[tracking_error_series.index] = tracking_error_series

            return {
                "beta": result_beta,
                "alpha": result_alpha,
                "r_squared": result_r2,
                "tracking_error": result_te,
            }

        except Exception as e:
            logger.error(f"Error calculating rolling beta: {str(e)}")
            empty_series = pd.Series(dtype=float)
            return {
                "beta": empty_series,
                "alpha": empty_series,
                "r_squared": empty_series,
                "tracking_error": empty_series,
            }

    @staticmethod
    def correlation_stability(
        returns1: pd.Series, returns2: pd.Series, window: int = 60, lookback: int = 252
    ) -> dict[str, pd.Series]:
        """
        Analyze correlation stability over time.

        Measures how stable the correlation relationship is,
        which indicates the reliability of diversification benefits.

        Args:
            returns1: First return series
            returns2: Second return series
            window: Rolling correlation window (default: 60)
            lookback: Lookback period for stability analysis (default: 252)

        Returns:
            Dictionary containing:
            - correlation: Rolling correlation
            - correlation_volatility: Volatility of correlation
            - correlation_trend: Trend in correlation
            - stability_score: Overall stability score (0-100)

        Example:
            >>> stability = CorrelationMetrics.correlation_stability(
            ...     stock_returns, market_returns)
            >>> current_stability = stability['stability_score'].iloc[-1]
        """
        try:
            # Input validation
            if not isinstance(returns1, pd.Series) or not isinstance(
                returns2, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Calculate rolling correlation
            rolling_corr = CorrelationMetrics.rolling_correlation(
                returns1, returns2, window
            )

            if rolling_corr.empty or rolling_corr.isna().all():
                empty_series = pd.Series(index=returns1.index, dtype=float)
                return {
                    "correlation": empty_series,
                    "correlation_volatility": empty_series,
                    "correlation_trend": empty_series,
                    "stability_score": empty_series,
                }

            # Calculate correlation volatility (rolling std of correlations)
            corr_volatility = rolling_corr.rolling(window=lookback).std()

            # Calculate correlation trend (slope of regression line)
            corr_trend = pd.Series(index=returns1.index, dtype=float)

            for i in range(lookback, len(rolling_corr) + 1):
                window_corr = rolling_corr.iloc[i - lookback : i].dropna()

                if len(window_corr) > 10:  # Need minimum data for trend
                    try:
                        x = np.arange(len(window_corr))
                        y = window_corr.values

                        # Linear regression slope
                        slope = np.polyfit(x, y, 1)[0]
                        corr_trend.iloc[i - 1] = slope

                    except Exception:
                        corr_trend.iloc[i - 1] = 0.0

            # Calculate stability score
            stability_score = pd.Series(index=returns1.index, dtype=float)

            for i in range(lookback, len(rolling_corr) + 1):
                try:
                    current_vol = (
                        corr_volatility.iloc[i - 1]
                        if i - 1 < len(corr_volatility)
                        else np.nan
                    )
                    current_trend = (
                        abs(corr_trend.iloc[i - 1])
                        if i - 1 < len(corr_trend)
                        else np.nan
                    )

                    if not np.isnan(current_vol) and not np.isnan(current_trend):
                        # Lower volatility and trend = higher stability
                        vol_score = max(
                            0, 100 - (current_vol * 200)
                        )  # Scale volatility
                        trend_score = max(
                            0, 100 - (current_trend * 1000)
                        )  # Scale trend

                        stability = (vol_score + trend_score) / 2
                        stability_score.iloc[i - 1] = min(100, max(0, stability))

                except Exception:
                    stability_score.iloc[i - 1] = 50.0  # Neutral score

            return {
                "correlation": rolling_corr,
                "correlation_volatility": corr_volatility,
                "correlation_trend": corr_trend,
                "stability_score": stability_score,
            }

        except Exception as e:
            logger.error(f"Error analyzing correlation stability: {str(e)}")
            empty_series = pd.Series(dtype=float)
            return {
                "correlation": empty_series,
                "correlation_volatility": empty_series,
                "correlation_trend": empty_series,
                "stability_score": empty_series,
            }

    @staticmethod
    def correlation_breakdown_risk(
        returns1: pd.Series, returns2: pd.Series, stress_threshold: float = -0.02
    ) -> dict[str, Union[float, pd.Series]]:
        """
        Analyze correlation breakdown during stress periods.

        Examines how correlations change during market stress,
        which is crucial for risk management and portfolio construction.

        Args:
            returns1: First return series
            returns2: Second return series
            stress_threshold: Threshold for defining stress periods (default: -2%)

        Returns:
            Dictionary containing:
            - normal_correlation: Correlation during normal periods
            - stress_correlation: Correlation during stress periods
            - correlation_breakdown: Difference between normal and stress correlations
            - stress_periods: Boolean series indicating stress periods

        Example:
            >>> breakdown = CorrelationMetrics.correlation_breakdown_risk(
            ...     stock_returns, market_returns)
            >>> breakdown_risk = breakdown['correlation_breakdown']
        """
        try:
            # Input validation
            if not isinstance(returns1, pd.Series) or not isinstance(
                returns2, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Input validation and alignment
            aligned_data = pd.concat(
                [returns1, returns2], axis=1, join="inner"
            ).dropna()

            if len(aligned_data) < 20:
                logger.warning("Insufficient data for correlation breakdown analysis")
                return {
                    "normal_correlation": 0.0,
                    "stress_correlation": 0.0,
                    "correlation_breakdown": 0.0,
                    "stress_periods": pd.Series(index=returns1.index, dtype=bool),
                }

            asset_rets = aligned_data.iloc[:, 0]
            market_rets = aligned_data.iloc[:, 1]

            # Identify stress periods (when market returns are below threshold)
            stress_periods = market_rets < stress_threshold

            # Calculate correlations for different periods
            normal_periods = ~stress_periods

            if stress_periods.sum() > 5 and normal_periods.sum() > 5:
                normal_correlation = asset_rets[normal_periods].corr(
                    market_rets[normal_periods]
                )
                stress_correlation = asset_rets[stress_periods].corr(
                    market_rets[stress_periods]
                )

                # Handle NaN correlations
                normal_correlation = (
                    normal_correlation if not np.isnan(normal_correlation) else 0.0
                )
                stress_correlation = (
                    stress_correlation if not np.isnan(stress_correlation) else 0.0
                )

                # Correlation breakdown (positive value indicates correlation increases in stress)
                correlation_breakdown = stress_correlation - normal_correlation

            else:
                # Not enough data for separate analysis
                overall_correlation = asset_rets.corr(market_rets)
                normal_correlation = overall_correlation
                stress_correlation = overall_correlation
                correlation_breakdown = 0.0

            # Expand stress periods to original index
            full_stress_periods = pd.Series(
                index=returns1.index, dtype=bool, data=False
            )
            full_stress_periods.loc[stress_periods.index] = stress_periods

            return {
                "normal_correlation": float(normal_correlation),
                "stress_correlation": float(stress_correlation),
                "correlation_breakdown": float(correlation_breakdown),
                "stress_periods": full_stress_periods,
            }

        except Exception as e:
            logger.error(f"Error analyzing correlation breakdown: {str(e)}")
            return {
                "normal_correlation": 0.0,
                "stress_correlation": 0.0,
                "correlation_breakdown": 0.0,
                "stress_periods": pd.Series(dtype=bool),
            }

    @staticmethod
    def multi_asset_correlation_matrix(
        returns_dict: dict[str, pd.Series], window: int = 60
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate rolling correlation matrices for multiple assets.

        Provides comprehensive correlation analysis across multiple
        assets, useful for portfolio and sector analysis.

        Args:
            returns_dict: Dictionary of asset name -> return series
            window: Rolling window size (default: 60)

        Returns:
            Dictionary containing:
            - correlation_matrices: List of correlation matrices over time
            - avg_correlation: Average correlation over time
            - correlation_eigenvalues: Principal eigenvalues over time

        Example:
            >>> returns_dict = {'AAPL': aapl_returns, 'MSFT': msft_returns}
            >>> corr_analysis = CorrelationMetrics.multi_asset_correlation_matrix(returns_dict)
        """
        try:
            if not returns_dict or len(returns_dict) < 2:
                logger.warning("Need at least 2 assets for correlation matrix")
                return {
                    "correlation_matrices": [],
                    "avg_correlation": pd.Series(dtype=float),
                    "correlation_eigenvalues": pd.Series(dtype=float),
                }

            # Combine all return series
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()

            if len(returns_df) < window:
                logger.warning(
                    f"Insufficient data for correlation matrix: {len(returns_df)} < {window}"
                )
                return {
                    "correlation_matrices": [],
                    "avg_correlation": pd.Series(dtype=float),
                    "correlation_eigenvalues": pd.Series(dtype=float),
                }

            correlation_matrices = []
            avg_correlations = []
            eigenvalues = []
            dates = []

            for i in range(window, len(returns_df) + 1):
                window_data = returns_df.iloc[i - window : i]

                try:
                    # Calculate correlation matrix
                    corr_matrix = window_data.corr()

                    # Calculate average correlation (excluding diagonal)
                    mask = np.ones_like(corr_matrix.values, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_corr = corr_matrix.values[mask].mean()

                    # Calculate principal eigenvalue
                    eigenvals = np.linalg.eigvals(corr_matrix.values)
                    max_eigenval = np.max(eigenvals.real)

                    correlation_matrices.append(corr_matrix)
                    avg_correlations.append(avg_corr)
                    eigenvalues.append(max_eigenval)
                    dates.append(returns_df.index[i - 1])

                except Exception as calc_error:
                    logger.warning(
                        f"Error calculating correlation matrix: {calc_error}"
                    )
                    continue

            # Create time series
            avg_correlation_series = pd.Series(avg_correlations, index=dates)
            eigenvalue_series = pd.Series(eigenvalues, index=dates)

            return {
                "correlation_matrices": correlation_matrices,
                "avg_correlation": avg_correlation_series,
                "correlation_eigenvalues": eigenvalue_series,
            }

        except Exception as e:
            logger.error(f"Error calculating multi-asset correlation matrix: {str(e)}")
            return {
                "correlation_matrices": [],
                "avg_correlation": pd.Series(dtype=float),
                "correlation_eigenvalues": pd.Series(dtype=float),
            }

    @staticmethod
    def correlation_risk_features(
        asset_returns: pd.Series, market_returns: pd.Series
    ) -> dict[str, float]:
        """
        Extract correlation-based risk features for ML models.

        Combines multiple correlation metrics into a comprehensive
        set of features for machine learning applications.

        Args:
            asset_returns: Asset return series
            market_returns: Market return series

        Returns:
            Dictionary of correlation risk features

        Example:
            >>> features = CorrelationMetrics.correlation_risk_features(
            ...     stock_returns, market_returns)
            >>> correlation_level = features['correlation_level']
        """
        try:
            # Input validation
            if not isinstance(asset_returns, pd.Series) or not isinstance(
                market_returns, pd.Series
            ):
                raise ValueError("Both inputs must be pandas Series")

            # Align series
            aligned_data = pd.concat(
                [asset_returns, market_returns], axis=1, join="inner"
            ).dropna()

            if len(aligned_data) < 30:
                logger.warning("Insufficient data for correlation risk features")
                return {
                    "correlation_level": 0.0,
                    "correlation_stability": 0.0,
                    "beta_level": 0.0,
                    "beta_stability": 0.0,
                    "tail_correlation": 0.0,
                }

            asset_rets = aligned_data.iloc[:, 0]
            market_rets = aligned_data.iloc[:, 1]

            # Basic correlation
            correlation_level = asset_rets.corr(market_rets)
            correlation_level = (
                correlation_level if not np.isnan(correlation_level) else 0.0
            )

            # Rolling correlation for stability
            rolling_corr = CorrelationMetrics.rolling_correlation(
                asset_returns, market_returns, window=60
            )
            correlation_stability = (
                1.0 - rolling_corr.std() if not rolling_corr.empty else 0.0
            )
            correlation_stability = (
                correlation_stability if not np.isnan(correlation_stability) else 0.0
            )

            # Beta analysis
            beta_result = CorrelationMetrics.rolling_beta(
                asset_returns, market_returns, window=60
            )
            beta_level = (
                beta_result["beta"].mean() if not beta_result["beta"].empty else 0.0
            )
            beta_level = beta_level if not np.isnan(beta_level) else 0.0

            beta_stability = (
                1.0 - beta_result["beta"].std()
                if not beta_result["beta"].empty
                else 0.0
            )
            beta_stability = beta_stability if not np.isnan(beta_stability) else 0.0

            # Tail correlation (correlation during extreme market moves)
            extreme_threshold = market_rets.quantile(0.05)  # Bottom 5%
            extreme_periods = market_rets <= extreme_threshold

            if extreme_periods.sum() > 5:
                tail_correlation = asset_rets[extreme_periods].corr(
                    market_rets[extreme_periods]
                )
                tail_correlation = (
                    tail_correlation if not np.isnan(tail_correlation) else 0.0
                )
            else:
                tail_correlation = correlation_level

            return {
                "correlation_level": float(correlation_level),
                "correlation_stability": float(correlation_stability),
                "beta_level": float(beta_level),
                "beta_stability": float(beta_stability),
                "tail_correlation": float(tail_correlation),
            }

        except Exception as e:
            logger.error(f"Error calculating correlation risk features: {str(e)}")
            return {
                "correlation_level": 0.0,
                "correlation_stability": 0.0,
                "beta_level": 0.0,
                "beta_stability": 0.0,
                "tail_correlation": 0.0,
            }
