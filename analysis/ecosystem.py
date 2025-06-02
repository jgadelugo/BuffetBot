"""
Ecosystem Analysis Module

This module provides ecosystem-based analysis for enhanced options recommendations.
It analyzes how related stocks (peers, suppliers, customers, sector members) are performing
to create a "network signal" that can confirm or veto trade recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from data.fetcher import DataFetcher
from data.peer_fetcher import PeerData, PeerFetchError, get_peer_info, get_peers
from utils.correlation_math import (
    CorrelationResult,
    EcosystemScore,
    calculate_ecosystem_correlations,
    calculate_ecosystem_score,
    calculate_returns,
)
from utils.errors import DataError, DataFetcherError, ErrorSeverity, handle_data_error
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@dataclass
class EcosystemMetrics:
    """Data class to hold ecosystem-wide metrics."""

    avg_rsi: float
    avg_implied_volatility: float | None = None
    avg_returns_1d: float = 0.0
    avg_returns_5d: float = 0.0
    avg_returns_30d: float = 0.0
    sector_momentum: float = 0.0
    volatility_regime: str = "normal"  # "low", "normal", "high"

    @property
    def momentum_signal(self) -> str:
        """Categorize momentum signal."""
        if self.avg_returns_5d > 0.02:  # >2% in 5 days
            return "strong_bullish"
        elif self.avg_returns_5d > 0.005:  # >0.5% in 5 days
            return "bullish"
        elif self.avg_returns_5d < -0.02:  # <-2% in 5 days
            return "strong_bearish"
        elif self.avg_returns_5d < -0.005:  # <-0.5% in 5 days
            return "bearish"
        else:
            return "neutral"


@dataclass
class EcosystemAnalysis:
    """Comprehensive ecosystem analysis results."""

    ticker: str
    peers: list[str]
    ecosystem_score: EcosystemScore
    ecosystem_metrics: EcosystemMetrics
    individual_correlations: dict[str, CorrelationResult]
    peer_data: list[PeerData]
    signal_strength: float  # 0-1, overall signal strength
    recommendation: str  # "confirm", "neutral", "veto"
    confidence: float  # 0-1, confidence in the recommendation
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "peers": self.peers,
            "ecosystem_score": {
                "normalized_score": self.ecosystem_score.normalized_score,
                "avg_correlation": self.ecosystem_score.avg_correlation,
                "confidence_score": self.ecosystem_score.confidence_score,
                "sample_size": self.ecosystem_score.sample_size,
            },
            "ecosystem_metrics": {
                "avg_rsi": self.ecosystem_metrics.avg_rsi,
                "avg_returns_1d": self.ecosystem_metrics.avg_returns_1d,
                "avg_returns_5d": self.ecosystem_metrics.avg_returns_5d,
                "avg_returns_30d": self.ecosystem_metrics.avg_returns_30d,
                "momentum_signal": self.ecosystem_metrics.momentum_signal,
                "volatility_regime": self.ecosystem_metrics.volatility_regime,
            },
            "signal_strength": self.signal_strength,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class EcosystemAnalyzer:
    """
    Main class for performing ecosystem analysis on stocks.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher | None = None,
        correlation_window: int = 60,  # days for correlation calculation
        min_correlation_periods: int = 30,
    ):
        """
        Initialize the ecosystem analyzer.

        Args:
            data_fetcher: DataFetcher instance for getting market data
            correlation_window: Number of days for correlation analysis
            min_correlation_periods: Minimum periods required for correlation
        """
        self.data_fetcher = data_fetcher or DataFetcher()
        self.correlation_window = correlation_window
        self.min_correlation_periods = min_correlation_periods

        logger.info(
            f"Initialized EcosystemAnalyzer with {correlation_window}d correlation window"
        )

    def _fetch_peer_price_data(
        self, peer_tickers: list[str], period: str = "6mo"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch price data for peer stocks.

        Args:
            peer_tickers: List of peer ticker symbols
            period: Time period for data fetching

        Returns:
            Dict mapping ticker to price DataFrame
        """
        peer_data = {}
        failed_tickers = []

        logger.info(f"Fetching price data for {len(peer_tickers)} peer stocks")

        for ticker in peer_tickers:
            try:
                price_data = self.data_fetcher.fetch_price_history(
                    ticker=ticker, period=period, interval="1d"
                )
                peer_data[ticker] = price_data
                logger.debug(
                    f"Successfully fetched data for {ticker}: {len(price_data)} records"
                )

            except (DataFetcherError, Exception) as e:
                logger.warning(f"Failed to fetch data for peer {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        if failed_tickers:
            logger.warning(
                f"Failed to fetch data for {len(failed_tickers)} peers: {failed_tickers}"
            )

        logger.info(
            f"Successfully fetched data for {len(peer_data)}/{len(peer_tickers)} peers"
        )
        return peer_data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI for a price series.

        Args:
            prices: Series of closing prices
            period: RSI calculation period

        Returns:
            Current RSI value
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data

        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_ecosystem_metrics(
        self, peer_data: dict[str, pd.DataFrame]
    ) -> EcosystemMetrics:
        """
        Calculate aggregate metrics across the ecosystem.

        Args:
            peer_data: Dictionary of peer price data

        Returns:
            EcosystemMetrics with calculated values
        """
        if not peer_data:
            return EcosystemMetrics(avg_rsi=50.0)

        rsi_values = []
        returns_1d = []
        returns_5d = []
        returns_30d = []
        volatilities = []

        for ticker, data in peer_data.items():
            if len(data) < 2:
                continue

            try:
                # Calculate RSI
                rsi = self._calculate_rsi(data["Close"])
                rsi_values.append(rsi)

                # Calculate returns
                close_prices = data["Close"]
                if len(close_prices) >= 2:
                    ret_1d = (close_prices.iloc[-1] / close_prices.iloc[-2]) - 1
                    returns_1d.append(ret_1d)

                if len(close_prices) >= 6:
                    ret_5d = (close_prices.iloc[-1] / close_prices.iloc[-6]) - 1
                    returns_5d.append(ret_5d)

                if len(close_prices) >= 31:
                    ret_30d = (close_prices.iloc[-1] / close_prices.iloc[-31]) - 1
                    returns_30d.append(ret_30d)

                # Calculate volatility (30-day)
                if len(close_prices) >= 30:
                    returns = close_prices.pct_change().dropna()
                    vol = returns.rolling(30).std().iloc[-1] * np.sqrt(
                        252
                    )  # Annualized
                    if not pd.isna(vol):
                        volatilities.append(vol)

            except Exception as e:
                logger.warning(f"Error calculating metrics for {ticker}: {str(e)}")
                continue

        # Calculate averages
        avg_rsi = np.mean(rsi_values) if rsi_values else 50.0
        avg_ret_1d = np.mean(returns_1d) if returns_1d else 0.0
        avg_ret_5d = np.mean(returns_5d) if returns_5d else 0.0
        avg_ret_30d = np.mean(returns_30d) if returns_30d else 0.0
        avg_volatility = np.mean(volatilities) if volatilities else 0.2

        # Determine volatility regime
        if avg_volatility < 0.15:
            vol_regime = "low"
        elif avg_volatility > 0.35:
            vol_regime = "high"
        else:
            vol_regime = "normal"

        # Calculate sector momentum (weighted average of different timeframes)
        sector_momentum = 0.5 * avg_ret_5d + 0.3 * avg_ret_30d + 0.2 * avg_ret_1d

        return EcosystemMetrics(
            avg_rsi=avg_rsi,
            avg_returns_1d=avg_ret_1d,
            avg_returns_5d=avg_ret_5d,
            avg_returns_30d=avg_ret_30d,
            sector_momentum=sector_momentum,
            volatility_regime=vol_regime,
        )

    def _generate_recommendation(
        self, ecosystem_score: EcosystemScore, ecosystem_metrics: EcosystemMetrics
    ) -> tuple[str, float, float]:
        """
        Generate trading recommendation based on ecosystem analysis.

        Args:
            ecosystem_score: Correlation-based ecosystem score
            ecosystem_metrics: Aggregate ecosystem metrics

        Returns:
            Tuple of (recommendation, signal_strength, confidence)
        """
        # Base signal from correlation strength
        correlation_signal = ecosystem_score.normalized_score

        # Momentum signal adjustment
        momentum_boost = 0.0
        if ecosystem_metrics.momentum_signal == "strong_bullish":
            momentum_boost = 0.3
        elif ecosystem_metrics.momentum_signal == "bullish":
            momentum_boost = 0.15
        elif ecosystem_metrics.momentum_signal == "strong_bearish":
            momentum_boost = -0.3
        elif ecosystem_metrics.momentum_signal == "bearish":
            momentum_boost = -0.15

        # RSI signal adjustment
        rsi_adjustment = 0.0
        if ecosystem_metrics.avg_rsi > 70:  # Overbought
            rsi_adjustment = -0.1
        elif ecosystem_metrics.avg_rsi < 30:  # Oversold
            rsi_adjustment = 0.1

        # Calculate combined signal strength
        signal_strength = correlation_signal + momentum_boost + rsi_adjustment
        signal_strength = max(0.0, min(1.0, signal_strength))  # Clamp to [0,1]

        # Generate recommendation
        if signal_strength >= 0.7:
            recommendation = "confirm"
        elif signal_strength <= 0.3:
            recommendation = "veto"
        else:
            recommendation = "neutral"

        # Calculate confidence
        confidence_factors = [
            ecosystem_score.confidence_score,  # Correlation confidence
            min(1.0, len(ecosystem_score.peers) / 5),  # Peer count confidence
            0.8
            if ecosystem_metrics.volatility_regime == "normal"
            else 0.6,  # Vol regime
        ]
        confidence = np.mean(confidence_factors)

        logger.info(
            f"Ecosystem recommendation: {recommendation} "
            f"(strength: {signal_strength:.3f}, confidence: {confidence:.3f})"
        )

        return recommendation, signal_strength, confidence

    def analyze_ecosystem(
        self, ticker: str, custom_peers: list[str] | None = None
    ) -> EcosystemAnalysis:
        """
        Perform comprehensive ecosystem analysis for a given ticker.

        Args:
            ticker: Target stock ticker
            custom_peers: Optional custom list of peer tickers

        Returns:
            EcosystemAnalysis with complete results

        Raises:
            PeerFetchError: If peer fetching fails
            DataFetcherError: If price data fetching fails
        """
        logger.info(f"Starting ecosystem analysis for {ticker}")

        try:
            # Step 1: Get peer stocks
            if custom_peers:
                peers = custom_peers
                logger.info(f"Using custom peers for {ticker}: {peers}")
            else:
                peers = get_peers(ticker)
                logger.info(f"Found {len(peers)} peers for {ticker}")

            # Step 2: Get detailed peer information
            peer_data_info = get_peer_info(ticker)

            # Step 3: Fetch target stock price data
            target_data = self.data_fetcher.fetch_price_history(
                ticker=ticker, period="6mo", interval="1d"
            )

            # Step 4: Fetch peer price data
            peer_price_data = self._fetch_peer_price_data(peers)

            if not peer_price_data:
                raise DataFetcherError(
                    DataError(
                        code="NO_PEER_DATA",
                        message=f"No peer price data available for {ticker}",
                        severity=ErrorSeverity.HIGH,
                    )
                )

            # Step 5: Calculate returns for correlation analysis
            target_returns = calculate_returns(target_data["Close"])
            target_returns.name = ticker

            peer_returns = {}
            for peer_ticker, price_data in peer_price_data.items():
                returns = calculate_returns(price_data["Close"])
                returns.name = peer_ticker
                peer_returns[peer_ticker] = returns

            # Step 6: Calculate correlations
            correlations = calculate_ecosystem_correlations(
                target_returns=target_returns,
                peer_returns_dict=peer_returns,
                min_periods=self.min_correlation_periods,
            )

            # Step 7: Calculate ecosystem score
            ecosystem_score = calculate_ecosystem_score(correlations)

            # Step 8: Calculate ecosystem metrics
            ecosystem_metrics = self._calculate_ecosystem_metrics(peer_price_data)

            # Step 9: Generate recommendation
            recommendation, signal_strength, confidence = self._generate_recommendation(
                ecosystem_score, ecosystem_metrics
            )

            # Step 10: Create comprehensive analysis result
            analysis = EcosystemAnalysis(
                ticker=ticker,
                peers=list(peer_price_data.keys()),  # Only successful peers
                ecosystem_score=ecosystem_score,
                ecosystem_metrics=ecosystem_metrics,
                individual_correlations=correlations,
                peer_data=peer_data_info,
                signal_strength=signal_strength,
                recommendation=recommendation,
                confidence=confidence,
            )

            logger.info(
                f"Ecosystem analysis complete for {ticker}: "
                f"{recommendation} (strength: {signal_strength:.3f})"
            )

            return analysis

        except Exception as e:
            error_msg = f"Ecosystem analysis failed for {ticker}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def analyze_multiple_tickers(
        self, tickers: list[str]
    ) -> dict[str, EcosystemAnalysis]:
        """
        Perform ecosystem analysis for multiple tickers.

        Args:
            tickers: List of ticker symbols to analyze

        Returns:
            Dictionary mapping ticker to EcosystemAnalysis
        """
        results = {}

        logger.info(f"Starting ecosystem analysis for {len(tickers)} tickers")

        for ticker in tickers:
            try:
                analysis = self.analyze_ecosystem(ticker)
                results[ticker] = analysis
                logger.debug(f"Analysis complete for {ticker}")

            except Exception as e:
                logger.error(f"Analysis failed for {ticker}: {str(e)}")
                continue

        logger.info(
            f"Completed ecosystem analysis for {len(results)}/{len(tickers)} tickers"
        )
        return results

    def get_ecosystem_summary(
        self, analysis: EcosystemAnalysis
    ) -> dict[str, str | float | int]:
        """
        Generate a summary of ecosystem analysis results.

        Args:
            analysis: EcosystemAnalysis object

        Returns:
            Dictionary with key summary metrics
        """
        return {
            "ticker": analysis.ticker,
            "peer_count": len(analysis.peers),
            "ecosystem_score": round(analysis.ecosystem_score.normalized_score, 3),
            "avg_correlation": round(analysis.ecosystem_score.avg_correlation, 3),
            "momentum_signal": analysis.ecosystem_metrics.momentum_signal,
            "avg_rsi": round(analysis.ecosystem_metrics.avg_rsi, 1),
            "volatility_regime": analysis.ecosystem_metrics.volatility_regime,
            "recommendation": analysis.recommendation,
            "signal_strength": round(analysis.signal_strength, 3),
            "confidence": round(analysis.confidence, 3),
            "strongest_correlation": max(
                analysis.individual_correlations.values(),
                key=lambda x: abs(x.correlation),
            ).ticker2
            if analysis.individual_correlations
            else "N/A",
        }
