"""
Peer Fetcher Module

This module provides functionality to fetch peer/competitor stocks for a given ticker.
It attempts to use free APIs first, then falls back to a static peer mapping.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

from utils.errors import DataError, ErrorSeverity, handle_data_error
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class PeerFetchError(Exception):
    """Exception raised when peer fetching fails or no peers are found."""

    def __init__(
        self,
        message: str,
        ticker: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.ticker = ticker
        self.details = details or {}
        super().__init__(message)


@dataclass
class PeerData:
    """Data class to hold peer information."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None


# Static peer mapping as fallback
STATIC_PEER_MAP: dict[str, list[str]] = {
    # Technology - AI/Semiconductors
    "NVDA": ["AMD", "INTC", "TSM", "AVGO", "QCOM", "MU", "AMAT", "LRCX"],
    "AMD": ["NVDA", "INTC", "TSM", "AVGO", "QCOM", "MU"],
    "INTC": ["NVDA", "AMD", "TSM", "AVGO", "QCOM", "MU"],
    "TSM": ["NVDA", "AMD", "INTC", "AVGO", "UMC"],
    # Technology - Software/Cloud
    "MSFT": ["GOOGL", "AMZN", "CRM", "ORCL", "SNOW", "PLTR"],
    "GOOGL": ["MSFT", "AMZN", "META", "CRM", "ORCL"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "TGT"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR"],
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META"],
    # Electric Vehicles/Clean Energy
    "TSLA": ["RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM"],
    "RIVN": ["TSLA", "LCID", "NIO", "F", "GM"],
    "LCID": ["TSLA", "RIVN", "NIO", "XPEV", "LI"],
    # Financial Services
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "USB", "PNC"],
    "GS": ["MS", "JPM", "BAC", "C"],
    "MS": ["GS", "JPM", "BAC", "C"],
    # Healthcare/Biotech
    "JNJ": ["PFE", "UNH", "ABBV", "BMY", "MRK"],
    "PFE": ["JNJ", "UNH", "ABBV", "BMY", "MRK"],
    "MRNA": ["BNTX", "PFE", "JNJ", "NVAX"],
    # Retail/E-commerce
    "WMT": ["AMZN", "TGT", "COST", "HD", "LOW"],
    "TGT": ["WMT", "AMZN", "COST", "HD"],
    # Energy
    "XOM": ["CVX", "COP", "EOG", "PXD", "SLB"],
    "CVX": ["XOM", "COP", "EOG", "PXD"],
    # Airlines
    "AAL": ["DAL", "UAL", "LUV", "JBLU"],
    "DAL": ["AAL", "UAL", "LUV", "JBLU"],
    "UAL": ["AAL", "DAL", "LUV", "JBLU"],
    # Streaming/Entertainment
    "NFLX": ["DIS", "ROKU", "PARA", "WBD"],
    "DIS": ["NFLX", "PARA", "WBD"],
    # Crypto-related
    "COIN": ["MSTR", "RIOT", "MARA", "HUT"],
    "MSTR": ["COIN", "RIOT", "MARA"],
}


def _validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbol.

    Args:
        ticker: Raw ticker symbol

    Returns:
        str: Normalized ticker symbol

    Raises:
        PeerFetchError: If ticker is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise PeerFetchError("Ticker must be a non-empty string", ticker=ticker)

    # Normalize ticker (uppercase, strip whitespace)
    normalized = ticker.strip().upper()

    if not normalized:
        raise PeerFetchError(
            "Ticker cannot be empty after normalization", ticker=ticker
        )

    # Basic ticker format validation (letters, numbers, dots, hyphens)
    if not normalized.replace(".", "").replace("-", "").isalnum():
        raise PeerFetchError(f"Invalid ticker format: {ticker}", ticker=ticker)

    return normalized


def _fetch_peers_from_yfinance(ticker: str) -> list[str]:
    """
    Attempt to fetch peer companies using yfinance.

    This method uses yfinance's company info to find similar companies
    based on sector and industry information.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List[str]: List of peer ticker symbols

    Raises:
        Exception: If API call fails
    """
    logger.info(f"Attempting to fetch peers for {ticker} using yfinance")

    try:
        # Get company info
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            raise Exception("No company info available")

        sector = info.get("sector")
        industry = info.get("industry")

        logger.debug(f"Company info for {ticker}: sector={sector}, industry={industry}")

        # This is a simplified approach - in a real implementation,
        # you might use sector/industry to query similar companies
        # For now, we'll return empty list to fall back to static mapping
        return []

    except Exception as e:
        logger.warning(f"yfinance peer fetch failed for {ticker}: {str(e)}")
        raise


def _fetch_peers_from_static_map(ticker: str) -> list[str]:
    """
    Fetch peers from static mapping.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List[str]: List of peer ticker symbols
    """
    logger.info(f"Fetching peers for {ticker} from static mapping")

    peers = STATIC_PEER_MAP.get(ticker, [])

    if peers:
        logger.info(f"Found {len(peers)} peers for {ticker} in static mapping")
        logger.debug(f"Peers for {ticker}: {peers}")
    else:
        logger.warning(f"No peers found for {ticker} in static mapping")

    return peers


def _filter_valid_peers(peers: list[str], original_ticker: str) -> list[str]:
    """
    Filter and validate peer tickers.

    Args:
        peers: List of peer ticker symbols
        original_ticker: Original ticker to exclude from peers

    Returns:
        List[str]: Filtered list of valid peer tickers
    """
    valid_peers = []

    for peer in peers:
        try:
            # Validate peer ticker
            normalized_peer = _validate_ticker(peer)

            # Don't include the original ticker as its own peer
            if normalized_peer != original_ticker:
                valid_peers.append(normalized_peer)
            else:
                logger.debug(
                    f"Excluding original ticker {original_ticker} from peers list"
                )

        except PeerFetchError as e:
            logger.warning(f"Invalid peer ticker {peer}: {str(e)}")
            continue

    return valid_peers


def get_peers(ticker: str) -> list[str]:
    """
    Get peer/competitor stocks for a given ticker.

    This function attempts to fetch peer companies using the following strategy:
    1. Try to fetch from yfinance API
    2. Fall back to static peer mapping
    3. Raise PeerFetchError if no peers found

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA')

    Returns:
        List[str]: List of peer ticker symbols

    Raises:
        PeerFetchError: If ticker is invalid or no peers found

    Example:
        >>> peers = get_peers('NVDA')
        >>> print(peers)
        ['AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMAT', 'LRCX']
    """
    try:
        # Validate and normalize ticker
        normalized_ticker = _validate_ticker(ticker)
        logger.info(f"Fetching peers for ticker: {normalized_ticker}")

        peers = []
        data_source = "unknown"

        # Strategy 1: Try yfinance API
        try:
            peers = _fetch_peers_from_yfinance(normalized_ticker)
            if peers:
                data_source = "yfinance_api"
                logger.info(
                    f"Successfully fetched {len(peers)} peers from yfinance API"
                )
        except Exception as e:
            logger.debug(f"yfinance API fetch failed: {str(e)}")

        # Strategy 2: Fall back to static mapping
        if not peers:
            peers = _fetch_peers_from_static_map(normalized_ticker)
            if peers:
                data_source = "static_mapping"

        # Filter and validate peers
        valid_peers = _filter_valid_peers(peers, normalized_ticker)

        if not valid_peers:
            error_msg = f"No peers found for ticker {normalized_ticker}"
            logger.error(error_msg)
            raise PeerFetchError(
                error_msg,
                ticker=normalized_ticker,
                details={"data_source_attempted": data_source},
            )

        # Log success
        logger.info(
            f"Successfully fetched {len(valid_peers)} peers for {normalized_ticker}",
            extra={
                "ticker": normalized_ticker,
                "peer_count": len(valid_peers),
                "data_source": data_source,
                "peers": valid_peers,
            },
        )

        return valid_peers

    except PeerFetchError:
        # Re-raise PeerFetchError as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error fetching peers for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise PeerFetchError(
            error_msg,
            ticker=ticker,
            details={"error_type": type(e).__name__, "error_message": str(e)},
        )


def get_peer_info(ticker: str) -> list[PeerData]:
    """
    Get detailed peer information including company names and sectors.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List[PeerData]: List of peer data objects with additional information

    Raises:
        PeerFetchError: If ticker is invalid or no peers found
    """
    peer_tickers = get_peers(ticker)
    peer_info = []

    logger.info(f"Fetching detailed info for {len(peer_tickers)} peers of {ticker}")

    for peer_ticker in peer_tickers:
        try:
            # Try to get additional info from yfinance
            stock = yf.Ticker(peer_ticker)
            info = stock.info

            peer_data = PeerData(
                ticker=peer_ticker,
                name=info.get("longName") if info else None,
                sector=info.get("sector") if info else None,
                industry=info.get("industry") if info else None,
            )

            peer_info.append(peer_data)
            logger.debug(f"Added peer info for {peer_ticker}: {peer_data.name}")

        except Exception as e:
            # If we can't get additional info, just add basic ticker info
            logger.warning(
                f"Could not fetch detailed info for peer {peer_ticker}: {str(e)}"
            )
            peer_info.append(PeerData(ticker=peer_ticker))

    logger.info(f"Successfully fetched detailed info for {len(peer_info)} peers")
    return peer_info


def add_static_peers(ticker: str, peers: list[str]) -> None:
    """
    Add or update peer mapping in the static peer map.

    This is useful for adding custom peer relationships that aren't
    available through APIs.

    Args:
        ticker: Stock ticker symbol
        peers: List of peer ticker symbols

    Raises:
        PeerFetchError: If ticker or peers are invalid
    """
    normalized_ticker = _validate_ticker(ticker)

    if not peers or not isinstance(peers, list):
        raise PeerFetchError("Peers must be a non-empty list", ticker=ticker)

    # Validate all peer tickers
    valid_peers = []
    for peer in peers:
        try:
            valid_peer = _validate_ticker(peer)
            valid_peers.append(valid_peer)
        except PeerFetchError as e:
            logger.warning(f"Skipping invalid peer {peer}: {str(e)}")

    if not valid_peers:
        raise PeerFetchError("No valid peers provided", ticker=ticker)

    STATIC_PEER_MAP[normalized_ticker] = valid_peers
    logger.info(
        f"Added/updated {len(valid_peers)} peers for {normalized_ticker} in static mapping"
    )
