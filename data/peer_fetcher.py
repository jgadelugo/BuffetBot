"""
Peer Fetcher Module

This module provides functionality to fetch peer/competitor stocks for a given ticker.
It attempts to use free APIs first, then falls back to a static peer mapping.

All functions implement robust fault handling - if real data is unavailable, the
functions return empty structures with clear metadata flags, ensuring
the advisory pipeline continues without breaking.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

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


class PeerResult(TypedDict):
    """Type definition for peer fetcher result structure."""

    peers: list[str]
    data_available: bool
    error_message: str | None
    ticker: str
    data_source: str | None
    total_peers_found: int


class PeerInfoResult(TypedDict):
    """Type definition for detailed peer info result structure."""

    peer_data: list[PeerData]
    data_available: bool
    error_message: str | None
    ticker: str
    successful_lookups: int
    failed_lookups: int


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


def get_peers(ticker: str) -> PeerResult:
    """
    Get peer/competitor stocks for a given ticker with robust error handling.

    This function attempts to fetch peer companies using the following strategy:
    1. Try to fetch from yfinance API
    2. Fall back to static peer mapping
    3. Return empty result with error information if no peers found

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA')

    Returns:
        PeerResult: Dictionary containing:
            - peers (List[str]): List of peer ticker symbols
            - data_available (bool): True if peers were found successfully
            - error_message (Optional[str]): Error description if data_available=False
            - ticker (str): The requested ticker symbol (normalized)
            - data_source (Optional[str]): Source of peer data ('yfinance_api' or 'static_mapping')
            - total_peers_found (int): Number of peers found

    Examples:
        >>> result = get_peers('NVDA')
        >>> if result['data_available']:
        ...     print(f"Found {result['total_peers_found']} peers: {result['peers']}")
        ... else:
        ...     print(f"Peers unavailable: {result['error_message']}")

        >>> # Example successful result:
        >>> {
        ...     'peers': ['AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMAT', 'LRCX'],
        ...     'data_available': True,
        ...     'error_message': None,
        ...     'ticker': 'NVDA',
        ...     'data_source': 'static_mapping',
        ...     'total_peers_found': 8
        ... }

    Note:
        This function never raises exceptions - it always returns a valid
        PeerResult structure. Check the 'data_available' flag to determine
        if peer data was successfully fetched.
    """
    # Initialize default response structure
    result: PeerResult = {
        "peers": [],
        "data_available": False,
        "error_message": None,
        "ticker": ticker,
        "data_source": None,
        "total_peers_found": 0,
    }

    try:
        # Validate and normalize ticker
        try:
            normalized_ticker = _validate_ticker(ticker)
        except PeerFetchError as e:
            error_msg = f"Invalid ticker: {str(e)}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result
        except Exception as e:
            error_msg = f"Unexpected error validating ticker '{ticker}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            return result

        result["ticker"] = normalized_ticker  # Update with normalized ticker
        logger.info(f"Fetching peers for ticker: {normalized_ticker}")

        peers = []
        data_source = None

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
            try:
                peers = _fetch_peers_from_static_map(normalized_ticker)
                if peers:
                    data_source = "static_mapping"
            except Exception as e:
                logger.warning(f"Static mapping fetch failed: {str(e)}")

        # Filter and validate peers
        if peers:
            try:
                valid_peers = _filter_valid_peers(peers, normalized_ticker)

                if valid_peers:
                    result.update(
                        {
                            "peers": valid_peers,
                            "data_available": True,
                            "error_message": None,
                            "data_source": data_source,
                            "total_peers_found": len(valid_peers),
                        }
                    )

                    logger.info(
                        f"Successfully fetched {len(valid_peers)} peers for {normalized_ticker}",
                        extra={
                            "ticker": normalized_ticker,
                            "peer_count": len(valid_peers),
                            "data_source": data_source,
                            "peers": valid_peers,
                        },
                    )
                    return result
                else:
                    error_msg = f"No valid peers found after filtering for ticker {normalized_ticker}"
                    logger.warning(error_msg)
                    result["error_message"] = error_msg
                    return result

            except Exception as e:
                error_msg = f"Error filtering peers for {normalized_ticker}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                result["error_message"] = error_msg
                return result
        else:
            error_msg = f"No peers found for ticker {normalized_ticker}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

    except Exception as e:
        # Catch-all for any unexpected errors
        error_msg = f"Critical error fetching peers for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
        return result


def get_peer_info(ticker: str) -> PeerInfoResult:
    """
    Get detailed peer information including company names and sectors with robust error handling.

    Args:
        ticker: Stock ticker symbol

    Returns:
        PeerInfoResult: Dictionary containing:
            - peer_data (List[PeerData]): List of peer data objects with additional information
            - data_available (bool): True if peer info was fetched successfully
            - error_message (Optional[str]): Error description if data_available=False
            - ticker (str): The requested ticker symbol (normalized)
            - successful_lookups (int): Number of peers with detailed info retrieved
            - failed_lookups (int): Number of peers where detailed info failed

    Examples:
        >>> result = get_peer_info('AAPL')
        >>> if result['data_available']:
        ...     for peer in result['peer_data']:
        ...         print(f"{peer.ticker}: {peer.name}")
        ... else:
        ...     print(f"Peer info unavailable: {result['error_message']}")

        >>> # Example successful result structure:
        >>> {
        ...     'peer_data': [
        ...         PeerData(ticker='MSFT', name='Microsoft Corporation', sector='Technology'),
        ...         PeerData(ticker='GOOGL', name='Alphabet Inc.', sector='Technology'),
        ...     ],
        ...     'data_available': True,
        ...     'error_message': None,
        ...     'ticker': 'AAPL',
        ...     'successful_lookups': 2,
        ...     'failed_lookups': 0
        ... }

    Note:
        This function never raises exceptions - it always returns a valid
        PeerInfoResult structure. Check the 'data_available' flag to determine
        if peer data was successfully fetched.
    """
    # Initialize default response structure
    result: PeerInfoResult = {
        "peer_data": [],
        "data_available": False,
        "error_message": None,
        "ticker": ticker,
        "successful_lookups": 0,
        "failed_lookups": 0,
    }

    try:
        # First get the list of peer tickers
        peer_result = get_peers(ticker)

        if not peer_result["data_available"]:
            error_msg = f"Cannot get peer info - peer lookup failed: {peer_result['error_message']}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            result["ticker"] = peer_result["ticker"]  # Use normalized ticker
            return result

        peer_tickers = peer_result["peers"]
        result["ticker"] = peer_result["ticker"]  # Use normalized ticker

        if not peer_tickers:
            error_msg = f"No peers found for {peer_result['ticker']}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

        logger.info(
            f"Fetching detailed info for {len(peer_tickers)} peers of {peer_result['ticker']}"
        )

        peer_info = []
        successful_count = 0
        failed_count = 0

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
                successful_count += 1
                logger.debug(f"Added peer info for {peer_ticker}: {peer_data.name}")

            except Exception as e:
                # If we can't get additional info, just add basic ticker info
                logger.warning(
                    f"Could not fetch detailed info for peer {peer_ticker}: {str(e)}"
                )
                peer_info.append(PeerData(ticker=peer_ticker))
                failed_count += 1

        # Set successful result
        result.update(
            {
                "peer_data": peer_info,
                "data_available": True,
                "error_message": None,
                "successful_lookups": successful_count,
                "failed_lookups": failed_count,
            }
        )

        logger.info(
            f"Successfully fetched detailed info for {len(peer_info)} peers "
            f"({successful_count} successful, {failed_count} failed lookups)"
        )
        return result

    except Exception as e:
        # Catch-all for any unexpected errors
        error_msg = f"Critical error fetching peer info for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
        return result


def add_static_peers(ticker: str, peers: list[str]) -> dict[str, Any]:
    """
    Add or update peer mapping in the static peer map with robust error handling.

    This is useful for adding custom peer relationships that aren't
    available through APIs.

    Args:
        ticker: Stock ticker symbol
        peers: List of peer ticker symbols

    Returns:
        Dict[str, Any]: Result dictionary containing:
            - success (bool): True if peers were added successfully
            - error_message (Optional[str]): Error description if success=False
            - ticker (Optional[str]): The normalized ticker symbol
            - valid_peers_added (int): Number of valid peers added
            - invalid_peers_skipped (int): Number of invalid peers skipped

    Examples:
        >>> result = add_static_peers('TSLA', ['RIVN', 'LCID', 'NIO'])
        >>> if result['success']:
        ...     print(f"Added {result['valid_peers_added']} peers for {result['ticker']}")
        ... else:
        ...     print(f"Failed to add peers: {result['error_message']}")

        >>> # Example successful result:
        >>> {
        ...     'success': True,
        ...     'error_message': None,
        ...     'ticker': 'TSLA',
        ...     'valid_peers_added': 3,
        ...     'invalid_peers_skipped': 0
        ... }

    Note:
        This function never raises exceptions - it always returns a valid
        result dictionary. Check the 'success' flag to determine if the
        operation completed successfully.
    """
    # Initialize default response structure
    result = {
        "success": False,
        "error_message": None,
        "ticker": None,
        "valid_peers_added": 0,
        "invalid_peers_skipped": 0,
    }

    try:
        # Validate ticker
        try:
            normalized_ticker = _validate_ticker(ticker)
            result["ticker"] = normalized_ticker
        except PeerFetchError as e:
            error_msg = f"Invalid ticker: {str(e)}"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result
        except Exception as e:
            error_msg = f"Unexpected error validating ticker '{ticker}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            return result

        # Validate peers input
        if not peers or not isinstance(peers, list):
            error_msg = "Peers must be a non-empty list"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            return result

        # Validate all peer tickers
        valid_peers = []
        invalid_count = 0

        for peer in peers:
            try:
                valid_peer = _validate_ticker(peer)
                valid_peers.append(valid_peer)
            except PeerFetchError as e:
                logger.warning(f"Skipping invalid peer {peer}: {str(e)}")
                invalid_count += 1
            except Exception as e:
                logger.warning(f"Unexpected error validating peer {peer}: {str(e)}")
                invalid_count += 1

        if not valid_peers:
            error_msg = "No valid peers provided after validation"
            logger.warning(error_msg)
            result["error_message"] = error_msg
            result["invalid_peers_skipped"] = invalid_count
            return result

        # Add to static mapping
        try:
            STATIC_PEER_MAP[normalized_ticker] = valid_peers

            result.update(
                {
                    "success": True,
                    "error_message": None,
                    "valid_peers_added": len(valid_peers),
                    "invalid_peers_skipped": invalid_count,
                }
            )

            logger.info(
                f"Added/updated {len(valid_peers)} peers for {normalized_ticker} in static mapping "
                f"(skipped {invalid_count} invalid peers)"
            )
            return result

        except Exception as e:
            error_msg = f"Failed to update static peer mapping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["error_message"] = error_msg
            result["valid_peers_added"] = 0
            result["invalid_peers_skipped"] = invalid_count
            return result

    except Exception as e:
        # Catch-all for any unexpected errors
        error_msg = f"Critical error adding static peers for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["error_message"] = error_msg
        return result
