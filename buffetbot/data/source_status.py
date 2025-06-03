"""
Data Source Status Module

This module provides centralized data availability status reporting for the BuffetBot
financial options app. It checks the health and availability of all major data sources
including forecast data, options data, and peer data.

The main function get_data_availability_status() tests each data fetcher without
running full analysis, providing a quick health check of the data pipeline.

All functions implement robust error handling and never crash, ensuring the status
checker itself doesn't break the application flow.
"""

import logging
from typing import Any, Dict

from buffetbot.data.forecast_fetcher import get_analyst_forecast
from buffetbot.data.options_fetcher import fetch_long_dated_calls
from buffetbot.data.peer_fetcher import get_peers
from buffetbot.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, "logs/source_status.log")


def get_data_availability_status(ticker: str) -> dict[str, Any]:
    """
    Get comprehensive data availability status for a given ticker.

    This function tests each of the existing data fetchers to determine if data
    can be successfully fetched from each source. It performs lightweight checks
    without running full analysis, using each fetcher's data_available and
    source_used metadata.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Dict[str, Any]: Dictionary containing availability status for each data source:
            {
                "ticker": "AAPL",
                "forecast": {"available": True, "source": "yahoo"},
                "options": {"available": True, "source": "yahoo"},
                "peers": {"available": True, "source": "fallback_static"},
                "overall_health": "healthy",  # "healthy", "partial", "unhealthy"
                "total_sources": 3,
                "available_sources": 3,
                "unavailable_sources": 0,
                "timestamp": "2024-01-15T10:30:00Z"
            }

    Examples:
        >>> status = get_data_availability_status('AAPL')
        >>> if status['forecast']['available']:
        ...     print(f"Forecast data available from {status['forecast']['source']}")
        >>> print(f"Overall health: {status['overall_health']}")

        >>> # Print formatted status
        >>> print_data_status(status)

    Note:
        This function never raises exceptions - it always returns a valid
        dictionary with status information for all sources, even if individual
        fetchers fail.
    """
    from datetime import datetime

    # Initialize default response structure
    status = {
        "ticker": ticker.upper().strip() if ticker else "UNKNOWN",
        "forecast": {"available": False, "source": "none"},
        "options": {"available": False, "source": "none"},
        "peers": {"available": False, "source": "none"},
        "overall_health": "unhealthy",
        "total_sources": 3,
        "available_sources": 0,
        "unavailable_sources": 3,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Input validation
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        logger.warning("Invalid ticker provided to get_data_availability_status")
        status["ticker"] = "INVALID"
        return status

    normalized_ticker = ticker.upper().strip()
    status["ticker"] = normalized_ticker

    logger.info(f"Checking data availability status for {normalized_ticker}")

    # Check forecast data availability
    try:
        logger.debug(f"Testing forecast data availability for {normalized_ticker}")
        forecast_result = get_analyst_forecast(normalized_ticker)

        if forecast_result.get("data_available", False):
            status["forecast"]["available"] = True
            status["forecast"]["source"] = forecast_result.get("source_used", "unknown")
            logger.info(
                f"✅ Forecast data available for {normalized_ticker} from {status['forecast']['source']}"
            )
        else:
            status["forecast"]["available"] = False
            status["forecast"]["source"] = "none"
            error_msg = forecast_result.get("error_message", "Unknown error")
            logger.warning(
                f"❌ Forecast data unavailable for {normalized_ticker}: {error_msg}"
            )

    except Exception as e:
        logger.warning(
            f"❌ Exception checking forecast data for {normalized_ticker}: {str(e)}"
        )
        status["forecast"]["available"] = False
        status["forecast"]["source"] = "error"

    # Check options data availability
    try:
        logger.debug(f"Testing options data availability for {normalized_ticker}")
        options_result = fetch_long_dated_calls(
            normalized_ticker, min_days_to_expiry=180
        )

        if options_result.get("data_available", False):
            status["options"]["available"] = True
            status["options"]["source"] = options_result.get("source_used", "unknown")
            logger.info(
                f"✅ Options data available for {normalized_ticker} from {status['options']['source']}"
            )
        else:
            status["options"]["available"] = False
            status["options"]["source"] = "none"
            error_msg = options_result.get("error_message", "Unknown error")
            logger.warning(
                f"❌ Options data unavailable for {normalized_ticker}: {error_msg}"
            )

    except Exception as e:
        logger.warning(
            f"❌ Exception checking options data for {normalized_ticker}: {str(e)}"
        )
        status["options"]["available"] = False
        status["options"]["source"] = "error"

    # Check peers data availability
    try:
        logger.debug(f"Testing peers data availability for {normalized_ticker}")
        peers_result = get_peers(normalized_ticker)

        if peers_result.get("data_available", False):
            status["peers"]["available"] = True
            status["peers"]["source"] = peers_result.get("source_used", "unknown")
            logger.info(
                f"✅ Peers data available for {normalized_ticker} from {status['peers']['source']}"
            )
        else:
            status["peers"]["available"] = False
            status["peers"]["source"] = "none"
            error_msg = peers_result.get("error_message", "Unknown error")
            logger.warning(
                f"❌ Peers data unavailable for {normalized_ticker}: {error_msg}"
            )

    except Exception as e:
        logger.warning(
            f"❌ Exception checking peers data for {normalized_ticker}: {str(e)}"
        )
        status["peers"]["available"] = False
        status["peers"]["source"] = "error"

    # Calculate overall health metrics
    available_count = sum(
        [
            status["forecast"]["available"],
            status["options"]["available"],
            status["peers"]["available"],
        ]
    )

    status["available_sources"] = available_count
    status["unavailable_sources"] = status["total_sources"] - available_count

    # Determine overall health
    if available_count == 3:
        status["overall_health"] = "healthy"
    elif available_count >= 2:
        status["overall_health"] = "partial"
    else:
        status["overall_health"] = "unhealthy"

    # Log summary
    logger.info(
        f"Data availability status for {normalized_ticker}: "
        f"{available_count}/{status['total_sources']} sources available "
        f"(Health: {status['overall_health']})"
    )

    return status


def print_data_status(status_dict: dict[str, Any]) -> None:
    """
    Print a nicely formatted data availability status report.

    This helper function takes the status dictionary returned by
    get_data_availability_status() and formats it for CLI/debug display.

    Args:
        status_dict: Status dictionary from get_data_availability_status()

    Examples:
        >>> status = get_data_availability_status('AAPL')
        >>> print_data_status(status)

        Data Availability Status for AAPL
        =====================================
        Overall Health: healthy (3/3 sources available)

        📊 Forecast Data: ✅ Available (Source: yahoo)
        📈 Options Data:  ✅ Available (Source: yahoo)
        👥 Peers Data:    ✅ Available (Source: fallback_static)

        Status checked at: 2024-01-15T10:30:00Z

    Note:
        This function never raises exceptions. If the status_dict is malformed,
        it will display an error message instead of crashing.
    """
    try:
        if not isinstance(status_dict, dict):
            print("❌ Error: Invalid status data provided")
            return

        ticker = status_dict.get("ticker", "UNKNOWN")
        overall_health = status_dict.get("overall_health", "unknown")
        available_sources = status_dict.get("available_sources", 0)
        total_sources = status_dict.get("total_sources", 0)
        timestamp = status_dict.get("timestamp", "Unknown")

        # Health status emoji
        health_emoji = {"healthy": "💚", "partial": "🟡", "unhealthy": "🔴"}.get(
            overall_health, "❓"
        )

        print(f"\nData Availability Status for {ticker}")
        print("=" * (len(f"Data Availability Status for {ticker}") + 1))
        print(
            f"Overall Health: {health_emoji} {overall_health} ({available_sources}/{total_sources} sources available)"
        )
        print()

        # Format each data source status
        sources = [
            ("📊 Forecast Data", "forecast"),
            ("📈 Options Data", "options"),
            ("👥 Peers Data", "peers"),
        ]

        for label, key in sources:
            source_info = status_dict.get(key, {})
            available = source_info.get("available", False)
            source = source_info.get("source", "unknown")

            status_icon = "✅" if available else "❌"
            availability_text = "Available" if available else "Unavailable"

            # Format the line with consistent spacing
            print(f"{label}: {status_icon} {availability_text:<12} (Source: {source})")

        print(f"\nStatus checked at: {timestamp}")
        print()

    except Exception as e:
        logger.error(f"Error formatting status display: {str(e)}")
        print(f"❌ Error formatting status display: {str(e)}")


def get_source_health_summary(tickers: list[str]) -> dict[str, Any]:
    """
    Get a health summary for multiple tickers.

    This function checks data availability across multiple tickers and provides
    an aggregate view of data source health.

    Args:
        tickers: List of ticker symbols to check

    Returns:
        Dict[str, Any]: Summary of data source health across all tickers:
            {
                "total_tickers_checked": 5,
                "healthy_tickers": 3,
                "partial_tickers": 1,
                "unhealthy_tickers": 1,
                "source_success_rates": {
                    "forecast": 0.8,  # 80% success
                    "options": 0.6,   # 60% success
                    "peers": 1.0      # 100% success
                },
                "most_reliable_source": "peers",
                "least_reliable_source": "options"
            }

    Examples:
        >>> summary = get_source_health_summary(['AAPL', 'MSFT', 'GOOGL'])
        >>> print(f"Most reliable source: {summary['most_reliable_source']}")
        >>> print(f"Forecast success rate: {summary['source_success_rates']['forecast']:.1%}")

    Note:
        This function never raises exceptions. If no valid tickers are provided,
        it returns a summary indicating zero checks performed.
    """
    try:
        if not tickers or not isinstance(tickers, list):
            logger.warning("Invalid tickers list provided to get_source_health_summary")
            return {
                "total_tickers_checked": 0,
                "healthy_tickers": 0,
                "partial_tickers": 0,
                "unhealthy_tickers": 0,
                "source_success_rates": {"forecast": 0.0, "options": 0.0, "peers": 0.0},
                "most_reliable_source": "none",
                "least_reliable_source": "none",
            }

        logger.info(f"Checking source health summary for {len(tickers)} tickers")

        health_counts = {"healthy": 0, "partial": 0, "unhealthy": 0}
        source_successes = {"forecast": 0, "options": 0, "peers": 0}
        total_checked = 0

        for ticker in tickers:
            if not ticker or not isinstance(ticker, str):
                continue

            try:
                status = get_data_availability_status(ticker)

                # Count health status
                health = status.get("overall_health", "unhealthy")
                health_counts[health] = health_counts.get(health, 0) + 1

                # Count source successes
                for source in ["forecast", "options", "peers"]:
                    if status.get(source, {}).get("available", False):
                        source_successes[source] += 1

                total_checked += 1

            except Exception as e:
                logger.warning(f"Error checking status for ticker {ticker}: {str(e)}")
                continue

        # Calculate success rates
        source_success_rates = {}
        if total_checked > 0:
            for source in ["forecast", "options", "peers"]:
                source_success_rates[source] = source_successes[source] / total_checked
        else:
            source_success_rates = {"forecast": 0.0, "options": 0.0, "peers": 0.0}

        # Find most and least reliable sources
        if source_success_rates:
            most_reliable = max(
                source_success_rates.keys(), key=lambda k: source_success_rates[k]
            )
            least_reliable = min(
                source_success_rates.keys(), key=lambda k: source_success_rates[k]
            )
        else:
            most_reliable = "none"
            least_reliable = "none"

        summary = {
            "total_tickers_checked": total_checked,
            "healthy_tickers": health_counts["healthy"],
            "partial_tickers": health_counts["partial"],
            "unhealthy_tickers": health_counts["unhealthy"],
            "source_success_rates": source_success_rates,
            "most_reliable_source": most_reliable,
            "least_reliable_source": least_reliable,
        }

        logger.info(
            f"Source health summary: {total_checked} tickers checked, "
            f"{health_counts['healthy']} healthy, {health_counts['partial']} partial, "
            f"{health_counts['unhealthy']} unhealthy"
        )

        return summary

    except Exception as e:
        logger.error(f"Error generating source health summary: {str(e)}")
        return {
            "total_tickers_checked": 0,
            "healthy_tickers": 0,
            "partial_tickers": 0,
            "unhealthy_tickers": 0,
            "source_success_rates": {"forecast": 0.0, "options": 0.0, "peers": 0.0},
            "most_reliable_source": "error",
            "least_reliable_source": "error",
        }
