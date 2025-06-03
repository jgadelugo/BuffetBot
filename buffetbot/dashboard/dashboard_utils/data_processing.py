"""Data processing utilities including caching and validation."""

# Path setup to ensure proper imports
import sys
from pathlib import Path

# Ensure project root is in path for absolute imports
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, Optional

import streamlit as st

from buffetbot.data.cleaner import clean_financial_data
from buffetbot.data.fetcher import fetch_stock_data
from buffetbot.utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(ticker: str, years: int = 5) -> dict[str, Any] | None:
    """Fetch and process stock data with caching.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data

    Returns:
        Dictionary containing processed stock data, or None if failed
    """
    try:
        # Validate ticker input first
        if not ticker or not isinstance(ticker, str):
            logger.error(f"Invalid ticker provided: {ticker}")
            st.error(f"Invalid ticker: {ticker}")
            return None

        # Normalize ticker for consistent caching
        normalized_ticker = ticker.upper().strip()
        logger.info(f"Fetching data for ticker: {normalized_ticker}")

        # Fetch raw data
        raw_data = fetch_stock_data(normalized_ticker, years)

        # Clean and process data
        cleaned_data = clean_financial_data(
            {
                "income_stmt": raw_data["income_stmt"],
                "balance_sheet": raw_data["balance_sheet"],
                "cash_flow": raw_data["cash_flow"],
            }
        )

        # Add price data and fundamentals
        cleaned_data["price_data"] = raw_data["price_data"]
        cleaned_data["fundamentals"] = raw_data["fundamentals"]
        cleaned_data["metrics"] = raw_data["metrics"]

        # Log successful data fetch for debugging
        logger.info(f"Successfully fetched and processed data for {normalized_ticker}")
        return cleaned_data

    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}", exc_info=True)
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def handle_ticker_change(current_ticker: str) -> bool:
    """Handle ticker changes and cache management.

    Args:
        current_ticker: The current ticker symbol

    Returns:
        True if ticker changed, False otherwise
    """
    # Initialize session state for tracking ticker changes
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None

    # Check if ticker has changed and clear cache if needed
    if st.session_state.last_ticker != current_ticker:
        logger.info(
            f"Ticker changed from {st.session_state.last_ticker} to {current_ticker}"
        )
        # Clear cache for the old ticker to ensure fresh data fetch
        get_stock_info.clear()
        # Show a brief message about data refresh (but not on first load)
        if st.session_state.last_ticker is not None:
            st.sidebar.success(f"Loading data for {current_ticker}...")
        # Update the last ticker
        st.session_state.last_ticker = current_ticker
        return True

    return False


def clear_cache() -> None:
    """Clear all cached data."""
    get_stock_info.clear()
    st.success("Cache cleared! Data will be refreshed.")
    st.rerun()
