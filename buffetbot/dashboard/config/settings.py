"""Dashboard configuration and setup utilities."""

import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from buffetbot.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def setup_project_path() -> str:
    """Set up project path and add to sys.path if needed.

    Returns:
        str: The project root path
    """
    project_root = str(Path(__file__).parent.parent.parent.absolute())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def configure_streamlit_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def setup_logging_config() -> None:
    """Initialize logging configuration."""
    setup_logging()


def initialize_session_state() -> None:
    """Initialize session state variables with default values."""
    session_defaults = {
        "last_ticker": None,
        "show_metric_definitions": True,
        "show_data_report": False,
        "glossary_category": "All",
        # Options analysis settings
        "options_settings": {
            "strategy_type": "Long Calls",
            "risk_tolerance": "Conservative",
            "time_horizon": "Medium-term (3-6 months)",
            "min_days": 180,
            "top_n": 5,
            "include_greeks": True,
            "volatility_analysis": False,
            "download_csv": False,
            "use_custom_weights": False,
            "custom_scoring_weights": {
                "rsi": 0.20,
                "beta": 0.20,
                "momentum": 0.20,
                "iv": 0.20,
                "forecast": 0.20,
            },
        },
        # Settings state tracking
        "settings_changed": False,
        "last_analysis_settings": None,
        "analysis_cache": None,
        "analysis_timestamp": None,
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Log initialization
    logger.debug("Session state initialized with default values")


def get_dashboard_config() -> dict[str, Any]:
    """Get dashboard configuration settings.

    Returns:
        Dict containing configuration settings
    """
    return {
        "cache_ttl": 3600,  # 1 hour cache
        "default_ticker": "AAPL",
        "default_years": 5,
        "min_years": 1,
        "max_years": 10,
        "default_min_days": 180,
        "min_min_days": 90,
        "max_min_days": 720,
        "default_top_n": 5,
        "min_top_n": 1,
        "max_top_n": 20,
    }


def update_options_setting(key: str, value: Any) -> None:
    """Update a specific options analysis setting and mark settings as changed.

    Args:
        key: Setting key to update
        value: New value for the setting
    """
    if "options_settings" not in st.session_state:
        st.session_state.options_settings = {}

    old_value = st.session_state.options_settings.get(key)
    if old_value != value:
        st.session_state.options_settings[key] = value
        st.session_state.settings_changed = True
        logger.info(f"Options setting updated: {key} = {value} (was: {old_value})")


def get_options_setting(key: str, default: Any = None) -> Any:
    """Get a specific options analysis setting.

    Args:
        key: Setting key to retrieve
        default: Default value if key doesn't exist

    Returns:
        The setting value or default
    """
    return st.session_state.get("options_settings", {}).get(key, default)


def get_current_settings_hash() -> str:
    """Generate a hash of current settings for change detection.

    Returns:
        String hash of current settings
    """
    import hashlib
    import json

    settings = st.session_state.get("options_settings", {})
    ticker = st.session_state.get("last_ticker", "")

    # Create a stable hash from settings
    settings_str = json.dumps({**settings, "ticker": ticker}, sort_keys=True)

    return hashlib.md5(settings_str.encode()).hexdigest()


def settings_have_changed() -> bool:
    """Check if analysis settings have changed since last analysis.

    Returns:
        True if settings have changed, False otherwise
    """
    current_hash = get_current_settings_hash()
    last_hash = st.session_state.get("last_analysis_settings")

    return current_hash != last_hash


def mark_settings_applied() -> None:
    """Mark current settings as applied by updating the last analysis settings hash."""
    current_hash = get_current_settings_hash()
    st.session_state.last_analysis_settings = current_hash
    st.session_state.settings_changed = False
    logger.debug(f"Settings marked as applied with hash: {current_hash}")


def clear_analysis_cache() -> None:
    """Clear the cached analysis results to force recalculation."""
    st.session_state.analysis_cache = None
    st.session_state.analysis_timestamp = None
    st.session_state.settings_changed = True
    logger.debug("Analysis cache cleared")


def validate_scoring_weights(weights: dict[str, float]) -> tuple[bool, str]:
    """Validate scoring weights to ensure they sum to 1.0.

    Args:
        weights: Dictionary of scoring weights

    Returns:
        Tuple of (is_valid, error_message)
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        return False, f"Weights must sum to 1.0 (current sum: {total:.3f})"

    for key, value in weights.items():
        if not 0 <= value <= 1:
            return False, f"Weight for {key} must be between 0 and 1 (current: {value})"

    return True, ""
