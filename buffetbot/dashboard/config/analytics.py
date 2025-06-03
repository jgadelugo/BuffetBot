"""
Analytics configuration for the dashboard.

This module contains configuration settings for Google Analytics and other
tracking integrations.
"""

import os
from typing import Any, Dict

# Google Analytics Configuration
GOOGLE_ANALYTICS_CONFIG = {
    "production": {
        "tracking_id": "G-ZCCK6W5VEF",
        "enabled": True,
        "debug_mode": False,
        "anonymize_ip": True,
        "cookie_consent": True,
    },
    "staging": {
        "tracking_id": "G-ZCCK6W5VEF",  # Consider using a separate staging ID
        "enabled": True,
        "debug_mode": True,
        "anonymize_ip": True,
        "cookie_consent": True,
    },
    "development": {
        "tracking_id": "G-ZCCK6W5VEF",
        "enabled": False,  # Disabled in development to avoid skewing data
        "debug_mode": True,
        "anonymize_ip": True,
        "cookie_consent": False,
    },
}


def get_environment() -> str:
    """Determine the current environment."""
    # Check environment variables first
    env = os.getenv("STREAMLIT_ENV", "").lower()
    if env in ["production", "staging", "development"]:
        return env

    # Check if running in Streamlit Cloud (production)
    if os.getenv("STREAMLIT_SHARING"):
        return "production"

    # Check for common development indicators
    if os.getenv("DEBUG") or os.getenv("DEVELOPMENT"):
        return "development"

    # Default to production for safety
    return "production"


def get_analytics_config(environment: str = None) -> dict[str, Any]:
    """Get analytics configuration for the specified environment.

    Args:
        environment: Environment name. If None, auto-detect.

    Returns:
        Analytics configuration dictionary
    """
    if environment is None:
        environment = get_environment()

    return GOOGLE_ANALYTICS_CONFIG.get(
        environment, GOOGLE_ANALYTICS_CONFIG["production"]
    )


def should_track_analytics(environment: str = None) -> bool:
    """Determine if analytics should be tracked in the current environment.

    Args:
        environment: Environment name. If None, auto-detect.

    Returns:
        True if analytics should be tracked
    """
    config = get_analytics_config(environment)
    return config.get("enabled", False)


def get_tracking_id(environment: str = None) -> str:
    """Get the Google Analytics tracking ID for the environment.

    Args:
        environment: Environment name. If None, auto-detect.

    Returns:
        Google Analytics tracking ID
    """
    config = get_analytics_config(environment)
    return config.get("tracking_id", "G-ZCCK6W5VEF")


# Custom event tracking configuration
CUSTOM_EVENTS = {
    "ticker_analysis": {
        "category": "Stock Analysis",
        "action": "Ticker Analyzed",
        "label": "ticker_symbol",
    },
    "tab_view": {"category": "Navigation", "action": "Tab Viewed", "label": "tab_name"},
    "data_export": {
        "category": "Data",
        "action": "Data Exported",
        "label": "export_format",
    },
    "error_occurred": {
        "category": "Errors",
        "action": "Error Occurred",
        "label": "error_type",
    },
}


def get_event_config(event_name: str) -> dict[str, str]:
    """Get configuration for a custom event.

    Args:
        event_name: Name of the custom event

    Returns:
        Event configuration dictionary
    """
    return CUSTOM_EVENTS.get(
        event_name,
        {"category": "Custom", "action": event_name, "label": "custom_event"},
    )
