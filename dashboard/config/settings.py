"""Dashboard configuration and setup utilities."""

import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from utils.logger import get_logger, setup_logging


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
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


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
