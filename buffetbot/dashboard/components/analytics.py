"""
Analytics integration for the dashboard.

This module handles Google Analytics and other tracking integrations
for the Stock Analysis Dashboard.
"""

from typing import Optional

import streamlit as st


def inject_google_analytics(tracking_id: str = "G-YEGLMK3LDR") -> None:
    """Inject Google Analytics tracking code into the Streamlit app.

    Args:
        tracking_id: Google Analytics tracking ID
    """

    # Google Analytics tracking code
    ga_code = f"""

        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={tracking_id}"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        gtag('config', {tracking_id});
        </script>

    """

    # Inject the code into the page head
    st.html(ga_code)


def track_page_view(page_name: str, ticker: str | None = None) -> None:
    """Track a page view event in Google Analytics.

    Args:
        page_name: Name of the page/view being tracked
        ticker: Optional ticker symbol for enhanced tracking
    """

    # Enhanced tracking with custom events
    if ticker:
        tracking_code = f"""
        <script>
        if (typeof gtag !== 'undefined') {{
            gtag('event', 'page_view', {{
                'page_title': '{page_name}',
                'page_location': window.location.href,
                'custom_parameter_ticker': '{ticker}'
            }});
        }}
        </script>
        """
    else:
        tracking_code = f"""
        <script>
        if (typeof gtag !== 'undefined') {{
            gtag('event', 'page_view', {{
                'page_title': '{page_name}',
                'page_location': window.location.href
            }});
        }}
        </script>
        """

    st.html(tracking_code)


def track_custom_event(event_name: str, parameters: dict = None) -> None:
    """Track a custom event in Google Analytics.

    Args:
        event_name: Name of the custom event
        parameters: Additional parameters for the event
    """

    params = parameters or {}
    params_js = ", ".join([f"'{k}': '{v}'" for k, v in params.items()])

    tracking_code = f"""
    <script>
    if (typeof gtag !== 'undefined') {{
        gtag('event', '{event_name}', {{
            {params_js}
        }});
    }}
    </script>
    """

    st.html(tracking_code)


def track_ticker_analysis(ticker: str, analysis_type: str) -> None:
    """Track when a user analyzes a specific ticker.

    Args:
        ticker: Stock ticker symbol
        analysis_type: Type of analysis performed (overview, risk, etc.)
    """

    track_custom_event(
        "ticker_analysis",
        {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "timestamp": "new Date().toISOString()",
        },
    )


def track_user_interaction(interaction_type: str, details: dict = None) -> None:
    """Track user interactions for UX analytics.

    Args:
        interaction_type: Type of interaction (button_click, tab_switch, etc.)
        details: Additional details about the interaction
    """

    params = details or {}
    params["interaction_type"] = interaction_type

    track_custom_event("user_interaction", params)


# Configuration for different environments
ANALYTICS_CONFIG = {
    "production": {"enabled": True, "tracking_id": "G-YEGLMK3LDR"},
    "development": {
        "enabled": False,  # Disable in development
        "tracking_id": "G-YEGLMK3LDR",
    },
    "staging": {
        "enabled": True,
        "tracking_id": "G-YEGLMK3LDR",  # You might want a separate ID for staging
    },
}


def get_analytics_config(environment: str = "production") -> dict:
    """Get analytics configuration for the specified environment.

    Args:
        environment: Environment name (production, development, staging)

    Returns:
        Analytics configuration dictionary
    """
    return ANALYTICS_CONFIG.get(environment, ANALYTICS_CONFIG["production"])


def initialize_analytics(environment: str = "production") -> None:
    """Initialize Google Analytics for the dashboard.

    Args:
        environment: Environment name to determine configuration
    """

    config = get_analytics_config(environment)

    if config["enabled"]:
        inject_google_analytics(config["tracking_id"])

        # Track initial page load
        track_page_view("Dashboard Load")

    # Store analytics state in session for tracking across the app
    st.session_state.analytics_enabled = config["enabled"]
    st.session_state.analytics_tracking_id = config["tracking_id"]
