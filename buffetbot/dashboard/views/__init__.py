"""
Dashboard Views Module

This module contains all view rendering functions for the financial analysis dashboard.
Each view is responsible for rendering a specific section or page of the application.

Views are organized by functionality:
- Core Analysis: overview, growth_metrics, risk_analysis
- Advanced Tools: options_advisor, analyst_forecast, price_analysis, financial_health
- Reference: glossary

All views follow the same interface pattern:
- render_*_view(data, ticker, **kwargs) -> None

This module also provides a modern view registry system for future extensibility.
"""

from typing import Any, Dict, Optional

# Advanced analysis views
from .analyst_forecast import render_analyst_forecast_tab

# Import base classes and registry
from .base import BaseView, ViewCategory, ViewMetadata, ViewRegistry, view_registry
from .financial_health import render_financial_health_page

# Reference views
from .glossary import render_glossary_tab
from .growth_metrics import render_growth_metrics_tab
from .options_advisor import render_options_advisor_tab

# Core analysis views
from .overview import render_overview_tab
from .price_analysis import render_price_analysis_page
from .risk_analysis import render_risk_analysis_tab

# Export all view functions and base classes
__all__ = [
    # Base classes for future development
    "BaseView",
    "ViewRegistry",
    "ViewMetadata",
    "ViewCategory",
    "view_registry",
    # Core analysis views
    "render_overview_tab",
    "render_growth_metrics_tab",
    "render_risk_analysis_tab",
    # Advanced analysis views
    "render_options_advisor_tab",
    "render_analyst_forecast_tab",
    "render_price_analysis_page",
    "render_financial_health_page",
    # Reference views
    "render_glossary_tab",
    # Helper functions
    "get_all_views",
    "get_view_function",
    "register_legacy_views",
]


def get_all_views() -> dict[str, dict[str, Any]]:
    """Get metadata about all available views.

    Returns:
        Dict containing view metadata including categories, descriptions, and functions
    """
    return {
        "core_analysis": {
            "overview": {
                "function": render_overview_tab,
                "title": "Portfolio Overview",
                "description": "High-level financial metrics and company overview",
                "icon": "📊",
                "requires_data": True,
            },
            "growth_metrics": {
                "function": render_growth_metrics_tab,
                "title": "Growth Analysis",
                "description": "Revenue, earnings, and growth trajectory analysis",
                "icon": "📈",
                "requires_data": True,
            },
            "risk_analysis": {
                "function": render_risk_analysis_tab,
                "title": "Risk Assessment",
                "description": "Comprehensive risk analysis and scoring",
                "icon": "⚠️",
                "requires_data": True,
            },
        },
        "advanced_tools": {
            "options_advisor": {
                "function": render_options_advisor_tab,
                "title": "Options Advisor",
                "description": "Options analysis and recommendations with Greeks",
                "icon": "🎯",
                "requires_data": True,
            },
            "analyst_forecast": {
                "function": render_analyst_forecast_tab,
                "title": "Analyst Forecast",
                "description": "Comprehensive analyst forecast and consensus analysis",
                "icon": "🔮",
                "requires_data": True,
            },
            "price_analysis": {
                "function": render_price_analysis_page,
                "title": "Price Valuation",
                "description": "Intrinsic value and valuation analysis",
                "icon": "💰",
                "requires_data": True,
            },
            "financial_health": {
                "function": render_financial_health_page,
                "title": "Financial Health",
                "description": "Financial ratios and health scoring",
                "icon": "🏥",
                "requires_data": True,
            },
        },
        "reference": {
            "glossary": {
                "function": render_glossary_tab,
                "title": "Metrics Glossary",
                "description": "Definitions and explanations of financial metrics",
                "icon": "📚",
                "requires_data": False,
            }
        },
    }


def get_view_function(view_name: str) -> Optional[callable]:
    """Get a view function by name.

    Args:
        view_name: Name of the view (e.g., 'overview', 'risk_analysis')

    Returns:
        View function if found, None otherwise
    """
    view_mapping = {
        "overview": render_overview_tab,
        "growth_metrics": render_growth_metrics_tab,
        "risk_analysis": render_risk_analysis_tab,
        "options_advisor": render_options_advisor_tab,
        "analyst_forecast": render_analyst_forecast_tab,
        "price_analysis": render_price_analysis_page,
        "financial_health": render_financial_health_page,
        "glossary": render_glossary_tab,
    }

    return view_mapping.get(view_name)


def register_legacy_views() -> None:
    """Register all legacy view functions in the view registry for future migration."""
    legacy_views = [
        (
            "overview",
            render_overview_tab,
            ViewMetadata(
                name="overview",
                title="Portfolio Overview",
                description="High-level financial metrics and company overview",
                icon="📊",
                category=ViewCategory.CORE_ANALYSIS,
                requires_data=True,
            ),
        ),
        (
            "growth_metrics",
            render_growth_metrics_tab,
            ViewMetadata(
                name="growth_metrics",
                title="Growth Analysis",
                description="Revenue, earnings, and growth trajectory analysis",
                icon="📈",
                category=ViewCategory.CORE_ANALYSIS,
                requires_data=True,
            ),
        ),
        (
            "risk_analysis",
            render_risk_analysis_tab,
            ViewMetadata(
                name="risk_analysis",
                title="Risk Assessment",
                description="Comprehensive risk analysis and scoring",
                icon="⚠️",
                category=ViewCategory.CORE_ANALYSIS,
                requires_data=True,
            ),
        ),
        (
            "options_advisor",
            render_options_advisor_tab,
            ViewMetadata(
                name="options_advisor",
                title="Options Advisor",
                description="Options analysis and recommendations with Greeks",
                icon="🎯",
                category=ViewCategory.ADVANCED_TOOLS,
                requires_data=True,
            ),
        ),
        (
            "analyst_forecast",
            render_analyst_forecast_tab,
            ViewMetadata(
                name="analyst_forecast",
                title="Analyst Forecast",
                description="Comprehensive analyst forecast and consensus analysis",
                icon="🔮",
                category=ViewCategory.ADVANCED_TOOLS,
                requires_data=True,
            ),
        ),
        (
            "price_analysis",
            render_price_analysis_page,
            ViewMetadata(
                name="price_analysis",
                title="Price Valuation",
                description="Intrinsic value and valuation analysis",
                icon="💰",
                category=ViewCategory.ADVANCED_TOOLS,
                requires_data=True,
            ),
        ),
        (
            "financial_health",
            render_financial_health_page,
            ViewMetadata(
                name="financial_health",
                title="Financial Health",
                description="Financial ratios and health scoring",
                icon="🏥",
                category=ViewCategory.ADVANCED_TOOLS,
                requires_data=True,
            ),
        ),
        (
            "glossary",
            render_glossary_tab,
            ViewMetadata(
                name="glossary",
                title="Metrics Glossary",
                description="Definitions and explanations of financial metrics",
                icon="📚",
                category=ViewCategory.REFERENCE,
                requires_data=False,
            ),
        ),
    ]

    for name, func, metadata in legacy_views:
        view_registry.register_legacy_function(name, func, metadata)


# Auto-register legacy views when module is imported
register_legacy_views()
