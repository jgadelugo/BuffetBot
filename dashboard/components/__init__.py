"""Dashboard components module."""

# Re-export commonly used components
try:
    from .analytics import (
        initialize_analytics,
        inject_google_analytics,
        track_custom_event,
        track_page_view,
        track_ticker_analysis,
        track_user_interaction,
    )
    from .charts import (
        create_enhanced_price_gauge,
        create_growth_chart,
        create_price_gauge,
        create_technical_analysis_chart,
        create_valuation_metrics_chart,
    )
    from .disclaimers import render_compliance_footer, render_investment_disclaimer
    from .glossary_utils import render_metric_card
    from .metrics import (
        create_comparison_table,
        create_progress_indicator,
        display_metric_with_info,
        display_metric_with_status,
        display_metrics_grid,
        display_metrics_grid_enhanced,
    )
    from .options_utils import (
        check_for_partial_data,
        create_styling_functions,
        get_data_score_badge,
        render_score_details_popover,
    )
    from .price_valuation import PriceValuationCard, create_valuation_summary
    from .sidebar import render_sidebar

    __all__ = [
        # Analytics components
        "initialize_analytics",
        "inject_google_analytics",
        "track_page_view",
        "track_custom_event",
        "track_ticker_analysis",
        "track_user_interaction",
        # Display components
        "display_metric_with_info",
        "display_metrics_grid",
        "display_metric_with_status",
        "display_metrics_grid_enhanced",
        "create_comparison_table",
        "create_progress_indicator",
        # Chart components
        "create_price_gauge",
        "create_growth_chart",
        "create_enhanced_price_gauge",
        "create_technical_analysis_chart",
        "create_valuation_metrics_chart",
        # Price valuation components
        "PriceValuationCard",
        "create_valuation_summary",
        # Sidebar and layout
        "render_sidebar",
        # Disclaimers and compliance
        "render_investment_disclaimer",
        "render_compliance_footer",
        # Utility components
        "render_metric_card",
        "render_score_details_popover",
        "get_data_score_badge",
        "check_for_partial_data",
        "create_styling_functions",
    ]

except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []
