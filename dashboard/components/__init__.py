"""Dashboard components module."""

# Re-export commonly used components
try:
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
        # Metrics components
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
        # Other components
        "render_sidebar",
        "render_investment_disclaimer",
        "render_compliance_footer",
        "render_metric_card",
        "render_score_details_popover",
        "get_data_score_badge",
        "check_for_partial_data",
        "create_styling_functions",
    ]

except ImportError as e:
    # If some components are missing, provide graceful fallback
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []
