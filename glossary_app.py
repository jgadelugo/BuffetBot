"""Financial Metrics Glossary - Interactive UI

A Streamlit application that provides an interactive interface for browsing
and searching financial metrics documentation.
"""

import streamlit as st
from typing import Dict, List, Optional
import pandas as pd
from glossary_data import (
    GLOSSARY,
    get_metrics_by_category,
    search_metrics,
    MetricDefinition
)

# Page configuration
st.set_page_config(
    page_title="Financial Metrics Glossary",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-name {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .metric-category {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .category-growth { background-color: #d4edda; color: #155724; }
    .category-value { background-color: #cce5ff; color: #004085; }
    .category-health { background-color: #fff3cd; color: #856404; }
    .category-risk { background-color: #f8d7da; color: #721c24; }
    .metric-description {
        color: #495057;
        line-height: 1.6;
        margin: 10px 0;
    }
    .metric-formula {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
        color: #212529;
    }
    .search-result-count {
        color: #6c757d;
        font-style: italic;
        margin: 10px 0;
    }
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(key: str, metric: MetricDefinition):
    """Render a single metric as a styled card."""
    category_class = f"category-{metric['category']}"
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-name">{metric['name']}</div>
        <span class="metric-category {category_class}">{metric['category'].upper()}</span>
        <div class="metric-description">{metric['description']}</div>
        <div style="margin-top: 15px;">
            <strong>Formula:</strong>
            <div class="metric-formula">{metric['formula']}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    # Header section
    st.markdown("""
    <div class="header-section">
        <h1 style="margin: 0; font-size: 2.5em;">📊 Financial Metrics Glossary</h1>
        <p style="margin-top: 10px; font-size: 1.1em;">
            Comprehensive guide to financial metrics and KPIs used in value investing analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for filtering
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Search functionality
        search_term = st.text_input(
            "Search metrics",
            placeholder="Enter term to search...",
            help="Search by metric name or description"
        )
        
        # Category filter
        st.subheader("Categories")
        categories = ["All", "Growth", "Value", "Health", "Risk"]
        selected_category = st.radio(
            "Select category",
            categories,
            index=0,
            help="Filter metrics by category"
        )
        
        # View options
        st.subheader("View Options")
        show_expanded = st.checkbox("Expand all sections", value=True)
        show_count = st.checkbox("Show metric count", value=True)
        
        # Stats section
        st.markdown("---")
        st.subheader("📈 Statistics")
        total_metrics = len(GLOSSARY)
        st.metric("Total Metrics", total_metrics)
        
        # Category breakdown
        for cat in ["growth", "value", "health", "risk"]:
            count = len(get_metrics_by_category(cat))
            st.metric(f"{cat.title()} Metrics", count)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Apply filters
        if search_term:
            filtered_metrics = search_metrics(search_term)
            if show_count:
                st.markdown(
                    f'<p class="search-result-count">Found {len(filtered_metrics)} metrics matching "{search_term}"</p>',
                    unsafe_allow_html=True
                )
        else:
            if selected_category == "All":
                filtered_metrics = GLOSSARY
            else:
                filtered_metrics = get_metrics_by_category(selected_category.lower())
        
        # Display metrics grouped by category
        if not search_term and selected_category == "All":
            # Group by category when showing all
            for category in ["growth", "value", "health", "risk"]:
                category_metrics = {k: v for k, v in filtered_metrics.items() 
                                  if v["category"] == category}
                
                if category_metrics:
                    # Category header with emoji
                    emoji_map = {
                        "growth": "📈",
                        "value": "💰",
                        "health": "💪",
                        "risk": "⚠️"
                    }
                    
                    with st.expander(
                        f"{emoji_map.get(category, '📊')} {category.upper()} METRICS ({len(category_metrics)} items)",
                        expanded=show_expanded
                    ):
                        for key, metric in category_metrics.items():
                            render_metric_card(key, metric)
        else:
            # Display filtered results without grouping
            if filtered_metrics:
                for key, metric in filtered_metrics.items():
                    render_metric_card(key, metric)
            else:
                st.info("No metrics found matching your criteria.")
    
    with col2:
        # Quick navigation or additional info
        st.subheader("📌 Quick Links")
        
        # Category navigation buttons
        for category in ["Growth", "Value", "Health", "Risk"]:
            if st.button(f"Go to {category}", key=f"nav_{category}", use_container_width=True):
                st.session_state.selected_category = category
                st.experimental_rerun()
        
        st.markdown("---")
        
        # Export functionality
        st.subheader("📥 Export")
        
        # Create DataFrame for export
        export_data = []
        for key, metric in GLOSSARY.items():
            export_data.append({
                "Key": key,
                "Name": metric["name"],
                "Category": metric["category"],
                "Description": metric["description"],
                "Formula": metric["formula"]
            })
        
        df = pd.DataFrame(export_data)
        
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="financial_metrics_glossary.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON download
        import json
        json_str = json.dumps(GLOSSARY, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name="financial_metrics_glossary.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 20px;">
        <p>Financial Metrics Glossary v1.0 | Built with Streamlit</p>
        <p>💡 Tip: Use the search bar to find specific metrics or filter by category</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 