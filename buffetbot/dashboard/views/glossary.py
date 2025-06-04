"""
Glossary view for the BuffetBot Dashboard.

This module provides a comprehensive glossary of financial metrics and formulas
used throughout the application.
"""

import json

import pandas as pd
import streamlit as st

from buffetbot.dashboard.components.glossary_utils import render_metric_card
from buffetbot.glossary import GLOSSARY, get_metrics_by_category, search_metrics


def render_glossary_tab() -> None:
    """Render the glossary tab content."""
    # Glossary header
    st.header("📚 Financial Metrics Glossary")
    st.markdown("Comprehensive guide to financial metrics used in this analysis")

    # Create two columns for layout
    col1, col2 = st.columns([1, 3])

    with col1:
        # Search and filter controls
        st.subheader("🔍 Search & Filter")

        # Search box
        search_term = st.text_input(
            "Search metrics", placeholder="Enter term...", key="glossary_search"
        )

        # Category filter
        st.subheader("Categories")
        categories = ["All", "Growth", "Value", "Health", "Risk"]

        # Use session state for selected category
        if "glossary_category" not in st.session_state:
            st.session_state.glossary_category = "All"

        selected_category = st.radio(
            "Filter by category",
            categories,
            index=categories.index(st.session_state.glossary_category),
            key="glossary_category_radio",
        )

        # Quick stats
        st.subheader("📊 Statistics")
        total_metrics = len(GLOSSARY)
        st.metric("Total Metrics", total_metrics)

        # Category counts
        for cat in ["growth", "value", "health", "risk"]:
            count = len(get_metrics_by_category(cat))
            st.caption(f"{cat.title()}: {count}")

    with col2:
        # Apply filters
        if search_term:
            filtered_metrics = search_metrics(search_term)
            st.caption(
                f"Found {len(filtered_metrics)} metrics matching '{search_term}'"
            )
        else:
            if selected_category == "All":
                filtered_metrics = GLOSSARY
            else:
                filtered_metrics = get_metrics_by_category(selected_category.lower())

        # Display metrics
        if filtered_metrics:
            # Group by category if showing all
            if not search_term and selected_category == "All":
                for category in ["growth", "value", "health", "risk"]:
                    category_metrics = {
                        k: v
                        for k, v in filtered_metrics.items()
                        if v["category"] == category
                    }

                    if category_metrics:
                        # Category header
                        emoji_map = {
                            "growth": "📈",
                            "value": "💰",
                            "health": "💪",
                            "risk": "⚠️",
                        }

                        with st.expander(
                            f"{emoji_map.get(category, '📊')} {category.upper()} METRICS ({len(category_metrics)} items)",
                            expanded=True,
                        ):
                            for key, metric in category_metrics.items():
                                render_metric_card(key, metric)
            else:
                # Display filtered results without grouping
                for key, metric in filtered_metrics.items():
                    render_metric_card(key, metric)
        else:
            st.info("No metrics found matching your criteria.")

        # Export options
        st.markdown("---")
        st.subheader("📥 Export Options")

        # Prepare data for export
        export_data = []
        for key, metric in GLOSSARY.items():
            export_data.append(
                {
                    "Key": key,
                    "Name": metric["name"],
                    "Category": metric["category"],
                    "Description": metric["description"],
                    "Formula": metric["formula"],
                }
            )

        df = pd.DataFrame(export_data)

        col1_export, col2_export = st.columns(2)

        with col1_export:
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name="financial_metrics_glossary.csv",
                mime="text/csv",
            )

        with col2_export:
            # JSON download
            json_str = json.dumps(GLOSSARY, indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_str,
                file_name="financial_metrics_glossary.json",
                mime="application/json",
            )
