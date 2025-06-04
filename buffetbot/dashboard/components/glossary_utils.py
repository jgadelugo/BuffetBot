"""Glossary utility functions for the dashboard."""

from typing import Any, Dict

import streamlit as st

from buffetbot.glossary import MetricDefinition


def render_metric_card(key: str, metric: MetricDefinition) -> None:
    """Render a single metric as a styled card.

    Args:
        key: The metric key
        metric: The metric definition
    """
    category_class = f"category-{metric['category']}"

    card_html = f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="font-size: 1.2em; font-weight: bold; color: #1f77b4; margin-bottom: 10px;">{metric['name']}</div>
        <span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; margin-bottom: 10px; background-color: {'#d4edda' if metric['category'] == 'growth' else '#cce5ff' if metric['category'] == 'value' else '#fff3cd' if metric['category'] == 'health' else '#f8d7da'}; color: {'#155724' if metric['category'] == 'growth' else '#004085' if metric['category'] == 'value' else '#856404' if metric['category'] == 'health' else '#721c24'};">{metric['category'].upper()}</span>
        <div style="color: #495057; line-height: 1.6; margin: 10px 0;">{metric['description']}</div>
        <div style="margin-top: 15px;">
            <strong>Formula:</strong>
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; color: #212529;">{metric['formula']}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
