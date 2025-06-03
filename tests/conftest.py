"""Pytest configuration for BuffetBot tests.

This file contains fixtures and configuration that will be automatically
available to all test files in the tests directory.
"""

import os
import sys

import pytest

# Add the parent directory to the Python path so we can import BuffetBot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_metrics():
    """Fixture providing sample calculated metrics for testing."""
    return {
        "current_ratio": 1.5,
        "debt_to_equity": 0.8,
        "pe_ratio": 15.2,
        "revenue_growth": 0.12,
        "return_on_equity": 0.18,
        "gross_margin": 0.35,
    }


@pytest.fixture
def sample_analysis_results():
    """Fixture providing sample analysis results for testing."""
    return {
        "growth_metrics": {
            "revenue_growth": 0.15,
            "earnings_growth": 0.18,
            "eps_growth": 0.12,
            "revenue_cagr": 0.14,
            "fcf_growth": 0.20,
        },
        "financial_ratios": {
            "current_ratio": 1.8,
            "debt_to_equity": 0.6,
            "return_on_equity": 0.22,
            "gross_margin": 0.35,
            "operating_margin": 0.18,
            "net_margin": 0.12,
        },
        "risk_metrics": {"beta": 1.2, "volatility": 0.25, "overall_risk_score": 45.5},
        "value_metrics": {"pe_ratio": 18.5, "pb_ratio": 3.2, "peg_ratio": 1.1},
    }
