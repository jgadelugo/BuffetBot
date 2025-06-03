"""Pytest configuration and common fixtures for the dashboard tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_stock_data():
    """Fixture providing sample stock data for testing."""
    return {
        "fundamentals": {
            "market_cap": 2.5e12,
            "pe_ratio": 28.5,
            "beta": 1.15,
            "dividend_yield": 0.021,
            "eps": 6.45,
        },
        "metrics": {
            "price_change": 0.032,
            "volatility": 0.28,
            "rsi": 58.3,
            "momentum": 0.67,
        },
        "price_data": pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=50, freq="D"),
                "Open": np.random.uniform(150, 160, 50),
                "High": np.random.uniform(155, 165, 50),
                "Low": np.random.uniform(145, 155, 50),
                "Close": np.random.uniform(150, 160, 50),
                "Volume": np.random.randint(1000000, 5000000, 50),
            }
        ),
        "income_stmt": pd.DataFrame(
            {
                "Date": ["2023-12-31", "2022-12-31", "2021-12-31"],
                "Revenue": [394328000000, 365817000000, 347155000000],
                "Net_Income": [96995000000, 99803000000, 94680000000],
                "Operating_Income": [114301000000, 119437000000, 108949000000],
            }
        ),
        "balance_sheet": pd.DataFrame(
            {
                "Date": ["2023-12-31", "2022-12-31", "2021-12-31"],
                "Total_Assets": [352755000000, 352583000000, 381191000000],
                "Total_Debt": [123930000000, 120069000000, 124719000000],
                "Shareholders_Equity": [74100000000, 50672000000, 63090000000],
            }
        ),
        "cash_flow": pd.DataFrame(
            {
                "Date": ["2023-12-31", "2022-12-31", "2021-12-31"],
                "Operating_Cash_Flow": [110543000000, 122151000000, 104038000000],
                "Free_Cash_Flow": [84726000000, 91546000000, 73365000000],
                "Capital_Expenditures": [10708000000, 10708000000, 10708000000],
            }
        ),
    }


@pytest.fixture
def sample_options_data():
    """Fixture providing sample options data for testing."""
    return pd.DataFrame(
        {
            "strike": [150, 155, 160, 165, 170],
            "expiry": [
                "2024-12-20",
                "2024-12-20",
                "2024-12-20",
                "2024-12-20",
                "2024-12-20",
            ],
            "lastPrice": [8.5, 6.2, 4.1, 2.8, 1.9],
            "bid": [8.3, 6.0, 3.9, 2.6, 1.7],
            "ask": [8.7, 6.4, 4.3, 3.0, 2.1],
            "impliedVolatility": [0.28, 0.32, 0.35, 0.38, 0.42],
            "volume": [150, 89, 234, 67, 23],
            "openInterest": [1200, 890, 1500, 450, 120],
            "RSI": [45, 52, 38, 67, 73],
            "Beta": [1.2, 1.2, 1.2, 1.2, 1.2],
            "Momentum": [0.65, 0.72, 0.58, 0.81, 0.69],
            "IV": [0.28, 0.32, 0.35, 0.38, 0.42],
            "ForecastConfidence": [0.75, 0.68, 0.82, 0.55, 0.48],
            "CompositeScore": [0.82, 0.76, 0.71, 0.63, 0.48],
            "score_details": [
                {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2},
                {"rsi": 0.2, "beta": 0.2, "momentum": 0.2, "iv": 0.2, "forecast": 0.2},
                {"rsi": 0.25, "beta": 0.25, "momentum": 0.25, "iv": 0.25},
                {"rsi": 0.33, "beta": 0.33, "momentum": 0.34},
                {"rsi": 0.5, "beta": 0.5},
            ],
        }
    )


@pytest.fixture
def sample_risk_analysis_result():
    """Fixture providing sample risk analysis result for testing."""
    return {
        "overall_risk": {
            "score": 72.3,
            "level": "Moderate",
            "factors": [
                "Market volatility exposure at 28%",
                "Beta above market average at 1.15",
                "Sector concentration in technology",
                "Currency exposure risk",
                "Regulatory changes impact",
            ],
            "warnings": [
                "Limited historical data for some metrics",
                "High correlation with market indices",
            ],
            "errors": [],
        },
        "market_risk": {
            "beta": 1.15,
            "volatility": 0.28,
            "correlation_spy": 0.85,
            "var_95": 0.045,
        },
        "financial_risk": {
            "debt_to_equity": 1.67,
            "interest_coverage": 8.9,
            "current_ratio": 1.23,
            "quick_ratio": 0.98,
        },
        "business_risk": {
            "operating_margin": 0.29,
            "revenue": 394328000000,
            "revenue_growth": 0.078,
            "margin_stability": 0.15,
        },
    }


@pytest.fixture
def mock_streamlit():
    """Fixture providing mocked streamlit components."""
    # Create mock objects
    mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()

    # Create a context manager that patches all streamlit functions
    patches = [
        patch("streamlit.set_page_config"),
        patch("streamlit.title"),
        patch("streamlit.header"),
        patch("streamlit.subheader"),
        patch("streamlit.markdown"),
        patch("streamlit.columns", return_value=[mock_col1, mock_col2, mock_col3]),
        patch("streamlit.metric"),
        patch("streamlit.dataframe"),
        patch("streamlit.info"),
        patch("streamlit.warning"),
        patch("streamlit.error"),
        patch("streamlit.success"),
        patch("streamlit.spinner"),
        patch("streamlit.progress"),
        patch("streamlit.empty"),
        patch("streamlit.button"),
        patch("streamlit.text_input"),
        patch("streamlit.slider"),
        patch("streamlit.selectbox"),
        patch("streamlit.checkbox"),
        patch("streamlit.radio"),
        patch("streamlit.tabs"),
        patch("streamlit.sidebar.header"),
        patch("streamlit.sidebar.text_input"),
        patch("streamlit.sidebar.slider"),
        patch("streamlit.sidebar.button"),
        patch("streamlit.sidebar.checkbox"),
        patch("streamlit.sidebar.markdown"),
        patch("streamlit.expander"),
        patch("streamlit.container"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.download_button"),
    ]

    # Start all patches
    started_patches = [p.start() for p in patches]

    try:
        yield {
            "columns": started_patches[5],  # The columns patch
            "col1": mock_col1,
            "col2": mock_col2,
            "col3": mock_col3,
        }
    finally:
        # Stop all patches
        for p in patches:
            p.stop()


@pytest.fixture
def mock_session_state():
    """Fixture providing mocked streamlit session state."""
    mock_state = {
        "last_ticker": None,
        "show_metric_definitions": True,
        "show_data_report": False,
        "glossary_category": "All",
    }

    with patch("streamlit.session_state", mock_state):
        yield mock_state


@pytest.fixture(autouse=True)
def setup_logging():
    """Fixture to set up logging for tests."""
    import logging

    # Set up basic logging configuration for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during tests
        format="%(name)s - %(levelname)s - %(message)s",
    )

    # Suppress specific loggers that might be noisy during tests
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("plotly").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


@pytest.fixture
def mock_config():
    """Fixture providing mock dashboard configuration."""
    return {
        "cache_ttl": 3600,
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


@pytest.fixture
def sample_glossary_data():
    """Fixture providing sample glossary data for testing."""
    return {
        "market_cap": {
            "name": "Market Capitalization",
            "category": "value",
            "description": "The total value of a company's shares in the stock market",
            "formula": "Market Cap = Share Price × Number of Outstanding Shares",
        },
        "pe_ratio": {
            "name": "Price-to-Earnings Ratio",
            "category": "value",
            "description": "A valuation ratio comparing a company's current share price to its per-share earnings",
            "formula": "P/E Ratio = Market Value per Share ÷ Earnings per Share",
        },
        "revenue_growth": {
            "name": "Revenue Growth",
            "category": "growth",
            "description": "The rate at which a company's revenue increases over time",
            "formula": "Revenue Growth = (Current Revenue - Previous Revenue) ÷ Previous Revenue × 100%",
        },
        "debt_to_equity": {
            "name": "Debt-to-Equity Ratio",
            "category": "health",
            "description": "A financial ratio indicating the relative proportion of shareholders' equity and debt",
            "formula": "D/E Ratio = Total Debt ÷ Total Shareholders' Equity",
        },
        "beta": {
            "name": "Beta",
            "category": "risk",
            "description": "A measure of a stock's volatility relative to the overall market",
            "formula": "Beta = Covariance(Stock Returns, Market Returns) ÷ Variance(Market Returns)",
        },
    }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


# Test utilities
class TestUtilities:
    """Utility functions for tests."""

    @staticmethod
    def create_mock_dataframe(rows=10, columns=None):
        """Create a mock DataFrame for testing."""
        if columns is None:
            columns = ["Date", "Value1", "Value2", "Value3"]

        data = {}
        for col in columns:
            if col == "Date":
                data[col] = pd.date_range(start="2024-01-01", periods=rows, freq="D")
            else:
                data[col] = np.random.uniform(0, 100, rows)

        return pd.DataFrame(data)

    @staticmethod
    def assert_no_exceptions(func, *args, **kwargs):
        """Assert that a function call doesn't raise any exceptions."""
        try:
            func(*args, **kwargs)
        except Exception as e:
            pytest.fail(f"Function {func.__name__} raised an exception: {str(e)}")


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtilities()
