"""
Sample market data for BuffetBot development.
"""

import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.market_data import MarketDataCache


async def create_sample_market_data(session: AsyncSession) -> None:
    """Create sample market data cache entries for development and testing."""

    # Sample market data for different tickers
    market_data_samples = [
        {
            "ticker": "AAPL",
            "data_type": "daily_prices",
            "data": {
                "current_price": 180.25,
                "open": 179.50,
                "high": 182.00,
                "low": 178.75,
                "volume": 58945123,
                "market_cap": 2832000000000,
                "pe_ratio": 29.4,
                "eps": 6.13,
                "dividend_yield": 0.47,
                "52_week_high": 199.62,
                "52_week_low": 164.08,
            },
        },
        {
            "ticker": "AAPL",
            "data_type": "financial_metrics",
            "data": {
                "revenue": 394328000000,
                "net_income": 99803000000,
                "total_debt": 123930000000,
                "total_cash": 29965000000,
                "book_value_per_share": 4.40,
                "return_on_equity": 1.569,
                "debt_to_equity": 1.73,
                "current_ratio": 0.94,
                "quick_ratio": 0.81,
            },
        },
        {
            "ticker": "MSFT",
            "data_type": "daily_prices",
            "data": {
                "current_price": 310.00,
                "open": 308.50,
                "high": 312.25,
                "low": 307.00,
                "volume": 28456789,
                "market_cap": 2304000000000,
                "pe_ratio": 32.1,
                "eps": 9.65,
                "dividend_yield": 0.73,
                "52_week_high": 384.30,
                "52_week_low": 213.43,
            },
        },
        {
            "ticker": "MSFT",
            "data_type": "financial_metrics",
            "data": {
                "revenue": 211915000000,
                "net_income": 72361000000,
                "total_debt": 58067000000,
                "total_cash": 104584000000,
                "book_value_per_share": 13.05,
                "return_on_equity": 0.428,
                "debt_to_equity": 0.35,
                "current_ratio": 1.76,
                "quick_ratio": 1.70,
            },
        },
        {
            "ticker": "GOOGL",
            "data_type": "daily_prices",
            "data": {
                "current_price": 140.50,
                "open": 139.25,
                "high": 142.00,
                "low": 138.75,
                "volume": 23567890,
                "market_cap": 1765000000000,
                "pe_ratio": 25.8,
                "eps": 5.44,
                "dividend_yield": 0.0,
                "52_week_high": 151.55,
                "52_week_low": 83.34,
            },
        },
        {
            "ticker": "TSLA",
            "data_type": "daily_prices",
            "data": {
                "current_price": 220.00,
                "open": 218.50,
                "high": 225.00,
                "low": 216.25,
                "volume": 95678123,
                "market_cap": 698000000000,
                "pe_ratio": 59.7,
                "eps": 3.68,
                "dividend_yield": 0.0,
                "52_week_high": 299.29,
                "52_week_low": 101.81,
            },
        },
        {
            "ticker": "JNJ",
            "data_type": "daily_prices",
            "data": {
                "current_price": 165.50,
                "open": 164.25,
                "high": 166.75,
                "low": 163.50,
                "volume": 12345678,
                "market_cap": 435000000000,
                "pe_ratio": 15.2,
                "eps": 10.89,
                "dividend_yield": 2.95,
                "52_week_high": 177.81,
                "52_week_low": 143.13,
            },
        },
        {
            "ticker": "JNJ",
            "data_type": "financial_metrics",
            "data": {
                "revenue": 94943000000,
                "net_income": 17941000000,
                "total_debt": 31665000000,
                "total_cash": 29556000000,
                "book_value_per_share": 25.89,
                "return_on_equity": 0.256,
                "debt_to_equity": 0.45,
                "current_ratio": 1.17,
                "quick_ratio": 0.86,
            },
        },
        {
            "ticker": "NVDA",
            "data_type": "daily_prices",
            "data": {
                "current_price": 450.00,
                "open": 445.25,
                "high": 458.50,
                "low": 442.00,
                "volume": 45123789,
                "market_cap": 1108000000000,
                "pe_ratio": 73.2,
                "eps": 6.15,
                "dividend_yield": 0.09,
                "52_week_high": 502.66,
                "52_week_low": 108.13,
            },
        },
        {
            "ticker": "VTI",
            "data_type": "daily_prices",
            "data": {
                "current_price": 225.00,
                "open": 224.50,
                "high": 226.00,
                "low": 223.75,
                "volume": 3456789,
                "market_cap": None,  # ETF
                "pe_ratio": None,
                "eps": None,
                "dividend_yield": 1.38,
                "52_week_high": 232.54,
                "52_week_low": 181.18,
            },
        },
    ]

    # Create market data cache entries
    for data_sample in market_data_samples:
        cache_entry = MarketDataCache(
            id=uuid.uuid4(),
            ticker=data_sample["ticker"],
            data_type=data_sample["data_type"],
            data=data_sample["data"],
            cached_at=datetime.utcnow() - timedelta(minutes=15),
            expires_at=datetime.utcnow()
            + timedelta(
                hours=23, minutes=45
            ),  # 24 hour cache with 15 min already passed
        )
        session.add(cache_entry)

    # Add some expired entries to test cleanup
    expired_entries = [
        MarketDataCache(
            id=uuid.uuid4(),
            ticker="AMD",
            data_type="daily_prices",
            data={"current_price": 95.00, "volume": 50000000},
            cached_at=datetime.utcnow() - timedelta(hours=25),
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired 1 hour ago
        ),
        MarketDataCache(
            id=uuid.uuid4(),
            ticker="AMD",
            data_type="financial_metrics",
            data={"revenue": 23601000000, "net_income": 1320000000},
            cached_at=datetime.utcnow() - timedelta(days=2),
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired 1 day ago
        ),
    ]

    for entry in expired_entries:
        session.add(entry)
