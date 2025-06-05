"""
Market data models for BuffetBot database.

Models for caching market data, price history, options data, and other financial information.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..connection import Base


class DataType(str, Enum):
    """Types of market data that can be cached."""

    PRICE_HISTORY = "price_history"
    FUNDAMENTALS = "fundamentals"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    OPTIONS_CHAIN = "options_chain"
    ANALYST_ESTIMATES = "analyst_estimates"
    PEERS = "peers"
    NEWS = "news"


class MarketDataCache(Base):
    """
    Market data cache model for storing various types of financial data.

    Provides a unified cache for all types of market data with proper
    expiration handling and data type categorization.
    """

    __tablename__ = "market_data_cache"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Cache key components
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    data_type: Mapped[DataType] = mapped_column(String(50), nullable=False, index=True)

    # Cached data
    data: Mapped[dict] = mapped_column(
        JSONB, nullable=False, comment="Cached market data as JSON"
    )

    # Cache metadata
    data_source: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Source of the data (e.g., 'yfinance', 'alpha_vantage')",
    )

    data_quality_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(3, 2),  # Quality score from 0.00 to 1.00
        nullable=True,
        comment="Quality score of the cached data (0.0 to 1.0)",
    )

    # Cache timing
    cached_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When this cached data expires",
    )

    # Additional parameters for cache key
    parameters: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional parameters used for fetching this data",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint for cache key
        UniqueConstraint(
            "ticker", "data_type", "parameters", name="unique_cache_entry"
        ),
        # Check constraints
        CheckConstraint(
            "data_quality_score >= 0 AND data_quality_score <= 1",
            name="valid_data_quality_score",
        ),
        CheckConstraint("expires_at > cached_at", name="valid_expiration_time"),
        # Performance indexes
        Index("idx_ticker_data_type", "ticker", "data_type"),
        Index("idx_expires_at", "expires_at"),
        Index("idx_cached_at", "cached_at"),
    )

    def __repr__(self) -> str:
        return f"<MarketDataCache(id={self.id}, ticker='{self.ticker}', type='{self.data_type}')>"

    @property
    def is_expired(self) -> bool:
        """Check if this cached data has expired."""
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)

    @property
    def age_in_hours(self) -> float:
        """Get the age of this cached data in hours."""
        delta = datetime.utcnow() - self.cached_at.replace(tzinfo=None)
        return delta.total_seconds() / 3600

    def get_data_value(self, key: str, default=None):
        """Safely get a value from the data JSON field."""
        if self.data is None:
            return default
        return self.data.get(key, default)


class PriceHistory(Base):
    """
    Price history model for storing historical price data.

    Optimized table for storing and querying historical price data
    with proper indexing for time-series queries.
    """

    __tablename__ = "price_history"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Stock identification
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Price data
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    open_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4), nullable=True  # Up to $99,999,999.9999
    )

    high_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)

    low_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)

    close_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)

    volume: Mapped[Optional[int]] = mapped_column(nullable=True)

    adjusted_close: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4), nullable=True
    )

    # Data metadata
    data_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default="yfinance"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Table constraints and indexes
    __table_args__ = (
        # Unique constraint for ticker + date
        UniqueConstraint("ticker", "date", name="unique_ticker_date"),
        # Check constraints
        CheckConstraint("open_price > 0", name="positive_open_price"),
        CheckConstraint("high_price > 0", name="positive_high_price"),
        CheckConstraint("low_price > 0", name="positive_low_price"),
        CheckConstraint("close_price > 0", name="positive_close_price"),
        CheckConstraint("volume >= 0", name="non_negative_volume"),
        # Performance indexes for time-series queries
        Index("idx_ticker_date", "ticker", "date"),
        Index("idx_date_ticker", "date", "ticker"),
        Index(
            "idx_ticker_date_desc", "ticker", "date", postgresql_ops={"date": "DESC"}
        ),
    )

    def __repr__(self) -> str:
        return f"<PriceHistory(ticker='{self.ticker}', date='{self.date}', close={self.close_price})>"


class OptionsData(Base):
    """
    Options data model for storing options chain information.

    Stores options data with expiration dates, strikes, and Greeks
    for analysis and caching purposes.
    """

    __tablename__ = "options_data"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Underlying stock
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Option details
    option_type: Mapped[str] = mapped_column(
        String(4), nullable=False  # 'call' or 'put'
    )

    strike_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)

    expiration_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Market data
    last_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)

    bid: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)

    ask: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4), nullable=True)

    volume: Mapped[Optional[int]] = mapped_column(nullable=True)

    open_interest: Mapped[Optional[int]] = mapped_column(nullable=True)

    # Greeks
    delta: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4), nullable=True  # Delta from -1.0000 to 1.0000
    )

    gamma: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6), nullable=True)

    theta: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6), nullable=True)

    vega: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6), nullable=True)

    implied_volatility: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4), nullable=True  # IV as decimal (e.g., 0.2500 for 25%)
    )

    # Data metadata
    data_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default="yfinance"
    )

    fetch_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Table constraints and indexes
    __table_args__ = (
        # Check constraints
        CheckConstraint("option_type IN ('call', 'put')", name="valid_option_type"),
        CheckConstraint("strike_price > 0", name="positive_strike_price"),
        CheckConstraint("last_price >= 0", name="non_negative_last_price"),
        CheckConstraint("bid >= 0", name="non_negative_bid"),
        CheckConstraint("ask >= 0", name="non_negative_ask"),
        CheckConstraint("volume >= 0", name="non_negative_volume"),
        CheckConstraint("open_interest >= 0", name="non_negative_open_interest"),
        CheckConstraint("delta >= -1 AND delta <= 1", name="valid_delta_range"),
        CheckConstraint("implied_volatility >= 0", name="non_negative_iv"),
        # Performance indexes
        Index("idx_ticker_expiration", "ticker", "expiration_date"),
        Index("idx_ticker_type_expiration", "ticker", "option_type", "expiration_date"),
        Index("idx_expiration_strike", "expiration_date", "strike_price"),
        Index("idx_fetch_timestamp", "fetch_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<OptionsData(ticker='{self.ticker}', type='{self.option_type}', strike={self.strike_price}, exp='{self.expiration_date}')>"

    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration from now."""
        delta = self.expiration_date.replace(tzinfo=None) - datetime.utcnow()
        return max(0, delta.days)

    @property
    def is_in_the_money(self) -> Optional[bool]:
        """
        Determine if option is in the money.
        Requires current stock price to be determined properly.
        """
        # This would need current stock price data
        # For now, return None - would be calculated with current market data
        return None
