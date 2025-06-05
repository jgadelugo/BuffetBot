"""
Portfolio models for BuffetBot database.

Models for user portfolios, positions, and risk management.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..connection import Base


class RiskTolerance(str, Enum):
    """Risk tolerance levels for portfolios."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Portfolio(Base):
    """
    Portfolio model representing a user's investment portfolio.

    A portfolio contains multiple positions and has an associated risk tolerance level.
    """

    __tablename__ = "portfolios"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Portfolio details
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    risk_tolerance: Mapped[RiskTolerance] = mapped_column(
        String(50), nullable=False, index=True
    )

    # Portfolio metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Target allocations (as percentages)
    target_cash_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),  # Up to 999.99%
        nullable=True,
        default=Decimal("5.00"),  # 5% default cash allocation
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="portfolios")

    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="portfolio", cascade="all, delete-orphan"
    )

    analysis_results: Mapped[list["AnalysisResult"]] = relationship(
        "AnalysisResult", back_populates="portfolio", cascade="all, delete-orphan"
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint(
            "risk_tolerance IN ('conservative', 'moderate', 'aggressive')",
            name="valid_risk_tolerance",
        ),
        CheckConstraint(
            "target_cash_percentage >= 0 AND target_cash_percentage <= 100",
            name="valid_cash_percentage",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Portfolio(id={self.id}, name='{self.name}', risk={self.risk_tolerance})>"
        )

    @property
    def total_positions(self) -> int:
        """Get the total number of positions in this portfolio."""
        return len(self.positions)

    @property
    def total_allocation_percentage(self) -> Decimal:
        """Calculate total allocation percentage of all positions."""
        return sum(pos.allocation_percentage or Decimal("0") for pos in self.positions)


class Position(Base):
    """
    Position model representing a stock position within a portfolio.

    Each position represents holdings of a specific ticker within a portfolio.
    """

    __tablename__ = "positions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Foreign key to portfolio
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Position details
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    shares: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4), nullable=True  # Up to 99,999,999.9999 shares
    )

    average_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4), nullable=True  # Up to $99,999,999.9999 per share
    )

    allocation_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2), nullable=True  # Up to 999.99%
    )

    # Position metadata
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio", back_populates="positions"
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint("shares >= 0", name="positive_shares"),
        CheckConstraint("average_cost >= 0", name="positive_average_cost"),
        CheckConstraint(
            "allocation_percentage >= 0 AND allocation_percentage <= 100",
            name="valid_allocation_percentage",
        ),
        # Unique constraint for ticker per portfolio
        UniqueConstraint("portfolio_id", "ticker", name="unique_ticker_per_portfolio"),
    )

    def __repr__(self) -> str:
        return f"<Position(id={self.id}, ticker='{self.ticker}', shares={self.shares})>"

    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate market value if shares and current price are available."""
        # This would need current price data which would come from market data models
        # For now, return None - this would be calculated with current market data
        return None

    @property
    def cost_basis(self) -> Optional[Decimal]:
        """Calculate total cost basis of the position."""
        if self.shares and self.average_cost:
            return self.shares * self.average_cost
        return None
