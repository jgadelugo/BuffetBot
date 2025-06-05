"""
Analysis models for BuffetBot database.

Models for storing analysis results, calculations, and recommendations.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..connection import Base


class AnalysisType(str, Enum):
    """Types of financial analysis that can be performed."""

    VALUE_ANALYSIS = "value_analysis"
    GROWTH_ANALYSIS = "growth_analysis"
    HEALTH_ANALYSIS = "health_analysis"
    RISK_ANALYSIS = "risk_analysis"
    OPTIONS_ANALYSIS = "options_analysis"
    ECOSYSTEM_ANALYSIS = "ecosystem_analysis"


class AnalysisResult(Base):
    """
    Analysis result model for storing computed financial analysis.

    Stores results from various analysis types (value, growth, health, etc.)
    with metadata about the analysis and expiration times for caching.
    """

    __tablename__ = "analysis_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )

    # Foreign key to portfolio (optional for single-stock analysis)
    portfolio_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Analysis details
    ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    analysis_type: Mapped[AnalysisType] = mapped_column(
        String(50), nullable=False, index=True
    )

    strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Specific strategy used (e.g., 'dcf', 'pe_ratio', 'long_calls')",
    )

    # Analysis results
    score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),  # Score from 0.00 to 999.99
        nullable=True,
        comment="Overall analysis score or rating",
    )

    analysis_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="Complete analysis results and metadata as JSON"
    )

    # Recommendations and insights
    recommendation: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Analysis recommendation (e.g., 'buy', 'hold', 'sell')",
    )

    confidence_level: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(3, 2),  # Confidence from 0.00 to 1.00
        nullable=True,
        comment="Confidence level in the analysis (0.0 to 1.0)",
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Human-readable summary of analysis results"
    )

    # Cache and expiration
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When this analysis result expires and should be recalculated",
    )

    # Analysis parameters for reproducibility
    parameters: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Parameters used for the analysis (for reproducibility)",
    )

    # Data quality and reliability
    data_quality_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(3, 2),  # Quality score from 0.00 to 1.00
        nullable=True,
        comment="Quality score of underlying data used (0.0 to 1.0)",
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
    portfolio: Mapped[Optional["Portfolio"]] = relationship(
        "Portfolio", back_populates="analysis_results"
    )

    # Table constraints
    __table_args__ = (
        CheckConstraint("score >= 0", name="positive_score"),
        CheckConstraint(
            "confidence_level >= 0 AND confidence_level <= 1",
            name="valid_confidence_level",
        ),
        CheckConstraint(
            "data_quality_score >= 0 AND data_quality_score <= 1",
            name="valid_data_quality_score",
        ),
        CheckConstraint(
            "analysis_type IN ('value_analysis', 'growth_analysis', 'health_analysis', 'risk_analysis', 'options_analysis', 'ecosystem_analysis')",
            name="valid_analysis_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, ticker='{self.ticker}', type='{self.analysis_type}', score={self.score})>"

    @property
    def is_expired(self) -> bool:
        """Check if this analysis result has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)

    @property
    def age_in_hours(self) -> float:
        """Get the age of this analysis in hours."""
        delta = datetime.utcnow() - self.calculated_at.replace(tzinfo=None)
        return delta.total_seconds() / 3600

    def get_metadata_value(self, key: str, default=None):
        """Safely get a value from the analysis_metadata JSON field."""
        if self.analysis_metadata is None:
            return default
        return self.analysis_metadata.get(key, default)

    def set_metadata_value(self, key: str, value) -> None:
        """Safely set a value in the analysis_metadata JSON field."""
        if self.analysis_metadata is None:
            self.analysis_metadata = {}
        self.analysis_metadata[key] = value
