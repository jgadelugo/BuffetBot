"""
Analysis repository for BuffetBot database layer.

Repository for managing analysis results with caching, expiration, and retrieval operations.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from buffetbot.utils.logger import setup_logger

from ..exceptions import EntityNotFoundError, RepositoryError, ValidationError
from .base import BaseRepository

# Import models - will be available when Phase 1a models are integrated
try:
    from ..models.analysis import AnalysisResult, AnalysisType
    from ..models.portfolio import Portfolio
except ImportError:
    # Placeholder classes for development
    class AnalysisResult:
        pass

    class AnalysisType:
        pass

    class Portfolio:
        pass


# Initialize logger
logger = setup_logger(__name__)


class AnalysisRepository(BaseRepository[AnalysisResult]):
    """
    Repository for analysis result management operations.

    Provides operations for storing, retrieving, and managing analysis results
    with support for caching, expiration, and analysis history.
    """

    def __init__(self, session):
        super().__init__(session, AnalysisResult)

    async def _validate_entity(
        self, entity: AnalysisResult, is_update: bool = False
    ) -> None:
        """Validate analysis result entity before database operations."""
        if not entity.ticker or not entity.ticker.strip():
            raise ValidationError("Ticker is required", field="ticker")

        if len(entity.ticker) > 10:
            raise ValidationError(
                "Ticker too long", field="ticker", value=entity.ticker
            )

        if not entity.analysis_type:
            raise ValidationError("Analysis type is required", field="analysis_type")

        if entity.score is not None and entity.score < 0:
            raise ValidationError(
                "Score cannot be negative", field="score", value=entity.score
            )

        if entity.confidence_level is not None:
            if entity.confidence_level < 0 or entity.confidence_level > 1:
                raise ValidationError(
                    "Confidence level must be between 0 and 1",
                    field="confidence_level",
                    value=entity.confidence_level,
                )

        if entity.data_quality_score is not None:
            if entity.data_quality_score < 0 or entity.data_quality_score > 1:
                raise ValidationError(
                    "Data quality score must be between 0 and 1",
                    field="data_quality_score",
                    value=entity.data_quality_score,
                )

        if entity.expires_at and entity.calculated_at:
            if entity.expires_at <= entity.calculated_at:
                raise ValidationError(
                    "Expiration time must be after calculation time",
                    field="expires_at",
                    value=entity.expires_at,
                )

    async def _apply_eager_loading(self, query: Select) -> Select:
        """Apply eager loading for analysis result relationships."""
        return query.options(selectinload(AnalysisResult.portfolio))

    async def get_recent_analysis(
        self,
        portfolio_id: UUID = None,
        ticker: str = None,
        analysis_type: str = None,
        max_age_hours: int = 24,
    ) -> Optional[AnalysisResult]:
        """
        Get the most recent analysis result within the specified age limit.

        Args:
            portfolio_id: Optional portfolio ID to filter by
            ticker: Optional ticker to filter by
            analysis_type: Type of analysis to retrieve
            max_age_hours: Maximum age of analysis in hours

        Returns:
            Optional[AnalysisResult]: Most recent analysis if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(
                f"Getting recent analysis: portfolio_id={portfolio_id}, "
                f"ticker={ticker}, type={analysis_type}, max_age={max_age_hours}h"
            )

            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

            # Build query
            query = select(AnalysisResult).where(
                AnalysisResult.calculated_at >= cutoff_time
            )

            # Apply filters
            if portfolio_id:
                query = query.where(AnalysisResult.portfolio_id == portfolio_id)

            if ticker:
                query = query.where(AnalysisResult.ticker == ticker.upper())

            if analysis_type:
                query = query.where(AnalysisResult.analysis_type == analysis_type)

            # Order by most recent and apply eager loading
            query = query.order_by(AnalysisResult.calculated_at.desc())
            query = await self._apply_eager_loading(query)

            result = await self.session.execute(query)
            analysis = result.scalars().first()

            if analysis:
                self.logger.debug(f"Found recent analysis: {analysis.id}")
            else:
                self.logger.debug("No recent analysis found matching criteria")

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to get recent analysis: {e}")
            raise RepositoryError(
                "Failed to get recent analysis",
                repository=self.__class__.__name__,
                operation="get_recent_analysis",
                details={
                    "error": str(e),
                    "portfolio_id": str(portfolio_id) if portfolio_id else None,
                    "ticker": ticker,
                    "analysis_type": analysis_type,
                    "max_age_hours": max_age_hours,
                },
            )

    async def get_portfolio_analysis_history(
        self, portfolio_id: UUID, analysis_type: str = None, limit: int = 50
    ) -> list[AnalysisResult]:
        """
        Get analysis history for a portfolio.

        Args:
            portfolio_id: ID of the portfolio
            analysis_type: Optional analysis type filter
            limit: Maximum number of results to return

        Returns:
            List[AnalysisResult]: List of analysis results ordered by date

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Getting analysis history for portfolio: {portfolio_id}")

            filters = {"portfolio_id": portfolio_id}
            if analysis_type:
                filters["analysis_type"] = analysis_type

            return await self.list_by_criteria(
                limit=limit, order_by="-calculated_at", **filters
            )

        except Exception as e:
            self.logger.error(
                f"Failed to get portfolio analysis history for {portfolio_id}: {e}"
            )
            raise RepositoryError(
                "Failed to get portfolio analysis history",
                repository=self.__class__.__name__,
                operation="get_portfolio_analysis_history",
                details={
                    "error": str(e),
                    "portfolio_id": str(portfolio_id),
                    "analysis_type": analysis_type,
                },
            )

    async def get_ticker_analysis_history(
        self, ticker: str, analysis_type: str = None, limit: int = 50
    ) -> list[AnalysisResult]:
        """
        Get analysis history for a specific ticker.

        Args:
            ticker: Stock ticker
            analysis_type: Optional analysis type filter
            limit: Maximum number of results to return

        Returns:
            List[AnalysisResult]: List of analysis results ordered by date
        """
        try:
            self.logger.debug(f"Getting analysis history for ticker: {ticker}")

            filters = {"ticker": ticker.upper()}
            if analysis_type:
                filters["analysis_type"] = analysis_type

            return await self.list_by_criteria(
                limit=limit, order_by="-calculated_at", **filters
            )

        except Exception as e:
            self.logger.error(
                f"Failed to get ticker analysis history for {ticker}: {e}"
            )
            raise RepositoryError(
                "Failed to get ticker analysis history",
                repository=self.__class__.__name__,
                operation="get_ticker_analysis_history",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "analysis_type": analysis_type,
                },
            )

    async def cleanup_expired_analysis(self, batch_size: int = 100) -> int:
        """
        Clean up expired analysis results.

        Args:
            batch_size: Number of records to delete in each batch

        Returns:
            int: Number of expired analysis results deleted

        Raises:
            RepositoryError: If cleanup fails
        """
        try:
            self.logger.debug("Starting cleanup of expired analysis results")

            current_time = datetime.utcnow()
            total_deleted = 0

            while True:
                # Find expired analysis results
                query = (
                    select(AnalysisResult)
                    .where(
                        and_(
                            AnalysisResult.expires_at.isnot(None),
                            AnalysisResult.expires_at <= current_time,
                        )
                    )
                    .limit(batch_size)
                )

                result = await self.session.execute(query)
                expired_analyses = result.scalars().all()

                if not expired_analyses:
                    break

                # Delete the batch
                for analysis in expired_analyses:
                    await self.session.delete(analysis)

                await self.session.flush()
                batch_deleted = len(expired_analyses)
                total_deleted += batch_deleted

                self.logger.debug(f"Deleted {batch_deleted} expired analysis results")

                # If we got less than batch_size, we're done
                if batch_deleted < batch_size:
                    break

            self.logger.info(
                f"Cleanup completed: deleted {total_deleted} expired analysis results"
            )
            return total_deleted

        except Exception as e:
            self.logger.error(f"Failed to cleanup expired analysis: {e}")
            raise RepositoryError(
                "Failed to cleanup expired analysis",
                repository=self.__class__.__name__,
                operation="cleanup_expired_analysis",
                details={"error": str(e), "batch_size": batch_size},
            )

    async def get_analysis_summary_by_type(
        self, portfolio_id: UUID = None, days_back: int = 30
    ) -> dict:
        """
        Get a summary of analysis results by type for the specified period.

        Args:
            portfolio_id: Optional portfolio ID to filter by
            days_back: Number of days to look back

        Returns:
            dict: Summary statistics by analysis type
        """
        try:
            self.logger.debug(f"Getting analysis summary for portfolio: {portfolio_id}")

            cutoff_time = datetime.utcnow() - timedelta(days=days_back)

            filters = {"calculated_at": {"operator": "gte", "value": cutoff_time}}

            if portfolio_id:
                filters["portfolio_id"] = portfolio_id

            # Get all analysis results for the period
            analyses = await self.list_by_criteria(
                limit=1000, **filters  # Large limit for summary
            )

            # Group by analysis type
            summary = {}
            for analysis in analyses:
                analysis_type = analysis.analysis_type
                if analysis_type not in summary:
                    summary[analysis_type] = {
                        "count": 0,
                        "avg_score": 0,
                        "avg_confidence": 0,
                        "latest": None,
                    }

                summary[analysis_type]["count"] += 1

                # Update averages
                if analysis.score is not None:
                    current_avg = summary[analysis_type]["avg_score"]
                    count = summary[analysis_type]["count"]
                    summary[analysis_type]["avg_score"] = (
                        current_avg * (count - 1) + float(analysis.score)
                    ) / count

                if analysis.confidence_level is not None:
                    current_avg = summary[analysis_type]["avg_confidence"]
                    count = summary[analysis_type]["count"]
                    summary[analysis_type]["avg_confidence"] = (
                        current_avg * (count - 1) + float(analysis.confidence_level)
                    ) / count

                # Track latest analysis
                if (
                    summary[analysis_type]["latest"] is None
                    or analysis.calculated_at > summary[analysis_type]["latest"]
                ):
                    summary[analysis_type][
                        "latest"
                    ] = analysis.calculated_at.isoformat()

            self.logger.debug(f"Generated analysis summary with {len(summary)} types")
            return summary

        except Exception as e:
            self.logger.error(f"Failed to get analysis summary: {e}")
            raise RepositoryError(
                "Failed to get analysis summary",
                repository=self.__class__.__name__,
                operation="get_analysis_summary_by_type",
                details={
                    "error": str(e),
                    "portfolio_id": str(portfolio_id) if portfolio_id else None,
                    "days_back": days_back,
                },
            )

    async def invalidate_analysis(
        self, ticker: str = None, portfolio_id: UUID = None, analysis_type: str = None
    ) -> int:
        """
        Invalidate analysis results by setting their expiration to now.

        Args:
            ticker: Optional ticker to filter by
            portfolio_id: Optional portfolio ID to filter by
            analysis_type: Optional analysis type to filter by

        Returns:
            int: Number of analysis results invalidated
        """
        try:
            self.logger.debug(
                f"Invalidating analysis: ticker={ticker}, portfolio_id={portfolio_id}, type={analysis_type}"
            )

            current_time = datetime.utcnow()

            # Build base query
            query = select(AnalysisResult)

            # Apply filters
            conditions = []
            if ticker:
                conditions.append(AnalysisResult.ticker == ticker.upper())
            if portfolio_id:
                conditions.append(AnalysisResult.portfolio_id == portfolio_id)
            if analysis_type:
                conditions.append(AnalysisResult.analysis_type == analysis_type)

            if conditions:
                query = query.where(and_(*conditions))

            # Get matching results
            result = await self.session.execute(query)
            analyses = result.scalars().all()

            # Update expiration times
            count = 0
            for analysis in analyses:
                analysis.expires_at = current_time
                count += 1

            await self.session.flush()

            self.logger.info(f"Invalidated {count} analysis results")
            return count

        except Exception as e:
            self.logger.error(f"Failed to invalidate analysis: {e}")
            raise RepositoryError(
                "Failed to invalidate analysis",
                repository=self.__class__.__name__,
                operation="invalidate_analysis",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "portfolio_id": str(portfolio_id) if portfolio_id else None,
                    "analysis_type": analysis_type,
                },
            )

    async def get_cached_or_expired_analysis(
        self, ticker: str, analysis_type: str, portfolio_id: UUID = None
    ) -> tuple[Optional[AnalysisResult], bool]:
        """
        Get analysis result and whether it's expired.

        Args:
            ticker: Stock ticker
            analysis_type: Type of analysis
            portfolio_id: Optional portfolio ID

        Returns:
            tuple: (AnalysisResult or None, is_expired boolean)
        """
        try:
            filters = {"ticker": ticker.upper(), "analysis_type": analysis_type}

            if portfolio_id:
                filters["portfolio_id"] = portfolio_id

            # Get most recent analysis
            analysis_list = await self.list_by_criteria(
                limit=1, order_by="-calculated_at", **filters
            )

            if not analysis_list:
                return None, True

            analysis = analysis_list[0]

            # Check if expired
            is_expired = False
            if analysis.expires_at:
                is_expired = datetime.utcnow() > analysis.expires_at.replace(
                    tzinfo=None
                )

            return analysis, is_expired

        except Exception as e:
            self.logger.error(f"Failed to get cached analysis: {e}")
            raise RepositoryError(
                "Failed to get cached analysis",
                repository=self.__class__.__name__,
                operation="get_cached_or_expired_analysis",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "analysis_type": analysis_type,
                    "portfolio_id": str(portfolio_id) if portfolio_id else None,
                },
            )
