"""
Market data repository for BuffetBot database layer.

Repository for managing cached market data with TTL, cleanup, and retrieval operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from buffetbot.utils.logger import setup_logger

from ..exceptions import (
    CacheError,
    EntityNotFoundError,
    RepositoryError,
    ValidationError,
)
from .base import BaseRepository

# Import models - will be available when Phase 1a models are integrated
try:
    from ..models.market_data import (
        DataType,
        MarketDataCache,
        OptionsData,
        PriceHistory,
    )
except ImportError:
    # Placeholder classes for development
    class MarketDataCache:
        pass

    class PriceHistory:
        pass

    class OptionsData:
        pass

    class DataType:
        pass


# Initialize logger
logger = setup_logger(__name__)


class MarketDataRepository(BaseRepository[MarketDataCache]):
    """
    Repository for market data cache management operations.

    Provides operations for storing, retrieving, and managing cached market data
    with support for TTL, expiration, and cleanup operations.
    """

    def __init__(self, session):
        super().__init__(session, MarketDataCache)

    async def _validate_entity(
        self, entity: MarketDataCache, is_update: bool = False
    ) -> None:
        """Validate market data cache entity before database operations."""
        if not entity.ticker or not entity.ticker.strip():
            raise ValidationError("Ticker is required", field="ticker")

        if len(entity.ticker) > 10:
            raise ValidationError(
                "Ticker too long", field="ticker", value=entity.ticker
            )

        if not entity.data_type:
            raise ValidationError("Data type is required", field="data_type")

        if not entity.data:
            raise ValidationError("Data is required", field="data")

        if not entity.expires_at:
            raise ValidationError("Expiration time is required", field="expires_at")

        if entity.data_quality_score is not None:
            if entity.data_quality_score < 0 or entity.data_quality_score > 1:
                raise ValidationError(
                    "Data quality score must be between 0 and 1",
                    field="data_quality_score",
                    value=entity.data_quality_score,
                )

        if entity.expires_at <= entity.cached_at:
            raise ValidationError(
                "Expiration time must be after cache time",
                field="expires_at",
                value=entity.expires_at,
            )

    async def _apply_eager_loading(self, query: Select) -> Select:
        """Apply eager loading for market data cache relationships."""
        # MarketDataCache doesn't have relationships, so return as-is
        return query

    async def get_cached_data(
        self, ticker: str, data_type: str, parameters: dict = None
    ) -> Optional[MarketDataCache]:
        """
        Get cached market data for a ticker and data type.

        Args:
            ticker: Stock ticker
            data_type: Type of market data
            parameters: Optional parameters for cache key

        Returns:
            Optional[MarketDataCache]: Cached data if found and not expired, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Getting cached data for {ticker}, type: {data_type}")

            # Build query
            query = select(MarketDataCache).where(
                and_(
                    MarketDataCache.ticker == ticker.upper(),
                    MarketDataCache.data_type == data_type,
                )
            )

            # Add parameters filter if provided
            if parameters:
                query = query.where(MarketDataCache.parameters == parameters)

            # Order by most recent
            query = query.order_by(MarketDataCache.cached_at.desc())

            result = await self.session.execute(query)
            cached_data = result.scalars().first()

            if cached_data:
                # Check if expired
                if cached_data.expires_at <= datetime.utcnow():
                    self.logger.debug(f"Cached data for {ticker} is expired")
                    return None

                self.logger.debug(f"Found valid cached data for {ticker}")
                return cached_data
            else:
                self.logger.debug(f"No cached data found for {ticker}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to get cached data for {ticker}: {e}")
            raise RepositoryError(
                "Failed to get cached data",
                repository=self.__class__.__name__,
                operation="get_cached_data",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "data_type": data_type,
                    "parameters": parameters,
                },
            )

    async def cache_market_data(
        self,
        ticker: str,
        data_type: str,
        data: dict,
        ttl_hours: int = 24,
        data_source: str = None,
        parameters: dict = None,
        data_quality_score: float = None,
    ) -> MarketDataCache:
        """
        Cache market data with specified TTL.

        Args:
            ticker: Stock ticker
            data_type: Type of market data
            data: Market data to cache
            ttl_hours: Time to live in hours
            data_source: Optional data source identifier
            parameters: Optional parameters for cache key
            data_quality_score: Optional data quality score

        Returns:
            MarketDataCache: Created cache entry

        Raises:
            RepositoryError: If caching fails
        """
        try:
            self.logger.debug(
                f"Caching data for {ticker}, type: {data_type}, TTL: {ttl_hours}h"
            )

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

            # Check if cache entry already exists
            existing_cache = await self.get_cached_data(ticker, data_type, parameters)

            if existing_cache:
                # Update existing cache entry
                existing_cache.data = data
                existing_cache.cached_at = datetime.utcnow()
                existing_cache.expires_at = expires_at
                existing_cache.data_source = data_source
                existing_cache.data_quality_score = data_quality_score

                updated_cache = await self.update(existing_cache)
                self.logger.info(f"Updated cache for {ticker}, type: {data_type}")
                return updated_cache
            else:
                # Create new cache entry
                cache_entry = MarketDataCache(
                    ticker=ticker.upper(),
                    data_type=data_type,
                    data=data,
                    expires_at=expires_at,
                    data_source=data_source,
                    parameters=parameters,
                    data_quality_score=data_quality_score,
                )

                created_cache = await self.create(cache_entry)
                self.logger.info(f"Created cache for {ticker}, type: {data_type}")
                return created_cache

        except Exception as e:
            self.logger.error(f"Failed to cache data for {ticker}: {e}")
            raise RepositoryError(
                "Failed to cache market data",
                repository=self.__class__.__name__,
                operation="cache_market_data",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "data_type": data_type,
                    "ttl_hours": ttl_hours,
                },
            )

    async def cleanup_expired_cache(self, batch_size: int = 100) -> int:
        """
        Clean up expired cache entries.

        Args:
            batch_size: Number of records to delete in each batch

        Returns:
            int: Number of expired cache entries deleted

        Raises:
            RepositoryError: If cleanup fails
        """
        try:
            self.logger.debug("Starting cleanup of expired cache entries")

            current_time = datetime.utcnow()
            total_deleted = 0

            while True:
                # Find expired cache entries
                query = (
                    select(MarketDataCache)
                    .where(MarketDataCache.expires_at <= current_time)
                    .limit(batch_size)
                )

                result = await self.session.execute(query)
                expired_entries = result.scalars().all()

                if not expired_entries:
                    break

                # Delete the batch
                for entry in expired_entries:
                    await self.session.delete(entry)

                await self.session.flush()
                batch_deleted = len(expired_entries)
                total_deleted += batch_deleted

                self.logger.debug(f"Deleted {batch_deleted} expired cache entries")

                # If we got less than batch_size, we're done
                if batch_deleted < batch_size:
                    break

            self.logger.info(
                f"Cleanup completed: deleted {total_deleted} expired cache entries"
            )
            return total_deleted

        except Exception as e:
            self.logger.error(f"Failed to cleanup expired cache: {e}")
            raise RepositoryError(
                "Failed to cleanup expired cache",
                repository=self.__class__.__name__,
                operation="cleanup_expired_cache",
                details={"error": str(e), "batch_size": batch_size},
            )

    async def invalidate_ticker_cache(self, ticker: str, data_type: str = None) -> int:
        """
        Invalidate cache entries for a specific ticker.

        Args:
            ticker: Stock ticker
            data_type: Optional data type filter

        Returns:
            int: Number of cache entries invalidated
        """
        try:
            self.logger.debug(f"Invalidating cache for ticker: {ticker}")

            current_time = datetime.utcnow()

            # Build query
            query = select(MarketDataCache).where(
                MarketDataCache.ticker == ticker.upper()
            )

            if data_type:
                query = query.where(MarketDataCache.data_type == data_type)

            # Get matching entries
            result = await self.session.execute(query)
            cache_entries = result.scalars().all()

            # Update expiration times
            count = 0
            for entry in cache_entries:
                entry.expires_at = current_time
                count += 1

            await self.session.flush()

            self.logger.info(f"Invalidated {count} cache entries for {ticker}")
            return count

        except Exception as e:
            self.logger.error(f"Failed to invalidate cache for {ticker}: {e}")
            raise RepositoryError(
                "Failed to invalidate ticker cache",
                repository=self.__class__.__name__,
                operation="invalidate_ticker_cache",
                details={"error": str(e), "ticker": ticker, "data_type": data_type},
            )

    async def get_cache_statistics(self, days_back: int = 7) -> dict:
        """
        Get cache usage statistics.

        Args:
            days_back: Number of days to look back for statistics

        Returns:
            dict: Cache statistics
        """
        try:
            self.logger.debug(f"Getting cache statistics for last {days_back} days")

            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            current_time = datetime.utcnow()

            # Get all cache entries from the period
            filters = {"cached_at": {"operator": "gte", "value": cutoff_time}}

            cache_entries = await self.list_by_criteria(
                limit=10000, **filters  # Large limit for statistics
            )

            # Calculate statistics
            total_entries = len(cache_entries)
            expired_entries = sum(
                1 for entry in cache_entries if entry.expires_at <= current_time
            )
            valid_entries = total_entries - expired_entries

            # Group by data type
            by_data_type = {}
            by_ticker = {}

            for entry in cache_entries:
                # By data type
                data_type = entry.data_type
                if data_type not in by_data_type:
                    by_data_type[data_type] = {"total": 0, "expired": 0, "valid": 0}

                by_data_type[data_type]["total"] += 1
                if entry.expires_at <= current_time:
                    by_data_type[data_type]["expired"] += 1
                else:
                    by_data_type[data_type]["valid"] += 1

                # By ticker (top 10)
                ticker = entry.ticker
                if ticker not in by_ticker:
                    by_ticker[ticker] = 0
                by_ticker[ticker] += 1

            # Get top 10 tickers
            top_tickers = sorted(by_ticker.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]

            statistics = {
                "period_days": days_back,
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "expiration_rate": round(expired_entries / total_entries * 100, 2)
                if total_entries > 0
                else 0,
                "by_data_type": by_data_type,
                "top_tickers": dict(top_tickers),
                "generated_at": current_time.isoformat(),
            }

            self.logger.debug(
                f"Generated cache statistics: {total_entries} total entries"
            )
            return statistics

        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            raise RepositoryError(
                "Failed to get cache statistics",
                repository=self.__class__.__name__,
                operation="get_cache_statistics",
                details={"error": str(e), "days_back": days_back},
            )

    async def get_cache_size_by_ticker(self, limit: int = 100) -> list[dict]:
        """
        Get cache size breakdown by ticker.

        Args:
            limit: Maximum number of tickers to return

        Returns:
            List[dict]: List of ticker cache statistics
        """
        try:
            self.logger.debug(f"Getting cache size by ticker (limit: {limit})")

            # Get all current valid cache entries
            current_time = datetime.utcnow()

            cache_entries = await self.list_by_criteria(
                expires_at={"operator": "gt", "value": current_time}, limit=10000
            )

            # Group by ticker
            ticker_stats = {}
            for entry in cache_entries:
                ticker = entry.ticker
                if ticker not in ticker_stats:
                    ticker_stats[ticker] = {
                        "ticker": ticker,
                        "entry_count": 0,
                        "data_types": set(),
                        "total_size_estimate": 0,
                        "oldest_entry": None,
                        "newest_entry": None,
                    }

                stats = ticker_stats[ticker]
                stats["entry_count"] += 1
                stats["data_types"].add(entry.data_type)

                # Estimate size (rough approximation)
                if entry.data:
                    stats["total_size_estimate"] += len(str(entry.data))

                # Track oldest and newest entries
                if (
                    stats["oldest_entry"] is None
                    or entry.cached_at < stats["oldest_entry"]
                ):
                    stats["oldest_entry"] = entry.cached_at

                if (
                    stats["newest_entry"] is None
                    or entry.cached_at > stats["newest_entry"]
                ):
                    stats["newest_entry"] = entry.cached_at

            # Convert to list and sort by entry count
            ticker_list = []
            for ticker, stats in ticker_stats.items():
                stats["data_types"] = list(stats["data_types"])
                stats["oldest_entry"] = (
                    stats["oldest_entry"].isoformat() if stats["oldest_entry"] else None
                )
                stats["newest_entry"] = (
                    stats["newest_entry"].isoformat() if stats["newest_entry"] else None
                )
                ticker_list.append(stats)

            # Sort by entry count descending
            ticker_list.sort(key=lambda x: x["entry_count"], reverse=True)

            # Apply limit
            result = ticker_list[:limit]

            self.logger.debug(
                f"Generated cache size breakdown for {len(result)} tickers"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to get cache size by ticker: {e}")
            raise RepositoryError(
                "Failed to get cache size by ticker",
                repository=self.__class__.__name__,
                operation="get_cache_size_by_ticker",
                details={"error": str(e), "limit": limit},
            )

    async def refresh_cache_entry(
        self,
        ticker: str,
        data_type: str,
        new_data: dict,
        ttl_hours: int = 24,
        parameters: dict = None,
    ) -> MarketDataCache:
        """
        Refresh an existing cache entry with new data.

        Args:
            ticker: Stock ticker
            data_type: Type of market data
            new_data: New data to cache
            ttl_hours: Time to live in hours
            parameters: Optional parameters for cache key

        Returns:
            MarketDataCache: Updated cache entry
        """
        try:
            self.logger.debug(f"Refreshing cache for {ticker}, type: {data_type}")

            # Get existing cache entry (including expired ones)
            query = select(MarketDataCache).where(
                and_(
                    MarketDataCache.ticker == ticker.upper(),
                    MarketDataCache.data_type == data_type,
                )
            )

            if parameters:
                query = query.where(MarketDataCache.parameters == parameters)

            query = query.order_by(MarketDataCache.cached_at.desc())

            result = await self.session.execute(query)
            existing_cache = result.scalars().first()

            if existing_cache:
                # Update existing entry
                existing_cache.data = new_data
                existing_cache.cached_at = datetime.utcnow()
                existing_cache.expires_at = datetime.utcnow() + timedelta(
                    hours=ttl_hours
                )

                updated_cache = await self.update(existing_cache)
                self.logger.info(f"Refreshed cache for {ticker}, type: {data_type}")
                return updated_cache
            else:
                # Create new entry if none exists
                return await self.cache_market_data(
                    ticker=ticker,
                    data_type=data_type,
                    data=new_data,
                    ttl_hours=ttl_hours,
                    parameters=parameters,
                )

        except Exception as e:
            self.logger.error(f"Failed to refresh cache for {ticker}: {e}")
            raise RepositoryError(
                "Failed to refresh cache entry",
                repository=self.__class__.__name__,
                operation="refresh_cache_entry",
                details={
                    "error": str(e),
                    "ticker": ticker,
                    "data_type": data_type,
                    "ttl_hours": ttl_hours,
                },
            )


class PriceHistoryRepository(BaseRepository[PriceHistory]):
    """Repository for price history operations."""

    def __init__(self):
        super().__init__(PriceHistory)


class OptionsDataRepository(BaseRepository[OptionsData]):
    """Repository for options data operations."""

    def __init__(self):
        super().__init__(OptionsData)
