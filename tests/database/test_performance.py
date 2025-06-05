"""
Performance tests for database operations.

Comprehensive performance testing suite that validates:
- CRUD operation performance benchmarks
- Concurrent access performance and safety
- Query optimization and indexing effectiveness
- Connection pool performance under load
- Memory usage and resource management
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import AnalysisResult, MarketDataCache, Portfolio, Position, User
from database.repositories import RepositoryRegistry
from database.repositories.analysis_repo import AnalysisRepository
from database.repositories.market_data_repo import MarketDataRepository
from database.repositories.portfolio_repo import PortfolioRepository, PositionRepository

from .conftest import (
    DatabaseTestUtils,
    generate_test_portfolios,
    generate_test_positions,
    performance_timer,
)

# Performance benchmarks (in seconds)
PERFORMANCE_BENCHMARKS = {
    "create_single": 0.1,  # 100ms max for single entity creation
    "create_bulk_100": 1.0,  # 1s max for 100 entities
    "read_by_id": 0.05,  # 50ms max for single entity read
    "read_list_100": 0.2,  # 200ms max for reading 100 entities
    "update_single": 0.1,  # 100ms max for single entity update
    "update_bulk_100": 1.5,  # 1.5s max for 100 entity updates
    "delete_single": 0.05,  # 50ms max for single entity deletion
    "complex_query": 0.5,  # 500ms max for complex queries
    "concurrent_reads": 1.0,  # 1s max for 10 concurrent reads
    "concurrent_writes": 2.0,  # 2s max for 10 concurrent writes
}


@pytest.mark.performance
@pytest.mark.asyncio
class TestRepositoryPerformance:
    """Performance tests for repository operations."""

    async def test_portfolio_repository_crud_performance(
        self, db_session: AsyncSession, sample_user: User, performance_metrics
    ):
        """Benchmark CRUD operations performance for portfolio repository."""
        repo = PortfolioRepository(db_session)

        # Test single create performance
        async with performance_timer(performance_metrics, "create_single"):
            portfolio = await repo.create(
                Portfolio(
                    user_id=sample_user.id,
                    name="Performance Test Portfolio",
                    description="Testing single create performance",
                    cash_balance=10000.0,
                )
            )

        assert (
            performance_metrics.get_average("create_single")
            < PERFORMANCE_BENCHMARKS["create_single"]
        )

        # Test read by ID performance
        async with performance_timer(performance_metrics, "read_by_id"):
            retrieved = await repo.get_by_id(portfolio.id)

        assert retrieved is not None
        assert (
            performance_metrics.get_average("read_by_id")
            < PERFORMANCE_BENCHMARKS["read_by_id"]
        )

        # Test update performance
        portfolio.name = "Updated Portfolio Name"
        async with performance_timer(performance_metrics, "update_single"):
            updated = await repo.update(portfolio)

        assert updated.name == "Updated Portfolio Name"
        assert (
            performance_metrics.get_average("update_single")
            < PERFORMANCE_BENCHMARKS["update_single"]
        )

        # Test delete performance
        async with performance_timer(performance_metrics, "delete_single"):
            deleted = await repo.delete(portfolio.id)

        assert deleted is True
        assert (
            performance_metrics.get_average("delete_single")
            < PERFORMANCE_BENCHMARKS["delete_single"]
        )

    async def test_bulk_operations_performance(
        self, db_session: AsyncSession, sample_user: User, performance_metrics
    ):
        """Test bulk operations performance."""
        repo = PortfolioRepository(db_session)

        # Generate test portfolios
        test_portfolios = generate_test_portfolios(100, sample_user.id)

        # Test bulk create performance
        async with performance_timer(performance_metrics, "create_bulk_100"):
            created_portfolios = await repo.bulk_create(test_portfolios)

        assert len(created_portfolios) == 100
        assert (
            performance_metrics.get_average("create_bulk_100")
            < PERFORMANCE_BENCHMARKS["create_bulk_100"]
        )

        # Test bulk read performance
        async with performance_timer(performance_metrics, "read_list_100"):
            portfolios = await repo.list_by_criteria(user_id=sample_user.id, limit=100)

        assert len(portfolios) >= 100
        assert (
            performance_metrics.get_average("read_list_100")
            < PERFORMANCE_BENCHMARKS["read_list_100"]
        )


@pytest.mark.performance
@pytest.mark.concurrent
@pytest.mark.asyncio
class TestConcurrentPerformance:
    """Performance tests for concurrent database operations."""

    @pytest.mark.parametrize("concurrent_sessions", [10], indirect=True)
    async def test_concurrent_read_performance(
        self,
        concurrent_sessions: list[AsyncSession],
        sample_user: User,
        sample_portfolio: Portfolio,
        performance_metrics,
    ):
        """Test concurrent read operations performance."""

        async def concurrent_read(session: AsyncSession):
            repo = PortfolioRepository(session)
            return await repo.get_by_id(sample_portfolio.id)

        # Execute concurrent reads
        async with performance_timer(performance_metrics, "concurrent_reads"):
            results = await asyncio.gather(
                *[concurrent_read(session) for session in concurrent_sessions]
            )

        # Verify all reads succeeded
        assert all(result is not None for result in results)
        assert all(result.id == sample_portfolio.id for result in results)
        assert (
            performance_metrics.get_average("concurrent_reads")
            < PERFORMANCE_BENCHMARKS["concurrent_reads"]
        )


@pytest.mark.performance
@pytest.mark.asyncio
class TestDatabaseConnectionPerformance:
    """Performance tests for database connection and session management."""

    async def test_connection_pool_performance(self, test_engine, performance_metrics):
        """Test connection pool performance under load."""

        async def execute_query():
            async with test_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar()

        # Test connection acquisition performance
        async with performance_timer(performance_metrics, "connection_pool"):
            results = await asyncio.gather(*[execute_query() for _ in range(50)])

        assert all(result == 1 for result in results)
        # Connection pool should handle 50 concurrent connections efficiently
        assert performance_metrics.get_average("connection_pool") < 2.0


# Performance test utilities
class PerformanceReporter:
    """Utility class for reporting performance test results."""

    @staticmethod
    def generate_performance_report(performance_metrics) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": {},
            "details": {},
            "benchmarks": PERFORMANCE_BENCHMARKS,
            "status": "PASS",
        }

        for operation, times in performance_metrics.metrics.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)

            report["details"][operation] = {
                "average": avg_time,
                "maximum": max_time,
                "minimum": min_time,
                "samples": len(times),
                "benchmark": PERFORMANCE_BENCHMARKS.get(operation, "N/A"),
            }

            # Check if operation meets benchmark
            benchmark = PERFORMANCE_BENCHMARKS.get(operation)
            if benchmark and avg_time > benchmark:
                report["status"] = "FAIL"
                report["details"][operation]["status"] = "FAIL"
            else:
                report["details"][operation]["status"] = "PASS"

        return report


@pytest.fixture
def performance_reporter():
    """Provide performance reporter utility."""
    return PerformanceReporter()
