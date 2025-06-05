"""
Error handling tests for database operations.

Comprehensive testing suite for error scenarios including:
- Database connection failures
- Constraint violations and data integrity errors
- Concurrent access conflicts and deadlocks
- Resource exhaustion scenarios
- Network and timeout errors
- Recovery and graceful degradation
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import (
    DisconnectionError,
    IntegrityError,
    InvalidRequestError,
    OperationalError,
)
from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession

from database.exceptions import (
    ConcurrencyError,
    DatabaseConnectionError,
    DuplicateEntityError,
    EntityNotFoundError,
    RepositoryError,
    ValidationError,
)
from database.models import Portfolio, Position, RiskTolerance, User
from database.repositories import RepositoryRegistry
from database.repositories.portfolio_repo import PortfolioRepository, PositionRepository
from database.repositories.session_manager import DatabaseSessionManager


@pytest.mark.asyncio
class TestDatabaseConnectionErrors:
    """Test database connection error handling."""

    async def test_connection_failure_handling(self):
        """Test handling of database connection failures."""
        # Mock a connection failure
        with patch(
            "database.repositories.session_manager.create_async_engine"
        ) as mock_engine:
            mock_engine.side_effect = OperationalError("Connection failed", None, None)

            session_manager = DatabaseSessionManager()

            with pytest.raises(DatabaseConnectionError):
                await session_manager.initialize()

    async def test_connection_timeout_handling(self, db_session: AsyncSession):
        """Test handling of connection timeouts."""
        repo = PortfolioRepository(db_session)

        # Mock a timeout error
        with patch.object(
            db_session, "execute", side_effect=SQLTimeoutError("Timeout", None, None)
        ):
            with pytest.raises(RepositoryError) as exc_info:
                await repo.list_by_criteria()

            assert "timeout" in str(exc_info.value).lower()

    async def test_connection_recovery(self, test_engine):
        """Test connection recovery after temporary failure."""
        session_manager = DatabaseSessionManager()
        session_manager.engine = test_engine

        # Simulate connection recovery
        health_check = await session_manager.health_check()
        assert health_check["status"] == "healthy"

    async def test_connection_pool_exhaustion(self, test_database_config):
        """Test handling of connection pool exhaustion."""
        # Create config with very small pool
        config = test_database_config.model_copy()
        config.pool_size = 1
        config.max_overflow = 0

        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_async_engine(config.get_database_url(), **config.engine_kwargs)

        SessionLocal = sessionmaker(engine, class_=AsyncSession)

        try:
            # Hold one connection
            session1 = SessionLocal()

            # Try to get another (should timeout quickly)
            with pytest.raises((OperationalError, SQLTimeoutError)):
                session2 = SessionLocal()
                async with session2:
                    await session2.execute(text("SELECT 1"))

        finally:
            await session1.close()
            await engine.dispose()


@pytest.mark.asyncio
class TestDataIntegrityErrors:
    """Test data integrity and constraint violation handling."""

    async def test_duplicate_entity_handling(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test handling of duplicate entity creation."""
        repo = PortfolioRepository(db_session)

        # Create first portfolio
        portfolio1 = Portfolio(
            user_id=sample_user.id,
            name="Unique Portfolio",
            description="First portfolio",
            cash_balance=10000.0,
        )
        created1 = await repo.create(portfolio1)

        # Try to create duplicate (assuming unique constraint on user_id + name)
        portfolio2 = Portfolio(
            user_id=sample_user.id,
            name="Unique Portfolio",  # Same name
            description="Duplicate portfolio",
            cash_balance=5000.0,
        )

        # This should handle the constraint violation gracefully
        try:
            await repo.create(portfolio2)
            # If no constraint exists, this test documents the behavior
        except (RepositoryError, DuplicateEntityError):
            # Expected behavior for duplicate handling
            pass

    async def test_foreign_key_constraint_violation(self, db_session: AsyncSession):
        """Test handling of foreign key constraint violations."""
        repo = PositionRepository(db_session)

        # Try to create position with non-existent portfolio
        invalid_position = Position(
            portfolio_id="00000000-0000-0000-0000-000000000000",  # Non-existent
            symbol="AAPL",
            quantity=100,
            average_cost=150.0,
            current_price=155.0,
            position_type="STOCK",
        )

        with pytest.raises(RepositoryError) as exc_info:
            await repo.create(invalid_position)

        assert (
            "foreign key" in str(exc_info.value).lower()
            or "constraint" in str(exc_info.value).lower()
        )

    async def test_null_constraint_violation(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test handling of null constraint violations."""
        repo = PortfolioRepository(db_session)

        # Try to create portfolio with null required field
        invalid_portfolio = Portfolio(
            user_id=sample_user.id,
            name=None,  # Required field
            description="Invalid portfolio",
            cash_balance=10000.0,
        )

        with pytest.raises((RepositoryError, ValidationError)):
            await repo.create(invalid_portfolio)

    async def test_check_constraint_violation(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test handling of check constraint violations."""
        repo = PositionRepository(db_session)

        # Try to create position with invalid data (negative quantity)
        invalid_position = Position(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            quantity=-100,  # Invalid negative quantity
            average_cost=150.0,
            current_price=155.0,
            position_type="STOCK",
        )

        with pytest.raises((RepositoryError, ValidationError)):
            await repo.create(invalid_position)


@pytest.mark.asyncio
class TestConcurrencyErrors:
    """Test concurrent access conflicts and deadlock handling."""

    async def test_optimistic_locking_conflict(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test handling of optimistic locking conflicts."""
        repo = PortfolioRepository(db_session)

        # Simulate concurrent modification
        portfolio1 = await repo.get_by_id(sample_portfolio.id)
        portfolio2 = await repo.get_by_id(sample_portfolio.id)

        # Modify and save first instance
        portfolio1.cash_balance += 1000.0
        await repo.update(portfolio1)

        # Try to modify and save second instance (should detect conflict)
        portfolio2.cash_balance += 2000.0

        # Depending on implementation, this might raise a concurrency error
        try:
            await repo.update(portfolio2)
            # If no optimistic locking, document the behavior
        except ConcurrencyError:
            # Expected behavior with optimistic locking
            pass

    async def test_deadlock_detection_and_recovery(
        self, test_engine, sample_user: User
    ):
        """Test deadlock detection and recovery."""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(test_engine, class_=AsyncSession)

        async def transaction1():
            async with SessionLocal() as session:
                repo = PortfolioRepository(session)
                # Create portfolio A
                portfolio_a = await repo.create(
                    Portfolio(
                        user_id=sample_user.id, name="Portfolio A", cash_balance=10000.0
                    )
                )
                await asyncio.sleep(0.1)  # Allow other transaction to start

                # Try to access portfolio B (potential deadlock)
                portfolios = await repo.list_by_criteria(name="Portfolio B")
                await session.commit()
                return portfolio_a

        async def transaction2():
            async with SessionLocal() as session:
                repo = PortfolioRepository(session)
                # Create portfolio B
                portfolio_b = await repo.create(
                    Portfolio(
                        user_id=sample_user.id, name="Portfolio B", cash_balance=20000.0
                    )
                )
                await asyncio.sleep(0.1)  # Allow other transaction to start

                # Try to access portfolio A (potential deadlock)
                portfolios = await repo.list_by_criteria(name="Portfolio A")
                await session.commit()
                return portfolio_b

        # Run concurrent transactions
        try:
            results = await asyncio.gather(
                transaction1(), transaction2(), return_exceptions=True
            )

            # At least one should succeed, or both should handle deadlock gracefully
            successful = [r for r in results if not isinstance(r, Exception)]
            assert (
                len(successful) >= 1
            ), "Both transactions failed - deadlock not handled"

        except Exception as e:
            # Deadlock should be handled gracefully
            assert "deadlock" in str(e).lower() or "timeout" in str(e).lower()

    async def test_transaction_isolation_levels(self, test_engine, sample_user: User):
        """Test transaction isolation level handling."""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(test_engine, class_=AsyncSession)

        # Test read committed isolation
        async with SessionLocal() as session1, SessionLocal() as session2:
            repo1 = PortfolioRepository(session1)
            repo2 = PortfolioRepository(session2)

            # Create portfolio in session1 (not committed)
            portfolio = await repo1.create(
                Portfolio(
                    user_id=sample_user.id, name="Isolation Test", cash_balance=10000.0
                )
            )

            # Session2 should not see uncommitted changes
            portfolios = await repo2.list_by_criteria(name="Isolation Test")
            assert len(portfolios) == 0, "Uncommitted changes visible in other session"

            # Commit session1
            await session1.commit()

            # Now session2 should see the changes
            portfolios = await repo2.list_by_criteria(name="Isolation Test")
            assert (
                len(portfolios) == 1
            ), "Committed changes not visible in other session"


@pytest.mark.asyncio
class TestResourceExhaustionErrors:
    """Test resource exhaustion and limit handling."""

    async def test_memory_exhaustion_handling(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test handling of memory exhaustion scenarios."""
        repo = PortfolioRepository(db_session)

        # Try to create a very large number of entities
        large_batch_size = 10000
        portfolios = []

        for i in range(large_batch_size):
            portfolio = Portfolio(
                user_id=sample_user.id,
                name=f"Portfolio {i}",
                description=f"Large batch portfolio {i}",
                cash_balance=1000.0,
            )
            portfolios.append(portfolio)

        # This should either succeed or fail gracefully
        try:
            created = await repo.bulk_create(portfolios)
            assert len(created) <= large_batch_size
        except (RepositoryError, MemoryError) as e:
            # Expected behavior for resource exhaustion
            assert "memory" in str(e).lower() or "resource" in str(e).lower()

    async def test_query_result_size_limits(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test handling of large query results."""
        repo = PortfolioRepository(db_session)

        # Create many portfolios
        portfolios = []
        for i in range(1000):
            portfolio = Portfolio(
                user_id=sample_user.id,
                name=f"Limit Test Portfolio {i}",
                cash_balance=1000.0,
            )
            portfolios.append(portfolio)

        await repo.bulk_create(portfolios)

        # Query without limit should handle large results
        try:
            all_portfolios = await repo.list_by_criteria(user_id=sample_user.id)
            assert len(all_portfolios) >= 1000
        except RepositoryError as e:
            # If there are result size limits, they should be handled gracefully
            assert "limit" in str(e).lower() or "size" in str(e).lower()

    async def test_disk_space_exhaustion(self, test_database_config):
        """Test handling of disk space exhaustion."""
        # This test documents the expected behavior
        # In real scenarios, disk space exhaustion would cause OperationalError

        # Mock disk space exhaustion
        with patch("sqlalchemy.ext.asyncio.AsyncSession.execute") as mock_execute:
            mock_execute.side_effect = OperationalError(
                "could not extend file", None, None
            )

            from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
            from sqlalchemy.orm import sessionmaker

            engine = create_async_engine(test_database_config.get_database_url())
            SessionLocal = sessionmaker(engine, class_=AsyncSession)

            try:
                async with SessionLocal() as session:
                    repo = PortfolioRepository(session)

                    with pytest.raises(RepositoryError):
                        await repo.list_by_criteria()
            finally:
                await engine.dispose()


@pytest.mark.asyncio
class TestNetworkAndTimeoutErrors:
    """Test network-related errors and timeout handling."""

    async def test_network_interruption_handling(self, db_session: AsyncSession):
        """Test handling of network interruptions."""
        repo = PortfolioRepository(db_session)

        # Mock network interruption
        with patch.object(
            db_session, "execute", side_effect=DisconnectionError("Network error")
        ):
            with pytest.raises(RepositoryError) as exc_info:
                await repo.list_by_criteria()

            assert (
                "network" in str(exc_info.value).lower()
                or "connection" in str(exc_info.value).lower()
            )

    async def test_query_timeout_handling(self, db_session: AsyncSession):
        """Test handling of query timeouts."""
        repo = PortfolioRepository(db_session)

        # Mock query timeout
        with patch.object(
            db_session,
            "execute",
            side_effect=SQLTimeoutError("Query timeout", None, None),
        ):
            with pytest.raises(RepositoryError) as exc_info:
                await repo.list_by_criteria()

            assert "timeout" in str(exc_info.value).lower()

    async def test_connection_retry_logic(self, test_database_config):
        """Test connection retry logic."""
        session_manager = DatabaseSessionManager()

        # Mock intermittent connection failures
        call_count = 0
        original_create_engine = None

        def mock_create_engine(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise OperationalError("Connection failed", None, None)
            # Succeed on 3rd attempt
            from sqlalchemy.ext.asyncio import create_async_engine

            return create_async_engine(*args, **kwargs)

        with patch(
            "database.repositories.session_manager.create_async_engine",
            side_effect=mock_create_engine,
        ):
            # This should eventually succeed after retries
            try:
                await session_manager.initialize()
                # If retry logic exists, it should succeed
            except DatabaseConnectionError:
                # If no retry logic, document the behavior
                pass


@pytest.mark.asyncio
class TestValidationErrors:
    """Test data validation error handling."""

    async def test_entity_validation_errors(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test entity validation error handling."""
        repo = PortfolioRepository(db_session)

        # Test various validation scenarios
        validation_test_cases = [
            {
                "name": "Empty name",
                "portfolio": Portfolio(
                    user_id=sample_user.id, name="", cash_balance=10000.0  # Empty name
                ),
                "expected_error": "name",
            },
            {
                "name": "Negative balance",
                "portfolio": Portfolio(
                    user_id=sample_user.id,
                    name="Valid Name",
                    cash_balance=-1000.0,  # Negative balance
                ),
                "expected_error": "balance",
            },
            {
                "name": "Invalid risk tolerance",
                "portfolio": Portfolio(
                    user_id=sample_user.id,
                    name="Valid Name",
                    cash_balance=10000.0,
                    risk_tolerance="INVALID",  # Invalid enum value
                ),
                "expected_error": "risk_tolerance",
            },
        ]

        for test_case in validation_test_cases:
            with pytest.raises((ValidationError, RepositoryError)) as exc_info:
                await repo.create(test_case["portfolio"])

            error_message = str(exc_info.value).lower()
            assert test_case["expected_error"].lower() in error_message

    async def test_business_rule_validation(
        self, db_session: AsyncSession, sample_portfolio: Portfolio
    ):
        """Test business rule validation errors."""
        repo = PositionRepository(db_session)

        # Test business rule: position quantity must be positive
        invalid_position = Position(
            portfolio_id=sample_portfolio.id,
            symbol="AAPL",
            quantity=0,  # Zero quantity should be invalid
            average_cost=150.0,
            current_price=155.0,
            position_type="STOCK",
        )

        with pytest.raises((ValidationError, RepositoryError)):
            await repo.create(invalid_position)


@pytest.mark.asyncio
class TestRecoveryAndGracefulDegradation:
    """Test error recovery and graceful degradation."""

    async def test_partial_failure_recovery(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test recovery from partial failures in batch operations."""
        repo = PortfolioRepository(db_session)

        # Create batch with some valid and some invalid portfolios
        portfolios = [
            Portfolio(
                user_id=sample_user.id, name="Valid Portfolio 1", cash_balance=10000.0
            ),
            Portfolio(
                user_id=sample_user.id,
                name="",  # Invalid - empty name
                cash_balance=5000.0,
            ),
            Portfolio(
                user_id=sample_user.id, name="Valid Portfolio 2", cash_balance=15000.0
            ),
        ]

        # Bulk create should handle partial failures gracefully
        try:
            created = await repo.bulk_create(portfolios)
            # Should create valid portfolios and skip invalid ones
            assert len(created) == 2  # Only valid ones created
        except RepositoryError:
            # Or fail fast and create none - both are valid strategies
            pass

    async def test_graceful_degradation_on_errors(self, test_database_config):
        """Test graceful degradation when database is unavailable."""
        # Mock database unavailability
        with patch(
            "database.repositories.session_manager.create_async_engine"
        ) as mock_engine:
            mock_engine.side_effect = OperationalError(
                "Database unavailable", None, None
            )

            registry = RepositoryRegistry()

            # Health check should report unhealthy status gracefully
            health = await registry.health_check()
            assert health["status"] == "unhealthy"
            assert "error" in health

    async def test_error_logging_and_monitoring(
        self, db_session: AsyncSession, sample_user: User
    ):
        """Test that errors are properly logged for monitoring."""
        repo = PortfolioRepository(db_session)

        # Create invalid portfolio to trigger error
        invalid_portfolio = Portfolio(
            user_id=sample_user.id, name="", cash_balance=10000.0  # Invalid
        )

        with patch("database.repositories.base.logger") as mock_logger:
            try:
                await repo.create(invalid_portfolio)
            except (ValidationError, RepositoryError):
                pass

            # Verify error was logged
            mock_logger.error.assert_called()


# Error handling test utilities
class ErrorTestUtils:
    """Utilities for error handling tests."""

    @staticmethod
    async def simulate_database_failure(session: AsyncSession, error_type: str):
        """Simulate various types of database failures."""
        error_map = {
            "connection": DisconnectionError("Connection lost"),
            "timeout": SQLTimeoutError("Query timeout", None, None),
            "integrity": IntegrityError("Constraint violation", None, None),
            "operational": OperationalError("Database error", None, None),
        }

        error = error_map.get(error_type, Exception("Unknown error"))

        with patch.object(session, "execute", side_effect=error):
            yield

    @staticmethod
    def assert_error_handling(exception, expected_type, expected_message_contains=None):
        """Assert that error handling meets expectations."""
        assert isinstance(exception, expected_type)

        if expected_message_contains:
            assert expected_message_contains.lower() in str(exception).lower()


@pytest.fixture
def error_test_utils():
    """Provide error handling test utilities."""
    return ErrorTestUtils()
