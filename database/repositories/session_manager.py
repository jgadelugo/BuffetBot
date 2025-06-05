"""
Database session manager for BuffetBot repository layer.

Manages database sessions, transactions, and connection pooling for async operations.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from buffetbot.utils.config import get_config
from buffetbot.utils.logger import setup_logger

from ..exceptions import DatabaseConnectionError, TransactionError

# Initialize logger
logger = setup_logger(__name__)


class DatabaseSessionManager:
    """
    Manages database sessions and transactions for async operations.

    Provides centralized session management with proper transaction handling,
    connection pooling, and error recovery.
    """

    def __init__(self, database_url: str = None, echo: bool = False):
        """
        Initialize the database session manager.

        Args:
            database_url: Database connection URL. If None, will use config.
            echo: Whether to echo SQL statements for debugging
        """
        self.database_url = database_url or self._get_database_url()
        self.echo = echo
        self._engine = None
        self._session_factory = None
        self._initialized = False

    def _get_database_url(self) -> str:
        """Get database URL from configuration or environment."""
        # For now, return a placeholder - this will be integrated with Phase 1a config
        # In a real implementation, this would read from config/environment
        return "postgresql+asyncpg://user:password@localhost/buffetbot"

    async def initialize(self):
        """Initialize the database engine and session factory."""
        if self._initialized:
            return

        try:
            # Create async engine with connection pooling
            self._engine = create_async_engine(
                self.database_url,
                echo=self.echo,
                poolclass=NullPool,  # For development, use proper pooling in production
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections after 1 hour
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False,
            )

            self._initialized = True
            logger.info("Database session manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database session manager: {e}")
            raise DatabaseConnectionError(
                "Failed to initialize database connection",
                details={"error": str(e), "database_url": self.database_url},
            )

    async def close(self):
        """Close the database engine and clean up resources."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("Database session manager closed")

    async def get_session(self) -> AsyncSession:
        """
        Get a new database session.

        Returns:
            AsyncSession: New database session

        Raises:
            DatabaseConnectionError: If session creation fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            session = self._session_factory()
            logger.debug("Created new database session")
            return session
        except Exception as e:
            logger.error(f"Failed to create database session: {e}")
            raise DatabaseConnectionError(
                "Failed to create database session", details={"error": str(e)}
            )

    async def commit_transaction(self, session: AsyncSession) -> None:
        """
        Commit a database transaction.

        Args:
            session: Database session to commit

        Raises:
            TransactionError: If commit fails
        """
        try:
            await session.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            logger.error(f"Failed to commit transaction: {e}")
            await self.rollback_transaction(session)
            raise TransactionError(
                "Failed to commit transaction",
                operation="commit",
                details={"error": str(e)},
            )

    async def rollback_transaction(self, session: AsyncSession) -> None:
        """
        Rollback a database transaction.

        Args:
            session: Database session to rollback
        """
        try:
            await session.rollback()
            logger.debug("Transaction rolled back successfully")
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
            # Don't raise exception here as this is typically called during error handling

    async def close_session(self, session: AsyncSession) -> None:
        """
        Close a database session.

        Args:
            session: Database session to close
        """
        try:
            await session.close()
            logger.debug("Database session closed")
        except Exception as e:
            logger.error(f"Failed to close session: {e}")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for database transactions.

        Automatically handles commit/rollback and session cleanup.

        Yields:
            AsyncSession: Database session within transaction

        Raises:
            TransactionError: If transaction fails
        """
        session = await self.get_session()

        try:
            logger.debug("Starting database transaction")
            yield session
            await self.commit_transaction(session)
            logger.debug("Database transaction completed successfully")
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            await self.rollback_transaction(session)
            raise TransactionError(
                "Database transaction failed",
                operation="transaction",
                details={"error": str(e)},
            )
        finally:
            await self.close_session(session)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for database sessions without automatic transactions.

        Yields:
            AsyncSession: Database session
        """
        session = await self.get_session()

        try:
            logger.debug("Starting database session")
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await self.rollback_transaction(session)
            raise
        finally:
            await self.close_session(session)

    async def health_check(self) -> dict:
        """
        Check database connection health.

        Returns:
            dict: Health check results
        """
        try:
            async with self.session() as session:
                # Simple query to test connection
                result = await session.execute("SELECT 1")
                result.scalar()

                return {
                    "status": "healthy",
                    "database_url": self.database_url,
                    "initialized": self._initialized,
                    "timestamp": asyncio.get_event_loop().time(),
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self.database_url,
                "initialized": self._initialized,
                "timestamp": asyncio.get_event_loop().time(),
            }


# Global session manager instance
_session_manager: Optional[DatabaseSessionManager] = None


def get_session_manager() -> DatabaseSessionManager:
    """
    Get the global database session manager instance.

    Returns:
        DatabaseSessionManager: Global session manager
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = DatabaseSessionManager()
    return _session_manager


async def init_database():
    """Initialize the global database session manager."""
    session_manager = get_session_manager()
    await session_manager.initialize()


async def close_database():
    """Close the global database session manager."""
    global _session_manager
    if _session_manager:
        await _session_manager.close()
        _session_manager = None
