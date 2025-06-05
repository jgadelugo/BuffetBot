"""
Database connection management for BuffetBot.

Provides async database connection management, session handling, and
database initialization using PostgreSQL and SQLAlchemy.
"""

import logging
import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Database:
    """
    Database connection manager with support for both sync and async operations.
    """

    def __init__(
        self, database_url: Optional[str] = None, async_url: Optional[str] = None
    ):
        """
        Initialize database connection manager.

        Args:
            database_url: Synchronous database URL for SQLAlchemy
            async_url: Asynchronous database URL for async operations
        """
        # Get database URLs from environment or use defaults
        self.database_url = database_url or self._get_database_url()
        self.async_url = async_url or self._get_async_database_url()

        # Initialize engines
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None

        # Connection pool configuration
        self.pool_config = {
            "pool_size": 20,
            "max_overflow": 30,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # 1 hour
        }

    def _get_database_url(self) -> str:
        """Get synchronous database URL from environment variables."""
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "buffetbot")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")

        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def _get_async_database_url(self) -> str:
        """Get asynchronous database URL from environment variables."""
        # Convert sync URL to async URL
        sync_url = self._get_database_url()
        return sync_url.replace("postgresql://", "postgresql+asyncpg://")

    def initialize(self) -> None:
        """Initialize database engines and session factories."""
        try:
            # Create synchronous engine
            self.engine = create_engine(
                self.database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                **self.pool_config,
            )

            # Create async engine
            self.async_engine = create_async_engine(
                self.async_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                **self.pool_config,
            )

            # Create session factories
            self.session_factory = sessionmaker(
                bind=self.engine, autocommit=False, autoflush=False
            )

            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )

            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a synchronous database session with automatic cleanup.

        Yields:
            Session: SQLAlchemy session
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an asynchronous database session with automatic cleanup.

        Yields:
            AsyncSession: SQLAlchemy async session
        """
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {str(e)}")
            raise
        finally:
            await session.close()

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.async_engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        from .models import Base  # Import here to avoid circular imports

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created successfully")

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self.async_engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        from .models import Base  # Import here to avoid circular imports

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        logger.info("Database tables dropped successfully")

    async def check_connection(self) -> bool:
        """
        Check if database connection is working.

        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False

    def close(self) -> None:
        """Close all database connections."""
        if self.engine:
            self.engine.dispose()
        if self.async_engine:
            self.async_engine.dispose()
        logger.info("Database connections closed")


# Global database instance
_database = None


def init_database(
    database_url: Optional[str] = None, async_url: Optional[str] = None
) -> Database:
    """
    Initialize the global database instance.

    Args:
        database_url: Optional synchronous database URL
        async_url: Optional asynchronous database URL

    Returns:
        Database: Initialized database instance
    """
    global _database

    if _database is None:
        _database = Database(database_url, async_url)
        _database.initialize()

    return _database


def get_database() -> Database:
    """
    Get the global database instance.

    Returns:
        Database: Database instance

    Raises:
        RuntimeError: If database is not initialized
    """
    global _database

    if _database is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    return _database


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """
    Convenience function to get a database session.

    Yields:
        Session: SQLAlchemy session
    """
    db = get_database()
    with db.get_session() as session:
        yield session


@asynccontextmanager
async def get_async_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function to get an async database session.

    Yields:
        AsyncSession: SQLAlchemy async session
    """
    db = get_database()
    async with db.get_async_session() as session:
        yield session
