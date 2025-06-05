"""
Database initialization and lifecycle management for BuffetBot.

Handles database initialization, schema creation, health checks, and seeding.
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from .config import DatabaseConfig, DatabaseEnvironment
from .connection import Base

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization, schema creation, and seeding."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database initializer.

        Args:
            config: Database configuration. If None, uses default config.
        """
        self.config = config or DatabaseConfig()
        self.engine = create_async_engine(
            self.config.get_database_url(), **self.config.engine_kwargs
        )
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """Create all database tables."""
        logger.info("Creating database tables...")
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as exc:
            logger.error(f"Failed to create database tables: {exc}")
            raise

    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        if self.config.is_production:
            raise RuntimeError("Cannot drop tables in production environment")

        logger.warning("Dropping all database tables...")
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped")
        except Exception as exc:
            logger.error(f"Failed to drop database tables: {exc}")
            raise

    async def check_database_health(self) -> bool:
        """
        Check if database is accessible and healthy.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as health_check"))
                health_value = result.scalar()
                if health_value == 1:
                    logger.info("Database health check passed")
                    return True

        except Exception as exc:
            logger.error(f"Database health check failed: {exc}")

        return False

    async def check_tables_exist(self) -> bool:
        """
        Check if required tables exist in the database.

        Returns:
            True if all required tables exist, False otherwise
        """
        try:
            async with self.engine.begin() as conn:
                # Check if tables exist by querying information_schema
                result = await conn.execute(
                    text(
                        """
                    SELECT COUNT(*) as table_count
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    AND table_name IN ('portfolios', 'positions', 'analysis_results', 'market_data_cache', 'users')
                """
                    )
                )
                table_count = result.scalar()

                # We expect at least 4 core tables (portfolios, positions, analysis_results, market_data_cache)
                if table_count >= 4:
                    logger.info(f"Found {table_count} required database tables")
                    return True
                else:
                    logger.warning(f"Only found {table_count} required database tables")
                    return False

        except Exception as exc:
            logger.error(f"Failed to check table existence: {exc}")
            return False

    async def seed_development_data(self) -> None:
        """Seed database with development data."""
        if not self.config.is_development:
            logger.warning("Skipping data seeding - not in development environment")
            return

        logger.info("Seeding development data...")
        try:
            from .seeds import create_sample_data

            async with self.SessionLocal() as session:
                await create_sample_data(session)
                await session.commit()

            logger.info("Development data seeding completed")

        except ImportError:
            logger.warning("Seeding module not found, skipping data seeding")
        except Exception as exc:
            logger.error(f"Failed to seed development data: {exc}")
            raise

    async def initialize_database(
        self,
        create_schema: bool = False,
        seed_data: bool = False,
        force_recreate: bool = False,
    ) -> None:
        """
        Complete database initialization process.

        Args:
            create_schema: Whether to create database schema
            seed_data: Whether to seed development data
            force_recreate: Whether to force recreation of tables
        """
        logger.info("Starting database initialization...")

        try:
            # Check database connectivity
            if not await self.check_database_health():
                raise RuntimeError("Cannot connect to database")

            # Handle schema creation
            if force_recreate and not self.config.is_production:
                logger.info("Force recreating database schema...")
                await self.drop_tables()
                await self.create_tables()
            elif create_schema:
                if self.config.is_production:
                    logger.warning(
                        "Schema creation skipped in production - use migrations instead"
                    )
                else:
                    tables_exist = await self.check_tables_exist()
                    if not tables_exist:
                        await self.create_tables()
                    else:
                        logger.info("Database tables already exist")

            # Seed development data if requested
            if seed_data:
                await self.seed_development_data()

            logger.info("Database initialization completed successfully")

        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            raise

    async def get_database_info(self) -> dict:
        """
        Get information about the database.

        Returns:
            Dictionary containing database information
        """
        try:
            async with self.engine.begin() as conn:
                # Get PostgreSQL version
                version_result = await conn.execute(text("SELECT version()"))
                version = version_result.scalar()

                # Get database size
                size_result = await conn.execute(
                    text(
                        """
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size
                """
                    )
                )
                size = size_result.scalar()

                # Get table count
                table_result = await conn.execute(
                    text(
                        """
                    SELECT COUNT(*) as table_count
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                """
                    )
                )
                table_count = table_result.scalar()

                # Get connection count
                conn_result = await conn.execute(
                    text(
                        """
                    SELECT COUNT(*) as connection_count
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """
                    )
                )
                connection_count = conn_result.scalar()

                return {
                    "database_name": self.config.database,
                    "host": self.config.host,
                    "port": self.config.port,
                    "environment": self.config.environment.value,
                    "version": version,
                    "size": size,
                    "table_count": table_count,
                    "active_connections": connection_count,
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                }

        except Exception as exc:
            logger.error(f"Failed to get database info: {exc}")
            return {
                "error": str(exc),
                "database_name": self.config.database,
                "host": self.config.host,
                "port": self.config.port,
                "environment": self.config.environment.value,
            }

    async def close(self) -> None:
        """Close database connections."""
        try:
            await self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as exc:
            logger.error(f"Error closing database connections: {exc}")


async def initialize_database(
    config: Optional[DatabaseConfig] = None,
    create_schema: bool = False,
    seed_data: bool = False,
) -> DatabaseInitializer:
    """
    Convenience function to initialize database.

    Args:
        config: Database configuration
        create_schema: Whether to create database schema
        seed_data: Whether to seed development data

    Returns:
        DatabaseInitializer instance
    """
    initializer = DatabaseInitializer(config)
    await initializer.initialize_database(
        create_schema=create_schema, seed_data=seed_data
    )
    return initializer


async def check_database_health(config: Optional[DatabaseConfig] = None) -> bool:
    """
    Convenience function to check database health.

    Args:
        config: Database configuration

    Returns:
        True if database is healthy, False otherwise
    """
    initializer = DatabaseInitializer(config)
    try:
        return await initializer.check_database_health()
    finally:
        await initializer.close()
