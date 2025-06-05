"""
Tests for database initialization and lifecycle management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from database.config import DatabaseConfig, DatabaseEnvironment
from database.initialization import (
    DatabaseInitializer,
    check_database_health,
    initialize_database,
)


@pytest.fixture
def mock_config():
    """Mock database configuration for testing."""
    config = MagicMock(spec=DatabaseConfig)
    config.environment = DatabaseEnvironment.TESTING
    config.is_production = False
    config.is_development = False
    config.is_testing = True
    config.get_database_url.return_value = (
        "postgresql+asyncpg://test:test@localhost:5432/test_db"
    )
    config.engine_kwargs = {
        "pool_size": 2,
        "max_overflow": 1,
        "pool_timeout": 10,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
        "echo": False,
    }
    return config


@pytest.fixture
def mock_engine():
    """Mock async database engine."""
    engine = AsyncMock()
    engine.begin = AsyncMock()
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def mock_session():
    """Mock async database session."""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


class TestDatabaseInitializer:
    """Test database initializer functionality."""

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    def test_initializer_creation(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test database initializer creation."""
        mock_create_engine.return_value = mock_engine

        initializer = DatabaseInitializer(mock_config)

        assert initializer.config == mock_config
        mock_create_engine.assert_called_once_with(
            mock_config.get_database_url(), **mock_config.engine_kwargs
        )
        mock_sessionmaker.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_create_tables(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test table creation."""
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        initializer = DatabaseInitializer(mock_config)
        await initializer.create_tables()

        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_drop_tables_development(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test table dropping in development environment."""
        mock_config.is_production = False
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        initializer = DatabaseInitializer(mock_config)
        await initializer.drop_tables()

        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_drop_tables_production_blocked(
        self, mock_sessionmaker, mock_create_engine, mock_config
    ):
        """Test that table dropping is blocked in production."""
        mock_config.is_production = True

        initializer = DatabaseInitializer(mock_config)

        with pytest.raises(
            RuntimeError, match="Cannot drop tables in production environment"
        ):
            await initializer.drop_tables()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_check_database_health_success(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test successful database health check."""
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result

        initializer = DatabaseInitializer(mock_config)
        is_healthy = await initializer.check_database_health()

        assert is_healthy is True
        mock_engine.begin.assert_called_once()
        mock_conn.execute.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_check_database_health_failure(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test failed database health check."""
        mock_create_engine.return_value = mock_engine
        mock_engine.begin.side_effect = Exception("Connection failed")

        initializer = DatabaseInitializer(mock_config)
        is_healthy = await initializer.check_database_health()

        assert is_healthy is False

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_check_tables_exist(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test checking if required tables exist."""
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 4  # 4 required tables found
        mock_conn.execute.return_value = mock_result

        initializer = DatabaseInitializer(mock_config)
        tables_exist = await initializer.check_tables_exist()

        assert tables_exist is True
        mock_conn.execute.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_seed_development_data_success(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_session
    ):
        """Test successful development data seeding."""
        mock_config.is_development = True
        mock_create_engine.return_value = mock_engine
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory

        with patch(
            "database.initialization.create_sample_data"
        ) as mock_create_sample_data:
            initializer = DatabaseInitializer(mock_config)
            await initializer.seed_development_data()

            mock_create_sample_data.assert_called_once_with(mock_session)
            mock_session.commit.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_seed_development_data_non_development(
        self, mock_sessionmaker, mock_create_engine, mock_config
    ):
        """Test that seeding is skipped in non-development environments."""
        mock_config.is_development = False

        initializer = DatabaseInitializer(mock_config)
        await initializer.seed_development_data()

        # Should complete without error but not actually seed data

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_initialize_database_full_flow(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test complete database initialization flow."""
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Mock health check success
        mock_result = AsyncMock()
        mock_result.scalar.return_value = 1
        mock_conn.execute.return_value = mock_result

        initializer = DatabaseInitializer(mock_config)

        with patch.object(
            initializer, "check_tables_exist", return_value=False
        ) as mock_check_tables, patch.object(
            initializer, "create_tables"
        ) as mock_create_tables, patch.object(
            initializer, "seed_development_data"
        ) as mock_seed_data:
            await initializer.initialize_database(
                create_schema=True, seed_data=True, force_recreate=False
            )

            mock_check_tables.assert_called_once()
            mock_create_tables.assert_called_once()
            mock_seed_data.assert_called_once()

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_get_database_info(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test getting database information."""
        mock_create_engine.return_value = mock_engine
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        # Mock multiple query results
        mock_results = [
            AsyncMock(scalar=lambda: "PostgreSQL 14.0"),  # version
            AsyncMock(scalar=lambda: "100 MB"),  # size
            AsyncMock(scalar=lambda: 5),  # table count
            AsyncMock(scalar=lambda: 3),  # connection count
        ]
        mock_conn.execute.side_effect = mock_results

        initializer = DatabaseInitializer(mock_config)
        info = await initializer.get_database_info()

        assert "version" in info
        assert "size" in info
        assert "table_count" in info
        assert "active_connections" in info
        assert info["environment"] == mock_config.environment.value

    @patch("database.initialization.create_async_engine")
    @patch("database.initialization.sessionmaker")
    async def test_close(
        self, mock_sessionmaker, mock_create_engine, mock_config, mock_engine
    ):
        """Test closing database connections."""
        mock_create_engine.return_value = mock_engine

        initializer = DatabaseInitializer(mock_config)
        await initializer.close()

        mock_engine.dispose.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for database operations."""

    @patch("database.initialization.DatabaseInitializer")
    async def test_initialize_database_function(self, mock_initializer_class):
        """Test the initialize_database convenience function."""
        mock_initializer = AsyncMock()
        mock_initializer_class.return_value = mock_initializer

        result = await initialize_database(
            config=None, create_schema=True, seed_data=True
        )

        mock_initializer_class.assert_called_once_with(None)
        mock_initializer.initialize_database.assert_called_once_with(
            create_schema=True, seed_data=True
        )
        assert result == mock_initializer

    @patch("database.initialization.DatabaseInitializer")
    async def test_check_database_health_function(self, mock_initializer_class):
        """Test the check_database_health convenience function."""
        mock_initializer = AsyncMock()
        mock_initializer.check_database_health.return_value = True
        mock_initializer_class.return_value = mock_initializer

        is_healthy = await check_database_health(config=None)

        assert is_healthy is True
        mock_initializer_class.assert_called_once_with(None)
        mock_initializer.check_database_health.assert_called_once()
        mock_initializer.close.assert_called_once()
