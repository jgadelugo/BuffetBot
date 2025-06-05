#!/usr/bin/env python3
"""Database Initialization System Test"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.config import DatabaseConfig
from database.initialization import DatabaseInitializer


async def test_database_initializer():
    """Test database initializer with mocked dependencies."""
    print("Testing DatabaseInitializer...")

    # Create test configuration
    config = DatabaseConfig(
        username="test_user", password="test_pass", database="test_db"
    )

    with patch("database.initialization.create_async_engine") as mock_engine, patch(
        "database.initialization.sessionmaker"
    ) as mock_sessionmaker:
        # Setup mocks
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance

        mock_session_class = AsyncMock()
        mock_sessionmaker.return_value = mock_session_class

        # Create initializer
        initializer = DatabaseInitializer(config)

        # Test that engine was created with correct parameters
        mock_engine.assert_called_once()
        call_args = mock_engine.call_args

        # Verify database URL
        assert call_args[0][0] == config.get_database_url()

        # Verify engine kwargs are passed
        kwargs = call_args[1]
        assert "pool_size" in kwargs
        assert kwargs["pool_size"] == config.pool_size

        print("✅ DatabaseInitializer created successfully")
        print(f"   Database URL: {config.get_database_url()}")
        print(f"   Engine kwargs: {config.engine_kwargs}")

        return initializer


async def test_health_check():
    """Test database health check functionality."""
    print("\nTesting health check...")

    config = DatabaseConfig(
        username="test_user", password="test_pass", database="test_db"
    )

    with patch("database.initialization.create_async_engine") as mock_engine:
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance

        # Mock successful connection
        mock_conn = AsyncMock()
        mock_engine_instance.begin.return_value.__aenter__.return_value = mock_conn

        initializer = DatabaseInitializer(config)

        # Test successful health check
        result = await initializer.check_database_health()
        assert result is True
        print("✅ Health check passed")

        # Test failed health check
        mock_engine_instance.begin.side_effect = Exception("Connection failed")
        result = await initializer.check_database_health()
        assert result is False
        print("✅ Health check failure handled correctly")


async def test_convenience_functions():
    """Test convenience functions."""
    print("\nTesting convenience functions...")

    with patch("database.initialization.DatabaseInitializer") as mock_init_class:
        mock_initializer = AsyncMock()
        mock_init_class.return_value = mock_initializer

        # Import and test convenience functions
        from database.initialization import (
            check_database_health,
            create_database_tables,
            initialize_database,
            seed_development_data,
        )

        # Test initialize_database
        await initialize_database()
        mock_init_class.assert_called()
        mock_initializer.initialize_database.assert_called_once()
        print("✅ initialize_database convenience function works")

        # Test check_database_health
        mock_initializer.reset_mock()
        mock_init_class.reset_mock()

        await check_database_health()
        mock_init_class.assert_called()
        mock_initializer.check_database_health.assert_called_once()
        print("✅ check_database_health convenience function works")


def main():
    """Run all database initialization tests."""
    print("🔧 BuffetBot Database Initialization Test")
    print("=" * 45)

    try:
        import asyncio

        # Run async tests
        asyncio.run(test_database_initializer())
        asyncio.run(test_health_check())
        asyncio.run(test_convenience_functions())

        print("\n🎉 All database initialization tests passed!")
        return 0

    except Exception as e:
        print(f"\n💥 Database initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
