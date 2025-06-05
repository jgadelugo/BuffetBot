#!/usr/bin/env python3
"""
Database Configuration Verification Script

Verifies that the database configuration system is properly configured
and can create valid database connections with different environments.
"""

import os
import sys
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.config import DatabaseConfig, DatabaseEnvironment


def test_configuration_creation() -> None:
    """Test basic configuration creation with minimal parameters."""
    print("Testing configuration creation...")

    # Set minimal required environment variables
    test_env = {
        "DB_USERNAME": "test_user",
        "DB_PASSWORD": "test_pass",
        "DB_NAME": "test_db",
    }

    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Force reload of the configuration with new environment
        config = DatabaseConfig(_env_file=None)  # Don't load from file, use env vars
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert config.database == "test_db"
        assert config.host == "localhost"  # default
        assert config.port == 5432  # default
        print("✅ Basic configuration creation successful")
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        raise
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_environment_validation() -> None:
    """Test environment validation and enum conversion."""
    print("Testing environment validation...")

    test_cases = [
        ("development", DatabaseEnvironment.DEVELOPMENT),
        ("PRODUCTION", DatabaseEnvironment.PRODUCTION),
        ("Testing", DatabaseEnvironment.TESTING),
        ("staging", DatabaseEnvironment.STAGING),
    ]

    for env_str, expected_enum in test_cases:
        # Set test environment
        os.environ.update(
            {
                "DB_USERNAME": "test",
                "DB_PASSWORD": "test",
                "DB_NAME": "test",
                "ENVIRONMENT": env_str,
            }
        )

        try:
            config = DatabaseConfig()
            assert config.environment == expected_enum
            print(f"✅ Environment '{env_str}' -> {expected_enum.value}")
        except Exception as e:
            print(f"❌ Environment validation failed for '{env_str}': {e}")
            raise
        finally:
            # Clean up
            for key in ["DB_USERNAME", "DB_PASSWORD", "DB_NAME", "ENVIRONMENT"]:
                os.environ.pop(key, None)


def test_database_url_generation() -> None:
    """Test database URL generation for different scenarios."""
    print("Testing database URL generation...")

    # Set test environment
    os.environ.update(
        {
            "DB_USERNAME": "myuser",
            "DB_PASSWORD": "mypass",
            "DB_NAME": "mydb",
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
        }
    )

    try:
        config = DatabaseConfig()

        # Test async URL
        async_url = config.get_database_url(async_driver=True)
        expected_async = "postgresql+asyncpg://myuser:mypass@localhost:5432/mydb"
        assert async_url == expected_async
        print(f"✅ Async URL: {async_url}")

        # Test sync URL
        sync_url = config.get_database_url(async_driver=False)
        expected_sync = "postgresql+psycopg2://myuser:mypass@localhost:5432/mydb"
        assert sync_url == expected_sync
        print(f"✅ Sync URL: {sync_url}")

        # Test migration URL
        migration_url = config.get_migration_url()
        assert migration_url == expected_sync
        print(f"✅ Migration URL: {migration_url}")

    except Exception as e:
        print(f"❌ URL generation failed: {e}")
        raise
    finally:
        # Clean up
        for key in ["DB_USERNAME", "DB_PASSWORD", "DB_NAME", "DB_HOST", "DB_PORT"]:
            os.environ.pop(key, None)


def test_engine_kwargs() -> None:
    """Test SQLAlchemy engine kwargs generation."""
    print("Testing engine kwargs generation...")

    # Set test environment
    os.environ.update(
        {
            "DB_USERNAME": "test",
            "DB_PASSWORD": "test",
            "DB_NAME": "test",
            "DB_POOL_SIZE": "10",
            "DB_POOL_MAX_OVERFLOW": "5",
            "DB_POOL_TIMEOUT": "60",
            "DB_ECHO_SQL": "true",
            "ENVIRONMENT": "development",
        }
    )

    try:
        config = DatabaseConfig()
        kwargs = config.engine_kwargs

        expected_keys = {
            "pool_size",
            "max_overflow",
            "pool_timeout",
            "pool_recycle",
            "pool_pre_ping",
            "echo",
        }

        assert all(key in kwargs for key in expected_keys)
        assert kwargs["pool_size"] == 10
        assert kwargs["max_overflow"] == 5
        assert kwargs["pool_timeout"] == 60
        assert (
            kwargs["echo"] is True
        )  # Should be True in development with echo_sql=true
        assert kwargs["pool_pre_ping"] is True

        print(f"✅ Engine kwargs: {kwargs}")

    except Exception as e:
        print(f"❌ Engine kwargs generation failed: {e}")
        raise
    finally:
        # Clean up
        for key in [
            "DB_USERNAME",
            "DB_PASSWORD",
            "DB_NAME",
            "DB_POOL_SIZE",
            "DB_POOL_MAX_OVERFLOW",
            "DB_POOL_TIMEOUT",
            "DB_ECHO_SQL",
            "ENVIRONMENT",
        ]:
            os.environ.pop(key, None)


def test_validation_errors() -> None:
    """Test that validation errors are properly raised."""
    print("Testing validation errors...")

    # Test invalid port
    os.environ.update(
        {
            "DB_USERNAME": "test",
            "DB_PASSWORD": "test",
            "DB_NAME": "test",
            "DB_PORT": "99999",  # Invalid port
        }
    )

    try:
        DatabaseConfig()
        print("❌ Should have raised validation error for invalid port")
        assert False, "Expected validation error"
    except ValueError as e:
        print(f"✅ Correctly caught port validation error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        raise
    finally:
        # Clean up
        for key in ["DB_USERNAME", "DB_PASSWORD", "DB_NAME", "DB_PORT"]:
            os.environ.pop(key, None)

    # Test invalid pool size
    os.environ.update(
        {
            "DB_USERNAME": "test",
            "DB_PASSWORD": "test",
            "DB_NAME": "test",
            "DB_POOL_SIZE": "0",  # Invalid pool size
        }
    )

    try:
        DatabaseConfig()
        print("❌ Should have raised validation error for invalid pool size")
        assert False, "Expected validation error"
    except ValueError as e:
        print(f"✅ Correctly caught pool size validation error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        raise
    finally:
        # Clean up
        for key in ["DB_USERNAME", "DB_PASSWORD", "DB_NAME", "DB_POOL_SIZE"]:
            os.environ.pop(key, None)


def main():
    """Run all configuration verification tests."""
    print("🔧 BuffetBot Database Configuration Verification")
    print("=" * 50)

    try:
        test_configuration_creation()
        print()

        test_environment_validation()
        print()

        test_database_url_generation()
        print()

        test_engine_kwargs()
        print()

        test_validation_errors()
        print()

        print("🎉 All configuration tests passed!")
        return 0

    except Exception as e:
        print(f"\n💥 Configuration verification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
