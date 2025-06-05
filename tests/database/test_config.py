"""
Tests for database configuration management.
"""

import os
from unittest.mock import patch

import pytest

from database.config import (
    DatabaseConfig,
    DatabaseEnvironment,
    get_database_config,
    get_test_database_config,
)


class TestDatabaseConfig:
    """Test database configuration functionality."""

    def test_default_configuration(self):
        """Test default configuration values."""
        with patch.dict(
            os.environ,
            {
                "DB_USERNAME": "test_user",
                "DB_PASSWORD": "test_pass",
                "DB_NAME": "test_db",
            },
            clear=True,
        ):
            config = DatabaseConfig()

            assert config.host == "localhost"
            assert config.port == 5432
            assert config.username == "test_user"
            assert config.password == "test_pass"
            assert config.database == "test_db"
            assert config.pool_size == 5
            assert config.environment == DatabaseEnvironment.DEVELOPMENT

    def test_environment_specific_configuration(self):
        """Test environment-specific configuration."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "production",
                "DB_USERNAME": "prod_user",
                "DB_PASSWORD": "prod_pass",
                "DB_NAME": "prod_db",
                "DB_POOL_SIZE": "20",
                "DB_ECHO_SQL": "false",
            },
            clear=True,
        ):
            config = DatabaseConfig()

            assert config.environment == DatabaseEnvironment.PRODUCTION
            assert config.pool_size == 20
            assert config.echo_sql is False
            assert config.is_production is True
            assert config.is_development is False

    def test_database_url_generation(self):
        """Test database URL generation for different drivers."""
        with patch.dict(
            os.environ,
            {
                "DB_USERNAME": "user",
                "DB_PASSWORD": "pass",
                "DB_HOST": "localhost",
                "DB_PORT": "5432",
                "DB_NAME": "testdb",
            },
            clear=True,
        ):
            config = DatabaseConfig()

            # Test async URL
            async_url = config.get_database_url(async_driver=True)
            assert async_url == "postgresql+asyncpg://user:pass@localhost:5432/testdb"

            # Test sync URL
            sync_url = config.get_database_url(async_driver=False)
            assert sync_url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"

            # Test migration URL
            migration_url = config.get_migration_url()
            assert migration_url == sync_url

    def test_engine_kwargs(self):
        """Test SQLAlchemy engine configuration."""
        with patch.dict(
            os.environ,
            {
                "DB_USERNAME": "user",
                "DB_PASSWORD": "pass",
                "DB_NAME": "testdb",
                "DB_POOL_SIZE": "10",
                "DB_POOL_MAX_OVERFLOW": "5",
                "DB_POOL_TIMEOUT": "60",
                "DB_POOL_RECYCLE": "7200",
                "ENVIRONMENT": "development",
                "DB_ECHO_SQL": "true",
            },
            clear=True,
        ):
            config = DatabaseConfig()
            kwargs = config.engine_kwargs

            assert kwargs["pool_size"] == 10
            assert kwargs["max_overflow"] == 5
            assert kwargs["pool_timeout"] == 60
            assert kwargs["pool_recycle"] == 7200
            assert kwargs["pool_pre_ping"] is True
            assert kwargs["echo"] is True  # Development environment with echo enabled

    def test_ssl_configuration(self):
        """Test SSL configuration handling."""
        with patch.dict(
            os.environ,
            {
                "DB_USERNAME": "user",
                "DB_PASSWORD": "pass",
                "DB_NAME": "testdb",
                "DB_SSL_MODE": "require",
                "DB_SSL_CERT": "/path/to/cert.pem",
                "DB_SSL_KEY": "/path/to/key.pem",
                "DB_SSL_CA": "/path/to/ca.pem",
            },
            clear=True,
        ):
            config = DatabaseConfig()
            kwargs = config.engine_kwargs

            assert "connect_args" in kwargs
            connect_args = kwargs["connect_args"]
            assert connect_args["sslmode"] == "require"
            assert connect_args["sslcert"] == "/path/to/cert.pem"
            assert connect_args["sslkey"] == "/path/to/key.pem"
            assert connect_args["sslca"] == "/path/to/ca.pem"

    def test_validation_errors(self):
        """Test configuration validation."""
        # Test invalid port
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            with patch.dict(
                os.environ,
                {
                    "DB_USERNAME": "user",
                    "DB_PASSWORD": "pass",
                    "DB_NAME": "testdb",
                    "DB_PORT": "70000",
                },
                clear=True,
            ):
                DatabaseConfig()

        # Test invalid pool size
        with pytest.raises(ValueError, match="Pool size must be positive"):
            with patch.dict(
                os.environ,
                {
                    "DB_USERNAME": "user",
                    "DB_PASSWORD": "pass",
                    "DB_NAME": "testdb",
                    "DB_POOL_SIZE": "0",
                },
                clear=True,
            ):
                DatabaseConfig()

    def test_environment_validation(self):
        """Test environment validation and fallback."""
        with patch.dict(
            os.environ,
            {
                "DB_USERNAME": "user",
                "DB_PASSWORD": "pass",
                "DB_NAME": "testdb",
                "ENVIRONMENT": "invalid_env",
            },
            clear=True,
        ):
            config = DatabaseConfig()
            # Should fallback to development for invalid environment
            assert config.environment == DatabaseEnvironment.DEVELOPMENT

    def test_get_database_config_function(self):
        """Test the get_database_config convenience function."""
        with patch.dict(
            os.environ,
            {"DB_USERNAME": "user", "DB_PASSWORD": "pass", "DB_NAME": "testdb"},
            clear=True,
        ):
            config = get_database_config()
            assert isinstance(config, DatabaseConfig)
            assert config.username == "user"

    def test_get_test_database_config_function(self):
        """Test the get_test_database_config convenience function."""
        with patch.dict(
            os.environ,
            {"DB_USERNAME": "user", "DB_PASSWORD": "pass", "DB_NAME": "testdb"},
            clear=True,
        ):
            config = get_test_database_config()
            assert isinstance(config, DatabaseConfig)
            assert config.environment == DatabaseEnvironment.TESTING
            assert config.database == "testdb_test"
            assert config.pool_size == 2
            assert config.echo_sql is False


class TestDatabaseEnvironment:
    """Test database environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert DatabaseEnvironment.DEVELOPMENT == "development"
        assert DatabaseEnvironment.TESTING == "testing"
        assert DatabaseEnvironment.STAGING == "staging"
        assert DatabaseEnvironment.PRODUCTION == "production"

    def test_environment_comparison(self):
        """Test environment comparison."""
        dev_env = DatabaseEnvironment.DEVELOPMENT
        prod_env = DatabaseEnvironment.PRODUCTION

        assert dev_env != prod_env
        assert dev_env == "development"
        assert prod_env == "production"
