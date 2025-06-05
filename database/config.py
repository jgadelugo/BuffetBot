"""
Database configuration management for BuffetBot.

Provides comprehensive database configuration with environment-specific settings,
connection pooling, security options, and validation.
"""

import os
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseEnvironment(str, Enum):
    """Database environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """Database configuration with environment-specific settings."""

    # Core database settings
    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    username: str = Field(alias="DB_USERNAME")
    password: str = Field(alias="DB_PASSWORD")
    database: str = Field(alias="DB_NAME")

    # Connection pool settings
    pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=0, alias="DB_POOL_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, alias="DB_POOL_RECYCLE")

    # Environment-specific settings
    environment: DatabaseEnvironment = Field(
        default=DatabaseEnvironment.DEVELOPMENT, alias="ENVIRONMENT"
    )
    echo_sql: bool = Field(default=False, alias="DB_ECHO_SQL")

    # SSL and security settings
    ssl_mode: str = Field(default="prefer", alias="DB_SSL_MODE")
    ssl_cert: Optional[str] = Field(default=None, alias="DB_SSL_CERT")
    ssl_key: Optional[str] = Field(default=None, alias="DB_SSL_KEY")
    ssl_ca: Optional[str] = Field(default=None, alias="DB_SSL_CA")

    # Migration settings
    migration_timeout: int = Field(default=300, alias="DB_MIGRATION_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # This allows using aliases for environment variables
        env_prefix="",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            try:
                return DatabaseEnvironment(v.lower())
            except ValueError:
                return DatabaseEnvironment.DEVELOPMENT
        return v

    @field_validator("pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        """Validate pool size is positive."""
        if v <= 0:
            raise ValueError("Pool size must be positive")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    def get_database_url(self, async_driver: bool = True) -> str:
        """
        Generate database URL for SQLAlchemy.

        Args:
            async_driver: Whether to use async driver (asyncpg) or sync (psycopg2)

        Returns:
            Database URL string
        """
        driver = "postgresql+asyncpg" if async_driver else "postgresql+psycopg2"
        return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_migration_url(self) -> str:
        """Generate synchronous database URL for Alembic migrations."""
        return self.get_database_url(async_driver=False)

    @property
    def engine_kwargs(self) -> dict[str, Any]:
        """Get SQLAlchemy engine configuration."""
        kwargs = {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": True,  # Enable connection health checks
            "echo": self.echo_sql
            and self.environment == DatabaseEnvironment.DEVELOPMENT,
        }

        # Add SSL configuration if specified
        if self.ssl_mode != "disable":
            connect_args = {"sslmode": self.ssl_mode}
            if self.ssl_cert:
                connect_args.update(
                    {
                        "sslcert": self.ssl_cert,
                        "sslkey": self.ssl_key,
                        "sslca": self.ssl_ca,
                    }
                )
            kwargs["connect_args"] = connect_args

        return kwargs

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == DatabaseEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == DatabaseEnvironment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == DatabaseEnvironment.TESTING


def get_database_config() -> DatabaseConfig:
    """
    Get database configuration instance.

    Returns:
        DatabaseConfig instance with current environment settings
    """
    return DatabaseConfig()


def get_test_database_config() -> DatabaseConfig:
    """
    Get database configuration for testing.

    Returns:
        DatabaseConfig instance configured for testing
    """
    # Get current config first to extract database name
    current_config = DatabaseConfig()

    # Create test config by modifying environment variables temporarily
    import os

    original_env = os.environ.get("DB_NAME")
    original_env_env = os.environ.get("ENVIRONMENT")

    try:
        # Set test-specific environment variables
        os.environ["DB_NAME"] = f"{current_config.database}_test"
        os.environ["ENVIRONMENT"] = "testing"

        test_config = DatabaseConfig(echo_sql=False, pool_size=2, max_overflow=1)
        return test_config
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["DB_NAME"] = original_env
        else:
            os.environ.pop("DB_NAME", None)

        if original_env_env is not None:
            os.environ["ENVIRONMENT"] = original_env_env
        else:
            os.environ.pop("ENVIRONMENT", None)
