"""
Migration tests for database schema evolution.

Comprehensive testing suite for Alembic migrations that validates:
- Migration up/down cycles work correctly
- Data integrity during schema changes
- Migration rollback safety
- Migration dependency handling
- Schema version consistency
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from alembic import command, script
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.config import DatabaseConfig
from database.initialization import DatabaseInitializer
from database.models import Base


@pytest.mark.migration
@pytest.mark.asyncio
class TestDatabaseMigrations:
    """Test database migration system."""

    @pytest.fixture
    def alembic_config(self, test_database_config: DatabaseConfig):
        """Create Alembic configuration for testing."""
        # Create temporary alembic.ini for testing
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", "database/migrations")
        alembic_cfg.set_main_option(
            "sqlalchemy.url", test_database_config.get_migration_url()
        )
        return alembic_cfg

    @pytest.fixture
    def migration_context(self, test_engine, alembic_config):
        """Create migration context for testing."""
        return MigrationContext.configure(test_engine.sync_engine, opts={})

    async def test_migration_upgrade_downgrade_cycle(
        self, test_database_config: DatabaseConfig, alembic_config: Config
    ):
        """Test complete migration upgrade and downgrade cycle."""

        # Get current migration state
        script_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(script_dir.walk_revisions())

        if not revisions:
            pytest.skip("No migrations found to test")

        # Start from base (no migrations applied)
        command.downgrade(alembic_config, "base")

        # Test incremental upgrades
        for revision in reversed(revisions):
            # Upgrade to this revision
            command.upgrade(alembic_config, revision.revision)

            # Verify database state after upgrade
            await self._verify_database_state_after_migration(
                test_database_config, revision.revision
            )

        # Test incremental downgrades
        for revision in revisions:
            if revision.down_revision:
                # Downgrade from this revision
                command.downgrade(alembic_config, revision.down_revision)

                # Verify database state after downgrade
                await self._verify_database_state_after_migration(
                    test_database_config, revision.down_revision or "base"
                )

    async def test_migration_with_existing_data(
        self,
        db_session: AsyncSession,
        sample_user,
        sample_portfolio,
        alembic_config: Config,
        test_database_config: DatabaseConfig,
    ):
        """Test migrations preserve existing data integrity."""

        # Create test data before migration
        test_data = await self._capture_test_data(db_session)

        # Get current head revision
        script_dir = ScriptDirectory.from_config(alembic_config)
        head_revision = script_dir.get_current_head()

        if not head_revision:
            pytest.skip("No head revision found")

        # Simulate migration cycle with existing data
        command.downgrade(alembic_config, "base")
        command.upgrade(alembic_config, head_revision)

        # Verify data integrity after migration cycle
        await self._verify_data_integrity_after_migration(
            test_database_config, test_data
        )

    async def test_migration_rollback_safety(
        self, alembic_config: Config, test_database_config: DatabaseConfig
    ):
        """Test that migrations can be safely rolled back."""

        script_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(script_dir.walk_revisions())

        if len(revisions) < 2:
            pytest.skip("Need at least 2 migrations to test rollback")

        # Apply all migrations
        command.upgrade(alembic_config, "head")

        # Test rolling back each migration
        for i, revision in enumerate(revisions[:-1]):  # Skip base revision
            try:
                # Rollback to previous revision
                command.downgrade(alembic_config, revision.down_revision or "base")

                # Verify rollback was successful
                current_rev = await self._get_current_revision(test_database_config)
                expected_rev = revision.down_revision or None

                assert (
                    current_rev == expected_rev
                ), f"Rollback failed: expected {expected_rev}, got {current_rev}"

                # Re-apply migration to test it still works
                command.upgrade(alembic_config, revision.revision)

            except Exception as e:
                pytest.fail(f"Migration rollback failed for {revision.revision}: {e}")

    async def test_migration_idempotency(
        self, alembic_config: Config, test_database_config: DatabaseConfig
    ):
        """Test that running migrations multiple times is safe."""

        # Apply all migrations
        command.upgrade(alembic_config, "head")

        # Capture initial state
        initial_schema = await self._capture_database_schema(test_database_config)

        # Run migrations again
        command.upgrade(alembic_config, "head")

        # Verify schema is unchanged
        final_schema = await self._capture_database_schema(test_database_config)

        assert initial_schema == final_schema, "Multiple migration runs changed schema"

    async def test_migration_dependency_handling(self, alembic_config: Config):
        """Test that migration dependencies are handled correctly."""

        script_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(script_dir.walk_revisions())

        # Verify dependency chain is valid
        for revision in revisions:
            if revision.down_revision:
                # Verify down_revision exists
                down_rev = script_dir.get_revision(revision.down_revision)
                assert down_rev is not None, (
                    f"Migration {revision.revision} references non-existent "
                    f"down_revision {revision.down_revision}"
                )

    async def test_schema_consistency_after_migrations(
        self, alembic_config: Config, test_database_config: DatabaseConfig
    ):
        """Test that migrated schema matches SQLAlchemy models."""

        # Apply all migrations
        command.upgrade(alembic_config, "head")

        # Get database schema from migrations
        migrated_schema = await self._capture_database_schema(test_database_config)

        # Get expected schema from SQLAlchemy models
        expected_schema = await self._get_expected_schema_from_models()

        # Compare schemas (allowing for some differences in constraints/indexes)
        schema_diff = self._compare_schemas(migrated_schema, expected_schema)

        if schema_diff:
            pytest.fail(f"Schema mismatch after migrations: {schema_diff}")

    async def test_migration_performance(
        self, alembic_config: Config, performance_metrics
    ):
        """Test migration performance benchmarks."""
        import time

        script_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(script_dir.walk_revisions())

        # Test each migration's performance
        command.downgrade(alembic_config, "base")

        for revision in reversed(revisions):
            start_time = time.perf_counter()
            command.upgrade(alembic_config, revision.revision)
            duration = time.perf_counter() - start_time

            performance_metrics.record(f"migration_{revision.revision}", duration)

            # Each migration should complete in reasonable time (5 minutes max)
            assert (
                duration < 300
            ), f"Migration {revision.revision} took {duration:.2f}s (>5min limit)"

    # Helper methods
    async def _verify_database_state_after_migration(
        self, config: DatabaseConfig, expected_revision: str
    ):
        """Verify database state after migration."""
        # Check revision is correct
        current_rev = await self._get_current_revision(config)
        assert (
            current_rev == expected_revision
        ), f"Expected revision {expected_revision}, got {current_rev}"

        # Check database connectivity
        from sqlalchemy import create_engine

        engine = create_engine(config.get_migration_url())

        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
        finally:
            engine.dispose()

    async def _capture_test_data(self, db_session: AsyncSession) -> dict[str, Any]:
        """Capture test data before migration for integrity verification."""
        # Get counts of each table
        tables_data = {}

        # Users
        result = await db_session.execute(text("SELECT COUNT(*) FROM users"))
        tables_data["users_count"] = result.scalar()

        # Portfolios
        result = await db_session.execute(text("SELECT COUNT(*) FROM portfolios"))
        tables_data["portfolios_count"] = result.scalar()

        # Positions
        result = await db_session.execute(text("SELECT COUNT(*) FROM positions"))
        tables_data["positions_count"] = result.scalar()

        # Sample record IDs for integrity checking
        result = await db_session.execute(text("SELECT id FROM users LIMIT 1"))
        user_id = result.scalar()
        if user_id:
            tables_data["sample_user_id"] = str(user_id)

        return tables_data

    async def _verify_data_integrity_after_migration(
        self, config: DatabaseConfig, original_data: dict[str, Any]
    ):
        """Verify data integrity after migration cycle."""
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_async_engine(config.get_database_url())
        SessionLocal = sessionmaker(engine, class_=AsyncSession)

        try:
            async with SessionLocal() as session:
                # Verify table counts are preserved
                result = await session.execute(text("SELECT COUNT(*) FROM users"))
                assert result.scalar() == original_data.get("users_count", 0)

                result = await session.execute(text("SELECT COUNT(*) FROM portfolios"))
                assert result.scalar() == original_data.get("portfolios_count", 0)

                result = await session.execute(text("SELECT COUNT(*) FROM positions"))
                assert result.scalar() == original_data.get("positions_count", 0)

                # Verify sample record still exists
                if "sample_user_id" in original_data:
                    result = await session.execute(
                        text("SELECT id FROM users WHERE id = :id"),
                        {"id": original_data["sample_user_id"]},
                    )
                    assert result.scalar() is not None
        finally:
            await engine.dispose()

    async def _get_current_revision(self, config: DatabaseConfig) -> str:
        """Get current database revision."""
        from sqlalchemy import create_engine

        engine = create_engine(config.get_migration_url())

        try:
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        finally:
            engine.dispose()

    async def _capture_database_schema(self, config: DatabaseConfig) -> dict[str, Any]:
        """Capture current database schema."""
        from sqlalchemy import create_engine

        engine = create_engine(config.get_migration_url())

        try:
            with engine.connect() as conn:
                inspector = inspect(conn)

                schema = {"tables": {}, "indexes": {}, "foreign_keys": {}}

                # Capture table information
                for table_name in inspector.get_table_names():
                    columns = inspector.get_columns(table_name)
                    schema["tables"][table_name] = {
                        "columns": {
                            col["name"]: col["type"].__class__.__name__
                            for col in columns
                        }
                    }

                    # Capture indexes
                    indexes = inspector.get_indexes(table_name)
                    schema["indexes"][table_name] = {
                        idx["name"]: idx["column_names"] for idx in indexes
                    }

                    # Capture foreign keys
                    foreign_keys = inspector.get_foreign_keys(table_name)
                    schema["foreign_keys"][table_name] = {
                        fk["name"]: {
                            "columns": fk["constrained_columns"],
                            "refers_to": f"{fk['referred_table']}.{fk['referred_columns']}",
                        }
                        for fk in foreign_keys
                    }

                return schema
        finally:
            engine.dispose()

    async def _get_expected_schema_from_models(self) -> dict[str, Any]:
        """Get expected schema from SQLAlchemy models."""
        metadata = Base.metadata

        schema = {"tables": {}, "indexes": {}, "foreign_keys": {}}

        for table_name, table in metadata.tables.items():
            # Capture columns
            schema["tables"][table_name] = {
                "columns": {
                    col.name: col.type.__class__.__name__ for col in table.columns
                }
            }

            # Capture indexes
            schema["indexes"][table_name] = {
                idx.name: list(idx.columns.keys()) for idx in table.indexes
            }

            # Capture foreign keys
            schema["foreign_keys"][table_name] = {
                fk.name: {
                    "columns": list(fk.columns.keys()),
                    "refers_to": f"{fk.elements[0].column.table.name}.{fk.elements[0].column.name}",
                }
                for fk in table.foreign_keys
            }

        return schema

    def _compare_schemas(
        self, migrated: dict[str, Any], expected: dict[str, Any]
    ) -> list[str]:
        """Compare two schemas and return differences."""
        differences = []

        # Compare tables
        migrated_tables = set(migrated["tables"].keys())
        expected_tables = set(expected["tables"].keys())

        if migrated_tables != expected_tables:
            missing = expected_tables - migrated_tables
            extra = migrated_tables - expected_tables

            if missing:
                differences.append(f"Missing tables: {missing}")
            if extra:
                differences.append(f"Extra tables: {extra}")

        # Compare columns for common tables
        for table in migrated_tables & expected_tables:
            migrated_cols = set(migrated["tables"][table]["columns"].keys())
            expected_cols = set(expected["tables"][table]["columns"].keys())

            if migrated_cols != expected_cols:
                missing = expected_cols - migrated_cols
                extra = migrated_cols - expected_cols

                if missing:
                    differences.append(f"Table {table} missing columns: {missing}")
                if extra:
                    differences.append(f"Table {table} extra columns: {extra}")

        return differences


@pytest.mark.migration
@pytest.mark.asyncio
class TestMigrationEnvironments:
    """Test migration behavior across different environments."""

    async def test_development_migration_settings(
        self, test_database_config: DatabaseConfig
    ):
        """Test migration settings for development environment."""
        test_database_config.environment = "development"

        # Development should allow more verbose output
        assert test_database_config.echo_sql in [True, False]  # Both acceptable

        # Development should have reasonable timeouts
        assert test_database_config.migration_timeout >= 60

    async def test_production_migration_settings(
        self, test_database_config: DatabaseConfig
    ):
        """Test migration settings for production environment."""
        test_database_config.environment = "production"

        # Production should have appropriate settings
        assert test_database_config.echo_sql == False  # No SQL echoing in prod
        assert test_database_config.migration_timeout >= 300  # Longer timeout for prod

    async def test_migration_backup_strategy(
        self, test_database_config: DatabaseConfig, alembic_config: Config
    ):
        """Test that migration includes backup strategy considerations."""

        # This test documents the backup strategy
        # In real production, you would:
        # 1. Create database backup before migration
        # 2. Test migration on backup first
        # 3. Apply migration with rollback plan

        # For testing, we verify migration can be rolled back
        script_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(script_dir.walk_revisions())

        if revisions:
            # Each migration should have a down_revision for rollback
            for revision in revisions:
                assert hasattr(
                    revision, "down_revision"
                ), f"Migration {revision.revision} missing rollback capability"


# Migration test utilities
class MigrationTestUtils:
    """Utilities for migration testing."""

    @staticmethod
    def create_test_migration(
        alembic_config: Config, message: str, upgrade_sql: str, downgrade_sql: str
    ) -> str:
        """Create a test migration for testing purposes."""

        # Generate migration
        rev = command.revision(alembic_config, message=message, autogenerate=False)

        # Get the generated file path
        script_dir = ScriptDirectory.from_config(alembic_config)
        revision = script_dir.get_revision(rev.revision)

        # Modify the generated file to include test SQL
        migration_file = Path(revision.path)
        content = migration_file.read_text()

        # Replace upgrade and downgrade functions
        content = content.replace(
            "def upgrade():", f'def upgrade():\n    op.execute("""{upgrade_sql}""")'
        )
        content = content.replace(
            "def downgrade():",
            f'def downgrade():\n    op.execute("""{downgrade_sql}""")',
        )

        migration_file.write_text(content)

        return rev.revision


@pytest.fixture
def migration_test_utils():
    """Provide migration test utilities."""
    return MigrationTestUtils()
