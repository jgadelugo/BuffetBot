"""
Database CLI commands for BuffetBot.

Provides command-line interface for database operations including migrations,
health checks, initialization, and development data seeding.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
from alembic import command
from alembic.config import Config

from .config import DatabaseConfig, DatabaseEnvironment
from .initialization import DatabaseInitializer

# Setup logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    alembic_cfg_path = Path(__file__).parent / "migrations" / "alembic.ini"
    if not alembic_cfg_path.exists():
        raise RuntimeError(f"Alembic config not found at {alembic_cfg_path}")

    alembic_cfg = Config(str(alembic_cfg_path))

    # Set the database URL for migrations
    db_config = DatabaseConfig()
    alembic_cfg.set_main_option("sqlalchemy.url", db_config.get_migration_url())

    return alembic_cfg


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def db(verbose: bool):
    """Database management commands for BuffetBot."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("alembic").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@db.command()
def init():
    """Initialize Alembic migration environment."""
    try:
        migrations_dir = Path(__file__).parent / "migrations"

        if (migrations_dir / "alembic.ini").exists():
            click.echo("⚠️  Alembic already initialized in database/migrations/")
            if not click.confirm("Reinitialize anyway?"):
                return

        # Initialize Alembic
        alembic_cfg = get_alembic_config()
        command.init(alembic_cfg, str(migrations_dir))

        click.echo("✅ Alembic migration environment initialized")

    except Exception as exc:
        click.echo(f"❌ Failed to initialize Alembic: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option("--message", "-m", required=True, help="Migration message")
@click.option(
    "--autogenerate/--no-autogenerate",
    default=True,
    help="Auto-generate migration from model changes",
)
def migrate(message: str, autogenerate: bool):
    """Create a new migration."""
    try:
        alembic_cfg = get_alembic_config()

        click.echo(f"Creating migration: {message}")
        command.revision(alembic_cfg, message=message, autogenerate=autogenerate)

        click.echo("✅ Migration created successfully")
        click.echo("💡 Review the migration file before applying with 'db upgrade'")

    except Exception as exc:
        click.echo(f"❌ Failed to create migration: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option(
    "--revision", "-r", default="head", help="Target revision (default: head)"
)
@click.option("--sql", is_flag=True, help="Show SQL instead of executing")
def upgrade(revision: str, sql: bool):
    """Run database migrations."""
    try:
        alembic_cfg = get_alembic_config()

        if sql:
            click.echo("-- SQL for migration:")
            command.upgrade(alembic_cfg, revision, sql=True)
        else:
            click.echo(f"Upgrading database to {revision}...")
            command.upgrade(alembic_cfg, revision)
            click.echo("✅ Database migrations completed")

    except Exception as exc:
        click.echo(f"❌ Migration failed: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option("--revision", "-r", default="-1", help="Target revision (default: -1)")
@click.option("--sql", is_flag=True, help="Show SQL instead of executing")
def downgrade(revision: str, sql: bool):
    """Rollback database migration."""
    try:
        config = DatabaseConfig()
        if config.is_production:
            click.echo(
                "❌ Cannot rollback migrations in production environment", err=True
            )
            sys.exit(1)

        if not sql and not click.confirm(
            f"⚠️  Are you sure you want to rollback to {revision}? This may cause data loss."
        ):
            return

        alembic_cfg = get_alembic_config()

        if sql:
            click.echo("-- SQL for rollback:")
            command.downgrade(alembic_cfg, revision, sql=True)
        else:
            click.echo(f"Rolling back to {revision}...")
            command.downgrade(alembic_cfg, revision)
            click.echo("✅ Database migration rolled back")

    except Exception as exc:
        click.echo(f"❌ Rollback failed: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option("--verbose", "-v", is_flag=True, help="Show verbose history")
def history(verbose: bool):
    """Show migration history."""
    try:
        alembic_cfg = get_alembic_config()
        command.history(alembic_cfg, verbose=verbose)

    except Exception as exc:
        click.echo(f"❌ Failed to show history: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option("--verbose", "-v", is_flag=True, help="Show verbose current state")
def current(verbose: bool):
    """Show current migration revision."""
    try:
        alembic_cfg = get_alembic_config()
        command.current(alembic_cfg, verbose=verbose)

    except Exception as exc:
        click.echo(f"❌ Failed to show current revision: {exc}", err=True)
        sys.exit(1)


@db.command()
@click.option("--seed", is_flag=True, help="Seed with development data")
@click.option("--force", is_flag=True, help="Force recreation of tables")
def create(seed: bool, force: bool):
    """Create database tables and optionally seed data."""

    async def _create():
        try:
            config = DatabaseConfig()
            initializer = DatabaseInitializer(config)

            click.echo(f"🔧 Initializing database: {config.database}")
            click.echo(f"📍 Environment: {config.environment.value}")

            if force and not config.is_production:
                click.echo(
                    "⚠️  Force recreation enabled - existing tables will be dropped"
                )
                if not click.confirm("Continue?"):
                    return

            await initializer.initialize_database(
                create_schema=True, seed_data=seed, force_recreate=force
            )

            if seed:
                click.echo("✅ Database created and seeded with development data")
            else:
                click.echo("✅ Database tables created successfully")

        except Exception as exc:
            click.echo(f"❌ Database creation failed: {exc}", err=True)
            raise
        finally:
            await initializer.close()

    try:
        asyncio.run(_create())
    except Exception:
        sys.exit(1)


@db.command()
def drop():
    """Drop all database tables (development only)."""

    async def _drop():
        try:
            config = DatabaseConfig()

            if config.is_production:
                click.echo("❌ Cannot drop tables in production environment", err=True)
                return False

            if not click.confirm(
                "⚠️  This will permanently delete ALL data. Are you absolutely sure?"
            ):
                return False

            initializer = DatabaseInitializer(config)
            await initializer.drop_tables()

            click.echo("✅ All database tables dropped")
            return True

        except Exception as exc:
            click.echo(f"❌ Failed to drop tables: {exc}", err=True)
            return False
        finally:
            await initializer.close()

    try:
        success = asyncio.run(_drop())
        if not success:
            sys.exit(1)
    except Exception:
        sys.exit(1)


@db.command()
@click.option(
    "--detailed", "-d", is_flag=True, help="Show detailed database information"
)
def health(detailed: bool):
    """Check database health and connectivity."""

    async def _health():
        try:
            config = DatabaseConfig()
            initializer = DatabaseInitializer(config)

            # Basic health check
            is_healthy = await initializer.check_database_health()

            if is_healthy:
                click.echo("✅ Database connection is healthy")

                if detailed:
                    info = await initializer.get_database_info()
                    click.echo("\n📊 Database Information:")
                    for key, value in info.items():
                        if key != "error":
                            click.echo(f"   {key}: {value}")

                return True
            else:
                click.echo("❌ Database health check failed")
                return False

        except Exception as exc:
            click.echo(f"❌ Health check error: {exc}", err=True)
            return False
        finally:
            await initializer.close()

    try:
        is_healthy = asyncio.run(_health())
        if not is_healthy:
            sys.exit(1)
    except Exception:
        sys.exit(1)


@db.command()
@click.option("--clear", is_flag=True, help="Clear existing data before seeding")
def seed(clear: bool):
    """Seed database with development data."""

    async def _seed():
        try:
            config = DatabaseConfig()

            if not config.is_development:
                click.echo("⚠️  Seeding is only available in development environment")
                if not click.confirm("Continue anyway?"):
                    return False

            initializer = DatabaseInitializer(config)

            if clear:
                click.echo("🧹 Clearing existing development data...")
                from .seeds import clear_all_data

                async with initializer.SessionLocal() as session:
                    await clear_all_data(session)
                    await session.commit()

            click.echo("🌱 Seeding development data...")
            await initializer.seed_development_data()

            click.echo("✅ Development data seeding completed")
            return True

        except Exception as exc:
            click.echo(f"❌ Seeding failed: {exc}", err=True)
            return False
        finally:
            await initializer.close()

    try:
        success = asyncio.run(_seed())
        if not success:
            sys.exit(1)
    except Exception:
        sys.exit(1)


@db.command()
def config():
    """Show current database configuration."""
    try:
        config = DatabaseConfig()

        click.echo("📋 Database Configuration:")
        click.echo(f"   Environment: {config.environment.value}")
        click.echo(f"   Host: {config.host}")
        click.echo(f"   Port: {config.port}")
        click.echo(f"   Database: {config.database}")
        click.echo(f"   Username: {config.username}")
        click.echo(f"   Pool Size: {config.pool_size}")
        click.echo(f"   Max Overflow: {config.max_overflow}")
        click.echo(f"   SSL Mode: {config.ssl_mode}")
        click.echo(f"   Echo SQL: {config.echo_sql}")

    except Exception as exc:
        click.echo(f"❌ Failed to load configuration: {exc}", err=True)
        sys.exit(1)


@db.command()
def reset():
    """Reset database (drop, create, migrate, seed) - development only."""

    async def _reset():
        try:
            config = DatabaseConfig()

            if config.is_production:
                click.echo("❌ Reset is not allowed in production environment", err=True)
                return False

            click.echo("🔄 Resetting database (this will delete all data)...")
            if not click.confirm("Continue?"):
                return False

            initializer = DatabaseInitializer(config)

            # Drop tables
            await initializer.drop_tables()
            click.echo("✅ Tables dropped")

            # Create tables
            await initializer.create_tables()
            click.echo("✅ Tables created")

            # Seed data
            await initializer.seed_development_data()
            click.echo("✅ Development data seeded")

            click.echo("🎉 Database reset completed successfully")
            return True

        except Exception as exc:
            click.echo(f"❌ Reset failed: {exc}", err=True)
            return False
        finally:
            await initializer.close()

    try:
        success = asyncio.run(_reset())
        if not success:
            sys.exit(1)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    db()
