"""
Database seeding for BuffetBot development environment.

Provides sample data creation for development and testing purposes.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .sample_analysis import create_sample_analysis_results
from .sample_market_data import create_sample_market_data
from .sample_portfolios import create_sample_portfolios

logger = logging.getLogger(__name__)


async def create_sample_data(session: AsyncSession) -> None:
    """
    Create all sample data for development.

    Args:
        session: Async SQLAlchemy session
    """
    logger.info("Creating sample data for development...")

    try:
        # Create sample portfolios and users first (required by other entities)
        await create_sample_portfolios(session)

        # Create sample market data
        await create_sample_market_data(session)

        # Create sample analysis results (depends on portfolios and market data)
        await create_sample_analysis_results(session)

        logger.info("Sample data creation completed successfully")

    except Exception as exc:
        logger.error(f"Failed to create sample data: {exc}")
        raise


async def clear_all_data(session: AsyncSession) -> None:
    """
    Clear all data from database (development only).

    Args:
        session: Async SQLAlchemy session
    """
    from ..models.analysis import AnalysisResult
    from ..models.market_data import MarketDataCache
    from ..models.portfolio import Portfolio, Position
    from ..models.user import User

    logger.warning("Clearing all data from database...")

    # Delete in reverse dependency order
    await session.execute("DELETE FROM analysis_results")
    await session.execute("DELETE FROM market_data_cache")
    await session.execute("DELETE FROM positions")
    await session.execute("DELETE FROM portfolios")
    await session.execute("DELETE FROM users")

    logger.info("All data cleared from database")
