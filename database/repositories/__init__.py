"""
Repository registry for BuffetBot database layer.

Centralized repository access for dependency injection and service layer integration.
"""

from typing import Optional

from buffetbot.utils.logger import setup_logger

from .analysis_repo import AnalysisRepository
from .market_data_repo import MarketDataRepository

# Import all repositories
from .portfolio_repo import PortfolioRepository, PositionRepository
from .session_manager import DatabaseSessionManager, get_session_manager

# Import models for type hints - will be available when Phase 1a models are integrated
try:
    from ..models.user import User
except ImportError:
    # Placeholder for development
    class User:
        pass


# Initialize logger
logger = setup_logger(__name__)


class RepositoryRegistry:
    """
    Centralized repository access for dependency injection.

    Provides a unified interface to access all repositories with proper
    session management and dependency injection support.
    """

    def __init__(self, session_manager: DatabaseSessionManager = None):
        """
        Initialize the repository registry.

        Args:
            session_manager: Database session manager instance
        """
        self.session_manager = session_manager or get_session_manager()
        self.logger = setup_logger(f"{__name__}.RepositoryRegistry")

        # Repository instances (will be created on demand)
        self._portfolio_repo = None
        self._position_repo = None
        self._analysis_repo = None
        self._market_data_repo = None

    async def get_portfolio_repository(self) -> PortfolioRepository:
        """
        Get portfolio repository instance.

        Returns:
            PortfolioRepository: Portfolio repository
        """
        if self._portfolio_repo is None:
            session = await self.session_manager.get_session()
            self._portfolio_repo = PortfolioRepository(session)
            self.logger.debug("Created portfolio repository instance")

        return self._portfolio_repo

    async def get_position_repository(self) -> PositionRepository:
        """
        Get position repository instance.

        Returns:
            PositionRepository: Position repository
        """
        if self._position_repo is None:
            session = await self.session_manager.get_session()
            self._position_repo = PositionRepository(session)
            self.logger.debug("Created position repository instance")

        return self._position_repo

    async def get_analysis_repository(self) -> AnalysisRepository:
        """
        Get analysis repository instance.

        Returns:
            AnalysisRepository: Analysis repository
        """
        if self._analysis_repo is None:
            session = await self.session_manager.get_session()
            self._analysis_repo = AnalysisRepository(session)
            self.logger.debug("Created analysis repository instance")

        return self._analysis_repo

    async def get_market_data_repository(self) -> MarketDataRepository:
        """
        Get market data repository instance.

        Returns:
            MarketDataRepository: Market data repository
        """
        if self._market_data_repo is None:
            session = await self.session_manager.get_session()
            self._market_data_repo = MarketDataRepository(session)
            self.logger.debug("Created market data repository instance")

        return self._market_data_repo

    async def cleanup_repositories(self):
        """Clean up repository instances and close sessions."""
        self._portfolio_repo = None
        self._position_repo = None
        self._analysis_repo = None
        self._market_data_repo = None
        self.logger.debug("Cleaned up repository instances")

    async def health_check(self) -> dict:
        """
        Perform health check on all repositories.

        Returns:
            dict: Health check results
        """
        try:
            self.logger.debug("Performing repository health check")

            # Check session manager health
            session_health = await self.session_manager.health_check()

            # Test each repository
            repo_tests = {}

            try:
                portfolio_repo = await self.get_portfolio_repository()
                await portfolio_repo.count_by_criteria()
                repo_tests["portfolio"] = {"status": "healthy"}
            except Exception as e:
                repo_tests["portfolio"] = {"status": "unhealthy", "error": str(e)}

            try:
                analysis_repo = await self.get_analysis_repository()
                await analysis_repo.count_by_criteria()
                repo_tests["analysis"] = {"status": "healthy"}
            except Exception as e:
                repo_tests["analysis"] = {"status": "unhealthy", "error": str(e)}

            try:
                market_data_repo = await self.get_market_data_repository()
                await market_data_repo.count_by_criteria()
                repo_tests["market_data"] = {"status": "healthy"}
            except Exception as e:
                repo_tests["market_data"] = {"status": "unhealthy", "error": str(e)}

            # Overall status
            all_healthy = session_health["status"] == "healthy" and all(
                test["status"] == "healthy" for test in repo_tests.values()
            )

            health_result = {
                "status": "healthy" if all_healthy else "unhealthy",
                "session_manager": session_health,
                "repositories": repo_tests,
                "timestamp": session_health.get("timestamp"),
            }

            self.logger.debug(
                f"Health check completed: {'healthy' if all_healthy else 'unhealthy'}"
            )
            return health_result

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "timestamp": None}


# Global registry instance
_registry: Optional[RepositoryRegistry] = None


def get_repository_registry() -> RepositoryRegistry:
    """
    Get the global repository registry instance.

    Returns:
        RepositoryRegistry: Global repository registry
    """
    global _registry
    if _registry is None:
        _registry = RepositoryRegistry()
        logger.debug("Created global repository registry")
    return _registry


async def init_repositories():
    """Initialize the global repository registry."""
    registry = get_repository_registry()
    await registry.session_manager.initialize()
    logger.info("Repository registry initialized")


async def close_repositories():
    """Close the global repository registry."""
    global _registry
    if _registry:
        await _registry.cleanup_repositories()
        await _registry.session_manager.close()
        _registry = None
        logger.info("Repository registry closed")


# Export all repositories and utilities
__all__ = [
    # Core registry
    "RepositoryRegistry",
    "get_repository_registry",
    "init_repositories",
    "close_repositories",
    # Individual repositories
    "PortfolioRepository",
    "PositionRepository",
    "AnalysisRepository",
    "MarketDataRepository",
    # Session management
    "DatabaseSessionManager",
    "get_session_manager",
]
