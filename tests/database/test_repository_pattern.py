"""
Tests for the repository pattern implementation.

Tests the base repository, domain-specific repositories, and repository registry.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from database.exceptions import (
    DatabaseConnectionError,
    EntityNotFoundError,
    RepositoryError,
    ValidationError,
)
from database.repositories import (
    RepositoryRegistry,
    close_repositories,
    get_repository_registry,
    init_repositories,
)

# Import repository components
from database.repositories.base import BaseRepository
from database.repositories.session_manager import DatabaseSessionManager


class MockEntity:
    """Mock entity for testing base repository."""

    def __init__(self, id=None, name="test", value=42):
        self.id = id or uuid4()
        self.name = name
        self.value = value


class MockRepository(BaseRepository[MockEntity]):
    """Mock repository for testing base functionality."""

    def __init__(self, session):
        super().__init__(session, MockEntity)

    async def _validate_entity(
        self, entity: MockEntity, is_update: bool = False
    ) -> None:
        """Mock validation."""
        if not entity.name:
            raise ValidationError("Name is required", field="name")

        if entity.value < 0:
            raise ValidationError(
                "Value cannot be negative", field="value", value=entity.value
            )

    async def _apply_eager_loading(self, query):
        """Mock eager loading."""
        return query


class TestBaseRepository:
    """Test the base repository functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.add = Mock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.merge = AsyncMock()
        session.delete = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def mock_repository(self, mock_session):
        """Create a mock repository instance."""
        return MockRepository(mock_session)

    @pytest.mark.asyncio
    async def test_create_entity_success(self, mock_repository, mock_session):
        """Test successful entity creation."""
        entity = MockEntity(name="test_entity", value=100)

        # Test creation
        result = await mock_repository.create(entity)

        # Verify session interactions
        mock_session.add.assert_called_once_with(entity)
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once_with(entity)

        assert result == entity

    @pytest.mark.asyncio
    async def test_create_entity_validation_error(self, mock_repository):
        """Test entity creation with validation error."""
        entity = MockEntity(name="", value=100)  # Invalid name

        with pytest.raises(RepositoryError) as exc_info:
            await mock_repository.create(entity)

        assert "Failed to create MockEntity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_repository, mock_session):
        """Test getting entity by ID when found."""
        entity_id = uuid4()
        entity = MockEntity(id=entity_id, name="found_entity")

        # Mock query result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = entity
        mock_session.execute.return_value = mock_result

        result = await mock_repository.get_by_id(entity_id)

        assert result == entity
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_repository, mock_session):
        """Test getting entity by ID when not found."""
        entity_id = uuid4()

        # Mock query result (not found)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await mock_repository.get_by_id(entity_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_entity_success(self, mock_repository, mock_session):
        """Test successful entity update."""
        entity = MockEntity(name="updated_entity", value=200)

        mock_session.merge.return_value = entity

        result = await mock_repository.update(entity)

        mock_session.merge.assert_called_once_with(entity)
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once_with(entity)

        assert result == entity

    @pytest.mark.asyncio
    async def test_delete_entity_success(self, mock_repository, mock_session):
        """Test successful entity deletion."""
        entity_id = uuid4()
        entity = MockEntity(id=entity_id)

        # Mock get_by_id to return the entity
        with patch.object(mock_repository, "get_by_id", return_value=entity):
            result = await mock_repository.delete(entity_id)

        mock_session.delete.assert_called_once_with(entity)
        mock_session.flush.assert_called_once()

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, mock_repository):
        """Test entity deletion when entity not found."""
        entity_id = uuid4()

        # Mock get_by_id to return None
        with patch.object(mock_repository, "get_by_id", return_value=None):
            result = await mock_repository.delete(entity_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_by_criteria(self, mock_repository, mock_session):
        """Test listing entities by criteria."""
        entities = [
            MockEntity(name="entity1", value=100),
            MockEntity(name="entity2", value=200),
        ]

        # Mock query result
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = entities
        mock_session.execute.return_value = mock_result

        result = await mock_repository.list_by_criteria(offset=0, limit=10, name="test")

        assert result == entities
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_by_criteria(self, mock_repository, mock_session):
        """Test counting entities by criteria."""
        expected_count = 5

        # Mock query result
        mock_result = Mock()
        mock_result.scalar.return_value = expected_count
        mock_session.execute.return_value = mock_result

        result = await mock_repository.count_by_criteria(name="test")

        assert result == expected_count
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_exists_true(self, mock_repository, mock_session):
        """Test entity existence check when entity exists."""
        entity_id = uuid4()

        # Mock count result
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        result = await mock_repository.exists(entity_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, mock_repository, mock_session):
        """Test entity existence check when entity doesn't exist."""
        entity_id = uuid4()

        # Mock count result
        mock_result = Mock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result

        result = await mock_repository.exists(entity_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_bulk_create_success(self, mock_repository, mock_session):
        """Test successful bulk entity creation."""
        entities = [
            MockEntity(name="bulk1", value=100),
            MockEntity(name="bulk2", value=200),
            MockEntity(name="bulk3", value=300),
        ]

        result = await mock_repository.bulk_create(entities)

        mock_session.add_all.assert_called_once_with(entities)
        mock_session.flush.assert_called_once()

        # Should refresh each entity
        assert mock_session.refresh.call_count == len(entities)

        assert result == entities


class TestSessionManager:
    """Test the database session manager."""

    @pytest.fixture
    def session_manager(self):
        """Create a session manager instance."""
        return DatabaseSessionManager(
            database_url="postgresql+asyncpg://test:test@localhost/test", echo=False
        )

    def test_session_manager_initialization(self, session_manager):
        """Test session manager initialization."""
        assert session_manager.database_url is not None
        assert session_manager.echo is False
        assert session_manager._initialized is False

    @pytest.mark.asyncio
    async def test_get_database_url(self, session_manager):
        """Test database URL generation."""
        url = session_manager._get_database_url()
        assert "postgresql+asyncpg://" in url

    @pytest.mark.asyncio
    @patch("database.repositories.session_manager.create_async_engine")
    @patch("database.repositories.session_manager.async_sessionmaker")
    async def test_initialize_success(
        self, mock_sessionmaker, mock_engine, session_manager
    ):
        """Test successful session manager initialization."""
        # Mock the engine and session factory
        mock_engine_instance = Mock()
        mock_sessionmaker_instance = Mock()

        mock_engine.return_value = mock_engine_instance
        mock_sessionmaker.return_value = mock_sessionmaker_instance

        await session_manager.initialize()

        assert session_manager._initialized is True
        assert session_manager._engine == mock_engine_instance
        assert session_manager._session_factory == mock_sessionmaker_instance

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, session_manager):
        """Test health check when not initialized."""
        result = await session_manager.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result


class TestRepositoryRegistry:
    """Test the repository registry."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        session_manager = Mock(spec=DatabaseSessionManager)
        session_manager.get_session = AsyncMock()
        session_manager.health_check = AsyncMock(return_value={"status": "healthy"})
        return session_manager

    @pytest.fixture
    def registry(self, mock_session_manager):
        """Create a repository registry with mock session manager."""
        return RepositoryRegistry(mock_session_manager)

    @pytest.mark.asyncio
    async def test_get_portfolio_repository(self, registry, mock_session_manager):
        """Test getting portfolio repository."""
        mock_session = Mock()
        mock_session_manager.get_session.return_value = mock_session

        repo = await registry.get_portfolio_repository()

        assert repo is not None
        assert registry._portfolio_repo is repo
        mock_session_manager.get_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analysis_repository(self, registry, mock_session_manager):
        """Test getting analysis repository."""
        mock_session = Mock()
        mock_session_manager.get_session.return_value = mock_session

        repo = await registry.get_analysis_repository()

        assert repo is not None
        assert registry._analysis_repo is repo

    @pytest.mark.asyncio
    async def test_get_market_data_repository(self, registry, mock_session_manager):
        """Test getting market data repository."""
        mock_session = Mock()
        mock_session_manager.get_session.return_value = mock_session

        repo = await registry.get_market_data_repository()

        assert repo is not None
        assert registry._market_data_repo is repo

    @pytest.mark.asyncio
    async def test_cleanup_repositories(self, registry):
        """Test repository cleanup."""
        # Set some mock repositories
        registry._portfolio_repo = Mock()
        registry._analysis_repo = Mock()
        registry._market_data_repo = Mock()

        await registry.cleanup_repositories()

        assert registry._portfolio_repo is None
        assert registry._analysis_repo is None
        assert registry._market_data_repo is None

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, registry, mock_session_manager):
        """Test health check when all components are healthy."""
        # Mock healthy session manager
        mock_session_manager.health_check.return_value = {"status": "healthy"}

        # Mock repository methods to succeed
        with patch.object(registry, "get_portfolio_repository") as mock_get_portfolio:
            with patch.object(registry, "get_analysis_repository") as mock_get_analysis:
                with patch.object(
                    registry, "get_market_data_repository"
                ) as mock_get_market:
                    # Mock repositories with successful count_by_criteria
                    mock_portfolio_repo = Mock()
                    mock_portfolio_repo.count_by_criteria = AsyncMock(return_value=0)
                    mock_get_portfolio.return_value = mock_portfolio_repo

                    mock_analysis_repo = Mock()
                    mock_analysis_repo.count_by_criteria = AsyncMock(return_value=0)
                    mock_get_analysis.return_value = mock_analysis_repo

                    mock_market_repo = Mock()
                    mock_market_repo.count_by_criteria = AsyncMock(return_value=0)
                    mock_get_market.return_value = mock_market_repo

                    result = await registry.health_check()

                    assert result["status"] == "healthy"
                    assert result["session_manager"]["status"] == "healthy"
                    assert all(
                        repo["status"] == "healthy"
                        for repo in result["repositories"].values()
                    )

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, registry, mock_session_manager):
        """Test health check when components are unhealthy."""
        # Mock unhealthy session manager
        mock_session_manager.health_check.return_value = {
            "status": "unhealthy",
            "error": "Connection failed",
        }

        result = await registry.health_check()

        assert result["status"] == "unhealthy"
        assert result["session_manager"]["status"] == "unhealthy"


class TestGlobalFunctions:
    """Test global repository functions."""

    @pytest.mark.asyncio
    async def test_get_repository_registry(self):
        """Test getting global repository registry."""
        # Clear any existing registry
        import database.repositories

        database.repositories._registry = None

        registry1 = get_repository_registry()
        registry2 = get_repository_registry()

        assert registry1 is registry2  # Should be singleton
        assert isinstance(registry1, RepositoryRegistry)

    @pytest.mark.asyncio
    @patch("database.repositories.get_repository_registry")
    async def test_init_repositories(self, mock_get_registry):
        """Test repository initialization."""
        mock_registry = Mock()
        mock_registry.session_manager = Mock()
        mock_registry.session_manager.initialize = AsyncMock()
        mock_get_registry.return_value = mock_registry

        await init_repositories()

        mock_registry.session_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("database.repositories._registry")
    async def test_close_repositories(self, mock_registry):
        """Test repository cleanup."""
        mock_registry.cleanup_repositories = AsyncMock()
        mock_registry.session_manager = Mock()
        mock_registry.session_manager.close = AsyncMock()

        await close_repositories()

        mock_registry.cleanup_repositories.assert_called_once()
        mock_registry.session_manager.close.assert_called_once()


class TestErrorHandling:
    """Test error handling in repositories."""

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling of database connection errors."""
        session_manager = DatabaseSessionManager("invalid://connection/string")

        with pytest.raises(DatabaseConnectionError):
            await session_manager.initialize()

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, mock_session):
        """Test validation error handling in repositories."""
        repository = MockRepository(mock_session)

        # Entity with invalid data
        invalid_entity = MockEntity(name="", value=-1)  # Empty name and negative value

        with pytest.raises(RepositoryError) as exc_info:
            await repository.create(invalid_entity)

        assert "Failed to create MockEntity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_session):
        """Test query error handling."""
        repository = MockRepository(mock_session)

        # Mock session to raise an exception
        mock_session.execute.side_effect = Exception("Database query failed")

        with pytest.raises(RepositoryError):
            await repository.get_by_id(uuid4())


if __name__ == "__main__":
    pytest.main([__file__])
