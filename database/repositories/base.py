"""
Base repository for BuffetBot database layer.

Abstract base repository providing common CRUD operations and patterns
for all domain-specific repositories.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from buffetbot.utils.logger import setup_logger

from ..exceptions import (
    EntityNotFoundError,
    QueryError,
    RepositoryError,
    ValidationError,
)

# Initialize logger
logger = setup_logger(__name__)

# Generic type for model entities
T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common CRUD operations.

    Provides standardized methods for Create, Read, Update, Delete operations
    along with query building helpers and error handling.
    """

    def __init__(self, session: AsyncSession, model_class: type[T]):
        """
        Initialize the base repository.

        Args:
            session: Database session for operations
            model_class: SQLAlchemy model class for this repository
        """
        self.session = session
        self.model_class = model_class
        self.model_name = model_class.__name__
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    async def create(self, entity: T) -> T:
        """
        Create a new entity in the database.

        Args:
            entity: Entity instance to create

        Returns:
            T: Created entity with generated fields

        Raises:
            RepositoryError: If creation fails
        """
        try:
            self.logger.debug(f"Creating new {self.model_name}")

            # Validate entity before creation
            await self._validate_entity(entity)

            # Add to session and flush to get generated values
            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)

            self.logger.info(
                f"Created {self.model_name} with ID: {getattr(entity, 'id', 'N/A')}"
            )
            return entity

        except Exception as e:
            self.logger.error(f"Failed to create {self.model_name}: {e}")
            raise RepositoryError(
                f"Failed to create {self.model_name}",
                repository=self.__class__.__name__,
                operation="create",
                details={"error": str(e)},
            )

    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """
        Get an entity by its ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            Optional[T]: Entity if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Getting {self.model_name} by ID: {entity_id}")

            query = select(self.model_class).where(self.model_class.id == entity_id)
            query = await self._apply_eager_loading(query)

            result = await self.session.execute(query)
            entity = result.scalar_one_or_none()

            if entity:
                self.logger.debug(f"Found {self.model_name} with ID: {entity_id}")
            else:
                self.logger.debug(f"No {self.model_name} found with ID: {entity_id}")

            return entity

        except Exception as e:
            self.logger.error(f"Failed to get {self.model_name} by ID {entity_id}: {e}")
            raise RepositoryError(
                f"Failed to get {self.model_name} by ID",
                repository=self.__class__.__name__,
                operation="get_by_id",
                details={"error": str(e), "entity_id": str(entity_id)},
            )

    async def update(self, entity: T) -> T:
        """
        Update an existing entity in the database.

        Args:
            entity: Entity instance to update

        Returns:
            T: Updated entity

        Raises:
            RepositoryError: If update fails
        """
        try:
            entity_id = getattr(entity, "id", None)
            self.logger.debug(f"Updating {self.model_name} with ID: {entity_id}")

            # Validate entity before update
            await self._validate_entity(entity, is_update=True)

            # Merge the entity into the session
            merged_entity = await self.session.merge(entity)
            await self.session.flush()
            await self.session.refresh(merged_entity)

            self.logger.info(f"Updated {self.model_name} with ID: {entity_id}")
            return merged_entity

        except Exception as e:
            self.logger.error(f"Failed to update {self.model_name}: {e}")
            raise RepositoryError(
                f"Failed to update {self.model_name}",
                repository=self.__class__.__name__,
                operation="update",
                details={"error": str(e)},
            )

    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete an entity by its ID.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            bool: True if entity was deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        try:
            self.logger.debug(f"Deleting {self.model_name} with ID: {entity_id}")

            # Check if entity exists first
            entity = await self.get_by_id(entity_id)
            if not entity:
                self.logger.debug(f"No {self.model_name} found with ID: {entity_id}")
                return False

            # Delete the entity
            await self.session.delete(entity)
            await self.session.flush()

            self.logger.info(f"Deleted {self.model_name} with ID: {entity_id}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to delete {self.model_name} with ID {entity_id}: {e}"
            )
            raise RepositoryError(
                f"Failed to delete {self.model_name}",
                repository=self.__class__.__name__,
                operation="delete",
                details={"error": str(e), "entity_id": str(entity_id)},
            )

    async def list_by_criteria(
        self, offset: int = 0, limit: int = 100, order_by: str = None, **filters
    ) -> list[T]:
        """
        List entities based on filter criteria.

        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field name to order by
            **filters: Filter criteria as keyword arguments

        Returns:
            List[T]: List of entities matching criteria

        Raises:
            RepositoryError: If query fails
        """
        try:
            self.logger.debug(f"Listing {self.model_name} with filters: {filters}")

            query = select(self.model_class)
            query = await self._apply_filters(query, filters)
            query = await self._apply_ordering(query, order_by)
            query = await self._apply_eager_loading(query)
            query = query.offset(offset).limit(limit)

            result = await self.session.execute(query)
            entities = result.scalars().all()

            self.logger.debug(f"Found {len(entities)} {self.model_name} entities")
            return list(entities)

        except Exception as e:
            self.logger.error(f"Failed to list {self.model_name} entities: {e}")
            raise RepositoryError(
                f"Failed to list {self.model_name} entities",
                repository=self.__class__.__name__,
                operation="list_by_criteria",
                details={"error": str(e), "filters": filters},
            )

    async def count_by_criteria(self, **filters) -> int:
        """
        Count entities matching filter criteria.

        Args:
            **filters: Filter criteria as keyword arguments

        Returns:
            int: Number of entities matching criteria

        Raises:
            RepositoryError: If count query fails
        """
        try:
            self.logger.debug(f"Counting {self.model_name} with filters: {filters}")

            query = select(func.count()).select_from(self.model_class)
            query = await self._apply_filters(query, filters)

            result = await self.session.execute(query)
            count = result.scalar()

            self.logger.debug(f"Found {count} {self.model_name} entities")
            return count

        except Exception as e:
            self.logger.error(f"Failed to count {self.model_name} entities: {e}")
            raise RepositoryError(
                f"Failed to count {self.model_name} entities",
                repository=self.__class__.__name__,
                operation="count_by_criteria",
                details={"error": str(e), "filters": filters},
            )

    async def exists(self, entity_id: UUID) -> bool:
        """
        Check if an entity exists by ID.

        Args:
            entity_id: ID of the entity to check

        Returns:
            bool: True if entity exists, False otherwise
        """
        try:
            query = (
                select(func.count())
                .select_from(self.model_class)
                .where(self.model_class.id == entity_id)
            )
            result = await self.session.execute(query)
            count = result.scalar()
            return count > 0

        except Exception as e:
            self.logger.error(
                f"Failed to check existence of {self.model_name} with ID {entity_id}: {e}"
            )
            raise RepositoryError(
                f"Failed to check existence of {self.model_name}",
                repository=self.__class__.__name__,
                operation="exists",
                details={"error": str(e), "entity_id": str(entity_id)},
            )

    async def bulk_create(self, entities: Sequence[T]) -> list[T]:
        """
        Create multiple entities in a single operation.

        Args:
            entities: List of entities to create

        Returns:
            List[T]: List of created entities

        Raises:
            RepositoryError: If bulk creation fails
        """
        try:
            self.logger.debug(
                f"Bulk creating {len(entities)} {self.model_name} entities"
            )

            # Validate all entities
            for entity in entities:
                await self._validate_entity(entity)

            # Add all entities to session
            self.session.add_all(entities)
            await self.session.flush()

            # Refresh all entities to get generated values
            for entity in entities:
                await self.session.refresh(entity)

            self.logger.info(f"Bulk created {len(entities)} {self.model_name} entities")
            return list(entities)

        except Exception as e:
            self.logger.error(f"Failed to bulk create {self.model_name} entities: {e}")
            raise RepositoryError(
                f"Failed to bulk create {self.model_name} entities",
                repository=self.__class__.__name__,
                operation="bulk_create",
                details={"error": str(e), "count": len(entities)},
            )

    # Abstract methods to be implemented by specific repositories

    @abstractmethod
    async def _validate_entity(self, entity: T, is_update: bool = False) -> None:
        """
        Validate an entity before database operations.

        Args:
            entity: Entity to validate
            is_update: Whether this is for an update operation

        Raises:
            ValidationError: If validation fails
        """
        pass

    @abstractmethod
    async def _apply_eager_loading(self, query: Select) -> Select:
        """
        Apply eager loading for relationships to avoid N+1 queries.

        Args:
            query: SQLAlchemy query to modify

        Returns:
            Select: Modified query with eager loading
        """
        pass

    # Helper methods for query building

    async def _apply_filters(self, query: Select, filters: dict[str, Any]) -> Select:
        """
        Apply filter criteria to a query.

        Args:
            query: SQLAlchemy query to modify
            filters: Dictionary of filter criteria

        Returns:
            Select: Modified query with filters applied
        """
        for field_name, value in filters.items():
            if hasattr(self.model_class, field_name):
                field = getattr(self.model_class, field_name)
                if isinstance(value, list):
                    query = query.where(field.in_(value))
                elif isinstance(value, dict) and "operator" in value:
                    # Support for complex operators like >, <, >=, <=
                    operator = value["operator"]
                    operand = value["value"]
                    if operator == "gt":
                        query = query.where(field > operand)
                    elif operator == "lt":
                        query = query.where(field < operand)
                    elif operator == "gte":
                        query = query.where(field >= operand)
                    elif operator == "lte":
                        query = query.where(field <= operand)
                    elif operator == "like":
                        query = query.where(field.like(f"%{operand}%"))
                else:
                    query = query.where(field == value)

        return query

    async def _apply_ordering(self, query: Select, order_by: str = None) -> Select:
        """
        Apply ordering to a query.

        Args:
            query: SQLAlchemy query to modify
            order_by: Field name to order by (prefix with '-' for descending)

        Returns:
            Select: Modified query with ordering applied
        """
        if order_by:
            if order_by.startswith("-"):
                field_name = order_by[1:]
                if hasattr(self.model_class, field_name):
                    field = getattr(self.model_class, field_name)
                    query = query.order_by(field.desc())
            else:
                if hasattr(self.model_class, order_by):
                    field = getattr(self.model_class, order_by)
                    query = query.order_by(field.asc())

        return query
