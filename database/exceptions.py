"""
Database exceptions for BuffetBot repository layer.

Custom exceptions for database operations with proper error handling and categorization.
"""


class DatabaseError(Exception):
    """Base database exception for all database-related errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code or "DATABASE_ERROR"
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class EntityNotFoundError(DatabaseError):
    """Entity not found in database."""

    def __init__(self, entity_type: str, identifier: str, details: dict = None):
        message = f"{entity_type} with identifier '{identifier}' not found"
        super().__init__(message, "ENTITY_NOT_FOUND", details)
        self.entity_type = entity_type
        self.identifier = identifier


class DuplicateEntityError(DatabaseError):
    """Attempt to create duplicate entity."""

    def __init__(self, entity_type: str, identifier: str, details: dict = None):
        message = f"{entity_type} with identifier '{identifier}' already exists"
        super().__init__(message, "DUPLICATE_ENTITY", details)
        self.entity_type = entity_type
        self.identifier = identifier


class DatabaseConnectionError(DatabaseError):
    """Database connection issues."""

    def __init__(
        self, message: str = "Database connection failed", details: dict = None
    ):
        super().__init__(message, "CONNECTION_ERROR", details)


class TransactionError(DatabaseError):
    """Transaction-related errors."""

    def __init__(self, message: str, operation: str = None, details: dict = None):
        super().__init__(message, "TRANSACTION_ERROR", details)
        self.operation = operation


class ValidationError(DatabaseError):
    """Data validation errors before database operations."""

    def __init__(
        self, message: str, field: str = None, value=None, details: dict = None
    ):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class QueryError(DatabaseError):
    """Database query execution errors."""

    def __init__(self, message: str, query: str = None, details: dict = None):
        super().__init__(message, "QUERY_ERROR", details)
        self.query = query


class CacheError(DatabaseError):
    """Cache-related database errors."""

    def __init__(self, message: str, cache_key: str = None, details: dict = None):
        super().__init__(message, "CACHE_ERROR", details)
        self.cache_key = cache_key


class RepositoryError(DatabaseError):
    """Repository-specific errors."""

    def __init__(
        self,
        message: str,
        repository: str = None,
        operation: str = None,
        details: dict = None,
    ):
        super().__init__(message, "REPOSITORY_ERROR", details)
        self.repository = repository
        self.operation = operation
