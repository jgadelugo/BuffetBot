# BuffetBot Repository Pattern

This directory contains the repository pattern implementation for BuffetBot's database layer. The repository pattern provides a clean abstraction over database operations and supports async operations for future FastAPI integration.

## Overview

The repository pattern separates business logic from data access logic by providing a unified interface for database operations. This implementation includes:

- **Base Repository**: Abstract base class with common CRUD operations
- **Domain Repositories**: Specific repositories for each domain entity
- **Session Management**: Async database session handling with transaction support
- **Registry Pattern**: Centralized repository access for dependency injection
- **Error Handling**: Custom exceptions for different error scenarios

## Architecture

```
database/repositories/
├── __init__.py              # Repository registry and exports
├── base.py                  # Abstract base repository
├── session_manager.py       # Database session management
├── exceptions.py            # Custom exception classes
├── portfolio_repo.py        # Portfolio and position repositories
├── analysis_repo.py         # Analysis result repository
├── market_data_repo.py      # Market data cache repository
└── examples/                # Usage examples and documentation
```

## Core Components

### 1. Base Repository (`base.py`)

The `BaseRepository` class provides common CRUD operations:

```python
from database.repositories.base import BaseRepository

class MyRepository(BaseRepository[MyEntity]):
    async def _validate_entity(self, entity, is_update=False):
        # Implement validation logic
        pass

    async def _apply_eager_loading(self, query):
        # Implement relationship loading
        return query
```

**Key Methods:**
- `create(entity)` - Create new entity
- `get_by_id(id)` - Get entity by ID
- `update(entity)` - Update existing entity
- `delete(id)` - Delete entity by ID
- `list_by_criteria(**filters)` - List entities with filters
- `count_by_criteria(**filters)` - Count entities
- `exists(id)` - Check if entity exists
- `bulk_create(entities)` - Create multiple entities

### 2. Session Manager (`session_manager.py`)

Manages database connections and transactions:

```python
from database.repositories.session_manager import get_session_manager

# Get session manager
session_manager = get_session_manager()

# Initialize database
await session_manager.initialize()

# Use transaction
async with session_manager.transaction() as session:
    # Database operations here
    pass

# Use session without transaction
async with session_manager.session() as session:
    # Database operations here
    pass
```

### 3. Repository Registry (`__init__.py`)

Centralized access to all repositories:

```python
from database.repositories import get_repository_registry

# Get registry
registry = get_repository_registry()

# Get specific repositories
portfolio_repo = await registry.get_portfolio_repository()
analysis_repo = await registry.get_analysis_repository()
market_data_repo = await registry.get_market_data_repository()

# Health check
health = await registry.health_check()
```

## Domain Repositories

### Portfolio Repository

Manages portfolios and positions:

```python
portfolio_repo = await registry.get_portfolio_repository()

# Get user portfolios
portfolios = await portfolio_repo.get_user_portfolios(user_id)

# Get portfolio by name
portfolio = await portfolio_repo.get_by_name_and_user(name, user_id)

# Add position to portfolio
updated_portfolio = await portfolio_repo.add_position(portfolio_id, position)

# Get portfolio summary
summary = await portfolio_repo.get_portfolio_summary(portfolio_id)
```

### Analysis Repository

Manages analysis results with caching:

```python
analysis_repo = await registry.get_analysis_repository()

# Get recent analysis
analysis = await analysis_repo.get_recent_analysis(
    ticker="AAPL",
    analysis_type="value_analysis",
    max_age_hours=24
)

# Get analysis history
history = await analysis_repo.get_portfolio_analysis_history(portfolio_id)

# Clean up expired analysis
deleted_count = await analysis_repo.cleanup_expired_analysis()

# Invalidate analysis
count = await analysis_repo.invalidate_analysis(ticker="AAPL")
```

### Market Data Repository

Manages cached market data:

```python
market_data_repo = await registry.get_market_data_repository()

# Cache market data
cached_data = await market_data_repo.cache_market_data(
    ticker="AAPL",
    data_type="fundamentals",
    data={"price": 150.25, "volume": 1000000},
    ttl_hours=24
)

# Get cached data
data = await market_data_repo.get_cached_data("AAPL", "fundamentals")

# Get cache statistics
stats = await market_data_repo.get_cache_statistics(days_back=7)

# Clean up expired cache
deleted_count = await market_data_repo.cleanup_expired_cache()
```

## Usage Examples

### Basic Usage

```python
import asyncio
from database.repositories import init_repositories, close_repositories, get_repository_registry

async def main():
    # Initialize repositories
    await init_repositories()

    try:
        # Get repository registry
        registry = get_repository_registry()

        # Get repositories
        portfolio_repo = await registry.get_portfolio_repository()

        # Use repositories
        portfolios = await portfolio_repo.get_user_portfolios(user_id)

    finally:
        # Clean up
        await close_repositories()

asyncio.run(main())
```

### Transaction Usage

```python
from database.repositories import get_repository_registry

async def create_portfolio_with_positions():
    registry = get_repository_registry()
    session_manager = registry.session_manager

    async with session_manager.transaction() as session:
        # Create repositories with transaction session
        portfolio_repo = PortfolioRepository(session)
        position_repo = PositionRepository(session)

        # Create portfolio
        portfolio = await portfolio_repo.create(my_portfolio)

        # Add positions
        for position_data in positions:
            position = Position(**position_data)
            await position_repo.create(position)

        # Transaction commits automatically
```

### Error Handling

```python
from database.exceptions import (
    EntityNotFoundError,
    ValidationError,
    RepositoryError
)

try:
    portfolio = await portfolio_repo.get_by_id(portfolio_id)
    if not portfolio:
        raise EntityNotFoundError("Portfolio", portfolio_id)

    # Update portfolio
    updated = await portfolio_repo.update(portfolio)

except ValidationError as e:
    print(f"Validation failed: {e}")
except EntityNotFoundError as e:
    print(f"Entity not found: {e}")
except RepositoryError as e:
    print(f"Repository error: {e}")
```

## Configuration

The repositories use the existing configuration system:

```python
# database/repositories/session_manager.py
def _get_database_url(self) -> str:
    # Integrates with Phase 1a configuration
    # In real implementation, reads from config/environment
    return "postgresql+asyncpg://user:password@localhost/buffetbot"
```

## Testing

Comprehensive tests are provided in `tests/database/test_repository_pattern.py`:

```bash
# Run repository tests
pytest tests/database/test_repository_pattern.py

# Run with coverage
pytest tests/database/test_repository_pattern.py --cov=database.repositories
```

## Integration with Existing Code

The repository pattern is designed to work alongside existing data access patterns:

1. **Additive Implementation**: Repositories don't replace existing data fetchers initially
2. **Backward Compatibility**: Existing Streamlit app continues to work
3. **Gradual Migration**: Analysis modules can gradually adopt repository pattern
4. **Configuration Integration**: Uses existing config patterns

## Performance Considerations

### Query Optimization

- **Eager Loading**: Repositories implement `_apply_eager_loading()` to avoid N+1 queries
- **Pagination**: Built-in support for offset/limit pagination
- **Filtering**: Efficient filtering with support for complex operators
- **Indexing**: Comments in models suggest appropriate indexes

### Caching Strategy

- **Analysis Results**: Automatic expiration and cleanup
- **Market Data**: TTL-based caching with configurable expiration
- **Statistics**: Efficient cache usage reporting

### Connection Management

- **Connection Pooling**: Configurable connection pool settings
- **Session Lifecycle**: Proper session creation and cleanup
- **Transaction Boundaries**: Clear transaction management

## Future Enhancements

### Phase 2 Integration

- **FastAPI Dependency Injection**: Repository registry designed for DI
- **Service Layer**: Repositories will be consumed by service classes
- **API Error Handling**: Consistent error responses

### Phase 3 Async Processing

- **Background Tasks**: Repositories support async operations
- **Bulk Operations**: Efficient bulk create/update operations
- **Event Sourcing**: Foundation for event-driven architecture

### Phase 4 Optimization

- **Advanced Caching**: Redis integration for distributed caching
- **Read Replicas**: Support for read/write splitting
- **Monitoring**: Performance metrics and query analysis

## Best Practices

### Repository Design

1. **Single Responsibility**: Each repository manages one domain entity
2. **Interface Segregation**: Minimal, focused interfaces
3. **Dependency Injection**: Use registry for repository access
4. **Error Handling**: Consistent exception handling patterns

### Performance

1. **Lazy Loading**: Load relationships only when needed
2. **Batch Operations**: Use bulk operations for multiple entities
3. **Query Optimization**: Implement efficient queries with proper indexes
4. **Connection Management**: Use connection pooling appropriately

### Testing

1. **Mock Dependencies**: Mock database sessions for unit tests
2. **Integration Tests**: Test with real database for complex operations
3. **Performance Tests**: Benchmark critical operations
4. **Error Scenarios**: Test all error conditions

## Migration Guide

### From Direct SQLAlchemy

```python
# Before (direct SQLAlchemy)
async with async_session() as session:
    result = await session.execute(
        select(Portfolio).where(Portfolio.user_id == user_id)
    )
    portfolios = result.scalars().all()

# After (repository pattern)
portfolio_repo = await registry.get_portfolio_repository()
portfolios = await portfolio_repo.get_user_portfolios(user_id)
```

### From Current Data Fetchers

```python
# Before (current fetcher)
from buffetbot.data.fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch_fundamentals("AAPL")

# After (repository pattern)
market_data_repo = await registry.get_market_data_repository()
cached_data = await market_data_repo.get_cached_data("AAPL", "fundamentals")
if not cached_data:
    # Fetch and cache new data
    fresh_data = fetcher.fetch_fundamentals("AAPL")
    cached_data = await market_data_repo.cache_market_data(
        "AAPL", "fundamentals", fresh_data
    )
```

## Contributing

When adding new repositories:

1. Extend `BaseRepository[T]` for the entity type
2. Implement `_validate_entity()` and `_apply_eager_loading()`
3. Add domain-specific methods as needed
4. Register in `RepositoryRegistry`
5. Add comprehensive tests
6. Update documentation

For questions or issues, refer to the repository pattern documentation or create an issue in the project repository.
